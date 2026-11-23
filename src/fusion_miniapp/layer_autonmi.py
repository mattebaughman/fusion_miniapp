"""The third agent layer: fusion's campaigns on AUTONMI's framework, as a workflow mini-app.

    python -m fusion_miniapp.layer_autonmi --preset smoke --sim fixture --agents 100 --populations 2

    # under Dragon: populations as Dragon processes, simulations on RHAPSODY's native backend, RaDEX
    python -m dragon -s src/fusion_miniapp/layer_autonmi.py --runtime dragon --populations 2 --workers 16 ...

Every campaign is the same three agents as in the other two layers (decide.py's Coordinator, Generator
and Evaluator), with the same messages in the same order, so its result is the same: `results_sha256`
equals the other layers' for the same knobs. What runs them is AUTONMI's:
- The mini-app is an AsyncFlow application. Its campaigns are split into populations (`--populations`;
  by default one per agent process of run.py), each an AsyncFlow node with an engine of its own for its
  simulations, a fleet for its model calls, and Flowgentic's agent host. Locally the populations run in
  this process; under Dragon (`--runtime dragon`) each is a Dragon process, its simulations run on
  RHAPSODY's native Dragon backend, and the store is RaDEX.
- A campaign is one AUTONMI scripted agent (`Script`): a step function over its three roles' memories.
  A generation's P evaluations are one fan-out of suspending `evaluate` task tools: the campaign is saved
  and dropped until all P end, the parent's generation barrier, as in the LangGraph layer. The
  Generator's model call is awaited inside its step, as decide.generator_step is written.
- Model calls go through Flowgentic's fleet: fusion's stub LLM as a transport, or (`--llm openai`) one
  Flowgentic `HTTPTransport` per server of the model job (`--endpoints-file`), which waits out a server
  that goes away instead of retrying (`llm.max_retries` does not apply).

It writes what run.py writes, with `agent_layer: autonmi`: `run_record.json` (record.write, with an
`autonmi` section), `agents.jsonl` and `campaigns/<id>/`; and AUTONMI's own folders (`launcher/`,
`populations/`, `app/`, `otel/`) and `summary.json`, its checks of the run's telemetry. Each population
streams its campaigns' lines to `agents-p<index>.jsonl`, as each agent process of run.py does, and
record.write merges them. Exit status as run.py's: 0 when the record passes and so do AUTONMI's checks,
1 when not, 143 when SIGTERM or Ctrl-C interrupted the run.

Needs the AUTONMI framework (the package `autonmi`, installed separately) and Flowgentic 0.6
(requirements-autonmi.txt).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

from autonmi import AgentSpec, Campaign, CampaignConfig, Check
from autonmi.population import PopulationConfig
from autonmi.scripted import Done, Lookup, Remember, Report, ScriptedAgent, Simulate, SimulateAll
from flowgentic.fleet import Reply, Usage
from flowgentic.host import current_agent
from flowgentic.tools import task_tool
from pydantic import BaseModel
from radical.asyncflow import WorkflowEngine
from wfMiniAPI.api import MiniApp

from fusion_miniapp import decide, knobs as knob_files, record, run, tools
from fusion_miniapp.llm import MAX_IN_FLIGHT, REPLY_TOKENS, StubLLM

ADMISSION_TOKENS = 2_000_000   # admitted tokens per model server, as the LangGraph layer's ADMISSION
# Under `python -m dragon -s .../layer_autonmi.py` (and `python -m`) this file runs as __main__. The
# definition and the simulation pools' factory are taken from the module of this name instead, so the
# populations' processes can unpickle them, and so _POOLS is the list the factory fills.
MODULE = "fusion_miniapp.layer_autonmi"
_POOLS: list = []   # this process's simulation pools, told to stop rebuilding on SIGTERM
_OWN_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)   # the run's, not its engines' (run_until_signalled)


class EvaluateArgs(BaseModel):
    params: dict[str, float | str]     # the parent's full candidate, with its "model" and "kind" strings
    candidate_idx: int
    generation: int
    seed: int


class StubTransport:
    """Fusion's stub LLM as a Flowgentic transport. Its replies depend on the campaign (the agent
    Flowgentic's host runs the call for) and the request's last line only, so the three layers agree."""

    model = "fusion-stub"

    def __init__(self) -> None:
        self.stub = StubLLM()

    async def complete(self, messages, tools=None, *, extra_body: dict | None = None, max_tokens: int | None = None,
                       **kwargs) -> Reply:
        b = int((extra_body or {}).get("thinking_token_budget", 0))
        r = max_tokens - b if max_tokens is not None else REPLY_TOKENS
        text, meta = await self.stub.call_with_meta(current_agent() or "", list(messages), b, reply_tokens=r)
        prompt = sum(len(str(m.get("content") or "")) for m in messages) // 4   # an estimate, for admission
        return Reply(content=text, finish_reason=meta["finish_reason"],
                     usage=Usage(prompt_tokens=prompt, completion_tokens=len(text) // 4))


def sim_pool(workers: int):
    """The executor of a population's real evaluations (AUTONMI's PopulationConfig.executor): spawned
    worker processes, as run.py's, which rebuild once when a worker dies and stop within run.SHUTDOWN_S."""
    from fusion_miniapp.pool import ResilientProcessPool

    pool = ResilientProcessPool(workers, mp_context=mp.get_context("spawn"), initializer=run._worker_init,
                                shutdown_timeout=run.SHUTDOWN_S)
    _POOLS.append(pool)
    return pool


def fixture_pool(workers: int):
    """The executor of a population's fixture evaluations: run.py's threads, which drop their queued work
    when they close (an interrupted run can hold tens of thousands of queued evaluations)."""
    return run.FixturePool(workers)


class Fusion:
    """Fusion as an AUTONMI agent definition (autonmi.AgentDefinition): its campaigns, how a population
    runs them (Script), their rows and the checks on them."""

    name = "fusion"

    def __init__(self, knobs: dict, sim: str, out: Path, work_root: str | None = None) -> None:
        self.knobs, self.sim, self.out, self.work_root = knobs, sim, Path(out), work_root

    def agents(self):
        for i in range(self.knobs["agents"]["count"]):   # as the other layers: campaign a<i>, seed i
            yield AgentSpec(f"a{i}", {"seed": i})

    def services(self, run_dir: Path, servers: int = 1) -> tuple[list, list[str]]:
        return [], []   # no MCP servers: the lookups run in the campaign's process (tools.lookup)

    def stub(self, replica: int) -> StubTransport:
        return StubTransport()

    def sim_knobs(self, seed: int) -> dict:
        """The `sim` knobs one campaign's evaluations get: its cost scale, and this run's own work folder."""
        s = decide.sim_knobs(self.knobs, seed)
        if self.work_root is not None:
            s["work_root"] = self.work_root
        return s

    def build(self, pop) -> ScriptedAgent:
        return Script(self, pop).agent()

    def rows(self, pop) -> list[dict]:
        """At the population's end, in its process: one row per campaign, whether it reported, and the
        evaluations it submitted, all of which end before it goes on (the generation's barrier): AUTONMI
        checks them against the tasks its engine completed."""
        s = pop.state
        if "lines" in s:
            s["lines"].close()
        return [{"agent_id": a, "finished": a in s.get("reported", ()), "simulations": s.get("sims", {}).get(a, 0),
                 "population": pop.index} for a in pop.agent_ids]

    def verify(self, rows: list[dict], store) -> list[Check]:
        """AUTONMI's checks of the campaigns; the run record checks their results (touches, the model)."""
        done, exp = [r for r in rows if r["finished"]], decide.expected(self.knobs)
        planned = all(r["simulations"] == exp["sims"] and 1 <= r["model_calls"] <= exp["llm_calls"] for r in done)
        return [Check("every_campaign_reports", len(done) == self.knobs["agents"]["count"],
                      {"reported": len(done), "campaigns": self.knobs["agents"]["count"]}),
                Check("evaluations_and_model_calls_as_planned", planned, exp)]


class Script:
    """One population's campaigns as AUTONMI scripted agents (autonmi.scripted): decide.py's steps, in
    the LangGraph layer's order, as a step function over a state that can be saved. A campaign's phases:

        look         the Coordinator's lookups, one at a time; then config.json and the Generator's
                     initialize, whose population the Coordinator sends to the Evaluator
        evaluating   the generation's P evaluations, out at once; the campaign is suspended until all end
        remembering  the Coordinator's selection, best-so-far and log done; its memory goes to the store
        reporting    the summary call done; its result goes to the campaign manager
    and after `remembering`, a generation left: the Generator's repopulate (with its model call in llm
    mode), and `evaluating` again.

    Each campaign's line (its result, or its error) goes to the population's agents-p<index>.jsonl."""

    def __init__(self, definition: Fusion, pop) -> None:
        self.knobs, self.sim, self.out, self.pop = definition.knobs, definition.sim, definition.out, pop
        self.send_budget = self.knobs["llm"].get("send_thinking_budget", True)
        self.calls: dict[str, list] = {}    # each campaign's model calls, until its line is written
        self.sims: Counter = Counter()      # the evaluations each campaign submitted
        self.reported: set[str] = set()
        self.lines = open(self.out / f"agents-p{pop.index:03d}.jsonl", "w", buffering=1)  # noqa: SIM115 (rows())
        pop.state.update(lines=self.lines, sims=self.sims, reported=self.reported)
        self.evaluate = task_tool(
            run.sim_function(definition.sim), suspend=True, name="evaluate", args_schema=EvaluateArgs,
            description="Evaluate one candidate profile with the turned-down IGNOS inner loop.",
            prepare=lambda agent_id, call_id, args: {**args, "knobs": definition.sim_knobs(args["seed"]),
                                                     "agent_id": agent_id})

    def agent(self) -> ScriptedAgent:
        return ScriptedAgent(start=self.start, step=self.step, simulate=self.evaluate, lookup=self.lookup,
                             remember=self.remember, report=self.report, on_call=self.on_call,
                             on_done=self.on_done, model_calls_limit=decide.expected(self.knobs)["llm_calls"])

    # --- the steps --------------------------------------------------------------------------------------

    def start(self, agent_id: str, spec: dict) -> dict:
        """A campaign's first state, when the host starts it: the lookups it makes."""
        return {"agent_id": agent_id, "seed": spec["seed"], "phase": "look", "t_start": time.time(),
                "todo": decide.lookups(self.knobs), "ctx": {}}

    async def step(self, s: dict, event, ask):
        agent_id, phase = s["agent_id"], s["phase"]
        run_dir = decide.campaign_dir(self.out, self.knobs, agent_id)
        if phase == "look":
            todo, ctx = s["todo"], s["ctx"]
            if event is not None:                               # the previous lookup's result
                todo, ctx = todo[1:], {**ctx, todo[0]: event}
            if todo:
                return {**s, "todo": todo, "ctx": ctx}, Lookup(todo[0])
            cst = decide.coordinator_state(agent_id, s["seed"], self.knobs, ctx, s["t_start"])
            msg = decide.start(cst, self.knobs, run_dir, self.sim)          # the Coordinator -> the Generator
            gst = decide.generator_state(agent_id, s["seed"], self.knobs)
            return await self._generate(s, cst, gst, msg, ask)
        if phase == "evaluating":                               # the Evaluator -> the Coordinator
            # as the LangGraph layer reads a task's result: through JSON, where NaN arrives as None
            results = json.loads(json.dumps(event, default=str))
            reply = {"results": results, "t0": s["t0"], "elapsed_s": time.time() - s["t0"]}
            cst, nxt = decide.on_evaluated(s["cst"], reply, self.knobs, run_dir)
            return _state(s, "remembering", cst=cst, gst=s["gst"], next=nxt), Remember(decide.memory(cst))
        if phase == "remembering":
            cst = s["cst"]
            if s["next"] is not None:                           # the Coordinator -> the Generator, again
                return await self._generate(s, cst, s["gst"], s["next"], ask)
            summary = await self._model(agent_id, ask, decide.summary_messages(cst), cst["budget"],
                                        decide.SUMMARY_TOKENS)
            return _state(s, "reporting"), Report(decide.finish(cst, summary, run_dir))
        if phase == "reporting":
            return s, Done()
        raise ValueError(f"unknown phase {phase!r}")

    async def _generate(self, s: dict, cst: dict, gst: dict, msg: dict, ask):
        """The Generator's step and its population to the Evaluator, whose P evaluations go out at once."""
        agent_id = s["agent_id"]
        gst, reply = await decide.generator_step(gst, msg, self.knobs,
                                                 lambda m, b, r: self._model(agent_id, ask, m, b, r))
        cst, msg = decide.on_population(cst, reply)
        calls = decide.evaluation_calls(msg["population"], msg["generation"])
        self.sims[agent_id] += len(calls)
        batch = tuple(Simulate({"params": c["params"], "candidate_idx": c["candidate_idx"],
                                "generation": c["generation"], "seed": s["seed"]}, call_id=c["id"]) for c in calls)
        return _state(s, "evaluating", cst=cst, gst=gst, t0=time.time()), SimulateAll(batch)

    async def _model(self, agent_id: str, ask, messages: list[dict], budget: int, reply_tokens: int) -> str:
        """decide.Ask through the population's fleet, with the request run.py's OpenAIChat sends:
        max_tokens = thinking budget + reply allowance, vLLM's thinking_token_budget (unless
        llm.send_thinking_budget is false), and no temperature (AUTONMI's transports set 0; None leaves the
        server's default). Admission reserves the budget. A call that fails is recorded, as run.Services
        records it."""
        kwargs = {"max_tokens": budget + reply_tokens, "reserve": budget + reply_tokens, "temperature": None,
                  "meta": {"budget": budget, "reply_tokens": reply_tokens}}
        if self.send_budget:
            kwargs["extra_body"] = {"thinking_token_budget": budget}
        t0 = time.perf_counter()
        try:
            return await ask(messages, **kwargs)
        except Exception as exc:
            self.calls.setdefault(agent_id, []).append(
                {"reply_tokens": reply_tokens, "budget": budget, "latency_s": round(time.perf_counter() - t0, 4),
                 "error": type(exc).__name__, "finish_reason": None, "empty": None})
            raise

    # --- the actions' services ----------------------------------------------------------------------------

    async def lookup(self, agent_id: str, what: str) -> dict:
        return await tools.lookup(what, agent_id, self.knobs["tools"]["latency_ms"])

    def remember(self, agent_id: str, state: dict) -> None:
        """The Coordinator's memory after each generation, in the population's store (RaDEX under Dragon)."""
        self.pop.store.put(f"campaign/{agent_id}", json.dumps(state, default=str))

    def report(self, agent_id: str, result: dict) -> None:
        """A campaign's result to the campaign manager: its line, as run.Services.report writes it."""
        self.reported.add(agent_id)
        self._line({**result, "llm_call_details": self.calls.pop(agent_id, [])})

    def on_call(self, rec, reply: Reply) -> None:
        """A model call that returned: its details, as run.Services keeps them for the campaign's line."""
        m = rec.meta or {}
        self.calls.setdefault(rec.agent_id, []).append({
            "reply_tokens": m.get("reply_tokens"), "budget": m.get("budget"),
            "latency_s": round(rec.service_s + rec.recovery_s, 4), "wait_s": round(rec.wait_s, 4),
            "finish_reason": reply.finish_reason, "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens, "reasoning_tokens": rec.reasoning_tokens or 0,
            "empty": not str(reply.content or "").strip()})

    def on_done(self, result, output) -> None:
        """A campaign that failed is a line with its error, as in the other layers; one cancelled (the run
        was interrupted) is not."""
        if result.status == "failed" and result.id not in self.reported:
            self._line({"agent_id": result.id, "error": result.error or result.status,
                        "llm_call_details": self.calls.pop(result.id, [])})

    def _line(self, line: dict) -> None:
        self.lines.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def _state(s: dict, phase: str, **memory) -> dict:
    """A campaign's state: its id, seed and phase, and what the phase needs (saved while it is suspended)."""
    return {"agent_id": s["agent_id"], "seed": s["seed"], "phase": phase, **memory}


# --- the mini-app ----------------------------------------------------------------------------------------

def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=int, help="campaigns (default: the knob file's agents.count)")
    p.add_argument("--preset", choices=knob_files.PRESETS, help="merged over the smoke preset")
    p.add_argument("--knobs", action="append", default=[], metavar="FILE",
                   help="a knob file (YAML) merged over the preset; repeatable, applied in order")
    p.add_argument("--sim", choices=("real", "fixture"), default="real")
    p.add_argument("--llm", choices=("stub", "openai"), help="overrides llm.mode")
    p.add_argument("--out", help="output directory (default: results/<time>-autonmi-<agents>-<pid>)")
    p.add_argument("--populations", type=int, help="populations of campaigns (default: as run.py's agent "
                                                   "processes, agents.count / agents.per_process)")
    p.add_argument("--workers", type=int, help="simulation workers per population "
                                               "(default: workers.sim_workers / populations)")
    p.add_argument("--runtime", choices=("local", "dragon"), default="local",
                   help="dragon: populations as Dragon processes; run under `python -m dragon -s`")
    p.add_argument("--endpoints-file", help="--llm openai: the model job's servers, one base URL per line "
                                            "(without /v1); each must answer /health")
    p.add_argument("--api-key-file", help="--llm openai: a file holding the servers' key, if they need one")
    p.add_argument("--model-wait", type=float, default=1800.0, help="seconds to wait for every server to answer")
    p.add_argument("--stagger-batch", type=int, default=500, help="first-wave campaign starts per batch")
    p.add_argument("--stagger-interval", type=float, default=0.01, help="seconds between those batches")
    return p.parse_args(argv)


def load_knobs(a: argparse.Namespace) -> dict:
    """As run.py loads them."""
    knobs = knob_files.load(preset=a.preset, paths=a.knobs)
    over = {"agents": {"count": a.agents} if a.agents is not None else {}, "llm": {"mode": a.llm} if a.llm else {}}
    knobs = knob_files.validate({**knobs, **{s: {**knobs[s], **v} for s, v in over.items()}})
    if knobs["llm"]["mode"] == "openai" and knobs["llm"]["model"] == "stub":
        raise knob_files.KnobError('llm.mode is openai but llm.model is still "stub"; set llm.model in a knob file')
    return knobs


def layout(knobs: dict, sim: str, populations: int | None = None, workers: int | None = None) -> tuple[int, int]:
    """Populations, and simulation workers in each: by default run.py's agent processes (agents.per_process
    campaigns each) and their share of workers.sim_workers, which with --sim real must give each one, as
    run.py requires."""
    n, w = knobs["agents"]["count"], knobs["workers"]["sim_workers"]
    populations = min(populations or -(-n // knobs["agents"]["per_process"]), n)
    if workers is None:
        if sim == "real" and populations > w:
            raise knob_files.KnobError(
                f"{populations} populations each need a simulation worker, but workers.sim_workers is {w}; raise "
                "workers.sim_workers or agents.per_process, or give --workers")
        workers = max(1, w // populations)
    return populations, workers


def own(out: Path) -> None:
    """--out as run.py takes it: not a folder that holds a run (the parent's logs append), and held by
    run.lock, also against a second run started at the same moment."""
    used = [p.name for p in out.glob("*") if p.name in ("run_record.json", "agents.jsonl", "campaigns", "run.lock")
            or p.name.startswith(("agents-", "part-"))] if out.is_dir() else []
    if used:
        raise knob_files.KnobError(f"--out {out} already holds a run ({', '.join(sorted(used))}); "
                                   "give a new directory")
    out.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(out / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise knob_files.KnobError(f"--out {out} is in use by another run (run.lock); give a new directory") from None


async def make_backend(a: argparse.Namespace):
    """The mini-app's own backend, which runs the populations: RHAPSODY's native one under Dragon (each
    population a process of its own), a thread pool locally (the populations are blocks of this process)."""
    if a.runtime == "dragon":
        from rhapsody.backends import DragonNativeExecutionBackend

        return await DragonNativeExecutionBackend(workers_per_node=2)
    from rhapsody.backends import ConcurrentExecutionBackend

    return await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=4))


async def run_miniapp(a: argparse.Namespace, knobs: dict, out: Path, work_root: str | None) -> int:
    backend = await make_backend(a)
    flow = await WorkflowEngine.create(backend=backend, work_dir=str(out))
    telemetry = await flow.start_telemetry(checkpoint_path=str(out / "launcher"))
    app = MiniApp(flow, telemetry)
    ok = False
    try:
        async with app, flow.workflow_scope("fusion"):
            ok = await miniapp_workflow(a, knobs, flow, out, app, work_root)
    finally:
        try:
            await flow.shutdown()
        finally:
            await telemetry.stop()
            app.verify()
    report = app.report
    print(f"[fusion/autonmi] checks {'pass' if ok and report.checks_pass else 'FAIL'}; telemetry "
          f"{'complete' if report.telemetry_complete else 'INCOMPLETE'}; {out}", file=sys.stderr, flush=True)
    return 0 if ok and report.exit_hint == 0 else 1


async def miniapp_workflow(a, knobs: dict, flow, out: Path, app, work_root: str | None) -> bool:
    k = knobs["llm"]
    model = None
    if k["mode"] == "openai":   # the model job's servers, once all answer; the thinking budget is per call
        from autonmi.models import wait_for_servers

        model = await wait_for_servers(a.endpoints_file, a.api_key_file, wait_s=a.model_wait, name=k["model"],
                                       timeout_s=k["timeout_s"], thinking_off={})
    definition = importlib.import_module(MODULE).Fusion(knobs, a.sim, out, work_root)
    # locally, the populations' simulations run on run.py's pools; under Dragon, on RHAPSODY's native backend
    executor = f"{MODULE}:{'sim_pool' if a.sim == 'real' else 'fixture_pool'}" if a.runtime == "local" else None
    campaign = Campaign(
        flow,
        definition,
        CampaignConfig(
            run_dir=str(out), populations=a.populations, runtime=a.runtime,
            population=PopulationConfig(
                workers=a.workers, stagger_batch=a.stagger_batch, stagger_interval_s=a.stagger_interval,
                cap_tokens=ADMISSION_TOKENS, cap_requests=MAX_IN_FLIGHT, max_live=knobs["agents"]["max_live"],
                executor=executor, model=model),
        ),
        app=app,
    )
    try:
        await campaign.start_services()
        try:
            await campaign.run()
        finally:
            await campaign.stop_services()
        return campaign.check()
    finally:
        await campaign.close()


async def run_until_signalled(a, knobs: dict, out: Path, work_root: str | None) -> tuple[int, bool, dict]:
    """run_miniapp, which SIGTERM or SIGINT cancels through the event loop, as run.run_slice is cancelled:
    the populations, their engines and pools close, and the run is recorded as interrupted. Returns
    run_miniapp's exit hint, whether the run was interrupted, and the event loop's lag (record.LoopLag)."""
    loop, me, stopped = asyncio.get_running_loop(), asyncio.current_task(), []
    pools = importlib.import_module(MODULE)._POOLS

    def on_term() -> None:
        if not stopped:
            stopped.append(time.time())
            for pool in pools:   # their workers are about to get the same signal: do not replace them
                pool.stop_rebuilds()
            me.cancel()
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in before:   # a plain handler that queues on_term, as run.run_slice installs
        with contextlib.suppress(ValueError):   # not the main thread: skip
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(on_term))
    # Every AsyncFlow engine registers SIGTERM, SIGINT and SIGHUP on its event loop for a shutdown of its
    # own, the last engine created winning (here, one population's): the run would end as that one
    # population's failure. This run owns its signals, as run.py does, so its loop declines them.
    register = loop.add_signal_handler
    loop.add_signal_handler = lambda sig, *args: None if sig in _OWN_SIGNALS else register(sig, *args)
    lag = record.LoopLag()
    lag.start()
    rc = 1
    try:
        rc = await run_miniapp(a, knobs, out, work_root)
    except asyncio.CancelledError:
        if not stopped:
            raise
    finally:
        if stopped and me.cancelling():   # the cancellation came from on_term (and may have been absorbed
            me.uncancel()                 # below, where a host ends its agents as cancelled)
        for sig, old in before.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(sig, signal.SIG_IGN if stopped else old)
        loop_lag = await lag.stop()
    return rc, bool(stopped), loop_lag


def probe(a: argparse.Namespace) -> dict | None:
    """record.device_probe where the simulations run, as run.Services.probe takes it: in a spawned process
    with --sim real, so that a broken torch or device stops the run before any campaign starts. None for
    the fixture, and under Dragon, whose simulation workers this process cannot reach."""
    if a.sim != "real" or a.runtime != "local":
        return None
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(record.device_probe, True).result()


def llm_service(knobs: dict, lines: list[dict]) -> dict:
    """These campaigns' model calls, as llm.py's recorder reports an agent process's (seen())."""
    details = [c for r in lines for c in r.get("llm_call_details", [])]
    ok = [c for c in details if not c.get("error")]
    return {"mode": knobs["llm"]["mode"], "calls": len(details),
            "budgets": dict(Counter(str(c["budget"]) for c in details)),
            "empty": sum(bool(c["empty"]) for c in ok), "errors": len(details) - len(ok),
            "retries": 0,   # AUTONMI's fleet waits out a server instead (layer.model_outages)
            "finish_reasons": dict(Counter(c["finish_reason"] for c in ok)),
            "tokens": {k: sum(c.get(f"{k}_tokens") or 0 for c in ok) for k in ("prompt", "completion", "reasoning")},
            "latency_s": [c["latency_s"] for c in details]}


def part(a: argparse.Namespace, knobs: dict, index: int, lines_file: Path, *, loop_lag, device, threads) -> dict:
    """Population `index` as record.write takes an agent process of run.py (run.Services.part): its campaign
    lines, and its counts, from them and from AUTONMI's population.json (written when it ends; an
    interrupted population has none). The counts from the lines are those of the campaigns that reported."""
    lines = [json.loads(x) for x in lines_file.read_text().splitlines() if x]
    done = [r for r in lines if "error" not in r]
    stats_file = Path(a.out) / "populations" / f"p{index:03d}" / "population.json"
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
    runner, local = stats.get("runner") or {}, a.runtime == "local"
    return {
        "agents_file": str(lines_file), "llm_service": llm_service(knobs, lines),
        "tool_calls": dict(Counter(what for _ in done for what in decide.lookups(knobs))),
        "sim_tasks": runner.get("simulations", sum(r["n_sims"] for r in done)),
        "memory_writes": sum(len(r["generations"]) for r in done), "reports": len(done),
        "layer": {"layer": "autonmi", "population": index, "runtime": a.runtime, "complete": bool(stats),
                  "suspensions": runner.get("suspensions"), "runner": runner, "tasks": stats.get("tasks"),
                  "outcomes": stats.get("outcomes"), "wall_s": stats.get("wall_s"),
                  "model_outages": stats.get("model_outages")},
        "sim_workers": a.workers, "sim_pool": ("process" if a.sim == "real" else "thread") if local else "dragon",
        "threads": threads, "pool": None, "pruned_tasks": 0,
        "backend": "ConcurrentExecutionBackend" if local else "DragonNativeExecutionBackend",
        # locally every population runs on this process's event loop; under Dragon, in processes of their own
        "loop_lag": loop_lag if local and index == 0 else None,
        "peak_rss_mb": record.peak_rss_mb() if local and index == 0 else None,
        "device": device or record.device_probe(), "frameworks": record.frameworks(),
    }


def framework_versions() -> dict:
    """AUTONMI's framework, AsyncFlow and wfMiniAPI, which record.versions() leaves out: their versions,
    and their commits when installed from git."""
    out = {}
    for dist in (*metadata.packages_distributions().get("autonmi", []), "radical.asyncflow", "wfMiniAPI"):
        try:
            d = metadata.distribution(dist)
        except metadata.PackageNotFoundError:
            continue
        vcs = json.loads(d.read_text("direct_url.json") or "{}").get("vcs_info", {})
        out[d.metadata["Name"]] = {"version": d.version, "commit": vcs.get("commit_id")}
    return out


def write_record(a, knobs: dict, out: Path, *, wall_s: float, t_run: float, interrupted: bool, crashed: bool,
                 rc: int, loop_lag, device, threads) -> dict:
    """run_record.json and agents.jsonl by record.write, one part per population; then AUTONMI's verdict
    (summary.json, by autonmi.assemble) in its `autonmi` section. A run whose AUTONMI checks fail fails."""
    from autonmi.assemble import assemble

    parts, failed = [], []
    for i in range(a.populations):
        f = out / f"agents-p{i:03d}.jsonl"
        if f.exists():
            parts.append(part(a, knobs, i, f, loop_lag=loop_lag, device=device, threads=threads))
        else:   # the population never started
            failed.append(i)
    rec = record.write(out, knobs=knobs, layer="autonmi", sim=a.sim, parts=parts, failed=failed, wall_s=wall_s,
                       interrupted=interrupted, t_run=t_run)
    try:
        summary = assemble(out)
    except Exception as exc:   # an interrupted or broken run may lack what AUTONMI assembles
        summary = {"ok": False, "checks": [{"name": "assembled", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}]}
    ok = rc == 0 and bool(summary.get("ok")) and not crashed
    rec["autonmi"] = {"ok": ok, "failed_checks": [c["name"] for c in summary.get("checks", []) if not c["ok"]],
                      "summary": "summary.json", "runtime": a.runtime, "populations": a.populations,
                      "workers_per_population": a.workers, "crashed": crashed, "versions": framework_versions()}
    rec["counts"]["suspensions"] = sum(p["layer"]["suspensions"] or 0 for p in parts)
    if rec["status"] == "pass" and not ok:
        rec["status"] = "fail"
    tmp = out / "run_record.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, out / "run_record.json")
    return rec


def main(argv=None) -> int:
    a = parse(argv)
    knobs = load_knobs(a)
    if knobs["llm"]["mode"] == "openai" and not a.endpoints_file:
        raise SystemExit("--llm openai needs --endpoints-file: the model job's servers")
    a.populations, a.workers = layout(knobs, a.sim, a.populations, a.workers)
    n = knobs["agents"]["count"]
    out = Path(a.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}-autonmi-{n}-{os.getpid()}").resolve()
    a.out = str(out)
    own(out)
    t0, t_run = time.perf_counter(), time.time()
    work_root, threads = None, None
    if a.sim == "real":
        from fusion_miniapp.sim import work_root as sim_work_root

        # before any worker starts: spawned workers inherit these, as run.Services sets them
        t = str(run.thread_limit(knobs))
        os.environ.update({v: t for v in run.THREAD_VARS}, FUSION_MINIAPP_THREADS=t)
        threads = {"per_worker": int(t), "cpus": run.cpus(), "knob": knobs["sim"]["threads_per_worker"]}
        if not knobs["sim"]["keep_artifacts"]:   # one folder for all evaluations' work, removed at the end
            work_root = tempfile.mkdtemp(prefix="fusion_miniapp_autonmi_", dir=sim_work_root(knobs["sim"]))
    rc, interrupted, crashed, loop_lag, device = 1, False, False, None, None
    try:   # the campaign configures the OpenTelemetry export (CampaignConfig.telemetry's defaults)
        device = probe(a)   # first: a broken torch or device stops the run before any campaign starts
        rc, interrupted, loop_lag = asyncio.run(run_until_signalled(a, knobs, out, work_root))
    except KeyboardInterrupt:   # SIGTERM or Ctrl-C before the event loop took the signals over
        interrupted = True
    except Exception:           # recorded as a failed run, with what finished
        traceback.print_exc()
        crashed = True
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
    rec = write_record(a, knobs, out, wall_s=time.perf_counter() - t0, t_run=t_run, interrupted=interrupted,
                       crashed=crashed, rc=rc, loop_lag=loop_lag, device=device, threads=threads)
    print(json.dumps({"run_record": str(out / "run_record.json"), "status": rec["status"], "agent_layer": "autonmi",
                      **rec["counts"], "touch_check": rec["touch_check"], "devices": rec["devices"],
                      "wall_seconds": rec["wall_seconds"], "peak_rss_mb": rec["peak_rss_mb"]}), flush=True)
    print(record.speed_line(rec), flush=True)
    for reason in rec["llm_check"]["fail"]:
        print(f"run failed: {reason}", file=sys.stderr)
    for name in rec["autonmi"]["failed_checks"]:
        print(f"run failed: AUTONMI's check {name}", file=sys.stderr)
    return {"pass": 0, "interrupted": 143}.get(rec["status"], 1)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, run._interrupt)   # until the event loop takes the signals over
    sys.exit(main())
