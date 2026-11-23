"""The third agent layer: fusion's campaigns on AUTONMI's framework, as a workflow mini-app.

    python -m fusion_miniapp.layer_autonmi --preset smoke --sim fixture --agents 100 --populations 2

    # under Dragon: populations as Dragon processes, simulations on RHAPSODY's native backend, RaDEX
    python -m dragon -s src/fusion_miniapp/layer_autonmi.py --runtime dragon --populations 2 --workers 16 ...

Every campaign is the same three roles as in the other two layers (decide.py's Coordinator, Generator
and Evaluator), with the same messages in the same order, so its result is the same: `results_sha256`
equals the other layers' for the same knobs. What runs them is AUTONMI's:
- The mini-app is an AsyncFlow application. Its campaigns are split into populations (`--populations`;
  by default one per agent process of run.py, or the next count up that divides the campaigns: layout),
  each an AsyncFlow node with an engine of its own for its simulations, a fleet for its model calls, and
  Flowgentic's agent host. Locally the populations run in this process; under Dragon (`--runtime dragon`)
  each is a Dragon process, its simulations run on RHAPSODY's native Dragon backend, and the store is RaDEX.
- A campaign is three AUTONMI scripted agents (`Script`), a<i>/coordinator, a<i>/generator and
  a<i>/evaluator, each with its own state, model calls, lookups and row, talking only through messages
  (`Mailbox`, one per population). The Evaluator's P evaluations are one fan-out of suspending `evaluate`
  task tools: it is saved and dropped until all P end, the parent's generation barrier, as in the LangGraph
  layer. The Generator's model call is awaited inside its step, as decide.generator_step is written. All
  three stay live until their campaign ends; with agents.start_together none begins generation 0 until
  every agent of its population has started. Every id seen outside the agents is the campaign's, a<i>.
- Model calls go through Flowgentic's fleet: fusion's stub LLM as a transport, or (`--llm openai`) one
  Flowgentic `HTTPTransport` per server of the model job (`--endpoints-file`), which waits out a server
  that goes away instead of retrying (`llm.max_retries` does not apply). The fleet admits at most
  `llm.max_in_flight` requests and `llm.admission_tokens` tokens per server at once, split over the
  populations. Before any campaign, each server must list `llm.model` in its /v1/models; its vLLM
  /metrics are sampled through the run (AUTONMI's VllmSampler, into vllm/vllm.json), unless
  --no-vllm-metrics.
- The head process prints one line to stderr per record.ticker interval (LiveLine): agents live, model
  requests open and queued at the client, sequences decoding at the model servers, agents done. The
  record keeps the same values as a series, and the counts' maxima, means and times at or above shares of
  the hosted agents and the admission capacity (`speed`; write_record).

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
import statistics
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

from autonmi import AgentSpec, Campaign, CampaignConfig, Check
from autonmi.population import PopulationConfig
from autonmi.scripted import Done, Lookup, Remember, Report, ScriptedAgent, Simulate, SimulateAll
from flowgentic.fleet import Reply, Usage
from flowgentic.host import AgentState, activity, current_agent
from flowgentic.tools import task_tool
from pydantic import BaseModel
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend
from wfMiniAPI.api import MiniApp

from fusion_miniapp import decide, knobs as knob_files, record, run, tools, trace
from fusion_miniapp.llm import REPLY_TOKENS, StubLLM

# The AUTONMI framework's commit this layer is tested on: origin/MTU-188-framework (its PR #16). An editable
# install runs whatever its clone has checked out (MTU-188-retry, for one, turns on reruns), so the run record
# names the commit imported, and a run on another commit, or on changed files, warns once (autonmi_pin).
AUTONMI_COMMIT = "e20861a8cf5d996c4b1fb332f57d7758bfeb4c75"
# Under `python -m dragon -s .../layer_autonmi.py` (and `python -m`) this file runs as __main__. The
# definition and the simulation pools' factory are taken from the module of this name instead, so the
# populations' processes can unpickle them, and so _POOLS is the list the factory fills.
MODULE = "fusion_miniapp.layer_autonmi"
_POOLS: list = []   # this process's simulation pools, told to stop rebuilding on SIGTERM
_OWN_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)   # the run's, not its engines' (run_until_signalled)


class _RunningSet(set):
    """AsyncFlow's set of running components, whose iteration yields only the members that have completed.

    AsyncFlow's engine loop finds completions by checking every running component's future on every pass
    (`for comp_uid in list(self.running)`, workflow_manager.py:1679-1684 at radical.asyncflow 3f18cc8): a cost
    that grows with the square of a run's tasks (2,000 campaigns: 209.5 s, against 82 s with this set; 10,000:
    2,039 s against 403 s; the same results). The loop iterates `running` only there, and only to find the
    completed ones, so a set that records completions by done-callback gives it the same answer. Remove once
    AsyncFlow tracks completions itself. FUSION_ASYNCFLOW_RUNNING_SHIM=0 turns it off."""

    def __init__(self, engine, members=()) -> None:
        super().__init__()
        self._engine, self._done = engine, set()
        for uid in members:
            self.add(uid)

    def add(self, uid) -> None:
        super().add(uid)
        self._engine.components[uid]["future"].add_done_callback(lambda _f, uid=uid: self._done.add(uid))

    def discard(self, uid) -> None:
        super().discard(uid)
        self._done.discard(uid)

    def clear(self) -> None:
        super().clear()
        self._done.clear()

    def __iter__(self):
        return iter([uid for uid in self._done if set.__contains__(self, uid)])


def _install_running_shim() -> bool:
    """Give every AsyncFlow engine of this process a _RunningSet (its `running` must be a plain set after
    __init__, or the engine keeps it, with one warning). Idempotent: this file runs as __main__ and as MODULE."""
    if os.environ.get("FUSION_ASYNCFLOW_RUNNING_SHIM", "1") == "0":
        return False
    engine_cls = WorkflowEngine
    if getattr(engine_cls, "_fusion_running_shim", False):
        return True
    init, warned = engine_cls.__init__, []

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if type(getattr(self, "running", None)) is set and isinstance(getattr(self, "components", None), dict):
            self.running = _RunningSet(self, self.running)
        elif not warned:
            warned.append(True)
            print("[fusion/autonmi] AsyncFlow's engine has no plain `running` set: its scan is left as it is",
                  file=sys.stderr, flush=True)

    engine_cls.__init__ = __init__
    engine_cls._fusion_running_shim = True
    return True


RUNNING_SHIM = _install_running_shim()   # recorded in the run record's `autonmi` section

ROLES = ("coordinator", "generator", "evaluator")   # a campaign's hosted agents, a<i>/<role>, in this order
AGENTS_PER_CAMPAIGN = len(ROLES)
FAIL_ENV = "FUSION_AUTONMI_FAIL"   # tests only: "<agent id>:<g>" fails that agent before its generation-g message


def campaign_of(agent_id: str) -> str:
    """The campaign of a hosted agent (a3/generator -> a3): the id seen everywhere outside the agents."""
    return agent_id.partition("/")[0]


def messages(knobs: dict) -> dict:
    """The messages each role sends and receives in one campaign of G generations: each generation the
    Coordinator sends make/g and evaluate/g, the Generator population/g and the Evaluator results/g; at the
    end the Coordinator sends done to the other two. 4G + 2 in all."""
    G = knobs["outer"]["generations"]
    return {"coordinator": {"sent": 2 * G + 2, "received": 2 * G}, "generator": {"sent": G, "received": G + 1},
            "evaluator": {"sent": G, "received": G + 1}}


class CampaignFailed(RuntimeError):
    """Another agent of the campaign failed: this one ends with an error instead of waiting for it."""


class InjectedFailure(RuntimeError):
    """FAIL_ENV's failure (tests only)."""


class Mailbox:
    """The messages between one population's agents (locally every population runs in one process, so one
    per population, in pop.state). A message is a JSON copy, as between processes, kept under its exact key
    <campaign>/<to>/<kind>/<g> in an asyncio future: `send` resolves it (making it, when the receiver has not
    asked yet), `receive` awaits it and deletes it. A second send of a key not yet received raises (a send
    after its receipt would stay in the box, which rows() reports). A waiting agent awaits inside its own
    step, counted by Flowgentic as waiting on a tool (TOOL_WAIT), and is woken once, never polled.
    `fail(campaign)` fails that campaign's waits, now and later, and drops its messages not yet received
    (counted per receiver in `dropped`). With agents.start_together, `all_started` is set when every agent
    of the population has started."""

    def __init__(self, agents: int) -> None:
        self.agents, self.started, self.all_started = agents, 0, asyncio.Event()
        self.box: dict[str, asyncio.Future] = {}
        self.failed: dict[str, str] = {}   # campaign -> why
        self.sent, self.received, self.waited = Counter(), Counter(), Counter()   # per agent; waited: seconds
        self.dropped = Counter()   # per receiving agent: messages sent but not received when its campaign failed

    def start(self) -> None:
        self.started += 1
        if self.started == self.agents:
            self.all_started.set()

    def _future(self, key: str) -> asyncio.Future:
        f = self.box.get(key)
        if f is None:
            f = self.box[key] = asyncio.get_running_loop().create_future()
        return f

    def send(self, sender: str, to: str, kind: str, g: int, value) -> None:
        campaign = campaign_of(sender)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        key = f"{campaign}/{to}/{kind}/{g}"
        f = self._future(key)
        if f.done():
            raise RuntimeError(f"message {key} sent twice")
        f.set_result(json.dumps(value, default=str))   # as the LangGraph layer reads a task's result
        self.sent[sender] += 1

    async def receive(self, me: str, kind: str, g: int):
        campaign = campaign_of(me)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        key, t = f"{me}/{kind}/{g}", time.perf_counter()
        try:
            with activity(AgentState.TOOL_WAIT):
                text = await self._future(key)
        finally:
            self.waited[me] += time.perf_counter() - t
            self.box.pop(key, None)
        self.received[me] += 1
        return json.loads(text)

    def fail(self, campaign: str, why: str) -> None:
        if campaign in self.failed:
            return
        self.failed[campaign] = why
        for key in [k for k in self.box if k.startswith(campaign + "/")]:
            f = self.box.pop(key)
            if not f.done():   # an agent waits on it
                f.set_exception(CampaignFailed(why))
            else:              # sent, never received
                self.dropped[key.rsplit("/", 2)[0]] += 1

    def left(self) -> Counter:
        """Messages never received, per receiving agent."""
        return Counter(key.rsplit("/", 2)[0] for key in self.box)


class EvaluateArgs(BaseModel):
    params: dict[str, float | str]     # the parent's full candidate, with its "model" and "kind" strings
    candidate_idx: int
    generation: int
    seed: int


class StubTransport:
    """Fusion's stub LLM as a Flowgentic transport. Its replies depend on the campaign (of the agent
    Flowgentic's host runs the call for) and the request's last line only, so the three layers agree.
    Each call takes `latency_s` (llm.stub_latency_s): an emulated model server's time, not its decoding."""

    model = "fusion-stub"

    def __init__(self, latency_s: float = 0.001) -> None:
        self.stub = StubLLM(latency_s=latency_s)

    async def complete(self, messages, tools=None, *, extra_body: dict | None = None, max_tokens: int | None = None,
                       **kwargs) -> Reply:
        b = int((extra_body or {}).get("thinking_token_budget", 0))
        r = max_tokens - b if max_tokens is not None else REPLY_TOKENS
        text, meta = await self.stub.call_with_meta(campaign_of(current_agent() or ""), list(messages), b,
                                                    reply_tokens=r)
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


class FixturePool(run.FixturePool):
    """run.py's fixture threads, each running the simulations' coroutines on one event loop of its own.

    Flowgentic makes every simulation a coroutine (AsyncFlow runs async tasks only), and RHAPSODY's thread
    backend runs each one with asyncio.run (`_run_in_thread`). On Python 3.12 asyncio.run ends by collecting
    every task of the process (asyncio.all_tasks walks one process-wide set), the live campaigns' included:
    a cost per evaluation that grows with the campaigns, so a run grew about with their square (2,000
    campaigns in the hundredk shape: 2,053 s, against 180 s on one loop per thread). RHAPSODY's own fix
    (reuse_event_loop, radical-cybertools/rhapsody#163) is not released; once it is, PopulationConfig(
    reuse_event_loop=True) replaces this class. Every other call is submitted unchanged, and so is this one
    if RHAPSODY stops making it. The fixture's coroutines leave no tasks behind on the loop."""

    def __init__(self, workers: int) -> None:
        super().__init__(workers)
        self._local = threading.local()
        self._loops: list = []
        self._loops_lock = threading.Lock()

    def submit(self, fn, /, *args, **kwargs):
        if fn is ConcurrentExecutionBackend._run_in_thread and len(args) == 3 and not kwargs \
                and asyncio.iscoroutinefunction(args[0]):   # RHAPSODY's own test, so both agree
            return super().submit(self._on_thread_loop, *args)
        return super().submit(fn, *args, **kwargs)

    def _on_thread_loop(self, func, args, kwargs):
        loop = getattr(self._local, "loop", None)
        if loop is None:
            loop = self._local.loop = asyncio.new_event_loop()
            with self._loops_lock:
                self._loops.append(loop)
        return loop.run_until_complete(func(*args, **kwargs))

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait:   # the threads have ended: their loops can close
            with self._loops_lock:
                loops, self._loops = self._loops, []
            for loop in loops:
                loop.close()


def fixture_pool(workers: int):
    """The executor of a population's fixture evaluations: run.py's threads, which drop their queued work
    when they close (an interrupted run can hold tens of thousands of queued evaluations), each with an
    event loop of its own (FixturePool)."""
    return FixturePool(workers)


class Fusion:
    """Fusion as an AUTONMI agent definition (autonmi.AgentDefinition): its campaigns, how a population
    runs them (Script), their rows and the checks on them."""

    name = "fusion"

    def __init__(self, knobs: dict, sim: str, out: Path, work_root: str | None = None) -> None:
        self.knobs, self.sim, self.out, self.work_root = knobs, sim, Path(out), work_root
        self.populations: dict[str, asyncio.Task] = {}   # p000...: the task running each population here (build)
        self.live: Counter = Counter()   # agents started and ended in this process (LiveLine)

    def agents(self):
        """Each campaign's three agents, one after another (AUTONMI cuts this list into contiguous
        populations, which layout() keeps on campaign boundaries)."""
        first = self.knobs["agents"]["first"]   # as the other layers: campaign a<i>, seed i
        for i in range(first, first + self.knobs["agents"]["count"]):
            for role in ROLES:
                yield AgentSpec(f"a{i}/{role}", {"seed": i})

    def services(self, run_dir: Path, servers: int = 1) -> tuple[list, list[str]]:
        return [], []   # no MCP servers: the lookups run in the campaign's process (tools.lookup)

    def stub(self, replica: int) -> StubTransport:
        return StubTransport(self.knobs["llm"]["stub_latency_s"])

    def sim_knobs(self, seed: int) -> dict:
        """The `sim` knobs one campaign's evaluations get: its cost scale, and this run's own work folder."""
        s = decide.sim_knobs(self.knobs, seed)
        if self.work_root is not None:
            s["work_root"] = self.work_root
        return s

    def build(self, pop) -> ScriptedAgent:
        """Called in the task that runs the population (locally, a block of the run's engine), which is
        kept so that an interrupted run can wait for the population to end (miniapp_workflow)."""
        self.populations[f"p{pop.index:03d}"] = asyncio.current_task()
        return Script(self, pop).agent()

    def rows(self, pop) -> list[dict]:
        """At the population's end, in its process: one row per hosted agent, with its role and campaign,
        whether it ended normally (`finished`; a Coordinator only once its campaign reported), the evaluations
        it submitted (the Evaluator's, all of which end before it goes on: AUTONMI checks them against the
        tasks its engine completed), its lookups, its messages sent, received, left unreceived and dropped
        when its campaign failed, and its seconds waiting for messages; the Coordinator's also its wait for
        every agent to start and when it began its campaign (`t_begin`, after that wait)."""
        sc = pop.state.get("script")
        if sc is None:   # the population ended before its agents were built
            return [{"agent_id": a, "finished": False, "simulations": 0, "population": pop.index}
                    for a in pop.agent_ids]
        sc.lines.close()
        mail, left, out = sc.mail, sc.mail.left(), []
        for a in pop.agent_ids:
            campaign, role = campaign_of(a), a.rpartition("/")[2]
            row = {"agent_id": a, "campaign": campaign, "role": role, "finished": a in sc.finished,
                   "simulations": sc.sims.get(a, 0), "population": pop.index, "lookups": sc.lookups_made[a],
                   "messages_sent": mail.sent[a], "messages_received": mail.received[a], "messages_left": left[a],
                   "messages_dropped": mail.dropped[a], "message_wait_s": round(mail.waited[a], 6)}
            if role == "coordinator":   # the campaign's model calls by its own count (decide.finish)
                row.update(llm_calls=sc.llm_calls.get(campaign), start_wait_s=sc.start_wait.get(a),
                           t_begin=sc.began.get(a))
            elif role == "evaluator":   # the inner-loop traces it kept (monitor.capture)
                row["traces"] = sc.traced[a]
            out.append(row)
        return out

    def verify(self, rows: list[dict], store) -> list[Check]:
        """AUTONMI's checks of the campaigns' agents; the run record checks their results (touches, the model).
        Per role, exactly: evaluations, model calls (AUTONMI's count of each agent's calls: the Coordinator's
        summary, the Generator's refills as many as its campaign counted, none by the Evaluator), lookups and
        messages; every message received (none left); the lookups against AUTONMI's own count; and the store
        at the end: no agent's saved state (ScriptRunner's script/p<index>/<agent>, kept while it is
        suspended), and each reported campaign's memory under its campaign id, campaign/a<i>."""
        n, exp, msgs = self.knobs["agents"]["count"], decide.expected_by_role(self.knobs), messages(self.knobs)
        coordinators = {r["campaign"]: r for r in rows if r.get("role") == "coordinator"}

        def planned(r: dict) -> bool:
            e, m = exp[r["role"]], msgs[r["role"]]
            calls = {"coordinator": 1, "evaluator": 0,
                     "generator": (coordinators[r["campaign"]].get("llm_calls") or 0) - 1}[r["role"]]
            return (r["simulations"] == e["sims"] and r["model_calls"] == calls <= e["llm_calls"]
                    and r["lookups"] == len(e["lookups"]) and (r["messages_sent"], r["messages_received"]) ==
                    (m["sent"], m["received"]))
        done = [r for r in rows if r["finished"]]
        off = [r["agent_id"] for r in done if not planned(r)]
        finished = {role: sum(r["finished"] for r in rows if r.get("role") == role) for role in ROLES}
        runners = [Path(self.out) / "populations" / f"p{i:03d}" / "population.json"
                   for i in sorted({r["population"] for r in rows})]
        autonmi = sum(json.loads(f.read_text())["runner"]["lookups"] for f in runners) \
            if all(f.exists() for f in runners) else None
        ours = sum(r.get("lookups", 0) for r in rows)
        saved = sum(store.contains(f"script/p{r['population']:03d}/{r['agent_id']}") for r in rows)
        reported = [r["campaign"] for r in done if r.get("role") == "coordinator"]
        memories = sum(store.contains(f"campaign/{c}") for c in reported)
        return [Check("every_campaign_reports", finished["coordinator"] == n,
                      {"reported": finished["coordinator"], "campaigns": n}),
                Check("every_agent_finishes", sum(finished.values()) == AGENTS_PER_CAMPAIGN * n,
                      {"finished": finished, "per_role": n}),
                Check("evaluations_and_model_calls_as_planned", not off,
                      {"per_campaign": exp, "messages": msgs, "not_as_planned": off[:10]}),
                Check("every_message_received", not any(r.get("messages_left") for r in rows),
                      {"left": sum(r.get("messages_left", 0) for r in rows)}),
                Check("lookups_match_autonmi", ours == autonmi, {"rows": ours, "autonmi": autonmi}),
                Check("store_holds_only_the_memories", saved == 0 and memories == len(reported),
                      {"saved_states_left": saved, "memories": memories, "campaigns_reported": len(reported)})]


class Script:
    """One population's campaigns as AUTONMI scripted agents (autonmi.scripted), three per campaign:
    decide.py's Coordinator, Generator and Evaluator, each a hosted agent of its own (a<i>/<role>) with its
    own state, model calls (under its own id) and lookups, which talks to the other two only through the
    population's Mailbox. One generation g:

        Coordinator -> Generator    make/g        initialize (g = 0, with the constraints lookup) or repopulate
        Generator -> Coordinator    population/g  decide.generator_step: in llm mode, its refill call
        Coordinator -> Evaluator    evaluate/g
        Evaluator                   the P evaluations at once (SimulateAll): suspended until all end
        Evaluator -> Coordinator    results/g     the Coordinator's selection, best-so-far and log; its memory

    then make/g+1, or after the last generation the Coordinator's summary call, done/G to the other two,
    and its Report. The Generator makes the abstract lookup when it starts (decide.LOOKUP_ROLE); the
    Coordinator makes the constraints lookup, after waiting (agents.start_together) for every agent of the
    population to start. The others end on done, so all three stay live until their campaign ends.

    Every id seen outside the agents is the campaign's, a<i>: the stub's key, the evaluations' agent_id
    (their cold starts and work folders), the lookups, the store key, the campaign's folder and its line
    (its result, or its error), which goes to the population's agents-p<index>.jsonl.

    With monitor.capture each evaluation's record carries its inner-loop trace (trace.py). The Evaluator
    takes the traces out of the results before it sends them, so decide.py (whose collect refuses a trace),
    the parent's files and the campaign's line never see one. `traces` (campaign -> generation -> the P
    traces, in candidate order) indexes them for the Monitor until the campaign's Evaluator ends. Popping it
    frees no memory: AsyncFlow and RHAPSODY keep each evaluate task's returned record, and RHAPSODY a text
    copy of it (its `stdout`), until the population ends, so the traces are held per population."""

    def __init__(self, definition: Fusion, pop) -> None:
        self.knobs, self.sim, self.out, self.pop = definition.knobs, definition.sim, definition.out, pop
        self.live = definition.live
        self.send_budget = self.knobs["llm"].get("send_thinking_budget", True)
        self.G, self.roles = self.knobs["outer"]["generations"], decide.expected_by_role(self.knobs)
        self.together = self.knobs["agents"]["start_together"]
        self.mail = Mailbox(len(pop.agent_ids))
        self.calls: dict[str, list] = {}    # each campaign's model calls, until its line is written
        self.sims: Counter = Counter()      # the evaluations each Evaluator submitted
        self.lookups_made: Counter = Counter()
        self.reported: set[str] = set()     # campaigns
        self.finished: set[str] = set()     # agents that ended normally
        self.llm_calls: dict[str, int] = {}   # each reported campaign's n_llm_calls
        self.start_wait: dict[str, float] = {}   # each Coordinator's wait for every agent to start
        self.began: dict[str, float] = {}        # when each Coordinator began its campaign (time.time())
        self.traces: dict[str, dict[int, list]] = {}   # each live campaign's traces, by generation
        self.traced: Counter = Counter()         # the traces each Evaluator kept
        self.fail_at = os.environ.get(FAIL_ENV)
        self.lines = open(self.out / f"agents-p{pop.index:03d}.jsonl", "w", buffering=1)  # noqa: SIM115 (rows())
        pop.state.update(script=self, mailbox=self.mail)
        self.evaluate = task_tool(
            run.sim_function(definition.sim), suspend=True, name="evaluate", args_schema=EvaluateArgs,
            description="Evaluate one candidate profile with the turned-down IGNOS inner loop.",
            prepare=lambda agent_id, call_id, args: {**args, "knobs": definition.sim_knobs(args["seed"]),
                                                     "monitor": self.knobs["monitor"],
                                                     "agent_id": campaign_of(agent_id)})

    def agent(self) -> ScriptedAgent:
        """One definition for all three roles (AUTONMI has one per population), so one model-call limit: the
        largest role's. verify() checks each role's exact count."""
        return ScriptedAgent(start=self.start, step=self.step, simulate=self.evaluate, lookup=self.lookup,
                             remember=self.remember, report=self.report, on_call=self.on_call,
                             on_done=self.on_done, model_calls_limit=max(r["llm_calls"] for r in self.roles.values()))

    # --- the steps --------------------------------------------------------------------------------------

    def start(self, agent_id: str, spec: dict) -> dict:
        """An agent's first state, when the host starts it."""
        self.live["started"] += 1
        self.mail.start()
        role = agent_id.rpartition("/")[2]
        s = {"agent_id": agent_id, "campaign": campaign_of(agent_id), "role": role, "seed": spec["seed"]}
        if role == "evaluator":
            return {**s, "phase": "waiting", "g": 0}
        return {**s, "phase": "start" if role == "coordinator" else "look", "t_start": time.time(),
                "todo": self.roles[role]["lookups"], "ctx": {}}

    async def step(self, s: dict, event, ask):
        return await {"coordinator": self._coordinator, "generator": self._generator,
                      "evaluator": self._evaluator}[s["role"]](s, event, ask)

    async def _coordinator(self, s: dict, event, ask):
        me, campaign, phase = s["agent_id"], s["campaign"], s["phase"]
        run_dir = decide.campaign_dir(self.out, self.knobs, campaign)
        if phase == "start":
            if self.together:   # every agent of the population started, so every one is live at once
                t = time.perf_counter()
                with activity(AgentState.TOOL_WAIT):
                    await self.mail.all_started.wait()
                self.start_wait[me] = round(time.perf_counter() - t, 6)
            self.began[me] = time.time()
            s, phase = {**s, "phase": "look"}, "look"
        if phase == "look":
            todo, ctx = s["todo"], s["ctx"]
            if event is not None:                               # the previous lookup's result
                todo, ctx = todo[1:], {**ctx, todo[0]: event}
            if todo:
                return {**s, "todo": todo, "ctx": ctx}, Lookup(todo[0])
            cst = decide.coordinator_state(campaign, s["seed"], self.knobs, ctx, s["t_start"])
            return await self._generation(s, cst, decide.start(cst, self.knobs, run_dir, self.sim), run_dir)
        if phase == "remembering":
            cst = s["cst"]
            if s["next"] is not None:
                return await self._generation(s, cst, s["next"], run_dir)
            summary = await self._model(campaign, ask, decide.summary_messages(cst), cst["budget"],
                                        decide.SUMMARY_TOKENS, "coordinator")
            for to in ("generator", "evaluator"):
                self._send(me, to, "done", self.G, None)
            return _state(s, "reporting"), Report(decide.finish(cst, summary, run_dir))
        if phase == "reporting":
            return s, Done()
        raise ValueError(f"unknown phase {phase!r}")

    async def _generation(self, s: dict, cst: dict, msg: dict, run_dir):
        """The Coordinator's generation: the Generator's population (at g = 0 with the count of its own
        lookups), then the Evaluator's results, then selection, best-so-far and log, and the memory."""
        me, g = s["agent_id"], cst["generation"]
        self._send(me, "generator", "make", g, msg)
        reply = await self.mail.receive(me, "population", g)
        cst = {**cst, "n_tool_calls": cst["n_tool_calls"] + reply.get("lookups", 0)}
        cst, msg = decide.on_population(cst, reply)
        self._send(me, "evaluator", "evaluate", g, msg)
        results = await self.mail.receive(me, "results", g)
        cst, nxt = decide.on_evaluated(cst, results, self.knobs, run_dir)
        return _state(s, "remembering", cst=cst, next=nxt), Remember(decide.memory(cst))

    async def _generator(self, s: dict, event, ask):
        """Its lookups, then each generation's population on make/g (decide.generator_step), then done."""
        me, campaign, todo, ctx = s["agent_id"], s["campaign"], s["todo"], s["ctx"]
        if event is not None:
            todo, ctx = todo[1:], {**ctx, todo[0]: event}
        if todo:
            return {**s, "todo": todo, "ctx": ctx}, Lookup(todo[0])
        gst = decide.generator_state(campaign, s["seed"], self.knobs)
        ask_model = lambda m, b, r: self._model(campaign, ask, m, b, r, "generator")  # noqa: E731
        for g in range(self.G):
            msg = await self.mail.receive(me, "make", g)
            if g == 0:   # the Coordinator's lookups, and its own: the refill prompt reads both
                msg = {**msg, "context": {**msg.get("context", {}), **ctx}}
            gst, reply = await decide.generator_step(gst, msg, self.knobs, ask_model)
            self._send(me, "coordinator", "population", g, {**reply, "lookups": len(ctx)} if g == 0 else reply)
        await self.mail.receive(me, "done", self.G)
        return s, Done()

    async def _evaluator(self, s: dict, event, ask):
        """On evaluate/g, the generation's P evaluations at once, suspended until all end; their results, in
        candidate order, to the Coordinator; after the last generation, done."""
        me, g = s["agent_id"], s["g"]
        if s["phase"] == "evaluating":   # resumed: the generation's evaluations, all ended
            results, traces = trace.split(event)   # the traces stay here, out of decide's way
            kept = sum(t is not None for t in traces)
            if kept:
                self.traces.setdefault(s["campaign"], {})[g] = traces
                self.traced[me] += kept
            self._send(me, "coordinator", "results", g,
                       {"results": results, "t0": s["t0"], "elapsed_s": time.time() - s["t0"]})
            g += 1
        if g == self.G:
            await self.mail.receive(me, "done", self.G)
            return s, Done()
        msg = await self.mail.receive(me, "evaluate", g)
        calls = decide.evaluation_calls(msg["population"], msg["generation"])
        self.sims[me] += len(calls)
        batch = tuple(Simulate({"params": c["params"], "candidate_idx": c["candidate_idx"],
                                "generation": c["generation"], "seed": s["seed"]}, call_id=c["id"]) for c in calls)
        return {**s, "phase": "evaluating", "g": g, "t0": time.time()}, SimulateAll(batch)

    def _send(self, me: str, to: str, kind: str, g: int, value) -> None:
        if self.fail_at == f"{me}:{g}":   # the tests' injected failure (FAIL_ENV)
            raise InjectedFailure(f"{me} failed before sending {kind}/{g} ({FAIL_ENV})")
        self.mail.send(me, to, kind, g, value)

    async def _model(self, campaign: str, ask, messages: list[dict], budget: int, reply_tokens: int,
                     role: str) -> str:
        """decide.Ask through the population's fleet, with the request run.py's OpenAIChat sends:
        max_tokens = thinking budget + reply allowance, vLLM's thinking_token_budget (unless
        llm.send_thinking_budget is false), and no temperature (AUTONMI's transports set 0; None leaves the
        server's default). Admission reserves the budget. A call that fails is recorded, as run.Services
        records it. `role`: the role whose call it is (generator: a refill; coordinator: the summary)."""
        kwargs = {"max_tokens": budget + reply_tokens, "reserve": budget + reply_tokens, "temperature": None,
                  "meta": {"budget": budget, "reply_tokens": reply_tokens, "role": role}}
        if self.send_budget:
            kwargs["extra_body"] = {"thinking_token_budget": budget}
        t0 = time.perf_counter()
        try:
            return await ask(messages, **kwargs)
        except Exception as exc:
            self.calls.setdefault(campaign, []).append(
                {"role": role, "reply_tokens": reply_tokens, "budget": budget,
                 "latency_s": round(time.perf_counter() - t0, 4), "error": type(exc).__name__, "finish_reason": None,
                 "empty": None})
            raise

    # --- the actions' services ----------------------------------------------------------------------------

    async def lookup(self, agent_id: str, what: str) -> dict:
        """Counted once it returns, as AUTONMI's runner counts it (lookups_match_autonmi)."""
        out = await tools.lookup(what, campaign_of(agent_id), self.knobs["tools"]["latency_ms"])
        self.lookups_made[agent_id] += 1
        return out

    def remember(self, agent_id: str, state: dict) -> None:
        """The Coordinator's memory after each generation, in the population's store (RaDEX under Dragon)."""
        self.pop.store.put(f"campaign/{campaign_of(agent_id)}", json.dumps(state, default=str))

    def report(self, agent_id: str, result: dict) -> None:
        """A campaign's result to the campaign manager: its line, as run.Services.report writes it."""
        campaign = campaign_of(agent_id)
        self.reported.add(campaign)
        self.llm_calls[campaign] = result["n_llm_calls"]
        self._line({**result, "llm_call_details": self.calls.pop(campaign, [])})

    def on_call(self, rec, reply: Reply) -> None:
        """A model call that returned: its details, as run.Services keeps them for the campaign's line, with
        its times (Flowgentic's CallRecord): it was in flight at the server from t_end - service_s to t_end,
        after recovery_s of failed tries (latency_s is their sum) and, before those, wait_s for admission."""
        m = rec.meta or {}
        self.calls.setdefault(campaign_of(rec.agent_id), []).append({
            "role": m.get("role"), "reply_tokens": m.get("reply_tokens"), "budget": m.get("budget"),
            "latency_s": round(rec.service_s + rec.recovery_s, 4), "wait_s": round(rec.wait_s, 4),
            "t_end": round(rec.t_end, 6), "service_s": round(rec.service_s, 6), "recovery_s": round(rec.recovery_s, 6),
            "finish_reason": reply.finish_reason, "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens, "reasoning_tokens": rec.reasoning_tokens or 0,
            "empty": not str(reply.content or "").strip()})

    def on_done(self, result, output) -> None:
        """An agent that ended normally is finished. The first of a campaign's agents to fail writes the
        campaign's line, with its error, its role and every role's model calls, as the other layers write a
        failed campaign; the other two then end with an error (CampaignFailed) instead of waiting. One
        cancelled (the run was interrupted) writes nothing."""
        self.live["ended"] += 1
        campaign = campaign_of(result.id)
        if result.id.rpartition("/")[2] == "evaluator":   # its campaign's index of traces ends with it
            self.traces.pop(campaign, None)
        if result.status == "done":
            self.finished.add(result.id)
        elif result.status == "failed":
            if campaign not in self.reported and campaign not in self.mail.failed:
                self._line({"agent_id": campaign, "error": result.error or result.status,
                            "role": result.id.rpartition("/")[2], "llm_call_details": self.calls.pop(campaign, [])})
            self.mail.fail(campaign, f"{result.id} failed: {result.error}")

    def _line(self, line: dict) -> None:
        self.lines.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def _state(s: dict, phase: str, **memory) -> dict:
    """An agent's state: its id, campaign, role, seed and phase, and what the phase needs."""
    return {k: s[k] for k in ("agent_id", "campaign", "role", "seed")} | {"phase": phase, **memory}


# --- the live line and the model service -----------------------------------------------------------------

def _model_states() -> tuple[int, int] | None:
    """Agents with a model request open (admitted and sent) and agents queued for admission, summed over
    this process's agent hosts: Flowgentic's own counts (its AgentStates), read as Flowgentic's
    OpenTelemetry gauge reads them (flowgentic._telemetry._observe_agents). None if Flowgentic stops
    keeping them there."""
    try:
        from flowgentic._agents import AgentState, _all_states

        opened = queued = 0
        for states in list(_all_states):
            counts = states._live_counts()   # None once that host's run has ended
            if counts is not None:
                opened += counts[AgentState.MODEL_REQUEST]
                queued += counts[AgentState.MODEL_QUEUED]
        return opened, queued
    except (ImportError, AttributeError, IndexError):
        return None


class LiveLine:
    """The line the head process prints to stderr once per record.ticker interval, and its series for the
    run record:

        agents live=<n> model_requests_open=<n> queued=<n> decoding_at_servers=<n> done=<n>

    live and done: hosted agents started and not ended, and ended (Script's counts); model_requests_open
    and queued: client side, Flowgentic's agent states (_model_states); decoding_at_servers: the sampler's
    latest reading of the model servers' running batches, summed ("-" without a sampler, as with the stub).
    Under Dragon the populations run in processes of their own, and only the servers' count is seen here."""

    FIELDS = ("t_s", "agents_live", "model_requests_open", "model_requests_queued", "decoding_at_servers", "done")

    def __init__(self, local: bool = True) -> None:
        self.local, self.counts, self.sampler, self.servers, self.served = local, Counter(), None, None, None
        self.t0, self.series, self.why_no_sampler = time.time(), [], None

    def read(self) -> list:
        states = _model_states() if self.local else None
        started, ended = self.counts["started"], self.counts["ended"]
        return [round(time.time() - self.t0, 2), started - ended if self.local else None,
                *(states or (None, None)), self._decoding(), ended if self.local else None]

    def emit(self, row: list) -> None:
        self.series.append(row)
        print("agents live={} model_requests_open={} queued={} decoding_at_servers={} done={}".format(
            *("-" if v is None else v for v in row[1:])), file=sys.stderr, flush=True)

    def _decoding(self) -> int | None:
        """The servers' latest running counts, summed, of those that answered within hold_s (a server that
        stopped answering is left out, as the record counts it, not held at its last reading)."""
        if self.sampler is None:
            return None
        series, now = list(self.sampler.series.values()), time.time()
        hold = hold_s(series, self.sampler.interval, recent=10)
        last = [s["num_requests_running"][-1] for s in series
                if s["t"] and s["num_requests_running"] and now < s["t"][-1] + hold]
        return int(sum(v or 0 for v in last)) if last else None

    @contextlib.asynccontextmanager
    async def during(self, out: Path):
        """The live line, and the sampler (if any) with a first reading before the run, around the run; at
        the end the sampler's last reading (which the last line shows), and its series in
        out/vllm/vllm.json, where AUTONMI's campaign finds it (its checks then compare the servers' token
        counts with the fleets')."""
        if self.sampler is not None:
            await asyncio.to_thread(self.sampler.sample)
            self.sampler.start()
        tick = asyncio.ensure_future(record.ticker(self.read, self.emit))
        interrupted = False
        try:
            yield
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            try:
                if self.sampler is not None:
                    if interrupted:   # SIGTERM: no last sweep (unreachable servers would take 5 s each), what
                        self.sampler._halt.set()   # was sampled is written; the run's TERM grace is short
                        await asyncio.to_thread(self.sampler.join, 5.0)
                    else:
                        await asyncio.to_thread(self.sampler.stop)
                    (out / "vllm").mkdir(parents=True, exist_ok=True)
                    (out / "vllm" / "vllm.json").write_text(json.dumps(self.sampler.raw()))
            finally:
                tick.cancel()
                await asyncio.gather(tick, return_exceptions=True)

    def section(self) -> dict:
        return {"interval_s": record.LIVE_INTERVAL_S, "t0": round(self.t0, 3), "fields": list(self.FIELDS),
                "series": self.series}


class ModelServiceError(RuntimeError):
    """The model servers cannot serve this run's model: the run is refused before any campaign starts."""


def _get(url: str, key: str | None, timeout: float = 10.0) -> bytes:
    headers = {"Authorization": f"Bearer {key}"} if key and key != "EMPTY" else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read()


def check_served_model(model: dict, name: str) -> list[dict]:
    """Every server of the model job lists `name` (llm.model) in its /v1/models, or ModelServiceError. The
    distinct entries they list for it (id and owned_by: who the server says serves it), for the record."""
    from autonmi.models import api_key

    key, served = api_key(model), []
    for url in model["endpoints"]:
        try:
            data = json.loads(_get(url.rstrip("/") + "/v1/models", key)).get("data", [])
        except Exception as exc:   # noqa: BLE001 - any failure refuses the run, with its reason
            raise ModelServiceError(f"{url}/v1/models did not answer: {type(exc).__name__}: {exc}") from None
        ids = [m.get("id") for m in data]
        if name not in ids:
            raise ModelServiceError(f"{url} serves {ids}, not llm.model {name!r}: set llm.model to the name the "
                                    "server serves (vLLM's --served-model-name)")
        entry = next({"id": m.get("id"), "owned_by": m.get("owned_by")} for m in data if m.get("id") == name)
        if entry not in served:
            served.append(entry)
    return served


def vllm_sampler(model: dict, tries: int = 3, pause_s: float = 2.0):
    """AUTONMI's VllmSampler over the model job's servers, every 2 s (5 s past 50 servers: it reads them one
    after another), when every server answers /metrics with vLLM's running count, each given `tries` tries
    `pause_s` apart; else (None, why not). All or none: AUTONMI compares the servers' token counts with all
    the fleets' calls."""
    from autonmi.models import api_key
    from rascal.telemetry.vllm import VllmSampler, fetch, parse

    key, urls = api_key(model), model["endpoints"]
    for url in urls:
        for attempt in range(tries):
            try:
                values, _ = parse(fetch(url, key))
                break
            except Exception as exc:   # noqa: BLE001 - no sampler, with the reason
                why = f"{url}/metrics: {type(exc).__name__}: {exc}"
                if attempt + 1 == tries:
                    return None, why
                time.sleep(pause_s)
        if "num_requests_running" not in values:
            return None, f"{url}/metrics has no vllm:num_requests_running"
    return VllmSampler(urls, key, 2.0 if len(urls) <= 50 else 5.0), None


VLLM_GAUGES = ("t", "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")   # what the record reads
HOLD_INTERVALS = 2.5   # a server's reading counts this many sampling intervals at most: one missed reading is bridged


def hold_s(series: list[dict], interval: float, recent: int | None = None) -> float:
    """How long a model server's reading of /metrics counts (record.summed's hold_s): 2.5 sampling intervals,
    or 2.5 times the servers' median gap between readings (of their last `recent`) when a sweep over them
    takes longer (the sampler reads them one after another). Past it, the server has stopped answering."""
    gaps = []
    for s in series:
        t = s["t"][-(recent + 1):] if recent else s["t"]
        gaps += [b - a for a, b in zip(t, t[1:])]
    return HOLD_INTERVALS * max(interval, statistics.median(gaps) if gaps else 0.0)


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
    p.add_argument("--populations", type=int, help="populations of campaigns; must divide the campaigns "
                                                   "(default: the smallest count that does from run.py's agent "
                                                   "processes, ceil(agents.count / agents.per_process), up to "
                                                   "twice that; else refused)")
    p.add_argument("--workers", type=int, help="simulation workers per population "
                                               "(default: workers.sim_workers / populations)")
    p.add_argument("--runtime", choices=("local", "dragon"), default="local",
                   help="dragon: populations as Dragon processes; run under `python -m dragon -s`")
    p.add_argument("--endpoints-file", help="--llm openai: the model job's servers, one base URL per line "
                                            "(without /v1); each must answer /health")
    p.add_argument("--api-key-file", help="--llm openai: a file holding the servers' key, if they need one")
    p.add_argument("--model-wait", type=float, default=1800.0, help="seconds to wait for every server to answer")
    p.add_argument("--no-vllm-metrics", action="store_true",
                   help="--llm openai: do not sample the servers' /metrics (no decoding counts, and none of "
                        "AUTONMI's checks against the servers' counters, which a server restart, a requeue, a "
                        "request the client gave up on, or servers shared with another run fail)")
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


def _divides(n: int, lo: int, hi: int) -> int | None:
    """The smallest population count from lo to hi that divides n campaigns, or None."""
    return next((p for p in range(lo, min(hi, n) + 1) if n % p == 0), None)


def layout(knobs: dict, sim: str, populations: int | None = None, workers: int | None = None) -> tuple[int, int]:
    """Populations, and simulation workers in each. AUTONMI cuts the agent list into contiguous equal blocks
    (divmod), so a campaign's three agents stay in one population only when the populations divide the
    campaigns: by default (None or 0) the smallest count that does from run.py's agent processes
    (agents.per_process campaigns each) up to twice that, else refused with nearby campaign counts that have
    one; a --populations that does not divide is refused. Workers: by default the population's share of
    workers.sim_workers, which with --sim real must give each one, as run.py requires."""
    n, w = knobs["agents"]["count"], knobs["workers"]["sim_workers"]
    whole = "a campaign's three agents must stay in one population, and AUTONMI splits the agents into equal blocks"
    if populations:
        populations = min(populations, n)
        if populations < 1 or n % populations:
            raise knob_files.KnobError(f"--populations {populations} does not divide the {n} campaigns: {whole}; "
                                       f"give a count that divides {n}")
    else:
        d = min(-(-n // knobs["agents"]["per_process"]), n)
        populations = _divides(n, d, 2 * d)
        if populations is None:
            below = next(m for m in range(n - 1, 0, -1) if _divides(m, d, 2 * d))
            above = next(m for m in range(n + 1, 2 * n + 2) if _divides(m, d, 2 * d))
            raise knob_files.KnobError(
                f"{n} campaigns divide by no population count from {d} (agents.per_process) to {2 * d}: {whole}; "
                f"run {below} or {above} campaigns, or give --populations")
    if workers is None:
        if sim == "real" and populations > w:
            raise knob_files.KnobError(
                f"{populations} populations each need a simulation worker, but workers.sim_workers is {w}; raise "
                "workers.sim_workers or agents.per_process, or give --workers")
        workers = max(1, w // populations)
    return populations, workers


def agents_live(knobs: dict) -> int | None:
    """AUTONMI's cap on the hosted agents live at once per population (None: all): agents.max_live keeps its
    meaning, campaigns, so three agents each. The host starts agents in order and the next only when one ends,
    so a cap below one campaign would leave a campaign's first agents waiting for good on peers that cannot
    start, and agents.start_together on any cap would leave every agent waiting for those past it: both are
    refused. With whole campaigns at most one is partly started, and its other agents start as others end."""
    a = knobs["agents"]
    if a["max_live"] is None:
        return None
    if a["start_together"]:
        raise knob_files.KnobError(f"agents.start_together needs agents.max_live: null, not {a['max_live']}: every "
                                   "agent waits until all have started, and with a cap those past it start only as "
                                   "others end")
    if a["max_live"] < 1:
        raise knob_files.KnobError(f"agents.max_live must hold one campaign ({AGENTS_PER_CAMPAIGN} agents) at least, "
                                   f"not {a['max_live']}")
    return AGENTS_PER_CAMPAIGN * a["max_live"]


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


async def run_miniapp(a: argparse.Namespace, knobs: dict, out: Path, work_root: str | None, live: LiveLine) -> int:
    backend = await make_backend(a)
    flow = await WorkflowEngine.create(backend=backend, work_dir=str(out))
    telemetry = await flow.start_telemetry(checkpoint_path=str(out / "launcher"))
    app = MiniApp(flow, telemetry)
    ok = False
    try:
        async with app, flow.workflow_scope("fusion"):
            ok = await miniapp_workflow(a, knobs, flow, out, app, work_root, live)
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


async def miniapp_workflow(a, knobs: dict, flow, out: Path, app, work_root: str | None, live: LiveLine) -> bool:
    k = knobs["llm"]
    model = None
    if k["mode"] == "openai":   # the model job's servers, once all answer; the thinking budget is per call
        from autonmi.models import wait_for_servers

        model = await wait_for_servers(a.endpoints_file, a.api_key_file, wait_s=a.model_wait, name=k["model"],
                                       timeout_s=k["timeout_s"], thinking_off={})
        live.served = await asyncio.to_thread(check_served_model, model, k["model"])   # before any campaign
        if a.no_vllm_metrics:
            live.why_no_sampler = "--no-vllm-metrics"
        else:
            live.sampler, live.why_no_sampler = await asyncio.to_thread(vllm_sampler, model)
        if live.sampler is None and not a.no_vllm_metrics:
            print(f"[fusion/autonmi] warning: the model servers' decoding counts are not recorded "
                  f"({live.why_no_sampler})", file=sys.stderr, flush=True)
    live.servers = len(model["endpoints"]) if model else 1   # the stub: one replica (PopulationConfig)
    definition = importlib.import_module(MODULE).Fusion(knobs, a.sim, out, work_root)
    definition.live = live.counts
    # locally, the populations' simulations run on run.py's pools; under Dragon, on RHAPSODY's native backend
    executor = f"{MODULE}:{'sim_pool' if a.sim == 'real' else 'fixture_pool'}" if a.runtime == "local" else None
    campaign = Campaign(
        flow,
        definition,
        CampaignConfig(
            run_dir=str(out), populations=a.populations, runtime=a.runtime,
            population=PopulationConfig(
                # a campaign's agents start together, so campaigns start as fast as one agent each did
                workers=a.workers, stagger_batch=AGENTS_PER_CAMPAIGN * a.stagger_batch,
                stagger_interval_s=a.stagger_interval, cap_tokens=k["admission_tokens"],
                cap_requests=k["max_in_flight"], max_live=agents_live(knobs),
                executor=executor, model=model),
        ),
        app=app,
    )
    try:
        await campaign.start_services()
        try:
            async with live.during(out):
                await campaign.run()
        except asyncio.CancelledError:
            # SIGTERM: the campaign cancels its populations, but neither waits for them nor records why, as it
            # does for populations it cancels itself. Each still shuts its engine down and writes its
            # telemetry and its report into the run's; wait for that (at most cancel_grace_s), so that the
            # run's verify() sees every population, and asyncio.run does not cancel one again mid-write. A
            # population cancelled before build (while its engine starts) is not waited for: it reports
            # nothing, and the run's telemetry says so. Under Dragon, build runs in the populations' own
            # processes and nothing is waited for here. Then the check the campaign records for its own
            # cancels, every_population_ran, fails with the populations the signal cancelled (under Dragon,
            # all of them: this process cannot tell which had ended).
            pops, grace, t = definition.populations, campaign.config.cancel_grace_s, time.perf_counter()
            names = [f"p{i:03d}" for i in range(campaign.config.populations)]
            cancelled = [n for n in names if n not in pops or not pops[n].done()]
            pending = [pops[n] for n in cancelled if n in pops]
            if pending:
                await asyncio.wait(pending, timeout=grace)
            running = [n for n in cancelled if n in pops and not pops[n].done()]
            if running:
                print(f"[fusion/autonmi] warning: {running} still running {grace} s (cancel_grace_s) after the "
                      "signal: their telemetry may be cut short", file=sys.stderr, flush=True)
            app.check("every_population_ran", ok=False, detail={
                "failed": {}, "cancelled": cancelled, "timed_out": [], "interrupted": True,
                "waited_s": round(time.perf_counter() - t, 1), "cancel_grace_s": grace, "still_running": running})
            raise
        finally:
            await campaign.stop_services()
        return campaign.check()
    finally:
        await campaign.close()


async def run_until_signalled(a, knobs: dict, out: Path, work_root: str | None,
                              live: LiveLine) -> tuple[int, bool, dict]:
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
        rc = await run_miniapp(a, knobs, out, work_root, live)
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


def autonmi_commit() -> dict:
    """The commit of the autonmi package this process imported: git's, in the clone it is imported from
    (an editable install), with whether tracked files differ; else its installer's record (direct_url.json,
    when installed from git). None when neither says."""
    import subprocess

    import autonmi

    src = Path(autonmi.__file__).resolve().parent

    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    try:
        git("ls-files", "--error-unmatch", "__init__.py")   # the package's own clone, not a repository above it
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "from": "git", "path": str(src)}
    except (OSError, subprocess.SubprocessError):
        pass
    for dist in metadata.packages_distributions().get("autonmi", []):
        vcs = json.loads(metadata.distribution(dist).read_text("direct_url.json") or "{}").get("vcs_info", {})
        if vcs.get("commit_id"):
            return {"commit": vcs["commit_id"], "dirty": None, "from": "direct_url.json", "path": str(src)}
    return {"commit": None, "dirty": None, "from": None, "path": str(src)}


def autonmi_pin() -> dict:
    """autonmi_commit() against AUTONMI_COMMIT, for the run record; one warning on stderr when the commit
    differs (or is unknown) or its tracked files are changed. The clone's path is in the warning only: run
    records name no local path."""
    got = autonmi_commit()
    path = got.pop("path")
    got.update(pin=AUTONMI_COMMIT, matches_pin=got["commit"] == AUTONMI_COMMIT and not got["dirty"])
    if not got["matches_pin"]:
        what = "an unknown commit" if got["commit"] is None else got["commit"][:12] + (
            " with changed files" if got["dirty"] else "")
        print(f"[fusion/autonmi] warning: the autonmi package imported ({path}) is at {what}, not at the "
              f"pinned {AUTONMI_COMMIT[:12]} (layer_autonmi.AUTONMI_COMMIT)", file=sys.stderr, flush=True)
    return got


TASK_COUNTS = ("created", "completed", "failed", "canceled", "abandoned", "not_submitted")


def telemetry_verdict(summary: dict) -> dict:
    """AUTONMI's verdict on the run's telemetry (its check telemetry_complete, in summary.json) and the
    task counts of the populations that reported into the run, each and summed. `abandoned`: tasks
    submitted that never ended, as when SIGTERM stops a run; workflow-mini-apps reports them, and they do
    not by themselves make the telemetry incomplete. `tasks` is None when no population reported."""
    check = next((c for c in summary.get("checks", []) if c["name"] == "telemetry_complete"),
                 {"ok": False, "detail": ["no telemetry_complete check in summary.json"]})
    pops = {p["population"]: {"complete": p.get("telemetry_complete"), "tasks": p.get("tasks")}
            for p in summary.get("per_population", [])}
    tasks = {k: sum((p["tasks"] or {}).get(k, 0) for p in pops.values()) for k in TASK_COUNTS} if pops else None
    return {"complete": bool(check.get("ok")), "reasons": check.get("detail", []), "tasks": tasks,
            "populations": pops}


def references(a, knobs: dict, live: LiveLine | None) -> dict:
    """What the counts of agents and model requests are measured against: the hosted agents (three per
    campaign), and the admission capacity, the requests the fleets admit at once (llm.max_in_flight per
    server, split over the populations as Flowgentic splits it; None when the servers are unknown)."""
    n, cap, servers = a.populations, knobs["llm"]["max_in_flight"], live.servers if live is not None else None
    return {"hosted_agents": AGENTS_PER_CAMPAIGN * knobs["agents"]["count"],
            "admission_capacity": n * max(1, int(cap * (1.0 / n))) * servers if servers else None}


def _rows(out: Path) -> list[dict]:
    """AUTONMI's rows, one per hosted agent (app/agents-p*.jsonl; a population the run cancelled wrote none)."""
    return [json.loads(x) for f in sorted((out / "app").glob("agents-p*.jsonl"))
            for x in f.read_text().splitlines() if x.strip()]


def by_role(rows: list[dict]) -> dict:
    """counts.by_role, from AUTONMI's rows: each role's agents, those that ended normally, their model calls
    (the fleets' count), lookups, messages sent, received and dropped (a failed campaign's), suspensions
    (AUTONMI's: the Evaluator's, one per generation unless every evaluation had ended when it asked), the
    inner-loop traces the Evaluators kept (monitor.capture), and each agent's seconds waiting for messages
    (their sum, and n, mean, median, 95th percentile and maximum over the agents)."""
    out = {}
    for role in ROLES:
        rs = [r for r in rows if r.get("role") == role]
        waits = [r.get("message_wait_s") or 0.0 for r in rs]
        out[role] = {"agents": len(rs), "finished": sum(bool(r.get("finished")) for r in rs),
                     **{k: sum(r.get(k) or 0 for r in rs) for k in ("model_calls", "lookups", "messages_sent",
                                                                     "messages_received", "messages_dropped",
                                                                     "suspensions", "traces")},
                     "message_wait_s": {"sum": round(sum(waits), 3), **record._pct(waits)}}
    return out


def begin_margin_s(rows: list[dict]) -> float | None:
    """The earliest Coordinator's t_begin (when it began its campaign) minus the last t_start of its
    population's hosted agents, the smallest over the populations: at least 0 when every campaign began
    after every agent of its population had started, as agents.start_together promises; below 0 when one
    began before. None without the times."""
    pops: dict = {}
    for r in rows:
        pops.setdefault(r.get("population"), []).append(r)
    margins = []
    for rs in pops.values():
        starts, begins = [r.get("t_start") for r in rs], [r["t_begin"] for r in rs if r.get("t_begin")]
        if not begins or not all(starts):
            return None
        margins.append(min(begins) - max(starts))
    return round(min(margins), 3) if margins else None


def decoding_metric(knobs: dict, out: Path, rec: dict, summary: dict, refs: dict, live: LiveLine | None,
                    rows: list[dict] | None = None) -> dict:
    """The record's `speed` counts besides the client's exact ones (record.write): agents live, from AUTONMI's
    rows (`rows`, else read from out/app), in all and per role, with how long all were live at once
    (`all_live_s`); AUTONMI's own client-side figures (summary.json), with Little's law across the sources; the event
    loop's lag beside the client count, flagged when high; the model servers' running batches, waiting
    requests and KV-cache use (vllm/vllm.json; null with the stub); the headline (decision 17: the servers'
    running count where there are servers, else the client-side count, emulated with the stub); and the live
    line's series."""
    s, stub = rec["speed"], knobs["llm"]["mode"] == "stub"
    timed = [r for r in (_rows(out) if rows is None else rows) if r.get("t_start") and r.get("t_end")]
    starts, ends = [r["t_start"] for r in timed], [r["t_end"] for r in timed]
    lo, hi = (min(starts), max(ends)) if starts else (None, None)
    agents = None
    if starts:
        n = refs["hosted_agents"]
        agents = {"source": "AUTONMI's rows (app/agents-p*.jsonl): each hosted agent's t_start to t_end",
                  "hosted_agents": n, "timed": len(starts),
                  **record.occupancy(starts, ends, lo, hi, refs={"hosted_agents": n}),
                  "ramp_up_s": round(max(starts) - lo, 3),
                  # every hosted agent live at once: from the last start to the first end
                  "all_live_s": round(max(0.0, min(ends) - max(starts)), 3) if len(timed) == n else 0.0}
        roles = [role for role in ROLES if any(r.get("role") == role for r in timed)]
        if roles:   # per role, against its own agents (one per campaign)
            agents["by_role"] = {}
            for role in roles:
                rs = [r for r in timed if r.get("role") == role]
                b, e = [r["t_start"] for r in rs], [r["t_end"] for r in rs]
                agents["by_role"][role] = {
                    "timed": len(rs), **record.occupancy(b, e, lo, hi, refs={"hosted_agents": n // len(ROLES)}),
                    "ramp_up_s": round(max(b) - min(b), 3)}
    phase, secs = s["agent_phase_s"], summary.get("agent_seconds_by_state") or {}

    def built_in(state: str) -> dict:
        x = secs.get(state)
        return {"agent_seconds": x, "mean": round(x / phase, 4) if x is not None and phase else None}
    opened, queued = s.get("model_requests_in_flight"), s.get("model_requests_queued")
    if opened is not None:
        sampled = (summary.get("agent_states_over_time") or {}).get("model_request") or []
        opened["autonmi_sampled"] = {
            "source": "AUTONMI's summary.json: agent_states_over_time.model_request (Flowgentic's agents in "
                      "MODEL_REQUEST, sampled at each export, about every 10 s) and agent_seconds_by_state."
                      "model_request (its agent-seconds) / agent_phase_s",
            "samples": len(sampled), "max_sampled": max(sampled) if sampled else None, **built_in("model_request")}
        rate = opened["calls_timed"] / phase if phase else None
        mean_s = opened["time_s"].get("mean")
        latency = knobs["llm"]["stub_latency_s"] if stub else None
        agent_s, wall = opened["autonmi_sampled"]["agent_seconds"], rec["wall_seconds"]
        opened["littles_law"] = {   # mean in flight = calls per second x time in flight, three ways
            "autonmi_agent_seconds_per_s": opened["autonmi_sampled"]["mean"],
            "fusion_calls_per_s_x_mean_time_in_flight": round(rate * mean_s, 4) if rate and mean_s else None,
            "calls_per_s_x_stub_latency_s": round(rate * latency, 4) if rate and stub else None,
            "autonmi_agent_seconds_per_wall_s": round(agent_s / wall, 4) if agent_s is not None and wall else None,
            "per": "per second of the agent phase (agent_phase_s), the calls' rate too; the last figure per "
                   "second of the whole run (wall_seconds: start-up and shut-down included)"}
        lag = next((x for x in s["loop_lag"] if x and x.get("samples")), None)
        opened["loop_lag"] = None if lag is None or not mean_s else {
            **{k: lag[k] for k in ("p95_ms", "max_ms", "share_over_100ms")},
            "high": lag["p95_ms"] / 1e3 >= 0.1 * mean_s,
            "rule": "high: the loop's 95th-percentile lag is at least 10% of the mean time in flight, which it "
                    "lands inside (the count then measures the loop as well as the model)"}
    if queued is not None:
        queued["autonmi_built_in"] = built_in("model_queued")
    decoding, why = None, ("the stub LLM: no model servers" if stub else
                           live.why_no_sampler if live is not None and live.why_no_sampler else "no vllm/vllm.json")
    vf = out / "vllm" / "vllm.json"
    if vf.exists():
        raw, series = json.loads(vf.read_text()), []
        for x in raw.get("series", {}).values():   # a sweep cut off by SIGTERM: its whole readings only
            n = min(len(x[k]) for k in VLLM_GAUGES)
            series.append({k: x[k][:n] for k in VLLM_GAUGES})
        hold, missing = hold_s(series, raw.get("interval_s") or 2.0), {}

        def gauge(name: str, scale: float = 1.0, references: dict | None = None) -> dict | None:
            """One gauge summed over the servers (record.summed, each reading counted for at most `hold`), or
            None when a server that answered never gave it (as an older vLLM's other names)."""
            lacking = [i for i, x in enumerate(series) if x["t"] and all(v is None for v in x[name])]
            if lacking:
                missing[name] = f"the servers {lacking} (in vllm.json's order) gave no vllm:{name}"
                return None
            t, v = record.summed([(x["t"], x[name]) for x in series], hold)
            return record.sampled(t, v / scale, lo, hi, references)
        times = sorted(t for x in series for t in x["t"])
        if lo is None and times:   # no rows: the sampler's own time
            lo, hi = times[0], times[-1]
        if times and hi > lo:
            running = gauge("num_requests_running", references=refs)
            decoding, why = None if running is None else {
                "source": "the model servers' /metrics, vllm:num_requests_running summed over the servers, sampled "
                          "by AUTONMI's VllmSampler (vllm/vllm.json)",
                "note": "the servers' running batches: prompt processing included (vLLM has no decode-only "
                        "count); total only (a server does not know which agent sent a request)",
                "servers": len(series), "interval_s": raw.get("interval_s"), "sample_errors": raw.get("errors"),
                "hold_s": round(hold, 3),
                "stale_s": [record.unheld_s(x["t"], lo, hi, hold) for x in series],
                "stale_rule": "a server's reading counts until its next one, at most hold_s (2.5 sampling intervals, "
                              "or 2.5 median gaps); stale_s: each server's seconds of the agent phase with no reading "
                              "counted (it stopped answering /metrics, or before its first), counted as 0",
                "running": running,
                "waiting": gauge("num_requests_waiting"),
                "kv_cache_usage": gauge("kv_cache_usage_perc", scale=len(series)),   # the servers' mean, a fraction
                "gauges_missing": missing or None}, missing.get("num_requests_running")
    if decoding is not None:
        headline = {"figure": "decoding_at_servers", "label": "the servers' running count",
                    "max": decoding["running"]["max"], "mean": decoding["running"]["mean"],
                    # who the servers' /v1/models say serves the model (llm_service.served_models)
                    "served_by": sorted({m["owned_by"] for m in (live.served if live is not None else None) or []
                                         if m["owned_by"]})}
    else:
        label = "client side" + (f", emulated (stub LLM, llm.stub_latency_s = {knobs['llm']['stub_latency_s']} s "
                                 "per call)" if stub else "")
        headline = {"figure": "model_requests_in_flight", "label": label,
                    "max": (opened or {}).get("max"), "mean": (opened or {}).get("mean")}
    return {"agents_live": agents, "decoding_at_servers": decoding, "decoding_at_servers_why_not": why,
            "headline": {**headline, "emulated": stub},
            "live": live.section() if live is not None else None}


def write_record(a, knobs: dict, out: Path, *, wall_s: float, t_run: float, interrupted: bool, crashed: bool,
                 rc: int, loop_lag, device, threads, pin: dict | None = None, live: LiveLine | None = None) -> dict:
    """run_record.json and agents.jsonl by record.write, one part per population; then AUTONMI's verdict
    (summary.json, by autonmi.assemble), its telemetry check with the populations' task counts
    (`telemetry_verdict`), and the framework's commit against its pin (`pin`, autonmi_pin) in its `autonmi`
    section; and the decoding metric's other counts in `speed` (decoding_metric). A run whose AUTONMI checks
    fail fails."""
    from autonmi.assemble import assemble

    parts, failed = [], []
    for i in range(a.populations):
        f = out / f"agents-p{i:03d}.jsonl"
        if f.exists():
            parts.append(part(a, knobs, i, f, loop_lag=loop_lag, device=device, threads=threads))
        else:   # the population never started
            failed.append(i)
    refs = references(a, knobs, live)
    rec = record.write(out, knobs=knobs, layer="autonmi", sim=a.sim, parts=parts, failed=failed, wall_s=wall_s,
                       interrupted=interrupted, t_run=t_run, model_refs=refs)
    try:
        summary = assemble(out)
    except Exception as exc:   # an interrupted or broken run may lack what AUTONMI assembles
        summary = {"ok": False, "checks": [{"name": "assembled", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}]}
    ok = rc == 0 and bool(summary.get("ok")) and not crashed
    rec["autonmi"] = {"ok": ok, "failed_checks": [c["name"] for c in summary.get("checks", []) if not c["ok"]],
                      "summary": "summary.json", "runtime": a.runtime, "populations": a.populations,
                      "workers_per_population": a.workers, "crashed": crashed, "versions": framework_versions(),
                      "framework_commit": pin, "asyncflow_running_shim": RUNNING_SHIM,
                      "telemetry": telemetry_verdict(summary)}
    rec["counts"]["suspensions"] = sum(p["layer"]["suspensions"] or 0 for p in parts)
    rows, n = _rows(out), knobs["agents"]["count"]
    # counts.agents keeps meaning campaigns (as in the other layers' records); the hosted agents are 3 per campaign
    rec["counts"].update(campaigns=n, hosted_agents=AGENTS_PER_CAMPAIGN * n, agents_per_campaign=AGENTS_PER_CAMPAIGN,
                         roles=list(ROLES), by_role=by_role(rows),
                         expected_by_role={role: {**e, "lookups": len(e["lookups"]), **messages(knobs)[role]}
                                           for role, e in decide.expected_by_role(knobs).items()})
    rec["speed"].update(decoding_metric(knobs, out, rec, summary, refs, live, rows))
    waits = [r["start_wait_s"] for r in rows if r.get("start_wait_s") is not None]
    rec["speed"]["start_together"] = {"on": knobs["agents"]["start_together"], "coordinator_wait_s": record._pct(waits),
                                      "begin_margin_s": begin_margin_s(rows)}
    if live is not None and live.served is not None:   # what each server's /v1/models said serves llm.model
        rec["llm_service"]["served_models"] = live.served
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
    agents_live(knobs)   # refuses a cap that could leave agents waiting for good, before anything starts
    pin = autonmi_pin()   # before the run, so that its warning comes first
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
    live = LiveLine(local=a.runtime == "local")
    try:   # the campaign configures the OpenTelemetry export (CampaignConfig.telemetry's defaults)
        device = probe(a)   # first: a broken torch or device stops the run before any campaign starts
        rc, interrupted, loop_lag = asyncio.run(run_until_signalled(a, knobs, out, work_root, live))
    except KeyboardInterrupt:   # SIGTERM or Ctrl-C before the event loop took the signals over
        interrupted = True
    except ModelServiceError as exc:   # before any campaign: the reason alone
        print(f"[fusion/autonmi] run refused: {exc}", file=sys.stderr, flush=True)
        crashed = True
    except Exception:           # recorded as a failed run, with what finished
        traceback.print_exc()
        crashed = True
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
    rec = write_record(a, knobs, out, wall_s=time.perf_counter() - t0, t_run=t_run, interrupted=interrupted,
                       crashed=crashed, rc=rc, loop_lag=loop_lag, device=device, threads=threads, pin=pin,
                       live=live)
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
