"""Entry point: python -m fusion_miniapp.run --agents N [--use-academy] [--preset NAME] [--knobs FILE ...]
                                       [--sim real|fixture] [--llm stub|openai] [--out DIR]

`--use-academy` is read once, here; the chosen agent layer is imported lazily, so a LangGraph run
never imports Academy and an Academy run never imports LangGraph. Agents are split into processes
of `agents.per_process` each; every process builds its own Services and RHAPSODY session. With
`--sim real` that session runs `evaluate` in spawned worker processes (torch never loads in an
agent process); the instant fixture runs on threads.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import itertools
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from fusion_miniapp import decide, knobs as knob_files, llm, record, tools
from fusion_miniapp.llm import REPLY_TOKENS


class FixturePool(ThreadPoolExecutor):
    """The fixture's threads. Closing drops queued work, as the process pool's bounded shutdown does: an
    interrupted 10,000-campaign run can hold 170,000 queued fixture tasks, and RHAPSODY's close calls
    shutdown(wait=True), which would run them all first."""

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=wait, cancel_futures=True)


class DictStore(dict):
    """In-process memory with the `put` of FlowGentic's KeyValueStore; a RadexStore drops in on a machine."""
    put = dict.__setitem__


SHUTDOWN_S = 20.0      # a simulation worker pool's shutdown deadline, after which leftovers are killed
PRUNE_S = 5.0          # how often finished tasks are dropped from RHAPSODY's bookkeeping


def _interrupt(*_) -> None:
    """SIGTERM in the main process of a fan-out (scancel, a time limit, kill) unwinds like Ctrl-C, so the
    agent processes get SIGTERM, write their counts and end; a second SIGTERM is ignored."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


def _end_with_parent() -> None:
    """When this spawned process's parent ends (killed), end it too: SIGTERM twice (one that lands inside a
    simulation task ends only that task), then a hard exit. Otherwise it would run on as an orphan."""
    parent = mp.parent_process()

    def watch():
        parent.join()
        for _ in range(2):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        os._exit(143)
    threading.Thread(target=watch, daemon=True).start()


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def cpus() -> int:
    """CPUs this process may use (its affinity mask, which Slurm sets per step; else all)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:   # macOS
        return os.cpu_count() or 1


def thread_limit(knobs: dict) -> int:
    """CPU threads per simulation worker: the knob, or (auto) the CPUs ÷ all simulation workers on the node."""
    t = knobs["sim"]["threads_per_worker"]
    return max(1, cpus() // knobs["workers"]["sim_workers"]) if t == "auto" else t


def _worker_init() -> None:
    """In each simulation worker: on SIGTERM, remove its assets and work folders and exit at once. An
    exception raised inside a task would only fail that task and leave the worker running (audit D1)."""
    def stop(*_):
        from fusion_miniapp import sim
        sim.cleanup()
        os._exit(143)
    signal.signal(signal.SIGTERM, stop)
    _end_with_parent()


def _prune(session) -> int:
    """Drop finished tasks from RHAPSODY's bookkeeping. Its Session and ConcurrentExecutionBackend keep
    every task they have run, with its result, for the session's life: 320,000 results per agent
    process at 100,000 campaigns x 32 evaluations. Each caller already holds its own future, and nothing
    reads a finished entry again. (Private attributes of rhapsody-py 0.7.0; tests/test_run_hygiene.py.)"""
    terminal = getattr(getattr(session, "_state_manager", None), "_terminal_states", ())
    tasks, n = getattr(session, "_tasks", {}), 0
    for uid in [u for u, t in tasks.items() if t.get("state") in terminal]:
        del tasks[uid]   # each still sits in a task -> future -> result -> task cycle until the cyclic
        n += 1           # collector runs; breaking it would change RHAPSODY's await task (left upstream)
    backends = getattr(session, "_exec_backends", {})
    for b in backends.values() if isinstance(backends, dict) else backends:
        bt = getattr(b, "tasks", None)
        if isinstance(bt, dict):
            for uid in [u for u, t in bt.items() if t.get("future") is not None and t["future"].done()]:
                del bt[uid]
    return n


class Services:
    """One agent process's services (the plan, section 3), the same object for both layers: the LLM,
    `evaluate` as a RHAPSODY function task, the lookups, memory, and the campaign manager that agents
    report to (a method call on this object, not a message). layer_langgraph also uses `session` and
    `sim_fn` to declare `evaluate` as a suspending task tool on the same session."""

    def __init__(self, knobs, sim_fn, model, session, store, log) -> None:
        self.knobs, self.sim_fn, self.model, self.session, self.store, self.log = (
            knobs, sim_fn, model, session, store, log)
        self.tool_calls: Counter = Counter()
        self.memory_writes = self.reports = self.sim_tasks = 0
        self.sim_workers: int | None = None
        self.sim_pool = "backend"
        self.device: dict | None = None   # the device probe, taken where simulations run
        self.layer: dict = {}   # what the agent layer reports about its stack
        self.sim = "real"                  # "fixture" when open() runs the fixture on threads
        self.threads: dict | None = None   # the thread limit the simulation workers started with
        self.pool = None                   # the simulation worker pool (pool.ResilientProcessPool), if processes
        self.run_tmp: Path | None = None   # every evaluation's work folder lives under this, removed at close
        self.pruned = 0
        self.calls: dict[str, list] = {}   # each campaign's model calls (details), until it reports
        self.backend: str | None = None
        self.loop_lag: dict | None = None
        self._pruner: asyncio.Task | None = None
        self._uid = itertools.count()

    @classmethod
    async def open(cls, knobs: dict, sim_fn, out: Path, part: int, workers: int, *, processes: bool = False,
                   backend=None, store=None, model=None) -> "Services":
        """Laptop defaults: RHAPSODY's concurrent backend on `workers` spawned processes (processes=True,
        for the real evaluate) or threads (the fixture), an in-process store, the LLM the knobs name.
        The process pool rebuilds itself once when a worker dies and resubmits the lost tasks, and its
        shutdown kills leftovers after SHUTDOWN_S. Not wired on the machines yet: a Dragon backend and a
        RadexStore would be passed here."""
        svc = cls(knobs, sim_fn, model or llm.make(knobs["llm"]), None,
                  DictStore() if store is None else store, open(out / f"agents-{part}.jsonl", "w"))
        try:
            await svc._open(knobs, out, part, workers, processes, backend)
        except BaseException:   # a half-open process closes what it opened
            if svc.session is None and svc.pool is not None:
                svc.pool.shutdown(wait=True)
            await svc.close()
            raise
        return svc

    async def _open(self, knobs, out, part, workers, processes, backend) -> None:
        import tempfile

        from fusion_miniapp.pool import ResilientProcessPool
        from rhapsody import Session
        from rhapsody.backends import ConcurrentExecutionBackend
        svc = self
        if not knobs["sim"]["keep_artifacts"]:
            # One folder this process owns for all its evaluations' assets and work folders, cold-start
            # children's included: removed after the pool has stopped, whatever killed a worker.
            from fusion_miniapp.sim import work_root
            svc.run_tmp = Path(tempfile.mkdtemp(prefix=f"fusion_miniapp_run{part}_", dir=work_root(knobs["sim"])))
        if backend is None:
            svc.sim_workers, svc.sim_pool = workers, "process" if processes else "thread"
            svc.sim = "real" if processes else "fixture"
            if processes:
                # Before the pool starts: spawned workers inherit these, and numpy's OpenBLAS, MKL and
                # torch read them when first imported (sim.py also calls torch.set_num_threads).
                n = str(thread_limit(knobs))
                os.environ.update({v: n for v in THREAD_VARS}, FUSION_MINIAPP_THREADS=n)
                svc.threads = {"per_worker": int(n), "cpus": cpus(), "knob": knobs["sim"]["threads_per_worker"]}
            # spawn, never fork: a forked worker would skip the exit hook that removes its assets directory
            if processes:
                svc.pool = ResilientProcessPool(workers, mp_context=mp.get_context("spawn"),
                                                initializer=_worker_init, shutdown_timeout=SHUTDOWN_S)
            backend = await ConcurrentExecutionBackend(svc.pool or FixturePool(workers))
        svc.backend = type(backend).__name__
        svc.session = Session(backends=[backend], work_dir=str(out))
        svc._pruner = asyncio.create_task(svc._prune_loop())

    async def _prune_loop(self) -> None:
        while True:
            await asyncio.sleep(PRUNE_S)
            self.pruned += _prune(self.session)

    async def close(self) -> None:
        """Every step runs even if an earlier one fails."""
        try:
            if self._pruner is not None:
                self._pruner.cancel()
                await asyncio.gather(self._pruner, return_exceptions=True)
            self.log.close()
        finally:
            try:
                if self.session is not None:
                    await self.session.close()   # joins the worker pool, within SHUTDOWN_S
            finally:
                try:
                    close = getattr(self.model, "close", None)
                    if close is not None:
                        await close()
                finally:
                    if self.run_tmp is not None:
                        import shutil
                        shutil.rmtree(self.run_tmp, ignore_errors=True)

    def sim_knobs(self, seed: int) -> dict:
        """The `sim` knobs one campaign's evaluations get: its cost scale, and this run's own work folder."""
        s = decide.sim_knobs(self.knobs, seed)
        if self.run_tmp is not None:
            s["work_root"] = str(self.run_tmp)
        return s

    async def _task(self, fn, name: str, **kwargs):
        """One RHAPSODY function task on this process's session; its return value."""
        from rhapsody import ComputeTask
        task = ComputeTask(function=fn, uid=f"{name}.{next(self._uid)}", kwargs=kwargs)
        (future,) = await self.session.submit_tasks([task])
        return (await future)["return_value"]

    async def llm(self, agent_id: str, messages: list[dict], thinking_budget: int,
                  reply_tokens: int = REPLY_TOKENS) -> str:
        return (await self.llm_meta(agent_id, messages, thinking_budget, reply_tokens))[0]

    async def llm_meta(self, agent_id: str, messages: list[dict], thinking_budget: int,
                       reply_tokens: int = REPLY_TOKENS) -> tuple[str, dict]:
        """The reply and the call's details (latency, finish reason, tokens, empty), which are kept for
        the campaign's line in agents.jsonl and passed to FlowGentic's token admission."""
        call, t0 = getattr(self.model, "call_with_meta", None), time.perf_counter()
        try:
            if call is None:
                text, meta = await self.model(agent_id, messages, thinking_budget, reply_tokens=reply_tokens), {}
            else:
                text, meta = await call(agent_id, messages, thinking_budget, reply_tokens=reply_tokens)
        except Exception as exc:   # a call that failed after its retries still leaves its record
            self.calls.setdefault(agent_id, []).append(
                {"reply_tokens": reply_tokens, "latency_s": round(time.perf_counter() - t0, 4),
                 "error": type(exc).__name__, "finish_reason": None, "empty": None})
            raise
        self.calls.setdefault(agent_id, []).append({"reply_tokens": reply_tokens, **meta})
        return text, meta

    async def simulate(self, agent_id: str, params: dict, candidate_idx: int, generation: int, seed: int) -> dict:
        self.sim_tasks += 1
        return await self._task(self.sim_fn, "evaluate", params=params, candidate_idx=candidate_idx,
                                generation=generation, knobs=self.sim_knobs(seed), agent_id=agent_id, seed=seed)

    async def probe(self) -> None:
        """The device probe as a task, so it describes a simulation worker's torch and devices."""
        self.device = await self._task(record.device_probe, "probe", load_torch=True)

    async def lookup(self, agent_id: str, what: str) -> dict:
        self.tool_calls[what] += 1
        return await tools.lookup(what, agent_id, self.knobs["tools"]["latency_ms"])

    async def remember(self, agent_id: str, state: dict) -> None:
        self.memory_writes += 1
        self.store.put(f"campaign/{agent_id}", json.dumps(state, default=str))

    async def report(self, agent_id: str, result: dict) -> None:
        self.reports += 1
        line = {**result, "llm_call_details": self.calls.pop(agent_id, [])}
        self.log.write(json.dumps(line, sort_keys=True, default=str) + "\n")

    def part(self) -> dict:
        """This agent process's counts; call after close(), so that its worker processes have ended."""
        return {"agents_file": self.log.name, "llm_service": self.model.seen(), "tool_calls": dict(self.tool_calls),
                "sim_tasks": self.sim_tasks, "memory_writes": self.memory_writes, "reports": self.reports,
                "layer": self.layer, "sim_workers": self.sim_workers, "sim_pool": self.sim_pool,
                "threads": self.threads, "pool": self.pool.stats() if self.pool else None,
                "backend": self.backend, "loop_lag": self.loop_lag,
                "pruned_tasks": self.pruned,
                "device": self.device or record.device_probe(), "frameworks": record.frameworks(),
                "peak_rss_mb": record.peak_rss_mb()}


def sim_function(mode: str):
    """`evaluate` or its fixture twin, by name from fusion_miniapp.sim."""
    from fusion_miniapp import sim
    return sim.evaluate if mode == "real" else sim.evaluate_fixture


async def run_slice(layer: str, knobs: dict, sim: str, out: Path, first: int, n: int, part: int, workers: int,
                    emit=None) -> tuple[list, dict]:
    """Agents first .. first+n-1 in this process, on the chosen layer. SIGTERM cancels the agents through
    the event loop (audit D3); the pool still closes, and the part (status "interrupted") still returns
    with the counts of the campaigns that finished."""
    run_agents = importlib.import_module(f"fusion_miniapp.layer_{layer}").run_agents
    svc = await Services.open(knobs, sim_function(sim), out, part, workers, processes=sim == "real")
    loop, me, stopped = asyncio.get_running_loop(), asyncio.current_task(), []
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def on_term() -> None:
        if not stopped:
            stopped.append(time.time())
            if svc.pool is not None:
                svc.pool.stop_rebuilds()   # workers are about to get the same signal: do not replace them
            me.cancel()
    # A plain handler that queues on_term, not loop.add_signal_handler: that one wakes the loop through a
    # self-pipe, which a busy loop at scale can fill, and then the signal is dropped (Codex 03:18-03:19).
    for sig in before:
        with contextlib.suppress(ValueError):   # not the main thread: skip
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(on_term))
    results, status = [], "complete"
    lag = record.LoopLag()
    lag.start()
    try:
        if sim == "real":
            await svc.probe()   # first: a broken torch or device stops the run before any agent starts
        results = await run_agents(n, knobs, lambda: svc, out, first=first, emit=emit)
    except asyncio.CancelledError:
        if not stopped:
            raise
        me.uncancel()
        status = "interrupted"
    finally:
        for sig, old in before.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(sig, signal.SIG_IGN if stopped else old)
        svc.loop_lag = await lag.stop()
        await svc.close()
    p = svc.part()
    p["status"] = status
    return results, p


def _child(layer, knobs, sim, out, first, n, part, workers, live) -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl-C reaches the whole group: the main process
                                                   # turns it into SIGTERM for each agent process
    _end_with_parent()

    def emit(c):
        live[4 * part:4 * part + 4] = list(c)
    _, p = asyncio.run(run_slice(layer, knobs, sim, Path(out), first, n, part, workers, emit))
    (Path(out) / f"part-{part}.json").write_text(json.dumps(p, default=str))


def _fan_out(layer, knobs, sim, out: Path, slices, workers) -> tuple[list[dict], list[int], bool]:
    """One agent process per slice: the parts they wrote, the indices of those that wrote none, and
    whether the run was interrupted (SIGTERM or Ctrl-C here)."""
    ctx = mp.get_context("spawn")
    live = ctx.Array("q", 4 * len(slices), lock=False)
    procs = [ctx.Process(target=_child, args=(layer, knobs, sim, str(out), first, n, k, workers, live))
             for k, (first, n) in enumerate(slices)]
    interrupted = False
    try:
        for p in procs:
            p.start()
        while any(p.is_alive() for p in procs):   # one live line for all processes
            time.sleep(record.LIVE_INTERVAL_S)
            record.print_live([sum(live[j::4]) for j in range(4)])
    except KeyboardInterrupt:
        interrupted = True
    finally:   # stop the children (each closes its pool within SHUTDOWN_S and writes its part), then kill
        for p in procs:
            if p.is_alive():
                p.terminate()
        deadline = time.monotonic() + SHUTDOWN_S + 25   # 10,000 campaigns unwind in about 15 s on a laptop
        for p in procs:
            if p.pid is not None:
                p.join(max(0.1, deadline - time.monotonic()))
                if p.is_alive():
                    p.kill()
                    p.join()
    record.print_live([sum(live[j::4]) for j in range(4)])
    parts, failed = [], []
    for k in range(len(slices)):
        f = out / f"part-{k}.json"
        if f.exists():
            parts.append(json.loads(f.read_text()))
            f.unlink()
        else:
            failed.append(k)
    return parts, failed, interrupted


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--agents", type=int, help="number of agents (default: the knob file's agents.count)")
    p.add_argument("--use-academy", action="store_true", help="Academy agents instead of LangGraph on FlowGentic")
    p.add_argument("--preset", choices=knob_files.PRESETS, help="merged over the smoke preset")
    p.add_argument("--knobs", action="append", default=[], metavar="FILE",
                   help="a knob file (YAML) merged over the preset; repeatable, applied in order "
                        "(e.g. a machine file from knobs/machines/)")
    p.add_argument("--sim", choices=("real", "fixture"), default="real")
    p.add_argument("--llm", choices=("stub", "openai"), help="overrides llm.mode")
    p.add_argument("--out", help="output directory (default: results/<time>-<layer>-<agents>-<pid>)")
    a = p.parse_args(argv)

    layer = "academy" if a.use_academy else "langgraph"   # the only place the flag is read
    knobs = knob_files.load(preset=a.preset, paths=a.knobs)
    over = {"agents": {"count": a.agents} if a.agents is not None else {}, "llm": {"mode": a.llm} if a.llm else {}}
    knobs = knob_files.validate({**knobs, **{s: {**knobs[s], **v} for s, v in over.items()}})
    if knobs["llm"]["mode"] == "openai" and knobs["llm"]["model"] == "stub":
        raise knob_files.KnobError('llm.mode is openai but llm.model is still "stub"; set llm.model in a knob file')
    n, per = knobs["agents"]["count"], knobs["agents"]["per_process"]
    slices = [(first, min(per, n - first)) for first in range(0, n, per)]
    if a.sim == "real" and len(slices) > knobs["workers"]["sim_workers"]:
        raise knob_files.KnobError(
            f"{len(slices)} agent processes each need a simulation worker, but workers.sim_workers is "
            f"{knobs['workers']['sim_workers']}; raise workers.sim_workers or agents.per_process")
    workers = max(1, knobs["workers"]["sim_workers"] // len(slices))   # simulation workers per agent process
    out = Path(a.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}-{layer}-{n}-{os.getpid()}")
    used = [p.name for p in out.glob("*") if p.name in ("run_record.json", "agents.jsonl", "campaigns", "run.lock")
            or p.name.startswith(("agents-", "part-"))] if out.is_dir() else []
    if used:   # the parent's logs append, so a second run would silently extend the first one's files
        raise knob_files.KnobError(f"--out {out} already holds a run ({', '.join(sorted(used))}); "
                                   "give a new directory")
    out.mkdir(parents=True, exist_ok=True)
    try:   # one run per directory, also against a second run started at the same moment
        os.close(os.open(out / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise knob_files.KnobError(f"--out {out} is in use by another run (run.lock); give a new directory") from None

    t0, t_run = time.perf_counter(), time.time()
    if len(slices) == 1:
        try:
            parts, failed = [asyncio.run(run_slice(layer, knobs, a.sim, out, 0, n, 0, workers))[1]], []
            interrupted = False
        except KeyboardInterrupt:   # before the agents' event loop took the signals over
            parts, failed, interrupted = [], [0], True
    else:
        parts, failed, interrupted = _fan_out(layer, knobs, a.sim, out, slices, workers)
    interrupted = interrupted or any(p.get("status") == "interrupted" for p in parts)
    rec = record.write(out, knobs=knobs, layer=layer, sim=a.sim, parts=parts, failed=failed,
                       wall_s=time.perf_counter() - t0, interrupted=interrupted, t_run=t_run)
    c = rec["counts"]
    print(json.dumps({"run_record": str(out / "run_record.json"), "status": rec["status"], "agent_layer": layer,
                      **c, "touch_check": rec["touch_check"], "devices": rec["devices"],
                      "wall_seconds": rec["wall_seconds"], "peak_rss_mb": rec["peak_rss_mb"]}))
    print(record.speed_line(rec))
    for reason in rec["llm_check"]["fail"]:
        print(f"run failed: {reason}", file=sys.stderr)
    return {"pass": 0, "interrupted": 143}.get(rec["status"], 1)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    sys.exit(main())
