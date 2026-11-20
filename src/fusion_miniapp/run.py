"""Entry point: python -m fusion_miniapp.run --agents N [--use-academy] [--preset NAME | --knobs FILE]
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


class DictStore(dict):
    """In-process memory with the `put` of FlowGentic's KeyValueStore; a RadexStore drops in on a machine."""
    put = dict.__setitem__


def _interrupt(*_) -> None:
    """SIGTERM (scancel, a time limit, kill) unwinds like Ctrl-C, so pools close and work folders go."""
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


def _worker_init() -> None:
    """In each simulation worker: exit normally on SIGTERM, so its exit hook removes its assets folder."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    _end_with_parent()


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
        self._uid = itertools.count()

    @classmethod
    async def open(cls, knobs: dict, sim_fn, out: Path, part: int, workers: int, *, processes: bool = False,
                   backend=None, store=None, model=None) -> "Services":
        """Laptop defaults: RHAPSODY's concurrent backend on `workers` spawned processes (processes=True,
        for the real evaluate) or threads (the fixture), an in-process store, the LLM the knobs name.
        A process worker is retired after round(1 / sim.cold_start_fraction) tasks, so that share of
        evaluations runs cold in a fresh process. Not wired on the machines yet: a Dragon backend and a
        RadexStore would be passed here."""
        from rhapsody import Session
        from rhapsody.backends import ConcurrentExecutionBackend
        svc = cls(knobs, sim_fn, model or llm.make(knobs["llm"]), None,
                  DictStore() if store is None else store, open(out / f"agents-{part}.jsonl", "w"))
        if backend is None:
            svc.sim_workers, svc.sim_pool = workers, "process" if processes else "thread"
            f = knobs["sim"]["cold_start_fraction"]
            # spawn, never fork: a forked worker would skip the exit hook that removes its assets directory
            backend = await ConcurrentExecutionBackend(
                ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                    max_tasks_per_child=max(1, round(1 / f)) if f else None) if processes
                else ThreadPoolExecutor(workers))
        svc.session = Session(backends=[backend], work_dir=str(out))
        return svc

    async def close(self) -> None:
        self.log.close()
        await self.session.close()   # joins the worker pool

    async def _task(self, fn, name: str, **kwargs):
        """One RHAPSODY function task on this process's session; its return value."""
        from rhapsody import ComputeTask
        task = ComputeTask(function=fn, uid=f"{name}.{next(self._uid)}", kwargs=kwargs)
        (future,) = await self.session.submit_tasks([task])
        return (await future)["return_value"]

    async def llm(self, agent_id: str, messages: list[dict], thinking_budget: int) -> str:
        return await self.model(agent_id, messages, thinking_budget)

    async def simulate(self, agent_id: str, candidate: dict, seed: int) -> dict:
        self.sim_tasks += 1
        return await self._task(self.sim_fn, "evaluate", candidate=candidate, knobs=decide.sim_knobs(self.knobs, seed),
                                agent_id=agent_id, seed=seed)

    async def probe(self) -> None:
        """The device probe as a task, so it describes a simulation worker's torch and devices."""
        self.device = await self._task(record.device_probe, "probe", load_torch=True)

    async def lookup(self, agent_id: str, what: str) -> dict:
        self.tool_calls[what] += 1
        return await tools.lookup(what, agent_id, self.knobs["tools"]["latency_ms"])

    async def remember(self, agent_id: str, state: dict) -> None:
        self.memory_writes += 1
        self.store.put(f"campaign/{agent_id}",
                       json.dumps({k: state[k] for k in ("round", "best", "history")}, default=str))

    async def report(self, agent_id: str, result: dict) -> None:
        self.reports += 1
        self.log.write(json.dumps(result, sort_keys=True, default=str) + "\n")

    def part(self) -> dict:
        """This agent process's counts; call after close(), so that its worker processes have ended."""
        return {"agents_file": self.log.name, "llm_service": self.model.seen(), "tool_calls": dict(self.tool_calls),
                "sim_tasks": self.sim_tasks, "memory_writes": self.memory_writes, "reports": self.reports,
                "layer": self.layer, "sim_workers": self.sim_workers, "sim_pool": self.sim_pool,
                "device": self.device or record.device_probe(), "frameworks": record.frameworks(),
                "peak_rss_mb": record.peak_rss_mb(),
                "sim_worker_peak_rss_mb": record.peak_rss_mb(children=True) if self.sim_pool == "process" else None}


def sim_function(mode: str):
    """`evaluate` or its fixture twin, by name from fusion_miniapp.sim."""
    from fusion_miniapp import sim
    return sim.evaluate if mode == "real" else sim.evaluate_fixture


async def run_slice(layer: str, knobs: dict, sim: str, out: Path, first: int, n: int, part: int, workers: int,
                    emit=None) -> tuple[list, dict]:
    """Agents first .. first+n-1 in this process, on the chosen layer."""
    run_agents = importlib.import_module(f"fusion_miniapp.layer_{layer}").run_agents
    svc = await Services.open(knobs, sim_function(sim), out, part, workers, processes=sim == "real")
    try:
        if sim == "real":
            await svc.probe()   # first: a broken torch or device stops the run before any agent starts
        results = await run_agents(n, knobs, lambda: svc, out, first=first, emit=emit)
    finally:
        await svc.close()
    return results, svc.part()


def _child(layer, knobs, sim, out, first, n, part, workers, live) -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    _end_with_parent()

    def emit(c):
        live[4 * part:4 * part + 4] = list(c)
    _, p = asyncio.run(run_slice(layer, knobs, sim, Path(out), first, n, part, workers, emit))
    (Path(out) / f"part-{part}.json").write_text(json.dumps(p, default=str))


def _fan_out(layer, knobs, sim, out: Path, slices, workers) -> tuple[list[dict], list[int]]:
    """One agent process per slice; the parts of those that ended cleanly, and the indices of those that did not."""
    ctx = mp.get_context("spawn")
    live = ctx.Array("q", 4 * len(slices), lock=False)
    procs = [ctx.Process(target=_child, args=(layer, knobs, sim, str(out), first, n, k, workers, live))
             for k, (first, n) in enumerate(slices)]
    try:
        for p in procs:
            p.start()
        while any(p.is_alive() for p in procs):   # one live line for all processes
            time.sleep(record.LIVE_INTERVAL_S)
            record.print_live([sum(live[j::4]) for j in range(4)])
    finally:   # on an interrupt: stop the children, give them time to close their pools, then kill
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            if p.pid is not None:
                p.join(20)
                if p.is_alive():
                    p.kill()
                    p.join()
    record.print_live([sum(live[j::4]) for j in range(4)])
    failed = [k for k, p in enumerate(procs) if p.exitcode]
    parts = []
    for k in range(len(slices)):
        f = out / f"part-{k}.json"
        if k not in failed:
            parts.append(json.loads(f.read_text()))
            f.unlink()
    return parts, failed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--agents", type=int, help="number of agents (default: the knob file's agents.count)")
    p.add_argument("--use-academy", action="store_true", help="Academy agents instead of LangGraph on FlowGentic")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preset", choices=knob_files.PRESETS)
    g.add_argument("--knobs", help="a knob file (YAML), merged over the smoke preset")
    p.add_argument("--sim", choices=("real", "fixture"), default="real")
    p.add_argument("--llm", choices=("stub", "openai"), help="overrides llm.mode")
    p.add_argument("--out", help="output directory (default: results/<time>-<layer>-<agents>-<pid>)")
    a = p.parse_args(argv)

    layer = "academy" if a.use_academy else "langgraph"   # the only place the flag is read
    knobs = knob_files.load(path=a.knobs, preset=a.preset)
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
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if len(slices) == 1:
        parts, failed = [asyncio.run(run_slice(layer, knobs, a.sim, out, 0, n, 0, workers))[1]], []
    else:
        parts, failed = _fan_out(layer, knobs, a.sim, out, slices, workers)
        if not parts:
            raise SystemExit(f"agent processes {failed} all failed; no run record written")
    rec = record.write(out, knobs=knobs, layer=layer, sim=a.sim, parts=parts, failed=failed,
                       wall_s=time.perf_counter() - t0)
    c = rec["counts"]
    print(json.dumps({"run_record": str(out / "run_record.json"), "agent_layer": layer, **c,
                      "touch_check": rec["touch_check"], "devices": rec["devices"],
                      "wall_seconds": rec["wall_seconds"], "peak_rss_mb": rec["peak_rss_mb"]}))
    # a failed agent or agent process, a missing agent or a missing touch fails the run
    return 1 if failed or c["agents"] != n or c["agents_failed"] or rec["touch_check"]["fail"] else 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    sys.exit(main())
