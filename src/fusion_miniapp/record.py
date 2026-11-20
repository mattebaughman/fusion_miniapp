"""The run record (<out>/run_record.json), the per-agent lines (<out>/agents.jsonl) and the live
counts line both layers print to stderr: `agents working=<n> waiting=<n> model_requests=<n> done=<n>`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import resource
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path

from fusion_miniapp import PARENT_DIR

LIVE_INTERVAL_S = 1.0
DISTS = ("fusion_miniapp", "numpy", "scipy", "netCDF4", "PyYAML", "torch", "langgraph", "langchain-core",
         "flowgentic", "rhapsody-py", "academy-py", "openai", "dragonhpc")


class Live:
    """Live counts from decide.run_campaign's on_state calls (the Academy layer's counter)."""

    def __init__(self) -> None:
        self.of: dict[str, str] = {}
        self.n: Counter = Counter()

    def __call__(self, agent_id: str, state: str) -> None:
        old = self.of.get(agent_id)
        if old is not None:
            self.n[old] -= 1
        self.of[agent_id] = state
        self.n[state] += 1

    def finish(self, agent_id: str) -> None:
        old = self.of.pop(agent_id, None)
        if old is not None:
            self.n[old] -= 1
        self.n["done"] += 1

    def counts(self) -> tuple[int, int, int, int]:
        return self.n["working"], self.n["waiting"], self.n["model_request"], self.n["done"]


def print_live(c) -> None:
    print("agents working={} waiting={} model_requests={} done={}".format(*c), file=sys.stderr, flush=True)


async def ticker(read, emit=None, interval: float = LIVE_INTERVAL_S) -> None:
    """Emit read() every interval until cancelled, then once more."""
    emit = emit or print_live
    try:
        while True:
            await asyncio.sleep(interval)
            emit(read())
    finally:
        emit(read())


def digest(lines) -> str:
    """SHA-256 of the per-agent results, independent of the order agents finished in."""
    rows = sorted(r if isinstance(r, str) else json.dumps(r, sort_keys=True, default=str) for r in lines)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def device_probe(load_torch: bool = False) -> dict:
    """What this process runs on. torch is described if it is loaded (or load_torch asks for it)."""
    out = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()}
    if load_torch:
        import torch  # noqa: F401
    torch = sys.modules.get("torch")
    if torch is None:
        return {**out, "torch": "not imported in this process"}
    # str(): torch.__version__ is a TorchVersion, which would import torch wherever it is unpickled
    return {**out, "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda, "hip_version": torch.version.hip,
            "mps_available": torch.backends.mps.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "visible_devices": {v: os.environ.get(v) for v in
                                ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}}


def frameworks() -> dict:
    return {m: m in sys.modules for m in ("academy", "langgraph", "flowgentic", "rhapsody", "torch")}


def versions() -> dict:
    out = {}
    for d in DISTS:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    try:  # FlowGentic 0.5.0 is installed from a git commit; record only the commit id
        out["flowgentic_commit"] = json.loads(metadata.distribution("flowgentic").read_text("direct_url.json")
                                              or "{}").get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        pass
    return out


def peak_rss_mb(children: bool = False) -> float:
    """This process's peak resident set size so far; children=True: the largest of its ended children."""
    scale = 2**20 if sys.platform == "darwin" else 2**10   # ru_maxrss: bytes on macOS, KiB on Linux
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return round(resource.getrusage(who).ru_maxrss / scale, 1)


def write(out: Path, *, knobs: dict, layer: str, sim: str, parts: list[dict], wall_s: float,
          failed: list[int] = ()) -> dict:
    """Merge the agent lines of this run's agent processes (`parts`, the ones that ended cleanly) into
    agents.jsonl and write run_record.json. `failed`: agent processes that did not end cleanly; their
    agents are missing from the record. Other files in `out` are never read.

    Fixture evaluations make no touches by design; they are counted apart from the touch check."""
    lines = []
    for part in parts:
        p = Path(part["agents_file"])
        lines += p.read_text().splitlines()
        p.unlink()
    lines.sort()
    (out / "agents.jsonl").write_text("".join(line + "\n" for line in lines))
    c = Counter()
    verdicts, devices, touches, errors, budgets, tool_calls, sources = (Counter() for _ in range(7))
    for line in lines:   # one agent at a time: 100,000 parsed results need not be held at once
        r = json.loads(line)
        if "error" in r:
            errors[r["error"].split(":", 1)[0]] += 1
            continue
        c.update(sims=r["n_sims"], llm_calls=r["n_llm_calls"], tool_calls=r["n_tool_calls"],
                 feasible_agents=bool(r["best"]["is_good"]))
        touches.update(r["touches"])
        for h in r["history"]:
            devices[str(h.get("device"))] += 1
            sources[h["source"]] += 1
            if h.get("fixture"):
                c["fixture"] += 1
            else:
                verdicts[str(h.get("touch_check"))] += 1
    for p in parts:
        budgets.update(p["llm_service"]["budgets"])
        tool_calls.update(p["tool_calls"])
    a = knobs["agents"]
    lock = json.loads((PARENT_DIR / "PARENT.lock").read_text())
    peaks = [p["peak_rss_mb"] for p in parts]
    sim_peaks = [p["sim_worker_peak_rss_mb"] for p in parts if p["sim_worker_peak_rss_mb"] is not None]
    rec = {
        "agent_layer": layer, "sim": sim, "llm": knobs["llm"]["mode"],
        "emulated": True, "method": "miniapp_standin",
        "note": "all physics output is synthetic (random-weight stand-in models)", "knobs": knobs,
        "counts": {
            "agents": len(lines), "agents_requested": a["count"], "agents_failed": sum(errors.values()),
            "agent_processes": len(parts) + len(failed), "failed_agent_processes": list(failed),
            **{k: c[k] for k in ("sims", "llm_calls", "tool_calls")},
            "proposals": {"llm": sources["llm"], "mutation": sources["mutation"]},
            "suspensions": sum(p["layer"]["suspensions"] for p in parts) if layer == "langgraph" else None,
            "sim_tasks": sum(p["sim_tasks"] for p in parts), "memory_writes": sum(p["memory_writes"] for p in parts),
            "reports": sum(p["reports"] for p in parts), "feasible_agents": c["feasible_agents"],
            "expected_per_agent": {"llm_calls": a["sims_per_agent"] + 1, "sims": a["sims_per_agent"]}},
        "llm_service": {"calls": sum(p["llm_service"]["calls"] for p in parts), "budgets": dict(budgets)},
        "lookups": dict(tool_calls),
        "touch_check": {"evaluations": sum(devices.values()), "fixture": c["fixture"],
                        "pass": verdicts.pop("pass", 0), "fail": dict(verdicts)},
        "touches": dict(touches), "devices": dict(devices),
        "wall_seconds": round(wall_s, 3), "results_sha256": digest(lines),
        "peak_rss_mb": {"main_process": peak_rss_mb(), "agent_process_max": max(peaks),
                        "agent_processes_sum": round(sum(peaks), 1),
                        "sim_worker_max": max(sim_peaks) if sim_peaks else None},
        "versions": versions(), "parent": {k: lock[k] for k in ("source", "branch", "commit")},
        "device_probe": parts[0]["device"], "frameworks_imported": parts[0]["frameworks"],
        "sim_pool": parts[0]["sim_pool"], "sim_workers_per_process": parts[0]["sim_workers"],
        "sim_workers_total": sum(p["sim_workers"] or 0 for p in parts),
        "layer_stats": [p["layer"] for p in parts],
    }
    if errors:
        rec["errors"] = dict(errors)
    (out / "run_record.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec
