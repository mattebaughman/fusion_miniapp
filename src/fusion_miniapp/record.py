"""The run record (<out>/run_record.json), the per-agent lines (<out>/agents.jsonl) and the live
counts line both layers print to stderr: `agents working=<n> waiting=<n> model_requests=<n> done=<n>`
(the AUTONMI layer prints its own, layer_autonmi.LiveLine).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

import numpy as np

from fusion_miniapp import PARENT_DIR, decide

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


# timings, memory, touch counts and device names: what depends on the machine or the scheduling
MACHINE_KEYS = ("sim_s", "wait_s", "t_submit", "t_start", "t_end", "worker_peak_rss_mb", "touches", "devices",
                "llm_call_details")
# per-candidate detail written with record.extras (the AUTONMI layer: the Monitor's numbers, and in the
# asynchronous mode each result's row and the replay orders): not the result
EXTRA_KEYS = ("monitor", "asynchronous")


def comparable(result: dict) -> dict:
    """A campaign's result without what depends on the machine or the scheduling: per-evaluation and
    per-generation times, touch counts (which evaluations met a cold worker), device names; and without
    the record toggle's per-candidate detail (EXTRA_KEYS)."""
    out = {k: v for k, v in result.items() if k not in MACHINE_KEYS and k not in EXTRA_KEYS}
    if "generations" in out:
        out["generations"] = [{k: v for k, v in g.items() if k not in ("elapsed_s", "t0", "fastest", "slowest")}
                              for g in out["generations"]]
    return out


def digest(lines) -> str:
    """SHA-256 over the sorted SHA-256s of the per-campaign results (their comparable parts): independent of
    the order they finished in, the same on any machine and in either agent layer (results_sha256)."""
    rows = sorted(hashlib.sha256(json.dumps(comparable(json.loads(r) if isinstance(r, str) else r),
                                            sort_keys=True, default=str).encode()).hexdigest() for r in lines)
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
    for dist, key in (("flowgentic", "flowgentic_commit"), ("rhapsody-py", "rhapsody_commit")):
        try:  # installed from a git commit: record only the commit id
            out[key] = json.loads(metadata.distribution(dist).read_text("direct_url.json")
                                  or "{}").get("vcs_info", {}).get("commit_id")
        except metadata.PackageNotFoundError:
            pass
    return out


def peak_rss_mb(children: bool = False) -> float:
    """This process's peak resident set size so far; children=True: the largest of its ended children."""
    scale = 2**20 if sys.platform == "darwin" else 2**10   # ru_maxrss: bytes on macOS, KiB on Linux
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return round(resource.getrusage(who).ru_maxrss / scale, 1)


def _pct(values) -> dict:
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"n": 0}
    q = np.percentile(a, [50, 95])
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "p50": round(float(q[0]), 4),
            "p95": round(float(q[1]), 4), "max": round(float(a.max()), 4)}


PCTS = (50, 90, 95)   # the shares of each reference whose time at or above occupancy(refs=) reports
STEADY = (0.15, 0.85)   # the steady window: these shares of the agent phase (its start-up and its tail left out)


def _held(held, dur, span: float, top, refs: dict | None, digits: int) -> dict:
    """The maximum, the time-weighted mean of levels `held` for durations `dur` over `span` seconds, and with
    refs ({name: n}) the seconds at or above 50, 90 and 95% of each n (None for an n that is unknown or 0)."""
    out = {"max": top, "mean": round(float((held * dur).sum() / span), digits)}
    if refs:
        out["time_ge_pct_s"] = {name: {"of": n, **{str(p): round(float(dur[held >= p / 100 * n].sum()), 3)
                                                  for p in PCTS}} if n else None for name, n in refs.items()}
    return out


def occupancy(starts, ends, t_lo: float, t_hi: float, n_ref: int | None = None, refs: dict | None = None,
              digits: int = 2) -> dict:
    """How many intervals [start, end) are open at once in the window [t_lo, t_hi]: the maximum, the
    time-weighted mean, with n_ref the seconds with at least 95% of n_ref open, and with refs ({name: n}, a
    list of references) the seconds at or above 50, 90 and 95% of each (`time_ge_pct_s`). At equal times an
    end counts before a start. The maximum too is within the window (of the levels held there for some time)."""
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    if not starts.size or t_hi <= t_lo:
        out = {"max": 0, "mean": 0.0, **({"time_ge_95pct_s": 0.0} if n_ref else {})}
        if refs:
            out["time_ge_pct_s"] = {name: {"of": n, **{str(p): 0.0 for p in PCTS}} if n else None
                                    for name, n in refs.items()}
        return out
    t = np.concatenate([starts, ends])
    d = np.concatenate([np.ones(starts.size), -np.ones(ends.size)])
    order = np.lexsort((d, t))
    t, level = t[order], np.cumsum(d[order])
    dur = np.diff(np.clip(t, t_lo, t_hi))
    inside = level[:-1][dur > 0]
    out = _held(level[:-1], dur, t_hi - t_lo, int(inside.max()) if inside.size else 0, refs, digits)
    if n_ref:
        out["time_ge_95pct_s"] = round(float(dur[level[:-1] >= 0.95 * n_ref].sum()), 3)
    return out


def sampled(t, values, t_lo: float, t_hi: float, refs: dict | None = None, digits: int = 4) -> dict | None:
    """occupancy() for a sampled count: each sample holds until the next one (the last until t_hi), within
    the window [t_lo, t_hi]. The maximum is over every sample. None without samples."""
    t, v = np.asarray(t, dtype=float), np.asarray(values, dtype=float)
    if not t.size or t_hi <= t_lo:
        return None
    dur = np.diff(np.clip(np.append(t, max(t_hi, t[-1])), t_lo, t_hi))
    top = float(v.max())
    return {"samples": int(t.size), **_held(v, dur, t_hi - t_lo, int(top) if top.is_integer() else round(top, 4),
                                            refs, digits)}


def summed(series: list[tuple[list, list]], hold_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Several sampled counts (one (times, values) per source, such as a model server) summed into one: at
    every time any source was sampled, each source's latest value (a missing value as 0). With hold_s a
    value counts for at most hold_s seconds, until the source's next sample if that comes sooner: a source
    that stopped answering counts 0 from hold_s after its last answer (the series then has a time there), not
    its last value to the end."""
    series = [(np.asarray(t, dtype=float), np.asarray([0.0 if x is None else x for x in v], dtype=float))
              for t, v in series if len(t)]
    cuts = [t for t, _ in series]
    if hold_s is not None:   # where a value stops counting before the source's next sample
        cuts += [(t + hold_s)[np.append(np.diff(t), np.inf) > hold_s] for t, _ in series]
    times = np.unique(np.concatenate(cuts)) if cuts else np.array([])
    total = np.zeros(times.size)
    for t, v in series:
        i = np.searchsorted(t, times, side="right") - 1
        counted = i >= 0
        if hold_s is not None:   # (as the cuts are computed: their own time is not counted)
            counted &= times < t[np.maximum(i, 0)] + hold_s
        total += np.where(counted, v[np.maximum(i, 0)], 0.0)
    return times, total


def unheld_s(t, t_lo: float, t_hi: float, hold_s: float) -> float:
    """The seconds of [t_lo, t_hi] in which a source sampled at times t had no value counted by
    summed(hold_s=): before its first sample, and more than hold_s after a sample with no next one yet."""
    t = np.asarray(t, dtype=float)
    if not t.size or t_hi <= t_lo:
        return round(max(0.0, t_hi - t_lo), 3)
    until = np.minimum(np.append(t[1:], np.inf), t + hold_s)
    held = np.clip(until, t_lo, t_hi) - np.clip(t, t_lo, t_hi)
    return round(float(t_hi - t_lo - held.sum()), 3)


class LoopLag:
    """How late this process's event loop wakes a 100 ms timer (FlowGentic's probe, in the stdlib, so both
    layers report the same measure): start() inside the loop, then summary()."""

    INTERVAL = 0.1

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        async def tick():
            loop = asyncio.get_running_loop()
            while True:
                t = loop.time()
                await asyncio.sleep(self.INTERVAL)
                self.samples.append(max(0.0, loop.time() - t - self.INTERVAL))
        self._task = asyncio.get_running_loop().create_task(tick())

    async def stop(self) -> dict:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        a = np.asarray(self.samples)
        if not a.size:
            return {"samples": 0}
        return {"samples": int(a.size), "p50_ms": round(1e3 * float(np.percentile(a, 50)), 1),
                "p95_ms": round(1e3 * float(np.percentile(a, 95)), 1), "max_ms": round(1e3 * float(a.max()), 1),
                "share_over_100ms": round(float((a > 0.1).mean()), 4)}


def git_state() -> dict:
    """The mini-app's own commit and whether tracked files differ (None outside a git checkout)."""
    import subprocess
    root = Path(__file__).resolve().parents[2]
    try:
        # only a checkout that is this source: git -C would otherwise find any repository above an install
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch",
                                  "src/fusion_miniapp/record.py"], capture_output=True, timeout=10)
        if Path(top).resolve() != root or tracked.returncode:
            return {"commit": None, "dirty": None}
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return {"commit": head, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def llm_check(knobs: dict, refill: dict, service: dict) -> dict:
    """llm.mode openai: the model must fill at least llm.min_model_share of the refill slots it was asked
    for, and at most llm.max_empty_share of its replies may be empty. No slots asked: no share verdict."""
    k = knobs["llm"]
    model = refill["filled"] / refill["asked"] if refill["asked"] else None     # raw ratios: compared unrounded
    empty = service["empty"] / service["calls"] if service["calls"] else None
    out = {"model_share": model, "empty_share": empty, "refill_slots": {"asked": refill["asked"],
           "filled": refill["filled"]}, "replies": {"calls": service["calls"], "empty": service["empty"]},
           "min_model_share": k["min_model_share"], "max_empty_share": k["max_empty_share"], "fail": []}
    if k["mode"] == "openai":
        if model is not None and model < k["min_model_share"]:
            out["fail"].append(f"the model filled {refill['filled']} of the {refill['asked']} refill slots it was "
                               f"asked for ({model:.4%}), below llm.min_model_share {k['min_model_share']}; the "
                               "rest are the parent's random mutants")
        if empty is not None and empty > k["max_empty_share"]:
            out["fail"].append(f"{service['empty']} of the model's {service['calls']} replies were empty "
                               f"({empty:.4%}), above llm.max_empty_share {k['max_empty_share']}")
    return out


def model_requests(starts, ends, lo: float, hi: float, refs: dict | None, **extra) -> dict:
    """Model requests open at once, from each call's interval (`model_requests_in_flight`, `_queued`): the
    maximum, the time-weighted mean over the campaigns' phase [lo, hi], the seconds at or above 50, 90 and
    95% of each reference, and the time each call spent so (`time_s`)."""
    spans = np.asarray(ends, dtype=float) - np.asarray(starts, dtype=float)
    return {**extra, **occupancy(starts, ends, lo, hi, refs=refs, digits=4), "time_s": _pct(spans)}


def steady(lo: float, hi: float) -> tuple[float, float]:
    """The steady window of the agent phase [lo, hi] (STEADY)."""
    return lo + STEADY[0] * (hi - lo), lo + STEADY[1] * (hi - lo)


def model_tokens(ends, prompt, generated, lo: float, hi: float, emulated: bool) -> dict:
    """Prompt and generated tokens per second over the steady window of the agent phase [lo, hi]: the usage
    counts of the calls answered in it (each call's t_end), per second of the window (`model_tokens_per_s`)."""
    w_lo, w_hi = steady(lo, hi)
    ends = np.asarray(ends, dtype=float)
    inside = (ends >= w_lo) & (ends <= w_hi)
    span = w_hi - w_lo
    def rate(a) -> float | None:
        return round(float(np.asarray(a, dtype=float)[inside].sum()) / span, 3) if span > 0 else None
    return {"window": list(STEADY), "window_s": round(span, 3), "calls": int(inside.sum()),
            "prompt_per_s": rate(prompt), "generated_per_s": rate(generated), "emulated": emulated}


def call_share(by_role: dict, roles: dict, campaigns: int, lo: float, hi: float) -> dict:
    """Each role's share of time in a model call over the steady window of the agent phase [lo, hi]: the mean
    number of its calls open at once (by_role: {role: (starts, ends)}, each call from t_end - service_s to t_end,
    of the campaigns that finished) over its agents (roles: {role: agents per campaign} x those campaigns); and a
    campaign's agents in a call, in all (`model_call_share`, record.extras). It takes every campaign as live
    through the whole window, as when all start at once (the hundredk shape); campaigns in waves
    (agents.max_live) would make it lower than the share of the agents live."""
    w_lo, w_hi = steady(lo, hi)
    share = {role: round(occupancy(*by_role.get(role, ((), ())), w_lo, w_hi, digits=6)["mean"] / (n * campaigns), 4)
             for role, n in roles.items()}
    return {"window": list(STEADY), "source": "each answered call's [t_end - service_s, t_end] in llm_call_details",
            "by_role": share, "agents_in_a_call_per_campaign": round(sum(share[r] * n for r, n in roles.items()), 4)}


def write(out: Path, *, knobs: dict, layer: str, sim: str, parts: list[dict], wall_s: float,
          failed: list[int] = (), interrupted: bool = False, t_run: float | None = None,
          model_refs: dict | None = None, roles: dict | None = None) -> dict:
    """Merge the campaign lines of this run's agent processes (`parts`, those that wrote their counts) into
    agents.jsonl and write run_record.json. `failed`: agent processes that wrote none; their campaigns are
    missing. `t_run`: when the run started (epoch). `model_refs`: the references ({name: n}) of the model
    requests open at once, which the record gives when the calls' details carry their times (t_end,
    service_s, wait_s, recovery_s: the AUTONMI layer's), or a part gives them apart from its lines
    (`call_times`: the calls' count, those retried, and `times`, an array of rows t_end, service_s, recovery_s,
    wait_s, prompt and completion tokens; the AUTONMI layer's calls summed instead of written in a line); from
    the same calls, prompt and generated tokens per second over the steady window (`model_tokens_per_s`).
    `roles` ({role: its agents per campaign}, the AUTONMI layer's): with record.extras, each role's share of
    time in a model call (`model_call_share`), from the calls' details in the lines. Other files in `out` are
    never read.

    The merge streams: each line is copied to agents.jsonl.tmp and counted as it passes, the file is
    renamed into place, and only then are the per-process files removed. Fixture evaluations make no
    touches by design; they are counted apart from the touch check."""
    from array import array
    from datetime import datetime, timezone
    c = Counter()
    verdicts, devices, touches, errors, budgets, tool_calls, sources, refill = (Counter() for _ in range(8))
    rows, n_lines = [], 0                            # each campaign's comparable hash, for the digest
    c_start, c_end, e_sub, e_start, e_end = (array("d") for _ in range(5))
    sim_s, wait_s, gen_s, worker_rss = array("d"), array("d"), array("d"), []
    # every model call, of every campaign: those whose details carry their times, as intervals at the client
    # (in flight: admitted and sent, its reply not back; queued: waiting for admission before that), and their
    # prompt and generated tokens
    m_start, m_end, q_start, q_end, calls = array("d"), array("d"), array("d"), array("d"), Counter()
    m_prompt, m_generated = array("d"), array("d")
    # record.extras, with the roles (the AUTONMI layer's): each role's calls in flight, from the lines' details
    by_role = {} if roles and knobs["record"]["extras"] else None
    tmp = out / "agents.jsonl.tmp"
    with tmp.open("w") as merged:
        for part in parts:
            t = part.get("call_times")
            if t is not None:   # the calls not in its lines, as their details would give them
                calls.update(all=t["calls"], retried=t["retried"])
                sent = t["times"][:, 0] - t["times"][:, 1]
                asked = sent - t["times"][:, 2]
                m_start.extend(sent.tolist())
                m_end.extend(t["times"][:, 0].tolist())
                q_start.extend((asked - t["times"][:, 3]).tolist())
                q_end.extend(asked.tolist())
                m_prompt.extend(t["times"][:, 4].tolist())
                m_generated.extend(t["times"][:, 5].tolist())
            with open(part["agents_file"]) as src:
                for line in src:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    merged.write(line + "\n")
                    n_lines += 1
                    r = json.loads(line)
                    rows.append(hashlib.sha256(json.dumps(comparable(r), sort_keys=True, default=str)
                                               .encode()).hexdigest())
                    for d in r.get("llm_call_details") or ():
                        calls["all"] += 1
                        if d.get("t_end") is None or d.get("service_s") is None:
                            continue
                        sent = d["t_end"] - d["service_s"]
                        asked = sent - (d.get("recovery_s") or 0.0)   # a retried call: approximate
                        m_start.append(sent)
                        m_end.append(d["t_end"])
                        q_start.append(asked - (d.get("wait_s") or 0.0))
                        q_end.append(asked)
                        m_prompt.append(d.get("prompt_tokens") or 0)
                        m_generated.append(d.get("completion_tokens") or 0)
                        calls["retried"] += bool(d.get("recovery_s"))
                        if by_role is not None and "error" not in r:   # the campaigns that finished, as `timed`
                            b, e = by_role.setdefault(d.get("role"), (array("d"), array("d")))
                            b.append(sent)
                            e.append(d["t_end"])
                    if "error" in r:
                        errors[r["error"].split(":", 1)[0]] += 1
                        continue
                    c.update(sims=r["n_sims"], llm_calls=r["n_llm_calls"], tool_calls=r["n_tool_calls"],
                             feasible_agents=int(bool(r["best"]["is_good"])), fixture=r["fixture"],
                             generations=len(r["generations"]))
                    touches.update(r["touches"])
                    devices.update(r["devices"])
                    verdicts.update(r["touch_checks"])
                    sources.update(r["sources"])
                    refill.update(r["refill"])
                    c_start.append(r["t_start"])
                    c_end.append(r["t_end"])
                    # each evaluation's submission: its generation's t0, or (the AUTONMI layer's asynchronous
                    # mode, which has no generation) its own, in t_submit
                    submitted = [g["t0"] for g in r["generations"] for _ in range(g["n_evaluated"])]
                    submitted += r.get("t_submit") or ()
                    gen_s.extend(g["elapsed_s"] for g in r["generations"])
                    for t0, w, d in zip(submitted, r["wait_s"], r["sim_s"]):
                        e_sub.append(t0)
                        e_start.append(t0 + w)
                        e_end.append(t0 + w + d)
                        wait_s.append(w)
                        sim_s.append(d)
                    if r.get("worker_peak_rss_mb") is not None:
                        worker_rss.append(r["worker_peak_rss_mb"])
    os.replace(tmp, out / "agents.jsonl")
    services = [p["llm_service"] for p in parts]
    for p, sv in zip(parts, services):
        budgets.update(sv["budgets"])
        tool_calls.update(p["tool_calls"])
    service = {"calls": sum(sv["calls"] for sv in services), "budgets": dict(budgets),
               **{k: sum(sv.get(k, 0) for sv in services) for k in ("empty", "errors", "retries")},
               "finish_reasons": dict(sum((Counter(sv.get("finish_reasons", {})) for sv in services), Counter())),
               "tokens": {k: sum(sv.get("tokens", {}).get(k, 0) for sv in services)
                          for k in ("prompt", "completion", "reasoning")}}
    latencies = [x for sv in services for x in sv.get("latency_s", [])]
    a = knobs["agents"]
    lock = json.loads((PARENT_DIR / "PARENT.lock").read_text())
    peaks = [p["peak_rss_mb"] for p in parts] or [None]
    first = parts[0] if parts else {}
    t_end_run = time.time()
    lo, hi = (min(c_start), max(c_end)) if c_start else (t_end_run, t_end_run)
    phase = hi - lo
    workers = sum(p["sim_workers"] or 0 for p in parts)
    rate = (lambda n: round(n / phase, 3)) if phase > 0 else (lambda n: None)
    camp = occupancy(c_start, c_end, lo, hi, n_ref=a["count"])
    timed = len(c_start)
    if not timed:   # nothing finished (an early interruption): unknown, not zero
        camp = {"max": None, "mean": None, "time_ge_95pct_s": None}
    speed = {
        # the speed numbers come from the campaigns that finished; an interrupted run's are partial
        "coverage": {"campaigns_timed": timed, "campaigns_requested": a["count"],
                     "complete": timed == a["count"] and not interrupted},
        "campaigns_live": {"max": camp["max"], "mean": camp["mean"], "requested": a["count"],
                           "time_ge_95pct_s": camp["time_ge_95pct_s"],
                           "ramp_up_s": round(max(c_start) - lo, 3) if c_start else None},
        "agent_phase_s": round(phase, 3) if timed else None,
        "startup_s": round(max(0.0, lo - t_run), 3) if t_run and timed else None,
        "shutdown_s": round(max(0.0, t_end_run - hi), 3) if timed else None,
        # model_calls: the outer loop's (counts.llm_calls); model_calls_all_agents: every call in the details
        "rates_per_s": {"evaluations": rate(len(sim_s)), "model_calls": rate(c["llm_calls"]),
                        "model_calls_all_agents": rate(calls["all"]),
                        "generations": rate(c["generations"]), "campaigns": rate(len(c_start))},
        # sims_in_flight: submitted, not yet returned (on the AUTONMI layer from when the candidate was put on its
        # campaign's queue, so with its wait for a free Monitor slot)
        "sims_in_flight": occupancy(e_sub, e_end, lo, hi) if timed else None,
        "sims_executing": occupancy(e_start, e_end, lo, hi) if timed else None,   # on a simulation worker
        "sim_s": _pct(sim_s), "queue_wait_s": _pct(wait_s), "generation_s": _pct(gen_s),
        "model_latency_s": _pct(latencies),
        "loop_lag": [p.get("loop_lag") for p in parts],
        "sim_workers": workers,
        # the simulations' own time spread over the workers, against the time the agents ran: 1 means
        # the run took no longer than its simulations had to
        "sim_bound_ratio": round(sum(sim_s) / workers / phase, 4) if workers and phase > 0 else None,
    }
    timed_calls = len(m_start)
    emulated = knobs["llm"]["mode"] == "stub"
    speed["model_requests_in_flight"] = model_requests(
        m_start, m_end, lo, hi, model_refs, side="client",
        what="model requests admitted and sent whose reply is not back (time queued inside a server included)",
        source="each call's [t_end - service_s, t_end] (its call record)", emulated=emulated,
        calls_timed=timed_calls, calls_untimed=calls["all"] - timed_calls,
        calls_retried=calls["retried"]) if timed_calls and timed else None
    speed["model_requests_queued"] = model_requests(
        q_start, q_end, lo, hi, model_refs, side="client",
        what="model requests waiting for the client's admission caps (llm.max_in_flight, llm.admission_tokens)",
        source="each call's wait_s before its time in flight", emulated=emulated) if timed_calls and timed else None
    # the usage counts of each call (with the stub, its estimate: emulated)
    speed["model_tokens_per_s"] = model_tokens(m_end, m_prompt, m_generated, lo, hi, emulated) \
        if timed_calls and timed else None
    if by_role is not None and timed:
        speed["model_call_share"] = call_share(by_role, roles, timed, lo, hi)
    rec = {
        "status": None, "agent_layer": layer, "sim": sim, "llm": knobs["llm"]["mode"],
        "emulated": True, "method": "miniapp_standin",
        "note": "all physics output is synthetic (random-weight stand-in models)", "knobs": knobs,
        "run": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "miniapp_git": git_state(),
                "backends": sorted({p.get("backend") for p in parts if p.get("backend")})},
        "counts": {
            "agents": n_lines, "agents_requested": a["count"], "agents_failed": sum(errors.values()),
            "agent_processes": len(parts) + len(failed), "failed_agent_processes": list(failed),
            **{k: c[k] for k in ("sims", "llm_calls", "tool_calls")},
            "generations": c["generations"],
            "candidates_by_source": {k: sources[k] for k in decide.SOURCES},
            "refill_slots": {"asked": refill["asked"], "filled_by_model": refill["filled"]},
            "suspensions": sum(p["layer"].get("suspensions") or 0 for p in parts) if layer == "langgraph" else None,
            "sim_tasks": sum(p["sim_tasks"] for p in parts), "memory_writes": sum(p["memory_writes"] for p in parts),
            "reports": sum(p["reports"] for p in parts), "feasible_agents": c["feasible_agents"],
            "expected_per_agent": decide.expected(knobs)},
        "llm_service": service,
        "llm_check": llm_check(knobs, refill, service),
        "lookups": dict(tool_calls),
        "touch_check": {"evaluations": sum(devices.values()), "fixture": c["fixture"],
                        "pass": verdicts.pop("pass", 0), "fail": dict(verdicts)},
        "touches": dict(touches), "devices": dict(devices),
        "wall_seconds": round(wall_s, 3), "results_sha256": hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest(),
        "speed": speed,
        "peak_rss_mb": {"main_process": peak_rss_mb(), "agent_process_max": max(peaks, key=lambda x: x or 0),
                        "agent_processes_sum": round(sum(x or 0 for x in peaks), 1),
                        "sim_worker_max": max(worker_rss) if worker_rss else None},
        "versions": versions(), "parent": {k: lock[k] for k in ("source", "branch", "commit")},
        "device_probe": first.get("device"), "frameworks_imported": first.get("frameworks"),
        "sim_pool": first.get("sim_pool"), "sim_workers_per_process": first.get("sim_workers"),
        "sim_workers_total": workers,
        "threads": first.get("threads"),
        "pools": [p.get("pool") for p in parts], "pruned_tasks": sum(p.get("pruned_tasks", 0) for p in parts),
        "layer_stats": [p["layer"] for p in parts],
    }
    if errors:
        rec["errors"] = dict(errors)
    c = rec["counts"]
    # pass: every campaign requested finished, in every agent process, every real evaluation passed its
    # touch check, and the model check passed; interrupted: SIGTERM or Ctrl-C; otherwise fail
    rec["status"] = "interrupted" if interrupted else "pass" if not failed and c["agents"] == a["count"] \
        and not c["agents_failed"] and not rec["touch_check"]["fail"] and not rec["llm_check"]["fail"] else "fail"
    tmp = out / "run_record.json.tmp"              # written whole, then renamed into place
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, out / "run_record.json")
    for part in parts:                             # only now: both files are in place
        Path(part["agents_file"]).unlink(missing_ok=True)
    return rec


def speed_line(rec: dict) -> str:
    """The run's main speed numbers on one parsable line (stdout)."""
    s = rec["speed"]
    s = {**s, **{k: s[k] or {} for k in ("sims_in_flight", "sims_executing")}}
    lag = [x.get("p95_ms") for x in s["loop_lag"] if x and x.get("samples")]
    f = {"status": rec["status"], "layer": rec["agent_layer"], "campaigns": rec["counts"]["agents"],
         "live_max": s["campaigns_live"]["max"], "live_mean": s["campaigns_live"]["mean"],
         "time_ge95_s": s["campaigns_live"]["time_ge_95pct_s"], "ramp_s": s["campaigns_live"]["ramp_up_s"],
         "phase_s": s["agent_phase_s"], "evals_per_s": s["rates_per_s"]["evaluations"],
         "llm_per_s": s["rates_per_s"]["model_calls"], "in_flight_max": s["sims_in_flight"].get("max"),
         "executing_max": s["sims_executing"].get("max"), "sim_p50_s": s["sim_s"].get("p50"),
         "wait_p95_s": s["queue_wait_s"].get("p95"), "gen_p95_s": s["generation_s"].get("p95"),
         "llm_p95_s": s["model_latency_s"].get("p95"), "lag_p95_ms": max(lag) if lag else None,
         "sim_bound": s["sim_bound_ratio"]}
    # the decoding metric's counts, in the records that have them (the AUTONMI layer's)
    if "agents_live" in s or s.get("model_requests_in_flight"):
        opened, queued, live, dec = (s.get(k) or {} for k in ("model_requests_in_flight", "model_requests_queued",
                                                              "agents_live", "decoding_at_servers"))
        running = dec.get("running") or {}
        f.update({"llm_all_per_s": s["rates_per_s"].get("model_calls_all_agents"), "agents_live_max": live.get("max"),
                  "requests_open_max": opened.get("max"), "requests_open_mean": opened.get("mean"),
                  "queued_max": queued.get("max"), "decoding_max": running.get("max"),
                  "decoding_mean": running.get("mean"), "emulated": opened.get("emulated")})
    return "speed: " + " ".join(f"{k}={v}" for k, v in f.items())
