"""Compare two run records, for checks across machines or layers:

    python -m fusion_miniapp.compare_records A/run_record.json B/run_record.json [--tol 0.25]

Results must agree: the counts and the knobs other than the machine's own (sim.device, sim.work_root,
sim.threads_per_worker, workers.sim_workers), and the campaigns. Identical results_sha256 settles the
campaigns at once. Otherwise each campaign in the two neighbouring agents.jsonl files is compared by
agent_id: ids, booleans, strings (selection basis) and integers (generations, counts, candidate numbers)
exactly, floats within --rtol/--atol (macOS and Linux, or CPU and GPU, differ in the last digits). Without
the agents files, unlike digests are unverifiable and fail. Suspensions are compared only between runs of
the same layer (Academy has none). Speed numbers are compared within --tol and reported, never failed on.
Exit 0 when the results agree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

MACHINE_KNOBS = {("sim", "device"), ("sim", "work_root"), ("sim", "threads_per_worker"), ("workers", "sim_workers"),
                 ("agents", "per_process"), ("agents", "max_live")}   # the machine's layout, not the problem
EXACT_COUNTS = ("agents", "agents_requested", "agents_failed", "sims", "llm_calls", "tool_calls", "generations",
                "candidates_by_source", "refill_slots", "sim_tasks", "memory_writes", "reports", "feasible_agents")
SPEED = (("campaigns_live", "max"), ("campaigns_live", "mean"), ("agent_phase_s",), ("rates_per_s", "evaluations"),
         ("rates_per_s", "model_calls"), ("sims_in_flight", "max"), ("sims_executing", "max"), ("sim_s", "p50"),
         ("queue_wait_s", "p95"), ("generation_s", "p95"), ("model_latency_s", "p95"), ("sim_bound_ratio",))


def _get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _close(x, y, path: str, rtol: float, atol: float, out: list) -> None:
    """Recursive comparison: numbers (not booleans) within tolerance, everything else exactly."""
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            _close(x.get(k), y.get(k), f"{path}.{k}", rtol, atol, out)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append(f"{path}: {len(x)} != {len(y)} items")
        for i, (u, v) in enumerate(zip(x, y)):
            _close(u, v, f"{path}[{i}]", rtol, atol, out)
    elif isinstance(x, float) or isinstance(y, float):
        num = isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) \
            and not isinstance(y, bool)
        if num and not (math.isfinite(x) and math.isfinite(y)):    # NaN equals NaN; infinities exactly
            ok = (math.isnan(x) and math.isnan(y)) or x == y
        else:
            ok = num and (x == y or abs(x - y) <= atol + rtol * max(abs(x), abs(y)))
        if not ok:
            out.append(f"{path}: {x!r} != {y!r}")
    elif x != y:
        out.append(f"{path}: {x!r} != {y!r}")


def compare_campaigns(lines_a, lines_b, rtol: float = 1e-6, atol: float = 1e-9, limit: int = 20) -> list[str]:
    """The comparable parts of two runs' campaigns, paired by agent_id (record.comparable)."""
    from fusion_miniapp.record import comparable
    a = {r["agent_id"]: comparable(r) for r in map(json.loads, lines_a)}
    b = {r["agent_id"]: comparable(r) for r in map(json.loads, lines_b)}
    out = [f"campaign {k} only in {'the first' if k in a else 'the second'} run" for k in sorted(set(a) ^ set(b))]
    for k in sorted(set(a) & set(b)):
        _close(a[k], b[k], k, rtol, atol, out)
        if len(out) >= limit:
            out.append("... (stopped after the first differences)")
            break
    return out


def _verify(rec: dict, lines) -> list[str]:
    """Whether a run's agents.jsonl is that run's: its campaign count, unique ids and digest."""
    from fusion_miniapp.record import digest
    if lines is None:
        return ["missing"]
    try:
        ids = [json.loads(line)["agent_id"] for line in lines]
    except (ValueError, KeyError, TypeError):
        return ["malformed"]
    out = []
    if len(ids) != rec["counts"].get("agents"):
        out.append(f"{len(ids)} campaigns, the record says {rec['counts'].get('agents')}")
    if len(set(ids)) != len(ids):
        out.append("repeated campaign ids")
    if not out and digest(lines) != rec.get("results_sha256"):
        out.append("its digest is not the record's results_sha256")
    return out


def compare(a: dict, b: dict, tol: float = 0.25, agents_a=None, agents_b=None, rtol: float = 1e-6,
            atol: float = 1e-9) -> dict:
    """{"results": [differences that fail], "speed": [measurements apart by more than tol]}. agents_a/b:
    the runs' agents.jsonl lines, for a campaign-by-campaign comparison when the digests differ."""
    bad = []
    for name, r in (("first", a), ("second", b)):   # only two passing runs can agree
        if r.get("status") != "pass":
            bad.append(f"the {name} run's status is {r.get('status')!r}, not 'pass'")
        if r.get("touch_check", {}).get("fail") or r.get("llm_check", {}).get("fail"):
            bad.append(f"the {name} run has failed touch or model checks")
    if a.get("results_sha256") != b.get("results_sha256"):
        agents_a, agents_b = (None if x is None else list(x) for x in
                              (x() if callable(x) else x for x in (agents_a, agents_b)))
        problems = [f"the {name} run's agents.jsonl: {p}" for name, rec, lines in
                    (("first", a, agents_a), ("second", b, agents_b)) for p in _verify(rec, lines)]
        bad += [f"results_sha256 differs and {p}: unverifiable" for p in problems] or \
            compare_campaigns(agents_a, agents_b, rtol, atol)
    if a.get("agent_layer") == b.get("agent_layer") and a["counts"].get("suspensions") != b["counts"].get("suspensions"):
        bad.append(f"counts.suspensions: {a['counts'].get('suspensions')} != {b['counts'].get('suspensions')}")
    for k in EXACT_COUNTS:
        if a["counts"].get(k) != b["counts"].get(k):
            bad.append(f"counts.{k}: {a['counts'].get(k)} != {b['counts'].get(k)}")
    for section in sorted(set(a["knobs"]) | set(b["knobs"])):
        sa, sb = a["knobs"].get(section, {}), b["knobs"].get(section, {})
        for key in sorted(set(sa) | set(sb)):
            if (section, key) not in MACHINE_KNOBS and sa.get(key) != sb.get(key):
                bad.append(f"knobs.{section}.{key}: {sa.get(key)!r} != {sb.get(key)!r}")
    apart = []
    for path in SPEED:
        x, y = _get(a.get("speed", {}), path), _get(b.get("speed", {}), path)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and max(abs(x), abs(y)) > 0:
            rel = abs(x - y) / max(abs(x), abs(y))
            if rel > tol:
                apart.append(f"speed.{'.'.join(path)}: {x} vs {y} ({rel:.0%} apart)")
    return {"results": bad, "speed": apart}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tol", type=float, default=0.25, help="relative tolerance for speed numbers (report only)")
    p.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for campaign floats")
    p.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance for campaign floats")
    args = p.parse_args(argv)
    a, b = (json.loads(Path(x).read_text()) for x in (args.a, args.b))
    def agents(x):   # read only when the digests differ
        f = Path(x).with_name("agents.jsonl")
        return (lambda: f.read_text().splitlines() if f.exists() else None)
    d = compare(a, b, args.tol, agents(args.a), agents(args.b), rtol=args.rtol, atol=args.atol)
    for line in d["results"]:
        print("DIFFERENT", line)
    for line in d["speed"]:
        print("speed", line)
    print("results agree" if not d["results"] else f"{len(d['results'])} result differences")
    return 1 if d["results"] else 0


if __name__ == "__main__":
    sys.exit(main())
