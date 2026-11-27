"""Combine completed launch records, including the partial records of interrupted launches.

    python -m fusion_miniapp.merge_records RUN_DIR... --out DIR

For launches run side by side (layer_autonmi's --slice; launch/stellar_ai_autonmi.sbatch runs this at its end
with FUSION_LAUNCHES > 1). The output is one run_record.json. Each source keeps its original campaign rows.
A null digest means at least one launch had no agents.jsonl. Durations and latency
percentiles are ranges across launches. Throughput sums describe the launches'
individual rates when their measurement windows overlap.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from bisect import bisect_right
from copy import deepcopy
from datetime import datetime
from math import isfinite
from pathlib import Path

from fusion_miniapp import record
from fusion_miniapp.compare_records import ADDED_KNOBS, MACHINE_KNOBS, _get


STATUS = ("pass", "interrupted", "fail")


def _same(values, field):
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"launches differ at {field}")
    return values[0]


def _total(values):
    """A missing measurement leaves the total unknown."""
    if any(value is None for value in values):
        return None
    if not all(type(value) in (float, int) for value in values):
        raise ValueError("expected numeric counts")
    return sum(values)


def _add_counts(values, path):
    """Add nested counters. A missing histogram bin contributes zero."""
    key = path[-1]
    if key in ("roles", "expected_by_role", "expected_per_agent", "agents_per_campaign"):
        return _same(values, ".".join(path))
    if any(value is None for value in values):
        return None
    if key == "parameters_filled":
        return [_total(column) for column in zip(*values, strict=True)]
    if key == "message_wait_s":
        n = _total([v["n"] for v in values])
        total = _total([v.get("sum", 0) for v in values])
        maxima = [v["max"] for v in values if v.get("max") is not None]
        return {"n": n, "sum": total, "mean": total / n if n and total is not None else None,
                "max": max(maxima, default=None)}
    if all(isinstance(value, dict) for value in values):
        sparse = path in {("errors",), ("touches",), ("lookups",), ("devices",), ("touch_check", "fail"),
                          ("llm_service", "finish_reasons"), ("llm_service", "budgets")}
        merged = {}
        for k in sorted(set().union(*values)):
            if path == ("counts",) and k == "failed_agent_processes":
                continue  # these indices are translated to merged population order below
            if not sparse and any(k not in v for v in values):
                merged[k] = None
            else:
                merged[k] = _add_counts([v[k] for v in values if k in v], (*path, k))
        return merged
    return _total(values)


def _read_launch(folder):
    rec = json.loads((folder / "run_record.json").read_text())
    if "merged_from" in rec:
        raise ValueError(f"{folder}: supply original launch records")
    if rec.get("status") not in STATUS:
        raise ValueError(f"{folder}: unknown or missing launch status")
    agents = rec["knobs"]["agents"]
    agents.setdefault("first", 0)
    if type(agents["first"]) is not int or agents["first"] < 0:
        raise ValueError(f"{folder}: invalid agents.first")
    if type(agents["count"]) is not int or agents["count"] <= 0:
        raise ValueError(f"{folder}: invalid agents.count")
    if rec["counts"]["agents_requested"] != agents["count"]:
        raise ValueError(f"{folder}: counts.agents_requested differs from the configured count")
    return rec


def _validate(records):
    layer = _same([r["agent_layer"] for r in records], "agent_layer")
    _same([r["sim"] for r in records], "sim")
    for r in records:
        if r["status"] == "pass" and (r["touch_check"]["fail"] or r["llm_check"]["fail"]
                                      or (layer == "autonmi" and r["autonmi"]["ok"] is not True)):
            raise ValueError("passing launch has failed source checks")
        mode = r["knobs"].get("campaign", {}).get("mode", "generations")
        if r.get("campaign_mode", mode) != mode:
            raise ValueError("campaign_mode disagrees with knobs.campaign.mode")
    _same([r.get("campaign_mode", r["knobs"].get("campaign", {}).get("mode", "generations"))
           for r in records], "campaign_mode")
    commits = [("run.miniapp_git.commit", [r["run"]["miniapp_git"]["commit"] for r in records])]
    if layer == "autonmi":
        commits.append(("autonmi.framework_commit.commit",
                        [r["autonmi"]["framework_commit"]["commit"] for r in records]))
    for key, ids in commits:
        if not _same(ids, key):
            raise ValueError(f"cannot verify {key}")
    knobs = [r["knobs"] for r in records]
    ignored = MACHINE_KNOBS | {("agents", "first"), ("agents", "count")}
    for section in sorted(set().union(*knobs)):
        sections = [k.get(section, {}) for k in knobs]
        for key in sorted(set().union(*sections)):
            path = (section, key)
            if path not in ignored:
                _same([s.get(key, ADDED_KNOBS.get(path)) for s in sections], f"knobs.{section}.{key}")


def _combined_digest(launches):
    # Reuse the writer's digest on both sides. Spooling verified rows bounds memory
    # to one campaign plus its hash per row, even with per-candidate extras enabled.
    complete = True
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as combined:
        for folder, rec in launches:
            source = folder / "agents.jsonl"
            if not source.exists():
                complete = False
                continue
            agents = rec["knobs"]["agents"]
            indices = range(agents["first"], agents["first"] + agents["count"])
            seen = set()

            def rows(stream):
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    aid = row["agent_id"]
                    index = int(aid[1:]) if isinstance(aid, str) and aid.startswith("a") else -1
                    if index not in indices or aid != f"a{index}" or index in seen:
                        raise ValueError(f"{folder}: duplicate or out-of-range campaign {aid!r}")
                    seen.add(index)
                    combined.write(line.rstrip("\n") + "\n")
                    yield row

            with source.open() as stream:
                digest = record.digest(rows(stream))
            if digest != rec["results_sha256"] or len(seen) != rec["counts"]["agents"]:
                raise ValueError(f"{source}: campaign rows differ from the source record")
        if complete:
            combined.seek(0)
            return record.digest(combined)
    return None


def merge(folders, out):
    """Build a merged record from the original launches in campaign order."""
    if any(folder.resolve() == out.resolve() for folder in folders):
        raise ValueError("the output directory is a source launch")
    launches = [(folder, _read_launch(folder)) for folder in folders]
    launches.sort(key=lambda item: item[1]["knobs"]["agents"]["first"])
    records = [r for _, r in launches]
    _validate(records)
    sources, gaps, end = [], [], None
    for folder, rec in launches:
        agents, run = rec["knobs"]["agents"], rec["run"]
        first, count = agents["first"], agents["count"]
        if end is not None and first < end:
            raise ValueError(f"{folder}: campaign ranges overlap at {first}")
        if end is not None and first > end:
            gaps.append([end, first])
        end = first + count
        sources.append({"folder": os.path.relpath(folder, out), "agents": {"first": first, "count": count},
                        "status": rec["status"], "run": {"timestamp": run["timestamp"],
                                                         "miniapp_git": run["miniapp_git"]}})
    first = records[0]
    knobs = deepcopy(first["knobs"])
    knobs["agents"]["count"] = sum(s["agents"]["count"] for s in sources)
    # A layout or machine setting that varies has no single value for the merged record.
    for section, key in MACHINE_KNOBS:
        settings = [r["knobs"].get(section, {}).get(key, ADDED_KNOBS.get((section, key))) for r in records]
        if any(value != settings[0] for value in settings[1:]):
            knobs.setdefault(section, {})[key] = None
    counts = _add_counts([r["counts"] for r in records], ("counts",))
    # Population IDs restart in each launch. Number them in merged_from order.
    offset, failed = 0, []
    for r in records:
        failed.extend(offset + i for i in r["counts"].get("failed_agent_processes", []))
        offset += r["counts"]["agent_processes"]
    counts["failed_agent_processes"] = failed
    rec = {k: first[k] for k in ("agent_layer", "sim", "llm", "method", "emulated", "note", "campaign_mode",
                                 "parent") if k in first}
    rec.update(status=max((r["status"] for r in records), key=STATUS.index), merged_from=sources,
               campaign_gaps=gaps, knobs=knobs, counts=counts,
               run={"miniapp_git": {"commit": first["run"]["miniapp_git"]["commit"],
                                    "dirty": _dirty([r["run"]["miniapp_git"] for r in records])}},
               results_sha256=_combined_digest(launches))
    for field in ("lookups", "touch_check", "touches", "devices"):
        rec[field] = _add_counts([r[field] for r in records], (field,))
    if any("errors" in r for r in records):
        rec["errors"] = _add_counts([r.get("errors", {}) for r in records], ("errors",))
    service = {}
    for field in ("calls", "empty", "errors", "retries", "budgets", "finish_reasons", "tokens"):
        service[field] = _add_counts([r["llm_service"][field] for r in records], ("llm_service", field))
    # Only launches that explicitly recorded extras can contribute refused replies.
    # Keep the original record's limit across the whole group.
    samples = {}
    for r in records:
        if r["knobs"].get("record", {}).get("extras"):
            for role, replies in (r["llm_service"].get("unusable_samples") or {}).items():
                remaining = 5 - len(samples.get(role, []))
                if replies and remaining:
                    samples.setdefault(role, []).extend(text[:200] for text in replies[:remaining])
    if samples:
        service["unusable_samples"] = samples
    rec["llm_service"] = service
    slots = counts["refill_slots"]
    rec["llm_check"] = record.llm_check(knobs, {"asked": slots["asked"], "filled": slots["filled_by_model"]}, service)
    for r in records:
        for problem in r["llm_check"]["fail"]:
            if problem not in rec["llm_check"]["fail"]:
                rec["llm_check"]["fail"].append(problem)
    if first["agent_layer"] == "autonmi":
        parts = [r["autonmi"] for r in records]
        rec["autonmi"] = {"ok": all(p["ok"] is True for p in parts),
                          "failed_checks": sorted({c for p in parts for c in p["failed_checks"]}),
                          "framework_commit": {"commit": parts[0]["framework_commit"]["commit"],
                                               "dirty": _dirty([p["framework_commit"] for p in parts])}}
        if any("inner_loop_gpus" in p for p in parts):
            rec["autonmi"]["inner_loop_gpus"] = {"workers_placed": _total(
                [p.get("inner_loop_gpus", {}).get("workers_placed") for p in parts])}
    rec["sim_workers_total"] = _total([r.get("sim_workers_total") for r in records])
    rec["peak_rss_mb"] = {"main_process": _total([r["peak_rss_mb"].get("main_process") for r in records])}
    # CROSS-CAMPAIGN [XC-K5] (see CROSS_CAMPAIGN_PLAN.md): P0 (planned): the cross_campaign sections of the launches
    #   read (control plane and ledger counters) would be summed here and, with --xc-dir, the entry point's summary
    #   added. A launch that wrote nothing stops merge above, so coverage against the run manifest
    #   (xc/manifest.json) is the entry point's to report.
    rec["wall_seconds"] = _range([r.get("wall_seconds") for r in records])
    rec["speed"] = _speed(launches)
    return rec


def _dirty(states):
    values = [s.get("dirty") for s in states]
    return True if True in values else None if None in values else False


def _range(values):
    measured = [v for v in values if v is not None]
    return {"min": min(measured, default=None), "max": max(measured, default=None)}


def _speed_sums(speeds, rate_keys):
    def summed(*path):
        return _total([_get(s, path) for s in speeds]) if speeds else None

    return {
        "rates_per_s": {key: summed("rates_per_s", key) for key in rate_keys},
        "model_tokens_per_s": {key: summed("model_tokens_per_s", key)
                               for key in ("prompt_per_s", "generated_per_s")},
        "model_requests_in_flight": {"mean": summed("model_requests_in_flight", "mean")},
        "decoding_at_servers": {"running": {"mean": summed("decoding_at_servers", "running", "mean")}},
    }


def _live_samples(rec):
    """Return the absolute sample times and safe outer bounds on this launch's client activity."""
    live = _get(rec, ("speed", "live")) or {}
    start = live.get("t0")
    if type(start) not in (float, int) or not isfinite(start):
        start = None
    rows = {}
    fields = live.get("fields") or []
    if start is not None and "t_s" in fields:
        for values in live.get("series") or []:
            row = dict(zip(fields, values))
            offset = row.get("t_s")
            if type(offset) in (float, int) and isfinite(offset):
                rows[round(start + offset, 6)] = (row.get("agents_live"), row.get("model_requests_open"))
    times = sorted(rows)
    interval = live.get("interval_s")
    if type(interval) not in (float, int) or not isfinite(interval) or interval <= 0:
        interval = 0  # exact samples only
    end = None
    try:
        stamp = datetime.fromisoformat(rec["run"]["timestamp"])
        if stamp.utcoffset() is not None:
            # record.write stamps the finished run with whole-second resolution.
            end = max(stamp.timestamp() + 1, times[-1]) if times else stamp.timestamp() + 1
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        pass
    return times, rows, interval, start, end


def _aligned_live(samples):
    """Align asynchronous samples on their union of epochs, bounding how long each reading lasts."""
    epochs = sorted({t for times, *_ in samples for t in times})
    origin = epochs[0] if epochs else None
    series = []
    for epoch in epochs:
        observed = []
        for times, rows, interval, start, end in samples:
            j = bisect_right(times, epoch) - 1
            if j >= 0 and epoch <= times[-1] and round(epoch - times[j], 6) <= interval:
                observed.append(rows[times[j]])
            elif (start is not None and epoch < start) or (end is not None and epoch >= end):
                observed.append((0, 0))
            else:
                observed.append((None, None))
        series.append([round(epoch - origin, 6), *(_total(column) for column in zip(*observed))])
    return {
        "t0": origin, "interval_s": None, "fields": ["t_s", "agents_live", "model_requests_open"], "series": series,
        "rule": "Grid is the union of observed epochs. Carry each reading for at most one sampling interval, "
                "through its last sample. Use zero before that launch's live clock starts or after its final "
                "record timestamp plus one second. Unknown times, missing readings and gaps yield null.",
    }


def _overlap_groups(windows):
    """Maximal sets sharing positive time. A-B and B-C can overlap while A-C never do."""
    edges = sorted({t for w in windows if w for t in w})
    groups = set()
    for start, end in zip(edges, edges[1:]):
        active = frozenset(i for i, w in enumerate(windows) if w and w[0] <= start and end <= w[1])
        if len(active) >= 2:
            groups.add(active)
    return sorted(sorted(g) for g in groups if not any(g < h for h in groups))


def _speed(launches):
    """Sum recorded launch rates only when their live-sampling windows intersect for positive time."""
    speeds = [rec.get("speed") or {} for _, rec in launches]
    slices = [{"folder": folder.name, "first": rec["knobs"]["agents"].get("first", 0)} for folder, rec in launches]
    samples = [_live_samples(rec) for _, rec in launches]
    windows = [(ts[0], ts[-1]) if ts else None for ts, *_ in samples]
    rate_keys = sorted(set().union(*(s.get("rates_per_s") or {} for s in speeds)))
    groups = []
    for indices in _overlap_groups(windows):
        groups.append({
            "launches": [slices[i] for i in indices], "start_epoch_s": max(windows[i][0] for i in indices),
            "end_epoch_s": min(windows[i][1] for i in indices),
            **_speed_sums([speeds[i] for i in indices], rate_keys),
        })
    all_overlap = bool(windows) and all(windows) and max(w[0] for w in windows) < min(w[1] for w in windows)
    lag = []
    for i, (_, rec) in enumerate(launches):
        rows = speeds[i].get("loop_lag") or []
        n = _get(rec, ("autonmi", "populations")) or len(rows)
        for p in range(max(len(rows), n)):
            row = (rows[p] if p < len(rows) else None) or {}
            lag.append({**slices[i], "population": p, "p95_ms": row.get("p95_ms"), "max_ms": row.get("max_ms")})

    def spread(*path):
        return _range(_get(s, path) for s in speeds)

    return {
        **_speed_sums(speeds if all_overlap else [], rate_keys),
        "overlap": {
            "all": bool(all_overlap),
            "rule": "Positive shared interval between first and last live samples. "
                    "Rate and mean sums use the full individual measurement windows.",
            "windows": [{**part, "start_epoch_s": w[0] if w else None, "end_epoch_s": w[1] if w else None}
                        for part, w in zip(slices, windows)],
            "groups": groups,
        },
        "agent_phase_s": spread("agent_phase_s"), "sim_bound_ratio": spread("sim_bound_ratio"),
        "agents_live": {"all_live_s": spread("agents_live", "all_live_s")},
        "model_latency_s": {k: spread("model_latency_s", k) for k in ("p50", "p95")},
        "loop_lag": {**{k: _range(row[k] for row in lag) for k in ("p95_ms", "max_ms")}, "per_population": lag},
        "live": _aligned_live(samples),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("launch", type=Path, nargs="+", help="directory with run_record.json")
    parser.add_argument("--out", type=Path, required=True, help="merged record directory")
    args = parser.parse_args(argv)
    try:
        result = merge(args.launch, args.out)
        args.out.mkdir(parents=True, exist_ok=True)
        # Share the exclusive marker used by the normal run entry points. A live
        # launch may hold only this file and have no result or campaign rows yet.
        lock = args.out / "run.lock"
        with lock.open("x"):
            try:
                dest = args.out / "run_record.json"
                existing = json.loads(dest.read_text()) if dest.exists() else {"merged_from": []}
                if not isinstance(existing, dict) or "merged_from" not in existing or (args.out / "agents.jsonl").exists():
                    raise ValueError(f"{args.out} contains an existing source run")
                with tempfile.NamedTemporaryFile("w", dir=args.out, delete=False) as temp:
                    try:
                        json.dump(result, temp, indent=1, allow_nan=False)
                        temp.write("\n")
                        temp.flush()
                        os.replace(temp.name, dest)
                    finally:
                        Path(temp.name).unlink(missing_ok=True)
            finally:
                lock.unlink()
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(1, f"merge_records: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
