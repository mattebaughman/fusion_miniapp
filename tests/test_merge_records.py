"""CPU fixture campaigns, archived record format, and synthetic merge cases."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fusion_miniapp import compare_records, record


ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results/laptop_2026-11-25/r500_generations_1ms/run_record.json"


def cli(module, *args, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-m", f"fusion_miniapp.{module}", *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=180)


def load(directory):
    return json.loads((directory / "run_record.json").read_text())


def write(directory, row):
    directory.mkdir(exist_ok=True)
    (directory / "run_record.json").write_text(json.dumps(row))


def merge(directory, *launches):
    result = cli("merge_records", *launches, "--out", directory, cwd=directory.parent)
    assert result.returncode == 0, result.stderr
    return load(directory)


@pytest.fixture
def pair(tmp_path):
    """Two synthetic launches with disjoint campaign indices, based on the archived stub record."""
    a, b = tmp_path / "a", tmp_path / "b"
    for directory, first in ((a, 0), (b, 500)):
        row = json.loads(SAVED.read_text())
        row["knobs"]["agents"]["first"] = first
        write(directory, row)
    return a, b


def test_generations_smoke_matches_a_single_launch(tmp_path):
    """Three actual CPU runs using the stub model and fixture inner loop."""
    pytest.importorskip("autonmi")
    pytest.importorskip("wfMiniAPI.api")
    pytest.importorskip("radical.asyncflow")
    full, left, right = (tmp_path / name for name in ("full", "left", "right"))
    for path, first, count in ((full, 0, 8), (left, 0, 4), (right, 4, 4)):
        config = tmp_path / f"{path.name}.yaml"
        config.write_text(json.dumps({"agents": {"first": first}, "campaign": {"mode": "generations"}}))
        result = cli("layer_autonmi", "--preset", "smoke", "--llm", "stub", "--sim", "fixture",
                     "--knobs", config, "--agents", count, "--out", path, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
    # Writers and readers allow blank separators between campaign rows.
    with (left / "agents.jsonl").open("a") as rows_file:
        rows_file.write("\n \n")
    # Reverse argument order so that the digest and ranges cannot depend on CLI order.
    merged = merge(tmp_path / "merged", right, left)
    single = load(full)
    lines = [line for path in (right, left) for line in (path / "agents.jsonl").read_text().splitlines() if line.strip()]
    assert record.digest(lines) == merged["results_sha256"] == single["results_sha256"]
    assert compare_records.compare(merged, single)["results"] == []

    def counts(rec):   # all but the agent processes (one per launch here) and the message waits (times: their
        c = rec["counts"]   # n, sum and max merged, the percentiles left out)
        return {k: {r: {x: y for x, y in v.items() if x != "message_wait_s"} for r, v in c[k].items()}
                if k == "by_role" else c[k] for k in c if k != "agent_processes"}
    assert counts(merged) == counts(single)
    assert merged["counts"]["agents_per_campaign"] == single["counts"]["agents_per_campaign"]
    assert merged["counts"]["by_role"]["coordinator"]["finished"] == 8
    assert merged["counts"]["agent_processes"] == 2
    assert sorted(x["agents"]["first"] for x in merged["merged_from"]) == [0, 4]
    assert merged["campaign_gaps"] == []
    # Each variant replaces one real sidecar row without changing its run record.
    rows = [json.loads(line) for line in (left / "agents.jsonl").read_text().splitlines() if line.strip()]
    for marker, damaged in (("duplicate", rows[1]), ("outside", {**rows[0], "agent_id": "a99"}),
                            ("digest", {**rows[0], "seed": -1})):
        (left / "agents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [damaged, *rows[1:]]))
        out = tmp_path / marker
        result = cli("merge_records", right, left, "--out", out, cwd=tmp_path)
        assert result.returncode != 0, marker
        assert "Traceback" not in result.stderr and not (out / "run_record.json").exists()


@pytest.mark.parametrize(("path", "value"), [
    (("knobs", "agents", "first"), 499),
    (("knobs", "outer", "rng"), "per_agent"),
    (("agent_layer",), "academy"),
    (("sim",), "real"),
    (("run", "miniapp_git", "commit"), "another-miniapp-commit"),
    (("autonmi", "framework_commit", "commit"), "another-framework-commit"),
])
def test_refuses_overlapping_or_incompatible_launches(pair, tmp_path, path, value):
    a, b = pair
    row = load(b)
    parent = row
    for key in path[:-1]:
        parent = parent[key]
    assert parent[path[-1]] != value
    parent[path[-1]] = value
    write(b, row)
    out = tmp_path / "merged"
    result = cli("merge_records", a, b, "--out", out, cwd=tmp_path)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr and "No module named" not in result.stderr
    assert not (out / "run_record.json").exists()


def test_refuses_to_replace_a_source(pair, tmp_path):
    a, b = pair
    original = (a / "run_record.json").read_bytes()
    result = cli("merge_records", a, b, "--out", a, cwd=tmp_path)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr and "No module named" not in result.stderr
    assert (a / "run_record.json").read_bytes() == original


@pytest.mark.parametrize(("left", "right", "worst"), [
    ("pass", "interrupted", "interrupted"), ("fail", "interrupted", "fail"),
])
def test_status_gaps_and_missing_campaign_rows(pair, tmp_path, left, right, worst):
    a, b = pair
    for path, status in ((a, left), (b, right)):
        row = load(path)
        row["status"] = status
        row["autonmi"]["ok"] = status == "pass"
        row["autonmi"]["failed_checks"] = [] if status == "pass" else [f"{path.name}_check"]
        if path == b:
            row["knobs"]["agents"]["first"] += 5
            row["counts"]["by_role"]["coordinator"]["finished"] = 0
        write(path, row)
    merged = merge(tmp_path / "merged", a, b)
    assert merged["status"] == worst and merged["autonmi"]["ok"] is False
    assert sorted(merged["autonmi"]["failed_checks"]) == (["b_check"] if left == "pass" else ["a_check", "b_check"])
    assert merged["campaign_gaps"] == [[500, 505]]
    assert merged["counts"]["agents"] == 1000 and merged["counts"]["by_role"]["coordinator"]["finished"] == 500
    assert merged.get("results_sha256") is None


def test_incomplete_async_counters_remain_unknown(tmp_path):
    # An interrupted run has real scored/kept counts, but no cycle or fill totals.
    # Give the archived pair disjoint ranges just for this merge fixture.
    a, b = tmp_path / "a", tmp_path / "b"
    for directory, source, first in ((a, "r10000_async_1ms", 0), (b, "r10000_async_sigterm", 10000)):
        row = load(SAVED.parents[1] / source)
        row["knobs"]["agents"]["first"] = first
        write(directory, row)
    merged = merge(tmp_path / "merged", a, b)
    assert merged["status"] == "interrupted"
    asynchronous = merged["counts"]["asynchronous"]
    assert asynchronous["results_scored"] == 320000 + 161920
    assert asynchronous["cycles"] is None and asynchronous["parameters_filled"] is None
    assert asynchronous["survivor_waits"] is None


@pytest.mark.parametrize("delta", [1, 30])
def test_live_series_and_rates_respect_actual_overlap(pair, tmp_path, delta):
    a, b = pair
    for path, start, live, calls, rate in ((a, 0, 10, 3, 11), (b, delta, 20, 7, 13)):
        row = load(path)
        row["run"]["timestamp"] = f"2026-11-25T00:00:{start + 4:02d}+00:00"
        row["wall_seconds"] = row["speed"]["agent_phase_s"] = 4.0
        row["speed"]["live"] = {"fields": ["t_s", "agents_live", "model_requests_open"], "interval_s": 1,
            "t0": 1795564800.0 + start, "series": [[t, live, calls] for t in range(4)] + [[4, 0, 0]]}
        row["speed"]["rates_per_s"]["model_calls_all_agents"] = rate
        write(path, row)
    merged = merge(tmp_path / "merged", a, b)
    assert merged["speed"]["agent_phase_s"] == {"min": 4.0, "max": 4.0}
    assert merged["speed"]["rates_per_s"]["model_calls_all_agents"] == (24 if delta == 1 else None)
    series = merged["speed"]["live"]
    for name, overlap_peak, separate_peak in (("agents_live", 30, 20), ("model_requests_open", 10, 7)):
        index = series["fields"].index(name)
        assert max(point[index] for point in series["series"] if point[index] is not None) == (
            overlap_peak if delta == 1 else separate_peak)


def test_missing_finished_count_and_different_machine_layout(pair, tmp_path):
    a, b = pair
    row = load(b)
    del row["counts"]["by_role"]["coordinator"]["finished"]
    row["knobs"]["workers"]["sim_workers"] += 1
    write(b, row)
    merged = merge(tmp_path / "merged", a, b)
    assert merged["counts"]["by_role"]["coordinator"]["finished"] is None
    assert merged["knobs"]["workers"]["sim_workers"] is None


def test_output_lock_and_error_counts(pair, tmp_path):
    a, b = pair
    for folder, errors in ((a, {"ModelError": 1}), (b, {"WorkerError": 1, "ModelError": 2})):
        row = load(folder)
        row.update(status="fail", errors=errors)
        row["counts"]["agents_failed"] = sum(errors.values())
        write(folder, row)
    out = tmp_path / "merged"
    out.mkdir()
    lock = out / "run.lock"
    lock.write_text("active")
    result = cli("merge_records", a, b, "--out", out, cwd=tmp_path)
    assert result.returncode != 0 and lock.read_text() == "active"
    assert not (out / "run_record.json").exists()
    lock.unlink()
    merged = merge(out, a, b)
    assert merged["errors"] == {"ModelError": 3, "WorkerError": 1}
    assert merged["counts"]["agents_failed"] == 4
    assert not lock.exists()


def test_refused_reply_samples_are_opt_in_and_capped_across_sources(pair, tmp_path):
    a, b = pair
    for i, folder in enumerate(pair):
        row = load(folder)
        row["knobs"]["record"]["extras"] = True
        row["llm_service"]["unusable_samples"] = {"evaluator": [str(i) * 200] * 4}
        write(folder, row)
    expected = {"evaluator": ["0" * 200] * 4 + ["1" * 200]}
    assert merge(tmp_path / "both", b, a)["llm_service"]["unusable_samples"] == expected
    for folder in pair:
        row = load(folder)
        row["knobs"]["record"]["extras"] = False
        write(folder, row)
    assert "unusable_samples" not in merge(tmp_path / "neither", a, b)["llm_service"]
