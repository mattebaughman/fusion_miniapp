"""End to end: the real evaluate (smoke preset, CPU) behind both agent layers, through the entry point.

4 campaigns x population 2 x 2 generations, stub LLM, flag off then on, each run in its own interpreter
as a user would start it. Simulations run in spawned worker processes. Laptop timings only.
"""
import json
import os
import subprocess
import sys

from fusion_miniapp import record

AGENTS, P, G = 4, 2, 2
SIMS = P * G
FLAGS = {"langgraph": [], "academy": ["--use-academy"]}
RECORD_KEYS = ("status", "threads", "agent_layer", "sim", "llm", "emulated", "method", "note", "knobs", "counts", "llm_service",
               "lookups", "touch_check", "touches", "devices", "wall_seconds", "results_sha256", "peak_rss_mb",
               "versions", "parent", "device_probe", "frameworks_imported", "sim_pool", "sim_workers_per_process",
               "sim_workers_total", "layer_stats")


def run(layer, tmp_path):
    knobs = tmp_path / "e2e.yaml"
    knobs.write_text(json.dumps({"sim": {"device": "cpu"}, "agents": {"count": AGENTS},
                                 "outer": {"population": P, "generations": G}}))
    out, tmp = tmp_path / layer, tmp_path / f"tmp-{layer}"
    tmp.mkdir()
    p = subprocess.run([sys.executable, "-m", "fusion_miniapp.run", "--knobs", str(knobs), "--sim", "real",
                        "--llm", "stub", "--out", str(out), *FLAGS[layer]],
                       capture_output=True, text=True, timeout=300, env={**os.environ, "TMPDIR": str(tmp)})
    assert p.returncode == 0, p.stderr[-3000:]
    assert not list(tmp.iterdir())          # work and assets directories removed, workers exited cleanly
    agents = [json.loads(line) for line in (out / "agents.jsonl").read_text().splitlines()]
    return json.loads((out / "run_record.json").read_text()), agents, p.stderr


def comparable(result):
    """A campaign's result without timings and touch counts (which evaluations met a cold worker, with
    3 checkpoint loads instead of 2, depends on scheduling)."""
    assert len(result["sim_s"]) == SIMS and all(t > 0 for t in result["sim_s"])
    return record.comparable(result)


def test_real_evaluate_through_both_layers_end_to_end(tmp_path):
    got = {layer: run(layer, tmp_path) for layer in FLAGS}
    for layer, (rec, agents, err) in got.items():
        assert set(RECORD_KEYS) <= set(rec)
        c = rec["counts"]
        assert (rec["status"], rec["agent_layer"], rec["sim"], rec["llm"]) == ("pass", layer, "real", "stub")
        assert rec["threads"]["per_worker"] >= 1
        assert (c["agents"], c["agents_failed"], c["agent_processes"]) == (AGENTS, 0, 1)
        assert (c["sims"], c["sim_tasks"], c["memory_writes"]) == (AGENTS * SIMS, AGENTS * SIMS, AGENTS * G)
        assert c["llm_calls"] == rec["llm_service"]["calls"] == AGENTS * G
        assert rec["lookups"] == {"constraints": AGENTS, "abstract": AGENTS} and c["reports"] == AGENTS
        assert c["candidates_by_source"]["start"] == AGENTS * P and rec["emulated"] is True
        assert c["candidates_by_source"]["model"] == c["refill_slots"]["filled_by_model"] > 0
        assert c["suspensions"] == (AGENTS * G if layer == "langgraph" else None)   # one per generation
        # every evaluation made every touch the knobs imply
        assert rec["touch_check"] == {"evaluations": 16, "fixture": 0, "pass": 16, "fail": {}}
        assert all(a["touch_checks"] == {"pass": SIMS} and a["devices"] == {"cpu": SIMS} for a in agents)
        assert rec["devices"] == {"cpu": 16}
        t, builds = rec["touches"], rec["touches"]["context_build"]
        assert 1 <= builds <= 2                                  # one cold start per worker that ran (2 workers)
        assert t["checkpoint_load"] == 2 * 16 + builds and t["graph_forward"] == 16 * 3
        assert t["artifact_write"] == 16 * 11 and t["handoff_file_rewrite"] == 16
        assert t["small_forward"] == 16 * 2 * 3 and t["library_import"] == 4 * builds   # puffs x cap; 4 per worker
        # simulations ran in worker processes: torch was never loaded in the agent process
        assert (rec["sim_pool"], rec["sim_workers_per_process"], rec["sim_workers_total"]) == ("process", 2, 2)
        assert not rec["frameworks_imported"]["torch"] and rec["frameworks_imported"][layer]
        assert "cuda_available" in rec["device_probe"] and rec["device_probe"]["torch"]   # CPU knobs on any host
        assert rec["peak_rss_mb"]["sim_worker_max"] > 0
        assert rec["parent"]["commit"] == "9768bcb" and rec["knobs"]["sim"]["device"] == "cpu"
        assert "agents working=" in err and f"done={AGENTS}" in err and "HDF5-DIAG" not in err
    lg = sorted((comparable(r) for r in got["langgraph"][1]), key=lambda r: r["agent_id"])
    ac = sorted((comparable(r) for r in got["academy"][1]), key=lambda r: r["agent_id"])
    assert [r["agent_id"] for r in lg] == [f"a{i}" for i in range(AGENTS)]
    assert lg == ac


def test_cold_start_fraction_runs_evaluations_in_fresh_processes(tmp_path):
    """cold_start_fraction 1: every evaluation runs in a fresh single-use process, so every evaluation
    imports the libraries and loads all three checkpoints, as the real loop does."""
    from fusion_miniapp import run as entry
    knobs = tmp_path / "cold.yaml"
    knobs.write_text(json.dumps({"sim": {"device": "cpu", "cold_start_fraction": 1.0},
                                 "agents": {"count": 1}, "workers": {"sim_workers": 1},
                                 "outer": {"population": 2, "generations": 1}}))
    assert entry.main(["--knobs", str(knobs), "--sim", "real", "--out", str(tmp_path / "out")]) == 0
    rec = json.loads((tmp_path / "out" / "run_record.json").read_text())
    t = rec["touches"]
    assert rec["touch_check"] == {"evaluations": 2, "fixture": 0, "pass": 2, "fail": {}}
    assert (t["context_build"], t["library_import"], t["checkpoint_load"]) == (2, 8, 6)
