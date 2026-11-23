"""evaluate / evaluate_fixture: contract keys, touch check, knob scaling, process safety."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest

from fusion_miniapp import PARENT_DIR, sim
from fusion_miniapp.sim import evaluate, evaluate_fixture   # torch loads at the first evaluate, not here
from fusion_miniapp.touches import KINDS

DEVICE = os.environ.get("FUSION_MINIAPP_TEST_DEVICE", "cpu")   # cpu, mps or cuda
DEV = {"device": DEVICE}
PARENT_KEYS = {
    "params", "candidate_idx", "input_te_core_keV", "te_core_metric", "te_core_keV", "ti_core_keV",
    "te_wall_eV", "ti_wall_eV", "wall_ok", "pedestal_ok", "elm_ok", "is_good", "te_ped_eV", "ne_ped_m3",
    "p_ped_Pa", "q_peak_proxy", "te_wall_div_eV", "ti_wall_div_eV", "puff_best", "summary_path", "inner_loop",
}
CONTRACT_KEYS = PARENT_KEYS | {"emulated", "device", "elapsed_seconds", "t_start", "worker_peak_rss_mb", "touches",
                                "touch_check"}


def _available(device):
    import torch
    return device == "cpu" or (device == "cuda" and torch.cuda.is_available()) or (
        device == "mps" and torch.backends.mps.is_available())


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_evaluate_contract_keys_and_touch_check(device):
    if not _available(device):
        pytest.skip(f"{device} not available here")
    r = evaluate({"a1_ne": 3.0, "Delta": 0.06}, {"device": device}, agent_id="t/1", seed=1)
    assert set(r) == CONTRACT_KEYS
    assert r["emulated"] is True and r["device"].startswith(device)
    assert set(r["touches"]) == set(KINDS)
    assert r["touch_check"] == "pass", r["touch_check"]
    assert r["params"]["a1_ne"] == 3.0 and r["params"]["Delta"] == 0.06
    json.dumps(r, allow_nan=False)                              # strict JSON: NaN became null


def test_context_build_makes_the_real_loops_counts(monkeypatch):
    evaluate({}, DEV)                                           # this process has loaded the libraries
    monkeypatch.setattr(sim, "_CONTEXTS", {})                   # a new context in a warm process
    r = evaluate({}, DEV)
    t = r["touches"]
    assert t["context_build"] == 1 and t["library_import"] == 0 and t["geometry_read"] == 2
    assert t["checkpoint_load"] == 3 and t["model_to_device"] == 3      # three loads per evaluation
    assert t["graph_forward"] == 3 and t["artifact_write"] == 11        # 1 + 2 forwards; 11 files
    assert (t["handoff_file_write"], t["handoff_file_rewrite"], t["handoff_file_read"]) == (1, 1, 2)
    assert t["result_return"] == 1 and r["touch_check"] == "pass"


def test_warm_smoke_call_is_fast():
    evaluate({}, DEV)
    r = evaluate({}, DEV)
    assert r["touches"]["context_build"] == 0 and r["touches"]["checkpoint_load"] == 2
    assert r["elapsed_seconds"] < 1.0                           # functional check, not a performance claim


@pytest.mark.parametrize("name, replacement, missing", [
    ("read_state_nc", lambda path: np.ones((3, 38, 98)), "handoff_file_read=1"),          # skip a file read
    ("forward_conditioned", lambda model, aux, hist, puff: np.asarray(hist[0]), "small_forward=0"),  # skip the GPU
])
def test_removed_touch_fails_check(monkeypatch, name, replacement, missing):
    from fusion_miniapp import standins
    monkeypatch.setattr(standins, name, replacement)
    monkeypatch.setattr(sim, "_CONTEXTS", {})                   # rebuild so the context picks up the patch
    r = evaluate({}, DEV)
    assert r["touch_check"].startswith("fail: ") and missing in r["touch_check"], r["touch_check"]
    monkeypatch.undo()
    assert evaluate({}, DEV)["touch_check"] == "pass"


def test_knobs_scale_the_counts():
    one = evaluate({}, {**DEV, "rollout_steps": 1})["touches"]
    three = evaluate({}, {**DEV, "rollout_steps": 3})["touches"]
    assert (one["graph_forward"], three["graph_forward"]) == (2, 4)
    assert (one["artifact_write"], three["artifact_write"]) == (9, 13)
    assert one["checkpoint_load"] == three["checkpoint_load"] == 2
    r = evaluate({}, {**DEV, "puffs": 3, "fixed_point_cap": 4})
    assert r["touches"]["small_forward"] == 12 and r["touch_check"] == "pass"   # every fixed point runs to the cap
    assert evaluate({}, {**DEV, "puffs": 2, "fixed_point_cap": 40})["touches"]["small_forward"] == 80


def test_parent_files_byte_identical_after_evaluate():
    evaluate({}, DEV)
    lock = json.loads((PARENT_DIR / "PARENT.lock").read_text())
    for rel, want in lock["sha256"].items():
        assert hashlib.sha256((PARENT_DIR / rel).read_bytes()).hexdigest() == want, rel


def test_no_nonpublic_modules_and_no_device_pinning(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    evaluate({}, DEV)
    for name in ("matey", "xgc_reader", "core_edge_optimization", "run_matey_inference_1step", "puff_optimizer"):
        assert name not in sys.modules, name
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_work_directory_deleted_unless_kept(tmp_path):
    r = evaluate({}, {**DEV, "work_root": str(tmp_path)})
    assert r["summary_path"] == "" and not list(tmp_path.glob("eval_*"))
    r = evaluate({}, {**DEV, "work_root": str(tmp_path), "keep_artifacts": True}, agent_id="kept")
    kept = list(tmp_path.glob("eval_kept_*"))
    assert len(kept) == 1 and sum(p.is_file() for p in kept[0].rglob("*")) == 11
    assert json.loads(open(r["summary_path"]).read())["method"] == sim.STANDIN_METHOD


def test_candidate_handling():
    with pytest.raises(ValueError, match="unknown candidate parameter"):
        evaluate_fixture({"a3": 1.0})
    with pytest.raises(ValueError, match="not finite"):
        evaluate_fixture({"a1_ne": float("nan")})
    assert evaluate_fixture({"Delta": 0.5})["params"]["Delta"] == 0.10     # clipped like the parent's mutation
    assert evaluate_fixture({}, seed=1)["params"] != evaluate_fixture({}, seed=2)["params"]


def test_evaluate_fixture_same_keys_and_deterministic():
    real = evaluate({"a2_Te": 0.6}, DEV, seed=3)
    fix = evaluate_fixture({"a2_Te": 0.6}, seed=3)
    assert set(fix) == set(real) | {"fixture"}
    assert fix["fixture"] is True and fix["emulated"] is True and fix["touch_check"].startswith("fail: fixture")
    again = evaluate_fixture({"a2_Te": 0.6}, seed=3)
    timing = ("elapsed_seconds", "t_start")
    assert {k: v for k, v in fix.items() if k not in timing} == {k: v for k, v in again.items() if k not in timing}
    json.dumps(fix, allow_nan=False)


def test_fixture_and_package_import_load_no_torch():
    code = ("import sys; import fusion_miniapp.sim as s; assert 'torch' not in sys.modules; "
            "s.evaluate_fixture({'a1_ne': 3.0}); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_evaluate_pickles_by_name_and_runs_in_spawned_pool():
    assert pickle.loads(pickle.dumps(evaluate)) is evaluate
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
        r = pool.submit(evaluate, {"gamma_Te": 0.1}, DEV, "spawned", 5).result(timeout=120)
        f = pool.submit(evaluate_fixture, {"gamma_Te": 0.1}, None, "spawned", 5).result(timeout=60)
    assert r["touch_check"] == "pass" and r["device"].startswith(DEVICE)
    t = r["touches"]                                            # a cold start: a fresh process
    assert (t["context_build"], t["library_import"], t["checkpoint_load"], t["model_to_device"]) == (1, 4, 3, 3)
    assert f["fixture"] is True


# --- the inner-loop trace (trace.py; monitor knobs): added instrumentation, not in IGNOS ---------------------

TIMING = ("elapsed_seconds", "t_start", "worker_peak_rss_mb")
HUNDREDK_SHAPE = {"rollout_steps": 2, "puffs": 3, "fixed_point_cap": 15}


def _same_but_trace(on: dict, off: dict) -> None:
    """on is off plus its trace: every other field byte-identical, in the same order."""
    assert "trace" not in off and set(on) == set(off) | {"trace"}
    strip = lambda r: {k: v for k, v in r.items() if k not in TIMING + ("trace",)}   # noqa: E731
    assert json.dumps(strip(on)) == json.dumps(strip(off))


def test_capture_changes_nothing_but_the_trace():
    """Real evaluations in a warm process, capture off and on, the same candidates: the same record (touch counts
    included), plus the trace; None (as the LangGraph and Academy layers call evaluate) records none."""
    from fusion_miniapp import trace
    for knobs in (DEV, {**DEV, **HUNDREDK_SHAPE, "nodes_per_plane": 400}):
        evaluate({}, knobs)                                     # warm: this context is built
        none = evaluate({"a1_Te": 0.75}, knobs, agent_id="c", seed=4)
        off = evaluate({"a1_Te": 0.75}, knobs, agent_id="c", seed=4, monitor={"capture": False})
        on = evaluate({"a1_Te": 0.75}, knobs, agent_id="c", seed=4, monitor={})
        _same_but_trace(on, off)
        _same_but_trace(on, none)
        assert on["touch_check"] == "pass" and on["touches"] == off["touches"]
        t = on["trace"]
        assert (t["kind"], t["synthetic"], t["complete"]) == ("inner_loop", False, True)
        shapes = {k: v.shape for k, v in trace.unpack(t).items()}
        assert shapes == trace.shapes(sim.sim_section(knobs))


def test_a_cold_start_keeps_its_trace():
    """A cold start runs in a fresh process, which gets the monitor knobs: the same record as with capture off,
    plus a complete trace; None records none."""
    knobs = {**DEV, "cold_start_fraction": 1.0}
    off = evaluate({"a1_Te": 0.75}, knobs, agent_id="c", seed=4, monitor={"capture": False})
    on = evaluate({"a1_Te": 0.75}, knobs, agent_id="c", seed=4, monitor={})
    _same_but_trace(on, off)
    assert on["cold_start"] is True and on["touch_check"] == "pass"
    assert (on["trace"]["kind"], on["trace"]["complete"]) == ("inner_loop", True)
    assert "trace" not in evaluate({"a1_Te": 0.75}, knobs, agent_id="c", seed=4)


def test_an_incomplete_recording_says_so():
    """pack checks every hook's count against the knobs' shape: a missing fixed-point iteration, or an XGC step
    too many, makes the trace incomplete, and a cell nothing came for is NaN."""
    from fusion_miniapp import trace
    shape = {"rollout_steps": 1, "puffs": 2, "fixed_point_cap": 2}
    psi, states = np.linspace(0, 1, 10), [np.full((10, 8), i + 1, np.float32) for i in range(4)]
    state = np.arange(3 * 5, dtype=float).reshape(3, 5)
    cells = lambda s: s                                                                     # noqa: E731
    values = lambda c: (c[..., 1, :].max(-1), c[..., 2, :].max(-1), c[..., 0, :].max(-1))   # noqa: E731

    def record(xgc_steps: int, skip: tuple | None) -> dict:
        with trace.recording(shape) as rec:
            for i in range(xgc_steps):
                trace.xgc_step(states[i], states[i + 1], psi)
            trace.handoff(np.zeros((3, 1, 1)), np.ones((3, 1, 1)))
            for p, puff in enumerate((1e21, 2e21)):
                trace.puff(puff)
                for it in range(2):
                    if (p, it) != skip:
                        trace.fixed_point_step(0.5 / (it + 1), state, cells, values)
            trace.summary({"scan": [{"puff": 1e21}, {"puff": 2e21}], "timings": {"step3_seconds": 0.1}})
        return trace.pack(rec)
    whole = record(2, None)
    assert whole["complete"] and np.isfinite(trace.unpack(whole)["fixed_point"]).all()
    assert list(trace.unpack(whole)["fixed_point"][1, 1]) == [0.25, 9, 14, 4]
    missing = record(2, (1, 1))
    assert not missing["complete"] and np.isnan(trace.unpack(missing)["fixed_point"][1, 1]).all()
    assert trace.unpack(missing)["fixed_point"][1, 0, 0] == 0.5
    extra = record(3, None)
    assert not extra["complete"] and trace.unpack(extra)["xgc"].shape == (2, 4)
    assert trace.unpack(whole)["xgc"].tolist() == trace.unpack(extra)["xgc"].tolist()


def test_decide_refuses_a_result_that_holds_its_trace():
    """The outer loop never sees a trace: decide.collect refuses a result that still holds one (the AUTONMI
    Evaluator takes them out first)."""
    from fusion_miniapp import decide
    population, result = [{"a1_Te": 0.5}], {"candidate_idx": 0, "params": {"a1_Te": 0.5}}
    assert decide.collect(population, [result]) == [result]
    with pytest.raises(RuntimeError, match="inner-loop trace"):
        decide.collect(population, [{**result, "trace": {"kind": "inner_loop"}}])


def test_real_series_end_at_the_records_values():
    """Each puff's fixed-point series ends at its own scan row (te_wall, ti_wall, q_peak_proxy, residual), and the
    selected puff's at the record's values; every fixed point runs to the cap; the XGC rows and the hand-off ranges
    are there (float32 throughout)."""
    from fusion_miniapp import trace
    knobs = {**DEV, "puffs": 3, "fixed_point_cap": 4, "rollout_steps": 3}
    r = evaluate({"a1_Te": 0.6, "a2_Te": 0.8}, knobs, seed=7, monitor={})
    a, f32 = trace.unpack(r["trace"]), np.float32
    scan, fp = a["scan"], a["fixed_point"]
    assert fp.shape == (3, 4, 4) and scan.shape == (3, 10) and a["xgc"].shape == (4, 4)
    for i in range(3):
        assert list(fp[i, -1]) == [scan[i, 8], scan[i, 1], scan[i, 2], scan[i, 6]]
    assert list(scan[:, 0]) == list(np.linspace(0.5e21, 4.0e21, 3).astype(f32))
    assert list(scan[:, 7]) == [4, 4, 4] and list(scan[:, 9]) == [0, 0, 0]   # the cap, never converged (FP_TOL 0)
    (sel,) = np.flatnonzero(scan[:, 0] == f32(r["puff_best"]))
    assert list(fp[sel, -1, 1:]) == [f32(r[k]) for k in ("te_wall_eV", "ti_wall_eV", "q_peak_proxy")]
    assert list(scan[sel, 1:7]) == [f32(r[k]) for k in ("te_wall_eV", "ti_wall_eV", "te_ped_eV", "ne_ped_m3",
                                                         "p_ped_Pa", "q_peak_proxy")]
    assert np.isfinite(a["xgc"]).all() and (a["xgc"][:, 0] > 0).all()
    assert (a["handoff"][:3] < a["handoff"][3:]).all() and (a["timings_s"] >= 0).all()
    assert np.isfinite(fp).all() and (np.diff(fp[:, :, 0], axis=1) != 0).all()   # a residual per iteration


def test_the_trace_survives_the_parents_json_value_json_and_pickle():
    """About 1.4 KB at the hundredk shape (232 values), in strings and a bool: the parent's _json_value, JSON and
    pickle give it back unchanged."""
    from fusion_miniapp import trace
    evo, fil = sim._parent()
    for r in (evaluate({}, DEV, monitor={}), evaluate_fixture({}, HUNDREDK_SHAPE, monitor={})):
        t = r["trace"]
        assert fil._json_value(t) == t and json.loads(json.dumps(t)) == t and pickle.loads(pickle.dumps(t)) == t
    assert sum(v.size for v in trace.unpack(t).values()) == 232 and len(json.dumps(t)) < 1500


@pytest.mark.parametrize("how", ["smooth", "hash"])
def test_fixture_fields_and_draws_are_unchanged_by_its_trace(how):
    """The fixture's record with a trace is its record without one, byte for byte (its five draws untouched), plus
    a synthetic trace that says so; the same parameters give the same trace."""
    from fusion_miniapp import trace
    for seed in range(5):
        off = evaluate_fixture({}, HUNDREDK_SHAPE, seed=seed)
        on = evaluate_fixture({}, HUNDREDK_SHAPE, seed=seed, monitor={"fixture_trace": how})
        _same_but_trace(on, off)
        _same_but_trace(on, evaluate_fixture({}, HUNDREDK_SHAPE, seed=seed, monitor={"capture": False}))
        assert (on["trace"]["kind"], on["trace"]["synthetic"]) == (f"synthetic_{how}", True)
        again = evaluate_fixture({}, HUNDREDK_SHAPE, seed=seed, monitor={"fixture_trace": how})
        assert on["trace"] == again["trace"]
        shapes = {k: v.shape for k, v in trace.unpack(on["trace"]).items()}
        assert shapes == trace.shapes(sim.sim_section(HUNDREDK_SHAPE))


def test_hash_fixture_trace_ends_at_the_records_values():
    from fusion_miniapp import trace
    f32 = np.float32
    for seed in range(5):
        r = evaluate_fixture({}, HUNDREDK_SHAPE, seed=seed, monitor={"fixture_trace": "hash"})
        a = trace.unpack(r["trace"])
        (sel,) = np.flatnonzero(a["scan"][:, 0] == f32(r["puff_best"]))
        wall = [f32(r[k]) for k in ("te_wall_eV", "ti_wall_eV", "q_peak_proxy")]
        assert list(a["fixed_point"][sel, -1, 1:]) == wall
        assert list(a["scan"][sel, 1:7]) == [f32(r[k]) for k in ("te_wall_eV", "ti_wall_eV", "te_ped_eV", "ne_ped_m3",
                                                                  "p_ped_Pa", "q_peak_proxy")]


def _gaps(t1: dict, t2: dict) -> np.ndarray:
    """The relative gap |a - b| / (|a| + |b| + eps) at each step of the inner loop (the XGC rows, then each puff's
    iterations), the largest over the step's values (the Monitor's measure, plan section 3.3)."""
    from fusion_miniapp import trace
    a, b = trace.unpack(t1), trace.unpack(t2)
    steps = lambda x: np.concatenate([x["xgc"], x["fixed_point"].reshape(-1, 4)]).astype(float)   # noqa: E731
    sa, sb = steps(a), steps(b)
    return (np.abs(sa - sb) / (np.abs(sa) + np.abs(sb) + 1e-30)).max(axis=1)


@pytest.mark.parametrize("shape", [HUNDREDK_SHAPE, {}])   # hundredk's and smoke's
def test_smooth_fixture_trace_has_its_planted_split(shape):
    """Two candidates either side of a1_Te's mid-range (normalised 0.47 and 0.53, everything else equal) agree to
    within 1% at every step before fixed-point iteration cap // 2 of the first puff, and part there (more than 5%);
    two on one side (0.60, 0.70) never part; the trace is smooth: a small step in a1_Te makes a small change."""
    from fusion_miniapp import decide, trace
    lo, hi = decide.ranges()[trace.SPLIT_PARAM]
    s = sim.sim_section(shape)
    k, c = s["rollout_steps"], s["fixed_point_cap"]

    def at(x: float) -> dict:
        return evaluate_fixture({trace.SPLIT_PARAM: lo + x * (hi - lo)}, shape, seed=11, monitor={})["trace"]
    across = _gaps(at(0.47), at(0.53))
    split = k + 1 + c // 2                                      # the XGC rows, then puff 0's iterations
    assert (across[:split] < 0.01).all() and split >= 2, across
    assert across[split] > 0.05 and np.flatnonzero(across > 0.05)[0] == split, across
    assert across[-1] > 0.05                                    # parted to the end: the outcomes differ
    assert (_gaps(at(0.60), at(0.70)) < 0.01).all()
    assert (_gaps(at(0.30), at(0.301)) < 1e-3).all()
