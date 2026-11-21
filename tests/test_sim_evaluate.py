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
