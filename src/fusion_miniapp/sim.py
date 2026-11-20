"""The simulation task: `evaluate`, the IGNOS inner loop turned down, and its fixture twin.

`evaluate` calls the parent's own worker `evolutionary_loop._evaluate_single_candidate`, which runs
the parent's `final_inner_loop.evaluate_final_innerloop`; both are unmodified and run over the
stand-ins in standins.py. The parent is configured only through its module attributes and the
injected context. All physics output is synthetic (random-weight stand-in models).

The parent's fixed-point tolerance is set to 0, so every fixed point runs to `fixed_point_cap` and
the cap sets the duration. `sim.cold_start_fraction` is not read here: a cold start is a fresh
worker process, which run.py makes by retiring each worker after round(1 / fraction) tasks.

Importing this module imports neither torch nor the parent files; `evaluate_fixture` never imports torch.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import PARENT_DIR
from .knobs import sim_section
from .touches import KINDS, check, recording, touch

STANDIN_METHOD = "miniapp_standin"          # replaces the parent's "validated_campaign3" label
PROFILE = ("three_region", "H")             # the parent's validated campaign profile family
_LOCK = threading.RLock()                   # the parent's module attributes and the touch recorder are per process
_CONTEXTS: dict[tuple, tuple[Path, dict]] = {}   # per-process context cache: context knobs -> (assets, context)
_CONTEXT_KNOBS = ("nodes_per_plane", "planes", "model_width", "checkpoint_mb", "device")


def _parent():
    """The parent modules, imported by name (parent/ has no __init__.py) and configured once."""
    with _LOCK:
        for d in (PARENT_DIR / "steps" / "step0", PARENT_DIR / "steps", PARENT_DIR):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        import evolutionary_loop as evo
        import final_inner_loop as fil
        for module in (evo, fil):
            if not Path(module.__file__).resolve().is_relative_to(PARENT_DIR.resolve()):
                raise ImportError(f"{module.__name__} resolved outside {PARENT_DIR}: {module.__file__}")
        if not hasattr(fil._clean_pfr, "parent_function"):     # observe the parent's in-place rewrite
            original = fil._clean_pfr

            def _clean_pfr(nc_path, **kwargs):
                original(nc_path, **kwargs)
                touch("handoff_file_rewrite")
            _clean_pfr.parent_function = original
            fil._clean_pfr = _clean_pfr
        fil.METHOD_NAME = STANDIN_METHOD
        return evo, fil


def _params(candidate: dict, seed: int, evo) -> dict:
    """The candidate's mutable parameters, clipped to the parent's ranges as its mutation does;
    every other parameter from the parent's own Latin-hypercube sampler at `seed`."""
    names = evo.MUTABLE_PARAMS[PROFILE[0]]
    unknown = sorted(set(candidate) - set(names))
    if unknown:
        raise ValueError(f"unknown candidate parameter(s) {unknown}; allowed: {names}")
    params = evo.initialize_population(1, *PROFILE, seed=int(seed), delta_bounds=evo.DELTA_CLIP)[0]
    ranges = evo.get_param_ranges(*PROFILE)
    for name, value in candidate.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"candidate parameter {name} is not finite: {value}")
        lo, hi = evo.DELTA_CLIP if name == "Delta" else ranges[name]
        params[name] = min(max(value, lo), hi)
    return params


def _root(sim: dict) -> Path | None:
    if sim["work_root"] is None:
        return None
    root = Path(os.path.expandvars(sim["work_root"])).expanduser()
    if "$" in str(root):
        raise ValueError(f"sim.work_root {sim['work_root']!r} names an unset environment variable")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _context(sim: dict, fil) -> dict:
    """The per-process context, built once per set of context knobs."""
    key = tuple(sim[k] for k in _CONTEXT_KNOBS)
    entry = _CONTEXTS.get(key)
    if entry is not None and entry[0].is_dir():
        return entry[1]
    from . import standins                                      # first use in a process imports torch here
    if entry is None or not entry[0].is_dir():
        assets = Path(tempfile.mkdtemp(prefix="fusion_miniapp_assets_", dir=_root(sim)))
        atexit.register(shutil.rmtree, assets, True)
        standins.make_assets(assets, sim)
    else:
        assets = entry[0]
    context = standins.build_context(assets, sim, fil)
    touch("context_build")
    _CONTEXTS[key] = (assets, context)
    return context


def evaluate(candidate: dict, knobs: dict | None = None, agent_id: str = "", seed: int = 0) -> dict:
    """Evaluate one candidate with the parent's inner loop over the stand-ins; return a JSON-safe record.

    candidate: any subset of the seven mutable profile parameters. knobs: the `sim` section of a knob
    file (None = smoke preset; a partial dict is merged over it). Calls in one process run one at a
    time (a lock); parallelism comes from processes. The worker never sets CUDA_VISIBLE_DEVICES:
    the launcher exposes the GPU and the device rule picks it up.
    """
    sim = sim_section(knobs)
    with _LOCK:
        start = time.perf_counter()
        evo, fil = _parent()
        params = _params(candidate or {}, seed, evo)
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(agent_id))[:40]
        run_dir = Path(tempfile.mkdtemp(prefix=f"eval_{slug}_", dir=_root(sim)))
        try:
            with recording() as counts:
                context = _context(sim, fil)
                context["method"] = fil.METHOD_NAME
                fil.N_XGC_STEPS = sim["rollout_steps"]
                fil.PUFFS = np.linspace(0.5e21, 4.0e21, sim["puffs"])
                fil.FP_CAP = sim["fixed_point_cap"]
                fil.FP_TOL = 0.0                                # never converged early: the cap sets duration
                kw = {"inner_loop": "validated", "model": PROFILE[0], "kind": PROFILE[1], "generation": 0,
                      "evo_run_dir": str(run_dir), "conditioned_ckpt": context["conditioned_checkpoint"],
                      "conditioned_yaml": context["conditioned_yaml"],
                      "conditioned_config_name": context["conditioned_config_name"],
                      "matey_model_dir": context["matey_model_dir"]}
                evo._FINAL_CONTEXT_CACHE.clear()                # the parent must find this context, never build its own
                evo._FINAL_CONTEXT_CACHE[(kw["conditioned_ckpt"], kw["conditioned_yaml"],
                                          kw["conditioned_config_name"], kw["matey_model_dir"])] = context
                record = evo._evaluate_single_candidate((params, None, 0, kw))   # gpu_id None
                touch("result_return")
                touch("artifact_write", sum(p.is_file() for p in run_dir.rglob("*")))
        finally:
            if not sim["keep_artifacts"]:
                shutil.rmtree(run_dir, ignore_errors=True)
        touches = {kind: counts[kind] for kind in KINDS}
        record.update(emulated=True, device=str(context["device"]),
                      elapsed_seconds=time.perf_counter() - start,
                      touches=touches, touch_check=check(touches, sim))
        if not sim["keep_artifacts"]:
            record["summary_path"] = ""                         # the work directory is gone
        return fil._json_value(record)                          # NaN -> None, numpy -> Python


def evaluate_fixture(candidate: dict, knobs: dict | None = None, agent_id: str = "", seed: int = 0) -> dict:
    """The same keys as evaluate, instantly and without torch, plus fixture: true. For large agent-count
    tests. Values are a deterministic function of the parameters, not a simulation."""
    start = time.perf_counter()
    sim = sim_section(knobs)
    evo, fil = _parent()
    params = _params(candidate or {}, seed, evo)
    profile = evo.generate_profile(psi=None, model=PROFILE[0], kind=PROFILE[1], params=dict(params))
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).digest()
    u = np.random.default_rng(int.from_bytes(digest[:8], "little")).uniform(size=5)
    te_core = float(profile["Te"][0]) * 1e3
    te_wall, ti_wall, te_ped = 50 + 1000 * u[0], 50 + 1000 * u[1], 100 + 900 * u[2]
    ne_ped = 1e19 * (1 + 4 * u[3])
    wall_ok = bool(te_wall <= fil.WALL_LIMIT_EV and ti_wall <= fil.WALL_LIMIT_EV)
    pedestal_ok = bool(te_ped >= fil.PEDESTAL_FLOOR_EV)
    record = {
        "params": params, "candidate_idx": 0, "input_te_core_keV": float(profile["Te"][0]),
        "te_core_metric": te_core, "te_core_keV": te_core / 1e3, "ti_core_keV": float("nan"),
        "te_wall_eV": te_wall, "ti_wall_eV": ti_wall, "wall_ok": wall_ok, "pedestal_ok": pedestal_ok,
        "elm_ok": None, "is_good": wall_ok and pedestal_ok, "te_ped_eV": te_ped, "ne_ped_m3": ne_ped,
        "p_ped_Pa": ne_ped * 2 * te_ped * 1.602e-19, "q_peak_proxy": ne_ped * te_wall ** 1.5,
        "te_wall_div_eV": te_wall, "ti_wall_div_eV": ti_wall,
        "puff_best": float(np.linspace(0.5e21, 4.0e21, sim["puffs"])[int(u[4] * sim["puffs"])]),
        "summary_path": "", "inner_loop": "fixture",
        "emulated": True, "fixture": True, "device": "none", "elapsed_seconds": time.perf_counter() - start,
        "touches": {kind: 0 for kind in KINDS}, "touch_check": "fail: fixture, no touches made",
    }
    return fil._json_value(record)
