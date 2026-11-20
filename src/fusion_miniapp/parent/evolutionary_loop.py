#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
Evolutionary optimisation loop for the core-edge plasma pipeline.

Wraps the existing pipeline (core_edge_optimization.py) in an evolutionary
algorithm that maintains a population of profile-generator parameterisations,
evaluates each through the inference pipeline, selects the best-performing
ones that satisfy physical constraints, mutates them, and iterates.

Candidates within each generation are evaluated **in parallel** across
multiple GPUs using multiprocessing (default: 4 workers for 4 A100s).

Usage:
  python evolutionary_loop.py -N 10 -G 20 --n-gpus 4
  python evolutionary_loop.py -N 8 -G 5 --inner-loop legacy --steps=-1,0,1,2,3,4,425,45,5,525,6,7
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths — mirror core_edge_optimization.py
# ---------------------------------------------------------------------------
DEMO_ROOT = Path(__file__).resolve().parent
STEPS_DIR = DEMO_ROOT / "steps"
STEP0_DIR = STEPS_DIR / "step0"
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(DEMO_ROOT.parent)))

# STEPS_DIR is needed for `import puff_optimizer` (campaign-consistent
# pedestal/wall metrics); it self-registers the rest of its own path needs.
for _d in (str(STEP0_DIR), str(STEPS_DIR), str(DEMO_ROOT)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from profile_generator import (
    PARAM_RANGES_SNYDER_H,
    PARAM_RANGES_SNYDER_L,
    PARAM_RANGES_THREE_REGION_H,
    PARAM_RANGES_THREE_REGION_L,
    generate_profile,
    sample_params_lhs,
)

# ---------------------------------------------------------------------------
# SOLPS output → core Te metric
# ---------------------------------------------------------------------------
SOLPS_SCALE2EV = 6.241509074460763e18  # J → eV


def extract_xgc_e_T_perp_metric(xgc_output: dict) -> float:
    """e_T_perp at node 0 from XGC step-4 output.

    Feature layout: [e_den, e_T_perp, e_T_para, e_u_para,
                     i_T_perp, i_T_para, i_u_para, dpot]

    Returns 0.0 if *xgc_output* is None or has an unexpected shape.
    """
    if xgc_output is None:
        return 0.0
    raw = xgc_output.get("output") if isinstance(xgc_output, dict) else xgc_output
    if raw is None:
        return 0.0
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0.0
    return float(arr[0, 1])


# Keys produced by extract_pedestal_wall_metrics, in CSV column order.
PEDESTAL_WALL_KEYS = (
    "te_ped_eV",
    "ne_ped_m3",
    "p_ped_Pa",
    "q_peak_proxy",
    "te_wall_div_eV",
    "ti_wall_div_eV",
)

_SOLPS_GEOM_CACHE: dict = {}


def _final_solps_state(result: dict) -> Optional[np.ndarray]:
    """Physical (3, 38, 98) [ne, Te_eV, Ti_eV] from the last SOLPS state.

    Prefers the final AR-SOLPS (step 525) iterate, then the step-5 output on
    disk, then the in-memory step-5 tensor. Returns None when none is available.
    """
    import torch

    out_dir = result.get("out_dir")
    candidates: List[Path] = []
    if out_dir:
        od = Path(out_dir)
        ar_base = od.parent / "ar_solps"
        if ar_base.is_dir():
            # ar_iter_{idx:03d}_{ts} — zero-padded, so the last name is the last step.
            iters = sorted(p for p in ar_base.glob("ar_iter_*") if p.is_dir())
            if iters:
                candidates.append(iters[-1] / "out" / "solps_output.pt")
        candidates.append(od / "solps_output.pt")

    raw = None
    for pt in candidates:
        if pt.exists():
            raw = torch.load(pt, map_location="cpu")
            break
    if raw is None:
        so = result.get("solps_output") or {}
        raw = so.get("output") if isinstance(so, dict) else None
    if raw is None:
        return None

    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64).squeeze()
    if arr.shape != (3, 38, 98):
        return None

    # The model emits Te/Ti in Joules; convert when the magnitudes say so.
    if np.nanmax(np.abs(arr[1])) < 1e-14:
        arr = arr.copy()
        arr[1] *= SOLPS_SCALE2EV
        arr[2] *= SOLPS_SCALE2EV
    return arr


def extract_pedestal_wall_metrics(result: dict) -> Dict[str, float]:
    """Campaign-consistent pedestal-top and divertor-target metrics.

    Delegates to steps/puff_optimizer.wall_pedestal_metrics — the same function
    behind the Te_ped vs Te_wall Pareto plane in figures/progress_5 — so these
    numbers are directly comparable with it. Te_ped is SOLPS row 0 at the outer
    midplane (psi_N ~ 0.891); the wall values are the max over SOL rows 19:38 at
    the inner/outer target columns.

    Reported as NaN, never 0.0, when the SOLPS state cannot be read: a zero here
    would be indistinguishable from a genuinely cold wall.
    """
    blank = {k: float("nan") for k in PEDESTAL_WALL_KEYS}
    try:
        import puff_optimizer as po

        state = _final_solps_state(result)
        if state is None:
            return blank
        if "geom" not in _SOLPS_GEOM_CACHE:
            _SOLPS_GEOM_CACHE["geom"] = po.load_geometry()
        m = po.wall_pedestal_metrics(state, _SOLPS_GEOM_CACHE["geom"])
    except Exception:
        return blank

    return {
        "te_ped_eV": float(m["te_ped_eV"]),
        "ne_ped_m3": float(m["ne_ped_m3"]),
        "p_ped_Pa": float(m["p_ped_Pa"]),
        "q_peak_proxy": float(m["q_peak_proxy"]),
        # Named _div to keep them distinct from the pipeline's te_wall_eV, which
        # is a whole-domain wall max rather than a divertor-target max.
        "te_wall_div_eV": float(m["te_wall_eV"]),
        "ti_wall_div_eV": float(m["ti_wall_eV"]),
    }


# ---------------------------------------------------------------------------
# Parameter-range helpers
# ---------------------------------------------------------------------------
_RANGE_MAP = {
    ("three_region", "H"): PARAM_RANGES_THREE_REGION_H,
    ("three_region", "L"): PARAM_RANGES_THREE_REGION_L,
    ("snyder", "H"): PARAM_RANGES_SNYDER_H,
    ("snyder", "L"): PARAM_RANGES_SNYDER_L,
}

MUTABLE_PARAMS = {
    "three_region": ["a1_ne", "gamma_ne", "Delta", "a2_ne", "a1_Te", "gamma_Te", "a2_Te"],
    "snyder": ["ne_sep", "a0_ne", "a1_ne", "Te_sep", "a0_Te", "a1_Te", "Delta"],
}

# Pedestal width bounds enforced on every candidate. The generator's own H-mode
# range is 0.04-0.10; the campaign widened the floor to 0.03 and clips on every
# draw and mutation to keep widths inside the surrogate's validated range.
DELTA_CLIP = (0.03, 0.10)

# Default step list for the explicit legacy pipeline: 425 = AR XGC, 525 = AR
# SOLPS. Step 35 is deliberately absent because core_edge_optimization rejects
# requests containing both step 35 and the XGC-to-SOLPS handoff step 45.
# Ordering guards live in core_edge_optimization.run_core_edge_optimization:
#   425 needs 3; 45 needs 4 or 425; 525 needs 35/45/425; 575 needs 525.
LEGACY_DEFAULT_STEP_IDS = [-1, 0, 1, 2, 3, 4, 425, 45, 5, 525, 6, 7]
# Backwards-compatible name used by the Academy driver and older scripts.
DEFAULT_STEP_IDS = LEGACY_DEFAULT_STEP_IDS

DEFAULT_INNER_LOOP = "validated"
INNER_LOOP_CHOICES = ("validated", "legacy")
DEFAULT_CONDITIONED_CKPT = (
    "/global/cfs/projectdirs/amsc007/models/SOLPS-MATEY/finetune_puff/"
    "condonly01/training_checkpoints/best_ckpt.tar"
)
DEFAULT_CONDITIONED_YAML = str(DEMO_ROOT / "config" / "finetune_puff.yaml")
DEFAULT_CONDITIONED_CONFIG_NAME = "finetune_puff"
DEFAULT_HELIOS_CKPT = (
    "/global/cfs/projectdirs/amsc007/models/HELIOS/outputs/"
    "BES-02-FLOW-DiT/2026-01-18/22-52-17/checkpoints"
)
DEFAULT_MATEY_MODEL_DIR = (
    "/global/cfs/projectdirs/amsc007/zhan1668/MATEY/models/"
    "Dev_Fusion_Demo_March2026_Final/demo_nbatchsloc100/"
)
VALIDATED_METHOD = {
    "name": "validated_campaign3",
    "profile_mode": "raw_campaign",
    "delta_bounds": [0.03, 0.10],
    "outer_objective": "max_final_ar_xgc_node0_e_T_perp",
    "no_feasible_policy": "normalised_core_wall_rank_blend",
    "xgc_ar_steps": 2,
    "psi_anchor_core": 0.85,
    "anchor_blend": 0.05,
    "anchor_channels": [0, 1, 2, 4, 5],
    "density_mode": "prescribed_seed",
    "step45_temperature_formula": "(2*T_perp+T_para)/3",
    "solps_alpha": 0.15,
    "solps_tolerance": 3e-3,
    "solps_cap": 300,
    "puffs_m3_s": [float(v) for v in np.linspace(0.5e21, 4.0e21, 6)],
    "wall_limit_eV": 100.0,
    "pedestal_floor_eV": 250.0,
    "elm_gate": False,
}

# Weight of the wall-temperature term when no candidate satisfies the wall
# constraint and selection falls back to ranking the infeasible population
# (see select_survivors). 0.0 ranks purely by core temperature, 1.0 purely by
# wall temperature.
#
# Low values are not merely weak, they are counterproductive: core and wall
# temperature correlate at +0.69 across the 1334 logged evaluations, so ranking
# mostly by core temperature actively selects for hotter walls. Replaying the
# loop against a surrogate fitted to those evaluations (held-out R^2 0.99),
# N=16 over 20 generations, best wall temperature reached:
#
#   w=0.0  636 eV    w=0.5  354 eV    w=0.8  211 eV
#   w=0.2  650 eV    w=0.6  218 eV    w=1.0  208 eV
#   w=0.4  533 eV    w=0.7  215 eV
#
# The knee sits between 0.5 and 0.6. The default takes some margin past it and
# keeps a little core-temperature signal, at a cost of ~7 eV against w=1.0.
DEFAULT_WALL_WEIGHT = 0.7


def _sha256_file(path: str) -> str:
    """Return a stable fingerprint for a model or source file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_source_sha256() -> str:
    """Fingerprint the branch-local files that define production coupling."""
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        DEMO_ROOT / "core_edge_optimization.py",
        STEPS_DIR / "final_inner_loop.py",
        STEPS_DIR / "puff_optimizer.py",
        STEPS_DIR / "run_matey_inference_1step.py",
        STEP0_DIR / "profile_generator.py",
    ):
        digest.update(str(path.relative_to(DEMO_ROOT)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_provenance() -> Dict[str, Any]:
    """Return the checked-out revision plus whether tracked files differ."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DEMO_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=DEMO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=DEMO_ROOT,
            text=True,
        ).strip()
        return {"head": head, "branch": branch, "tracked_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "branch": None, "tracked_dirty": None}


def resolve_inner_loop_settings(
    *,
    inner_loop: str,
    step_ids: Optional[List[int]],
    n_ar_steps: Optional[int],
    conditioned_ckpt: Optional[str],
    conditioned_yaml: Optional[str],
    conditioned_config_name: str,
    model: str,
    kind: str,
) -> Dict[str, Any]:
    """Validate mode-specific options and return resolved worker settings."""
    mode = inner_loop.lower()
    if mode not in INNER_LOOP_CHOICES:
        raise ValueError(f"inner_loop must be one of {INNER_LOOP_CHOICES}, got {inner_loop!r}")

    if mode == "validated":
        if model.lower() != "three_region" or kind.upper() != "H":
            raise ValueError("validated inner loop requires model=three_region and kind=H")
        if step_ids is not None:
            raise ValueError("--steps is legacy-only; validated mode uses the fixed campaign chain")
        if n_ar_steps not in (None, 2):
            raise ValueError("validated inner loop fixes XGC AR at 2 steps")

        ckpt = str(Path(conditioned_ckpt or DEFAULT_CONDITIONED_CKPT).expanduser().resolve())
        yaml_path = str(Path(conditioned_yaml or DEFAULT_CONDITIONED_YAML).expanduser().resolve())
        for label, path in (("conditioned checkpoint", ckpt), ("conditioned YAML", yaml_path)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        return {
            "inner_loop": mode,
            "step_ids": None,
            "n_ar_steps": 2,
            "conditioned_ckpt": ckpt,
            "conditioned_yaml": yaml_path,
            "conditioned_config_name": conditioned_config_name,
            "profile_mode": "raw_campaign",
        }

    resolved_steps = list(LEGACY_DEFAULT_STEP_IDS if step_ids is None else step_ids)
    if 35 in resolved_steps and 45 in resolved_steps:
        raise ValueError("legacy step list cannot contain both mutually exclusive steps 35 and 45")
    return {
        "inner_loop": mode,
        "step_ids": resolved_steps,
        "n_ar_steps": 10 if n_ar_steps is None else int(n_ar_steps),
        "conditioned_ckpt": None,
        "conditioned_yaml": None,
        "conditioned_config_name": None,
        "profile_mode": "pipeline_shaped",
    }


def get_param_ranges(model: str, kind: str) -> dict:
    return _RANGE_MAP[(model.lower(), kind.upper())]


def resolve_profile_shaping(constraints: dict) -> Dict[str, Any]:
    """Step-0 profile shaping pulled out of the constraints file.

    Resolved once per run and shipped to every worker so candidates are built in
    the same regime the pipeline uses (pedestal cap, separatrix anchor, ordered
    core/join/pedestal/separatrix anchors) instead of raw generator output.
    Defaults mirror core_edge_optimization.run_core_edge_optimization.

    Shared by run_evolutionary_loop and evo_agents.build_run_context so the two
    drivers cannot drift apart on the worker contract.
    """
    def _keV(key):
        v = constraints.get(key)
        return None if v is None else float(v) / 1e3

    return {
        "T_sep_keV": _keV("target_separatrix_temperature_eV"),
        "T_ped_max_keV": _keV("max_pedestal_temperature_eV"),
        "pedestal_psi_min": float(constraints.get("pedestal_region_psi_min", 0.88)),
        "pedestal_cap_transition_psi": float(
            constraints.get("pedestal_cap_transition_psi", 0.03)
        ),
        "separatrix_anchor_width_psi": float(
            constraints.get("separatrix_anchor_width_psi", 0.04)
        ),
        "config_profile_params": dict(constraints.get("profile_params") or {}),
    }


def clip_delta(
    params: dict,
    bounds: Tuple[float, float] = DELTA_CLIP,
) -> dict:
    """Clip the pedestal width in place to explicit *bounds*."""
    if "Delta" in params:
        lo, hi = bounds
        params["Delta"] = float(np.clip(params["Delta"], lo, hi))
    return params


def merge_profile_params(
    candidate_params: dict, config_params: dict, model: str
) -> dict:
    """Combine a candidate with config.yaml's ``profile_params`` overrides.

    The two disagree about ownership: config.yaml pins Delta/a1_Te/a2_Te, which are
    also parameters the EA mutates. Resolution — the EA owns everything in
    MUTABLE_PARAMS, and config.yaml owns the rest. That keeps the frozen shape
    parameters (a3, a4, core_linear_frac_*, bcore_*) at their experimental-like
    values from shot 174310 @ 3500 ms instead of leaving them wherever LHS
    happened to drop them, while still letting the search vary its own knobs.
    """
    merged = copy.deepcopy(candidate_params)
    mutable = set(MUTABLE_PARAMS.get(model.lower(), []))
    for k, v in config_params.items():
        if k not in mutable:
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Population initialisation (LHS)
# ---------------------------------------------------------------------------
def initialize_population(
    n: int,
    model: str,
    kind: str,
    seed: int = 42,
    delta_bounds: Optional[Tuple[float, float]] = None,
) -> List[Dict]:
    population = sample_params_lhs(model, kind, n_samples=n, seed=seed)
    if delta_bounds is not None and model.lower() == "three_region" and kind.upper() == "H":
        # profile_generator's historical H prior starts Delta at 0.04. The
        # validated campaign contract widens it to 0.03 while retaining the LHS
        # quantiles rather than replacing them with independent random draws.
        source_lo, source_hi = PARAM_RANGES_THREE_REGION_H["Delta"]
        target_lo, target_hi = delta_bounds
        for params in population:
            quantile = (float(params["Delta"]) - source_lo) / (source_hi - source_lo)
            params["Delta"] = target_lo + quantile * (target_hi - target_lo)
        population = [clip_delta(params, delta_bounds) for params in population]
    return population


_FINAL_CONTEXT_CACHE: Dict[Tuple[str, str, str, str], Any] = {}


def _evaluate_validated_candidate(
    params: dict,
    gpu_id: Optional[int],
    candidate_idx: int,
    kw: dict,
) -> dict:
    """Evaluate one raw campaign profile through the validated inner loop."""
    from final_inner_loop import (  # imported only after CUDA visibility is pinned
        build_final_innerloop_context,
        evaluate_final_innerloop,
    )

    profile_full = generate_profile(
        psi=None,
        model=kw["model"],
        kind=kw["kind"],
        params=copy.deepcopy(params),
    )
    profile = {
        "psi": np.asarray(profile_full["psi"], dtype=float),
        "ne": np.asarray(profile_full["ne"], dtype=float),
        "Te": np.asarray(profile_full["Te"], dtype=float),
    }

    cache_key = (
        str(kw["conditioned_ckpt"]),
        str(kw["conditioned_yaml"]),
        str(kw["conditioned_config_name"]),
        str(kw["matey_model_dir"]),
    )
    context = _FINAL_CONTEXT_CACHE.get(cache_key)
    if context is None:
        context = build_final_innerloop_context(
            device="cuda",
            conditioned_ckpt=kw["conditioned_ckpt"],
            conditioned_yaml=kw["conditioned_yaml"],
            conditioned_config_name=kw["conditioned_config_name"],
            matey_model_dir=kw["matey_model_dir"],
        )
        _FINAL_CONTEXT_CACHE[cache_key] = context

    generation = int(kw.get("generation", 0))
    work_dir = (
        Path(kw["evo_run_dir"])
        / f"gen{generation}_candidate{candidate_idx}"
        / "final_innerloop"
    )
    summary = evaluate_final_innerloop(
        profile=profile,
        context=context,
        work_dir=work_dir,
        gpu_id=gpu_id,
    )

    te_core_metric = float(summary["te_core_metric"])
    te_wall_eV = float(summary["te_wall_eV"])
    ti_wall_eV = float(summary["ti_wall_eV"])
    te_ped_eV = float(summary["te_ped_eV"])
    summary_path = summary.get("summary_path", str(work_dir / "summary.json"))
    return {
        "params": params,
        "candidate_idx": candidate_idx,
        "input_te_core_keV": float(profile["Te"][0]),
        "te_core_metric": te_core_metric,
        "te_core_keV": te_core_metric / 1e3,
        "ti_core_keV": float("nan"),
        "te_wall_eV": te_wall_eV,
        "ti_wall_eV": ti_wall_eV,
        "wall_ok": bool(summary["wall_ok"]),
        "pedestal_ok": bool(summary["pedestal_ok"]),
        "elm_ok": None,
        "is_good": bool(summary["is_good"]),
        "te_ped_eV": te_ped_eV,
        "ne_ped_m3": float(summary.get("ne_ped_m3", float("nan"))),
        "p_ped_Pa": float(summary.get("p_ped_Pa", float("nan"))),
        "q_peak_proxy": float(summary.get("q_peak_proxy", float("nan"))),
        "te_wall_div_eV": te_wall_eV,
        "ti_wall_div_eV": ti_wall_eV,
        "puff_best": float(summary["puff_best"]),
        "summary_path": str(summary_path),
        "inner_loop": "validated",
    }


# ---------------------------------------------------------------------------
# Single-candidate worker (runs in a child process)
# ---------------------------------------------------------------------------
def _evaluate_single_candidate(args_tuple: tuple) -> dict:
    """Top-level function so it is pickleable for multiprocessing.

    ``args_tuple`` is ``(params, gpu_id, candidate_idx, shared_kwargs)``.

    Each worker:
      1. Pins itself to one GPU via CUDA_VISIBLE_DEVICES.
      2. Generates a profile from the candidate params.
      3. Monkey-patches ``manufacture_profiles_from_generator`` in
         core_edge_optimization so step 0 uses our profile.
      4. Calls ``run_core_edge_optimization`` for a single pass.
      5. Returns a lightweight metrics dict.
    """
    params, gpu_id, candidate_idx, kw = args_tuple

    # Pin GPU **before** torch is imported. gpu_id is None when the caller has
    # already constrained device visibility for us, in which case overriding it
    # would point the worker at the wrong physical device.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if kw.get("inner_loop", DEFAULT_INNER_LOOP) == "validated":
        return _evaluate_validated_candidate(params, gpu_id, candidate_idx, kw)

    # Imports inside the worker so each process gets its own module state
    import core_edge_optimization as ceo  # noqa: E402

    # Give each worker its own STEP4_DIR to avoid concurrent file conflicts.
    # Scoped under the run directory rather than under steps/ so that concurrent
    # jobs (e.g. several sweep arms sharing this checkout) cannot collide on the
    # same worker index and race on the solps_input_D3D.nc copy below.
    worker_step4_dir = Path(kw["evo_run_dir"]) / "_step4" / f"worker_{candidate_idx}"
    worker_step4_dir.mkdir(parents=True, exist_ok=True)
    (worker_step4_dir / "valid").mkdir(exist_ok=True)
    # Copy the static input file so the loader can find it
    import shutil
    src = ceo.STEP4_DIR / "valid" / "solps_input_D3D.nc"
    dst = worker_step4_dir / "valid" / "solps_input_D3D.nc"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
    ceo.STEP4_DIR = worker_step4_dir

    # Give each worker its own OUT_DIR/VIZ_DIR to avoid viz overwrites.
    # run_core_edge_optimization (use_run_dir=False) derives OUT_DIR from DEMO_OUTPUT_BASE,
    # so patch that to make VIZ_DIR unique per generation+candidate.
    gen = kw.get("generation", 0)
    ceo.DEMO_OUTPUT_BASE = Path(kw["evo_run_dir"]) / f"gen{gen}_candidate{candidate_idx}"

    model = kw["model"]
    kind = kw["kind"]
    n_core_points = kw["n_core_points"]

    # Build the profile through the pipeline's own step 0 rather than calling
    # generate_profile directly. Step 0 applies the ordered core/join/pedestal/
    # separatrix anchors, the pedestal cap and the SOL continuation, which is what
    # puts a candidate in the solps-matey data regime. A bare generate_profile call
    # skips all of it, so candidates would land outside the surrogate's training
    # distribution and their pedestals would not close on the LCFS.
    psi = np.linspace(0.0, 1.05, 256)
    original_fn = ceo.step0_manufacture_profiles_from_generator
    shaping = kw.get("profile_shaping") or {}
    T_core_keV = kw.get("T_core_keV")

    profile_dict = original_fn(
        psi=psi,
        model=model,
        kind=kind,
        seed=None,
        n_points=len(psi),
        T_core_keV=T_core_keV,
        T_sep_keV=shaping.get("T_sep_keV"),
        T_ped_max_keV=shaping.get("T_ped_max_keV"),
        pedestal_psi_min=shaping.get("pedestal_psi_min", 0.88),
        pedestal_cap_transition_psi=shaping.get("pedestal_cap_transition_psi", 0.03),
        separatrix_anchor_width_psi=shaping.get("separatrix_anchor_width_psi", 0.04),
        profile_params=merge_profile_params(
            params, shaping.get("config_profile_params") or {}, model
        ),
    )

    # Monkey-patch step 0 so the pipeline's own step-0 call returns this profile

    def _patched(**_kwargs):
        return profile_dict

    ceo.step0_manufacture_profiles_from_generator = _patched

    try:
        result = ceo.run_core_edge_optimization(
            constraints_path=kw["constraints_path"],
            helios_ckpt_path=kw["helios_ckpt_path"],
            matey_model_dir=kw["matey_model_dir"],
            max_iter=1,
            step_ids=kw["step_ids"],
            device="cuda",
            verbose=kw["verbose"],
            flash=kw["flash"],
            use_run_dir=False,
            use_solps_inference=kw["use_solps_inference"],
            solps_data_dir=kw["solps_data_dir"],
            use_xgc_inference=kw["use_xgc_inference"],
            xgc_data_dir=kw["xgc_data_dir"],
            use_ddp=False,
            profile_source="parametric",
            profile_model=model,
            profile_kind=kind,
            elmimpute=kw["elmimpute"],
            save_viz=kw["save_viz"],
            n_ar_steps=kw.get("n_ar_steps", 10),
        )
    finally:
        ceo.step0_manufacture_profiles_from_generator = original_fn

    # Extract metrics
    xgc_output = result.get("xgc_output")
    solps_output = result.get("solps_output") or {}
    best = result.get("best_result") or {}
    constraints = result.get("constraints") or {}

    te_core_metric = extract_xgc_e_T_perp_metric(xgc_output)
    te_core_keV = float(best.get("Te_core_keV", solps_output.get("Te_core_keV", 0.0)))
    ti_core_keV = float(best.get("Ti_core_keV", solps_output.get("Ti_core_keV", 0.0)))
    te_wall_eV = float(best.get("te_wall", 0.0))
    ti_wall_eV = float(best.get("ti_wall", 0.0))

    # If step 6 didn't populate best_result wall values, check solps_output directly
    if te_wall_eV == 0.0 and "Te_wall" in solps_output:
        te_wall_eV = float(np.max(solps_output["Te_wall"]))
    if ti_wall_eV == 0.0 and "Ti_wall" in solps_output:
        ti_wall_eV = float(np.max(solps_output["Ti_wall"]))

    # Prefer the pipeline's own verdicts. Step 2 runs in the default step list, so
    # hardcoding elm_ok=True here would discard the ELM check and leave half of
    # convergence_check() fed by a constant. Fall back to a local wall check only
    # when the pipeline did not report one.
    wall_ok_calc, _, _ = ceo.check_wall_temperature_constraint(
        te_wall_eV, ti_wall_eV, constraints
    )
    elm_ok = bool(result.get("elm_ok", True))
    wall_ok = bool(result.get("wall_ok", wall_ok_calc))
    is_good = ceo.convergence_check(elm_ok, wall_ok)

    # Logged for analysis only — selection still uses te_core_metric.
    ped_wall = extract_pedestal_wall_metrics(result)

    # Free GPU memory
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return {
        "params": params,
        "candidate_idx": candidate_idx,
        "te_core_metric": te_core_metric,
        "te_core_keV": te_core_keV,
        "ti_core_keV": ti_core_keV,
        "te_wall_eV": te_wall_eV,
        "ti_wall_eV": ti_wall_eV,
        "wall_ok": wall_ok,
        "pedestal_ok": None,
        "elm_ok": elm_ok,
        "is_good": is_good,
        "puff_best": float("nan"),
        "summary_path": "",
        "input_te_core_keV": float(profile_dict["Te"][0]),
        "inner_loop": "legacy",
        **ped_wall,
    }


# ---------------------------------------------------------------------------
# Parallel population evaluation
# ---------------------------------------------------------------------------
def _slurm_node_list() -> List[str]:
    """Hostnames this driver may place candidates on, in SLURM order.

    EVO_NODES (comma-separated hostnames) confines the driver to a slice of the
    allocation, so several arms of a sweep can share one job without competing
    for the same nodes. Unset means the whole allocation.
    """
    explicit = os.environ.get("EVO_NODES")
    if explicit:
        return [h for h in explicit.replace(",", " ").split() if h]

    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        raise RuntimeError(
            "--dispatch srun requires a SLURM allocation, but neither EVO_NODES "
            "nor SLURM_JOB_NODELIST is set"
        )
    out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
    return [h for h in out.split() if h]


def _dispatch_srun(tasks: List[tuple], gpus_per_node: int) -> List[dict]:
    """Run every candidate concurrently, one single-node srun step per candidate.

    Placement is computed here rather than delegated to SLURM's GPU arbitration:
    candidate *i* is pinned to ``node[i // gpus_per_node]`` and to local device
    ``i % gpus_per_node``, which the worker applies through the same
    CUDA_VISIBLE_DEVICES path used by the in-process pool. srun therefore only
    has to place the step on the right node, which keeps this independent of how
    the site configures GPU binding for overlapping steps.

    Steps are launched with --overlap so they can share the allocation
    concurrently. Extra srun flags can be injected via EVO_SRUN_EXTRA without
    editing this file (default: --cpus-per-task=16); if steps come up with no
    visible GPU on a given system, add "--gpus-per-node=<n>" there.
    """
    nodes = _slurm_node_list()
    capacity = len(nodes) * gpus_per_node
    if len(tasks) > capacity:
        raise RuntimeError(
            f"{len(tasks)} candidates but only {capacity} GPU slots available "
            f"({len(nodes)} nodes x {gpus_per_node} GPUs)"
        )

    shared = tasks[0][3]
    gen = shared.get("generation", 0)
    stage = Path(shared["evo_run_dir"]) / "_dispatch" / f"gen{gen}"
    stage.mkdir(parents=True, exist_ok=True)

    # gpu_ss11 nodes are 128 CPUs / 4 GPUs, so one candidate's share is 32 CPUs.
    extra = os.environ.get("EVO_SRUN_EXTRA", "--cpus-per-task=32").split()
    self_path = str(Path(__file__).resolve())

    launched = []
    for i, (params, gpu_id, candidate_idx, kw) in enumerate(tasks):
        task_f = stage / f"task_{candidate_idx}.json"
        res_f = stage / f"result_{candidate_idx}.json"
        log_f = stage / f"worker_{candidate_idx}.log"
        with open(task_f, "w") as f:
            json.dump(
                {"params": params, "gpu_id": gpu_id,
                 "candidate_idx": candidate_idx, "shared_kwargs": kw},
                f,
            )
        cmd = [
            "srun", "--nodes=1", "--ntasks=1", "--overlap",
            f"--nodelist={nodes[i // gpus_per_node]}",
            *extra,
            sys.executable, self_path,
            "--worker-task", str(task_f), "--worker-result", str(res_f),
        ]
        lf = open(log_f, "w")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        launched.append((candidate_idx, res_f, log_f, proc, lf))

    results: List[dict] = []
    failures = []
    for candidate_idx, res_f, log_f, proc, lf in launched:
        rc = proc.wait()
        lf.close()
        if rc != 0 or not res_f.exists():
            failures.append((candidate_idx, rc, log_f))
            continue
        with open(res_f) as f:
            results.append(json.load(f))

    if failures:
        idx, rc, log_f = failures[0]
        try:
            tail = "".join(open(log_f).readlines()[-25:])
        except OSError:
            tail = "(log unavailable)"
        raise RuntimeError(
            f"{len(failures)} of {len(tasks)} candidate steps failed under "
            f"--dispatch srun; first failure was candidate {idx} (rc={rc}).\n"
            f"Tail of {log_f}:\n{tail}"
        )

    results.sort(key=lambda r: r["candidate_idx"])
    return results


def evaluate_population_parallel(
    population: List[Dict],
    n_gpus: int,
    shared_kwargs: dict,
    dispatch: str = "pool",
    gpus_per_node: int = 4,
    gpu_offset: int = 0,
) -> List[dict]:
    """Evaluate all candidates, dispatching batches of *n_gpus* in parallel.

    With ``dispatch="srun"`` the whole generation is instead evaluated at once
    across every node in the allocation, so there is no batching loop and the
    generation costs one candidate-evaluation of wall time regardless of N.

    ``gpu_offset`` shifts the local device indices this driver pins, so that two
    arms sharing one node can occupy disjoint GPUs. It applies to the pool
    backend only; an srun-dispatched driver owns whole nodes and starts at 0.
    """
    results: List[dict] = []
    n = len(population)

    if dispatch == "srun":
        tasks = [
            (params, i % gpus_per_node, i, shared_kwargs)
            for i, params in enumerate(population)
        ]
        return _dispatch_srun(tasks, gpus_per_node)

    for batch_start in range(0, n, n_gpus):
        batch_end = min(batch_start + n_gpus, n)
        batch = population[batch_start:batch_end]

        tasks = [
            (params, gpu_offset + gpu_id, batch_start + i, shared_kwargs)
            for i, (gpu_id, params) in enumerate(zip(range(len(batch)), batch))
        ]

        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx) as executor:
            batch_results = list(executor.map(_evaluate_single_candidate, tasks))

        results.extend(batch_results)

    return results


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def _normalised_ranks(values: List[float], higher_is_better: bool) -> np.ndarray:
    """Map *values* onto [0, 1] by rank: best -> 1.0, worst -> 0.0.

    Rank rather than min-max because the two quantities being combined below
    live on unrelated scales (te_core_metric spans a factor of ~1.5 across a
    generation, wall temperature a factor of ~4) and a raw sum would be
    dominated by whichever happens to have the larger spread.
    """
    n = len(values)
    if n <= 1:
        return np.ones(n)
    arr = np.asarray(values, dtype=float)
    worst = -np.inf if higher_is_better else np.inf
    arr = np.where(np.isfinite(arr), arr, worst)
    order = np.argsort(arr, kind="stable")
    if higher_is_better:
        order = order[::-1]
    u = np.empty(n)
    u[order] = np.linspace(1.0, 0.0, n)
    return u


def _rank_infeasible(
    evaluated: List[dict],
    wall_weight: float,
) -> List[dict]:
    """Order a population in which nothing satisfies the wall constraint.

    Scored on two normalised ranks: core temperature (maximised) and wall
    temperature (minimised), blended by *wall_weight*. Candidates that do meet
    the wall constraint -- reachable only if they failed some other check --
    are pinned to the top of the wall term rather than being rewarded for
    running colder than required, which is the "only if it isn't met" part.

    Ranking infeasible candidates by raw max(Te_wall, Ti_wall) is equivalent to
    ranking them by constraint violation, since Te and Ti share one limit in
    config.yaml and subtracting a constant does not reorder them.
    """
    u_core = _normalised_ranks(
        [e["te_core_metric"] for e in evaluated], higher_is_better=True
    )
    # -1.0 is below any physical temperature, so any wall_ok candidate sorts
    # ahead of every violating one regardless of how far under the limit it is.
    wall_key = [
        -1.0 if e["wall_ok"] else max(e["te_wall_eV"], e["ti_wall_eV"])
        for e in evaluated
    ]
    u_wall = _normalised_ranks(wall_key, higher_is_better=False)

    score = (1.0 - wall_weight) * u_core + wall_weight * u_wall
    order = sorted(range(len(evaluated)), key=lambda i: -score[i])
    return [evaluated[i] for i in order]


def select_survivors(
    evaluated: List[dict],
    rng: np.random.Generator,
    wall_weight: float = DEFAULT_WALL_WEIGHT,
) -> List[dict]:
    """Probabilistic rank-based selection.

    Ranking depends on whether anything cleared the constraints:

    * At least one ``is_good`` candidate -> rank those by te_core_metric and
      ignore the rest, as before.
    * None -> rank the whole population by a blend of core temperature and wall
      temperature (see _rank_infeasible) instead of returning nothing. Returning
      nothing sends repopulate down its full-random-restart path, which throws
      away every evaluation of the generation; on this problem the wall
      constraint has never been met, so that path fired every single generation
      and the loop degenerated into independent random sampling.

    Given the ranking, survival is unchanged:

    * Best (rank 0): always kept.
    * Worst (rank n-1): always discarded.
    * Others: P(keep) = 1 - rank / (n - 1)  (i.e. percentile).
    """
    if not 0.0 <= wall_weight <= 1.0:
        raise ValueError(f"wall_weight must be in [0, 1], got {wall_weight}")

    good = [e for e in evaluated if e["is_good"]]
    if good:
        ranked = sorted(good, key=lambda e: e["te_core_metric"], reverse=True)
    elif evaluated:
        ranked = _rank_infeasible(evaluated, wall_weight)
        best = ranked[0]
        print(f"    No candidate met the constraints — carrying forward the "
              f"{len(ranked)} evaluated, ranked by core/wall blend "
              f"(wall_weight={wall_weight:.2f}); "
              f"best: e_T_perp[0]={best['te_core_metric']:.1f} "
              f"wall={max(best['te_wall_eV'], best['ti_wall_eV']):.1f} eV")
    else:
        return []

    if len(ranked) <= 1:
        return list(ranked)

    n = len(ranked)
    survivors = []

    for rank_idx, candidate in enumerate(ranked):
        if rank_idx == 0:
            survivors.append(candidate)
        elif rank_idx == n - 1:
            pass  # always discard worst
        else:
            percentile = 1.0 - rank_idx / (n - 1)
            if rng.random() < percentile:
                survivors.append(candidate)

    return survivors


def empty_best_result() -> Dict[str, Any]:
    return {
        "te_core_metric": 0.0,
        "params": None,
        "generation": -1,
        "candidate_idx": None,
        "is_good": False,
        "selection_basis": "none",
    }


def update_best_result(
    best_ever: Dict[str, Any],
    evaluated: List[dict],
    survivors: List[dict],
    generation: int,
) -> Dict[str, Any]:
    """Track best feasible core, or the latest fallback winner if none exist."""
    feasible = [candidate for candidate in evaluated if candidate.get("is_good")]
    if feasible:
        choice = max(feasible, key=lambda item: item["te_core_metric"])
        if (
            best_ever.get("selection_basis") == "feasible_core"
            and choice["te_core_metric"] <= best_ever["te_core_metric"]
        ):
            return best_ever
        basis = "feasible_core"
    else:
        if best_ever.get("selection_basis") == "feasible_core" or not survivors:
            return best_ever
        # select_survivors always retains rank zero, so this is the generation's
        # core/wall-blend winner rather than an arbitrary probabilistic survivor.
        choice = survivors[0]
        basis = "no_feasible_core_wall_blend"

    te_ped = choice.get("te_ped_eV")
    te_ped = (
        float(te_ped)
        if te_ped is not None and math.isfinite(float(te_ped))
        else None
    )
    return {
        "te_core_metric": float(choice["te_core_metric"]),
        "params": copy.deepcopy(choice["params"]),
        "generation": int(generation),
        "candidate_idx": int(choice["candidate_idx"]),
        "is_good": bool(choice.get("is_good")),
        "selection_basis": basis,
        "te_wall_eV": float(choice["te_wall_eV"]),
        "ti_wall_eV": float(choice["ti_wall_eV"]),
        "te_ped_eV": te_ped,
    }


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------
def mutate_params(
    parent_params: dict,
    model: str,
    kind: str,
    noise_amplitude: float,
    rng: np.random.Generator,
    delta_bounds: Optional[Tuple[float, float]] = None,
) -> dict:
    """Gaussian perturbation of mutable parameters, clipped to valid ranges."""
    ranges = get_param_ranges(model, kind)
    mutable = MUTABLE_PARAMS[model.lower()]
    child = copy.deepcopy(parent_params)

    for name in mutable:
        if name not in ranges or name not in child:
            continue
        lo, hi = delta_bounds if name == "Delta" and delta_bounds else ranges[name]
        sigma = noise_amplitude * (hi - lo)
        child[name] = float(np.clip(child[name] + rng.normal(0, sigma), lo, hi))

    return clip_delta(child, delta_bounds) if delta_bounds else child


# ---------------------------------------------------------------------------
# Repopulation
# ---------------------------------------------------------------------------
def repopulate(
    survivors: List[dict],
    population_size: int,
    model: str,
    kind: str,
    noise_amplitude: float,
    rng: np.random.Generator,
    generation: int,
    delta_bounds: Optional[Tuple[float, float]] = None,
) -> List[Dict]:
    """Fill population back to *population_size*.

    If no survivors: full random restart via LHS.
    Otherwise: keep survivor params and fill the rest by mutating randomly-
    chosen survivors.
    """
    if not survivors:
        seed = int(rng.integers(0, 2**31)) + generation * 1000
        print("    No survivors — full random restart (LHS)")
        return initialize_population(
            population_size, model, kind, seed, delta_bounds=delta_bounds
        )

    new_pop = [copy.deepcopy(s["params"]) for s in survivors]

    while len(new_pop) < population_size:
        parent = survivors[int(rng.integers(0, len(survivors)))]["params"]
        new_pop.append(
            mutate_params(
                parent, model, kind, noise_amplitude, rng,
                delta_bounds=delta_bounds,
            )
        )

    return new_pop


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _best_finite(evaluated: List[dict], key: str, want_max: bool):
    """Best finite value of *key* over feasible candidates, or None if there are none."""
    vals = [
        e[key] for e in evaluated
        if e.get("is_good") and e.get(key) is not None and math.isfinite(e[key])
    ]
    if not vals:
        return None
    return float(max(vals) if want_max else min(vals))


def log_generation(
    output_dir: Path,
    generation: int,
    evaluated: List[dict],
    survivors: List[dict],
) -> None:
    """Write per-candidate CSV and per-generation JSONL."""
    csv_path = output_dir / "evolution_log.csv"
    jsonl_path = output_dir / "evolution_log.jsonl"

    write_header = not csv_path.exists() or generation == 0
    param_keys = sorted(evaluated[0]["params"].keys()) if evaluated else []

    fieldnames = [
        "generation",
        "candidate_idx",
        "inner_loop",
        "input_te_core_keV",
        "te_core_metric",
        "te_core_keV",
        "ti_core_keV",
        "te_wall_eV",
        "ti_wall_eV",
        "wall_ok",
        "pedestal_ok",
        "elm_ok",
        "is_good",
        "puff_best",
        "summary_path",
        "survived",
    ] + list(PEDESTAL_WALL_KEYS) + [f"param_{k}" for k in param_keys]

    survivor_param_ids = {id(s["params"]) for s in survivors}

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for e in evaluated:
            row = {
                "generation": generation,
                "candidate_idx": e["candidate_idx"],
                "inner_loop": e.get("inner_loop", "legacy"),
                "input_te_core_keV": (
                    "" if not math.isfinite(float(e.get("input_te_core_keV", float("nan"))))
                    else f"{float(e['input_te_core_keV']):.6f}"
                ),
                "te_core_metric": f"{e['te_core_metric']:.6f}",
                "te_core_keV": f"{e['te_core_keV']:.6f}",
                "ti_core_keV": (
                    "" if not math.isfinite(float(e.get("ti_core_keV", float("nan"))))
                    else f"{float(e['ti_core_keV']):.6f}"
                ),
                "te_wall_eV": f"{e['te_wall_eV']:.2f}",
                "ti_wall_eV": f"{e['ti_wall_eV']:.2f}",
                "wall_ok": e["wall_ok"],
                "pedestal_ok": "" if e.get("pedestal_ok") is None else e["pedestal_ok"],
                "elm_ok": "" if e.get("elm_ok") is None else e["elm_ok"],
                "is_good": e["is_good"],
                "puff_best": (
                    "" if not math.isfinite(float(e.get("puff_best", float("nan"))))
                    else f"{float(e['puff_best']):.6g}"
                ),
                "summary_path": e.get("summary_path", ""),
                "survived": id(e["params"]) in survivor_param_ids,
            }
            for k in PEDESTAL_WALL_KEYS:
                v = e.get(k)
                # Blank rather than a fabricated 0.0 when extraction failed.
                row[k] = "" if v is None or not math.isfinite(v) else f"{v:.6g}"
            for k in param_keys:
                row[f"param_{k}"] = e["params"].get(k, "")
            writer.writerow(row)

    gen_summary = {
        "generation": generation,
        "n_evaluated": len(evaluated),
        "n_good": sum(1 for e in evaluated if e["is_good"]),
        "n_survivors": len(survivors),
        "best_te_core_metric": max((e["te_core_metric"] for e in evaluated), default=0.0),
        "mean_te_core_metric": float(np.mean([e["te_core_metric"] for e in evaluated])) if evaluated else 0.0,
        # Panel-(h) coordinates over the feasible set: the utopia direction is
        # high pedestal Te, low divertor wall Te.
        "best_te_ped_eV": _best_finite(evaluated, "te_ped_eV", want_max=True),
        "min_te_wall_div_eV": _best_finite(evaluated, "te_wall_div_eV", want_max=False),
        "timestamp": datetime.now().isoformat(),
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(gen_summary) + "\n")


# ===========================================================================
# Main evolutionary loop
# ===========================================================================
def run_evolutionary_loop(
    population_size: int = 4,
    n_generations: int = 5,
    noise_amplitude: float = 0.1,
    n_gpus: int = 4,
    model: str = "three_region",
    kind: str = "H",
    seed: int = 42,
    step_ids: Optional[List[int]] = None,
    n_core_points: int = 5,
    constraints_path: str = "config/config.yaml",
    helios_ckpt_path: str = DEFAULT_HELIOS_CKPT,
    matey_model_dir: str = DEFAULT_MATEY_MODEL_DIR,
    use_solps_inference: bool = True,
    solps_data_dir: Optional[str] = None,
    use_xgc_inference: bool = True,
    xgc_data_dir: Optional[str] = None,
    verbose: int = 0,
    flash: bool = True,
    elmimpute: bool = True,
    save_viz: bool = True,
    dispatch: str = "pool",
    gpus_per_node: int = 4,
    gpu_offset: int = 0,
    run_tag: Optional[str] = None,
    n_ar_steps: Optional[int] = None,
    inner_loop: str = DEFAULT_INNER_LOOP,
    conditioned_ckpt: Optional[str] = None,
    conditioned_yaml: Optional[str] = None,
    conditioned_config_name: str = DEFAULT_CONDITIONED_CONFIG_NAME,
    wall_weight: float = DEFAULT_WALL_WEIGHT,
) -> dict:
    """Run the full evolutionary optimisation."""
    rng = np.random.default_rng(seed)
    if not 0.0 <= wall_weight <= 1.0:
        raise ValueError(f"wall_weight must be in [0, 1], got {wall_weight}")

    settings = resolve_inner_loop_settings(
        inner_loop=inner_loop,
        step_ids=step_ids,
        n_ar_steps=n_ar_steps,
        conditioned_ckpt=conditioned_ckpt,
        conditioned_yaml=conditioned_yaml,
        conditioned_config_name=conditioned_config_name,
        model=model,
        kind=kind,
    )
    step_ids = settings["step_ids"]
    n_ar_steps = settings["n_ar_steps"]

    # Output directory. The tag and job id keep concurrently-started arms of a
    # sweep from landing in the same second-resolution timestamp directory.
    _output_base = Path(os.environ.get("DEMO_SCRATCH", str(DEMO_ROOT)))
    _name = f"evo_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    if run_tag:
        _name += f"_{run_tag}"
    _jobid = os.environ.get("SLURM_JOB_ID")
    if _jobid:
        _name += f"_j{_jobid}"
    evo_run_dir = _output_base / "runs" / _name
    evo_run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve constraints path
    cp = Path(constraints_path)
    if not cp.is_absolute():
        cp = DEMO_ROOT / cp
    constraints_path_abs = str(cp)

    # Load constraints for the T_core_keV target
    import yaml

    with open(constraints_path_abs) as f:
        constraints = yaml.safe_load(f) or {}
    T_core_keV = constraints.get("target_core_temperature_keV")

    profile_shaping = (
        resolve_profile_shaping(constraints)
        if settings["inner_loop"] == "legacy"
        else None
    )

    checkpoint_sha256 = (
        _sha256_file(settings["conditioned_ckpt"])
        if settings["conditioned_ckpt"]
        else None
    )

    # Save run config
    run_config = {
        "population_size": population_size,
        "n_generations": n_generations,
        "noise_amplitude": noise_amplitude,
        # The independent variable of the dual-vs-core ablation. Recorded here so
        # a run's condition is recoverable from the run itself rather than only
        # from its directory name.
        "wall_weight": wall_weight,
        "n_gpus": n_gpus,
        "model": model,
        "kind": kind,
        "seed": seed,
        "inner_loop": settings["inner_loop"],
        "profile_mode": settings["profile_mode"],
        "step_ids": step_ids,
        "n_core_points": n_core_points,
        "T_core_keV": T_core_keV,
        "n_ar_steps": n_ar_steps,
        "dispatch": dispatch,
        "gpus_per_node": gpus_per_node,
        "gpu_offset": gpu_offset,
        "run_tag": run_tag,
        "constraints_path": constraints_path_abs,
        "matey_model_dir": str(Path(matey_model_dir).expanduser().resolve()),
        "helios_ckpt_path": (
            str(Path(helios_ckpt_path).expanduser().resolve())
            if settings["inner_loop"] == "legacy" else None
        ),
        "profile_shaping": profile_shaping,
        "validated_method": (
            VALIDATED_METHOD if settings["inner_loop"] == "validated" else None
        ),
        "conditioned_ckpt": settings["conditioned_ckpt"],
        "conditioned_ckpt_sha256": checkpoint_sha256,
        "conditioned_yaml": settings["conditioned_yaml"],
        "conditioned_yaml_sha256": (
            _sha256_file(settings["conditioned_yaml"])
            if settings["conditioned_yaml"] else None
        ),
        "conditioned_config_name": settings["conditioned_config_name"],
        "runtime_source_sha256": _runtime_source_sha256(),
        "git": _git_provenance(),
    }
    with open(evo_run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Shared kwargs passed to every worker
    shared_kwargs: Dict[str, Any] = {
        "evo_run_dir": str(evo_run_dir),
        "model": model,
        "kind": kind,
        "n_core_points": n_core_points,
        "T_core_keV": T_core_keV,
        "inner_loop": settings["inner_loop"],
        "constraints_path": constraints_path_abs,
        "helios_ckpt_path": helios_ckpt_path,
        "matey_model_dir": matey_model_dir,
        "use_solps_inference": use_solps_inference,
        "solps_data_dir": solps_data_dir,
        "use_xgc_inference": use_xgc_inference,
        "xgc_data_dir": xgc_data_dir,
        "step_ids": step_ids,
        "verbose": verbose,
        "flash": flash,
        "elmimpute": elmimpute,
        "save_viz": save_viz,
        "n_ar_steps": n_ar_steps,
        "profile_shaping": profile_shaping,
        "conditioned_ckpt": settings["conditioned_ckpt"],
        "conditioned_yaml": settings["conditioned_yaml"],
        "conditioned_config_name": settings["conditioned_config_name"],
    }

    # ------------------------------------------------------------------
    # Initialise population
    # ------------------------------------------------------------------
    delta_bounds = DELTA_CLIP if settings["inner_loop"] == "validated" else None
    population = initialize_population(
        population_size, model, kind, seed, delta_bounds=delta_bounds
    )

    best_ever = empty_best_result()

    print(f"Evolutionary loop: {n_generations} generations, population {population_size}, "
          f"{n_gpus} GPUs, inner_loop={settings['inner_loop']}")
    print(f"Output: {evo_run_dir}")

    for gen in range(n_generations):
        gen_start = time.perf_counter()
        print(f"\n{'=' * 60}")
        print(f"GENERATION {gen + 1}/{n_generations}  (population={len(population)})")
        print(f"{'=' * 60}")

        shared_kwargs["generation"] = gen

        # --------------------------------------------------------------
        # Evaluate (4 candidates in parallel)
        # --------------------------------------------------------------
        evaluated = evaluate_population_parallel(
            population, n_gpus, shared_kwargs, dispatch=dispatch,
            gpus_per_node=gpus_per_node, gpu_offset=gpu_offset,
        )

        gen_elapsed = time.perf_counter() - gen_start

        # Print per-candidate summary
        for e in evaluated:
            tag = "GOOD" if e["is_good"] else "FAIL"
            print(f"  [{tag}] candidate {e['candidate_idx']:2d}  "
                  f"e_T_perp[0]={e['te_core_metric']:.6f}  "
                  f"Te_wall={e['te_wall_eV']:.1f} eV  "
                  f"Ti_wall={e['ti_wall_eV']:.1f} eV")

        # --------------------------------------------------------------
        # Select survivors
        # --------------------------------------------------------------
        survivors = select_survivors(evaluated, rng, wall_weight)
        best_ever = update_best_result(best_ever, evaluated, survivors, gen)

        # Log
        log_generation(evo_run_dir, gen, evaluated, survivors)

        n_good = sum(1 for e in evaluated if e["is_good"])
        print(f"\n  Generation {gen + 1}: {len(evaluated)} evaluated, "
              f"{n_good} good, {len(survivors)} survivors  ({gen_elapsed:.1f}s)")
        if survivors:
            best_gen = max(survivors, key=lambda s: s["te_core_metric"])
            print(f"  Best this gen: e_T_perp[0]={best_gen['te_core_metric']:.6f}")
        print(f"  Best ever:     e_T_perp[0]={best_ever['te_core_metric']:.6f} "
              f"(gen {best_ever['generation'] + 1})")

        # --------------------------------------------------------------
        # Repopulate
        # --------------------------------------------------------------
        if gen < n_generations - 1:
            population = repopulate(
                survivors, population_size, model, kind, noise_amplitude, rng, gen,
                delta_bounds=delta_bounds,
            )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("EVOLUTIONARY OPTIMISATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best e_T_perp[0]: {best_ever['te_core_metric']:.6f}")
    print(f"Found in generation: {best_ever['generation'] + 1}")
    if best_ever["params"]:
        print(f"Best params: {json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in best_ever['params'].items()}, indent=2)}")
    print(f"Log directory: {evo_run_dir}")

    with open(evo_run_dir / "best_result.json", "w") as f:
        json.dump(best_ever, f, indent=2, default=str)

    return best_ever


# ===========================================================================
# CLI
# ===========================================================================
def _worker_main() -> None:
    """Evaluate exactly one candidate from a task file and write its result.

    This is the entry point each srun step runs under --dispatch srun.
    """
    wp = argparse.ArgumentParser()
    wp.add_argument("--worker-task", required=True)
    wp.add_argument("--worker-result", required=True)
    wargs, _ = wp.parse_known_args()

    with open(wargs.worker_task) as f:
        t = json.load(f)

    result = _evaluate_single_candidate(
        (t["params"], t["gpu_id"], t["candidate_idx"], t["shared_kwargs"])
    )

    # Write via a temp file so the driver can never read a partial result.
    out = Path(wargs.worker_result)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f)
    tmp.replace(out)


def main():
    if "--worker-task" in sys.argv:
        _worker_main()
        return

    parser = argparse.ArgumentParser(
        description="Evolutionary optimisation for core-edge pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evolutionary parameters
    parser.add_argument("-N", "--population-size", type=int, default=4,
                        help="Number of candidates per generation")
    parser.add_argument("-G", "--generations", type=int, default=5,
                        help="Number of evolutionary generations")
    parser.add_argument("--noise-amplitude", type=float, default=0.1,
                        help="Mutation sigma as fraction of parameter range")
    parser.add_argument("--wall-weight", type=float, default=DEFAULT_WALL_WEIGHT,
                        help="When no candidate meets the wall constraint, how "
                             "strongly to rank survivors by low wall temperature "
                             "vs high core temperature. 1.0 = wall only, "
                             "0.0 = core only. Ignored once anything is is_good")
    parser.add_argument("--n-core-points", type=int, default=5,
                        help="Number of core radial points for Te metric")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPUs / parallel workers")
    parser.add_argument("--model", type=str, default="three_region",
                        choices=["three_region", "snyder"])
    parser.add_argument("--kind", type=str, default="H", choices=["H", "L"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--inner-loop", choices=INNER_LOOP_CHOICES, default=DEFAULT_INNER_LOOP,
        help="validated: campaign-3 anchored XGC + constrained puff scan; "
             "legacy: generic numbered core-edge pipeline",
    )
    parser.add_argument("--conditioned-ckpt", type=str, default=DEFAULT_CONDITIONED_CKPT,
                        help="Conditioned SOLPS checkpoint for validated mode")
    parser.add_argument("--conditioned-yaml", type=str, default=DEFAULT_CONDITIONED_YAML,
                        help="MATEY YAML containing the conditioned model config")
    parser.add_argument("--conditioned-config-name", type=str,
                        default=DEFAULT_CONDITIONED_CONFIG_NAME,
                        help="Top-level conditioned model config in --conditioned-yaml")

    # Generic legacy pipeline only.
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Legacy-only comma-separated pipeline steps; omitted uses the safe "
             "AR default (425 = AR XGC, 525 = AR SOLPS)",
    )
    parser.add_argument(
        "--n-ar-steps", type=int, default=None,
        help="Legacy generic AR count. Validated mode is fixed at two XGC steps",
    )

    # Pass-through pipeline args
    parser.add_argument("--constraints", type=str,
                        default=str(DEMO_ROOT / "config" / "config.yaml"))
    parser.add_argument("--helios-ckpt", type=str,
                        default=DEFAULT_HELIOS_CKPT)
    parser.add_argument("--matey-model-dir", type=str,
                        default=DEFAULT_MATEY_MODEL_DIR)
    parser.add_argument("--use-solps-inference", action="store_true", default=True)
    parser.add_argument("--no-solps-inference", action="store_false", dest="use_solps_inference")
    parser.add_argument("--solps-data-dir", type=str, default=None)
    parser.add_argument("--use-xgc-inference", action="store_true", default=True)
    parser.add_argument("--no-xgc-inference", action="store_false", dest="use_xgc_inference")
    parser.add_argument("--xgc-data-dir", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="count", default=1)
    parser.add_argument("--flash", action="store_true")
    parser.add_argument("--elmimpute", action="store_true", default=True,
                        help="ELM imputation mode: mask BES+FS+ECEI, given TECE+bt+ip+r0+v1-v3")
    parser.add_argument("--no-elmimpute", action="store_false", dest="elmimpute")
    parser.add_argument("--save-viz", action="store_true", default=True,
                        help="Save visualisation outputs to run directory")
    parser.add_argument("--no-save-viz", action="store_false", dest="save_viz")

    # Dispatch backend
    parser.add_argument("--dispatch", type=str, default="pool", choices=["pool", "srun"],
                        help="pool: in-process workers, batches of --n-gpus on one node. "
                             "srun: one srun step per candidate across the whole allocation, "
                             "so a generation costs one evaluation regardless of population")
    parser.add_argument("--gpus-per-node", type=int, default=4,
                        help="GPUs per node, used for srun placement")
    parser.add_argument("--gpu-offset", type=int, default=0,
                        help="First local GPU index this driver may use, so two arms "
                             "sharing a node occupy disjoint devices (pool dispatch only)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Suffix for the run directory name (e.g. the sweep arm)")

    args = parser.parse_args()

    step_ids = (
        [int(s.strip()) for s in args.steps.split(",")]
        if args.steps is not None
        else None
    )

    run_evolutionary_loop(
        population_size=args.population_size,
        n_generations=args.generations,
        noise_amplitude=args.noise_amplitude,
        wall_weight=args.wall_weight,
        n_gpus=args.n_gpus,
        model=args.model,
        kind=args.kind,
        seed=args.seed,
        step_ids=step_ids,
        n_core_points=args.n_core_points,
        constraints_path=args.constraints,
        helios_ckpt_path=args.helios_ckpt,
        matey_model_dir=args.matey_model_dir,
        use_solps_inference=args.use_solps_inference,
        solps_data_dir=args.solps_data_dir,
        use_xgc_inference=args.use_xgc_inference,
        xgc_data_dir=args.xgc_data_dir,
        verbose=args.verbose,
        flash=args.flash,
        elmimpute=args.elmimpute,
        save_viz=args.save_viz,
        dispatch=args.dispatch,
        gpus_per_node=args.gpus_per_node,
        gpu_offset=args.gpu_offset,
        run_tag=args.run_tag,
        n_ar_steps=args.n_ar_steps,
        inner_loop=args.inner_loop,
        conditioned_ckpt=args.conditioned_ckpt,
        conditioned_yaml=args.conditioned_yaml,
        conditioned_config_name=args.conditioned_config_name,
    )


if __name__ == "__main__":
    main()
