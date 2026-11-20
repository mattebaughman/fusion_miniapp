#!/usr/bin/env python3
"""Import-safe implementation of the validated production inner loop.

This module deliberately contains no campaign driver and performs no work at
import time.  A worker first pins its GPU, then calls
``build_final_innerloop_context`` once and reuses the returned process-local
context for one or more calls to ``evaluate_final_innerloop``.

The evaluated chain is the final campaign-3 method:

    profile -> step 3 -> two anchored XGC AR steps -> corrected step 4.5
    -> six independent conditioned SOLPS fixed points -> min-wall selection

Large model objects and CUDA tensors remain in the context.  The evaluation
result is JSON-safe and points to candidate-local artifacts on disk.
"""

from __future__ import annotations

import importlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np


_STEPS_DIR = Path(__file__).resolve().parent
_DEMO_ROOT = _STEPS_DIR.parent

METHOD_NAME = "validated_campaign3"
N_XGC_STEPS = 2
PSI_ANCHOR_CORE = 0.85
ANCHOR_BLEND = 0.05
ANCHOR_CHANNELS = (0, 1, 2, 4, 5)
FP_ALPHA = 0.15
FP_TOL = 3.0e-3
FP_CAP = 300
MAIN_CHAMBER_COL_START = 24
MAIN_CHAMBER_COL_STOP = 73
PUFFS = np.linspace(0.5e21, 4.0e21, 6)
WALL_LIMIT_EV = 100.0
PEDESTAL_FLOOR_EV = 250.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_file(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a readable file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable: {path}")
    return path


def _require_dir(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def _import_local_module(name: str, expected_path: Path) -> ModuleType:
    """Import a dependency while refusing a same-named external checkout."""
    for path in (str(_DEMO_ROOT), str(_STEPS_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    module = importlib.import_module(name)
    actual = Path(module.__file__).resolve()
    expected = expected_path.resolve()
    if actual != expected:
        raise ImportError(
            f"{name} resolved outside this checkout: {actual} (expected {expected})"
        )
    return module


def _add_matey_dependency_roots() -> None:
    """Expose the shared MATEY library without importing another demo tree."""
    project_root = Path(
        os.environ.get("PROJECT_ROOT", "/global/cfs/projectdirs/amsc007")
    ).expanduser().resolve()
    # Insert PROJECT_ROOT first so the more specific MATEY source root ends up
    # ahead of it.  These provide third-party/library packages only; all demo
    # workflow modules are still resolved and verified against this checkout.
    for root in (project_root, project_root / "asvillar" / "src" / "MATEY"):
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)


def _json_value(value: Any) -> Any:
    """Recursively convert NumPy/path values to strict JSON primitives."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _profile_arrays(profile: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = {"psi", "ne", "Te"}.difference(profile)
    if missing:
        raise KeyError(f"Profile is missing required fields: {sorted(missing)}")
    arrays = {
        key: np.asarray(profile[key], dtype=np.float64).copy()
        for key in ("psi", "ne", "Te")
    }
    shapes = {key: value.shape for key, value in arrays.items()}
    if any(value.ndim != 1 for value in arrays.values()):
        raise ValueError(f"Profile fields must be one-dimensional, got {shapes}")
    if len({value.size for value in arrays.values()}) != 1:
        raise ValueError(f"Profile fields must have the same length, got {shapes}")
    if arrays["psi"].size < 2:
        raise ValueError("Profile must contain at least two radial points")
    for key, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Profile field {key!r} contains non-finite values")
    return arrays


def _clean_pfr(
    nc_path: Path,
    *,
    jsep: int = 17,
    leftcut: int = 24,
    rightcut: int = 72,
) -> None:
    """Clamp the step-4.5 private-flux interpolation artifact."""
    import netCDF4

    with netCDF4.Dataset(str(nc_path), "a") as dataset:
        for name in ("te2d", "ti2d"):
            values = dataset.variables[name][:]
            for time_index in range(values.shape[0]):
                field = values[time_index]
                pfr_columns = list(range(0, leftcut)) + list(
                    range(rightcut + 1, field.shape[1])
                )
                for column in pfr_columns:
                    field[: jsep + 2, column] = np.minimum(
                        field[: jsep + 2, column], field[jsep + 2, column]
                    )
                values[time_index] = field
            dataset.variables[name][:] = values


def _conditioned_fixed_point(
    initial_state: np.ndarray,
    puff: float,
    context: Mapping[str, Any],
    aux: Mapping[str, Any],
) -> tuple[np.ndarray, int, float, bool]:
    """Run one independent constrained, damped SOLPS fixed point."""
    po = context["puff_optimizer"]
    model = context["conditioned_model"]
    state0 = np.asarray(initial_state, dtype=np.float64)
    state = state0.copy()
    freeze_mask = context["temperature_freeze_mask"]
    density_weights = context["density_weights"]
    columns = context["main_chamber_columns"]
    final_residual = float("nan")
    converged = False

    for iteration in range(1, FP_CAP + 1):
        mapped = po.forward_conditioned(
            model, aux, [state] * po.N_STEPS, float(puff)
        )
        next_state = (1.0 - FP_ALPHA) * state + FP_ALPHA * np.asarray(
            mapped, dtype=np.float64
        )

        # Freeze Te/Ti on closed-flux rows in the main chamber.
        for channel in (1, 2):
            next_state[channel][freeze_mask] = state0[channel][freeze_mask]

        # Prescribe density inside row 21, blend across rows 21/22, and
        # leave rows >=23 free.  PFR and SOL columns remain unconstrained.
        next_state[0][:, columns] = (
            (1.0 - density_weights[:, columns]) * state0[0][:, columns]
            + density_weights[:, columns] * next_state[0][:, columns]
        )

        final_residual = float(po.rel_l2(next_state, state))
        if not np.isfinite(final_residual):
            raise RuntimeError(
                f"SOLPS fixed point produced a non-finite residual at puff {puff:.6e}"
            )
        state = next_state
        if final_residual < FP_TOL:
            converged = True
            break

    return state, iteration, final_residual, converged


def _artifact_paths(work_dir: Path) -> dict[str, Any]:
    return {
        "profile": work_dir / "profile.npz",
        "ar_xgc_dir": work_dir / "ar_xgc",
        "xgc_handoff": work_dir / "xgc_handoff.npz",
        "solps_input_dir": work_dir / "solps_input",
        "solps_input": work_dir / "solps_input" / "valid" / "solps_input_D3D.nc",
        "puff_scan": work_dir / "puff_scan.json",
        "selected_solps_state": work_dir / "selected_solps_state.npz",
        "summary": work_dir / "summary.json",
    }


def build_final_innerloop_context(
    device: Any,
    conditioned_ckpt: str | os.PathLike[str],
    conditioned_yaml: str | os.PathLike[str],
    conditioned_config_name: str,
    matey_model_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the process-local geometry and model context.

    Call this only after the worker has set its CUDA visibility/device.  The
    explicit checkpoint, YAML, config name, and reference model directory are
    authoritative; no campaign or external-clone defaults are consulted.
    """
    ckpt_path = _require_file(conditioned_ckpt, "conditioned checkpoint")
    yaml_path = _require_file(conditioned_yaml, "conditioned model YAML")
    model_dir = _require_dir(matey_model_dir, "MATEY model directory")
    _require_file(model_dir / "hyperparams.yaml", "MATEY hyperparameters")
    _require_file(
        model_dir / "training_checkpoints" / "ckpt.tar", "XGC AR checkpoint"
    )
    if not isinstance(conditioned_config_name, str) or not conditioned_config_name:
        raise ValueError("conditioned_config_name must be a non-empty string")

    # run_matey_inference_1step imports matey.utils at module load time.  Make
    # the production library roots available before importing any local
    # workflow modules that may transitively load it.
    _add_matey_dependency_roots()
    ceo = _import_local_module(
        "core_edge_optimization", _DEMO_ROOT / "core_edge_optimization.py"
    )
    rmi = _import_local_module(
        "run_matey_inference_1step", _STEPS_DIR / "run_matey_inference_1step.py"
    )
    po = _import_local_module("puff_optimizer", _STEPS_DIR / "puff_optimizer.py")

    import torch
    from matplotlib.tri import LinearTriInterpolator
    from scipy.io import loadmat
    from xgc_reader import base as xgc_base

    torch_device = torch.device(device)
    xgc_mesh_dir = _require_dir(ceo._XGC_MESH_FOR_STEP3, "XGC mesh directory")
    solps_geometry = _require_file(ceo.SOLPS_GEOM, "SOLPS geometry")

    # xgc_reader changes the process cwd while loading a case.  Context setup
    # must not leak that global side effect into a long-lived worker.
    original_cwd = Path.cwd()
    try:
        xgc = xgc_base.xgc1(str(xgc_mesh_dir))
        xgc.setup_mesh()
        xgc.load_units()
    finally:
        os.chdir(original_cwd)
    solps_instance = loadmat(
        str(solps_geometry), squeeze_me=True, struct_as_record=False
    )
    geometry = po.load_geometry()
    rc = np.asarray(solps_instance["Geo"].prcen, dtype=np.float64).reshape(38, 98)
    zc = np.asarray(solps_instance["Geo"].pzcen, dtype=np.float64).reshape(38, 98)
    jxa = int(np.argmax(rc[18]))
    nnodes = int(xgc.mesh.nnodes)
    psi_n_nodes = (
        np.asarray(xgc.mesh.psi, dtype=np.float64).flatten()[:nnodes]
        / float(xgc.psix)
    )
    psi_interpolator = LinearTriInterpolator(xgc.mesh.triobj, psi_n_nodes)
    psi_omp = np.asarray(psi_interpolator(rc[:, jxa], zc[:, jxa]), dtype=float)

    temperature_freeze_mask = np.zeros((38, 98), dtype=bool)
    confined_rows = np.flatnonzero(np.isfinite(psi_omp) & (psi_omp < 1.0))
    main_chamber_columns = np.arange(
        MAIN_CHAMBER_COL_START, MAIN_CHAMBER_COL_STOP
    )
    temperature_freeze_mask[np.ix_(confined_rows, main_chamber_columns)] = True

    density_row_weights = np.zeros(38, dtype=np.float64)
    density_row_weights[21] = 0.33
    density_row_weights[22] = 0.67
    density_row_weights[23:] = 1.0
    density_weights = np.tile(density_row_weights[:, None], (1, 98))

    conditioned_model = po.load_conditioned_model(
        str(ckpt_path), str(yaml_path), conditioned_config_name, torch_device
    )
    if not any("inconMLP" in name for name, _ in conditioned_model.named_parameters()):
        raise RuntimeError(
            "The loaded checkpoint/model has no actuator-conditioning parameters "
            "(inconMLP); use an input_control_act-trained checkpoint and config"
        )

    return {
        "method": METHOD_NAME,
        "device": torch_device,
        "conditioned_checkpoint": str(ckpt_path),
        "conditioned_yaml": str(yaml_path),
        "conditioned_config_name": conditioned_config_name,
        "matey_model_dir": str(model_dir),
        "xgc_mesh_dir": str(xgc_mesh_dir),
        "solps_geometry": str(solps_geometry),
        "xgc": xgc,
        "solps_instance": solps_instance,
        "nnodes": nnodes,
        "psi_n_nodes": psi_n_nodes,
        "psi_omp": psi_omp,
        "jxa": jxa,
        "geometry": geometry,
        "temperature_freeze_mask": temperature_freeze_mask,
        "density_weights": density_weights,
        "main_chamber_columns": main_chamber_columns,
        "conditioned_model": conditioned_model,
        "core_edge_optimization": ceo,
        "run_matey_inference": rmi,
        "puff_optimizer": po,
    }


def evaluate_final_innerloop(
    profile: Mapping[str, Any],
    context: Mapping[str, Any],
    work_dir: str | os.PathLike[str],
    gpu_id: int | str | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate with the validated final inner loop.

    The returned mapping contains no arrays or model objects.  All large
    states are written below ``work_dir`` and referenced by absolute path.
    """
    if context.get("method") != METHOD_NAME:
        raise ValueError(
            f"Context method is {context.get('method')!r}; expected {METHOD_NAME!r}"
        )
    ceo = context["core_edge_optimization"]
    rmi = context["run_matey_inference"]
    po = context["puff_optimizer"]

    candidate_dir = Path(work_dir).expanduser().resolve()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(candidate_dir)
    paths["ar_xgc_dir"].mkdir(parents=True, exist_ok=True)
    arrays = _profile_arrays(profile)
    np.savez_compressed(paths["profile"], **arrays)

    started_at = _utc_now()
    start_time = time.perf_counter()
    step_timings: dict[str, float] = {}
    ar_iteration_artifacts: list[str] = []

    try:
        stage_start = time.perf_counter()
        edge_bcs = ceo.step3_construct_full_profiles_and_extract_edge_bcs(
            arrays,
            xgc_instance=context["xgc"],
            shot=174310,
            time_ms=3500,
            gpath=str(ceo.STEP1_DIR),
        )
        seed_state = np.asarray(
            rmi._xgc_node_values_from_edge_bcs(edge_bcs), dtype=np.float32
        )
        if seed_state.ndim != 2 or seed_state.shape[1] < 6:
            raise RuntimeError(f"Unexpected XGC seed-state shape: {seed_state.shape}")
        if seed_state.shape[0] % int(context["nnodes"]) != 0:
            raise RuntimeError(
                "XGC seed node count is not divisible by the configured mesh size: "
                f"{seed_state.shape[0]} vs {context['nnodes']}"
            )
        nplanes = seed_state.shape[0] // int(context["nnodes"])
        psi_n_full = np.tile(context["psi_n_nodes"], nplanes)
        step_timings["step3_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        ar_results = rmi.run_matey_ar_xgc_inference(
            model_dir=context["matey_model_dir"],
            edge_bcs=edge_bcs,
            xgc_case_dir=context["xgc_mesh_dir"],
            n_ar_steps=N_XGC_STEPS,
            output_base_dir=str(paths["ar_xgc_dir"]),
            save_viz=False,
            verbose=False,
            distributed=False,
            psi_n_full=psi_n_full,
            psi_anchor_core=PSI_ANCHOR_CORE,
            anchor_blend=ANCHOR_BLEND,
            anchor_channels=list(ANCHOR_CHANNELS),
            anchor_state=seed_state,
        )
        if len(ar_results) != N_XGC_STEPS:
            raise RuntimeError(
                f"XGC AR returned {len(ar_results)} steps; expected {N_XGC_STEPS}"
            )
        final_xgc = np.asarray(ar_results[-1]["output"], dtype=np.float64).copy()
        if final_xgc.shape != seed_state.shape:
            raise RuntimeError(
                f"Final XGC state shape {final_xgc.shape} does not match seed "
                f"shape {seed_state.shape}"
            )
        # Density is prescribed end-to-end by the candidate profile.
        final_xgc[:, 0] = seed_state[:, 0]
        te_core_metric = float(final_xgc[0, 1])
        if not np.isfinite(te_core_metric):
            raise RuntimeError("Final XGC core metric is not finite")
        for result in ar_results:
            artifact = Path(result["iter_dir"]) / "out" / "xgc_output.npz"
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing XGC AR artifact: {artifact}")
            ar_iteration_artifacts.append(str(artifact.resolve()))
        np.savez_compressed(
            paths["xgc_handoff"],
            output=final_xgc.astype(np.float32),
            seed_state=seed_state,
            psi_n_full=psi_n_full,
            density_prescribed=np.array(True),
            te_core_metric=np.array(te_core_metric),
        )
        step_timings["xgc_ar_seconds"] = time.perf_counter() - stage_start

        # The AR helper constructs its model per call.  Release cached blocks
        # before the conditioned SOLPS scan while retaining the context model.
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stage_start = time.perf_counter()
        solps_input_dir = Path(
            ceo.step45_xgc2solps(
                context["xgc"],
                context["solps_instance"],
                final_xgc,
                output_dir=str(paths["solps_input_dir"]),
            )
        ).resolve()
        solps_input_nc = solps_input_dir / "valid" / "solps_input_D3D.nc"
        if not solps_input_nc.is_file():
            raise FileNotFoundError(f"Step 4.5 did not create {solps_input_nc}")
        _clean_pfr(solps_input_nc)
        step_timings["step45_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        aux = po.load_aux(
            str(solps_input_dir), context["matey_model_dir"], context["device"]
        )
        initial_solps = np.asarray(po.read_state_nc(str(solps_input_dir)), dtype=float)
        scan_rows: list[dict[str, Any]] = []
        scan_states: list[np.ndarray] = []
        for puff in PUFFS:
            state, iterations, residual, converged = _conditioned_fixed_point(
                initial_solps, float(puff), context, aux
            )
            metrics = po.wall_pedestal_metrics(state, context["geometry"])
            required_metrics = ("te_wall_eV", "ti_wall_eV", "te_ped_eV")
            if any(not np.isfinite(float(metrics[name])) for name in required_metrics):
                raise RuntimeError(f"Non-finite SOLPS metrics at puff {puff:.6e}")
            row = {
                "puff": float(puff),
                **{name: float(value) for name, value in metrics.items()},
                "fp_iterations": int(iterations),
                "residual": float(residual),
                "converged": bool(converged),
            }
            scan_rows.append(row)
            scan_states.append(state)

        wall_feasible_indices = [
            index
            for index, row in enumerate(scan_rows)
            if row["te_wall_eV"] <= WALL_LIMIT_EV
            and row["ti_wall_eV"] <= WALL_LIMIT_EV
        ]
        selection_pool = wall_feasible_indices or list(range(len(scan_rows)))
        selected_index = min(
            selection_pool,
            key=lambda index: max(
                scan_rows[index]["te_wall_eV"], scan_rows[index]["ti_wall_eV"]
            ),
        )
        best = scan_rows[selected_index]
        selected_state = scan_states[selected_index]
        wall_ok = bool(
            best["te_wall_eV"] <= WALL_LIMIT_EV
            and best["ti_wall_eV"] <= WALL_LIMIT_EV
        )
        pedestal_ok = bool(best["te_ped_eV"] >= PEDESTAL_FLOOR_EV)
        is_good = wall_ok and pedestal_ok

        _write_json(
            paths["puff_scan"],
            {
                "puffs": PUFFS,
                "rows": scan_rows,
                "selected_index": selected_index,
                "selection": (
                    "min_max_wall_among_wall_feasible"
                    if wall_feasible_indices
                    else "min_max_wall_among_all"
                ),
                "wall_limit_eV": WALL_LIMIT_EV,
            },
        )
        np.savez_compressed(
            paths["selected_solps_state"],
            state=np.asarray(selected_state, dtype=np.float32),
            initial_state=np.asarray(initial_solps, dtype=np.float32),
            selected_index=np.array(selected_index),
            puff=np.array(best["puff"]),
            residual=np.array(best["residual"]),
            fp_iterations=np.array(best["fp_iterations"]),
            converged=np.array(best["converged"]),
        )
        step_timings["solps_scan_seconds"] = time.perf_counter() - stage_start

        ended_at = _utc_now()
        elapsed_seconds = time.perf_counter() - start_time
        artifact_summary = {
            "profile": str(paths["profile"]),
            "ar_xgc_dir": str(paths["ar_xgc_dir"]),
            "ar_xgc_iterations": ar_iteration_artifacts,
            "xgc_handoff": str(paths["xgc_handoff"]),
            "solps_input": str(solps_input_nc),
            "puff_scan": str(paths["puff_scan"]),
            "selected_solps_state": str(paths["selected_solps_state"]),
            "summary": str(paths["summary"]),
        }
        summary = {
            "schema_version": 1,
            "status": "ok",
            "method": METHOD_NAME,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "gpu_id": gpu_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(context["device"]),
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_seconds": float(elapsed_seconds),
            "constants": {
                "xgc_ar_steps": N_XGC_STEPS,
                "psi_anchor_core": PSI_ANCHOR_CORE,
                "anchor_blend": ANCHOR_BLEND,
                "anchor_channels": ANCHOR_CHANNELS,
                "density_prescribed": True,
                "temperature_convention": "(2*T_perp+T_para)/3",
                "puffs": PUFFS,
                "fp_alpha": FP_ALPHA,
                "fp_tolerance": FP_TOL,
                "fp_iteration_cap": FP_CAP,
                "main_chamber_columns": [
                    MAIN_CHAMBER_COL_START,
                    MAIN_CHAMBER_COL_STOP,
                ],
                "temperature_freeze_psi_max": 1.0,
                "density_row_weights": {"21": 0.33, "22": 0.67, "23+": 1.0},
                "wall_limit_eV": WALL_LIMIT_EV,
                "pedestal_floor_eV": PEDESTAL_FLOOR_EV,
            },
            "model": {
                "conditioned_checkpoint": context["conditioned_checkpoint"],
                "conditioned_yaml": context["conditioned_yaml"],
                "conditioned_config_name": context["conditioned_config_name"],
                "matey_model_dir": context["matey_model_dir"],
            },
            "timings": step_timings,
            "artifacts": artifact_summary,
            "summary_path": str(paths["summary"]),
            "scan": scan_rows,
            "selected_index": int(selected_index),
            "puff_best": float(best["puff"]),
            "te_core_metric": te_core_metric,
            "te_wall_eV": float(best["te_wall_eV"]),
            "ti_wall_eV": float(best["ti_wall_eV"]),
            "te_ped_eV": float(best["te_ped_eV"]),
            "ne_ped_m3": float(best["ne_ped_m3"]),
            "p_ped_Pa": float(best["p_ped_Pa"]),
            "q_peak_proxy": float(best["q_peak_proxy"]),
            "wall_ok": wall_ok,
            "pedestal_ok": pedestal_ok,
            "is_good": is_good,
            "feasible": is_good,
        }
        _write_json(paths["summary"], summary)
        return _json_value(summary)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "error",
            "method": METHOD_NAME,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "gpu_id": gpu_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(context.get("device")),
            "started_at": started_at,
            "ended_at": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timings": step_timings,
            "summary_path": str(paths["summary"]),
            "artifacts": {
                "profile": str(paths["profile"]),
                "ar_xgc_iterations": ar_iteration_artifacts,
                "summary": str(paths["summary"]),
            },
        }
        _write_json(paths["summary"], failure)
        raise


__all__ = ["build_final_innerloop_context", "evaluate_final_innerloop"]
