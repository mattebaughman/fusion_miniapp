"""Stand-ins for the three non-public modules the parent inner loop reaches through `context`.

The parent's `evaluate_final_innerloop` (parent/steps/final_inner_loop.py, unmodified) gets all
model code from context["core_edge_optimization"], context["run_matey_inference"] and
context["puff_optimizer"]. This module supplies small public replacements with the same call
signatures, a one-time generator for the files the real loop finds on the site file system
(mesh, geometry, checkpoints and their YAML configs) and a replacement for the context builder.

Random weights and a generated mesh: this reproduces the data flow, library loads and resource
touches of the real loop, not its physics. Core PyTorch only, so one code path serves CUDA and
ROCm builds. Lifted from research_notes/.../prototypes/inner_loop/inject_probe.py, plus the second
graph-checkpoint load, xgc_input.npz per step, the colour-range file and the touch records.

The calls to trace.py record per-step values for the Monitor (added instrumentation, not in IGNOS). They
only read arrays, and do nothing unless the evaluation records a trace.
"""
from __future__ import annotations

import functools
import importlib
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.io import loadmat, savemat

from . import trace
from .touches import touch

H, W = 38, 98                 # SOLPS grid at its real size; the parent's _clean_pfr indexes it
N_HIST = 2                    # time history the grid model takes (po.N_STEPS in the real loop)
PUFF_SCALE = 4.785693e21      # puff normalisation used by the real conditioned model
CONFIG_NAME = "miniapp"
LIBRARIES = ("torch", "numpy", "scipy.io", "netCDF4")
_LOADED: set[str] = set()     # libraries this process's context builds have loaded (once per process)


def pick_device(name: str | None = "auto") -> torch.device:
    """auto: cuda if torch.cuda.is_available() (CUDA and ROCm builds alike), else cpu. Never picks an index."""
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("sim.device is 'cuda' but torch.cuda.is_available() is False")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("sim.device is 'mps' but torch.backends.mps.is_available() is False")
    return torch.device(name)


class GraphNet(nn.Module):
    """XGC graph-surrogate stand-in: linear encoder, message passing with index_add_, linear decoder."""

    def __init__(self, width: int = 32, layers: int = 2, f_in: int = 11, f_out: int = 8):
        super().__init__()
        self.enc = nn.Linear(f_in, width)
        self.mp = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(layers)])
        self.dec = nn.Linear(width, f_out)

    def forward(self, x, edge_index):
        h = torch.relu(self.enc(x))
        src, dst = edge_index[0], edge_index[1]
        for lin in self.mp:
            agg = torch.zeros_like(h).index_add_(0, dst, h[src])
            h = torch.relu(lin(torch.cat([h, agg], dim=-1)))
        return torch.sigmoid(self.dec(h))


class GridNet(nn.Module):
    """Actuator-conditioned SOLPS-surrogate stand-in: 2x2 patches, transformer encoder, puff conditioning."""

    def __init__(self, width: int = 32, layers: int = 2, heads: int = 2, c: int = 3, t: int = N_HIST):
        super().__init__()
        self.patch = nn.Conv2d(c * t, width, kernel_size=2, stride=2)
        self.cond = nn.Sequential(nn.Linear(t + 1, width), nn.ReLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width, batch_first=True)
        self.proc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, c * 4)

    def forward(self, x, cond):                                  # x (T, B, C, H, W), cond (B, T + 1)
        t, b, c, h, w = x.shape
        z = self.patch(x.permute(1, 0, 2, 3, 4).reshape(b, t * c, h, w))
        hh, ww = z.shape[-2:]
        tok = z.flatten(2).transpose(1, 2) + self.cond(cond).unsqueeze(1)
        y = self.head(self.proc(tok))
        y = y.view(b, hh, ww, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return torch.sigmoid(y)


def _up(array: np.ndarray, device: torch.device) -> torch.Tensor:
    touch("host_to_device")                                     # a no-op copy when the device is the CPU
    return torch.as_tensor(np.ascontiguousarray(array)).to(device)


def _down(tensor: torch.Tensor) -> np.ndarray:
    touch("device_to_host")
    return tensor.detach().cpu().numpy()


def _save_model(model: nn.Module, path: Path, pad_mb: float = 0.0) -> None:
    sd = OrderedDict(("module." + k, v) for k, v in model.state_dict().items())   # DDP-style prefix, as on site
    torch.save({"model_state": sd, "pad": torch.zeros(int(pad_mb * 2**20) // 4)}, path)


def load_model(cls, path: Path, config: dict, device: torch.device) -> nn.Module:
    """torch.load onto the device, build from the config, strip the prefix, load weights, eval()."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    touch("checkpoint_load")
    model = cls(**config).to(device)
    touch("model_to_device")
    model.load_state_dict(OrderedDict((k.replace("module.", "", 1), v) for k, v in ckpt["model_state"].items()))
    return model.eval()


def make_assets(root: Path, sim: dict) -> Path:
    """Write, once, the files the real loop reads from the site file system. Deterministic."""
    npp, npl = sim["nodes_per_plane"], sim["planes"]
    rng = np.random.default_rng(0)
    psi = np.sort(rng.uniform(0.0, 1.05, npp))                  # one plane, ordered core -> edge
    src = np.arange(npp - 1)
    edges = np.stack([np.r_[src, src + 1], np.r_[src + 1, src]])
    edge_index = np.concatenate([edges + p * npp for p in range(npl)], axis=1).astype(np.int64)
    np.savez(root / "mesh.npz", psi=psi, theta=rng.uniform(0, 2 * np.pi, npp), edge_index=edge_index)
    prcen = np.tile(1.0 + 1.3 * np.sin(np.pi * np.arange(W) / (W - 1)), (H, 1))   # outer midplane mid-grid
    pzcen = np.tile(np.linspace(-1.2, 1.2, H)[:, None], (1, W))
    savemat(root / "geometry.mat", {"Geo": {"prcen": prcen.ravel(), "pzcen": pzcen.ravel(),
                                            "psi_row": np.linspace(0.85, 1.05, H)}})
    config = {"width": sim["model_width"], "layers": 2}
    model_dir = root / "model"
    (model_dir / "training_checkpoints").mkdir(parents=True, exist_ok=True)
    (model_dir / "hyperparams.yaml").write_text(yaml.safe_dump(config))
    (root / "conditioned.yaml").write_text(yaml.safe_dump({CONFIG_NAME: config}))
    with torch.random.fork_rng(devices=[]):                     # leave the caller's RNG untouched
        torch.manual_seed(0)
        _save_model(GraphNet(**config), model_dir / "training_checkpoints" / "ckpt.tar", sim["checkpoint_mb"])
        _save_model(GridNet(**config), root / "conditioned_best_ckpt.tar")
    return root


def read_mesh(case_dir) -> dict:
    with np.load(Path(case_dir) / "mesh.npz") as f:
        mesh = {k: f[k] for k in f.files}
    touch("mesh_read")
    return mesh


def _read_nc(solps_dir) -> np.ndarray:
    with netCDF4.Dataset(str(Path(solps_dir) / "valid" / "solps_input_D3D.nc")) as f:
        state = np.stack([np.asarray(f.variables[n][0], np.float64) for n in ("ne2d", "te2d", "ti2d")])
    touch("handoff_file_read")
    return state


# ------------------------------ context["core_edge_optimization"] ------------------------------
def step3(arrays, xgc_instance=None, shot=None, time_ms=None, gpath=None, timings=None):
    """Step 3: the 1-D candidate profile onto every graph node by np.interp, tiled over planes."""
    psi_n = np.asarray(xgc_instance.mesh.psi, np.float64) / float(xgc_instance.psix)
    n = xgc_instance.n_planes
    return {"ne2dxgc": np.tile(np.interp(psi_n, arrays["psi"], arrays["ne"] * 1e20), n).astype(np.float32),
            "te2dxgc": np.tile(np.interp(psi_n, arrays["psi"], arrays["Te"] * 1e3), n).astype(np.float32)}


def step45(xgc_instance, solps_instance, xgc_output, output_dir=None):
    """Step 4.5: plane average, nodes binned by psi onto the 38 x 98 grid, netCDF hand-off file."""
    touch("handoff_memory")                                     # the final XGC state arrives in memory
    npl, npp = xgc_instance.n_planes, xgc_instance.mesh.nnodes
    pa = np.asarray(xgc_output).reshape(npl, npp, -1).mean(axis=0)
    psi_n = np.asarray(xgc_instance.mesh.psi) / xgc_instance.psix
    row = np.clip(np.searchsorted(np.asarray(solps_instance["Geo"].psi_row), psi_n) - 1, 0, H - 1)
    count = np.maximum(np.bincount(row, minlength=H), 1)
    fields = {"ne2d": pa[:, 0], "te2d": (2 * pa[:, 1] + pa[:, 2]) / 3, "ti2d": (2 * pa[:, 4] + pa[:, 5]) / 3}
    path = Path(output_dir) / "valid" / "solps_input_D3D.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), mode="w", format="NETCDF4") as f:
        f.createDimension("time", 2)
        f.createDimension("nr", H)
        f.createDimension("nz", W)
        for name, values in fields.items():
            rv = np.bincount(row, weights=values, minlength=H) / count
            rv[rv <= 0] = rv[rv > 0].min() if (rv > 0).any() else 1.0
            f.createVariable(name, "f4", ("time", "nr", "nz"))[:] = np.broadcast_to(rv[:, None], (2, H, W))
    touch("handoff_file_write")
    return Path(output_dir)


# ------------------------------ context["run_matey_inference"] ------------------------------
def node_values(edge_bcs):
    """Seed state [N, 8]: density, electron and ion temperatures from step 3; flows zero."""
    ne, te = edge_bcs["ne2dxgc"], edge_bcs["te2dxgc"]
    z = np.zeros_like(ne)
    return np.stack([ne, te, te, z, te, te, z, z], axis=1)


def run_ar(model_dir, edge_bcs, xgc_case_dir, n_ar_steps=2, output_base_dir=None, save_viz=False,
           verbose=False, distributed=False, psi_n_full=None, psi_anchor_core=None, anchor_blend=0.05,
           anchor_channels=None, anchor_state=None, *, device):
    """XGC stage: a priming forward with its own model load, then an AR helper that reads the mesh
    and loads the checkpoint again and runs n_ar_steps anchored rollout steps, as the real one does."""
    model_dir, out = Path(model_dir), Path(output_base_dir)
    config = yaml.safe_load((model_dir / "hyperparams.yaml").read_text())
    ckpt = model_dir / "training_checkpoints" / "ckpt.tar"
    seed = node_values(edge_bcs).astype(np.float32)
    lo_np, hi_np = seed.min(0), seed.max(0) + 1e-6

    def batch():                                                # the real batch builder reads the mesh case
        mesh = read_mesh(xgc_case_dir)
        npp = mesh["psi"].size
        npl = seed.shape[0] // npp
        pos = np.stack([np.tile(mesh["psi"], npl), np.tile(mesh["theta"], npl), np.repeat(np.arange(npl), npp)], 1)
        return (_up(pos.astype(np.float32), device), _up(mesh["edge_index"], device),
                _up(lo_np, device), _up(hi_np, device))

    def forward(model, pos, ei, lo, hi, state):
        x = torch.cat([pos, (state - lo) / (hi - lo)], dim=-1)
        with torch.no_grad():
            y = model(x, ei) * (hi - lo) + lo
        touch("graph_forward")
        return y, x

    model = load_model(GraphNet, ckpt, config, device)          # priming call
    pos, ei, lo, hi = batch()
    y, _ = forward(model, pos, ei, lo, hi, _up(seed, device))
    primed = _down(y)
    trace.xgc_step(seed, primed, psi_n_full)
    touch("handoff_memory")                                     # priming output -> rollout input, via the host

    pos, ei, lo, hi = batch()                                   # AR helper: mesh again, checkpoint again
    model = load_model(GraphNet, ckpt, config, device)
    state = _up(primed, device)
    ne_prev, te_prev = np.asarray(edge_bcs["ne2dxgc"]), np.asarray(edge_bcs["te2dxgc"])
    np.savez(out / "ar_xgc_clim_ranges.npz", ne_min=max(float(ne_prev.min()) * 0.75, 1e15),
             ne_max=float(ne_prev.max()) * 1.25, te_min=max(float(te_prev.min()) / 1e3 * 0.75, 1e-6),
             te_max=float(te_prev.max()) / 1e3 * 1.25)
    weight = np.ones_like(seed)
    edge = np.clip((np.asarray(psi_n_full) - (psi_anchor_core - anchor_blend)) / anchor_blend, 0, 1)
    weight[:, list(anchor_channels)] = edge.astype(np.float32)[:, None]
    anchor = np.asarray(anchor_state, np.float32)
    results = []
    for k in range(n_ar_steps):
        it = out / f"ar_iter_{k:03d}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        (it / "out").mkdir(parents=True, exist_ok=True)
        np.savez(it / "out" / "xgc_input.npz", ne2dxgc=ne_prev, te2dxgc=te_prev)
        y, x = forward(model, pos, ei, lo, hi, state)
        w, a = _up(weight, device), _up(anchor, device)
        state = w * y + (1 - w) * a                             # core anchor, blended on the device
        y_np = _down(state)
        trace.xgc_step(results[-1]["output"] if results else primed, y_np, psi_n_full)
        np.savez(it / "out" / "xgc_output.npz", output=y_np, input=_down(x))
        results.append({"output": y_np, "input": _down(x), "iter_idx": k, "iter_dir": str(it)})
        read_mesh(xgc_case_dir)                                 # the real step reopens the case for a unit
        ne_prev, te_prev = y_np[:, 0], (2 * y_np[:, 1] + y_np[:, 2]) / 3
        if k < n_ar_steps - 1:
            touch("handoff_memory")                             # stays on the device as the next input
    return results


# ------------------------------ context["puff_optimizer"] ------------------------------
def load_aux(solps_data_dir, reference_model_dir, device):
    """Normalisation ranges from the hand-off file, uploaded to the device."""
    s = _read_nc(solps_data_dir)
    lo, hi = s.min(axis=(1, 2), keepdims=True), s.max(axis=(1, 2), keepdims=True) * 1.5 + 1e-6
    trace.handoff(lo, hi)
    return {"lo": _up(lo.astype(np.float32), device), "hi": _up(hi.astype(np.float32), device), "device": device}


def read_state_nc(path):
    """The second reader of the hand-off file: the SOLPS initial state (3, 38, 98)."""
    return _read_nc(path)


def forward_conditioned(model, aux, hist, puff):
    """One fixed-point iteration's model call: upload, normalise, forward, denormalise, clamp, download."""
    trace.puff(puff)
    lo, hi = aux["lo"], aux["hi"]
    x = (_up(np.stack(hist).astype(np.float32), aux["device"]) - lo) / (hi - lo)
    cond = torch.full((1, len(hist) + 1), float(puff) / PUFF_SCALE, device=aux["device"])
    with torch.no_grad():
        y = model(x.unsqueeze(1), cond)[0] * (hi - lo) + lo
    touch("small_forward")
    return _down(torch.maximum(torch.minimum(y, hi), lo)).astype(np.float64)


def rel_l2(a, b):
    """The fixed point's residual: called once per iteration, with the new state and the one before."""
    residual = float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))
    trace.fixed_point_step(residual, a, wall_cells, wall_values)
    return residual


# a state's wall cells, rows 19-37 at columns 1 and 96 of each of its 3 fields, as flat indices into (3, 38, 98)
WALL_CELLS = (np.arange(3)[:, None, None] * H * W + np.arange(19, 38)[:, None] * W + np.array([1, 96])).reshape(3, -1)


def wall_cells(state) -> np.ndarray:
    """A state's (3, 38, 98) wall cells: density, electron and ion temperature, each (38,)."""
    return np.asarray(state).reshape(-1)[WALL_CELLS]


def wall_values(cells) -> tuple:
    """te_wall_eV, ti_wall_eV and q_peak_proxy: the largest over the wall cells, of one state (3, 38) or of
    several at once (n, 3, 38)."""
    ne, te, ti = cells[..., 0, :], cells[..., 1, :], cells[..., 2, :]
    return te.max(-1), ti.max(-1), (ne * np.abs(te) ** 1.5).max(-1)


def wall_pedestal_metrics(state, geom):
    ne, te, ti = state
    jxa = geom["jxa"]
    te_wall, ti_wall, q_peak = wall_values(wall_cells(state))
    return {"te_wall_eV": float(te_wall), "ti_wall_eV": float(ti_wall),
            "te_ped_eV": float(te[0, jxa]), "ne_ped_m3": float(ne[0, jxa]),
            "p_ped_Pa": float(ne[0, jxa] * (te[0, jxa] + ti[0, jxa]) * 1.602e-19),
            "q_peak_proxy": float(q_peak)}


# ------------------------------ the context builder ------------------------------
def build_context(assets: Path, sim: dict, fil) -> dict:
    """Replacement for final_inner_loop.build_final_innerloop_context: the same keys, public packages only."""
    for name in sorted(set(LIBRARIES) - _LOADED):              # the real cost is paid once per process
        importlib.import_module(name)
        _LOADED.add(name)
        touch("library_import")
    device = pick_device(sim["device"])
    mesh = read_mesh(assets)
    solps_instance = loadmat(str(assets / "geometry.mat"), squeeze_me=True, struct_as_record=False)
    touch("geometry_read")
    geo = loadmat(str(assets / "geometry.mat"), squeeze_me=True, struct_as_record=False)["Geo"]
    touch("geometry_read")                                      # the real po.load_geometry() reads it again
    jxa = int(np.argmax(np.asarray(geo.prcen).reshape(H, W)[18]))
    psi_omp = np.asarray(geo.psi_row, np.float64)
    cols = np.arange(fil.MAIN_CHAMBER_COL_START, fil.MAIN_CHAMBER_COL_STOP)
    freeze = np.zeros((H, W), dtype=bool)
    freeze[np.ix_(np.flatnonzero(psi_omp < 1.0), cols)] = True
    row_weights = np.zeros(H)
    row_weights[21], row_weights[22], row_weights[23:] = 0.33, 0.67, 1.0
    config = yaml.safe_load((assets / "conditioned.yaml").read_text())[CONFIG_NAME]
    npp = int(mesh["psi"].size)
    return {
        "method": fil.METHOD_NAME, "device": device,
        "conditioned_checkpoint": str(assets / "conditioned_best_ckpt.tar"),
        "conditioned_yaml": str(assets / "conditioned.yaml"), "conditioned_config_name": CONFIG_NAME,
        "matey_model_dir": str(assets / "model"), "xgc_mesh_dir": str(assets),
        "solps_geometry": str(assets / "geometry.mat"),
        "xgc": SimpleNamespace(mesh=SimpleNamespace(nnodes=npp, psi=mesh["psi"]), psix=1.0, n_planes=sim["planes"]),
        "solps_instance": solps_instance, "nnodes": npp, "psi_n_nodes": mesh["psi"].astype(np.float64),
        "psi_omp": psi_omp, "jxa": jxa, "geometry": {"jxa": jxa},
        "temperature_freeze_mask": freeze, "density_weights": np.tile(row_weights[:, None], (1, W)),
        "main_chamber_columns": cols,
        "conditioned_model": load_model(GridNet, assets / "conditioned_best_ckpt.tar", config, device),
        "core_edge_optimization": SimpleNamespace(
            STEP1_DIR=str(assets), step3_construct_full_profiles_and_extract_edge_bcs=step3,
            step45_xgc2solps=step45),
        "run_matey_inference": SimpleNamespace(
            _xgc_node_values_from_edge_bcs=node_values,
            run_matey_ar_xgc_inference=functools.partial(run_ar, device=device)),
        "puff_optimizer": SimpleNamespace(
            N_STEPS=N_HIST, load_aux=load_aux, read_state_nc=read_state_nc,
            forward_conditioned=forward_conditioned, rel_l2=rel_l2, wall_pedestal_metrics=wall_pedestal_metrics),
    }
