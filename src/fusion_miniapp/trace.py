"""The inner-loop trace: one evaluation's inner loop, step by step, for the Monitor. Added instrumentation,
not in IGNOS.

IGNOS keeps only each puff's end values, and its validated evaluator drops even those. With the knob
`monitor.capture`, an evaluation's record gains `trace`, about 230 numbers in the hundredk shape:
- the parent's own: the summary's per-puff scan rows (puff, the six metrics, iterations, residual,
  converged) and its four stage timings, which `_evaluate_validated_candidate` drops (sim.py wraps
  `evaluate_final_innerloop` to keep them);
- per step, recorded in fusion's stand-ins (standins.py): for the XGC stage's priming state and each
  rollout step, the relative change and the mean density, electron and ion temperature over the nodes
  outside the anchored core (psi > EDGE_PSI); for every iteration of every puff's fixed point, the
  residual and the te_wall, ti_wall and q_peak_proxy of its new state (the cells wall_pedestal_metrics
  reads); and the hand-off's normalisation ranges (load_aux).
The hooks only read arrays, so results, touches and the parent's files are unchanged.

In the record: {"kind", "synthetic", "complete", "sections": "<name>:<shape> ...", "f32": the values as
little-endian float32, section after section, in base64}, about 1.4 KB in the hundredk shape; strings, a
bool and no floats, so it passes the parent's _json_value, JSON and pickle unchanged. `unpack` returns the
arrays; COLUMNS names their last axis. `complete` is false when the hooks saw other counts than the knobs
give (the missing cells are NaN). Timings are not reproducible; everything else repeats.

The fixture (`evaluate_fixture`) has no inner loop: its trace has the same shape, is synthetic and says
so (`kind` synthetic_smooth or synthetic_hash, `synthetic` true), and uses none of the fixture's own random
draws (knob `monitor.fixture_trace`):
- smooth (the default): a smooth function of the normalised mutable parameters with one planted split.
  Candidates on either side of SPLIT_PARAM's mid-range follow each other until fixed-point iteration
  fixed_point_cap // 2 of every puff, then part in te_wall, ti_wall and q_peak_proxy. Its series end at
  their own values, not the record's, which come from a hash of the parameters.
- hash: random values from a second generator seeded by the parameters' digest; the selected puff's
  series and scan row end at the record's own te_wall, ti_wall and q_peak_proxy (its scan row also at the
  record's pedestal values).

One evaluation per process records at a time (sim.evaluate's lock), as touches.recording does.
"""
from __future__ import annotations

import base64
import functools
import math
from contextlib import contextmanager

import numpy as np

EDGE_PSI = 0.80             # the XGC means and change cover the nodes outside the anchored core
SCAN = ("puff", "te_wall_eV", "ti_wall_eV", "te_ped_eV", "ne_ped_m3", "p_ped_Pa", "q_peak_proxy", "fp_iterations",
        "residual", "converged")                                      # the parent's per-puff scan row
TIMINGS = ("step3_seconds", "xgc_ar_seconds", "step45_seconds", "solps_scan_seconds")   # the parent's stages
COLUMNS = {
    "xgc": ("rel_change", "ne_m3", "te_eV", "ti_eV"),                  # rows: priming, then each rollout step
    "fixed_point": ("residual", "te_wall_eV", "ti_wall_eV", "q_peak_proxy"),   # [puff, iteration]
    "handoff": ("ne_lo", "te_lo", "ti_lo", "ne_hi", "te_hi", "ti_hi"),  # load_aux's normalisation ranges
    "scan": SCAN,                                                       # [puff]
    "timings_s": TIMINGS,
}
SPLIT_PARAM, SPLIT_AT, SPLIT_WIDTH = "a1_Te", 0.5, 0.02   # the smooth fixture's planted split (normalised)

_active: Recorder | None = None


class Recorder:
    """What the hooks gave during one evaluation, kept as given (each hook does as little as it can); pack
    computes the trace from it once the evaluation has ended."""

    def __init__(self, sim: dict) -> None:
        self.sim = sim
        self.xgc: list = []                  # the XGC states seen, each once, in order
        self.steps: list[tuple] = []         # each XGC step's (state before, state after), as indices into xgc
        self.psi = None                      # the nodes' normalised psi
        self.fixed_point: list[tuple] = []   # (puff index, iteration, residual, the new state's wall cells)
        self.wall_values = None              # wall cells -> te_wall, ti_wall, q_peak_proxy
        self.puffs: dict = {}                # each puff's index, in the order run
        self.iterations: list[int] = []      # each puff's fixed-point iterations
        self.puff: float | None = None       # the puff of the iterations that follow
        self.handoff: tuple | None = None    # load_aux's (low, high)
        self.scan: list | None = None
        self.timings: list | None = None


@contextmanager
def recording(sim: dict | None):
    """Record one evaluation's trace, in the shape the `sim` knobs give: yields the Recorder, or None when sim
    is None (every hook is then a no-op). Not thread-safe; sim.evaluate holds a lock."""
    global _active
    _active = None if sim is None else Recorder(sim)
    try:
        yield _active
    finally:
        _active = None


# --- the hooks (standins.py, and sim.py's wrapper of the parent's inner loop) ------------------------------

def xgc_step(before, after, psi) -> None:
    """An XGC state (the priming output, or a rollout step's) and the one before it. pack takes, over the
    nodes with psi > EDGE_PSI, the relative L2 change and the mean density, electron and ion temperature (the
    parent's (2 T_perp + T_para) / 3). The arrays are kept, not copied: neither the stand-ins nor the parent
    change them in place (the parent copies the last state before it prescribes the density)."""
    rec = _active
    if rec is not None:
        if not rec.xgc or rec.xgc[-1] is not before:          # each rollout step starts from the last one's state
            rec.xgc.append(before)
        rec.xgc.append(after)
        rec.steps.append((len(rec.xgc) - 2, len(rec.xgc) - 1))
        rec.psi = psi


def puff(value) -> None:
    """The puff of the fixed-point iterations that follow (each iteration's model call)."""
    if _active is not None:
        _active.puff = float(value)


def fixed_point_step(residual: float, state, cells, values) -> None:
    """One fixed-point iteration (its residual is computed once per iteration): the residual, and a copy of
    the new state's wall cells, cells(state). values(cells) gives their te_wall, ti_wall and q_peak_proxy;
    pack applies it to every iteration's cells at once."""
    rec = _active
    if rec is None:
        return
    i = rec.puffs.setdefault(rec.puff, len(rec.puffs))
    if i == len(rec.iterations):
        rec.iterations.append(0)
    j = rec.iterations[i]
    rec.iterations[i] = j + 1
    rec.fixed_point.append((i, j, residual, cells(state)))
    rec.wall_values = values


def handoff(lo, hi) -> None:
    """The hand-off file's normalisation ranges (density, electron and ion temperature: low, then high)."""
    if _active is not None:
        _active.handoff = (lo, hi)


def summary(s: dict) -> None:
    """The parent's inner-loop summary: its per-puff scan rows and its stage timings."""
    rec = _active
    if rec is not None:
        rec.scan = [[_num(row.get(k)) for k in SCAN] for row in s.get("scan", [])]
        rec.timings = [_num(s.get("timings", {}).get(k)) for k in TIMINGS]


def _num(v) -> float:
    return math.nan if v is None else float(v)


# --- the record's field --------------------------------------------------------------------------------------

def shapes(sim: dict) -> dict[str, tuple]:
    """Each section's shape for the `sim` knobs."""
    k, p, c = sim["rollout_steps"], sim["puffs"], sim["fixed_point_cap"]
    return {"xgc": (k + 1, len(COLUMNS["xgc"])), "fixed_point": (p, c, len(COLUMNS["fixed_point"])),
            "handoff": (len(COLUMNS["handoff"]),), "scan": (p, len(SCAN)), "timings_s": (len(TIMINGS),)}


@functools.lru_cache(maxsize=64)
def _header(shape: tuple) -> str:
    """The sections header, one string per shape (shared by the records of one process)."""
    return " ".join(f"{name}:{'x'.join(map(str, dims))}" for name, dims in shape)


def _field(kind: str, header: str, values: np.ndarray, complete: bool) -> dict:
    return {"kind": kind, "synthetic": kind != "inner_loop", "complete": bool(complete), "sections": header,
            "f32": base64.b64encode(values.astype("<f4").tobytes()).decode("ascii")}


def encode(kind: str, arrays: dict[str, np.ndarray], complete: bool = True) -> dict:
    """The record's `trace`: the sections in order, as float32."""
    return _field(kind, _header(tuple((name, a.shape) for name, a in arrays.items())),
                  np.concatenate([np.ravel(a) for a in arrays.values()]), complete)


def unpack(trace: dict) -> dict[str, np.ndarray]:
    """A record's `trace` as arrays (float32), by section."""
    data, out, i = np.frombuffer(base64.b64decode(trace["f32"]), "<f4"), {}, 0
    for part in trace["sections"].split():
        name, dims = part.split(":")
        shape = tuple(int(d) for d in dims.split("x"))
        n = math.prod(shape)
        out[name] = data[i:i + n].reshape(shape)
        i += n
    if i != data.size:
        raise ValueError(f"trace holds {data.size} values, its sections {i}")
    return out


# density, and the electron and ion temperature (2 T_perp + T_para) / 3, from the XGC channels' means
_XGC_MEANS = np.zeros((8, 3))
_XGC_MEANS[0, 0], _XGC_MEANS[[1, 2], 1], _XGC_MEANS[[4, 5], 2] = 1, (2 / 3, 1 / 3), (2 / 3, 1 / 3)


@functools.lru_cache(maxsize=64)
def _layout(k: int, p: int, c: int) -> tuple:
    """For rollout_steps, puffs and fixed_point_cap: the sections header, the number of values, and each
    section's (name, offset, shape)."""
    shape, at, i = shapes({"rollout_steps": k, "puffs": p, "fixed_point_cap": c}), [], 0
    for name, dims in shape.items():
        at.append((name, i, dims))
        i += math.prod(dims)
    return _header(tuple(shape.items())), i, tuple(at)


def pack(rec: Recorder) -> dict:
    """What a recording saw, as the record's `trace`, in the shape the knobs give (NaN where nothing came). It
    is complete when every count fit that shape exactly: k + 1 XGC steps, the hand-off's 6 values, and per
    puff the cap's iterations and a scan row."""
    header, size, at = _layout(rec.sim["rollout_steps"], rec.sim["puffs"], rec.sim["fixed_point_cap"])
    values = np.full(size, np.nan)
    out = {name: values[i:i + math.prod(dims)].reshape(dims) for name, i, dims in at}
    xgc, fp, n_handoff = out["xgc"], out["fixed_point"], 0
    if rec.steps:
        steps = rec.steps[:len(xgc)]
        edge = np.flatnonzero(np.asarray(rec.psi) > EDGE_PSI)
        x = np.stack(rec.xgc).take(edge, axis=1).astype(np.float64)
        if steps == [(i, i + 1) for i in range(len(steps))]:  # each step from the last one's state (run_ar)
            b, a = x[:len(steps)], x[1:len(steps) + 1]
        else:
            b, a = x[[s[0] for s in steps]], x[[s[1] for s in steps]]
        d = a - b
        xgc[:len(steps), 0] = np.sqrt(np.einsum("sij,sij->s", d, d) / np.einsum("sij,sij->s", b, b))
        xgc[:len(steps), 1:] = (np.add.reduce(a, axis=1) / a.shape[1]) @ _XGC_MEANS
    rows = [r for r in rec.fixed_point if r[0] < fp.shape[0] and r[1] < fp.shape[1]]
    if rows:
        i, j, residual, cells = zip(*rows)
        fp[i, j] = np.array((residual, *rec.wall_values(np.stack(cells)))).T   # every iteration's at once
    if rec.handoff is not None:
        h = np.concatenate([np.ravel(v) for v in rec.handoff])
        n_handoff = len(h)
        out["handoff"][:n_handoff] = h[:len(out["handoff"])]
    for name, given in (("scan", rec.scan), ("timings_s", rec.timings)):
        if given:
            out[name][:len(given)] = given[:len(out[name])]
    counts = (len(rec.steps), n_handoff, len(rec.scan or ()), len(rec.timings or ()), rec.iterations)
    complete = counts == (len(xgc), len(out["handoff"]), fp.shape[0], len(out["timings_s"]), [fp.shape[1]] * fp.shape[0])
    return _field("inner_loop", header, values, complete)


def split(results: list[dict]) -> tuple[list[dict], list]:
    """Each evaluation record without its trace (a new dict: the record is left as it is), and the traces
    (None where a record has none), in the same order."""
    return ([{k: v for k, v in r.items() if k != "trace"} if "trace" in r else r for r in results],
            [r.get("trace") for r in results])


# --- the fixture's synthetic trace ---------------------------------------------------------------------------

# Four smooth features of the seven normalised parameters: each the mean of four of them (fixed weights).
_W = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]],
              float)
_W /= _W.sum(axis=0)


@functools.lru_cache(maxsize=64)
def _basis(k: int, p: int, c: int) -> dict:
    """What the smooth trace's shape fixes: each series with its features at 1 (they only scale it), the puffs
    (as sim.py sets them), and the planted split's ramp (0 before iteration c // 2, then rising towards 0.3)."""
    t, n = np.arange(k + 1), np.arange(c)
    decay, puffs = np.exp(-n / max(1.0, c / 4)), np.linspace(0.5e21, 4.0e21, p)
    v = puffs / 4.0e21
    end = (1.2 - 0.4 * v)[:, None]                         # where the wall temperatures settle, by puff
    scan = np.zeros((p, len(SCAN)))
    scan[:, 0], scan[:, 7] = puffs, c                       # the puff; iterations: the cap (converged 0)
    return {"xgc": np.stack([0.2 * 0.5 ** t, 1e19 * (1 + 0.02 * t), 200 * (1 - 0.03 * t), 220 * (1 - 0.03 * t)],
                            axis=1),
            "te": 150 * (end + (2 - end) * decay), "ti": 170 * end + (320 - 170 * end) * decay,
            "residual": 0.05 * (1 + 0.5 * v)[:, None] * decay, "ne": 1e19 * (0.5 + v)[:, None],
            "te_ped": 400 * (1 - 0.1 * v), "ne_ped": 3e19 * (1 + 0.2 * v), "scan": scan,
            "handoff": np.array([1e18, 10, 10, 1.5e20, 1500, 1500]),
            "ramp": 0.3 * np.where(n >= c // 2, 1 - np.exp(-(n - c // 2 + 1.0)), 0.0)}


_NO_TIMINGS = np.full(len(TIMINGS), np.nan)   # the fixture runs no stage


def synthetic(how: str, params: dict, bounds: dict, sim: dict, record: dict, digest: bytes, selected: int) -> dict:
    """The fixture's trace (`monitor.fixture_trace` smooth or hash), in the shape the knobs give. bounds: the
    mutable parameters' ranges, in order; record: the fixture's record (hash ends at its values); digest: its
    parameters' sha256; selected: its selected puff's index."""
    k, p, c = sim["rollout_steps"], sim["puffs"], sim["fixed_point_cap"]
    if how == "smooth":
        x = np.clip([(float(params[n]) - lo) / (hi - lo) for n, (lo, hi) in bounds.items()], 0.0, 1.0)
        return encode("synthetic_smooth", _smooth(x, list(bounds).index(SPLIT_PARAM), k, p, c))
    if how == "hash":
        rng = np.random.default_rng(int.from_bytes(digest[8:16], "little"))
        return encode("synthetic_hash", _hash(rng, k, p, c, record, selected))
    raise ValueError(f"unknown fixture trace {how!r}")


def _smooth(x: np.ndarray, j: int, k: int, p: int, c: int) -> dict:
    b = _basis(k, p, c)
    f = 1 + 0.5 * (x @ _W)                                  # the four features, as factors from 1 to 1.5
    split = 1 + math.tanh((x[j] - SPLIT_AT) / SPLIT_WIDTH) * b["ramp"]   # the planted split, by iteration
    te = b["te"] * (f[2] * split)
    fp = np.empty((p, c, len(COLUMNS["fixed_point"])))
    fp[..., 0], fp[..., 1], fp[..., 2] = b["residual"] * f[0], te, b["ti"] * (f[3] * split)
    fp[..., 3] = b["ne"] * f[1] * te ** 1.5
    scan = b["scan"].copy()
    scan[:, [1, 2, 6, 8]] = fp[:, -1, [1, 2, 3, 0]]         # each puff's end
    scan[:, 3], scan[:, 4] = b["te_ped"] * f[3], b["ne_ped"] * f[1]
    scan[:, 5] = scan[:, 4] * 2 * scan[:, 3] * 1.602e-19
    return {"xgc": b["xgc"] * f, "fixed_point": fp, "handoff": b["handoff"] * f[[1, 2, 3, 1, 2, 3]], "scan": scan,
            "timings_s": _NO_TIMINGS}


def _hash(rng: np.random.Generator, k: int, p: int, c: int, record: dict, selected: int) -> dict:
    u = rng.uniform(size=(5, k + 1))
    xgc = np.stack([0.3 * u[0], 1e19 * (1 + 4 * u[1]), 50 + 1000 * u[2], 50 + 1000 * u[3]], axis=1)
    f = rng.uniform(size=(4, p, c))
    fp = np.stack([0.1 * f[0], 50 + 1000 * f[1], 50 + 1000 * f[2], 1e19 * (1 + 4 * f[3]) * (50 + 1000 * f[1]) ** 1.5],
                  axis=2)
    fp[selected, -1, 1:] = record["te_wall_eV"], record["ti_wall_eV"], record["q_peak_proxy"]
    s = rng.uniform(size=(2, p))
    te_ped, ne_ped = 100 + 900 * s[0], 1e19 * (1 + 4 * s[1])
    scan = np.stack([np.linspace(0.5e21, 4.0e21, p), fp[:, -1, 1], fp[:, -1, 2], te_ped, ne_ped,
                     ne_ped * 2 * te_ped * 1.602e-19, fp[:, -1, 3], np.full(p, float(c)), fp[:, -1, 0], np.zeros(p)],
                    axis=1)
    scan[selected, 3:6] = record["te_ped_eV"], record["ne_ped_m3"], record["p_ped_Pa"]
    h = rng.uniform(size=6)
    handoff = np.array([1e18, 10, 10, 5e19, 500, 500]) * (1 + h)
    return {"xgc": xgc, "fixed_point": fp, "handoff": handoff, "scan": scan, "timings_s": _NO_TIMINGS}
