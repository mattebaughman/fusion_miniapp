"""Touch recorder and checker: what "validated" means for this mini-app.

The stand-ins call `touch(kind)` at each touch the real IGNOS inner loop makes; sim.py adds the
ones it observes from outside (context build, files on disk, result returned). `check` compares
the counts of one evaluation with what the knobs imply. The kinds follow the checklist in the
plan's section 2; the numbers in brackets are rows of the 50-touch list in
research_notes/.../ignos_purpose_and_touches.md, section 2.1.

Per evaluation, with k = rollout_steps, p = puffs, c = fixed_point_cap, b = 1 if the context was
built in this call else 0, s = small forwards observed:
  checkpoint_load, model_to_device  b + 2     (the real loop always pays b = 1: three loads)
  graph_forward                     1 + k
  small_forward                     p * c     (sim.py sets the parent's tolerance to 0)
  artifact_write                    7 + 2k    (11 files at k = 2, as in the real loop)
  library_import                    4 on a worker process's first build, 0 on a later build in the
                                    same process (another set of context knobs); 0 without a build
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

KINDS = {
    "context_build": "per-process context built in this call (first call in a worker, or new context knobs) [9-13]",
    "library_import": "torch, numpy, scipy.io, netCDF4 loaded by the worker process's first context build [2-8]",
    "mesh_read": "generated mesh file read: context, priming, AR helper, each AR step [9, 17, 23, 32]",
    "geometry_read": "generated .mat geometry read by scipy loadmat, twice per context [10]",
    "checkpoint_load": "torch.load of a checkpoint onto the device: context, priming, AR helper [12, 18, 23]",
    "model_to_device": "model built from its YAML config and moved to the device [13, 18, 23]",
    "graph_forward": "graph model forward on the device: priming + one per rollout step [20, 27]",
    "host_to_device": "host-to-device copies [19, 24, 28, 41, 43]",
    "device_to_host": "device-to-host copies [21, 30, 45]",
    "handoff_memory": "a stage's output becomes the next stage's input in memory [22, 33, 37]",
    "handoff_file_write": "netCDF hand-off file written by step 4.5 [38]",
    "handoff_file_rewrite": "the same file reopened and rewritten in place (the parent's own _clean_pfr) [39]",
    "handoff_file_read": "the hand-off file read by two readers [40, 42]",
    "small_forward": "fixed-point iteration: small grid forward with a host round trip [43-46]",
    "artifact_write": "files in the evaluation's work directory when it returns [14, 25, 26, 31, 35, 38, 49]",
    "result_return": "scalar result record returned by the parent's worker function [50]",
}

_active: Counter | None = None


def touch(kind: str, n: int = 1) -> None:
    """Record n touches of one kind in the active recording (a no-op outside one)."""
    if kind not in KINDS:
        raise KeyError(f"unknown touch kind {kind!r}")
    if _active is not None:
        _active[kind] += n


@contextmanager
def recording():
    """Collect touches for one evaluation. Not thread-safe; sim.evaluate holds a lock."""
    global _active
    _active = Counter()
    try:
        yield _active
    finally:
        _active = None


def expected(sim: dict, counts: dict) -> dict:
    """Allowed (low, high) count of every touch kind, from the knobs and the observed cold start."""
    b = int(counts.get("context_build", 0))
    k, p, c = sim["rollout_steps"], sim["puffs"], sim["fixed_point_cap"]
    s = int(counts.get("small_forward", 0))
    exact = {
        "mesh_read": b + 2 + k,
        "geometry_read": 2 * b,
        "checkpoint_load": b + 2,
        "model_to_device": b + 2,
        "graph_forward": 1 + k,
        "host_to_device": 12 + 2 * k + s,   # 5 priming + 5 AR helper + 2 per step + 2 aux + 1 per small forward
        "device_to_host": 1 + 3 * k + s,    # 1 priming + 3 per step + 1 per small forward
        "handoff_memory": 1 + k,
        "handoff_file_write": 1,
        "handoff_file_rewrite": 1,
        "handoff_file_read": 2,
        "artifact_write": 7 + 2 * k,
        "result_return": 1,
    }
    want = {kind: (n, n) for kind, n in exact.items()}
    want["context_build"] = (0, 1)
    want["library_import"] = (0, 4) if b else (0, 0)    # 4 or 0: standins loads all four or none
    want["small_forward"] = (p * c, p * c)
    return want


def check(counts: dict, sim: dict) -> str:
    """'pass', or 'fail: <kind>=<got> (want <range>); ...' naming every kind out of range."""
    bad = []
    for kind, (lo, hi) in expected(sim, counts).items():
        got = int(counts.get(kind, 0))
        if not lo <= got <= hi:
            bad.append(f"{kind}={got} (want {lo})" if lo == hi else f"{kind}={got} (want {lo}..{hi})")
    return "pass" if not bad else "fail: " + "; ".join(bad)
