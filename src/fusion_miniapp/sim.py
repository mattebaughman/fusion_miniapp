"""The simulation task: `evaluate`, the IGNOS inner loop turned down, and its fixture twin.

`evaluate` calls the parent's own worker `evolutionary_loop._evaluate_single_candidate`, which runs
the parent's `final_inner_loop.evaluate_final_innerloop`; both are unmodified and run over the
stand-ins in standins.py. The parent is configured only through its module attributes and the
injected context. All physics output is synthetic (random-weight stand-in models).

The parent's fixed-point tolerance is set to 0, so every fixed point runs to `fixed_point_cap` and
the cap sets the duration. A cold start (`sim.cold_start_fraction`) runs one evaluation in a fresh,
single-use Python process started from inside the worker's task (`python -m fusion_miniapp.sim`), so it
still counts against the worker pool's size. Which evaluations are cold is a fixed function of the
campaign, generation and candidate number, so a run repeats exactly.

With the `monitor` knobs (capture), the record also carries the inner-loop trace (trace.py: added
instrumentation, not in IGNOS): the parent summary's per-puff rows and stage timings, which its evaluator
drops (kept by wrapping `evaluate_final_innerloop`, as `_clean_pfr` is wrapped), and per-step values
recorded in the stand-ins. `evaluate_fixture`'s trace is synthetic.

Importing this module imports neither torch nor the parent files; `evaluate_fixture` never imports torch.
"""
from __future__ import annotations

import atexit
import functools
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import PARENT_DIR, trace
from .knobs import monitor_section, sim_section
from .touches import KINDS, check, recording, touch

STANDIN_METHOD = "miniapp_standin"          # replaces the parent's "validated_campaign3" label
PROFILE = ("three_region", "H")             # the parent's validated campaign profile family
_LOCK = threading.RLock()                   # the parent's module attributes and the touch recorder are per process
_CONTEXTS: dict[tuple, tuple[Path, dict]] = {}   # per-process context cache: context knobs -> (assets, context)
_CONTEXT_KNOBS = ("nodes_per_plane", "planes", "model_width", "checkpoint_mb", "device")
_RUN_DIRS: set[Path] = set()                # work directories of evaluations in progress (for cleanup)
_CHILDREN: set = set()                      # cold-start processes in progress (for cleanup)
COLD_CHILD = "FUSION_MINIAPP_COLD_CHILD"     # set in a cold-start process, which never starts another


def cleanup() -> None:
    """Stop this process's cold-start children and remove its assets and in-progress work directories;
    for a worker about to os._exit."""
    for child in list(_CHILDREN):
        child.kill()
        child.wait()
    for path in [assets for assets, _ in _CONTEXTS.values()] + list(_RUN_DIRS):
        shutil.rmtree(path, ignore_errors=True)


def is_cold(fraction: float, agent_id: str, generation: int, candidate_idx: int) -> bool:
    """Whether this evaluation is a cold start: a hash of its place in the run below the fraction."""
    if fraction <= 0:
        return False
    h = hashlib.sha256(f"{agent_id}/{generation}/{candidate_idx}".encode()).digest()
    return int.from_bytes(h[:8], "little") / 2**64 < fraction


def _cold(**kwargs) -> dict:
    """One evaluation in a fresh Python process (it loads torch, builds its context, then exits). The
    child works under a folder this worker owns, so the worker's cleanup removes it even after the child
    is killed (unless sim.keep_artifacts)."""
    t_start = time.time()                                # the worker starts the task: a cold start's
    sim = sim_section(kwargs["knobs"])                   # interpreter launch is simulation time, not queue
    root = Path(tempfile.mkdtemp(prefix="cold_", dir=work_root(sim)))
    if not sim["keep_artifacts"]:
        _RUN_DIRS.add(root)
    kwargs = {**kwargs, "knobs": {**sim, "work_root": str(root)}}
    try:
        record = _cold_child(kwargs)
        record.update(t_start=t_start, elapsed_seconds=time.time() - t_start)
        return record
    finally:
        if not sim["keep_artifacts"]:
            _RUN_DIRS.discard(root)
            shutil.rmtree(root, ignore_errors=True)


def _cold_child(kwargs: dict) -> dict:
    """The child gets the read end of a pipe whose only write end this worker holds. However the worker
    ends, SIGKILL included, the child then reads end-of-file and exits, so it never overlaps the retry."""
    life_r, life_w = os.pipe()
    env = {**os.environ, COLD_CHILD: "1", "FUSION_MINIAPP_LIFELINE": str(life_r)}
    try:
        child = subprocess.Popen([sys.executable, "-m", "fusion_miniapp.sim"], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True,
                                 pass_fds=(life_r,))
    except BaseException:
        os.close(life_w)
        raise
    finally:
        os.close(life_r)
    _CHILDREN.add(child)
    try:
        out, err = child.communicate(json.dumps(kwargs), timeout=3600)
    finally:
        _CHILDREN.discard(child)
        if child.poll() is None:
            child.kill()
            child.wait()
        os.close(life_w)
    if child.returncode:
        raise RuntimeError(f"cold-start evaluation failed ({child.returncode}): {err[-2000:]}")
    record = json.loads(out.splitlines()[-1])
    record["cold_start"] = True
    return record


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
        if not hasattr(fil.evaluate_final_innerloop, "parent_function"):   # the trace keeps the summary's
            inner = fil.evaluate_final_innerloop        # scan and timings, which the parent's evaluator drops;
                                                        # the parent imports this by name on every call

            def evaluate_final_innerloop(*args, **kwargs):
                summary = inner(*args, **kwargs)
                trace.summary(summary)
                return summary
            evaluate_final_innerloop.parent_function = inner
            fil.evaluate_final_innerloop = evaluate_final_innerloop
        fil.METHOD_NAME = STANDIN_METHOD
        return evo, fil


def _params(candidate: dict | None, seed: int, evo, params: dict | None = None) -> dict:
    """A full parameter dict (`params`, as the parent's outer loop makes them) passes unchanged, as the
    parent's evaluator takes it. Otherwise `candidate`: some of the seven mutable parameters, clipped to
    the parent's ranges as its mutation does, every other parameter from the parent's own Latin-hypercube
    sampler at `seed`."""
    if params is not None:
        if candidate:
            raise ValueError("give a full params dict or a partial candidate, not both")
        for name, value in params.items():
            if not isinstance(value, str) and not math.isfinite(float(value)):
                raise ValueError(f"parameter {name} is not finite: {value}")
        return dict(params)
    candidate = candidate or {}
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


def peak_rss_mb() -> float:
    """This process's peak resident memory. Linux: VmHWM, which starts afresh at exec (ru_maxrss would
    carry the parent's size over a spawn's fork and exec); elsewhere ru_maxrss."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1)


def work_root(sim: dict) -> Path | None:
    """sim.work_root with $VARS and ~ expanded, created; None for the system temp. An unset variable is
    an error, never a literal folder name."""
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
    if os.environ.get("FUSION_MINIAPP_THREADS"):                # run.py's per-worker thread limit
        import torch
        torch.set_num_threads(int(os.environ["FUSION_MINIAPP_THREADS"]))
    if entry is None or not entry[0].is_dir():
        assets = Path(tempfile.mkdtemp(prefix="fusion_miniapp_assets_", dir=work_root(sim)))
        atexit.register(shutil.rmtree, assets, True)
        standins.make_assets(assets, sim)
    else:
        assets = entry[0]
    context = standins.build_context(assets, sim, fil)
    touch("context_build")
    _CONTEXTS[key] = (assets, context)
    return context


def evaluate(candidate: dict | None = None, knobs: dict | None = None, agent_id: str = "", seed: int = 0, *,
             params: dict | None = None, candidate_idx: int = 0, generation: int = 0,
             monitor: dict | None = None) -> dict:
    """Evaluate one candidate with the parent's inner loop over the stand-ins; return a JSON-safe record.

    params: a full parameter dict from the parent's outer loop, passed to the parent's worker as is;
    candidate_idx and generation are its place in the campaign (the parent's work directory is
    gen{generation}_candidate{candidate_idx}). Or candidate: any subset of the seven mutable profile
    parameters, the rest sampled at `seed`. knobs: the `sim` section of a knob
    file (None = smoke preset; a partial dict is merged over it). monitor: the `monitor` section (a
    partial dict is merged over the smoke preset's); with capture on, the record gains `trace` (trace.py).
    None, as the LangGraph and Academy layers call it, records none. Calls in one process run one at a
    time (a lock); parallelism comes from processes. evaluate sets no device: the launcher exposes the GPU,
    or (the AUTONMI layer with workers.gpu_per_worker) the worker's initializer gives each worker one through
    CUDA_VISIBLE_DEVICES (pool.on_gpu), and the device rule picks it up.
    """
    sim, watch = sim_section(knobs), monitor_section(monitor)
    if not os.environ.get(COLD_CHILD) and is_cold(sim["cold_start_fraction"], agent_id, generation, candidate_idx):
        return _cold(candidate=candidate, knobs=knobs, agent_id=agent_id, seed=seed, params=params,
                     candidate_idx=candidate_idx, generation=generation, monitor=monitor)
    with _LOCK:
        t_start = time.time()   # start and duration on one clock: the speed section builds intervals of them
        evo, fil = _parent()
        params = _params(candidate, seed, evo, params)
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(agent_id))[:40]
        run_dir = Path(tempfile.mkdtemp(prefix=f"eval_{slug}_", dir=work_root(sim)))
        _RUN_DIRS.add(run_dir)
        try:
            with recording() as counts, trace.recording(sim if watch and watch["capture"] else None) as steps:
                context = _context(sim, fil)
                context["method"] = fil.METHOD_NAME
                fil.N_XGC_STEPS = sim["rollout_steps"]
                fil.PUFFS = np.linspace(0.5e21, 4.0e21, sim["puffs"])
                fil.FP_CAP = sim["fixed_point_cap"]
                fil.FP_TOL = 0.0                                # never converged early: the cap sets duration
                kw = {"inner_loop": "validated", "model": PROFILE[0], "kind": PROFILE[1], "generation": int(generation),
                      "evo_run_dir": str(run_dir), "conditioned_ckpt": context["conditioned_checkpoint"],
                      "conditioned_yaml": context["conditioned_yaml"],
                      "conditioned_config_name": context["conditioned_config_name"],
                      "matey_model_dir": context["matey_model_dir"]}
                evo._FINAL_CONTEXT_CACHE.clear()                # the parent must find this context, never build its own
                evo._FINAL_CONTEXT_CACHE[(kw["conditioned_ckpt"], kw["conditioned_yaml"],
                                          kw["conditioned_config_name"], kw["matey_model_dir"])] = context
                record = evo._evaluate_single_candidate((params, None, int(candidate_idx), kw))   # gpu_id None
                touch("result_return")
                touch("artifact_write", sum(p.is_file() for p in run_dir.rglob("*")))
        finally:
            _RUN_DIRS.discard(run_dir)
            if not sim["keep_artifacts"]:
                shutil.rmtree(run_dir, ignore_errors=True)
        touches = {kind: counts[kind] for kind in KINDS}
        record.update(emulated=True, device=str(context["device"]), t_start=t_start,
                      elapsed_seconds=time.time() - t_start, worker_peak_rss_mb=peak_rss_mb(),
                      touches=touches, touch_check=check(touches, sim))
        if steps is not None:
            record["trace"] = trace.pack(steps)
        if not sim["keep_artifacts"]:
            record["summary_path"] = ""                         # the work directory is gone
        return fil._json_value(record)                          # NaN -> None, numpy -> Python


@functools.lru_cache(maxsize=1)
def _bounds(evo) -> dict:
    """The seven mutable parameters' ranges, in order, as `_params` clips them."""
    ranges = evo.get_param_ranges(*PROFILE)
    return {n: tuple(evo.DELTA_CLIP if n == "Delta" else ranges[n]) for n in evo.MUTABLE_PARAMS[PROFILE[0]]}


def evaluate_fixture(candidate: dict | None = None, knobs: dict | None = None, agent_id: str = "", seed: int = 0, *,
                     params: dict | None = None, candidate_idx: int = 0, generation: int = 0,
                     monitor: dict | None = None) -> dict:
    """The same keys as evaluate, instantly and without torch, plus fixture: true. For large agent-count
    tests. Values are a deterministic function of the parameters, not a simulation; so is its trace
    (monitor capture), which is synthetic (monitor.fixture_trace) and draws none of the five values."""
    t_start = time.time()
    sim, watch = sim_section(knobs), monitor_section(monitor)
    evo, fil = _parent()
    params = _params(candidate, seed, evo, params)
    profile = evo.generate_profile(psi=None, model=PROFILE[0], kind=PROFILE[1], params=dict(params))
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).digest()
    u = np.random.default_rng(int.from_bytes(digest[:8], "little")).uniform(size=5)
    selected = int(u[4] * sim["puffs"])
    te_core = float(profile["Te"][0]) * 1e3
    te_wall, ti_wall, te_ped = 50 + 1000 * u[0], 50 + 1000 * u[1], 100 + 900 * u[2]
    ne_ped = 1e19 * (1 + 4 * u[3])
    wall_ok = bool(te_wall <= fil.WALL_LIMIT_EV and ti_wall <= fil.WALL_LIMIT_EV)
    pedestal_ok = bool(te_ped >= fil.PEDESTAL_FLOOR_EV)
    record = {
        "params": params, "candidate_idx": int(candidate_idx), "input_te_core_keV": float(profile["Te"][0]),
        "te_core_metric": te_core, "te_core_keV": te_core / 1e3, "ti_core_keV": float("nan"),
        "te_wall_eV": te_wall, "ti_wall_eV": ti_wall, "wall_ok": wall_ok, "pedestal_ok": pedestal_ok,
        "elm_ok": None, "is_good": wall_ok and pedestal_ok, "te_ped_eV": te_ped, "ne_ped_m3": ne_ped,
        "p_ped_Pa": ne_ped * 2 * te_ped * 1.602e-19, "q_peak_proxy": ne_ped * te_wall ** 1.5,
        "te_wall_div_eV": te_wall, "ti_wall_div_eV": ti_wall,
        "puff_best": float(np.linspace(0.5e21, 4.0e21, sim["puffs"])[selected]),
        "summary_path": "", "inner_loop": "fixture",
        "emulated": True, "fixture": True, "device": "none", "t_start": t_start, "worker_peak_rss_mb": None,
        "elapsed_seconds": time.time() - t_start,
        "touches": {kind: 0 for kind in KINDS}, "touch_check": "fail: fixture, no touches made",
    }
    if watch and watch["capture"]:
        record["trace"] = trace.synthetic(watch["fixture_trace"], params, _bounds(evo), sim, record, digest, selected)
    return fil._json_value(record)


if __name__ == "__main__":   # a cold-start process: one evaluation, its record as the last stdout line
    import signal

    def _stop(*_):
        cleanup()
        os._exit(143)
    signal.signal(signal.SIGTERM, _stop)

    def _watch(fd: int) -> None:   # end-of-file: the worker that started this process is gone
        while os.read(fd, 1):
            pass
        cleanup()
        os._exit(143)
    if os.environ.get("FUSION_MINIAPP_LIFELINE"):
        threading.Thread(target=_watch, args=(int(os.environ["FUSION_MINIAPP_LIFELINE"]),), daemon=True).start()
    _record = evaluate(**json.loads(sys.stdin.read()))
    sys.stdout.write("\n" + json.dumps(_record) + "\n")
