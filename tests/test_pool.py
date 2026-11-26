"""Package 1 tests for pool.py: ResilientProcessPool recovery on SIGKILL and bounded shutdown."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from fusion_miniapp.pool import ResilientProcessPool


def _square(x: int) -> int:
    return x * x


def _kill_once_then_succeed(marker: str, value: int) -> int:
    """First time the marker file does not exist, create it and SIGKILL this worker."""
    path = Path(marker)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        os.kill(os.getpid(), signal.SIGKILL)
    except FileExistsError:
        pass
    return value * 10


def _always_sigkill(_: int) -> int:
    os.kill(os.getpid(), signal.SIGKILL)
    return 0


def _exit_sigterm_once_then_succeed(marker: str, value: int) -> int:
    path = Path(marker)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        os._exit(128 + signal.SIGTERM)
    except FileExistsError:
        pass
    return value * 10


def _ignore_sigterm_and_sleep(pid_file: str, seconds: float) -> int:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    tmp = Path(pid_file + ".tmp")
    tmp.write_text(str(os.getpid()))
    tmp.replace(pid_file)
    time.sleep(seconds)
    return 1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_isinstance_and_clean_run():
    ctx = mp.get_context("spawn")
    pool = ResilientProcessPool(max_workers=2, mp_context=ctx, shutdown_timeout=5.0)
    assert isinstance(pool, ProcessPoolExecutor)
    try:
        futs = [pool.submit(_square, i) for i in range(6)]
        assert [f.result(timeout=30) for f in futs] == [i * i for i in range(6)]
    finally:
        pool.shutdown(wait=True)
    assert pool.stats() == {"rebuilds": 0, "resubmitted": 0, "killed_at_shutdown": 0}


def test_sigkill_mid_task_rebuilds_once_and_delivers_all_results(tmp_path):
    ctx = mp.get_context("spawn")
    marker = str(tmp_path / "killed_once")
    pool = ResilientProcessPool(max_workers=2, mp_context=ctx, shutdown_timeout=5.0)
    try:
        futs = [pool.submit(_kill_once_then_succeed, marker, i) for i in range(6)]
        got = [f.result(timeout=30) for f in futs]
    finally:
        pool.shutdown(wait=True)
    assert got == [i * 10 for i in range(6)]
    st = pool.stats()
    assert st["rebuilds"] == 1
    assert st["resubmitted"] >= 1
    assert st["killed_at_shutdown"] == 0


def test_poison_task_fails_after_single_retry():
    ctx = mp.get_context("spawn")
    pool = ResilientProcessPool(max_workers=1, mp_context=ctx, shutdown_timeout=5.0)
    try:
        fut = pool.submit(_always_sigkill, 1)
        with pytest.raises(BrokenProcessPool):
            fut.result(timeout=30)
    finally:
        pool.shutdown(wait=True)
    st = pool.stats()
    assert st["rebuilds"] == 1
    assert st["resubmitted"] == 1


def test_shutdown_kills_sigterm_ignoring_worker_within_deadline(tmp_path):
    ctx = mp.get_context("spawn")
    pid_file = tmp_path / "stuck.pid"
    deadline = 1.5
    pool = ResilientProcessPool(max_workers=1, mp_context=ctx, shutdown_timeout=deadline)
    pool.submit(_ignore_sigterm_and_sleep, str(pid_file), 60.0)
    # Wait until the worker has installed SIG_IGN and entered sleep.
    wait_until = time.monotonic() + 15.0
    while not pid_file.exists() and time.monotonic() < wait_until:
        time.sleep(0.05)
    assert pid_file.exists(), "worker never started"
    worker_pid = int(pid_file.read_text().strip())

    t0 = time.monotonic()
    pool.shutdown(wait=True)
    elapsed = time.monotonic() - t0

    assert elapsed < deadline + 1.0, f"shutdown took {elapsed:.2f}s (deadline {deadline}s)"
    assert pool.stats()["killed_at_shutdown"] == 1
    # Give the OS a brief moment to finish reaping if needed.
    reap_until = time.monotonic() + 2.0
    while _pid_alive(worker_pid) and time.monotonic() < reap_until:
        time.sleep(0.05)
    assert not _pid_alive(worker_pid)


def test_rhapsody_backend_uses_process_path_and_recovers(tmp_path):
    from rhapsody import ComputeTask, Session
    from rhapsody.backends import ConcurrentExecutionBackend

    marker = str(tmp_path / "rhap_kill_once")

    async def go():
        pool = ResilientProcessPool(max_workers=2, mp_context=mp.get_context("spawn"), shutdown_timeout=5.0)
        backend = await ConcurrentExecutionBackend(pool)
        session = Session(backends=[backend], work_dir=str(tmp_path))
        try:
            tasks = [ComputeTask(function=_kill_once_then_succeed, uid=f"t.{i}", args=[marker, i]) for i in range(4)]
            futs = await session.submit_tasks(tasks)
            vals = [(await f)["return_value"] for f in futs]
            return vals, pool.stats()
        finally:
            await session.close()

    vals, st = asyncio.run(go())
    assert vals == [0, 10, 20, 30]
    assert st["rebuilds"] == 1 and st["resubmitted"] >= 1


def test_isolated_sigterm_worker_still_rebuilds_and_recovers(tmp_path):
    """An isolated worker exiting 143 (SIGTERM) without stop_rebuilds() must still recover."""
    ctx = mp.get_context("spawn")
    marker = str(tmp_path / "term_once")
    pool = ResilientProcessPool(max_workers=1, mp_context=ctx, shutdown_timeout=5.0)
    try:
        fut = pool.submit(_exit_sigterm_once_then_succeed, marker, 7)
        assert fut.result(timeout=30) == 70
    finally:
        pool.shutdown(wait=True)
    st = pool.stats()
    assert st["rebuilds"] == 1 and st["resubmitted"] == 1


def test_stop_rebuilds_suppresses_recovery():
    """Authoritative stop_rebuilds() (called on job-wide SIGTERM) suppresses rebuilds."""
    ctx = mp.get_context("spawn")
    pool = ResilientProcessPool(max_workers=1, mp_context=ctx, shutdown_timeout=5.0)
    pool.stop_rebuilds()
    try:
        fut = pool.submit(_always_sigkill, 1)
        with pytest.raises(BrokenProcessPool):
            fut.result(timeout=30)
    finally:
        pool.shutdown(wait=True)
    st = pool.stats()
    assert st["rebuilds"] == 0 and st["resubmitted"] == 0


def _placed(barrier) -> tuple:
    """A task: once as many workers run it as the barrier's parties (so each pool's workers have all started),
    this worker's number and GPU (pool.PLACED, set by on_gpu) and the GPUs it sees."""
    from fusion_miniapp import pool

    barrier.wait(timeout=120)
    return pool.PLACED, os.environ.get("CUDA_VISIBLE_DEVICES")


def test_each_inner_loop_worker_sees_gpu_i_mod_the_gpu_count_across_populations(monkeypatch):
    """workers.gpu_per_worker on the AUTONMI layer, with a faked count of 3 GPUs (FIXTURE: no GPU is used here):
    two populations' pools (layer_autonmi.sim_pool, 2 workers each) take their workers' numbers from one counter, so
    the 4 workers are 0 .. 3, the second pool's after the first's, and worker i sees GPU i mod 3 alone
    (CUDA_VISIBLE_DEVICES). Real placement on the local runtime was seen on one StellarAI node on 2026-11-25
    (README); under Dragon it has not run on GPUs."""
    layer = pytest.importorskip("fusion_miniapp.layer_autonmi")
    monkeypatch.setattr(layer, "_GPUS", ["0", "1", "2"])
    monkeypatch.setattr(layer, "_GPU_COUNTER", None)
    monkeypatch.setattr(layer, "_POOLS", [])
    pools, got = [layer.sim_pool(2), layer.sim_pool(2)], []
    with mp.get_context("spawn").Manager() as manager:
        try:
            for p in pools:   # each pool's 2 workers busy at once, so both start, before the next pool's
                barrier = manager.Barrier(2)
                got.append(sorted(f.result(timeout=180) for f in [p.submit(_placed, barrier) for _ in range(2)]))
        finally:
            for p in pools:
                p.shutdown(wait=True)
    assert got == [[((0, "0"), "0"), ((1, "1"), "1")], [((2, "2"), "2"), ((3, "0"), "0")]]
    assert layer._GPU_COUNTER.value == 4


def _placed_by_file(barrier, counter_file: str, gpus: list) -> tuple:
    """A task as a worker under Dragon starts sim.evaluate_placed: once as many workers run it as the barrier's
    parties, this worker places itself from inside the task (pool.on_gpu_file), and a second call changes nothing;
    its number and GPU (pool.PLACED) and the GPUs it sees."""
    from fusion_miniapp import pool

    barrier.wait(timeout=120)
    first = pool.on_gpu_file(counter_file, gpus)
    assert pool.on_gpu_file(counter_file, gpus) == first == pool.PLACED   # once per process
    return first, os.environ.get("CUDA_VISIBLE_DEVICES")


def test_each_worker_placed_from_its_task_sees_gpu_i_mod_the_gpu_count(tmp_path):
    """workers.gpu_per_worker under Dragon, whose pool workers start without this package's initializer: two
    populations' pools (plain spawned pools of 2 workers, one after the other; FIXTURE: a faked list of 3 GPUs, no
    GPU used) place their workers from inside a task on one counter file (pool.on_gpu_file), so the 4 workers are
    0 .. 3, the second pool's after the first's, worker i sees GPU i mod 3 alone, and the file holds the count. No
    laptop test reaches Dragon or a GPU: this placement has not run under Dragon."""
    counter, gpus, got = str(tmp_path / "inner_loop_gpus.count"), ["0", "1", "2"], []
    spawn = mp.get_context("spawn")
    with spawn.Manager() as manager:
        for _ in range(2):   # each pool's 2 workers busy at once, so both start, before the next pool's
            barrier = manager.Barrier(2)
            with ProcessPoolExecutor(2, mp_context=spawn) as p:
                got.append(sorted(f.result(timeout=180) for f in [p.submit(_placed_by_file, barrier, counter, gpus)
                                                                  for _ in range(2)]))
    assert got == [[((0, "0"), "0"), ((1, "1"), "1")], [((2, "2"), "2"), ((3, "0"), "0")]]
    assert (tmp_path / "inner_loop_gpus.count").read_text() == "4"


def test_evaluate_placed_places_its_worker_before_it_evaluates(tmp_path, monkeypatch):
    """sim.evaluate_placed (the task under Dragon with workers.gpu_per_worker): the worker's number and GPU from the
    counter file first (here the fifth worker, of a FIXTURE list of 3 GPUs), then evaluate with the other arguments,
    which sees that GPU alone (evaluate is a stand-in here: no inner loop runs)."""
    from fusion_miniapp import pool, sim

    monkeypatch.setattr(pool, "PLACED", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")   # put back after the test
    monkeypatch.setattr(sim, "evaluate", lambda **kw: {**kw, "seen": os.environ["CUDA_VISIBLE_DEVICES"]})
    counter = tmp_path / "inner_loop_gpus.count"
    counter.write_text("4")
    got = sim.evaluate_placed(gpu_counter=str(counter), gpus=["0", "1", "2"], params={"a": 1.0}, seed=3)
    assert got == {"params": {"a": 1.0}, "seed": 3, "seen": "1"}
    assert pool.PLACED == (4, "1") and counter.read_text() == "5"


def test_no_worker_is_placed_at_default_knobs_or_without_a_gpu(monkeypatch):
    """workers.gpu_per_worker off (the default), or on where the device probe saw no GPU (this laptop, the fixture):
    no GPU list, so sim_pool's workers get run.py's initializer alone and set no device. On, with GPUs:
    CUDA_VISIBLE_DEVICES's entries where it is set, else 0 .. n - 1. Under Dragon (no probe there): the head's
    CUDA_VISIBLE_DEVICES entries, none when it is unset."""
    layer = pytest.importorskip("fusion_miniapp.layer_autonmi")
    from fusion_miniapp import knobs as knob_files, run

    off = knob_files.base()
    on = knob_files.validate({"workers": {"gpu_per_worker": True}}, modes=("generations", "async"))
    assert off["workers"]["gpu_per_worker"] is False
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert layer.inner_loop_gpus(off, {"gpus": ["B200"] * 8}) == []
    assert layer.inner_loop_gpus(on, {"gpus": []}) == layer.inner_loop_gpus(on, None) == []
    assert layer.inner_loop_gpus(on, {"gpus": ["B200"] * 3}) == ["0", "1", "2"]
    assert layer.inner_loop_gpus(on, None, dragon=True) == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert layer.inner_loop_gpus(on, {"gpus": ["B200"] * 4}) == ["4", "5", "6", "7"]
    assert layer.inner_loop_gpus(on, None, dragon=True) == ["4", "5", "6", "7"]
    assert layer.inner_loop_gpus(off, None, dragon=True) == []
    monkeypatch.setattr(layer, "_GPUS", [])
    monkeypatch.setattr(layer, "_POOLS", [])
    p = layer.sim_pool(1)
    try:
        assert (p._initializer, p._initargs) == (run._worker_init, ())
    finally:
        p.shutdown(wait=True)
    with pytest.raises(knob_files.KnobError, match="workers.gpu_per_worker"):   # run.py's layers refuse it set
        knob_files.validate({"workers": {"gpu_per_worker": True}})
