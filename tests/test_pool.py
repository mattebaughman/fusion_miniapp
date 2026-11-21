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
