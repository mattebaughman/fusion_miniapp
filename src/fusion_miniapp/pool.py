"""Resilient ProcessPoolExecutor for RHAPSODY's ConcurrentExecutionBackend.

Subclasses `ProcessPoolExecutor` so RHAPSODY takes its cloudpickle process path
(`isinstance(executor, ProcessPoolExecutor)`), while adding two protections:
- On `BrokenProcessPool` (e.g. a SIGKILLed / OOM-killed worker), rebuilds the inner pool once per
  break and resubmits lost tasks at most once each.
- Bounded `shutdown(wait=True)`: waits briefly for clean exit, sends SIGTERM to leftovers, then
  SIGKILLs any workers still alive before the deadline so `session.close()` can never hang.

`on_gpu` is a worker initializer that gives each worker one GPU (the AUTONMI layer's workers.gpu_per_worker).
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, InvalidStateError, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

PLACED: tuple[int, str] | None = None   # in a worker that on_gpu placed: its number and its GPU


def on_gpu(counter, gpus: list[str], init=None) -> None:
    """A worker initializer: this worker takes the next number i from `counter` (a multiprocessing Value that
    every pool placing its workers shares), and sees GPU gpus[i % len(gpus)] alone (CUDA_VISIBLE_DEVICES, set
    before this process starts CUDA; its cold-start children inherit it); then `init` (the pool's own
    initializer)."""
    global PLACED
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    PLACED = (i, gpus[i % len(gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = PLACED[1]
    if init is not None:
        init()


def _safe_set_result(fut: Future, value) -> None:
    try:
        fut.set_result(value)
    except InvalidStateError:
        pass


def _safe_set_exception(fut: Future, exc: BaseException) -> None:
    try:
        fut.set_exception(exc)
    except InvalidStateError:
        pass


def _stop_pool(
    pool: ProcessPoolExecutor,
    timeout: float,
    *,
    cancel_futures: bool = True,
    snapshot: tuple[object, object] | None = None,
) -> int:
    """Shut down one inner ProcessPoolExecutor within `timeout` seconds; return count SIGKILLed."""
    t0 = time.monotonic()
    deadline = t0 + timeout
    grace_end = t0 + min(1.0, 0.25 * timeout)
    kill_at = t0 + 0.75 * timeout

    # Snapshot BEFORE pool.shutdown(wait=False), which sets pool._processes and
    # pool._executor_manager_thread to None while the manager thread still owns the dict.
    if snapshot is not None:
        procs_dict, mgr = snapshot
    else:
        procs_dict = getattr(pool, "_processes", None)
        mgr = getattr(pool, "_executor_manager_thread", None)
    seen_procs: dict[int, object] = {}

    def live() -> list:
        if procs_dict:
            try:
                for p in list(procs_dict.values()):
                    seen_procs[id(p)] = p
            except RuntimeError:
                pass
        return [p for p in seen_procs.values() if p.is_alive()]

    live()
    pool.shutdown(wait=False, cancel_futures=cancel_futures)

    def all_done() -> bool:
        return not live() and not (mgr is not None and mgr.is_alive())

    while not all_done() and time.monotonic() < grace_end:
        time.sleep(0.02)
    if all_done():
        return 0

    # Past grace: stop queued work and ask live workers to exit cleanly (SIGTERM).
    pool.shutdown(wait=False, cancel_futures=True)
    for p in live():
        try:
            p.terminate()
        except Exception:
            pass

    while not all_done() and time.monotonic() < kill_at:
        time.sleep(0.02)
    if all_done():
        return 0

    # Still alive after SIGTERM: hard-kill leftovers and reap within remaining budget.
    killed = 0
    still = live()
    for p in still:
        try:
            p.kill()
            killed += 1
        except Exception:
            pass
    for p in still:
        p.join(timeout=max(0.05, deadline - time.monotonic()))
    if mgr is not None and mgr.is_alive():
        mgr.join(timeout=max(0.05, deadline - time.monotonic()))
    return killed


class ResilientProcessPool(ProcessPoolExecutor):
    """A ProcessPoolExecutor that recovers once per break and never hangs on shutdown."""

    def __init__(
        self,
        max_workers: int | None = None,
        mp_context=None,
        initializer=None,
        initargs: tuple = (),
        *,
        shutdown_timeout: float = 10.0,
    ) -> None:
        # Intentionally do NOT call super().__init__: delegate to a replaceable inner pool
        # while preserving isinstance(..., ProcessPoolExecutor) for RHAPSODY.
        if shutdown_timeout <= 0:
            raise ValueError(f"shutdown_timeout must be > 0, got {shutdown_timeout!r}")
        self._max_workers = max_workers
        self._mp_context = mp_context
        self._initializer = initializer
        self._initargs = initargs
        self._shutdown_timeout = float(shutdown_timeout)
        self._lock = threading.Lock()
        self._closed = False
        self._no_rebuild = False
        self._gen = 0
        self._rebuilds = 0
        self._resubmitted = 0
        self._killed_at_shutdown = 0
        self._retired: list[ProcessPoolExecutor] = []
        self._stopping: list[tuple[ProcessPoolExecutor, tuple[object, object]]] = []
        self._pool = self._make_pool()

    def stop_rebuilds(self) -> None:
        """Disable future pool rebuilds (e.g. from a SIGTERM callback before shutdown() runs)."""
        with self._lock:
            self._no_rebuild = True

    def _make_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=self._mp_context,
            initializer=self._initializer,
            initargs=self._initargs,
        )

    def _rebuild_locked(self, broken_gen: int) -> ProcessPoolExecutor | None:
        """Rebuild once if `broken_gen` is still the current generation; return retired pool if rebuilt."""
        if self._gen != broken_gen:
            return None
        dead = self._pool
        self._pool = self._make_pool()
        self._gen += 1
        self._rebuilds += 1
        self._retired.append(dead)
        return dead

    def _retire_async(self, dead: ProcessPoolExecutor) -> None:
        """Clean up a broken inner pool off its own manager thread (avoiding self-join deadlock)."""
        def reap() -> None:
            _stop_pool(dead, min(self._shutdown_timeout, 2.0), cancel_futures=True)
            with self._lock:
                if dead in self._retired:
                    self._retired.remove(dead)
        threading.Thread(target=reap, daemon=True).start()

    def submit(self, fn, /, *args, **kwargs) -> Future:
        outer: Future = Future()
        self._dispatch(outer, fn, args, kwargs, attempt=0)
        return outer

    def _dispatch(self, outer: Future, fn, args: tuple, kwargs: dict, *, attempt: int) -> None:
        with self._lock:
            if self._closed:
                if attempt == 0:
                    raise RuntimeError("cannot schedule new futures after shutdown")
                _safe_set_exception(outer, BrokenProcessPool("executor shut down during retry"))
                return
            pool, gen = self._pool, self._gen

        try:
            inner = pool.submit(fn, *args, **kwargs)
        except BrokenProcessPool as exc:
            if attempt >= 1:
                _safe_set_exception(outer, exc)
                return
            dead = None
            with self._lock:
                if self._closed:
                    raise RuntimeError("cannot schedule new futures after shutdown") from exc
                if self._no_rebuild:
                    _safe_set_exception(outer, exc)
                    return
                dead = self._rebuild_locked(gen)
                self._resubmitted += 1
            if dead is not None:
                self._retire_async(dead)
            self._dispatch(outer, fn, args, kwargs, attempt=1)
            return

        outer.add_done_callback(lambda f: inner.cancel() if f.cancelled() else None)

        def on_done(fut: Future) -> None:
            if outer.done():
                return
            if fut.cancelled():
                outer.cancel()
                return
            exc = fut.exception()
            if exc is None:
                _safe_set_result(outer, fut.result())
                return
            if isinstance(exc, BrokenProcessPool) and attempt == 0:
                dead = None
                with self._lock:
                    if self._closed or self._no_rebuild:
                        _safe_set_exception(outer, exc)
                        return
                    dead = self._rebuild_locked(gen)
                    self._resubmitted += 1
                if dead is not None:
                    self._retire_async(dead)
                self._dispatch(outer, fn, args, kwargs, attempt=1)
                return
            _safe_set_exception(outer, exc)

        inner.add_done_callback(on_done)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            self._closed = True
            self._no_rebuild = True
            fresh = [self._pool, *self._retired]
            self._retired.clear()
            self._pool = None
            tracked = [(p, (getattr(p, "_processes", None), getattr(p, "_executor_manager_thread", None)))
                       for p in fresh if p is not None]
            if not wait:
                self._stopping.extend(tracked)
                to_stop = tracked
            else:
                to_stop = [*self._stopping, *tracked]
                self._stopping.clear()
        if not wait:
            for p, _ in to_stop:
                p.shutdown(wait=False, cancel_futures=cancel_futures)
            return
        deadline = time.monotonic() + self._shutdown_timeout
        killed = 0
        for p, snap in to_stop:
            remaining = max(0.2, deadline - time.monotonic())
            killed += _stop_pool(p, remaining, cancel_futures=cancel_futures, snapshot=snap)
        if killed:
            with self._lock:
                self._killed_at_shutdown += killed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "rebuilds": self._rebuilds,
                "resubmitted": self._resubmitted,
                "killed_at_shutdown": self._killed_at_shutdown,
            }
