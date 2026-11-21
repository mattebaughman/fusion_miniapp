"""Processes end with their owners: a cold-start child whose simulation worker is SIGKILLed exits at once
(the plan, package 1, and Codex's 02:39 finding), so it never runs beside the pool's retry."""
import os
import signal
import subprocess
import time
from multiprocessing import get_context


def _worker(q):
    from fusion_miniapp import sim
    spawn = subprocess.Popen

    def spy(*a, **k):
        p = spawn(*a, **k)
        q.put(p.pid)
        return p
    subprocess.Popen = spy
    sim.evaluate(knobs={"device": "cpu", "cold_start_fraction": 1.0, "fixed_point_cap": 300, "puffs": 6},
                 agent_id="x", seed=0)


def test_a_cold_child_exits_when_its_worker_is_killed():
    ctx = get_context("spawn")
    q = ctx.Queue()
    worker = ctx.Process(target=_worker, args=(q,))
    worker.start()
    child = q.get(timeout=60)
    time.sleep(1.0)                                   # the child is evaluating
    os.kill(worker.pid, signal.SIGKILL)
    worker.join()
    t = time.time()
    while time.time() - t < 10:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(child, signal.SIGKILL)
    raise AssertionError("the cold child outlived its worker by 10 s")


def test_sigterm_to_a_busy_two_process_run_keeps_both_processes_counts(tmp_path):
    """SIGTERM while 2 x 2,000 campaigns keep both event loops busy: each agent process still gets the
    signal (a plain handler, not the loop's self-pipe, which a busy loop can fill), unwinds in time and
    writes its counts; the run records itself as interrupted and exits 143 well within 60 s."""
    import json
    import sys
    (tmp_path / "k.yaml").write_text(json.dumps({"agents": {"per_process": 2000}, "workers": {"sim_workers": 4},
                                                 "outer": {"population": 16, "generations": 2, "logs": "none",
                                                           "parent_stdout": "drop"}}))
    p = subprocess.Popen([sys.executable, "-m", "fusion_miniapp.run", "--agents", "4000", "--sim", "fixture",
                          "--knobs", str(tmp_path / "k.yaml"), "--out", str(tmp_path / "r")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    t = time.time()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 143
    assert time.time() - t < 45
    rec = json.loads((tmp_path / "r" / "run_record.json").read_text())
    assert rec["status"] == "interrupted" and rec["counts"]["failed_agent_processes"] == []
    assert rec["counts"]["sim_tasks"] > 0 and len(rec["layer_stats"]) == 2
