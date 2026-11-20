"""Run `python -m fusion_miniapp.run` once and sample the summed resident memory of its process tree.

usage: python ladder.py OUT_DIR <run arguments...>
Writes OUT_DIR/stderr.log (the run's stderr) and OUT_DIR/ladder.json (exit code, outer wall time,
the largest summed RSS of the process tree seen at 1 s sampling, and the run's one-line summary).
A laptop functional check, not a benchmark.
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def tree_rss_mb(root: int) -> float:
    rows = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout.splitlines()
    kids, rss = {}, {}
    for row in filter(str.strip, rows):
        pid, ppid, kib = map(int, row.split())
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = kib
    total, todo = 0, [root]
    while todo:
        pid = todo.pop()
        total += rss.get(pid, 0)
        todo += kids.get(pid, [])
    return total / 1024


out, args = Path(sys.argv[1]), sys.argv[2:]
out.mkdir(parents=True, exist_ok=True)
t0, peak = time.perf_counter(), 0.0
with open(out / "stderr.log", "w") as err:
    p = subprocess.Popen([sys.executable, "-m", "fusion_miniapp.run", *args, "--out", str(out)],
                         stdout=subprocess.PIPE, stderr=err, text=True)
    while p.poll() is None:
        peak = max(peak, tree_rss_mb(p.pid))
        time.sleep(1.0)
    stdout = p.stdout.read().strip()
res = {"args": args, "exit": p.returncode, "outer_wall_s": round(time.perf_counter() - t0, 1),
       "tree_rss_peak_mb_sampled_1s": round(peak, 1),
       "summary": json.loads(stdout.splitlines()[-1]) if stdout else None}
(out / "ladder.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
