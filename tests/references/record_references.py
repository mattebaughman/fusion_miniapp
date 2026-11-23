"""Record the references tests/test_layer_autonmi.py compares the AUTONMI layer with: the LangGraph layer's
runs (python -m fusion_miniapp.run) of that file's cases (REFERENCES), from the code of one commit.

    python tests/references/record_references.py [--commit REV | --working-tree] [--out DIR] [--work DIR]

Run it in the AUTONMI layer's environment (requirements-autonmi.txt), where those tests run.

The commit (default HEAD) is exported with `git archive` into a temporary folder, and each run imports
fusion_miniapp from there, so the runs are that commit's code, whatever the working tree holds
(--working-tree: the installed code instead, with its commit and whether tracked files differ). The cases
and the digests of their output (test_layer_autonmi: REFERENCES, parent_hashes, campaign_hashes;
record.comparable; compare_records.EXACT_COUNTS) come from this script's own checkout, which the header
names too ("helpers"); a count the commit's records lack is left out.

Each case gives DIR/<name>.json (default: this folder): a header saying what produced it (the LangGraph
layer, the commit, the command, the date, the machine) and what the tests compare: results_sha256 and each
campaign's sha256, the exact counts, the sha256 of each campaign's parent files, the real case's campaigns
(record.comparable, compared within a tolerance where another OS or BLAS build gives other last digits)
and the model-server case's request bodies, sorted. `reference_sha256` is the sha256 of all but the
header: two recordings of one commit must give the same. The model server is the tests' FIXTURE
(fake_vllm.FakeVllm), not a model; all physics output is synthetic, and the fixture cases simulate nothing.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))   # tests/: the cases, their comparisons and the model server stand-in

import test_layer_autonmi as cases  # noqa: E402
from fake_vllm import FakeVllm  # noqa: E402

from fusion_miniapp import compare_records, record  # noqa: E402


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def export(rev: str, into: Path) -> dict:
    """The commit's tree in `into` (git archive); what it is."""
    commit = git("rev-parse", "--verify", f"{rev}^{{commit}}")
    data = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", commit], capture_output=True,
                          check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(into, filter="data")
    return {"commit": commit, "subject": git("log", "-1", "--format=%s", commit), "dirty": False,
            "how": "git archive of the commit, run with PYTHONPATH=<export>/src"}


def run_case(name: str, case: dict, work: Path, env: dict) -> tuple[dict, dict]:
    """One case on the LangGraph layer in work/<name>: the reference's body, and its command."""
    folder = work / name
    folder.mkdir(parents=True)
    knobs_file = folder / "knobs.yaml"
    knobs_file.write_text(json.dumps(case["knobs"]))   # JSON is YAML
    out = folder / "langgraph"
    cmd = [sys.executable, "-m", "fusion_miniapp.run", *case["args"], "--knobs", str(knobs_file), "--out", str(out)]
    with FakeVllm() if case.get("server") else contextlib.nullcontext() as server:
        if server is not None:
            env = {**env, "OPENAI_BASE_URL": f"http://127.0.0.1:{server.port}/v1"}
        proc = subprocess.run(cmd, cwd=folder, env=env, capture_output=True, text=True, timeout=900, check=False)
        sent = server.take() if server is not None else []
    (folder / "stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"{name}: exit {proc.returncode}\n{proc.stderr[-3000:]}")
    rec = json.loads((out / "run_record.json").read_text())
    body = {"name": name, "knobs": case["knobs"], "args": list(case["args"]), "status": rec["status"],
            "results_sha256": rec["results_sha256"], "campaign_sha256": cases.campaign_hashes(out),
            "counts": {k: rec["counts"][k] for k in compare_records.EXACT_COUNTS if k in rec["counts"]}}
    assert hashlib.sha256("\n".join(sorted(body["campaign_sha256"].values())).encode()).hexdigest() == \
        rec["results_sha256"], f"{name}: the campaigns' hashes do not make the record's results_sha256"
    if case.get("real"):
        lines = [json.loads(x) for x in (out / "agents.jsonl").read_text().splitlines()]
        body["campaigns"] = sorted((record.comparable(r) for r in lines), key=lambda r: r["agent_id"])
    if (out / "campaigns").is_dir():
        body["parent_files"] = cases.parent_hashes(out)
    if server is not None:
        body["requests"] = [json.loads(b) for b in cases.request_bodies(sent)]
    shown = [("<knobs.yaml>" if c == str(knobs_file) else "<out>" if c == str(out) else c) for c in cmd[1:]]
    env_shown = "OPENAI_BASE_URL=<the FakeVllm fixture>/v1 " if server is not None else ""
    command = env_shown + "python " + " ".join(shown)
    return body, {"command": command, "threads": rec.get("threads"),
                  "versions": rec.get("versions"), "miniapp_git": rec["run"]["miniapp_git"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = p.add_mutually_exclusive_group()
    which.add_argument("--commit", default="HEAD", help="the commit whose code records them (default HEAD)")
    which.add_argument("--working-tree", action="store_true", help="the installed code instead of an export")
    p.add_argument("--out", default=str(HERE), help="where the <name>.json files go (default: this folder)")
    p.add_argument("--work", help="keep the runs in this folder (default: a temporary one, removed)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="fusion_references_"))
    work.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        if a.working_tree:
            code = {**record.git_state(), "how": "the installed fusion_miniapp (the working tree)"}
        else:
            src = work / "export"
            code = export(a.commit, src)
            env["PYTHONPATH"] = str(src / "src")
        imported = Path(subprocess.run([sys.executable, "-c", "import fusion_miniapp; print(fusion_miniapp.__file__)"],
                                       env=env, cwd=work, capture_output=True, text=True, check=True).stdout.strip())
        base = ROOT if a.working_tree else work / "export"
        if not imported.is_relative_to(base):
            raise SystemExit(f"the runs would import fusion_miniapp from {imported}, not from {base}")
        code["imported"] = ("" if a.working_tree else "<export>/") + str(imported.relative_to(base))
        recorder = "python tests/references/record_references.py " + (
            "--working-tree" if a.working_tree else f"--commit {a.commit}")
        helpers = {"commit": git("rev-parse", "HEAD"),
                   "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                   "how": "the cases and digests (test_layer_autonmi, record.comparable, compare_records.EXACT_COUNTS) "
                          "from the recorder's own checkout"}
        for name, case in cases.REFERENCES.items():
            t0 = time.perf_counter()
            body, how = run_case(name, case, work, env)
            ref = {"recorded_from": "the LangGraph layer (python -m fusion_miniapp.run), dev's frozen comparison",
                   "header": {"code": code, "helpers": helpers, "command": how["command"], "recorder": recorder,
                              "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "machine": {"platform": platform.platform(), "python": platform.python_version(),
                                          "cpus": os.cpu_count()},
                              "threads_per_sim_worker": how["threads"], "versions": how["versions"],
                              "run_miniapp_git": how["miniapp_git"],
                              "note": "all physics output is synthetic; the fixture cases simulate nothing; the "
                                      "model server is a test FIXTURE, not a model"},
                   "reference_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                   **body}
            (out / f"{name}.json").write_text(json.dumps(ref, indent=1, sort_keys=False) + "\n")
            print(f"{name}: reference_sha256 {ref['reference_sha256']} results_sha256 {body['results_sha256']} "
                  f"({time.perf_counter() - t0:.1f} s)", flush=True)
    finally:
        if not a.work:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
