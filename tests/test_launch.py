"""Check the site launchers without a scheduler. srun is a recording fixture."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fusion_miniapp import knobs

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("site,gpu_request", [
    ("perlmutter", "--gpus-per-task=1"), ("stellar_ai", "--gres=gpu:1"), ("lux", "--gpus-per-task=1"),
])
def test_batch_launch(site, gpu_request, tmp_path):
    scratch = tmp_path / "scratch with spaces"
    scratch.mkdir()
    marker = tmp_path / "steps.jsonl"
    srun = tmp_path / "srun"
    srun.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
args = sys.argv[1:]
with open(os.environ["STEP_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if "probe" in args:
    with open(args[args.index("--out") + 1], "w") as f:
        json.dump({"tmpdir": os.environ.get("TMPDIR")}, f)
    print("scheduler prologue")
    sys.exit(int(os.environ.get("PROBE_EXIT", 0)))
""")
    srun.chmod(0o755)
    (tmp_path / "module").write_text("#!/bin/sh\nexit ${MODULE_EXIT:-0}\n")
    (tmp_path / "module").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "FUSION_MINIAPP_DIR": str(ROOT),
           "FUSION_ENV": sys.prefix, "PSCRATCH": str(scratch), "FUSION_SCRATCH": str(scratch),
           "MEMBERWORK": str(scratch), "SLURM_JOB_ACCOUNT": "TEST", "SLURM_JOB_ID": "23",
           "SLURM_JOB_NAME": "test", "SLURM_CPUS_PER_TASK": "32", "SLURM_RESTART_COUNT": "0",
           "SLURM_TMPDIR": str(tmp_path / "node scratch"), "STEP_LOG": str(marker)}
    user_knobs = tmp_path / "custom knobs.yaml"
    user_knobs.write_text("sim:\n  device: cpu\n")
    script = ROOT / "launch" / f"{site}.sbatch"
    directives = {line.removeprefix("#SBATCH ") for line in script.read_text().splitlines()
                  if line.startswith("#SBATCH ")}
    assert "--signal=TERM@120" in directives
    assert ("--gpus=1" if site == "lux" else gpu_request) in directives
    if site == "perlmutter":
        assert {"--constraint=gpu", "--qos=shared", "--cpus-per-task=32"} <= directives
    cmd = ["bash", str(script), "--preset", "ignos_shape", "--knobs", str(user_knobs)]
    done = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    probe, run = [json.loads(line) for line in marker.read_text().splitlines()]
    assert gpu_request in probe and gpu_request in run
    base = scratch / "test" if site == "lux" else scratch
    out = base / "fusion-miniapp" / "test-23"
    assert probe[-4:] == ["probe", "--require-gpu", "--out", str(out / "probe.json")]
    observed = json.loads((out / "probe.json").read_text())
    assert observed["tmpdir"] == (env["SLURM_TMPDIR"] if site == "lux" else env.get("TMPDIR"))
    machine_file = ROOT / "knobs/machines" / f"{site}.yaml"
    assert run[-8:] == ["--preset", "ignos_shape", "--knobs", str(user_knobs),
                        "--knobs", str(machine_file), "--out", str(out / "run")]
    merged = knobs.load(preset="ignos_shape", paths=[user_knobs, machine_file])
    assert merged["sim"]["device"] == "cuda" and merged["outer"]["population"] == 16
    if site == "lux":
        assert (tmp_path / "node scratch/miopen/kernels").is_dir()
        assert (tmp_path / "node scratch/miopen/db").is_dir()

    # Early failures must never start the application step.
    marker.unlink()
    bad = subprocess.run(cmd, cwd=tmp_path, env={**env, "MODULE_EXIT": "2"}, capture_output=True)
    assert bad.returncode != 0 and not marker.exists()
    refused = subprocess.run(cmd, cwd=tmp_path, env={**env, "SLURM_JOB_ID": "24", "PROBE_EXIT": "1"},
                             capture_output=True)
    assert refused.returncode != 0 and len(marker.read_text().splitlines()) == 1


def test_cpu_probe_writes_evidence_and_refuses_required_gpu(tmp_path):
    # Use this exact interpreter and emulate a venv activation banner/hook.
    activate = tmp_path / "bin/activate"
    activate.parent.mkdir()
    activate.write_text('echo "activating $OPTIONAL_TEST_HOOK"\nexport PATH="$TEST_PYTHON_BIN:$PATH"\n')
    env = {**os.environ, "FUSION_ENV": str(tmp_path), "TEST_PYTHON_BIN": str(Path(sys.executable).parent),
           "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
    cmd = [str(ROOT / "launch/run.sh"), "probe"]
    dest = tmp_path / "probe.json"
    p = subprocess.run(cmd + ["--require-gpu", "--out", str(dest)], env=env, capture_output=True, text=True, timeout=45)
    assert p.returncode != 0 and "requires a visible GPU" in p.stderr
    assert p.stdout == ""
    data = json.loads(dest.read_text())
    assert data["cuda_available"] is False and data["forward"]["finite"] is True
    assert data["forward"]["conv_out"] == [[2, 2, 6, 6], "cpu"]
    assert data["threads"]["torch"] >= 1
    assert data["gpu_management"] is None
