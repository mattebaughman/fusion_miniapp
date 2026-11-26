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


# --- the StellarAI run path: launch/stellar_ai_autonmi.sbatch, stellar_ai_model.env, probe_model.py -------------
# Laptop checks only, with FIXTURES for Slurm, the GPUs and the model servers: nothing here runs on the machine.

import importlib.util
import re
import signal
import stat
import time

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
STELLAR = ROOT / "launch" / "stellar_ai_autonmi.sbatch"
MODEL_ENV = ROOT / "launch" / "stellar_ai_model.env"
MACHINE_KNOBS = [ROOT / "knobs/machines/stellar_ai.yaml", ROOT / "knobs/machines/stellar_ai_nemotron.yaml"]
AUTONMI = pytest.mark.skipif(importlib.util.find_spec("autonmi") is None or importlib.util.find_spec("aiohttp") is None,
                             reason="needs the AUTONMI layer's environment (requirements-autonmi.txt and autonmi)")


def test_the_stellarai_knob_file_loads_with_its_settings():
    """knobs/machines/stellar_ai_nemotron.yaml over the hundredk preset and stellar_ai.yaml: the asynchronous mode
    with 2 Monitor slots (5 hosted agents a campaign, 100,000 at the scale run's 20,000 campaigns), thinking off, a
    reply cap of 4,096, no per-call time limit in practice (86,400 s), a model server waited out for 300 s,
    NVIDIA's sampling, the client's admission caps out of reach, one GPU per inner-loop worker; the model's name
    is the one the env file serves. knobs/hundredk.yaml itself sets none of these."""
    k = knobs.validate(knobs.load(preset="hundredk", paths=MACHINE_KNOBS), modes=("generations", "async"))
    assert (k["agents"]["count"], k["campaign"]["mode"], k["campaign"]["slots"]) == (20000, "async", 2)
    assert k["agents"]["count"] * (3 + k["campaign"]["slots"]) == 100000
    assert k["llm"] | {} == {**k["llm"], "mode": "openai", "model": NEMOTRON, "thinking": "off", "reply_cap": 4096,
                             "timeout_s": 86400, "outage_s": 300, "temperature": 1.0, "top_p": 0.95,
                             "max_in_flight": 1000000, "admission_tokens": 0, "send_thinking_budget": False}
    assert k["workers"]["gpu_per_worker"] and k["sim"]["device"] == "cuda" and k["sim"]["work_root"] == "$TMPDIR"
    assert (k["outer"]["population"], k["outer"]["generations"]) == (16, 2)
    plain = knobs.load(preset="hundredk")
    assert (plain["llm"]["mode"], plain["campaign"]["mode"], plain["llm"]["outage_s"]) == ("stub", "generations", 1800)
    served = re.search(r"^RASCAL_MODEL_NAME=(\S+)$", MODEL_ENV.read_text(), re.M).group(1)
    assert served == k["llm"]["model"]


def test_the_model_jobs_env_file_sets_the_serving_flags_and_no_secret(tmp_path):
    """launch/stellar_ai_model.env, sourced under `set -u` as AUTONMI's jobs/local.env sources it: 3 restarts per
    server, the serving flags (at most 4,096 requests running per server, FP8 KV cache, no prefix caching, CUDA
    graphs to 4,096) as words that model.sh's unquoted expansion keeps whole (the graphs' JSON one word), the
    offline variables, the vLLM environment first on PATH. It needs FUSION_SCRATCH and FUSION_ENV, and it sets no
    variable whose name says it holds a secret."""
    if importlib.util.find_spec("autonmi") is None:   # the copy's own rule, layer_autonmi.SECRET_NAME
        secret = re.compile(r"(^|[_.-])(API_?KEYS?|KEYS?|TOKEN|SECRETS?|PASSWORDS?|PASSWD|CREDENTIALS?)($|[_.-])", re.I)
    else:
        from fusion_miniapp.layer_autonmi import SECRET_NAME as secret
    script = ('set -u; source "$1"; set -- $RASCAL_VLLM_ARGS; printf "%s\\n" "$@"; echo "--"; '
              'echo "$RASCAL_VLLM_RESTARTS|$RASCAL_MODEL|$RASCAL_VENV|${PATH%%:*}|$HF_HUB_OFFLINE|$TMPDIR"')
    env = {"PATH": os.environ["PATH"], "FUSION_SCRATCH": str(tmp_path / "scratch"), "FUSION_ENV": str(tmp_path / "venv")}
    done = subprocess.run(["bash", "-c", script, "-", str(MODEL_ENV)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    words, rest = done.stdout.split("--\n")
    words = words.split()
    assert words[words.index("--max-num-seqs") + 1] == "4096" and "--no-enable-prefix-caching" in words
    assert words[words.index("--kv-cache-dtype") + 1] == "fp8"
    graphs = [w for w in words if w.startswith("--compilation-config=")]
    assert len(graphs) == 1
    config = json.loads(graphs[0].split("=", 1)[1])
    assert config["cudagraph_capture_sizes"][-1] == 4096 and config["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert rest.strip() == "|".join(["3", f"{tmp_path}/scratch/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                                     str(tmp_path / "venv"), f"{tmp_path}/scratch/envs/vllm/bin", "1", "/tmp"])
    unset = subprocess.run(["bash", "-c", 'set -u; source "$1"', "-", str(MODEL_ENV)], env={"PATH": env["PATH"]},
                           capture_output=True, text=True)
    assert unset.returncode != 0 and "FUSION_SCRATCH" in unset.stderr
    names = re.findall(r"^(?:export )?([A-Z_][A-Z0-9_]*)=", MODEL_ENV.read_text(), re.M)
    names += [n for line in re.findall(r"^export (.+)$", MODEL_ENV.read_text(), re.M)
              for n in re.findall(r"([A-Z_][A-Z0-9_]*)=", line)]
    assert names and not [n for n in names if secret.search(n)]


# The assignments of AUTONMI's jobs/local.env.example (commit e20861a), a FIXTURE copy so that the test needs no
# AUTONMI checkout: its RASCAL_VENV and RASCAL_MODEL are placeholders.
LOCAL_ENV_EXAMPLE = """RASCAL_VENV=/path/to/venv
RASCAL_OVERLAY=
RASCAL_RUNS=/path/to/runs
RASCAL_MODEL=${RASCAL_MODEL:-/path/to/Qwen3-8B}
RASCAL_CUDART=
SLURM_ACCOUNT=
SLURM_CPU_PARTITION=
SLURM_GPU_PARTITION=
SLURM_GPU_GRES=gpu:1
"""


def test_the_env_file_sourced_last_in_local_env_wins_over_the_examples_placeholders(tmp_path):
    """The README's one-time setup: AUTONMI's jobs/local.env copied from its local.env.example, the Slurm values
    set, and the line that sources launch/stellar_ai_model.env last; sourced as jobs/submit and model.sh source it
    (set -euo pipefail). The model job gets the campaign venv and the Nemotron weights on scratch, not the
    example's placeholders, also with a RASCAL_MODEL left in the environment; FUSION_MODEL_PATH names other
    weights."""
    local = tmp_path / "local.env"
    local.write_text(LOCAL_ENV_EXAMPLE + "SLURM_ACCOUNT=x\nSLURM_GPU_PARTITION=gpu\nSLURM_GPU_GRES=gpu:8\n"
                     'source "$FUSION_MINIAPP_DIR/launch/stellar_ai_model.env"\n')
    env = {"PATH": os.environ["PATH"], "FUSION_MINIAPP_DIR": str(ROOT), "FUSION_SCRATCH": "/scratch/me",
           "FUSION_ENV": "/scratch/me/venv"}

    def sourced(**extra) -> list[str]:
        script = 'set -euo pipefail; source "$1"; echo "$RASCAL_MODEL|$RASCAL_VENV|$SLURM_GPU_GRES"'
        done = subprocess.run(["bash", "-c", script, "-", str(local)], env={**env, **extra}, capture_output=True,
                              text=True)
        assert done.returncode == 0, done.stderr
        return done.stdout.strip().split("|")

    weights = "/scratch/me/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    assert sourced() == sourced(RASCAL_MODEL="/old/Qwen3-8B") == [weights, "/scratch/me/venv", "gpu:8"]
    assert sourced(FUSION_MODEL_PATH="/scratch/me/other")[0] == "/scratch/me/other"


def test_the_stellarai_launch_script_parses_and_asks_slurm_for_a_whole_node():
    """bash -n; the job asks for a whole 8-GPU node (--mem in MB, since StellarAI rejects --mem=0) and for SIGTERM
    to the batch shell 15 minutes before its time limit; its name has no "dragon" (AUTONMI's guard); and the whole
    script is one function read before it runs (`main "$@"; exit`), so an edit during a run cannot reach it."""
    assert subprocess.run(["bash", "-n", str(STELLAR)]).returncode == 0
    text = STELLAR.read_text()
    directives = {line.removeprefix("#SBATCH ") for line in text.splitlines() if line.startswith("#SBATCH ")}
    assert {"--signal=B:TERM@900", "--gres=gpu:8", "--exclusive", "--mem=950000M", "--nodes=1"} <= directives
    assert "--time=01:00:00" in directives   # one node's 8 servers start in about 16 minutes, the warning at 15 before
    assert not any(d.startswith("--mem=0") for d in directives)
    name = next(d.split("=", 1)[1] for d in directives if d.startswith("--job-name="))
    assert "dragon" not in name
    assert text.rstrip().splitlines()[-1] == 'main "$@"; exit'


def stellar_env(tmp_path, job: str, **extra) -> dict:
    """The launch script's environment on the laptop: this checkout, a venv whose activation puts this Python
    first, a scratch folder, a fake AUTONMI checkout (FIXTURE), and FIXTURE tools first on PATH."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "activate").write_text(f'export PATH="{Path(sys.executable).parent}:$PATH"\n')
    (tmp_path / "autonmi" / "jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "autonmi" / "jobs" / "local.env").write_text("")
    return {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "FUSION_MINIAPP_DIR": str(ROOT),
            "FUSION_ENV": str(tmp_path / "venv"), "FUSION_SCRATCH": str(tmp_path / "scratch"),
            "RASCAL_REPO": str(tmp_path / "autonmi"), "SLURM_JOB_ID": job, "SLURM_JOB_NAME": "test",
            "TOOL_LOG": str(tmp_path / "tools.log"), **extra}


def fixture_tool(tmp_path, name: str, body: str) -> None:
    """A FIXTURE command on the launch script's PATH; each call is logged with its stdin."""
    tool = tmp_path / "bin" / name
    tool.parent.mkdir(exist_ok=True)
    tool.write_text(f"#!{sys.executable}\n" + "import json, os, sys\n"
                    "stdin = '' if sys.stdin.isatty() else sys.stdin.read()\n"
                    "with open(os.environ['TOOL_LOG'], 'a') as f:\n"
                    f"    f.write(json.dumps([{name!r}, *sys.argv[1:], stdin]) + '\\n')\n" + body)
    tool.chmod(0o755)


def tool_calls(tmp_path) -> list[list[str]]:
    log = tmp_path / "tools.log"
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


DRY = {"this node": {}, "four model nodes": {"FUSION_MODEL_NODES": "4", "FUSION_MODEL_TIME": "03:00:00"},
       "an earlier fleet": {"FUSION_MODEL_FLEET": "/scratch/earlier/fleet"}}


@pytest.mark.parametrize("how", DRY)
def test_a_dry_run_prints_the_launch_and_runs_nothing(tmp_path, how):
    """FUSION_DRY_RUN=1, without Slurm: each command is printed and none is run (every Slurm, GPU and MPS tool
    on PATH fails if called), no folder is made. The servers on this node: model.sh, one fleet folder; on four
    model nodes: four model jobs through AUTONMI's jobs/submit with the model jobs' size, an MPS daemon for the
    inner loop, quit at the end; reusing an earlier fleet: no model job, no stop file, and the check asks for idle
    servers. Always: a key (never printed), the setup-file list, launch/probe_model.py with the run's own
    arguments, then the run with the hundredk preset, the two StellarAI knob files, the script's arguments after
    them, --llm openai, the fleet's endpoints and key files, --model-wait 3600 and a new --out; then the stop
    files when this launch started the servers, the GPU check and the node-local folder removed. The run folder is
    named for FUSION_RUN_NAME, else the job name; launch.pid is not written in a dry run."""
    for name in ("sbatch", "srun", "squeue", "scancel", "nvidia-smi", "nvidia-cuda-mps-control"):
        fixture_tool(tmp_path, name, "sys.exit(99)\n")
    env = stellar_env(tmp_path, "41", FUSION_DRY_RUN="1", **DRY[how])
    done = subprocess.run(["bash", str(STELLAR), "--agents", "100", "--sim", "fixture"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert tool_calls(tmp_path) == [] and not (tmp_path / "scratch").exists()
    out = done.stdout
    run_dir = f"{tmp_path}/scratch/fusion-miniapp/test-41"
    fleet = DRY[how].get("FUSION_MODEL_FLEET", f"{run_dir}/fleet")
    dry = [line.removeprefix("[dry run] ").rstrip() for line in out.splitlines() if line.startswith("[dry run] ")]
    started = how != "an earlier fleet"
    assert sum("jobs/model.sh" in x for x in dry) == (how == "this node")
    submits = [x for x in dry if "jobs/submit model" in x]
    assert len(submits) == (4 if how == "four model nodes" else 0)
    for k, x in enumerate(submits):
        assert x.split()[2:] == [f"{fleet}/model-{k}", "--parsable", "--cpus-per-task=96", "--mem=950000M",
                                 "--exclusive", "--time=03:00:00", "-J", "fusion-model"]
    assert ("nvidia-cuda-mps-control -d" in out) == (how != "this node")
    assert ("echo quit | nvidia-cuda-mps-control" in out) == (how != "this node")
    assert (f"write a new key to {fleet}/api-key" in out) == started
    assert (f"write {fleet}/setup-files" in out) == started
    assert sum(x.startswith("touch ") and x.endswith("/stop") for x in dry) == {"this node": 1, "four model nodes": 4,
                                                                              "an earlier fleet": 0}[how]
    probe = next(x.split() for x in dry if "probe_model.py" in x)
    run = next(x.split() for x in dry if "-m fusion_miniapp.layer_autonmi" in x)
    assert probe[:4] == ["python", f"{ROOT}/launch/probe_model.py", "--logs", fleet]
    assert ("--idle" in probe) == (how == "an earlier fleet")
    assert run[:3] == ["python", "-m", "fusion_miniapp.layer_autonmi"] and probe[-len(run) + 3:] == run[3:]
    assert run[3:] == ["--preset", "hundredk", "--knobs", str(MACHINE_KNOBS[0]), "--knobs", str(MACHINE_KNOBS[1]),
                       "--agents", "100", "--sim", "fixture", "--llm", "openai", "--endpoints-file", f"{fleet}/endpoints",
                       "--api-key-file", f"{fleet}/api-key", "--model-wait", "3600", "--out", f"{run_dir}/run"]
    assert any(x.startswith("nvidia-smi --query-compute-apps") for x in dry)
    assert any(x == f"rm -rf /tmp/fusion-{os.environ.get('USER', 'user')}-test-41" for x in dry)
    both = subprocess.run(["bash", str(STELLAR)], cwd=tmp_path, env={**env, "FUSION_MODEL_NODES": "2",
                                                                       "FUSION_MODEL_FLEET": fleet},
                          capture_output=True, text=True, timeout=60)
    assert both.returncode == 2 and "not both" in both.stderr
    if how == "this node":   # a dry run needs no AUTONMI checkout; a launch without its scratch folder does not start
        bare = {k: v for k, v in env.items() if k != "RASCAL_REPO"}
        dry_bare = subprocess.run(["bash", str(STELLAR)], cwd=tmp_path, env=bare, capture_output=True, text=True,
                                  timeout=60)
        assert dry_bare.returncode == 0 and "bash \\$RASCAL_REPO/jobs/model.sh" in dry_bare.stdout, dry_bare.stderr
        unset = subprocess.run(["bash", str(STELLAR)], cwd=tmp_path, env={k: v for k, v in env.items()
                                                                           if k != "FUSION_SCRATCH"},
                               capture_output=True, text=True, timeout=60)
        assert unset.returncode != 0 and "FUSION_SCRATCH" in unset.stderr
        for named, end in (({"FUSION_RUN_NAME": "a"}, "a-7"), ({}, "x-7")):   # the run name, else the job name
            step = subprocess.run(["bash", str(STELLAR)], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                                  env={**env, "SLURM_JOB_ID": "7", "SLURM_JOB_NAME": "x", **named})
            run_dir = f"{tmp_path}/scratch/fusion-miniapp/{end}"
            assert step.returncode == 0 and f"run folder {run_dir};" in step.stdout, step.stderr
            assert f"[dry run] write this script's PID to {run_dir}/launch.pid" in step.stdout
            assert f"rm -rf /tmp/fusion-{os.environ.get('USER', 'user')}-{end}" in step.stdout
        assert tool_calls(tmp_path) == [] and not (tmp_path / "scratch").exists()


def test_a_dry_run_under_dragon_prints_dragons_launcher(tmp_path):
    """FUSION_RUNTIME=dragon (on a reused fleet, as the runs on StellarAI were), in a dry run: the run is the layer's
    script under Dragon's launcher in its single-node mode, `python -m dragon -s <checkout>/src/fusion_miniapp/
    layer_autonmi.py`, with the local run's arguments and --runtime dragon after them, which the check gets too;
    DRAGON_DEFAULT_SEG_SZ is 64 GiB unless set; dragon-cleanup runs at the end, before the MPS daemon quits. The
    job name keeps no "dragon". FUSION_RUNTIME must be local or dragon. Not in a dry run, a run folder that exists is
    refused before anything starts. Nothing runs: no laptop test reaches Dragon."""
    for name in ("sbatch", "srun", "squeue", "scancel", "nvidia-smi", "nvidia-cuda-mps-control", "dragon-cleanup"):
        fixture_tool(tmp_path, name, "sys.exit(99)\n")
    fleet = "/scratch/earlier/fleet"
    env = stellar_env(tmp_path, "42", FUSION_DRY_RUN="1", FUSION_MODEL_FLEET=fleet, FUSION_RUNTIME="dragon")
    env.pop("DRAGON_DEFAULT_SEG_SZ", None)

    def dry_run(**extra) -> tuple[str, list[str]]:
        done = subprocess.run(["bash", str(STELLAR), "--agents", "100", "--sim", "fixture"], cwd=tmp_path,
                              env={**env, **extra}, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout, [x.removeprefix("[dry run] ").rstrip() for x in done.stdout.splitlines()
                             if x.startswith("[dry run] ")]
    out, dry = dry_run()
    assert tool_calls(tmp_path) == [] and not (tmp_path / "scratch").exists()
    run_dir = f"{tmp_path}/scratch/fusion-miniapp/test-42"
    probe = next(x.split() for x in dry if "probe_model.py" in x)
    run = next(x.split() for x in dry if "-m dragon" in x)
    assert run[:4] == ["python", "-m", "dragon", "-s"] and run[4] == f"{ROOT}/src/fusion_miniapp/layer_autonmi.py"
    assert run[5:] == ["--preset", "hundredk", "--knobs", str(MACHINE_KNOBS[0]), "--knobs", str(MACHINE_KNOBS[1]),
                       "--agents", "100", "--sim", "fixture", "--llm", "openai", "--endpoints-file", f"{fleet}/endpoints",
                       "--api-key-file", f"{fleet}/api-key", "--model-wait", "3600", "--out", f"{run_dir}/run",
                       "--runtime", "dragon"]
    assert probe[-len(run) + 5:] == run[5:] and "--idle" in probe
    assert not any("-m fusion_miniapp.layer_autonmi" in x for x in dry)
    assert f"under Dragon: DRAGON_DEFAULT_SEG_SZ={64 * 1024 ** 3}" in out
    assert "under Dragon: DRAGON_DEFAULT_SEG_SZ=1234" in dry_run(DRAGON_DEFAULT_SEG_SZ="1234")[0]
    lines = out.splitlines()
    at = lambda text: next(i for i, x in enumerate(lines) if text in x)   # noqa: E731
    assert at("[dry run] timeout 120 dragon-cleanup") < at("echo quit | nvidia-cuda-mps-control") \
        < at("nvidia-smi --query-compute-apps")
    local = subprocess.run(["bash", str(STELLAR)], cwd=tmp_path, env={**env, "FUSION_RUNTIME": "local"},
                           capture_output=True, text=True, timeout=60)
    assert local.returncode == 0 and "dragon" not in local.stdout.replace(str(tmp_path), "")
    bad = subprocess.run(["bash", str(STELLAR)], cwd=tmp_path, env={**env, "FUSION_RUNTIME": "Dragon"},
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2 and "FUSION_RUNTIME is local or dragon" in bad.stderr
    Path(run_dir, "run").mkdir(parents=True)   # not a dry run: under Dragon the run folder must be new
    used = subprocess.run(["bash", str(STELLAR)], cwd=tmp_path, env={**env, "FUSION_DRY_RUN": "0"},
                          capture_output=True, text=True, timeout=60)
    assert used.returncode == 2 and "under Dragon the run folder must be new" in used.stderr
    assert tool_calls(tmp_path) == [] and "ended with status" not in used.stdout   # before the teardown is set


def test_a_dry_run_of_several_launches_prints_one_run_per_launch_then_the_merge(tmp_path):
    """FUSION_LAUNCHES=2 in a dry run (on a reused fleet): the fleet's endpoints split into one block per launch, the
    check once on the whole fleet with --slice 0/2, one MPS daemon, then two runs, launch i with the run's arguments
    and after them --slice i/2, its own endpoints file launch-<i>/endpoints and --out run-<i>, then one merge of
    run-0 and run-1 into run. Nothing is made. FUSION_LAUNCHES=1 prints what no FUSION_LAUNCHES does. Refused (exit 2,
    before the teardown is set): a value that is not a number of 1 or more, FUSION_RUNTIME=dragon with 2 launches,
    and (not in a dry run) a run folder run or run-<i> that exists."""
    for name in ("sbatch", "srun", "squeue", "scancel", "nvidia-smi", "nvidia-cuda-mps-control"):
        fixture_tool(tmp_path, name, "sys.exit(99)\n")
    fleet = "/scratch/earlier/fleet"
    env = stellar_env(tmp_path, "43", FUSION_DRY_RUN="1", FUSION_MODEL_FLEET=fleet, FUSION_LAUNCHES="2")
    args = ["bash", str(STELLAR), "--agents", "12", "--sim", "fixture"]
    done = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert tool_calls(tmp_path) == [] and not (tmp_path / "scratch").exists()
    run_dir = f"{tmp_path}/scratch/fusion-miniapp/test-43"
    dry = [x.removeprefix("[dry run] ").rstrip() for x in done.stdout.splitlines() if x.startswith("[dry run] ")]
    assert f"split {fleet}/endpoints into 2 blocks in order" in done.stdout
    probes = [x.split() for x in dry if "probe_model.py" in x]
    runs = [x.split() for x in dry if "-m fusion_miniapp.layer_autonmi" in x]
    merges = [x.split() for x in dry if "-m fusion_miniapp.merge_records" in x]
    assert len(probes) == 1 and len(runs) == 2 and len(merges) == 1
    assert probes[0][:7] == ["python", f"{ROOT}/launch/probe_model.py", "--logs", fleet, "--idle", "--slice", "0/2"]
    one = ["--preset", "hundredk", "--knobs", str(MACHINE_KNOBS[0]), "--knobs", str(MACHINE_KNOBS[1]), "--agents", "12",
           "--sim", "fixture", "--llm", "openai", "--endpoints-file", f"{fleet}/endpoints", "--api-key-file",
           f"{fleet}/api-key", "--model-wait", "3600", "--out", f"{run_dir}/run"]
    assert probes[0][7:] == one
    for i, run in enumerate(runs):
        assert run == ["python", "-m", "fusion_miniapp.layer_autonmi", *one, "--slice", f"{i}/2", "--endpoints-file",
                       f"{run_dir}/launch-{i}/endpoints", "--out", f"{run_dir}/run-{i}"]
    assert merges[0][3:] == [f"{run_dir}/run-0", f"{run_dir}/run-1", "--out", f"{run_dir}/run"]
    assert sum("nvidia-cuda-mps-control -d" in x for x in dry) == 1
    assert done.stdout.count("echo quit | nvidia-cuda-mps-control") == 1

    def stamped(out: str) -> str:   # each line without its time
        return re.sub(r"^\[fusion-autonmi [^]]*\]", "", out, flags=re.M)
    without = {k: v for k, v in env.items() if k != "FUSION_LAUNCHES"}
    plain, single = (subprocess.run(args, cwd=tmp_path, env=e, capture_output=True, text=True, timeout=60)
                     for e in (without, {**env, "FUSION_LAUNCHES": "1"}))
    assert plain.returncode == single.returncode == 0 and stamped(single.stdout) == stamped(plain.stdout)
    assert "--slice" not in plain.stdout and "merge_records" not in plain.stdout
    for bad, why in ((dict(FUSION_LAUNCHES="0"), "a number of launches"),
                     (dict(FUSION_LAUNCHES="two"), "a number of launches"),
                     (dict(FUSION_RUNTIME="dragon"), "one Dragon runtime per node")):
        refused = subprocess.run(args, cwd=tmp_path, env={**env, **bad}, capture_output=True, text=True, timeout=60)
        assert refused.returncode == 2 and why in refused.stderr and "ended with status" not in refused.stdout
    for used in ("run", "run-1"):   # not a dry run: the records merged must be this launch's
        Path(run_dir, used).mkdir(parents=True)
        refused = subprocess.run(args, cwd=tmp_path, env={**env, "FUSION_DRY_RUN": "0"}, capture_output=True,
                                 text=True, timeout=60)
        assert refused.returncode == 2 and f"{run_dir}/{used} exists" in refused.stderr
        assert "ended with status" not in refused.stdout and tool_calls(tmp_path) == []
        Path(run_dir, used).rmdir()


def fake_submit(tmp_path) -> None:
    """AUTONMI's jobs/submit as a FIXTURE: `submit model FOLDER ...` is a model job that has started at once: it
    writes the folder's endpoints (one fake vLLM server's URL each, from FAKE_URLS in order), gpu.csv, a server log
    that holds the key it was given (as a careless server might print it), its job output, and a file the copy
    must not take (a bare FAKE secret under a name that says nothing); it prints its job id, as sbatch
    --parsable does."""
    submit = tmp_path / "autonmi" / "jobs" / "submit"
    submit.parent.mkdir(parents=True, exist_ok=True)
    submit.write_text(f"#!{sys.executable}\n" + """import json, os, sys
from pathlib import Path
with open(os.environ["TOOL_LOG"], "a") as f:
    f.write(json.dumps(["submit", *sys.argv[1:]]) + "\\n")
folder = Path(sys.argv[2])
k = int(folder.name.rsplit("-", 1)[1])
job = 9100 + k
(folder / "endpoints").write_text(os.environ["FAKE_URLS"].split(",")[k] + "\\n")
(folder / "gpu.csv").write_text("2026/11/24 00:00:00.000, 0, 97, 150000\\n")
key = (folder / "api-key").read_text().strip()
(folder / "vllm-0.log").write_text("".join(f"INFO {i}\\n" for i in range(700)).replace("INFO 3\\n", f"key {key}\\n"))
(folder / f"model-{job}.out").write_text(f"model job {job}\\n")
(folder / "notes.txt").write_text("planted-bare-secret-in-an-unnamed-file-7q\\n")
print(job)
""")
    submit.chmod(0o755)


USER_KNOBS = {"outer": {"population": 4, "generations": 2, "parent_stdout": "drop"}, "record": {"extras": True},
              "workers": {"sim_workers": 2}}


def stellar_run(tmp_path, servers: int, job: str, *, latency_s: float = 0.0, **extra):
    """The launch script on the laptop against `servers` fake vLLM servers (FIXTURES serving the model's name), one
    model job each (fake_submit), 2 fixture campaigns of 8 candidates with record.extras; a Popen, the servers and
    the script's environment (with `extra`)."""
    from fake_vllm import FakeVllm

    fake_submit(tmp_path)
    fixture_tool(tmp_path, "squeue", "print('RUNNING')\n")
    fixture_tool(tmp_path, "scancel", "")
    fixture_tool(tmp_path, "nvidia-smi", "")
    fixture_tool(tmp_path, "nvidia-cuda-mps-control", "")
    fakes = [FakeVllm(model=NEMOTRON, latency_s=latency_s).__enter__() for _ in range(servers)]
    env = stellar_env(tmp_path, job, FUSION_MODEL_NODES=str(servers), FUSION_MODEL_TIME="00:20:00",
                      FAKE_URLS=",".join(f.url for f in fakes), **extra)
    user = tmp_path / "user knobs.yaml"
    user.write_text(json.dumps(USER_KNOBS))
    p = subprocess.Popen(["bash", str(STELLAR), "--agents", "2", "--sim", "fixture", "--knobs", str(user)],
                         cwd=tmp_path, env=env, stdout=open(tmp_path / "job.out", "w"), stderr=subprocess.STDOUT)
    return p, fakes, env


@AUTONMI
def test_the_launch_against_fake_servers_on_two_model_nodes(tmp_path):
    """The whole launch on the laptop with FIXTURES for Slurm, the GPUs and two model nodes' servers (fake vLLM
    servers on localhost, one per fake model job): one key for both jobs (owner only, never printed); the
    combined endpoints file in the jobs' order; the setup-file list; the check (launch/probe_model.py) reports both
    servers; the run passes with the StellarAI settings on every request and the script's knob file applied last;
    its record's copy of the model servers' setup files takes each job's listed files and nothing else, so
    neither the key nor the bare FAKE secret of a file the list does not name reaches any file of the run; then
    each job's stop file, the MPS daemon started for the inner loop and quit, the GPUs checked, the node-local
    folder removed, and the script exits 0."""
    job = f"77{os.getpid()}"
    p, fakes, env = stellar_run(tmp_path, 2, job)
    try:
        assert p.wait(timeout=600) == 0, (tmp_path / "job.out").read_text()[-4000:]
    finally:
        for f in fakes:
            f.__exit__(None, None, None)
    log = (tmp_path / "job.out").read_text()
    run_dir = tmp_path / "scratch" / "fusion-miniapp" / f"test-{job}"
    fleet = run_dir / "fleet"
    key = (fleet / "api-key").read_text().strip()
    assert len(key) == 48 and stat.S_IMODE((fleet / "api-key").stat().st_mode) == 0o600
    assert all((fleet / f"model-{k}" / "api-key").read_text().strip() == key for k in range(2))
    assert key not in log
    assert (fleet / "endpoints").read_text() == "".join(f.url + "\n" for f in fakes)
    assert (fleet / "setup-files").read_text().split() == ["endpoints"] + [
        f"model-{k}/{name}" for k in range(2) for name in ("endpoints", "gpu.csv", "vllm-*.log", "model-*.out")]
    calls = tool_calls(tmp_path)
    assert [c[:3] for c in calls if c[0] == "submit"] == [["submit", "model", str(fleet / f"model-{k}")] for k in range(2)]
    mps = [c for c in calls if c[0] == "nvidia-cuda-mps-control"]
    assert [c[1:] for c in mps] == [["-d", ""], ["quit\n"]]
    assert all((fleet / f"model-{k}" / "stop").exists() for k in range(2))
    assert not Path(f"/tmp/fusion-{os.environ.get('USER', 'user')}-test-{job}").exists()
    probed = [json.loads(x) for x in log.splitlines() if x.startswith('{"server"')]
    assert [x["server"] for x in probed] == [f.url for f in fakes] and all(x["problems"] == [] for x in probed)
    assert all([c["role"] for c in x["calls"]] == ["generator", "monitor", "evaluator", "coordinator"] for x in probed)
    rec = json.loads((run_dir / "run" / "run_record.json").read_text())
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    k = rec["knobs"]
    assert (k["agents"]["count"], k["outer"]["population"], k["campaign"]["mode"], k["llm"]["reply_cap"]) == (2, 4, "async", 4096)
    for body in [b for f in fakes for b in f.bodies]:
        assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["model"] == NEMOTRON
        assert (body["max_tokens"], body["temperature"], body["top_p"]) == (4096, 1.0, 0.95)
    copied = rec["llm_service"]["model_service"]
    assert copied["files"] == sorted(["endpoints", "environment.txt"] + [
        f"model-{j}/{name}" for j in range(2) for name in ("endpoints", "gpu.csv", "vllm-0.log", f"model-910{j}.out")])
    assert copied["removed_by_the_check"] == [] and copied["secrets_left_out"] == 2   # the key, in each job's log
    for f in (run_dir / "run").rglob("*"):   # every file of the run (fleet/ holds the model jobs' own)
        if f.is_file():
            data = f.read_bytes()
            assert key.encode() not in data and b"planted-bare-secret" not in data, f
    head = (run_dir / "run" / "model_service" / "model-1" / "vllm-0.log").read_text().splitlines()
    assert len(head) == 500 and head[3] == "key [secret left out]"


@AUTONMI
def test_sigterm_to_the_launch_reaches_the_run_and_releases_the_servers(tmp_path):
    """Slurm's warning before the time limit (SIGTERM to the batch shell, --signal=B:TERM@900) while the run is
    under way, its fake server (a FIXTURE) answering each call in 1 s: the script sends SIGTERM to the run's PID
    alone, the run writes its interrupted record and exits 143, and the script still writes the model job's stop
    file and quits the MPS daemon, and exits 143. The script's PID, the one signalled, is in <run folder>/launch.pid."""
    job = f"78{os.getpid()}"
    p, fakes, env = stellar_run(tmp_path, 1, job, latency_s=1.0)
    run = tmp_path / "scratch" / "fusion-miniapp" / f"test-{job}" / "run"
    try:
        deadline = time.time() + 300
        while not list(run.glob("agents-p*.jsonl")):   # the campaigns are being built: the run's handler is set
            assert p.poll() is None and time.time() < deadline, (tmp_path / "job.out").read_text()[-4000:]
            time.sleep(0.2)
        time.sleep(1.0)
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=120) == 143, (tmp_path / "job.out").read_text()[-4000:]
    finally:
        for f in fakes:
            f.__exit__(None, None, None)
    log = (tmp_path / "job.out").read_text()
    assert "signal: SIGTERM to the run" in log and "the run exited with status 143" in log
    assert (run.parent / "launch.pid").read_text() == f"{p.pid}\n"   # the PID signalled, as kill -TERM would
    assert json.loads((run / "run_record.json").read_text())["status"] == "interrupted"
    assert (run.parent / "fleet" / "model-0" / "stop").exists()
    assert [c[1:] for c in tool_calls(tmp_path) if c[0] == "nvidia-cuda-mps-control"] == [["-d", ""], ["quit\n"]]


@AUTONMI
@pytest.mark.parametrize("interrupt", [False, True], ids=["complete", "sigterm"])
def test_several_launches_against_fake_servers_on_two_model_nodes(tmp_path, interrupt):
    """FUSION_LAUNCHES=2 on two fake model nodes (FIXTURES, as above), --agents 2: launch i runs campaign a<i> alone
    (--slice i/2) against server i alone (launch-<i>/endpoints, one block of the fleet's, in order), each into its
    own run folder run-<i>, under the one MPS daemon of the script (one start, one quit); then one record of the
    two in run, by merge_records. Complete: both pass, so does the merged record, and the script exits 0.
    `sigterm`: Slurm's warning to the script while both run reaches each launch, both records are interrupted, so
    is the merged one, and the script exits 143. Either way each model job's stop file is written."""
    job = f"80{os.getpid()}{int(interrupt)}"
    p, fakes, env = stellar_run(tmp_path, 2, job, latency_s=1.0 if interrupt else 0.0, FUSION_LAUNCHES="2")
    base = tmp_path / "scratch" / "fusion-miniapp" / f"test-{job}"
    dirs = [base / f"run-{i}" for i in range(2)]
    try:
        if interrupt:
            deadline = time.time() + 300
            while not all(list(d.glob("agents-p*.jsonl")) for d in dirs):   # both launches' handlers are set
                assert p.poll() is None and time.time() < deadline, (tmp_path / "job.out").read_text()[-4000:]
                time.sleep(0.2)
            time.sleep(1.0)
            p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=600) == (143 if interrupt else 0), (tmp_path / "job.out").read_text()[-4000:]
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=60)
        for f in fakes:
            f.__exit__(None, None, None)
    log = (tmp_path / "job.out").read_text()
    status = "interrupted" if interrupt else "pass"
    for i, d in enumerate(dirs):
        rec = json.loads((d / "run_record.json").read_text())
        assert rec["status"] == status, (i, rec["status"], log[-4000:])
        assert (rec["knobs"]["agents"]["first"], rec["knobs"]["agents"]["count"]) == (i, 1)
        assert (base / f"launch-{i}" / "endpoints").read_text() == fakes[i].url + "\n"
        assert (base / f"launch-{i}" / "setup-files").read_text() == "endpoints\n"
        if not interrupt:
            assert [json.loads(x)["agent_id"] for x in (d / "agents.jsonl").read_text().splitlines()] == [f"a{i}"]
            assert rec["llm_service"]["model_service"]["files"] == ["endpoints", "environment.txt"]
        assert (base / "fleet" / f"model-{i}" / "stop").exists()
        assert f"launch {i}/2 exited with status {143 if interrupt else 0}" in log
    assert ("signal: SIGTERM to the run (" in log) == interrupt
    merged = json.loads((base / "run" / "run_record.json").read_text())
    assert merged["status"] == status and merged["knobs"]["agents"]["count"] == 2
    assert [(m["agents"]["first"], m["status"]) for m in merged["merged_from"]] == [(0, status), (1, status)]
    assert "were not merged" not in log and f"ended with status {143 if interrupt else 0}" in log
    assert [c[1:] for c in tool_calls(tmp_path) if c[0] == "nvidia-cuda-mps-control"] == [["-d", ""], ["quit\n"]]


# Dragon's launcher as a FIXTURE package (`python -m dragon -s SCRIPT ARGS`), since Dragon cannot run on a laptop:
# it runs SCRIPT as a child process with --runtime dragon removed (the layer then runs locally), writes the child's
# PID to <--out>/head.pid as the layer itself does under Dragon (test_layer_autonmi checks that write; here once
# --out holds run.lock, or a file matching FAKE_DRAGON_HEAD_AFTER), never
# forwards a signal (it logs one it gets), and exits FAKE_DRAGON_EXIT, or else as its child did.
FAKE_DRAGON = """import os, signal, subprocess, sys, time
from pathlib import Path
assert sys.argv[1] == "-s", sys.argv
script, args = sys.argv[2], sys.argv[3:]
at = args.index("--runtime")
assert args[at + 1] == "dragon", args
del args[at:at + 2]
out = Path(args[args.index("--out") + 1])
child = subprocess.Popen([sys.executable, script, *args])
signal.signal(signal.SIGTERM, lambda *_: open(os.environ["FAKE_DRAGON_LOG"], "a").write("launcher: SIGTERM\\n"))
while not list(out.glob(os.environ.get("FAKE_DRAGON_HEAD_AFTER", "run.lock"))) and child.poll() is None:
    time.sleep(0.05)
(out / "head.pid").write_text(f"{child.pid}\\n")
rc = child.wait()
print(f"+++ head proc exited, code {rc}", flush=True)
sys.exit(int(os.environ.get("FAKE_DRAGON_EXIT", rc)))
"""


@AUTONMI
@pytest.mark.parametrize("when", ["running", "starting"])
def test_sigterm_under_dragon_reaches_the_layers_head_not_dragons_launcher(tmp_path, when):
    """FUSION_RUNTIME=dragon with a FIXTURE Dragon launcher (FAKE_DRAGON, first on PYTHONPATH) that exits 0 whatever
    happens: Slurm's warning while the run is under way reaches the layer's head process through head.pid, not the
    launcher; the run writes its interrupted record; the launch takes its exit status from that record (143), not
    from the launcher; and at the end dragon-cleanup runs (through `timeout`, a FIXTURE on PATH here) before the
    MPS daemon quits. `starting`: the signal comes once the launcher has started and before head.pid is there
    (FAKE_DRAGON writes it once the campaigns are being built); the script holds it and sends it to the head when
    the file appears. What Dragon itself does on a signal is not known: no laptop test reaches Dragon."""
    fake = tmp_path / "fake_dragon" / "dragon"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    (fake / "__main__.py").write_text(FAKE_DRAGON)
    fixture_tool(tmp_path, "dragon-cleanup", "")
    fixture_tool(tmp_path, "timeout", "import subprocess\nsys.exit(subprocess.call(sys.argv[2:]))\n")
    path = os.pathsep.join(x for x in (str(fake.parent), os.environ.get("PYTHONPATH")) if x)
    job = f"79{os.getpid()}"
    p, fakes, env = stellar_run(tmp_path, 1, job, latency_s=1.0, FUSION_RUNTIME="dragon", PYTHONPATH=path,
                                FAKE_DRAGON_LOG=str(tmp_path / "launcher.log"), FAKE_DRAGON_EXIT="0",
                                FAKE_DRAGON_HEAD_AFTER="agents-p*.jsonl" if when == "starting" else "run.lock")
    run = tmp_path / "scratch" / "fusion-miniapp" / f"test-{job}" / "run"
    try:
        deadline = time.time() + 300
        while not (list(run.glob("agents-p*.jsonl")) if when == "running" else   # the run's handler is set
                   "Dragon's launcher:" in (tmp_path / "job.out").read_text()):
            assert p.poll() is None and time.time() < deadline, (tmp_path / "job.out").read_text()[-4000:]
            time.sleep(0.2)
        if when == "running":
            time.sleep(1.0)
        else:
            assert not (run / "head.pid").exists()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=120) == 143, (tmp_path / "job.out").read_text()[-4000:]
    finally:
        for f in fakes:
            f.__exit__(None, None, None)
    log = (tmp_path / "job.out").read_text()
    head = (run / "head.pid").read_text().strip()
    launcher = re.search(r"Dragon's launcher: (\d+)", log).group(1)
    assert head != launcher and f"the run's PID: {head}" in log and f"signal: SIGTERM to the run ({head})" in log
    assert ("signal: held until the run's head process has written its PID" in log) == (when == "starting")
    assert not (tmp_path / "launcher.log").exists()   # the launcher got no signal
    assert "+++ head proc exited, code 143" in log and "the run exited with status 0" in log
    assert "its record gives status 143" in log and "ended with status 143" in log
    assert json.loads((run / "run_record.json").read_text())["status"] == "interrupted"
    calls = [c[0] for c in tool_calls(tmp_path)]
    assert calls.index("dragon-cleanup") < len(calls) - 1 - calls[::-1].index("nvidia-cuda-mps-control")
    assert [c for c in tool_calls(tmp_path) if c[0] in ("timeout", "dragon-cleanup")] == [
        ["timeout", "120", "dragon-cleanup", ""], ["dragon-cleanup", ""]]
    assert (run.parent / "fleet" / "model-0" / "stop").exists()


# Model jobs as processes (FIXTURES), for the paths that need servers to start late, end on their stop file, or
# outlive a launch: FAKE_JOB is one model job, a fake vLLM server in a process of its own (with the folder's key)
# that writes its endpoints after FAKE_START_S seconds (a port nobody listens on for the jobs FAKE_DOWN lists),
# answers in 3 s while the file FAKE_HOLD exists, and ends when its stop file appears.
FAKE_JOB = """import os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["FAKE_TESTS"])
from fake_vllm import FakeVllm
folder, job = Path(sys.argv[1]), sys.argv[2]
hold = os.environ.get("FAKE_HOLD", "")
with FakeVllm(model=os.environ["FAKE_MODEL"], api_key=(folder / "api-key").read_text().strip()) as server:
    (folder / "stop").unlink(missing_ok=True)   # as model.sh does at each start
    (folder / "gpu.csv").write_text("2026/11/24 00:00:00.000, 0, 97, 150000\\n")
    (folder / "vllm-0.log").write_text("INFO ready\\n")
    time.sleep(float(os.environ.get("FAKE_START_S", "0")))
    down = job in os.environ.get("FAKE_DOWN", "").split(",")
    (folder / "endpoints").write_text(("http://127.0.0.1:9" if down else server.url) + "\\n")
    print(f"model job {job}: serving", flush=True)
    while not (folder / "stop").exists():
        server.latency_s = 3.0 if hold and os.path.exists(hold) else 0.0
        time.sleep(0.1)
print(f"model job {job}: stop seen", flush=True)
(folder / "ended").write_text(job + "\\n")
"""

# jobs/submit model FOLDER ...: job 9200 + k for FOLDER model-<k>; it starts FAKE_JOB at once and returns once its
# endpoints are written, or with FAKE_STATES' k-th entry set the job stays in that state and never starts.
FAKE_JOB_SUBMIT = """import json, os, subprocess, sys, time
from pathlib import Path
with open(os.environ["TOOL_LOG"], "a") as f:
    f.write(json.dumps(["submit", *sys.argv[1:]]) + "\\n")
folder = Path(sys.argv[2])
k = int(folder.name.rsplit("-", 1)[1])
job, jobs = str(9200 + k), Path(os.environ["FAKE_JOBS"])
jobs.mkdir(exist_ok=True)
states = os.environ.get("FAKE_STATES", "").split(",")
if k < len(states) and states[k]:
    (jobs / job).write_text(states[k])
else:
    p = subprocess.Popen([sys.executable, os.environ["FAKE_JOB"], str(folder), job], start_new_session=True,
                         stdout=open(folder / f"model-{job}.out", "a"), stderr=subprocess.STDOUT)
    (jobs / job).write_text(f"pid {p.pid}")
    deadline = time.time() + 60
    while not (folder / "endpoints").exists() and p.poll() is None and time.time() < deadline:
        time.sleep(0.05)
print(job)
"""

# squeue -h -j ID -o %T: RUNNING while a started job's process lives, nothing once it ended, else its state.
FAKE_SQUEUE = """jobs = os.path.join(os.environ["FAKE_JOBS"], sys.argv[sys.argv.index("-j") + 1])
state = open(jobs).read() if os.path.exists(jobs) else ""
if state.startswith("pid "):
    try:
        os.kill(int(state[4:]), 0)
        print("RUNNING")
    except OSError:
        pass
elif state:
    print(state)
"""


def fake_jobs(d: Path, model_sh: str | None = None) -> dict:
    """FIXTURES for the launch script in folder d: model jobs as processes (FAKE_JOB, through a fake jobs/submit),
    with model_sh a fake jobs/model.sh (default: FAKE_JOB in the foreground, as the job's own); squeue, scancel,
    nvidia-smi and nvidia-cuda-mps-control. The variables they read, for stellar_env."""
    (d / "fake_job.py").write_text(FAKE_JOB)
    jobs = d / "autonmi" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "submit").write_text(f"#!{sys.executable}\n" + FAKE_JOB_SUBMIT)
    (jobs / "submit").chmod(0o755)
    own = 'exec "$FAKE_PYTHON" "$FAKE_JOB" "$1" "$SLURM_JOB_ID"\n'
    (jobs / "model.sh").write_text("#!/bin/bash\n" + (model_sh or own))
    fixture_tool(d, "squeue", FAKE_SQUEUE)
    for name in ("scancel", "nvidia-smi", "nvidia-cuda-mps-control"):
        fixture_tool(d, name, "")
    return {"FAKE_JOB": str(d / "fake_job.py"), "FAKE_JOBS": str(d / "jobs"), "FAKE_PYTHON": sys.executable,
            "FAKE_TESTS": str(Path(__file__).parent), "FAKE_MODEL": NEMOTRON}


def stop_fake_jobs(d: Path) -> None:
    """Ends every FIXTURE model job a test left serving: each sees its stop file, or is killed after 10 s."""
    for folder in d.glob("**/fleet/model-*"):
        (folder / "stop").touch()
    deadline = time.time() + 10
    for f in (d / "jobs").glob("*") if (d / "jobs").exists() else []:
        state = f.read_text()
        if state.startswith("pid "):
            pid = int(state[4:])
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


def launch(d: Path, env: dict, *args: str, **kw) -> subprocess.CompletedProcess:
    knob = d / "user knobs.yaml"
    knob.write_text(json.dumps(USER_KNOBS))
    return subprocess.run(["bash", str(STELLAR), "--agents", "2", "--sim", "fixture", "--knobs", str(knob), *args],
                          cwd=d, env=env, capture_output=True, text=True, timeout=300, **kw)


@AUTONMI
def test_the_wiring_check_on_one_node_waits_for_model_sh_and_ends_it(tmp_path):
    """FUSION_MODEL_NODES=0, the README's wiring check, with a FIXTURE jobs/model.sh (one fake vLLM server that
    writes its endpoints after 1 s and ends on its stop file): the launch waits for the endpoints, the check and the
    run pass; while the run is under way the job's endpoints file is rewritten, as a requeued job's would be, and
    fleet/endpoints follows it; at the end the stop file, model.sh ended before the launch exits, no MPS daemon
    (the servers have the GPUs), the node-local folder removed, exit 0, and no "Terminated" line in the output."""
    job = f"71{os.getpid()}"
    hold = tmp_path / "hold"   # while it exists, each reply takes 3 s: the run outlasts the refresh
    env = stellar_env(tmp_path, job, FAKE_START_S="1", FAKE_HOLD=str(hold), **fake_jobs(tmp_path))
    knob = tmp_path / "user knobs.yaml"
    knob.write_text(json.dumps(USER_KNOBS))
    p = subprocess.Popen(["bash", str(STELLAR), "--agents", "2", "--sim", "fixture", "--knobs", str(knob)],
                         cwd=tmp_path, env=env, stdout=open(tmp_path / "job.out", "w"), stderr=subprocess.STDOUT)
    run_dir = tmp_path / "scratch" / "fusion-miniapp" / f"test-{job}"
    fleet = run_dir / "fleet"
    try:
        deadline = time.time() + 120
        while "the run's PID" not in (tmp_path / "job.out").read_text():
            assert p.poll() is None and time.time() < deadline, (tmp_path / "job.out").read_text()[-4000:]
            time.sleep(0.1)
        hold.touch()
        requeued = (fleet / "model-0" / "endpoints").read_text().replace("127.0.0.1", "localhost")
        (fleet / "model-0" / "endpoints").write_text(requeued)
        while (fleet / "endpoints").read_text() != requeued:
            assert p.poll() is None and time.time() < deadline + 30, (tmp_path / "job.out").read_text()[-4000:]
            time.sleep(0.2)
        hold.unlink()
        assert p.wait(timeout=300) == 0, (tmp_path / "job.out").read_text()[-4000:]
    finally:
        if p.poll() is None:   # a failed check: Slurm's signal, so the run ends too
            p.terminate()
            p.wait(timeout=120)
        stop_fake_jobs(tmp_path)
    log = (tmp_path / "job.out").read_text()
    assert "Terminated" not in log and "model.sh ended" not in log
    rec = json.loads((run_dir / "run" / "run_record.json").read_text())
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    assert (fleet / "model-0" / "stop").exists() and (fleet / "model-0" / "ended").read_text().strip() == job
    assert (fleet / "setup-files").read_text().split() == ["endpoints", "model-0/endpoints", "model-0/gpu.csv",
                                                          "model-0/vllm-*.log", "model-0/model-*.out"]
    assert f"model job {job}: stop seen" in (fleet / "model-0" / f"model-{job}.out").read_text()
    calls = tool_calls(tmp_path)
    assert [c[0] for c in calls if c[0] in ("submit", "scancel", "nvidia-cuda-mps-control")] == []
    assert not Path(f"/tmp/fusion-{os.environ.get('USER', 'user')}-test-{job}").exists()


@AUTONMI
def test_a_launch_that_cannot_use_its_servers_stops_and_releases_them(tmp_path):
    """Launches that never reach the run, each exit 1 with its reason, and their teardown: model.sh (FIXTURE) that
    ends at once; two model jobs still PENDING and CONFIGURING when FUSION_MODEL_WAIT (1 s) runs out, both stop
    files written and both jobs cancelled (each would remove its stop file when it starts); two running model jobs
    (FIXTURES), one of whose servers is down, so the check fails and no run starts, both stop files written, both
    jobs end and neither is cancelled, and a server process still on the GPUs is warned about. And a step of the
    launch's own shell that fails under set -u once the teardown is set (an activation that unsets the script's
    variable `fleet`, before any server is started): exit 1, where bash 3.2 alone would give 0. Under
    FUSION_RUNTIME=dragon a launch that stops before the run started no Dragon, so runs no dragon-cleanup (a
    FIXTURE here)."""
    cases = {
        "model.sh ends": ({}, 'echo "vLLM 0 exited before it was ready"; exit 1\n'),
        "jobs not started": ({"FUSION_MODEL_NODES": "2", "FUSION_MODEL_WAIT": "1",
                              "FAKE_STATES": "PENDING,CONFIGURING"}, None),
        "a server down": ({"FUSION_MODEL_NODES": "2", "FAKE_DOWN": "9201"}, None),
        "unset variable": ({}, None),
        "model.sh ends, under Dragon": ({"FUSION_RUNTIME": "dragon"}, 'echo "vLLM 0 exited"; exit 1\n'),
    }
    done = {}
    for i, (case, (extra, model_sh)) in enumerate(cases.items()):
        d = tmp_path / f"case{i}"
        d.mkdir()
        env = stellar_env(d, f"72{i}", FUSION_MODEL_TIME="00:20:00", **fake_jobs(d, model_sh), **extra)
        if case == "unset variable":
            with open(d / "venv" / "bin" / "activate", "a") as f:
                f.write("unset fleet\n")
        for tool in ("dragon-cleanup", "timeout") if "FUSION_RUNTIME" in extra else ():
            fixture_tool(d, tool, "")
        if case == "a server down":   # a server process still on this node, of a launch that stops its servers
            fixture_tool(d, "nvidia-smi", "print('1, VLLM::EngineCore, 100 MiB')\n")
        try:
            done[case] = launch(d, env)
        finally:
            stop_fake_jobs(d)
        out = done[case].stdout + done[case].stderr
        assert done[case].returncode == 1 and "ended with status 1" in out, out[-3000:]
        fleet = d / "scratch" / "fusion-miniapp" / f"test-72{i}" / "fleet"
        n = 0 if case == "unset variable" else 2 if "FUSION_MODEL_NODES" in extra else 1
        assert all((fleet / f"model-{k}" / "stop").exists() for k in range(n)), case
        assert not (fleet.parent / "run").exists() and "the run's PID" not in out, case
        cancelled = [c[1] for c in tool_calls(d) if c[0] == "scancel"]
        assert cancelled == (["9200", "9201"] if case == "jobs not started" else []), case
        assert not Path(f"/tmp/fusion-{os.environ.get('USER', 'user')}-test-72{i}").exists()
        assert not [c for c in tool_calls(d) if c[0] in ("dragon-cleanup", "timeout")], case
    assert "model.sh ended before its servers were ready" in done["model.sh ends"].stderr
    assert "not every model job wrote its endpoints within 1 s" in done["jobs not started"].stderr
    down = done["a server down"].stderr
    assert "http://127.0.0.1:9: /v1/models did not answer" in down and "the model servers failed the check" in down
    assert "warning: still on the GPUs: 1, VLLM::EngineCore, 100 MiB" in down   # its servers are not left: warned
    fleet = tmp_path / "case2" / "scratch" / "fusion-miniapp" / "test-722" / "fleet"
    assert all((fleet / f"model-{k}" / "ended").exists() for k in range(2))
    assert "fleet: unbound variable" in done["unset variable"].stderr
    assert not (tmp_path / "case3" / "scratch" / "fusion-miniapp" / "test-723" / "fleet").exists()


# squeue -h -j ID -o %T as a FIXTURE answering from FAKE_ANSWERS in turn (the last again after the end): a state, or
# FAIL, squeue failing (nothing printed, exit 1), as when slurmctld does not answer in time.
SCRIPTED_SQUEUE = """count = os.path.join(os.environ["FAKE_JOBS"], "squeue-calls")
n = int(open(count).read()) if os.path.exists(count) else 0
open(count, "w").write(str(n + 1))
answers = os.environ["FAKE_ANSWERS"].split(",")
if answers[min(n, len(answers) - 1)] == "FAIL":
    sys.exit(1)
print(answers[min(n, len(answers) - 1)])
"""


def test_a_model_job_has_ended_only_on_an_end_state_or_after_three_answers_without_one(tmp_path):
    """One model job that never starts (FIXTURE, PENDING), against a FIXTURE squeue answering in turn: one failed
    squeue call does not end the launch's wait (FUSION_MODEL_WAIT, 1 s, does); an end state (FAILED) ends it at
    once, with FUSION_MODEL_WAIT's 60 s left; a job whose state squeue does not give at the end is cancelled, as
    one not yet serving. Each launch exits 1 before the run, with the job's stop file written."""
    cases = {
        "a failed answer": ("FAIL,PENDING", "1", "not every model job wrote its endpoints within 1 s"),
        "an end state": ("FAILED", "60", "model job 9200 ended (FAILED) before its servers were ready"),
        "no answer at the end": ("PENDING,FAIL", "1", "not every model job wrote its endpoints within 1 s"),
    }
    for i, (case, (answers, wait, reason)) in enumerate(cases.items()):
        d = tmp_path / f"case{i}"
        d.mkdir()
        env = stellar_env(d, f"73{i}", FUSION_MODEL_NODES="1", FUSION_MODEL_TIME="00:20:00", FUSION_MODEL_WAIT=wait,
                          FAKE_STATES="PENDING", FAKE_ANSWERS=answers, **fake_jobs(d))
        fixture_tool(d, "squeue", SCRIPTED_SQUEUE)
        t0 = time.time()
        done = launch(d, env)
        out = done.stdout + done.stderr
        assert done.returncode == 1 and reason in done.stderr and "the run's PID" not in out, (case, out[-3000:])
        assert time.time() - t0 < 40, case
        assert (d / "scratch" / "fusion-miniapp" / f"test-73{i}" / "fleet" / "model-0" / "stop").exists(), case
        assert [c[1] for c in tool_calls(d) if c[0] == "scancel"] == ["9200"], case


@AUTONMI
def test_servers_kept_by_one_launch_are_reused_by_the_next(tmp_path):
    """FUSION_KEEP_SERVERS=1 with one model job (FIXTURE): the run passes, and the launch leaves the job serving (no
    stop file, nothing cancelled). A second launch with FUSION_MODEL_FLEET set to that fleet folder checks that its
    servers are idle, submits no job, runs against them with the same key (the MPS daemon started and quit for its
    inner loop) and leaves them serving. Once the stop file is written the job ends, and a third launch on that
    fleet fails its check and starts no run. FUSION_KEEP_SERVERS=1 without FUSION_MODEL_NODES is refused, since
    model.sh on the launch's own node ends with its job. The GPU check at the end of a launch that leaves the
    servers running (kept or reused) counts their VLLM::EngineCore processes (a FIXTURE nvidia-smi) and warns only
    about any other process."""
    fake = fake_jobs(tmp_path)
    fixture_tool(tmp_path, "nvidia-smi", "print('1, VLLM::EngineCore, 100 MiB')\n")   # a server left on this node
    runs = tmp_path / "scratch" / "fusion-miniapp"
    fleet = runs / "test-731" / "fleet"
    try:
        kept = launch(tmp_path, stellar_env(tmp_path, "731", FUSION_MODEL_NODES="1", FUSION_MODEL_TIME="00:20:00",
                                            FUSION_KEEP_SERVERS="1", **fake))
        assert kept.returncode == 0, (kept.stdout + kept.stderr)[-3000:]
        assert f"kept for FUSION_MODEL_FLEET={fleet}" in kept.stdout
        assert not (fleet / "model-0" / "stop").exists() and not (fleet / "model-0" / "ended").exists()
        reused = launch(tmp_path, stellar_env(tmp_path, "732", FUSION_MODEL_FLEET=str(fleet), **fake))
        assert reused.returncode == 0, (reused.stdout + reused.stderr)[-3000:]
        for done in (kept, reused):   # the servers it leaves are counted, not warned about
            assert "warning: still on the GPUs" not in done.stderr
            assert "on the GPUs: only the model servers this launch leaves running (1)" in done.stdout
        for name in ("731", "732"):
            rec = json.loads((runs / f"test-{name}" / "run" / "run_record.json").read_text())
            assert rec["status"] == "pass" and rec["autonmi"]["ok"], (name, rec["autonmi"])
        assert not (fleet / "model-0" / "stop").exists() and not (fleet / "model-0" / "ended").exists()
        assert not (runs / "test-732" / "fleet").exists() and f"reusing the servers of {fleet}" in reused.stdout
        calls = tool_calls(tmp_path)
        assert [c[0] for c in calls].count("submit") == 1 and not [c for c in calls if c[0] == "scancel"]
        assert [c[1:] for c in calls if c[0] == "nvidia-cuda-mps-control"] == [["-d", ""], ["quit\n"]] * 2
        (fleet / "model-0" / "stop").touch()
        deadline = time.time() + 30
        while not (fleet / "model-0" / "ended").exists():
            assert time.time() < deadline
            time.sleep(0.1)
        fixture_tool(tmp_path, "nvidia-smi", "print('1, VLLM::EngineCore, 100 MiB\\n2, python, 10 MiB')\n")
        gone = launch(tmp_path, stellar_env(tmp_path, "733", FUSION_MODEL_FLEET=str(fleet), **fake))
        assert gone.returncode == 1 and "/v1/models did not answer" in gone.stderr, gone.stderr[-3000:]
        assert not (runs / "test-733" / "run").exists()
        warned = [x for x in gone.stderr.splitlines() if "still on the GPUs" in x]   # only the other process
        assert len(warned) == 1 and warned[0].endswith("warning: still on the GPUs: 2, python, 10 MiB"), gone.stderr
        assert "VLLM::EngineCore" not in gone.stderr
    finally:
        stop_fake_jobs(tmp_path)
    alone = launch(tmp_path, stellar_env(tmp_path, "734", FUSION_KEEP_SERVERS="1", **fake))
    assert alone.returncode == 2 and "FUSION_KEEP_SERVERS=1 needs FUSION_MODEL_NODES" in alone.stderr


def probe_model(tmp_path, servers, *extra, key: str | None = None):
    (tmp_path / "endpoints").write_text("".join(s + "\n" for s in servers))
    args = [sys.executable, str(ROOT / "launch/probe_model.py"), *extra, "--preset", "hundredk",
            *[x for f in MACHINE_KNOBS for x in ("--knobs", str(f))], "--llm", "openai",
            "--endpoints-file", str(tmp_path / "endpoints")]
    if key is not None:
        (tmp_path / "api-key").write_text(key + "\n")
        args += ["--api-key-file", str(tmp_path / "api-key")]
    return subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=tmp_path)


@AUTONMI
def test_probe_model_reports_both_servers_and_stops_the_launch_when_one_is_down(tmp_path):
    """launch/probe_model.py against two fake vLLM servers (FIXTURES) with a key: each server gets one request of
    each role with the run's settings (thinking off, the reply cap, temperature, top_p; the Generator's schema),
    and each reply is reported (usable, the Generator's 7 parameters filled for each of its 2 candidates); exit 0.
    Then the stops, each exit 1 with the server and the reason: one server down; one that serves another model;
    one whose context cannot hold a prompt and the reply cap; with --idle, one with requests running; a server
    log with a Traceback, and one with a Traceback after a restart. A server's own shutdown trace (the lines of a
    real one: after [shutdown], until the next start) does not stop the launch. The key is never printed."""
    from fake_vllm import FakeVllm

    key = "FAKE-probe-key-3c9d1e7a"
    with FakeVllm(model=NEMOTRON, api_key=key) as one, FakeVllm(model=NEMOTRON, api_key=key) as two:
        ok = probe_model(tmp_path, [one.url, two.url], key=key)
        assert ok.returncode == 0, ok.stderr
        reports = [json.loads(x) for x in ok.stdout.splitlines()]
        assert [r["server"] for r in reports] == [one.url, two.url]
        for r in reports:
            assert r["problems"] == [] and r["max_model_len"] == 32768
            assert [c["role"] for c in r["calls"]] == ["generator", "monitor", "evaluator", "coordinator"]
            assert all(c["usable"] and c["finish_reason"] == "stop" and c["max_tokens"] == 4096 for c in r["calls"])
            assert r["calls"][0]["parameters_filled"] == [7, 7]
        for server in (one, two):
            bodies = server.take()
            assert len(bodies) == 4 and sum("response_format" in b for b in bodies) == 1
            assert all(b["chat_template_kwargs"] == {"enable_thinking": False} and b["temperature"] == 1.0 and
                       b["top_p"] == 0.95 and b["max_tokens"] == 4096 and b["model"] == NEMOTRON for b in bodies)
        assert "2 servers answered" in ok.stderr
        down_url = "http://127.0.0.1:9"   # nothing listens there
        down = probe_model(tmp_path, [one.url, down_url], key=key)
        assert down.returncode == 1 and f"{down_url}: /v1/models did not answer" in down.stderr
        assert [json.loads(x)["problems"] == [] for x in down.stdout.splitlines()] == [True, False]
        one.running = 3   # as another run's requests would show on its /metrics
        busy = probe_model(tmp_path, [one.url, two.url], "--idle", key=key)
        assert busy.returncode == 1 and f"{one.url}: not idle: 3 requests running" in busy.stderr
        one.running = 0
        assert probe_model(tmp_path, [one.url, two.url], "--idle", key=key).returncode == 0
        (tmp_path / "model-0").mkdir()
        (tmp_path / "model-0" / "vllm-1.log").write_text("INFO ready\nTraceback (most recent call last):\n")
        logged = probe_model(tmp_path, [one.url], "--logs", str(tmp_path), key=key)
        assert logged.returncode == 1 and "vllm-1.log: line 2: Traceback" in logged.stderr
        # what a vLLM 0.27.1 server stopped by SIGTERM left in its log on StellarAI (2026-11-25): skipped, until the
        # next start (model.sh's "=== " line)
        stopped = ("(APIServer pid=2832910) INFO 11-25 14:07:46 [core_client.py:690] [shutdown] MPClient: engine "
                   "manager stopped\n"
                   "(APIServer pid=2832910) ERROR 11-25 14:07:46 [async_llm.py:724] Traceback (most recent call "
                   "last):\n"
                   "(APIServer pid=2832910) ERROR 11-25 14:07:46 [async_llm.py:724] vllm.v1.engine.exceptions."
                   "EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.\n")
        scans, start = {}, "=== start vllm 1\n"
        for case, text in (("stopped", stopped), ("restarted", stopped + start + "INFO ready\n"),
                           ("failed after a restart", stopped + start + "Traceback (most recent call last):\n")):
            (tmp_path / "model-0" / "vllm-1.log").write_text(text)
            scans[case] = probe_model(tmp_path, [one.url], "--logs", str(tmp_path), key=key)
        assert scans["stopped"].returncode == 0 and scans["restarted"].returncode == 0, scans["stopped"].stderr
        failed = scans["failed after a restart"]
        assert failed.returncode == 1 and "vllm-1.log: line 5: Traceback" in failed.stderr
        for done in (ok, down, busy, logged, *scans.values()):
            assert key not in done.stdout + done.stderr
    with FakeVllm(model="another-model") as other, FakeVllm(model=NEMOTRON, max_model_len=4096) as small:
        wrong = probe_model(tmp_path, [other.url, small.url])
        assert wrong.returncode == 1
        assert f"{other.url}: serves ['another-model'], not llm.model '{NEMOTRON}'" in wrong.stderr
        assert f"{small.url}: generator: " in wrong.stderr and "> max_model_len 4096" in wrong.stderr
        assert small.take() == []   # nothing too long is sent


@AUTONMI
def test_probe_model_stops_the_launch_on_an_http_error_an_empty_reply_or_a_time_out(tmp_path):
    """The stops a server that lists the model can still cause (as a server that answers its health check and then
    fails every request would): beside a good one, a fake vLLM server (FIXTURE) that answers every request with
    HTTP 400, one whose replies are empty, and one slower than --timeout (1 s); each exit 1, naming the server, each
    role and the reason, with the good server reported without problems."""
    from fake_vllm import FakeVllm

    cases = {"HTTP 400 Bad Request": {"fail": lambda body: True}, "an empty reply": {"reply": lambda content: ""},
             "no reply within 1 s": {"latency_s": 3.0}}
    for reason, kw in cases.items():
        with FakeVllm(model=NEMOTRON) as good, FakeVllm(model=NEMOTRON, **kw) as bad:
            done = probe_model(tmp_path, [good.url, bad.url], *(("--timeout", "1") if "within" in reason else ()))
        assert done.returncode == 1, reason
        for role in ("generator", "monitor", "evaluator", "coordinator"):
            assert f"{bad.url}: {role}: {reason}" in done.stderr, (reason, done.stderr)
        assert [json.loads(x)["problems"] == [] for x in done.stdout.splitlines()] == [True, False], reason
