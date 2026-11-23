"""The mini-app's campaigns are the parent's outer loop exactly (the plan, package 2).

For each seed, the unmodified parent driver (tests/parent_oracle.py: run_evolutionary_loop, or
evo_agents' three Academy agents for outer.rng: per_agent) and one mini-app campaign in each agent layer
(generator: mutation) must write identical evolution_log.csv, evolution_log.jsonl (timestamps aside) and
best_result.json. The CSV holds every generation's candidates (all parameters), their results and which
survived, so this also checks the candidates and the survivors. The AUTONMI layer runs the same cases
through its own entry point, one campaign a<seed> (agents.first = seed); without the AUTONMI framework
those cases are skipped.

Seeds 0-2 only ever take the parent's all-infeasible branch; fixture seed 99 also reaches a feasible
candidate (feasible_core), as Codex found.
"""
import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fusion_miniapp import knobs as knob_files, run

HERE = Path(__file__).resolve().parent
LAYERS = ("langgraph", "academy")


def oracle_knobs(rng: str) -> dict:
    return knob_files.validate({
        "agents": {"count": 1, "per_process": 1},
        "sim": {"device": "cpu", "threads_per_worker": 1},
        "workers": {"sim_workers": 1},
        "outer": {"population": 4, "generations": 3, "generator": "mutation", "rng": rng, "logs": "per_agent",
                  "parent_stdout": "drop"}})


def parent_files(tmp: Path, mode: str, seed: int, rng: str, knobs: dict) -> dict:
    out = tmp / f"parent-{mode}-{seed}-{rng}"
    out.mkdir()
    subprocess.run([sys.executable, str(HERE / "parent_oracle.py"), str(out), mode, str(seed), rng, json.dumps(knobs)],
                   check=True, timeout=600)
    return read(Path((out / "run_dir.txt").read_text()))


def miniapp_files(tmp: Path, layer: str, mode: str, seed: int, knobs: dict) -> dict:
    out = tmp / f"{layer}-{mode}-{seed}"
    out.mkdir()
    results, _ = asyncio.run(run.run_slice(layer, knobs, mode, out, seed, 1, 0, 1))
    assert len(results) == 1 and "error" not in results[0], results
    return read(out / "campaigns" / f"a{seed}")


def autonmi_files(tmp: Path, mode: str, seed: int, knobs: dict) -> dict:
    """One campaign, a<seed> with seed `seed`, through python -m fusion_miniapp.layer_autonmi: a process of
    its own, started in tmp (AsyncFlow leaves a folder there)."""
    path = tmp / f"knobs-{mode}-{seed}.yaml"
    path.write_text(json.dumps({**knobs, "agents": {**knobs["agents"], "first": seed}}))   # JSON is YAML
    out = tmp / f"autonmi-{mode}-{seed}"
    proc = subprocess.run([sys.executable, "-m", "fusion_miniapp.layer_autonmi", "--knobs", str(path), "--sim", mode,
                           "--out", str(out)], cwd=tmp, capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rec = json.loads((out / "run_record.json").read_text())
    assert (rec["status"], rec["counts"]["agents"], rec["autonmi"]["ok"]) == ("pass", 1, True), rec["autonmi"]
    assert [json.loads(x)["agent_id"] for x in (out / "agents.jsonl").read_text().splitlines()] == [f"a{seed}"]
    return read(out / "campaigns" / f"a{seed}")


def autonmi_missing() -> str | None:
    """What the AUTONMI layer lacks here (as tests/test_layer_autonmi.py checks), or None."""
    try:
        for module in ("autonmi", "wfMiniAPI.api", "radical.asyncflow"):
            if importlib.util.find_spec(module) is None:
                return module
        from flowgentic.host import current_agent  # noqa: F401  (Flowgentic 0.6)
    except ImportError as exc:
        return str(exc)
    return None


def read(d: Path) -> dict:
    jsonl = [json.loads(line) for line in (d / "evolution_log.jsonl").read_text().splitlines()]
    for row in jsonl:
        row.pop("timestamp")
    return {"csv": (d / "evolution_log.csv").read_text(), "jsonl": jsonl,
            "best": json.loads((d / "best_result.json").read_text())}


CASES = [("fixture", s, "shared") for s in (0, 1, 2, 99)] + [("fixture", s, "per_agent") for s in (0, 99)] \
    + [("real", s, "shared") for s in (0, 1, 2)] + [("real", 0, "per_agent")]


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory):
    """The parent driver's files for a case, run once for both tests below, and checked to be what the case
    is for: P x G candidates, and for fixture seed 99 the feasible branch."""
    done = {}

    def get(mode: str, seed: int, rng: str) -> dict:
        if (mode, seed, rng) not in done:
            want = parent_files(tmp_path_factory.mktemp("parent"), mode, seed, rng, oracle_knobs(rng))
            assert want["csv"].count("\n") == 1 + 4 * 3           # header + P x G candidates
            if (mode, seed) == ("fixture", 99):
                assert want["best"]["selection_basis"] == "feasible_core"
            done[mode, seed, rng] = want
        return done[mode, seed, rng]
    return get


@pytest.mark.parametrize("mode,seed,rng", CASES)
def test_campaign_is_the_parent_loop(tmp_path, parent_run, mode, seed, rng):
    knobs = oracle_knobs(rng)
    want = parent_run(mode, seed, rng)
    for layer in LAYERS:
        got = miniapp_files(tmp_path, layer, mode, seed, knobs)
        assert got["csv"] == want["csv"], layer
        assert got["jsonl"] == want["jsonl"], layer
        assert got["best"] == want["best"], layer


@pytest.mark.parametrize("mode,seed,rng", CASES)
def test_campaign_on_autonmi_is_the_parent_loop(tmp_path, parent_run, mode, seed, rng):
    missing = autonmi_missing()
    if missing:
        pytest.skip(f"needs the AUTONMI layer's environment (requirements-autonmi.txt): {missing}")
    want = parent_run(mode, seed, rng)
    got = autonmi_files(tmp_path, mode, seed, oracle_knobs(rng))
    assert got["csv"] == want["csv"]
    assert got["jsonl"] == want["jsonl"]
    assert got["best"] == want["best"]
