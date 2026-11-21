"""The mini-app's campaigns are the parent's outer loop exactly (the plan, package 2).

For each seed, the unmodified parent driver (tests/parent_oracle.py: run_evolutionary_loop, or
evo_agents' three Academy agents for outer.rng: per_agent) and one mini-app campaign in each agent layer
(generator: mutation) must write identical evolution_log.csv, evolution_log.jsonl (timestamps aside) and
best_result.json. The CSV holds every generation's candidates (all parameters), their results and which
survived, so this also checks the candidates and the survivors.

Seeds 0-2 only ever take the parent's all-infeasible branch; fixture seed 99 also reaches a feasible
candidate (feasible_core), as Codex found.
"""
import asyncio
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


def read(d: Path) -> dict:
    jsonl = [json.loads(line) for line in (d / "evolution_log.jsonl").read_text().splitlines()]
    for row in jsonl:
        row.pop("timestamp")
    return {"csv": (d / "evolution_log.csv").read_text(), "jsonl": jsonl,
            "best": json.loads((d / "best_result.json").read_text())}


CASES = [("fixture", s, "shared") for s in (0, 1, 2, 99)] + [("fixture", s, "per_agent") for s in (0, 99)] \
    + [("real", s, "shared") for s in (0, 1, 2)] + [("real", 0, "per_agent")]


@pytest.mark.parametrize("mode,seed,rng", CASES)
def test_campaign_is_the_parent_loop(tmp_path, mode, seed, rng):
    knobs = oracle_knobs(rng)
    want = parent_files(tmp_path, mode, seed, rng, knobs)
    assert want["csv"].count("\n") == 1 + 4 * 3           # header + P x G candidates
    for layer in LAYERS:
        got = miniapp_files(tmp_path, layer, mode, seed, knobs)
        assert got["csv"] == want["csv"], layer
        assert got["jsonl"] == want["jsonl"], layer
        assert got["best"] == want["best"], layer
    if (mode, seed) == ("fixture", 99):
        assert want["best"]["selection_basis"] == "feasible_core"
