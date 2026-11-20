"""Agent side: the two layers agree, the flag picks the layer, the run record's counts are right.

All with the fixture simulation and the stub LLM (or a recording stand-in); laptop timings only.
"""
import asyncio
import importlib
import json
import os
import subprocess
import sys

import pytest

from fusion_miniapp import decide, knobs as knob_files, record, run, sim as sim_module
from fusion_miniapp.knobs import KnobError
from fusion_miniapp.llm import StubLLM

LAYERS = ("langgraph", "academy")


def knobs_for(n, sims=2, tools=None, **agents):
    return knob_files.validate({"agents": {"count": n, "per_process": n, "sims_per_agent": sims, **agents},
                                "tools": tools or {}})


def run_layer(layer, knobs, out, model=None, sim="fixture", sim_fn=None, n=None):
    """One agent process on one layer, with the services run.py builds (and optional stand-ins for the
    LLM and the simulation function). n agents; default: the knobs' count."""
    async def go():
        svc = await run.Services.open(knobs, sim_fn or run.sim_function(sim), out, 0, 1 if sim == "real" else 2,
                                      processes=sim == "real", model=model)
        try:
            agents = importlib.import_module(f"fusion_miniapp.layer_{layer}").run_agents
            results = await agents(n or knobs["agents"]["count"], knobs, lambda: svc, out)
        finally:
            await svc.close()
        return results, svc
    out.mkdir(parents=True, exist_ok=True)
    return asyncio.run(go())


def test_layers_give_identical_results_for_50_agents(tmp_path):
    knobs = knobs_for(50)
    got = {layer: run_layer(layer, knobs, tmp_path / layer)[0] for layer in LAYERS}
    for results in got.values():
        assert len(results) == 50 and not [r for r in results if "error" in r]
        assert {r["agent_id"] for r in results} == {f"a{i}" for i in range(50)}
        assert all(r["n_sims"] == 2 and r["n_llm_calls"] == 3 for r in results)
    assert record.digest(got["langgraph"]) == record.digest(got["academy"])
    a0 = next(r for r in got["academy"] if r["agent_id"] == "a0")
    assert a0 == next(r for r in got["langgraph"] if r["agent_id"] == "a0")
    assert [h["source"] for h in a0["history"]] == ["llm", "llm"]          # the stub's JSON parsed


def test_real_evaluate_through_both_layers_on_cpu(tmp_path):
    """The turned-down inner loop itself (torch on the CPU), not the fixture, behind each layer."""
    knobs = knob_files.validate({"agents": {"count": 2, "per_process": 2, "sims_per_agent": 2},
                                 "sim": {"device": "cpu"}})
    got = {layer: run_layer(layer, knobs, tmp_path / layer, sim="real")[0] for layer in LAYERS}
    for results in got.values():
        hist = [h for r in results for h in r["history"]]
        assert len(hist) == 4 and not any(h["fixture"] for h in hist)
        assert {h["touch_check"] for h in hist} == {"pass"} and {h["device"] for h in hist} == {"cpu"}
    # touches differ only by the one cold start the first layer paid in this process
    def strip(rs):
        return sorted(({k: v for k, v in r.items() if k != "touches"} for r in rs), key=lambda r: r["agent_id"])
    assert strip(got["langgraph"]) == strip(got["academy"])


SEEN = []


def recording_fixture(candidate, knobs=None, agent_id="", seed=0):
    """The fixture, noting the simulation knobs each agent's evaluate received."""
    SEEN.append((agent_id, knobs["nodes_per_plane"], knobs["fixed_point_cap"]))
    return sim_module.evaluate_fixture(candidate, knobs, agent_id, seed)


def test_per_agent_cost_scale_sets_simulation_knobs_and_budget(tmp_path):
    knobs = knobs_for(20, sims=3, thinking_budget=100, cost_scale=[0.5, 1, 2], budget_scales_with_cost=True)
    want = {f"a{i}": [0.5, 1, 2][i % 3] for i in range(20)}             # agent i uses entry i % 3
    got = {}
    for layer in LAYERS:
        model = StubLLM()
        SEEN.clear()
        got[layer], _ = run_layer(layer, knobs, tmp_path / layer, model=model, sim_fn=recording_fixture)
        assert model.calls == 20 * (3 + 1)                                  # S + 1 calls per agent
        assert dict(model.budgets) == {50: 7 * 4, 100: 7 * 4, 200: 6 * 4}  # each agent's budget, on every call
        assert {r["agent_id"]: r["budget"] for r in got[layer]} == {a: round(100 * c) for a, c in want.items()}
        assert sorted(set(SEEN)) == sorted({(a, round(200 * c), round(3 * c)) for a, c in want.items()})
        assert len(SEEN) == 20 * 3
    assert record.digest(got["langgraph"]) == record.digest(got["academy"])


def test_best_so_far_is_a_best_not_the_last_candidate(tmp_path):
    """Feasible: the highest feasible core. Infeasible: no other candidate has both a higher core metric
    and a cooler wall (the parent's core/wall ranking, applied one candidate at a time)."""
    results, _ = run_layer("academy", knobs_for(60, sims=6), tmp_path)
    last = 0
    for r in results:
        b, hist = r["best"], r["history"]
        assert {k: b["params"][k] for k in decide.ranges()} == hist[b["candidate_idx"]]["candidate"]
        feasible = [h["te_core_metric"] for h in hist if h["is_good"]]
        if feasible:
            assert b["is_good"] and b["te_core_metric"] == max(feasible)
            continue
        wall = max(b["te_wall_eV"], b["ti_wall_eV"])
        assert not [h for h in hist if h["te_core_metric"] > b["te_core_metric"]
                    and max(h["te_wall_eV"], h["ti_wall_eV"]) < wall]
        last += b["candidate_idx"] == len(hist) - 1
    assert last < len(results) / 2                                          # not simply the latest candidate


class InFlight(StubLLM):
    """A stub with 20 ms of latency that counts the calls in flight at once."""

    def __init__(self):
        super().__init__(latency_s=0.02)
        self.now = self.peak = 0

    async def __call__(self, agent_id, messages, thinking_budget):
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            return await super().__call__(agent_id, messages, thinking_budget)
        finally:
            self.now -= 1


def test_both_layers_admit_the_same_share_of_model_calls(tmp_path):
    knobs = knobs_for(1000, sims=1)
    knobs["agents"]["per_process"] = 10                                 # this process's share: 512 x 1% = 5
    for layer in LAYERS:
        model = InFlight()
        results, _ = run_layer(layer, knobs, tmp_path / layer, model=model, n=50)
        assert len(results) == 50 and not [r for r in results if "error" in r]
        assert model.peak == 5, (layer, model.peak)


class NoJSON(StubLLM):
    """An LLM whose replies never parse."""

    async def __call__(self, agent_id, messages, thinking_budget):
        await super().__call__(agent_id, messages, thinking_budget)
        return "I would try a wider pedestal next."


def test_unparsable_reply_falls_back_to_the_parents_mutation(tmp_path):
    knobs = knobs_for(10, sims=3)
    got = {}
    for layer in LAYERS:
        got[layer], _ = run_layer(layer, knobs, tmp_path / layer, model=NoJSON())
        hist = [h for r in got[layer] for h in r["history"]]
        assert len(hist) == 30 and {h["source"] for h in hist} == {"mutation"}
        for h in hist:
            assert all(lo <= h["candidate"][k] <= hi for k, (lo, hi) in decide.ranges().items())
    assert record.digest(got["langgraph"]) == record.digest(got["academy"])  # the mutation is seeded


def test_counts_in_the_run_record_for_1000_agents_in_both_layers(tmp_path):
    (tmp_path / "k.yaml").write_text(json.dumps({"agents": {"count": 1000, "per_process": 1000, "sims_per_agent": 2}}))
    recs = {}
    for layer in LAYERS:
        flag = ["--use-academy"] if layer == "academy" else []
        assert run.main(["--knobs", str(tmp_path / "k.yaml"), "--sim", "fixture",
                         "--out", str(tmp_path / layer), *flag]) == 0
        recs[layer] = rec = json.loads((tmp_path / layer / "run_record.json").read_text())
        c = rec["counts"]
        assert rec["agent_layer"] == layer
        assert (c["agents"], c["agents_failed"], c["sims"], c["sim_tasks"]) == (1000, 0, 2000, 2000)
        assert c["llm_calls"] == rec["llm_service"]["calls"] == 1000 * 3
        assert rec["llm_service"]["budgets"] == {"64": 3000}
        assert c["tool_calls"] == 1000 * 2                                  # constraints, abstract
        assert rec["lookups"] == {"constraints": 1000, "abstract": 1000}
        assert c["proposals"] == {"llm": 2000, "mutation": 0}               # every stub reply parsed
        assert (c["agents_requested"], c["failed_agent_processes"]) == (1000, [])
        assert rec["emulated"] is True and "synthetic" in rec["note"]
        assert (c["memory_writes"], c["reports"]) == (2000, 1000)
        assert c["suspensions"] == (2000 if layer == "langgraph" else None)
        assert rec["touch_check"] == {"evaluations": 2000, "fixture": 2000, "pass": 0, "fail": {}}
        assert len((tmp_path / layer / "agents.jsonl").read_text().splitlines()) == 1000
    assert recs["langgraph"]["results_sha256"] == recs["academy"]["results_sha256"]


def missed_touch(candidate, knobs=None, agent_id="", seed=0):
    """A non-fixture result whose touch check failed."""
    from fusion_miniapp import sim
    return {**sim.evaluate_fixture(candidate, knobs, agent_id, seed), "fixture": False,
            "touch_check": "fail: mesh_read=0 (want 4)"}


def test_a_failed_touch_check_fails_the_run(tmp_path, monkeypatch):
    (tmp_path / "agents-7.jsonl").write_text('{"agent_id": "stale", "error": "from an earlier run"}\n')
    monkeypatch.setattr(run, "sim_function", lambda mode: missed_touch)
    assert run.main(["--agents", "3", "--sim", "fixture", "--out", str(tmp_path)]) == 1
    rec = json.loads((tmp_path / "run_record.json").read_text())
    assert rec["touch_check"] == {"evaluations": 6, "fixture": 0, "pass": 0, "fail": {"fail: mesh_read=0 (want 4)": 6}}
    assert (rec["counts"]["agents"], rec["counts"]["agents_failed"]) == (3, 0)   # the stale file was not read


def test_command_line_refusals(tmp_path):
    k = tmp_path / "k.yaml"
    k.write_text(json.dumps({"agents": {"per_process": 1}}))
    with pytest.raises(KnobError, match="agents.count must be in"):
        run.main(["--agents", "0", "--sim", "fixture", "--out", str(tmp_path / "zero")])
    with pytest.raises(KnobError, match="llm.model is still"):
        run.main(["--agents", "1", "--llm", "openai", "--sim", "fixture", "--out", str(tmp_path / "llm")])
    with pytest.raises(KnobError, match="3 agent processes each need a simulation worker"):
        run.main(["--agents", "3", "--knobs", str(k), "--sim", "real", "--out", str(tmp_path / "workers")])


def test_flag_selects_the_layer_and_imports_only_that_framework(tmp_path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    (tmp_path / "k.yaml").write_text(json.dumps({"agents": {"per_process": 3}}))   # 6 agents in 2 processes
    for layer in LAYERS:
        out = tmp_path / layer
        flag = ["--use-academy"] if layer == "academy" else []
        p = subprocess.run([sys.executable, "-m", "fusion_miniapp.run", "--agents", "6", "--sim", "fixture",
                            "--knobs", str(tmp_path / "k.yaml"), "--out", str(out), *flag],
                           capture_output=True, text=True, env=env, timeout=120)
        assert p.returncode == 0, p.stderr[-2000:]
        rec = json.loads((out / "run_record.json").read_text())
        assert rec["agent_layer"] == layer
        assert rec["counts"]["agent_processes"] == 2 and rec["counts"]["agents"] == 6
        other = "langgraph" if layer == "academy" else "academy"
        assert rec["frameworks_imported"][layer] and not rec["frameworks_imported"][other]
        assert "agents working=" in p.stderr and "done=6" in p.stderr
