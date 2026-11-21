"""Agent side: the two layers agree, the flag picks the layer, the run record's counts are right.

Every campaign is the parent's outer loop (Coordinator, Generator, Evaluator); tests/test_parent_oracle.py
checks it against the parent itself. All here with the fixture simulation and the stub LLM (or a
recording stand-in); laptop timings only.
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


def knobs_for(n, P=4, G=2, tools=None, outer=None, **agents):
    return knob_files.validate({"agents": {"count": n, "per_process": n, **agents}, "tools": tools or {},
                                "outer": {"population": P, "generations": G, "parent_stdout": "drop",
                                          **(outer or {})}})


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
        assert all(r["n_sims"] == 8 and r["n_llm_calls"] == 2 and len(r["generations"]) == 2 for r in results)
    assert record.digest(got["langgraph"]) == record.digest(got["academy"])
    a0 = next(r for r in got["academy"] if r["agent_id"] == "a0")
    assert record.comparable(a0) == record.comparable(next(r for r in got["langgraph"] if r["agent_id"] == "a0"))
    s = a0["sources"]                                                    # the stub's JSON list parsed
    assert s["start"] == 4 and s["mutation"] == 0 and s["model"] == a0["refill"]["asked"] == 4 - s["survivor"]
    for layer in LAYERS:                                                 # the parent's per-campaign files
        d = tmp_path / layer / "campaigns" / "a0"
        assert sorted(p.name for p in d.iterdir()) == ["best_result.json", "config.json", "evolution_log.csv",
                                                       "evolution_log.jsonl"]
        assert len((d / "evolution_log.csv").read_text().splitlines()) == 1 + 8


def test_real_evaluate_through_both_layers_on_cpu(tmp_path):
    """The turned-down inner loop itself (torch on the CPU), not the fixture, behind each layer."""
    knobs = knob_files.validate({
        "agents": {"count": 2, "per_process": 2}, "sim": {"device": "cpu"},
        "outer": {"population": 3, "generations": 2, "parent_stdout": "drop"}})
    got = {layer: run_layer(layer, knobs, tmp_path / layer, sim="real")[0] for layer in LAYERS}
    for results in got.values():
        assert not [r for r in results if "error" in r]
        assert sum(r["n_sims"] for r in results) == 2 * 3 * 2 and not sum(r["fixture"] for r in results)
        assert {k for r in results for k in r["touch_checks"]} == {"pass"}
        assert {k for r in results for k in r["devices"]} == {"cpu"}
    assert record.digest(got["langgraph"]) == record.digest(got["academy"])


SEEN = []


def recording_fixture(candidate=None, knobs=None, agent_id="", seed=0, **where):
    """The fixture, noting the simulation knobs each campaign's evaluate received."""
    SEEN.append((agent_id, knobs["nodes_per_plane"], knobs["fixed_point_cap"]))
    return sim_module.evaluate_fixture(candidate, knobs, agent_id, seed, **where)


def test_per_agent_cost_scale_sets_simulation_knobs_and_budget(tmp_path):
    knobs = knobs_for(20, thinking_budget=100, cost_scale=[0.5, 1, 2], budget_scales_with_cost=True)
    want = {f"a{i}": [0.5, 1, 2][i % 3] for i in range(20)}             # campaign i uses entry i % 3
    got = {}
    for layer in LAYERS:
        model = StubLLM()
        SEEN.clear()
        got[layer], _ = run_layer(layer, knobs, tmp_path / layer, model=model, sim_fn=recording_fixture)
        assert model.calls == 20 * 2                                        # G calls per campaign
        assert dict(model.budgets) == {50: 7 * 2, 100: 7 * 2, 200: 6 * 2}  # each campaign's budget, every call
        assert {r["agent_id"]: r["budget"] for r in got[layer]} == {a: round(100 * c) for a, c in want.items()}
        assert sorted(set(SEEN)) == sorted({(a, round(200 * c), round(3 * c)) for a, c in want.items()})
        assert len(SEEN) == 20 * 8
        cfg = json.loads((tmp_path / layer / "campaigns" / "a2" / "config.json").read_text())
        assert cfg["sim_knobs"]["fixed_point_cap"] == 6 and cfg["sim"] == "fixture"   # a2's own cost scale
    assert record.digest(got["langgraph"]) == record.digest(got["academy"])


class InFlight(StubLLM):
    """A stub with 20 ms of latency that counts the calls in flight at once."""

    def __init__(self):
        super().__init__(latency_s=0.2)   # long enough that all 10 campaigns' refills overlap under load
        self.now = self.peak = 0

    async def call_with_meta(self, agent_id, messages, thinking_budget, reply_tokens=256):   # both layers' path
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            return await super().call_with_meta(agent_id, messages, thinking_budget, reply_tokens)
        finally:
            self.now -= 1


def test_both_layers_admit_the_same_share_of_model_calls(tmp_path):
    knobs = knobs_for(1000)                                             # this process runs 10 of 1,000:
    for layer in LAYERS:                                                # its share is 512 x 1% = 5
        model = InFlight()
        results, _ = run_layer(layer, knobs, tmp_path / layer, model=model, n=10)
        assert len(results) == 10 and not [r for r in results if "error" in r]
        assert model.peak == 5, (layer, model.peak)


class NoJSON(StubLLM):
    """An LLM whose replies never parse."""

    async def call_with_meta(self, agent_id, messages, thinking_budget, reply_tokens=256):
        _, meta = await super().call_with_meta(agent_id, messages, thinking_budget, reply_tokens)
        return "I would try a wider pedestal next.", meta


def test_unparsable_reply_keeps_the_parents_mutants(tmp_path):
    knobs = knobs_for(10, G=3)
    got = {}
    for layer in LAYERS:
        got[layer], _ = run_layer(layer, knobs, tmp_path / layer, model=NoJSON())
        for r in got[layer]:
            assert r["sources"]["model"] == 0 and r["sources"]["mutation"] == r["refill"]["asked"] > 0
            assert r["refill"]["filled"] == 0 and r["n_llm_calls"] == 3
    assert record.digest(got["langgraph"]) == record.digest(got["academy"])  # the mutation is the parent's


def test_mutation_mode_makes_no_refill_calls(tmp_path):
    knobs = knobs_for(5, G=3, outer={"generator": "mutation"})
    for layer in LAYERS:
        model = StubLLM()
        results, _ = run_layer(layer, knobs, tmp_path / layer, model=model)
        assert model.calls == 5 and all(r["n_llm_calls"] == 1 and r["refill"]["asked"] == 0 for r in results)


def test_counts_in_the_run_record_for_1000_agents_in_both_layers(tmp_path):
    (tmp_path / "k.yaml").write_text(json.dumps({"agents": {"count": 1000, "per_process": 1000},
                                                 "outer": {"parent_stdout": "drop"}}))
    recs = {}
    for layer in LAYERS:
        flag = ["--use-academy"] if layer == "academy" else []
        assert run.main(["--knobs", str(tmp_path / "k.yaml"), "--sim", "fixture",
                         "--out", str(tmp_path / layer), *flag]) == 0
        recs[layer] = rec = json.loads((tmp_path / layer / "run_record.json").read_text())
        c = rec["counts"]
        assert rec["agent_layer"] == layer and rec["status"] == "pass"
        assert (c["agents"], c["agents_failed"], c["sims"], c["sim_tasks"]) == (1000, 0, 8000, 8000)
        assert c["llm_calls"] == rec["llm_service"]["calls"] == 1000 * 2
        assert rec["llm_service"]["budgets"] == {"64": 2000}
        assert c["tool_calls"] == 1000 * 2                                  # constraints, abstract
        assert rec["lookups"] == {"constraints": 1000, "abstract": 1000}
        s = c["candidates_by_source"]
        assert s["start"] == 4000 and s["survivor"] + s["model"] == 4000 and s["mutation"] == s["restart"] == 0
        assert c["refill_slots"] == {"asked": s["model"], "filled_by_model": s["model"]}   # every reply parsed
        assert (c["agents_requested"], c["failed_agent_processes"], c["generations"]) == (1000, [], 2000)
        assert rec["emulated"] is True and "synthetic" in rec["note"]
        assert (c["memory_writes"], c["reports"]) == (2000, 1000)
        assert c["suspensions"] == (2000 if layer == "langgraph" else None)     # one per generation
        assert rec["touch_check"] == {"evaluations": 8000, "fixture": 8000, "pass": 0, "fail": {}}
        assert len((tmp_path / layer / "agents.jsonl").read_text().splitlines()) == 1000
    assert recs["langgraph"]["results_sha256"] == recs["academy"]["results_sha256"]


def missed_touch(candidate=None, knobs=None, agent_id="", seed=0, **where):
    """A non-fixture result whose touch check failed."""
    from fusion_miniapp import sim
    return {**sim.evaluate_fixture(candidate, knobs, agent_id, seed, **where), "fixture": False,
            "touch_check": "fail: mesh_read=0 (want 4)"}


def test_a_failed_touch_check_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "sim_function", lambda mode: missed_touch)
    assert run.main(["--agents", "3", "--sim", "fixture", "--out", str(tmp_path)]) == 1
    rec = json.loads((tmp_path / "run_record.json").read_text())
    assert rec["status"] == "fail"
    assert rec["touch_check"] == {"evaluations": 24, "fixture": 0, "pass": 0, "fail": {"fail: mesh_read=0 (want 4)": 24}}
    assert (rec["counts"]["agents"], rec["counts"]["agents_failed"]) == (3, 0)


def test_an_output_directory_holding_a_run_is_refused(tmp_path):
    (tmp_path / "agents-7.jsonl").write_text('{"agent_id": "stale", "error": "from an earlier run"}\n')
    with pytest.raises(KnobError, match="already holds a run"):
        run.main(["--agents", "1", "--sim", "fixture", "--out", str(tmp_path)])


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
