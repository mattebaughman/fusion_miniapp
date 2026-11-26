"""The run record's speed section, model check and comparison (the plan, packages 4 and 5), and RHAPSODY's
bookkeeping being pruned. Fixture simulation and stub or local fixture endpoints; laptop timings only."""
import asyncio
import json
import threading

import numpy as np
from aiohttp import web

from fusion_miniapp import compare_records, record, run


def test_occupancy_counts_open_intervals():
    occ = record.occupancy([0, 1, 2], [3, 3, 4], 0, 4, n_ref=3)
    assert occ["max"] == 3 and occ["mean"] == round((1 + 2 + 3 + 1) / 4, 2)   # 1, 2, 3, then 1 open
    assert occ["time_ge_95pct_s"] == 1.0                                     # all 3 open from 2 to 3
    assert record.occupancy([0, 1], [1, 2], 0, 2)["max"] == 1                # an end counts before a start


def test_the_loop_lags_summary_without_awaiting_is_stops_result():
    """record.LoopLag.summary(), for code that cannot await (a population's rows under Dragon), ends the sampling and
    gives what stop() gives; with no sample, {"samples": 0}."""
    async def sampled():
        lag = record.LoopLag()
        lag.start()
        await asyncio.sleep(0.45)
        now = lag.summary()
        await asyncio.sleep(0.25)   # no sample after it
        return now, await lag.stop()
    now, stopped = asyncio.run(sampled())
    assert now == stopped and now["samples"] >= 2 and set(now) == {"samples", "p50_ms", "p95_ms", "max_ms",
                                                                    "share_over_100ms"}
    assert record.LoopLag().summary() == {"samples": 0}


def test_the_steady_windows_tokens_per_second_and_share_of_time_in_a_call():
    """The steady window is 15% to 85% of the agent phase: [15, 85] of [0, 100]. Tokens per second count the calls
    answered in it (t_end 10 and 90 are out); a role's share of time in a call is its calls' time open in it over
    its agents' (here 2 campaigns, 1 coordinator and 2 monitors each), and a campaign's agents in a call their sum."""
    tok = record.model_tokens([10, 20, 50, 85, 90], [100] * 5, [10, 20, 30, 40, 50], 0, 100, False)
    assert tok == {"window": [0.15, 0.85], "window_s": 70.0, "calls": 3, "prompt_per_s": round(300 / 70, 3),
                   "generated_per_s": round(90 / 70, 3), "emulated": False}
    assert record.model_tokens([5], [1], [1], 3, 3, True)["prompt_per_s"] is None   # no phase: no rate
    by_role = {"coordinator": ([10, 60], [50, 90]), "monitor": ([0], [100])}   # open 35 + 25 s, and all 70 s
    got = record.call_share(by_role, {"coordinator": 1, "generator": 1, "monitor": 2}, 2, 0, 100)
    assert got["by_role"] == {"coordinator": round(60 / 70 / 2, 4), "generator": 0.0, "monitor": 0.25}
    assert got["agents_in_a_call_per_campaign"] == round(round(60 / 70 / 2, 4) + 2 * 0.25, 4)


def test_speed_section_and_line(tmp_path, capsys):
    assert run.main(["--agents", "200", "--sim", "fixture", "--out", str(tmp_path / "r")]) == 0
    rec = json.loads((tmp_path / "r" / "run_record.json").read_text())
    s, c = rec["speed"], rec["counts"]
    assert s["campaigns_live"]["requested"] == 200 and 0 < s["campaigns_live"]["max"] <= 200
    assert s["sim_s"]["n"] == s["queue_wait_s"]["n"] == c["sims"] == 200 * 8
    assert s["generation_s"]["n"] == c["generations"] == 400
    assert s["model_latency_s"]["n"] == c["llm_calls"] == rec["llm_service"]["calls"]
    assert s["sims_executing"]["max"] <= rec["sim_workers_total"] == 2           # two fixture threads
    assert s["sims_executing"]["max"] <= s["sims_in_flight"]["max"] <= 200 * 4   # at most P per campaign
    assert s["agent_phase_s"] > 0 and s["startup_s"] >= 0 and s["shutdown_s"] >= 0
    assert s["rates_per_s"]["evaluations"] > 0 and s["loop_lag"][0]["samples"] >= 1
    line0 = json.loads((tmp_path / "r" / "agents.jsonl").read_text().splitlines()[0])
    g0 = line0["generations"][0]
    assert g0["fastest"]["sim_s"] <= g0["slowest"]["sim_s"] and 0 <= g0["slowest"]["candidate_idx"] < 4
    assert len(line0["llm_call_details"]) == 2 and line0["llm_call_details"][0]["finish_reason"] == "stop"
    assert rec["run"]["host"] and set(rec["run"]["miniapp_git"]) == {"commit", "dirty"}
    chk = rec["llm_check"]
    assert (chk["model_share"], chk["empty_share"], chk["fail"]) == (1.0, 0.0, [])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("speed: ")]
    assert len(line) == 1
    fields = dict(kv.split("=", 1) for kv in line[0][len("speed: "):].split())
    assert fields["status"] == "pass" and int(fields["campaigns"]) == 200
    # the agents file was streamed into place and the per-process files removed
    assert not list((tmp_path / "r").glob("agents-*.jsonl")) and not (tmp_path / "r" / "agents.jsonl.tmp").exists()
    assert len((tmp_path / "r" / "agents.jsonl").read_text().splitlines()) == 200


def test_comparison_of_non_finite_values_and_an_early_interrupted_speed_line():
    out = []
    compare_records._close(float("nan"), float("nan"), "x", 1e-6, 1e-9, out)
    compare_records._close(float("inf"), float("inf"), "y", 1e-6, 1e-9, out)
    assert out == []
    compare_records._close(float("inf"), float("-inf"), "z", 1e-6, 1e-9, out)
    assert out == ["z: inf != -inf"]
    rec = {"status": "interrupted", "agent_layer": "langgraph", "counts": {"agents": 0},
           "speed": {"campaigns_live": {"max": None, "mean": None, "time_ge_95pct_s": None, "ramp_up_s": None},
                     "agent_phase_s": None, "rates_per_s": {"evaluations": None, "model_calls": None},
                     "sims_in_flight": None, "sims_executing": None, "sim_s": {"n": 0}, "queue_wait_s": {"n": 0},
                     "generation_s": {"n": 0}, "model_latency_s": {"n": 0}, "loop_lag": [], "sim_bound_ratio": None}}
    assert record.speed_line(rec).startswith("speed: status=interrupted")


def test_the_model_check_compares_unrounded_shares():
    knobs = {"llm": {"mode": "openai", "min_model_share": 0.5, "max_empty_share": 0.1}}
    chk = record.llm_check(knobs, {"asked": 100_000, "filled": 49_999}, {"calls": 100_000, "empty": 10_001})
    assert len(chk["fail"]) == 2 and "49999 of the 100000" in chk["fail"][0] and "10001 of" in chk["fail"][1]
    chk = record.llm_check(knobs, {"asked": 0, "filled": 0}, {"calls": 0, "empty": 0})   # no slots: no verdict
    assert chk["model_share"] is None and chk["fail"] == []


def test_campaign_time_starts_before_its_lookups(tmp_path):
    (tmp_path / "k.yaml").write_text(json.dumps({"tools": {"latency_ms": 300}}))
    for layer, flag in (("lg", []), ("ac", ["--use-academy"])):
        assert run.main(["--agents", "3", "--sim", "fixture", "--knobs", str(tmp_path / "k.yaml"),
                         "--out", str(tmp_path / layer), *flag]) == 0
        rec = json.loads((tmp_path / layer / "run_record.json").read_text())
        assert rec["speed"]["agent_phase_s"] >= 0.6, layer           # two 300 ms lookups are campaign time


def test_compare_records_across_layers(tmp_path):
    for layer, flag in (("lg", []), ("ac", ["--use-academy"])):
        assert run.main(["--agents", "30", "--sim", "fixture", "--out", str(tmp_path / layer), *flag]) == 0
    a, b = (json.loads((tmp_path / x / "run_record.json").read_text()) for x in ("lg", "ac"))
    assert compare_records.compare(a, b)["results"] == []          # suspensions differ by design: not compared
    assert compare_records.main([str(tmp_path / x / "run_record.json") for x in ("lg", "ac")]) == 0
    b["knobs"]["sim"]["device"] = "cuda"                            # a machine knob: not a difference
    assert compare_records.compare(a, b)["results"] == []
    b["knobs"]["outer"]["population"] = 5
    assert compare_records.compare(a, b)["results"] == ["knobs.outer.population: 4 != 5"]
    b["knobs"]["outer"]["population"] = 4
    del a["knobs"]["agents"]["first"]                               # a record from before the knob: first 0
    assert compare_records.compare(a, b)["results"] == []
    b["knobs"]["agents"]["first"] = 5
    assert compare_records.compare(a, b)["results"] == ["knobs.agents.first: 0 != 5"]


def test_compare_records_skips_suspensions_across_the_monitor_slots():
    """On the AUTONMI layer a record from before the Monitor slots (no `campaign` knobs: one suspension per
    generation) and one from after (one per inner loop) agree when their results do; two records from after, or
    two from another layer, must still suspend alike."""
    new = {"status": "pass", "agent_layer": "autonmi", "results_sha256": "x", "counts": {"suspensions": 16000},
           "knobs": {"outer": {"population": 16}, "campaign": {"slots": 2, "mode": "generations", "warmup": 4}}}
    old = {**new, "counts": {"suspensions": 1000}, "knobs": {"outer": {"population": 16}}}
    assert compare_records.compare(old, new)["results"] == []
    assert compare_records.compare({**new, "counts": {"suspensions": 1000}}, new)["results"] == [
        "counts.suspensions: 1000 != 16000"]
    lg = {**old, "agent_layer": "langgraph"}
    assert compare_records.compare(lg, {**new, "agent_layer": "langgraph"})["results"] == [
        "counts.suspensions: 1000 != 16000"]


def test_compare_records_compares_asynchronous_records_on_counts_and_invariants():
    """Two records of the asynchronous mode (whose results differ from run to run with the order of arrival)
    agree when both pass (AUTONMI's checks of the mode's invariants are in the status), and their knobs, their
    exact counts other than those that follow the order (feasible_agents, memory_writes) and, with a real model,
    its replies (candidates_by_source, refill_slots), and their results scored agree; their digests are not
    compared. An asynchronous and a generations record differ in the mode."""
    knobs = {"outer": {"population": 16}, "campaign": {"slots": 2, "mode": "async", "warmup": 4}}
    a = {"status": "pass", "agent_layer": "autonmi", "results_sha256": "x", "knobs": knobs,
         "counts": {"sims": 32, "feasible_agents": 0, "memory_writes": 30, "suspensions": 32,
                    "asynchronous": {"results_scored": 32, "results_kept": 17}}}
    b = {**a, "results_sha256": "y", "counts": {**a["counts"], "feasible_agents": 1, "memory_writes": 28,
                                                "asynchronous": {"results_scored": 32, "results_kept": 15}}}
    assert compare_records.compare(a, b)["results"] == []
    c = {**b, "counts": {**b["counts"], "sims": 31, "asynchronous": {"results_scored": 31}}}
    assert compare_records.compare(a, c)["results"] == ["counts.asynchronous.results_scored: 32 != 31",
                                                        "counts.sims: 32 != 31"]
    assert compare_records.compare(a, {**b, "status": "fail"})["results"] == [
        "the second run's status is 'fail', not 'pass'"]
    g = {**b, "knobs": {**knobs, "campaign": {**knobs["campaign"], "mode": "generations"}}}
    assert "knobs.campaign.mode: 'async' != 'generations'" in compare_records.compare(a, g)["results"]
    # the counts that follow the model's replies (a candidate is the model's when its reply filled a parameter):
    # left out with a real model (llm.mode openai), compared with the stub, whose replies are fixed
    stub = {**a, "counts": {**a["counts"], "candidates_by_source": {"model": 32}, "refill_slots": {"filled_by_model": 32}}}
    model = {**b, "counts": {**b["counts"], "candidates_by_source": {"model": 30, "mutation": 2},
                             "refill_slots": {"filled_by_model": 30}}}
    assert compare_records.compare(stub, model)["results"] == [
        "counts.candidates_by_source: {'model': 32} != {'model': 30, 'mutation': 2}",
        "counts.refill_slots: {'filled_by_model': 32} != {'filled_by_model': 30}"]
    openai = {**knobs, "llm": {"mode": "openai"}}
    assert compare_records.compare({**stub, "knobs": openai}, {**model, "knobs": openai})["results"] == []


def test_compare_records_needs_a_digest_for_generations():
    base = {"status": "pass", "agent_layer": "autonmi", "counts": {}, "knobs": {}}
    with_digest = {**base, "results_sha256": "abc"}
    without_digest = {**base, "results_sha256": None}
    reason = ["missing results_sha256: campaign agreement is unverifiable"]
    assert compare_records.compare(base, without_digest)["results"] == reason
    assert compare_records.compare(with_digest, without_digest)["results"] == reason
    assert compare_records.compare(with_digest, with_digest)["results"] == []
    asynchronous = {**without_digest, "knobs": {"campaign": {"mode": "async"}},
                    "counts": {"asynchronous": {"results_scored": 8}}}
    assert compare_records.compare(asynchronous, asynchronous)["results"] == []


def test_compare_records_checks_which_campaigns_an_async_merge_covers():
    # Two contiguous pairs with first=0 and total=4. The second pair starts at 2, 3 or 4.
    base = {"status": "pass", "agent_layer": "autonmi",
            "knobs": {"agents": {"first": 0, "count": 4}, "campaign": {"mode": "async"}},
            "counts": {"agents_requested": 4, "asynchronous": {"results_scored": 32}}}
    contiguous = {**base, "campaign_gaps": []}
    one_gap = {**base, "campaign_gaps": [[2, 3]]}
    two_missing = {**base, "campaign_gaps": [[2, 4]]}
    assert compare_records.compare(base, contiguous)["results"] == []
    assert compare_records.compare(one_gap, one_gap)["results"] == []
    assert compare_records.compare(base, one_gap)["results"] == ["campaign_gaps: [] != [[2, 3]]"]
    assert compare_records.compare(one_gap, two_missing)["results"] == ["campaign_gaps: [[2, 3]] != [[2, 4]]"]


def test_compare_records_tolerates_last_digit_differences(tmp_path):
    assert run.main(["--agents", "5", "--sim", "fixture", "--out", str(tmp_path / "a")]) == 0
    rec = json.loads((tmp_path / "a" / "run_record.json").read_text())
    lines = (tmp_path / "a" / "agents.jsonl").read_text().splitlines()
    nudged = []
    for line in lines:
        r = json.loads(line)
        r["best"]["te_core_metric"] *= 1 + 1e-9                     # another machine's last digits
        nudged.append(json.dumps(r))
    other = dict(rec, results_sha256=record.digest(nudged))
    assert compare_records.compare(rec, other)["results"] == [
        "results_sha256 differs and the first run's agents.jsonl: missing: unverifiable",
        "results_sha256 differs and the second run's agents.jsonl: missing: unverifiable"]
    assert compare_records.compare(rec, other, agents_a=lines, agents_b=[])["results"] == [
        "results_sha256 differs and the second run's agents.jsonl: 0 campaigns, the record says 5: unverifiable"]
    assert compare_records.compare(rec, other, agents_a=lines, agents_b=nudged)["results"] == []
    r = json.loads(nudged[0])
    r["best"]["is_good"] = not r["best"]["is_good"]                 # a decision: must agree exactly
    nudged[0] = json.dumps(r)
    other["results_sha256"] = record.digest(nudged)
    diffs = compare_records.compare(rec, other, agents_a=lines, agents_b=nudged)["results"]
    assert len(diffs) == 1 and diffs[0].endswith(".best.is_good: " + repr(not r["best"]["is_good"]) + " != "
                                                 + repr(r["best"]["is_good"]))


def _endpoint(content: str):
    """A local fixture endpoint (not a model) that answers every chat request with `content`."""
    async def chat(request):
        body = await request.json()
        return web.json_response({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    async def serve():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return runner, site._server.sockets[0].getsockname()[1]

    loop = asyncio.new_event_loop()
    runner, port = loop.run_until_complete(serve())
    thread = threading.Thread(target=loop.run_forever)
    thread.start()

    def stop():
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.run_until_complete(runner.cleanup())
        loop.close()
    return port, stop


def test_a_model_that_fills_no_slots_fails_the_run(tmp_path, monkeypatch, capsys):
    """The audit's silent all-mutation run: every reply empty, so every refill is the parent's mutants."""
    port, stop = _endpoint("")
    try:
        monkeypatch.setenv("MINIAPP_TEST_LLM_URL", f"http://127.0.0.1:{port}/v1")
        (tmp_path / "k.yaml").write_text(json.dumps({"llm": {"mode": "openai", "model": "test-model",
                                                             "base_url_env": "MINIAPP_TEST_LLM_URL",
                                                             "api_key_env": "MINIAPP_TEST_LLM_KEY"}}))
        assert run.main(["--agents", "5", "--sim", "fixture", "--knobs", str(tmp_path / "k.yaml"),
                         "--out", str(tmp_path / "r")]) == 1
    finally:
        stop()
    rec = json.loads((tmp_path / "r" / "run_record.json").read_text())
    assert rec["status"] == "fail" and rec["counts"]["agents_failed"] == 0      # every campaign finished
    chk = rec["llm_check"]
    assert chk["model_share"] == 0.0 and chk["empty_share"] == 1.0 and len(chk["fail"]) == 2
    assert "below llm.min_model_share" in capsys.readouterr().err


def test_finished_tasks_are_pruned_from_rhapsody(tmp_path):
    """RHAPSODY's Session and backend would keep every finished task; the pruner empties both."""
    from fusion_miniapp import decide, knobs as knob_files

    async def go():
        knobs = knob_files.validate({"agents": {"count": 1, "per_process": 1}})
        svc = await run.Services.open(knobs, run.sim_function("fixture"), tmp_path, 0, 2)
        evo = decide.parent()
        pop = evo.initialize_population(4, "three_region", "H", 0, delta_bounds=evo.DELTA_CLIP)
        try:
            for g in range(3):
                await decide.evaluate_population(svc, "a0", 0, pop, g)
            before = len(svc.session._tasks)
            pruned = run._prune(svc.session)
            backend_left = sum(len(b.tasks) for b in svc.session._exec_backends.values())
            return before, pruned, len(svc.session._tasks), backend_left
        finally:
            await svc.close()
    before, pruned, after, backend_left = asyncio.run(go())
    assert before == pruned == 12 and after == 0 and backend_left == 0
    assert np.isfinite(before)
