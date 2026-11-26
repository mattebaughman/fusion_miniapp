"""The asynchronous mode's rule (campaign.mode: async), as decide.py's pure functions: each agent's state and what
arrived give its new state and what it sends. No agent framework: these run in the base environment.

The results are the fixture's (FIXTURE data: a function of the profile, not an inner loop), a few marked
feasible by hand so that both of the parent's rankings are used."""
import asyncio
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from async_replay import fixture, replay

from fusion_miniapp import decide, knobs as knob_files, monitor, record, sim, tools
from fusion_miniapp.llm import StubLLM, numbers_only


def async_knobs(weight: float = 0.25, population: int = 16, generations: int = 2) -> dict:
    k = knob_files.validate({"campaign": {"mode": "async"}, "monitor": {"weight": weight},
                             "outer": {"population": population, "generations": generations, "parent_stdout": "drop"}},
                            modes=("generations", "async"))
    return k


def results(n: int = 48) -> list[dict]:
    """n fixture results (the parent's initial populations of seeds 0, 1, ...), as JSON carries them, with
    three marked feasible and one te_core_metric not finite."""
    evo, out = decide.parent(), []
    for seed in range(-(-n // 16)):
        for i, p in enumerate(evo.initialize_population(16, decide.MODEL, decide.KIND, seed, delta_bounds=evo.DELTA_CLIP)):
            out.append(sim.evaluate_fixture(params=decide._plain(p), candidate_idx=16 * seed + i))
    out = out[:n]
    for i in (20, 23, 31):
        if i < n:
            out[i].update(wall_ok=True, pedestal_ok=True, is_good=True)
    if n > 26:
        out[26]["te_core_metric"] = float("nan")
    return [json.loads(json.dumps(r).replace("NaN", "null")) for r in out]   # as a message carries them


def reports(rs: list[dict], m=lambda k: 0.0) -> list[dict]:
    return [{"candidate": k, "result": r, "t0": 0.0, "source": "start", "m": m(k)} for k, r in enumerate(rs)]


def evaluate_in(rs: list[dict], knobs: dict, batches=(1, 3, 2, 5)) -> list[dict]:
    """The Evaluator's cycles over the reports in arrival order, in batches of the given sizes, in turn."""
    window, out, i, b = [], [], 0, 0
    while i < len(rs):
        n = batches[b % len(batches)]
        window, scored, digest = decide.async_evaluate(window, rs[i:i + n], knobs)
        assert digest["candidates"] == [x["candidate"] for x in rs[i:i + n]]
        out += scored
        i, b = i + n, b + 1
    return out


def test_the_score_at_weight_0_is_the_parents_keep_probability_in_the_window(capsys):
    """At monitor.weight 0 each result's score is q, whatever the Monitor's m: the probability that the parent's
    select_survivors keeps it in the set of the campaign's last outer.population scored results and itself (in
    arrival order, it last). Checked as package 1 checks q: replaying select_survivors with seeded generators,
    the result survives exactly when the draw the parent makes for it is below q (the best: q 1, kept with no
    draw; the worst: q 0). The Evaluator's batches do not matter: one result at a time, in arrival order."""
    evo, k = decide.parent(), async_knobs(weight=0.0)
    rs = results()
    scored = evaluate_in(reports(rs, m=lambda i: (-1) ** i * 0.8), k)
    assert [x["candidate"] for x in scored] == list(range(len(rs)))
    assert evaluate_in(reports(rs, m=lambda i: (-1) ** i * 0.8), k, batches=(1,)) == scored
    for i, x in enumerate(scored):
        window = [decide.native(r) for r in rs[max(0, i - 16):i + 1]]
        q = decide.keep_probabilities(window, k["outer"]["wall_weight"])
        assert x["score"] == x["q"] == q[-1], i
        middle = sorted((j for j, v in enumerate(q) if 0.0 < v < 1.0), key=lambda j: -q[j])
        for seed in range(40):
            evaluated = copy.deepcopy(window)
            rng = np.random.default_rng(seed)
            kept = {id(e) for e in evo.select_survivors(evaluated, rng, k["outer"]["wall_weight"])}
            replay_ = np.random.default_rng(seed)
            draws = {j: replay_.random() for j in middle}
            assert replay_.bit_generator.state == rng.bit_generator.state
            last = len(window) - 1
            assert (id(evaluated[-1]) in kept) == (q[last] == 1.0 or (last in draws and draws[last] < q[last])), (i, seed)
    assert {x["q"] for x in scored} >= {0.0, 1.0} and any(0 < x["q"] < 1 for x in scored)
    capsys.readouterr()   # the parent's prints for a set with nothing feasible


def test_the_score_adds_the_monitors_m_by_its_weight_clipped_to_0_1():
    """score = q + monitor.weight x m, clipped to 0..1; m None (no trace) counts 0."""
    rs = reports(results(30), m=lambda i: [1.0, -1.0, 0.5, None][i % 4])
    plain = evaluate_in(rs, async_knobs(weight=0.0))
    for x, y in zip(plain, evaluate_in(rs, async_knobs(weight=0.25))):
        assert y["q"] == x["q"] and y["score"] == min(1.0, max(0.0, x["q"] + 0.25 * (y["m"] or 0.0)))
    assert any(0 < x["score"] < 1 and x["m"] for x in plain)


def test_survival_is_the_draw_below_the_score_and_warm_up_keeps_every_result(tmp_path):
    """The Coordinator, in batches: one draw per result from stream 1 of outer.rng per_agent, in arrival order;
    kept exactly when draw < score, and always for the first campaign.warmup results; the survivors the last
    outer.population kept, the oldest leaving; best-so-far the best feasible result so far once there is one,
    else the top-ranked of the current window and result (the parent's update_best_result); one parent log row
    per result, its arrival number as the generation and `survived` its kept flag."""
    k, seed = async_knobs(), 7
    rs = results()
    scored = evaluate_in(reports(rs, m=lambda i: math.sin(i)), k)
    cst = decide.async_coordinator_state("a7", seed, k, {}, 0.0)
    rows, i = [], 0
    for n in (2, 1, 4, 3, 7, 1, 30):
        cst, r = decide.async_on_scored(cst, scored[i:i + n], k, tmp_path)
        rows, i = rows + r, i + n
    assert i == len(rs) and cst["arrived"] == len(rs) and cst["cycles"] == 7
    stream1 = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    warmup = k["campaign"]["warmup"]
    for n, (row, x) in enumerate(zip(rows, scored)):
        assert (row["arrival"], row["candidate"], row["score"]) == (n, x["candidate"], x["score"])
        assert row["draw"] == stream1.random()
        assert row["kept"] == (n < warmup or row["draw"] < row["score"]), row
    kept = [r["candidate"] for r in rows if r["kept"]]
    assert all(r["kept"] for r in rows[:warmup]) and 0 < len(kept) < len(rows)
    assert [s["candidate"] for s in cst["survivors"]] == kept[-16:]
    assert cst["kept"] == len(kept) and cst["scored_after_warmup"] == len(rs) - warmup
    assert cst["kept_after_warmup"] == sum(r["kept"] for r in rows[warmup:])
    feasible = [j for j, r in enumerate(rs) if r["is_good"]]
    assert cst["best"]["candidate_idx"] == max(feasible, key=lambda j: rs[j]["te_core_metric"])
    assert cst["best"]["selection_basis"] == "feasible_core"
    early = decide.async_coordinator_state("a7", seed, k, {}, 0.0)   # before any feasible result: the window's top
    early, _ = decide.async_on_scored(early, scored[:12], k, None)
    top = decide.ranked([decide.native(r) for r in rs[:12]], k["outer"]["wall_weight"])[0]
    assert early["best"]["candidate_idx"] == top["candidate_idx"]
    assert early["best"]["selection_basis"] == "no_feasible_core_wall_blend" and early["best"]["generation"] == 11
    with (tmp_path / "evolution_log.csv").open() as f:
        log = list(csv.DictReader(f))
    assert [(int(x["generation"]), int(x["candidate_idx"]), x["survived"] == "True") for x in log] == \
        [(r["arrival"], r["candidate"], r["kept"]) for r in rows]
    assert cst["gens"] == [] and len(cst["t_submit"]) == len(cst["sim_s"]) == len(cst["wait_s"]) == len(rs)
    line = decide.async_finish(cst)   # no generations; each evaluation's times; the counts
    assert (line["generations"], line["n_sims"], line["t_submit"], line["n_cycles"]) == ([], len(rs), cst["t_submit"], 7)
    assert line["survival"] == {"results_scored": len(rs), "results_kept": len(kept),
                                "results_scored_after_warmup": len(rs) - warmup,
                                "results_kept_after_warmup": cst["kept_after_warmup"]}
    assert cst["counts"]["start"] == len(rs) and decide.async_memory(cst)["survivors"] == kept[-16:]
    msg = decide.async_survivors(cst)
    assert msg["n"] == 7 and msg["survivors"] == [decide.native(rs[c])["params"] for c in kept[-16:]]


def test_the_generator_takes_the_parents_profiles_two_at_a_time():
    """Candidates 0 .. P-1 are the parent's initial population (drawn once, the campaign's seed); each later one
    is the parent's mutate_params of a survivor chosen at random from the newest survivors message held, from
    stream 0 as repopulate chooses and mutates; a take stops at a profile that needs a survivor when none is
    held, and the Generator's only wait is then for its first survivors (candidate P is due); an older survivors
    message never replaces a newer one; the budget ends the takes."""
    evo, k, seed = decide.parent(), async_knobs(population=5, generations=2), 3
    gst = decide.async_generator_state("a3", seed, k)
    made = []
    while not decide.async_waits(gst, k):
        gst, calls = decide.async_profiles(gst, k)
        assert 1 <= len(calls) <= decide.TAKE
        made += calls
    assert [c["candidate_idx"] for c in made] == [0, 1, 2, 3, 4] and gst["made"] == 5   # 5 waits for a survivor
    assert decide.async_profiles(gst, k) == (gst, [])
    initial = evo.initialize_population(5, decide.MODEL, decide.KIND, seed, delta_bounds=evo.DELTA_CLIP)
    survivors = [decide._plain(p) for p in initial[:3]]
    gst = decide.async_on_survivors(gst, [{"n": 2, "survivors": survivors}, {"n": 1, "survivors": []}])
    assert (gst["held"], gst["survivors"]) == (2, survivors)
    assert decide.async_on_survivors(gst, [{"n": 1, "survivors": []}]) == gst
    sizes = []
    while gst["made"] < 10:
        gst, calls = decide.async_profiles(gst, k)
        made, sizes = made + calls, sizes + [len(calls)]
    assert sizes == [2, 2, 1]   # the budget's last take is one
    assert [c["id"] for c in made] == [f"c{i}" for i in range(10)] and gst["initial"] is None
    assert [(c["generation"], c["source"]) for c in made] == [(i // 5, "start" if i < 5 else "mutation")
                                                                 for i in range(10)]
    assert [c["params"] for c in made[:5]] == [decide._plain(p) for p in initial]
    stream0 = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
    for c in made[5:]:
        chosen = survivors[int(stream0.integers(0, len(survivors)))]
        assert c["params"] == decide._plain(evo.mutate_params(chosen, decide.MODEL, decide.KIND, k["outer"]["noise"],
                                                              stream0, delta_bounds=evo.DELTA_CLIP))
    assert decide.async_profiles(gst, k)[1] == []


ORDER = [[k] for k in range(16)]


def orders(swap: bool) -> tuple[list, list, list]:
    """Evaluator and Coordinator cycles (one result each), in candidate order or with each later pair swapped, and
    the survivors message held at each take: the one after all but the last two results scored."""
    order = [[k + 1 - 2 * (k % 2)] if swap and k >= 4 else [k] for k in range(16)]
    held = [0 if k < 4 else 2 * (k // 2) - 2 for k in range(16)]
    return order, [list(x) for x in order], held


def test_one_order_and_one_seed_give_one_campaign():
    """The pure functions driven twice in one order of arrival (with the same seed and the Monitor's m) give the
    same states and results; another order gives another campaign (its survival, survivors and best-so-far)."""
    k, seed = async_knobs(population=4, generations=4), 11
    m = {c: ((c * 37) % 11 - 5) / 5 for c in range(16)}
    runs = [replay(seed, k, *orders(swap), m, fixture(k, seed)) for swap in (False, False, True)]
    same = [json.dumps({"cst": {x: v for x, v in cst.items() if x not in record.MACHINE_KEYS}, "rows": rows,
                        "calls": calls}, sort_keys=True) for cst, rows, calls in runs]   # less the machine's timings
    assert same[0] == same[1]
    (a, rows_a, _), (b, rows_b, _) = runs[0], runs[2]
    assert a["arrived"] == b["arrived"] == 16 and a["cycles"] == b["cycles"] == 16
    assert [(r["candidate"], r["kept"]) for r in rows_a] != [(r["candidate"], r["kept"]) for r in rows_b]
    assert ([s["candidate"] for s in a["survivors"]], a["best"]) != ([s["candidate"] for s in b["survivors"]], b["best"])
    with pytest.raises(RuntimeError, match="cannot be replayed"):   # a result scored before it was made
        replay(seed, k, [[5], *ORDER], ORDER, orders(False)[2], m, fixture(k, seed))


def test_a_trace_is_held_until_its_result_is_scored():
    """Rolling.take_one gives one held trace (the asynchronous mode labels each as its result is scored), once."""
    stats = monitor.Rolling(warmup=4, window=16)
    x = np.zeros((3, 4))
    stats.hold(1, 17, x, True)
    stats.hold(1, 18, x, False)
    assert stats.take_one(1, 17)[1] is True and stats.take_one(1, 17) is None and stats.held == {1: {18: (x, False)}}
    assert stats.take_one(1, 18)[1] is False and stats.held == {} and stats.take_one(0, 0) is None
    assert stats.recent == [] and stats.ends == []


def stub(messages: list[dict], agent: str = "a3") -> str:
    """The stub LLM's reply (FIXTURE), as the AUTONMI layer's stub transport asks it for campaign `agent`."""
    return asyncio.run(StubLLM(latency_s=0).call_with_meta(agent, messages, 0))[0]


def test_the_generators_reply_replaces_the_evolutionary_loops_values():
    """The asynchronous Generator's call: its prompt shows the take's profiles, the newest survivors, the
    Evaluator's latest scores and the Coordinator's guidance, and its last line names the candidates (the stub
    seeds its reply by the campaign and that line); its schema (sent as response_format) is a JSON list of as many
    objects as candidates. The reply is the last such list in the text: each value in it replaces the profile's,
    clipped to range (then the parent's clip_delta), a key left out keeps the evolutionary loop's value, and the
    parameters each filled are counted (source `model` when any). A reply that is not such a list cannot be used."""
    k = async_knobs(population=4, generations=8)
    gst, calls = decide.async_profiles(decide.async_generator_state("a3", 3, k, {"abstract": {"text": "Pedestals."}}), k)
    content = decide.async_generator_messages(gst, calls, k)[-1]["content"]
    assert content.splitlines()[-1].startswith("Candidates 0 and 1 of 32: reply with a JSON list of 2 objects,")
    assert "No survivors yet." in content and all(f"candidate {c['candidate_idx']} (start)" in content for c in calls)
    gst2 = {**gst, "survivors": [calls[0]["params"]], "scores": [[0, 0.5], [1, 0.25]], "guidance": "Lower a1_Te."}
    text = decide.async_generator_messages(gst2, calls, k)
    assert "Background: Pedestals." in text[0]["content"]
    assert all(x in text[1]["content"] for x in ("newest survivors (1)", "0: 0.5, 1: 0.25", "guidance: Lower a1_Te."))
    one = decide.async_generator_messages(gst, calls[1:], k)[-1]["content"].splitlines()[-1]
    assert one.startswith("Candidate 1 of 32: reply with a JSON list of 1 objects")
    schema = decide.candidates_schema(2)
    assert schema["type"] == "json_schema" and schema["json_schema"]["schema"]["minItems"] == 2 == \
        schema["json_schema"]["schema"]["maxItems"]
    assert set(schema["json_schema"]["schema"]["items"]["properties"]) == set(decide.ranges())
    lo, hi = decide.ranges()["a1_Te"]
    props = decide.parse_reconcile('[1, 2] then [{"a1_Te": %r, "other": 1}, {"Delta": "wide"}]' % (hi + 5), 2)
    assert props == [{"a1_Te": hi}, {}]
    out, filled = decide.async_reconcile(calls, props)
    assert filled == [1, 0] and (out[0]["source"], out[0]["filled"], out[1]["filled"]) == ("model", 1, 0)
    assert out[0]["params"] == {**calls[0]["params"], "a1_Te": hi} and out[1] == {**calls[1], "filled": 0}
    for bad in ("", "no JSON", "[{}]", '[{"a1_Te": 1}, 3]', '{"a1_Te": 1}'):
        assert decide.parse_reconcile(bad, 2) is None, bad
    got = decide.parse_reconcile(stub(decide.async_generator_messages(gst, calls, k)), 2)   # FIXTURE
    assert len(got) == 2 and all(set(x) == set(decide.ranges()) for x in got)


def test_an_integer_too_large_for_a_float_in_a_reply_is_not_a_number():
    """A JSON integer of 309 to 4,300 digits (json reads it as an int, which no float can hold): the Generator's and
    the Evaluator's parse functions and the generations mode's refill take it as they take 1e400, as no number, and
    raise nothing. So such a reply is not filled, or cannot be used and is tried again, as the README says."""
    big = "1" + "0" * 400
    assert decide.parse_reconcile('[{"a1_Te": %s}, {}]' % big, 2) == [{}, {}]
    assert decide.parse_reconcile('[{"a1_Te": 1e400}, {}]', 2) == [{}, {}]
    assert decide.parse_scores('{"5": %s}' % big, [5]) is None
    assert decide.parse_scores('{"5": 0.25} then {"5": %s}' % big, [5]) == {5: 0.25}
    assert decide.parse_candidates('[{"a1_Te": %s}]' % big, 1) == []


def test_the_evaluators_reply_is_the_score_and_the_stub_answers_the_score_from_numbers():
    """The Evaluator's call: its prompt gives each result's q and the Monitor's m (in full), the Monitor's other
    numbers and note, and neither the score from numbers nor its formula; its second-to-last line names the reply's
    keys (the candidate numbers as JSON strings) and its last line is SCORES_ASK; its reply is one JSON object with a
    keep-probability for every candidate, each clipped to 0..1, and becomes the score (and the digest); a reply
    without every candidate cannot be used. The stub (and the fake vLLM server) answer it with each score from
    numbers exactly, at their monitor.weight (a result without a trace: q), a Monitor's note and the
    Coordinator's guidance with nothing."""
    k = async_knobs(population=4, generations=4)
    rs = results(6)
    _, scored, _ = decide.async_evaluate([], reports(rs, m=lambda c: None if c == 2 else 0.5 - c / 5), k)
    items = [{**x, "numbers": None if x["m"] is None else {"m": x["m"], "left_at": 3, "back_at": None, "split_at": 5},
              "note": "steady"} for x in scored]
    messages = decide.scores_messages(items)
    asks = messages[-1]["content"].splitlines()
    assert asks[-1] == decide.SCORES_ASK   # the stub and the fake vLLM server know the call by its last line
    # the line before it names the reply's keys as parse_scores wants them (a model otherwise writes "candidate 0")
    assert asks[-2] == ('Use exactly these JSON keys, the candidate numbers alone as strings: '
                        '"0", "1", "2", "3", "4", "5".')
    shown = json.dumps([messages[0]["content"], *asks[:-2], asks[-1]])   # the prompt but for that line's wording
    assert "numbers alone" not in shown and "x m" not in shown
    assert "candidate 2: q " in messages[-1]["content"] and "no trace" in messages[-1]["content"]
    cands = [x["candidate"] for x in scored]
    from_numbers = {x["candidate"]: x["score"] for x in scored}
    assert decide.parse_scores(stub(messages), cands) == from_numbers   # exactly the score from numbers
    assert numbers_only(messages[-1]["content"]) == stub(messages)
    half = asyncio.run(StubLLM(latency_s=0, weight=0.5).call_with_meta("a3", messages, 0))[0]   # another weight
    assert decide.parse_scores(half, cands) == {x["candidate"]: decide.numbers_score(x["q"], x["m"], 0.5) for x in scored}
    assert decide.parse_scores(half, cands) != from_numbers
    new, digest = decide.async_model_scores(scored, dict.fromkeys(cands, 0.875))
    assert [x["score"] for x in new] == [0.875] * 6 and digest == {"candidates": cands, "scores": [0.875] * 6}
    assert [x["q"] for x in new] == [x["q"] for x in scored]
    assert decide.parse_scores('{"0": 2, "1": -1, "2": 0.5, "3": 0, "4": 1, "5": 0.25}', cands) == {
        0: 1.0, 1: 0.0, 2: 0.5, 3: 0.0, 4: 1.0, 5: 0.25}
    for bad in ('{"0": 0.5}', '{"0": "x", "1": 1, "2": 1, "3": 1, "4": 1, "5": 1}', "[]", ""):
        assert decide.parse_scores(bad, cands) is None, bad
    note = decide.note_messages(rs[0], items[0]["numbers"])
    assert note[-1]["content"].splitlines()[-1] == decide.NOTE_ASK and stub(note) == ""
    cst, rows = decide.async_on_scored(decide.async_coordinator_state("a3", 3, k, {}, 0.0), new, k, None)
    guidance = decide.guidance_messages(cst, rows, k)[-1]["content"]
    assert guidance.splitlines()[-1] == decide.GUIDANCE_ASK and stub(decide.guidance_messages(cst, rows, k)) == ""
    assert all(f"candidate {c}: {'kept' if r['kept'] else 'dropped'}" in guidance for c, r in zip(cands, rows))
    assert decide.parse_text(" two\n lines ") == "two lines" and decide.parse_text("") == ""


def test_the_calls_per_campaign_are_bounded_by_role():
    """The most calls each role asks in a campaign of the asynchronous mode (decide.async_calls): the hundredk shape
    (16 x 2) gives the Generator 16, the Monitors 32, the Evaluator and the Coordinator 32 each, 112 in all; none
    with llm.mode off; and the Generator's limit grows by one take when an odd P splits a take."""
    hundredk = knob_files.load(preset="hundredk")
    hundredk["campaign"]["mode"] = "async"
    assert decide.async_calls(hundredk) == {"coordinator": 32, "generator": 16, "evaluator": 32, "monitor": 32}
    assert decide.expected(hundredk)["llm_calls"] == 112
    off = {**hundredk, "llm": {**hundredk["llm"], "mode": "off"}}
    assert sum(decide.async_calls(off).values()) == decide.expected(off)["llm_calls"] == 0
    assert decide.async_calls(async_knobs(population=5, generations=2))["generator"] == 6
    assert decide.role_calls(hundredk) and not decide.role_calls(off)
    gen = knob_files.base()
    assert not decide.role_calls(gen) and decide.role_calls({**gen, "campaign": {**gen["campaign"], "notes": True}})


CONTEXT, REPLY_CAP = 32768, 4096                # the servers' context and the reply cap of the StellarAI knob file
PROMPT_BOUND = (CONTEXT - REPLY_CAP) * 3 // 4   # 21,504 characters: a quarter of what is left kept free


def test_every_roles_longest_prompt_fits_the_context_by_construction():
    """Each role's longest prompt in the asynchronous mode, at the StellarAI knob file's shape (the hundredk preset:
    population 16, 32 candidates, both lookups), built from decide's bounds: a Monitor's note and the Coordinator's
    guidance far past their reply caps are cut to NOTE_CHARS where a later prompt shows them, and one Evaluator call
    shows SHOWN_RESULTS results. The longest of each part is taken: every survivor and score the Generator can
    hold, the last two candidates' numbers, the longest abstract, every number shown at its widest plausible form
    (results of nine characters, q at full precision, m and the steps as monitor.score gives them), and for the
    Coordinator every result of the campaign judged in one cycle. Each prompt (its characters, as the layer counts
    them) is under PROMPT_BOUND, so it fits the context with the reply cap and a quarter to spare."""
    root = Path(__file__).resolve().parents[1]
    k = knob_files.validate(knob_files.load(preset="hundredk", paths=[root / "knobs/machines/stellar_ai.yaml",
                                                                       root / "knobs/machines/stellar_ai_nemotron.yaml"]),
                            modes=("generations", "async"))
    P, B, ranges = k["outer"]["population"], decide.async_budget(k), decide.ranges()
    assert (P, B, k["llm"]["reply_cap"], decide.asynchronous(k)) == (16, 32, REPLY_CAP, True)
    profile = {key: lo + (hi - lo) * 0.123456789 for key, (lo, hi) in ranges.items()}   # 6 places each
    wide = 1234567.8   # a shown number of nine characters

    def result(i: int) -> dict:
        return {"candidate_idx": i, "params": profile, **{key: wide for key in decide.SHOWN[:4]},
                "wall_ok": False, "pedestal_ok": False, "is_good": False}
    long_text, q = "y" * 5 * decide.NOTE_CHARS, math.pi / 300   # q: 20 characters at full precision
    numbers = {"m": -0.1234, "left_at": 47, "back_at": 48, "split_at": 46}

    abstract = {"abstract": max((asyncio.run(tools.abstract(f"a{i}")) for i in range(30)), key=lambda a: len(a["text"]))}
    gst = {**decide.async_generator_state("a99999", 99999, k, abstract), "survivors": [profile] * P,
           "scores": [[B - 1, 0.1234]] * P, "guidance": long_text, "held": 1}
    calls = [{"id": f"c{i}", "params": profile, "candidate_idx": i, "generation": i // P, "source": "mutation",
              "filled": None} for i in (B - 2, B - 1)]
    items = [{"candidate": B - 1 - j, "result": result(B - 1 - j), "q": q, "numbers": numbers, "note": long_text}
             for j in range(decide.SHOWN_RESULTS)]
    cst = {**decide.async_coordinator_state("a99999", 99999, k, {"constraints": asyncio.run(tools.constraints())}, 0.0),
           "window": [{key: result(i).get(key) for key in decide.WINDOW_KEYS} for i in range(B - P, B)],
           "arrived": B, "kept": B, "survivors": [{"candidate": i, "params": profile} for i in range(B - P, B)],
           "best": {**decide.parent().empty_best_result(), "te_core_metric": wide, "te_wall_eV": wide,
                    "te_ped_eV": wide, "is_good": False}}
    rows = [{"candidate": i, "kept": False, "score": 0.1234} for i in range(B)]
    prompts = {"generator": decide.async_generator_messages(gst, calls, k),
               "monitor": decide.note_messages(result(B - 1), numbers),
               "evaluator": decide.scores_messages(items),
               "coordinator": decide.guidance_messages(cst, rows, k)}
    sizes = {role: sum(len(m["content"]) for m in messages) for role, messages in prompts.items()}
    assert PROMPT_BOUND == 21504 and all(size < PROMPT_BOUND for size in sizes.values()), sizes
    for role, letter in (("generator", "y"), ("evaluator", "y")):   # each note and guidance cut where it is shown
        text = prompts[role][-1]["content"]
        assert letter * decide.NOTE_CHARS in text and letter * (decide.NOTE_CHARS + 1) not in text, role
    assert prompts["evaluator"][-1]["content"].count("note: ") == decide.SHOWN_RESULTS
    assert decide._cut("") == "" and decide._cut("abc") == "abc"
