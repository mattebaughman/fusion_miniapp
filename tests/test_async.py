"""The asynchronous mode's rule (campaign.mode: async), as decide.py's pure functions: each agent's state and what
arrived give its new state and what it sends. No agent framework: these run in the base environment.

The results are the fixture's (FIXTURE data: a function of the profile, not an inner loop), a few marked
feasible by hand so that both of the parent's rankings are used."""
import copy
import csv
import json
import math

import numpy as np
import pytest
from async_replay import fixture, replay

from fusion_miniapp import decide, knobs as knob_files, monitor, record, sim


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
