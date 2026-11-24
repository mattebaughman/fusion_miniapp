"""monitor.py, the Monitor's trace statistics and numbers, and decide.keep_probabilities, the parent's own
keep-probability. No agent framework: these run in the base environment.

The traces are the fixture's synthetic smooth trace (trace.py), which has one planted split: candidates on either
side of the middle of a1_Te's range follow each other until fixed-point iteration fixed_point_cap // 2 of each
puff, then part. FIXTURE data, not an inner loop."""
import copy
import math

import numpy as np
import pytest

from fusion_miniapp import decide, knobs, monitor, sim, trace

SIM = knobs.load(preset="hundredk")["sim"]                    # 2 rollout steps, 3 puffs, cap 15: 48 steps
K, C = SIM["rollout_steps"], SIM["fixed_point_cap"]
SPLIT = K + 1 + C // 2                                        # puff 0's iteration c // 2: step 10
LO, HI = decide.ranges()["a1_Te"]


def population(seed: int) -> list[dict]:
    """The parent's initial population of 16, as a campaign of this seed draws it."""
    evo = decide.parent()
    return evo.initialize_population(16, decide.MODEL, decide.KIND, seed, delta_bounds=evo.DELTA_CLIP)


def steps(params: dict) -> np.ndarray:
    return monitor.steps(sim.evaluate_fixture(knobs=SIM, params=params, monitor={"capture": True})["trace"])


def side(params: dict) -> float:
    """a1_Te's place in its range: below 0.5 the low branch of the planted split, above it the high one."""
    return (params["a1_Te"] - LO) / (HI - LO)


def labelled(seed: int, warmup: int = 4) -> monitor.Rolling:
    """A campaign's statistics after its first generation, labelled as the layer labels a fixture's traces:
    good when the keep-probability from the trace's own end values is 0.5 or more, in candidate order."""
    xs = [steps(p) for p in population(seed)]
    stats = monitor.Rolling(warmup, 16)
    for x, q in zip(xs, monitor.end_keep(xs)):
        stats.add(x, q >= 0.5)
    return stats


def test_steps_are_the_inner_loops_steps_in_order():
    """The XGC stage's rows, then each puff's fixed-point iterations in turn, each with its four values."""
    rec = sim.evaluate_fixture(knobs=SIM, params=population(0)[0], monitor={"capture": True})
    x, a = monitor.steps(rec["trace"]), trace.unpack(rec["trace"])
    assert x.shape == (K + 1 + SIM["puffs"] * C, 4)
    assert np.array_equal(x[:K + 1], a["xgc"]) and np.array_equal(x[K + 1 + C + 3], a["fixed_point"][1, 3])


def test_a_side_is_the_running_mean_and_variance_then_exponentially_weighted():
    """Up to `window` traces a side's mean and variance are the plain ones (numpy's, over the traces); after that
    each new trace counts 1/window. A value that is not finite leaves that value's statistics as they were."""
    rng = np.random.default_rng(7)
    xs = rng.normal(size=(9, 3, 2))
    s = monitor.Side((3, 2), window=5)
    for i, x in enumerate(xs[:5]):
        s.add(x)
        assert np.allclose(s.mean, xs[:i + 1].mean(axis=0)) and np.allclose(s.var, xs[:i + 1].var(axis=0))
    mean, var = s.mean.copy(), s.var.copy()
    s.add(xs[5])
    d = xs[5] - mean
    assert np.allclose(s.mean, mean + d / 5) and np.allclose(s.var, (1 - 1 / 5) * (var + d * d / 5))
    x = xs[6].copy()
    x[0, 0] = np.nan
    mean, var = s.mean.copy(), s.var.copy()
    s.add(x)
    assert s.mean[0, 0] == mean[0, 0] and s.var[0, 0] == var[0, 0] and s.mean[1, 1] != mean[1, 1]
    assert (s.traces, s.n[0, 0], s.n[1, 1]) == (7, 6, 7)


def test_no_numbers_until_both_sides_hold_warmup_traces():
    x = steps(population(0)[0])
    stats = monitor.Rolling(warmup=4, window=16)
    for _ in range(3):
        stats.add(x, True)
        stats.add(x * 1.5, False)
    assert stats.score(x) == monitor.NONE and not stats.ready()
    stats.add(x, True)
    assert stats.score(x) == monitor.NONE              # the bad side holds 3
    stats.add(x * 1.5, False)
    assert stats.ready() and stats.score(x)["m"] > 0.9


@pytest.mark.parametrize("seed", range(8))
def test_the_planted_split_is_found(seed):
    """After one generation of 16, labelled from the traces' end values: the sides part at the planted step, and
    every candidate of the next generation that is clearly on one branch is scored like it, m > 0 on the low
    branch (lower wall temperatures at the end: the good side) and m < 0 on the high one."""
    stats = labelled(seed)
    assert stats.ready() and stats.fit()[5] == SPLIT
    for p in population(seed + 1000):
        got = stats.score(steps(p))
        assert got["split_at"] == SPLIT
        if abs(side(p) - 0.5) > 0.05:
            assert math.copysign(1, got["m"]) == (1 if side(p) < 0.5 else -1), (side(p), got)


def test_a_candidate_on_the_ridge_leaves_the_band_at_the_split_and_comes_back_at_the_next_puff():
    """A candidate in the middle of a1_Te's range follows neither branch: it leaves the typical band once the
    sides have parted (from puff 0's iteration c // 2 on) and comes back at puff 1's first iteration, where the
    paths rejoin."""
    stats = labelled(0)
    ridge = {**population(5)[0], "a1_Te": (LO + HI) / 2}
    got = stats.score(steps(ridge))
    assert SPLIT <= got["left_at"] < K + 1 + C and got["back_at"] == K + 1 + C, got


def test_an_excursion_is_found_where_it_leaves_and_comes_back():
    """A typical trace with puff 0's te_wall tripled at iterations 2 to 4 (before the split): it left the typical
    band at iteration 2 and came back at iteration 5; unchanged, it never leaves."""
    stats = labelled(0)
    p = next(p for p in population(3) if side(p) < 0.4)
    x = steps(p)
    assert stats.score(x)["left_at"] is None
    x[K + 1 + 2:K + 1 + 5, 1] *= 3
    got = stats.score(x)
    assert (got["left_at"], got["back_at"]) == (K + 1 + 2, K + 1 + 5), got


def test_end_keep_ranks_by_the_wall_at_the_trace_end():
    """The larger of te_wall and ti_wall at the last step, as the parent's wall key; lower is better, as the
    parent ranks the wall term: best 1, worst 0, by position; not finite last."""
    def x(te, ti):
        a = np.zeros((2, 4))
        a[-1, 1], a[-1, 2] = te, ti
        return a
    assert monitor.end_keep([x(100, 300), x(200, 50), x(np.nan, 10), x(250, 10)]) == pytest.approx([1 / 3, 1, 0, 2 / 3])
    assert monitor.end_keep([x(1, 1)]) == [1.0] and monitor.end_keep([]) == []


# --- q: the parent's keep-probability -------------------------------------------------------------------------

def result(te_core, good=False, wall=500.0, wall_ok=False) -> dict:
    return {"te_core_metric": te_core, "is_good": good, "wall_ok": wall_ok, "te_wall_eV": wall, "ti_wall_eV": wall - 1}


GENERATIONS = {
    "mixed, with ties and a NaN among the feasible": [
        result(900, True), result(950, True), result(950, True), result(float("nan"), True), result(700, True),
        result(990), result(400, wall=100), result(800, True), result(800, True), result(1200), result(600, True),
        result(990, wall=80, wall_ok=True), result(300), result(500, True), result(100), result(650)],
    "all infeasible, with ties and NaNs": [
        result(900, wall=400), result(900, wall=400), result(float("nan"), wall=300), result(850, wall=float("nan")),
        result(700, wall=200, wall_ok=True), result(1000, wall=900), result(900, wall=400), result(600, wall=250),
        result(950, wall=500), result(950, wall=500), result(400, wall=100), result(880, wall=420)],
    "one feasible": [result(500), result(600, True), result(700), result(800)],
    "two": [result(500, wall=300), result(600, wall=200)],
    "one": [result(500)],
}


@pytest.mark.parametrize("name", GENERATIONS)
def test_q_is_the_parents_keep_probability(name, capsys):
    """Replaying the parent's select_survivors with a seeded generator, a result survives exactly when the draw
    made for it is below its q (best: q 1, kept with no draw; worst and unranked: q 0, never kept); the draws go
    to the ranked results from the second best on, in the order of their q, as the parent draws them."""
    evo, gen = decide.parent(), GENERATIONS[name]
    for w in (0.0, 0.7, 1.0):
        q = decide.keep_probabilities(gen, w)
        assert len(q) == len(gen) and all(0.0 <= v <= 1.0 for v in q) and max(q) == 1.0
        middle = sorted((i for i, v in enumerate(q) if 0.0 < v < 1.0), key=lambda i: -q[i])
        for seed in range(200):
            evaluated = copy.deepcopy(gen)
            rng = np.random.default_rng(seed)
            kept = {id(e) for e in evo.select_survivors(evaluated, rng, w)}
            replay = np.random.default_rng(seed)
            draws = {i: replay.random() for i in middle}
            assert replay.bit_generator.state == rng.bit_generator.state   # as many draws as the parent made
            for i, e in enumerate(evaluated):
                assert (id(e) in kept) == (q[i] == 1.0 or (i in draws and draws[i] < q[i])), (name, w, seed, i)
    capsys.readouterr()   # the parent's prints for a generation with nothing feasible
