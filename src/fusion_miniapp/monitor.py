"""The Monitor's numbers: each campaign's rolling trace statistics, and how one inner-loop trace compares with
them. Added instrumentation, not in IGNOS. Pure numpy and deterministic; no agent framework, no parent file.

A trace's steps (`steps`) are its inner loop's steps in the order they run: the XGC stage's rows (priming,
then each rollout step), then every fixed-point iteration of each puff in turn, each with its four values
(trace.COLUMNS "xgc" and "fixed_point"). Step k + 1 + j * c + i is puff j's iteration i (k rollout steps, c
the fixed-point cap). The scan rows, hand-off ranges and stage timings are summaries, not steps.

Each campaign keeps one `Rolling`: for every step and value an exponentially weighted mean and variance, one
side for good candidates and one for bad ones (`Side`). A side's n-th trace has weight max(1/n, 1/window):
its plain running mean and variance until it holds `window` traces, then each new trace counts 1/window.
Who is good is the caller's (the AUTONMI layer labels a generation's traces at its barrier, or in the
asynchronous mode each trace as the Evaluator scores its result).

For one trace (`Rolling.score`), with d_good and d_bad its distance at a step from each side's mean, in units
of the two sides' pooled spread there (root mean square over the step's values):
- left_at: the first step where both distances exceed BAND, so it looks like neither side (it left the
  typical band); None if it never does;
- back_at: the first step after that where it is within BAND of a side again (it came back); None if not;
- m, from -1 (its path looked like the bad, low-ranked candidates') to +1 (like the good, high-ranked
  ones'): (d_bad² - d_good²) / (d_bad² + d_good²) at each step, averaged with the two sides' squared
  separation at that step as weights, so the steps where the outcomes part count most;
- split_at: the first step where the two sides' means are more than BAND spreads apart, where the outcomes
  part (the campaign's, at the time of scoring).
Until both sides hold `warmup` traces, m is 0 and the steps are None.
"""
from __future__ import annotations

import numpy as np

from fusion_miniapp import trace

BAND = 2.0         # the typical band, and the sides' separation: 2 pooled spreads (root mean square over values)
REL_FLOOR = 1e-3   # a spread counts at least this share of the larger mean there (the traces are float32)
NONE = {"m": 0.0, "left_at": None, "back_at": None, "split_at": None}   # before warm-up


def steps(tr: dict) -> np.ndarray:
    """A trace's steps by their four values (rows by columns, float32), from trace.unpack."""
    a = trace.unpack(tr)
    fp = a["fixed_point"]
    return np.concatenate([a["xgc"], fp.reshape(-1, fp.shape[-1])])


def end_keep(xs: list[np.ndarray]) -> list[float]:
    """The keep-probability of each trace by its own end values, for the fixture's synthetic labels: ranked by
    the larger of te_wall and ti_wall at its last step, lower first, as the parent ranks the wall term
    (_normalised_ranks: the best 1, the worst 0, by position; a value that is not finite last)."""
    wall = np.array([max(float(x[-1, 1]), float(x[-1, 2])) for x in xs])
    order = np.argsort(np.where(np.isfinite(wall), wall, np.inf), kind="stable")
    u = np.empty(len(xs))
    u[order] = np.linspace(1.0, 0.0, len(xs))
    return u.tolist()


class Side:
    """One side's exponentially weighted mean and variance per step and value, and the traces it holds."""

    def __init__(self, shape: tuple, window: int) -> None:
        self.window, self.traces = window, 0
        self.n = np.zeros(shape, dtype=np.int64)   # the finite values seen, per step and value
        self.mean, self.var = np.zeros(shape), np.zeros(shape)

    def add(self, x: np.ndarray) -> None:
        ok = np.isfinite(x)
        self.n += ok
        a = np.where(ok, np.maximum(1.0 / np.maximum(self.n, 1), 1.0 / self.window), 0.0)
        d = np.where(ok, x - self.mean, 0.0)
        self.mean += a * d
        self.var = (1.0 - a) * (self.var + a * d * d)
        self.traces += 1


class Rolling:
    """One campaign's trace statistics (see the module's docstring), and the traces held until labelled: the
    Monitors hold each trace they score, and the caller takes a generation's to label them (or, in the
    asynchronous mode, each one as its result is scored). In the asynchronous mode it also holds the campaign's
    window: its last `window` scored results in arrival order (`recent`, decide.async_score's), and each one's
    last trace step when its trace is synthetic (`ends`, for the next labels; else None)."""

    def __init__(self, warmup: int, window: int) -> None:
        self.warmup, self.window = warmup, window
        self.good: Side | None = None
        self.bad: Side | None = None
        self.held: dict[int, dict[int, tuple[np.ndarray, bool]]] = {}   # generation -> candidate -> (steps, synthetic)
        self._fit: tuple | None = None   # what score needs, made at the first score after an add
        self.recent: list[dict] = []
        self.ends: list[np.ndarray | None] = []

    def hold(self, generation: int, candidate: int, x: np.ndarray, synthetic: bool) -> None:
        self.held.setdefault(generation, {})[candidate] = (x, synthetic)

    def take(self, generation: int) -> dict[int, tuple[np.ndarray, bool]]:
        return self.held.pop(generation, {})

    def take_one(self, generation: int, candidate: int) -> tuple[np.ndarray, bool] | None:
        """One held trace (the asynchronous mode), or None."""
        held = self.held.get(generation)
        out = held.pop(candidate, None) if held else None
        if held is not None and not held:
            del self.held[generation]
        return out

    def add(self, x: np.ndarray, good: bool) -> None:
        if self.good is None:
            self.good, self.bad = Side(x.shape, self.window), Side(x.shape, self.window)
        (self.good if good else self.bad).add(x)
        self._fit = None

    def ready(self) -> bool:
        return self.good is not None and min(self.good.traces, self.bad.traces) >= self.warmup

    def fit(self) -> tuple:
        """The sides' means, their pooled spread (at least REL_FLOOR of the larger of the two means; 1 where
        both are 0 and never varied), which values both sides have seen, the squared separation per step, and
        split_at."""
        if self._fit is None:
            g, b = self.good, self.bad
            known = (g.n > 0) & (b.n > 0)
            spread = np.sqrt((g.var + b.var) / 2)
            spread = np.maximum(spread, REL_FLOOR * np.maximum(abs(g.mean), abs(b.mean)))
            spread = np.where(spread > 0, spread, 1.0)
            sep2 = np.square(np.where(known, (g.mean - b.mean) / spread, 0.0)).mean(axis=1)
            apart = np.flatnonzero(sep2 > BAND ** 2)
            self._fit = (g.mean, b.mean, spread, known, sep2, int(apart[0]) if apart.size else None)
        return self._fit

    def score(self, x: np.ndarray) -> dict:
        """One trace's numbers (the module's docstring): left_at, back_at, m and split_at."""
        if not self.ready():
            return dict(NONE)
        good, bad, spread, known, sep2, split_at = self.fit()
        ok = known & np.isfinite(x)
        dg = np.square(np.where(ok, (x - good) / spread, 0.0)).mean(axis=1)
        db = np.square(np.where(ok, (x - bad) / spread, 0.0)).mean(axis=1)
        out = np.minimum(dg, db) > BAND ** 2
        left = back = None
        if out.any():
            left = int(np.argmax(out))
            inside = np.flatnonzero(~out[left + 1:])
            back = left + 1 + int(inside[0]) if inside.size else None
        both = dg + db
        r = np.where(both > 0, (db - dg) / np.where(both > 0, both, 1.0), 0.0)
        w = float(sep2.sum())
        m = float((sep2 * r).sum() / w) if w > 0 else 0.0
        return {"m": round(m, 4), "left_at": left, "back_at": back, "split_at": split_at}
