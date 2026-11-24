"""The decoding metric's arithmetic (record.occupancy with references, record.sampled, record.summed) against
brute-force recounts, and AUTONMI's vLLM sampler against the fake vLLM server's /metrics (a FIXTURE, not a
model: tests/fake_vllm.py). The runs of the AUTONMI layer that use them are in tests/test_layer_autonmi.py."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
from fake_vllm import FakeVllm

from fusion_miniapp import record


def brute_force(starts, ends, lo, hi, refs):
    """Intervals [start, end) open on each stretch between two consecutive event times, counted one by one;
    the maximum over the stretches inside the window [lo, hi]."""
    cuts = np.unique(np.concatenate([starts, ends, [lo, hi]]))
    top, area, at = 0, 0.0, {name: {p: 0.0 for p in record.PCTS} for name in refs}
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = (a + b) / 2
        n = sum(1 for s, e in zip(starts, ends) if s <= m < e)
        w = max(0.0, min(b, hi) - max(a, lo))
        top = max(top, n) if w > 0 else top
        area += n * w
        for name, ref in refs.items():
            for p in record.PCTS:
                at[name][p] += w if n >= p / 100 * ref else 0.0
    return top, area / (hi - lo), at


@pytest.mark.parametrize("ties", (False, True))
def test_occupancy_matches_a_brute_force_recount(ties):
    """The maximum, the time-weighted mean and the seconds at or above 50, 90 and 95% of each reference, in a
    window that cuts intervals off at both ends (half the trials: a window well inside, so that the busiest
    stretch can lie outside it, which the maximum must not count); with whole-second times (ties), an end
    counts before a start."""
    rng = np.random.default_rng(7)
    for trial in range(40):
        n = int(rng.integers(1, 120))
        starts = rng.uniform(0, 100, n)
        ends = starts + rng.exponential(10, n)
        if ties:
            starts, ends = np.floor(starts), np.floor(ends) + 1
        lo, hi = float(rng.uniform(-5, 20)), float(rng.uniform(60, 130))
        if trial % 2:
            lo, hi = float(rng.uniform(20, 45)), float(rng.uniform(50, 75))
        refs = {"hosted": n, "cap": int(rng.integers(1, max(2, n // 3))), "unknown": None}
        got = record.occupancy(starts, ends, lo, hi, refs=refs, digits=6)
        top, mean, at = brute_force(starts, ends, lo, hi, {k: v for k, v in refs.items() if v})
        assert got["max"] == top, trial
        assert got["mean"] == pytest.approx(mean, abs=1e-6), trial
        assert got["time_ge_pct_s"]["unknown"] is None
        for name in ("hosted", "cap"):
            assert got["time_ge_pct_s"][name]["of"] == refs[name]
            for p in record.PCTS:
                assert got["time_ge_pct_s"][name][str(p)] == pytest.approx(at[name][p], abs=2e-3), (trial, name, p)


def test_occupancy_keeps_its_old_fields():
    """campaigns_live's n_ref and two-decimal mean are unchanged; no intervals give zeros, references too."""
    assert record.occupancy([0, 1, 2], [3, 3, 4], 0, 4, n_ref=3) == {"max": 3, "mean": 1.75, "time_ge_95pct_s": 1.0}
    empty = record.occupancy([], [], 0, 4, refs={"a": 2})
    assert empty == {"max": 0, "mean": 0.0, "time_ge_pct_s": {"a": {"of": 2, "50": 0.0, "90": 0.0, "95": 0.0}}}


def test_a_sampled_count_holds_each_sample_until_the_next():
    """record.sampled: a step function of samples, against a brute-force sum over a fine grid; record.summed
    adds sources sampled at different times, each at its latest value."""
    t, v = [0.0, 2.0, 5.0, 6.0], [1, 4, 0, 3]
    got = record.sampled(t, v, 1.0, 8.0, refs={"r": 4})
    # held: 1 on [1, 2), 4 on [2, 5), 0 on [5, 6), 3 on [6, 8): mean (1 + 12 + 0 + 6) / 7
    assert got == {"samples": 4, "max": 4, "mean": round(19 / 7, 4),
                   "time_ge_pct_s": {"r": {"of": 4, "50": 5.0, "90": 3.0, "95": 3.0}}}
    assert record.sampled([], [], 0, 1) is None
    times, total = record.summed([([0.0, 2.0], [1, 3]), ([1.0], [10]), ([], [])])
    assert times.tolist() == [0.0, 1.0, 2.0] and total.tolist() == [1.0, 11.0, 13.0]
    _, with_gap = record.summed([([0.0, 1.0], [None, 2])])
    assert with_gap.tolist() == [0.0, 2.0]


def test_a_server_that_stops_answering_is_not_held_at_its_last_reading():
    """record.summed(hold_s=): server A read every 2 s with 5 running, server B only at 0 and 2 s with 100: B
    counts until 2 + hold_s (5 s), then 0, not 100 to the end; record.unheld_s gives B's 3 s without a reading
    and A's none. Without hold_s (the old sum) B would be held, a mean of 105."""
    a = ([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], [5] * 6)
    b = ([0.0, 2.0], [100, 100])
    t, total = record.summed([a, b], hold_s=5.0)
    assert t.tolist() == [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 15.0]   # and A's own end, hold_s after its last
    assert total.tolist() == [105.0, 105.0, 105.0, 105.0, 5.0, 5.0, 5.0, 0.0]
    assert record.sampled(t, total, 0.0, 10.0)["mean"] == (105 * 7 + 5 * 3) / 10
    assert record.sampled(*record.summed([a, b]), 0.0, 10.0)["mean"] == 105.0
    assert (record.unheld_s(a[0], 0.0, 10.0, 5.0), record.unheld_s(b[0], 0.0, 10.0, 5.0)) == (0.0, 3.0)
    # a gap shorter than hold_s (one missed reading) is bridged; one longer is not, and the server comes back
    t, total = record.summed([([0.0, 4.0, 12.0], [1, 2, 3])], hold_s=5.0)
    assert (t.tolist(), total.tolist()) == ([0.0, 4.0, 9.0, 12.0, 17.0], [1.0, 2.0, 0.0, 3.0, 0.0])
    assert record.unheld_s([0.0, 4.0, 12.0], -1.0, 14.0, 5.0) == 1.0 + 3.0


def test_summed_with_a_hold_matches_a_brute_force_recount():
    """Random servers, each read at random gaps (some longer than hold_s): at the middle of every stretch of
    the summed series, its value is each server's latest reading still within hold_s, added up."""
    rng = np.random.default_rng(11)
    for trial in range(20):
        hold = float(rng.uniform(1, 6))
        series = []
        for _ in range(int(rng.integers(1, 6))):
            t = np.cumsum(rng.choice([0.5, 1.0, 2.0, 8.0], size=int(rng.integers(1, 15))))
            series.append((t.tolist(), rng.integers(0, 50, t.size).tolist()))
        t, total = record.summed(series, hold_s=hold)
        for a, b, got in zip(t, np.append(t[1:], t[-1] + 1), total):
            m = (a + b) / 2
            want = 0
            for ts, vs in series:
                i = int(np.searchsorted(ts, m, side="right")) - 1
                want += vs[i] if i >= 0 and m < ts[i] + hold else 0
            assert got == want, (trial, m)


def test_the_decoding_metric_leaves_out_stale_readings_and_missing_gauges(tmp_path):
    """layer_autonmi.decoding_metric on a vllm.json written by hand: server B stops answering after 2 s (its
    reading counts 2.5 intervals more, then 0, with its stale seconds recorded) and gives no KV-cache gauge (an
    older vLLM's name): that statistic is null with the reason, not zeros. A sweep cut off mid-write (SIGTERM:
    a time without its gauges) is cut to whole readings."""
    layer = pytest.importorskip("fusion_miniapp.layer_autonmi", reason="needs the AUTONMI layer's environment")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "agents-p000.jsonl").write_text(json.dumps({"t_start": 100.0, "t_end": 110.0}) + "\n")
    a = {"t": [100.0 + 2 * i for i in range(6)], "num_requests_running": [5] * 6, "num_requests_waiting": [1] * 6,
         "kv_cache_usage_perc": [0.2] * 6}
    b = {"t": [100.0, 102.0, 104.5], "num_requests_running": [100, 100], "num_requests_waiting": [3, 3],
         "kv_cache_usage_perc": [None, None]}
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "vllm.json").write_text(json.dumps({"interval_s": 2.0, "errors": 4,
                                                             "series": {"http://a": a, "http://b": b}}))
    knobs = {"llm": {"mode": "openai", "stub_latency_s": 0.001}}
    rec = {"speed": {"agent_phase_s": 10.0, "loop_lag": []}, "wall_seconds": 12.0}
    got = layer.decoding_metric(knobs, tmp_path, rec, {}, {"hosted_agents": 1}, None)
    d = got["decoding_at_servers"]
    assert d["hold_s"] == 5.0 and d["stale_s"] == [0.0, 3.0] and d["sample_errors"] == 4
    assert d["running"]["max"] == 105 and d["running"]["mean"] == (105 * 7 + 5 * 3) / 10
    assert d["waiting"]["mean"] == (4 * 7 + 1 * 3) / 10
    assert d["kv_cache_usage"] is None and "kv_cache_usage_perc" in d["gauges_missing"]["kv_cache_usage_perc"]
    assert got["headline"]["figure"] == "decoding_at_servers" and got["headline"]["served_by"] == []


def test_the_live_line_leaves_out_a_server_that_stopped_answering():
    """LiveLine's decoding_at_servers sums the servers' latest running counts, only of readings within hold_s
    (2.5 intervals): a server last read 28 s ago is left out, not held at its last value."""
    layer = pytest.importorskip("fusion_miniapp.layer_autonmi", reason="needs the AUTONMI layer's environment")
    now = time.time()

    class Sampler:
        interval = 2.0
        series = {"a": {"t": [now - 4, now - 2, now - 0.5], "num_requests_running": [1, 2, 3]},
                  "b": {"t": [now - 30, now - 28], "num_requests_running": [50, 60]}}
    live = layer.LiveLine()
    live.sampler = Sampler()
    assert live._decoding() == 3
    Sampler.series["b"]["t"].append(now - 0.2)
    Sampler.series["b"]["num_requests_running"].append(70)
    assert live._decoding() == 73


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def test_autonmis_sampler_reads_the_fake_vllm_servers_metrics():
    """AUTONMI's VllmSampler parses the fixture's /metrics: 5 requests at once against a batch of 2 are 2
    running and 3 waiting, with KV-cache use for the 2; afterwards none, and the counters AUTONMI compares
    with the fleets' (prompt and generated tokens, successful requests) count all 5."""
    pytest.importorskip("rascal.telemetry.vllm", reason="needs the AUTONMI framework's clone (rascal)")
    from rascal.telemetry.vllm import VllmSampler, fetch, parse

    body = {"model": "stand-in", "messages": [{"role": "user", "content": "Summarise.\nnothing more"}]}
    with FakeVllm(latency_s=1.5, batch=2) as server:
        threads = [threading.Thread(target=_post, args=(server.url, body)) for _ in range(5)]
        for x in threads:
            x.start()
        deadline = time.time() + 10
        while server.running + server.waiting < 5:
            assert time.time() < deadline
            time.sleep(0.01)
        values, cache = parse(fetch(server.url, None))
        assert (values["num_requests_running"], values["num_requests_waiting"]) == (2.0, 3.0)
        assert values["kv_cache_usage_perc"] == pytest.approx(2 * 150 / 65_536)
        assert cache == {"block_size": "16", "num_gpu_blocks": "4096"}
        for x in threads:
            x.join()
        sampler = VllmSampler([server.url], None, interval=0.05)
        sampler.sample()
        raw = sampler.raw()
    s = raw["series"][server.url]
    assert raw["errors"] == 0 and len(s["t"]) == 1
    assert (s["num_requests_running"], s["num_requests_waiting"]) == ([0.0], [0.0])
    assert (s["prompt_tokens_total"], s["generation_tokens_total"], s["request_success_total"]) == ([500.0], [250.0],
                                                                                                    [5.0])
    assert server.running_max == 2 and server.waiting_max == 3


def test_the_fake_vllm_server_answers_what_a_run_asks_first():
    """/health without a key; /v1/models (the served name) and chat completions only with it, when it has one."""
    with FakeVllm(model="served-name", api_key="k") as server:
        with urllib.request.urlopen(server.url + "/health", timeout=10) as r:
            assert r.status == 200
        req = urllib.request.Request(server.url + "/v1/models", headers={"Authorization": "Bearer k"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert [m["id"] for m in json.loads(r.read())["data"]] == ["served-name"]
        with pytest.raises(urllib.error.HTTPError) as no_key:
            urllib.request.urlopen(server.url + "/v1/models", timeout=10)
        assert no_key.value.code == 401
