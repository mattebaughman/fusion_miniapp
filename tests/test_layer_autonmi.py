"""The AUTONMI layer (layer_autonmi.py) against the LangGraph layer: the same campaigns, the same run record.

Each run is a process of its own, as in production (OpenTelemetry's providers are set once per process),
started in the test's own folder (AsyncFlow leaves a folder for every engine in the working directory).
Two runs agree when compare_records says so, as for two runs on two machines. The model server is a
FIXTURE (not a model): a stand-in for a vLLM server on 127.0.0.1, on a thread of the test (fake_vllm.py).

On the AUTONMI layer each campaign is 3 + campaign.slots hosted agents (a<i>/coordinator, a<i>/generator,
a<i>/evaluator and the Monitor slots a<i>/monitor0, ...) that talk through messages and the campaign's candidate
queue; the tests also check their rows, messages and per-role counts, and that the results do not depend on how
the campaigns are laid out (populations, caps, workers, starting together, Monitor slots).

Each AUTONMI run is also compared with a reference recorded from the LangGraph layer
(tests/references/<name>.json, written by tests/references/record_references.py, which runs these
cases): its results_sha256, exact counts, parent files and model requests. The live LangGraph run is
compared with the same reference, so a change to either layer shows.

Needs the AUTONMI framework (the package `autonmi`, installed separately) and Flowgentic 0.6
(requirements-autonmi.txt); skipped without them.
"""
import asyncio
import contextlib
import copy
import csv
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
from fake_vllm import FakeVllm

from fusion_miniapp import compare_records, decide, knobs as knob_files, record


def _missing(*modules):
    out = []
    for m in modules:
        try:
            if importlib.util.find_spec(m) is None:
                out.append(m)
        except ModuleNotFoundError:
            out.append(m)
    return out


WHY = "the AUTONMI layer's environment: the AUTONMI framework, installed separately, and requirements-autonmi.txt"
pytest.importorskip("autonmi", reason=f"needs {WHY}")
for _module in ("wfMiniAPI.api", "radical.asyncflow"):
    pytest.importorskip(_module, reason=f"needs {WHY}")
try:
    from flowgentic.host import current_agent  # noqa: F401  (Flowgentic 0.6's public runner API)
except ImportError:
    pytest.skip(f"needs Flowgentic 0.6: {WHY}", allow_module_level=True)

ENTRY = {"autonmi": "fusion_miniapp.layer_autonmi", "langgraph": "fusion_miniapp.run"}
ROLES = ("coordinator", "generator", "evaluator", "monitor")   # each campaign's roles on the AUTONMI layer


def names(slots: int = 2) -> list[str]:
    """A campaign's hosted agents, in order: the three outer-loop roles, then the Monitor slots."""
    return ["coordinator", "generator", "evaluator", *(f"monitor{k}" for k in range(slots))]


def knob_file(tmp_path, knobs: dict) -> str:
    path = tmp_path / "knobs.yaml"
    path.write_text(json.dumps(knobs))   # JSON is YAML
    return str(path)


def run(layer, tmp_path, *args, env=None, code=0) -> dict:
    """One run of a layer in tmp_path/<layer>; its run record."""
    proc = subprocess.run([sys.executable, "-m", ENTRY[layer], *args, "--out", str(tmp_path / layer)],
                          cwd=tmp_path, env=env, check=False, timeout=900, capture_output=True, text=True)
    assert proc.returncode == code, proc.stderr[-3000:]
    return json.loads((tmp_path / layer / "run_record.json").read_text())


def lines(tmp_path, layer) -> list[dict]:
    return [json.loads(x) for x in (tmp_path / layer / "agents.jsonl").read_text().splitlines()]


def agree(tmp_path, a: dict, b: dict) -> list[str]:
    """compare_records' differences between the AUTONMI run `a` and the LangGraph run `b`."""
    text = {layer: (tmp_path / layer / "agents.jsonl").read_text().splitlines() for layer in ENTRY}
    return compare_records.compare(a, b, agents_a=text["autonmi"], agents_b=text["langgraph"])["results"]


CASES = {
    "llm refills, one random stream, all live": {},
    "mutation, two random streams, no logs, three live": {
        "outer": {"generator": "mutation", "rng": "per_agent", "logs": "none"}, "agents": {"max_live": 3}},
}


def fixture_knobs(case: str) -> dict:
    over = CASES[case]
    return {"outer": {"population": 4, "generations": 2, "parent_stdout": "drop", **over.get("outer", {})},
            "agents": over.get("agents", {})}


FIXTURE_ARGS = ("--preset", "smoke", "--sim", "fixture", "--agents", "20")
REAL_KNOBS = {"sim": {"device": "cpu"}, "outer": {"population": 3, "generations": 2, "parent_stdout": "drop"},
              "workers": {"sim_workers": 2}}
REAL_ARGS = ("--preset", "smoke", "--sim", "real", "--agents", "2")
SERVER_KNOBS = {"llm": {"mode": "openai", "model": "stand-in"},
                "outer": {"population": 4, "generations": 3, "parent_stdout": "drop"}}
SERVER_ARGS = ("--preset", "smoke", "--sim", "fixture", "--agents", "12")

# The recorded references, tests/references/<name>.json: each case's knobs and arguments (besides --knobs and
# --out) as the tests below run them. The real stand-ins' floats could differ in the last digits on another OS
# or BLAS build (a precaution: on the laptop that recorded it, 1, 2, 3 and 6 threads per worker gave the same
# digest), so that reference also keeps its campaigns, compared within compare_records' tolerance when the
# digests differ, and then of its parent files only config.json, which holds no results.
REFERENCES = {
    "fixture_llm_refills": {"knobs": fixture_knobs("llm refills, one random stream, all live"), "args": FIXTURE_ARGS},
    "fixture_mutation_no_logs": {"knobs": fixture_knobs("mutation, two random streams, no logs, three live"),
                                 "args": FIXTURE_ARGS},
    "real_cpu": {"knobs": REAL_KNOBS, "args": REAL_ARGS, "real": True},
    "model_server": {"knobs": SERVER_KNOBS, "args": SERVER_ARGS, "server": True},
}
REFERENCE_OF = dict(zip(CASES, ("fixture_llm_refills", "fixture_mutation_no_logs")))
REFERENCES_DIR = Path(__file__).resolve().parent / "references"
PARENT_FILES = ("config.json", "evolution_log.csv", "best_result.json")
CHECKOUT_FIELDS = ("git", "constraints_path")   # config.json's fields that name the checkout: its commit, its path


def parent_hashes(out: Path) -> dict:
    """sha256 of each campaign's parent files, config.json without the fields that name the checkout."""
    hashes = {}
    for d in sorted((out / "campaigns").glob("a*")):
        hashes[d.name] = {}
        for f in PARENT_FILES:
            data = (d / f).read_bytes()
            if f == "config.json":   # written with json.dumps(indent=2): a round trip keeps the other bytes
                data = json.dumps({k: v for k, v in json.loads(data).items() if k not in CHECKOUT_FIELDS},
                                  indent=2).encode()
            hashes[d.name][f] = hashlib.sha256(data).hexdigest()
    return hashes


def campaign_hashes(out: Path) -> dict:
    """Each campaign's sha256 (its comparable part), as results_sha256 is made of them (record.write)."""
    rows = [json.loads(x) for x in (out / "agents.jsonl").read_text().splitlines()]
    return {r["agent_id"]: hashlib.sha256(json.dumps(record.comparable(r), sort_keys=True, default=str).encode())
            .hexdigest() for r in rows}


def request_bodies(sent: list[dict]) -> list[str]:
    """The model requests, sorted, without their unset fields."""
    return sorted(json.dumps({k: v for k, v in body.items() if v is not None}, sort_keys=True) for body in sent)


def reference(name: str) -> dict:
    """A recorded reference, checked to be of the case the test runs (otherwise it must be recorded again)."""
    ref = json.loads((REFERENCES_DIR / f"{name}.json").read_text())
    case = REFERENCES[name]
    assert (ref["knobs"], ref["args"]) == (case["knobs"], list(case["args"])), \
        f"tests/references/{name}.json is of another case: record it again (record_references.py)"
    return ref


def against_reference(out: Path, rec: dict, ref: dict) -> list[str]:
    """A run's differences from its recorded reference: results_sha256 (for the real stand-ins, else each
    campaign within compare_records' tolerance), the exact counts, and the parent files (for the real
    stand-ins with another digest, config.json alone)."""
    bad = []
    same = rec["results_sha256"] == ref["results_sha256"]
    if not same:
        if "campaigns" in ref:
            bad += compare_records.compare_campaigns((out / "agents.jsonl").read_text().splitlines(),
                                                     [json.dumps(c) for c in ref["campaigns"]])
        else:
            got, want = campaign_hashes(out), ref["campaign_sha256"]
            bad.append(f"results_sha256 {rec['results_sha256']} != the reference's {ref['results_sha256']}; campaigns "
                       f"that differ: {sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))[:20]}")
    bad += [f"counts.{k}: {rec['counts'].get(k)} != {v}" for k, v in ref["counts"].items() if rec["counts"].get(k) != v]
    if "parent_files" in ref:
        got, want = parent_hashes(out), ref["parent_files"]
        if not same and "campaigns" in ref:   # floats apart within the tolerance: the files' bytes differ too
            got, want = ({c: {"config.json": h["config.json"]} for c, h in x.items()} for x in (got, want))
        if got != want:
            bad.append("the parent files differ from the reference's")
    return bad


@pytest.mark.parametrize("case", CASES)
def test_the_autonmi_layer_gives_the_langgraph_layers_results(tmp_path, case):
    """20 fixture campaigns in 2 populations: the same campaigns (results_sha256), counts and knobs as the
    LangGraph layer's run, and the same parent files for each campaign; and both as their recorded reference."""
    from fusion_miniapp.layer_autonmi import AUTONMI_COMMIT, FULL_GC_THRESHOLD

    knobs = fixture_knobs(case)
    common = (*FIXTURE_ARGS, "--knobs", knob_file(tmp_path, knobs))
    a = run("autonmi", tmp_path, *common, "--populations", "2")
    b = run("langgraph", tmp_path, *common)
    assert (a["agent_layer"], a["status"], a["counts"]["agents"], a["counts"]["sims"]) == ("autonmi", "pass", 20, 160)
    assert agree(tmp_path, a, b) == []
    ref = reference(REFERENCE_OF[case])
    for layer, rec in (("autonmi", a), ("langgraph", b)):
        assert against_reference(tmp_path / layer, rec, ref) == [], layer
    pin = a["autonmi"]["framework_commit"]   # the commit imported: unknown only when neither git nor the installer says
    assert pin["pin"] == AUTONMI_COMMIT and isinstance(pin["matches_pin"], bool) and "path" not in pin, pin
    assert pin["commit"] or pin["from"] is None, pin
    assert a["autonmi"]["ok"] and a["autonmi"]["populations"] == 2, a["autonmi"]
    tel = a["autonmi"]["telemetry"]   # every task ended, in both populations
    assert (tel["complete"], tel["reasons"], sorted(tel["populations"])) == (True, [], ["p000", "p001"])
    assert tel["tasks"]["abandoned"] == 0 and tel["tasks"]["created"] == tel["tasks"]["completed"] > 0
    assert a["autonmi"]["asyncflow_running_shim"] is True   # the run that agreed had the shim (on by default)
    assert a["autonmi"]["gc"] == {"automatic_full_collections": False,   # and no automatic full collections
                                  "thresholds": [700, 10, FULL_GC_THRESHOLD]}, a["autonmi"]["gc"]
    assert 0 < a["counts"]["suspensions"] <= 160        # at most one per evaluation: a Monitor's wait for it
    assert_agents_per_campaign(tmp_path / "autonmi", a)
    if knobs["outer"].get("logs", "per_agent") == "per_agent":
        for f in ("config.json", "evolution_log.csv", "best_result.json"):   # the parent's per-campaign files
            for c in ("a0", "a19"):
                assert (tmp_path / "autonmi" / "campaigns" / c / f).read_bytes() == \
                    (tmp_path / "langgraph" / "campaigns" / c / f).read_bytes(), (c, f)
    assert not list(tmp_path.glob("autonmi/agents-p*.jsonl"))   # merged into agents.jsonl, then removed


def app_rows(out: Path) -> list[dict]:
    """AUTONMI's rows, one per hosted agent."""
    return [json.loads(x) for f in sorted((out / "app").glob("agents-p*.jsonl")) for x in f.read_text().splitlines()]


def assert_agents_per_campaign(out: Path, rec: dict) -> None:
    """Each campaign is 3 + campaign.slots hosted agents (a<i>/coordinator, generator, evaluator, monitor0, ...),
    consecutive and in one population, each ended normally (finished); every message sent was received once, the
    count computed from P, G, the slots and the roles, with none left (the queue's included); each role's model
    calls exactly as its campaign counted them (the Coordinator's summary, the Generator's refills, none by the
    Evaluator or the Monitors) and its lookups as decision 4(b) splits them; the campaign's P x G evaluations are
    its Monitors', each with its trace (monitor.capture) and exactly one report; only the Monitors suspend; the
    lookups are AUTONMI's count; the store holds no saved state at the end, and each campaign's memory under
    campaign/a<i>."""
    knobs, n = rec["knobs"], rec["counts"]["agents"]
    G, P, S = knobs["outer"]["generations"], knobs["outer"]["population"], knobs["campaign"]["slots"]
    lines = {r["agent_id"]: r for r in lines_of(out)}
    rows = app_rows(out)
    assert [r["agent_id"] for r in rows] == [f"{c}/{name}" for c in sorted(lines, key=lambda c: int(c[1:]))
                                            for name in names(S)]
    assert len({(r["campaign"], r["population"]) for r in rows}) == n   # never split over populations
    assert all(r["finished"] and r["status"] == "done" and r["messages_left"] == r["messages_dropped"] == 0
               for r in rows)
    sent = {role: sum(r["messages_sent"] for r in rows if r["role"] == role) for role in ROLES}
    received = {role: sum(r["messages_received"] for r in rows if r["role"] == role) for role in ROLES}
    assert sent == {"coordinator": n * (G + 2), "generator": n * (G + P * G + S), "evaluator": n * G,
                    "monitor": n * P * G}
    assert received == {"coordinator": n * 2 * G, "generator": n * (G + 1), "evaluator": n * (P * G + 1),
                        "monitor": n * (P * G + S)}
    assert sum(sent.values()) == sum(received.values()) == n * (3 * G + 2 + 2 * P * G + S)   # each once, none left
    for r in rows:
        line = lines[r["campaign"]]
        calls = {"coordinator": 1, "generator": line["n_llm_calls"] - 1, "evaluator": 0, "monitor": 0}[r["role"]]
        roles_calls = [c for c in line["llm_call_details"] if c["role"] == r["role"]]
        assert r["model_calls"] == calls == len(roles_calls), r
        assert r["lookups"] == {"coordinator": 1, "generator": 1, "evaluator": 0, "monitor": 0}[r["role"]], r
        assert r["suspensions"] == 0 or r["role"] == "monitor", r
        if r["role"] == "monitor":   # one evaluation, one trace (monitor.capture) and one report per candidate taken
            assert r["simulations"] == r["reports"] == r["messages_received"] - 1, r
            assert r["traces"] == knobs["monitor"]["capture"] * r["simulations"], r
        else:
            assert r["simulations"] == 0, r
    for c in lines:   # a campaign's Monitors evaluate its P x G candidates between them
        assert sum(r["simulations"] for r in rows if r["campaign"] == c and r["role"] == "monitor") == P * G, c
    by = rec["counts"]["by_role"]
    assert {role: by[role]["model_calls"] for role in ROLES} == {
        "coordinator": n, "generator": rec["counts"]["llm_calls"] - n, "evaluator": 0, "monitor": 0}
    assert by["monitor"]["suspensions"] == rec["counts"]["suspensions"]
    assert by["monitor"]["simulations"] == by["monitor"]["reports"] == rec["counts"]["sims"]   # a report per inner loop
    assert by["monitor"]["traces"] == knobs["monitor"]["capture"] * rec["counts"]["sims"]
    assert all(by[role][k] == 0 for role in ROLES[:3] for k in ("traces", "reports", "simulations"))
    assert (rec["counts"]["campaigns"], rec["counts"]["hosted_agents"], rec["counts"]["agents_per_campaign"]) == \
        (n, (3 + S) * n, 3 + S)
    checks = {c["name"]: c for c in json.loads((out / "app" / "checks.json").read_text())}
    assert {k: c["ok"] for k, c in checks.items()} == dict.fromkeys((
        "every_campaign_reports", "every_agent_finishes", "evaluations_and_model_calls_as_planned",
        "every_message_received", "lookups_match_autonmi", "store_holds_only_the_memories",
        "every_agent_has_a_row"), True), checks
    assert checks["store_holds_only_the_memories"]["detail"] == {"saved_states_left": 0, "memories": n,
                                                                  "campaigns_reported": n}


def lines_of(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "agents.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("per_process", (1000, 1))
@pytest.mark.parametrize("layer", ENTRY)
def test_an_offset_run_hosts_the_same_campaigns(tmp_path, layer, per_process):
    """agents.first = 18, agents.count = 2: campaigns a18 and a19 alone, with seeds 18 and 19, give what they give
    in the 20-campaign reference (their results and parent files), in one agent process or population and in
    one each."""
    knobs = fixture_knobs("llm refills, one random stream, all live")
    knobs["agents"] = {**knobs["agents"], "first": 18, "per_process": per_process}
    rec = run(layer, tmp_path, "--preset", "smoke", "--sim", "fixture", "--agents", "2",
              "--knobs", knob_file(tmp_path, knobs))
    assert (rec["status"], rec["counts"]["agents"], rec["knobs"]["agents"]["first"]) == ("pass", 2, 18)
    ref = reference("fixture_llm_refills")
    assert campaign_hashes(tmp_path / layer) == {k: ref["campaign_sha256"][k] for k in ("a18", "a19")}
    assert parent_hashes(tmp_path / layer) == {k: ref["parent_files"][k] for k in ("a18", "a19")}


@pytest.mark.parametrize("monitor", ({"capture": False}, {"fixture_trace": "hash"}))
def test_the_trace_changes_no_result(tmp_path, monitor):
    """The llm-refill case with no trace, or with the hash fixture trace (the smooth one is the default every
    other run here uses): the reference's results and parent files; the Evaluators keep one trace per
    evaluation, or none; no campaign's line holds one."""
    knobs = {**fixture_knobs("llm refills, one random stream, all live"), "monitor": monitor}
    rec = run("autonmi", tmp_path, *FIXTURE_ARGS, "--populations", "2", "--knobs", knob_file(tmp_path, knobs))
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    assert against_reference(tmp_path / "autonmi", rec, reference("fixture_llm_refills")) == []
    assert_agents_per_campaign(tmp_path / "autonmi", rec)
    assert rec["counts"]["by_role"]["monitor"]["traces"] == (160 if monitor.get("capture", True) else 0)
    assert '"trace"' not in (tmp_path / "autonmi" / "agents.jsonl").read_text()


@pytest.mark.parametrize("slots", (1, 16))
@pytest.mark.parametrize("case", CASES)
def test_the_results_do_not_depend_on_the_monitor_slots(tmp_path, case, slots):
    """campaign.slots 1 (one candidate at a time) and 16 (more slots than the 4 candidates of a generation): the
    reference's results and parent files, with 3 + slots agents per campaign, as at the default 2."""
    knobs = {**fixture_knobs(case), "campaign": {"slots": slots}}
    rec = run("autonmi", tmp_path, *FIXTURE_ARGS, "--populations", "2", "--knobs", knob_file(tmp_path, knobs))
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    assert against_reference(tmp_path / "autonmi", rec, reference(REFERENCE_OF[case])) == []
    assert_agents_per_campaign(tmp_path / "autonmi", rec)
    assert rec["counts"]["hosted_agents"] == 20 * (3 + slots)


@pytest.mark.parametrize("extras", (False, True))
def test_the_monitors_numbers_are_written_only_with_the_record_toggle(tmp_path, extras):
    """record.extras off (the default): the record has the Monitor's counts (one report per inner loop) and no
    per-candidate numbers; on: each campaign's line also has every candidate's numbers (its generation and
    candidate, q, the Monitor's m, left_at, back_at and split_at, and its label: from the trace's end values,
    synthetic, on the fixture), outside the digest, which stays the reference's."""
    knobs = {**fixture_knobs("llm refills, one random stream, all live"), "record": {"extras": extras}}
    rec = run("autonmi", tmp_path, *FIXTURE_ARGS, "--populations", "2", "--knobs", knob_file(tmp_path, knobs))
    assert rec["status"] == "pass" and against_reference(tmp_path / "autonmi", rec, reference("fixture_llm_refills")) == []
    assert rec["counts"]["by_role"]["monitor"]["reports"] == rec["counts"]["sims"] == 160
    rows = lines_of(tmp_path / "autonmi")
    if not extras:
        assert not any("monitor" in r for r in rows)
        return
    for r in rows:
        got = r["monitor"]
        assert [(x["generation"], x["candidate_idx"]) for x in got] == [(g, i) for g in range(2) for i in range(4)]
        assert all(set(x) == {"generation", "candidate_idx", "q", "m", "left_at", "back_at", "split_at", "label",
                              "label_from"} for x in got), got[0]
        assert {x["label_from"] for x in got} == {"trace_end_synthetic"} and {x["label"] for x in got} <= {"good", "bad"}
        assert sorted(x["q"] for x in got if x["generation"] == 0) == [0.0, 0.333333, 0.666667, 1.0]
        assert all(x["m"] == 0.0 for x in got)   # smoke's 4 candidates a generation never reach warm-up (4 a side)


SPLIT_KNOBS = {"outer": {"logs": "per_agent"}, "record": {"extras": True}}   # logs: each candidate's a1_Te


def test_the_monitors_score_after_warm_up_and_find_the_planted_split(tmp_path):
    """The hundredk shape (16 candidates a generation, 2 generations; fixture, smooth synthetic trace) in campaigns
    a0 and a1, with record.extras. Generation 0 is scored before warm-up (m 0, no steps); at its barrier 8 traces
    join each side (labelled from their end values, synthetic), so every candidate of generation 1 is scored:
    split_at at the planted step (rollout_steps + 1 + fixed_point_cap // 2), and m of its branch's sign for every
    candidate more than 0.05 of a1_Te's range from the middle (positive on the low branch, whose wall is cooler
    at the end: labelled good). With 16 slots (the whole generation at once) the results and every Monitor
    number are those of 2 slots: both read one campaign's statistics, which change only at the barrier.
    FIXTURE data. (Campaigns a0 and a1 are not among the few seeds whose sides are apart before the planted
    step: see the package's evidence.)"""
    got = {}
    for slots in (2, 16):
        d = tmp_path / f"slots{slots}"
        d.mkdir()
        knobs = {**SPLIT_KNOBS, "campaign": {"slots": slots}}
        rec = run("autonmi", d, "--preset", "hundredk", "--sim", "fixture", "--agents", "2",
                  "--knobs", knob_file(d, knobs))
        assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
        assert rec["counts"]["by_role"]["monitor"]["reports"] == rec["counts"]["sims"] == 64
        assert_agents_per_campaign(d / "autonmi", rec)
        got[slots] = (rec, lines_of(d / "autonmi"))
    (rec, rows), (rec16, rows16) = got[2], got[16]
    assert rec16["results_sha256"] == rec["results_sha256"]
    # by campaign: agents.jsonl has its lines in the order the campaigns end
    assert {r["agent_id"]: r["monitor"] for r in rows16} == {r["agent_id"]: r["monitor"] for r in rows}
    s = rec["knobs"]["sim"]
    planted, (lo, hi) = s["rollout_steps"] + 1 + s["fixed_point_cap"] // 2, decide.ranges()["a1_Te"]
    clear = 0
    for r in rows:
        with (tmp_path / "slots2" / "autonmi" / "campaigns" / r["agent_id"] / "evolution_log.csv").open() as f:
            u = {(int(x["generation"]), int(x["candidate_idx"])): (float(x["param_a1_Te"]) - lo) / (hi - lo)
                 for x in csv.DictReader(f)}
        g0, g1 = ([x for x in r["monitor"] if x["generation"] == g] for g in (0, 1))
        assert all(x["m"] == 0.0 and x["split_at"] is None for x in g0), r["agent_id"]
        assert [x["label"] for x in g0].count("good") == 8 and {x["label_from"] for x in r["monitor"]} == {
            "trace_end_synthetic"}
        assert all(x["split_at"] == planted for x in g1), (r["agent_id"], [x["split_at"] for x in g1])
        for x in g1:
            side = u[1, x["candidate_idx"]]
            if abs(side - 0.5) > 0.05:
                clear += 1
                assert (x["m"] > 0) == (side < 0.5), (r["agent_id"], side, x)
    assert clear >= 16, clear


def test_the_barrier_labels_a_real_trace_by_its_keep_probability():
    """At a generation's barrier (Script._label) each held trace that is not synthetic (a real inner loop's)
    joins the good side when q, the parent's keep-probability of its result in the generation, is 0.5 or more,
    else the bad side; a synthetic one is labelled from its own end values. With record.extras each candidate's
    q, label and label_from are kept for its campaign's line."""
    import numpy as np

    from fusion_miniapp import monitor
    from fusion_miniapp.layer_autonmi import Script

    sc = Script.__new__(Script)   # only what _label reads
    sc.knobs, sc.extras, sc.stats, sc.numbers = {"outer": {"wall_weight": 0.7}}, True, {}, {}
    stats = sc.stats["a0"] = monitor.Rolling(warmup=1, window=16)
    xs = [np.full((3, 4), float(i)) for i in range(4)]
    for i, x in enumerate(xs):   # candidate 3's trace is synthetic: its label is its end values' (alone: good)
        stats.hold(0, i, x, synthetic=i == 3)
    results = [{"candidate_idx": i, "is_good": True, "te_core_metric": t} for i, t in enumerate((900, 700, 800, 600))]
    sc._label("a0", 0, results, [dict(monitor.NONE)] * 4)
    assert [(x["q"], x["label"], x["label_from"]) for x in sc.numbers["a0"]] == [
        (1.0, "good", "q"), (0.333333, "bad", "q"), (0.666667, "good", "q"), (0.0, "good", "trace_end_synthetic")]
    assert (stats.good.traces, stats.bad.traces) == (3, 1) and stats.held == {}
    assert np.allclose(stats.good.mean, 5 / 3) and np.allclose(stats.bad.mean, 1.0)


@pytest.mark.skipif(bool(_missing("torch")), reason="needs torch")
def test_real_evaluate_through_the_autonmi_layer_on_cpu(tmp_path):
    common = (*REAL_ARGS, "--knobs", knob_file(tmp_path, REAL_KNOBS))
    a = run("autonmi", tmp_path, *common)
    b = run("langgraph", tmp_path, *common)
    assert a["touch_check"] == {"evaluations": 12, "fixture": 0, "pass": 12, "fail": {}}
    assert a["counts"]["by_role"]["monitor"]["traces"] == 12   # the real inner loop's, across processes
    assert a["device_probe"]["torch"]                     # taken in a spawned process, where simulations run
    assert (a["sim_pool"], a["sim_workers_total"]) == ("process", 2)
    assert agree(tmp_path, a, b) == []
    ref = reference("real_cpu")
    for layer, rec in (("autonmi", a), ("langgraph", b)):
        assert against_reference(tmp_path / layer, rec, ref) == [], layer


# the AUTONMI layer's main() with its device probe reporting 3 FAKE GPUs (the probe itself still runs)
FAKE_GPUS_MAIN = """import sys
from fusion_miniapp import layer_autonmi as L
real = L.probe
L.probe = lambda a: {**(real(a) or {}), "gpus": ["FAKE-GPU"] * 3}
sys.exit(L.main(sys.argv[1:]))
"""


@pytest.mark.skipif(bool(_missing("torch")), reason="needs torch")
def test_the_gpu_knob_places_the_real_inner_loops_workers_from_the_device_probe(tmp_path):
    """workers.gpu_per_worker on, --sim real on this machine's CPU (sim.device cpu), with the device probe made to
    report 3 FAKE GPUs: main() hands the probe's GPUs to sim_pool, both inner-loop workers are placed
    (autonmi.inner_loop_gpus), and the results are the real_cpu reference's. No GPU is used: real placement on
    GPUs is unverified until a run on GPU nodes."""
    knobs = {**REAL_KNOBS, "workers": {**REAL_KNOBS["workers"], "gpu_per_worker": True}}
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    proc = subprocess.run([sys.executable, "-c", FAKE_GPUS_MAIN, *REAL_ARGS, "--knobs", knob_file(tmp_path, knobs),
                           "--out", str(tmp_path / "autonmi")], cwd=tmp_path, env=env, check=False, timeout=900,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rec = json.loads((tmp_path / "autonmi" / "run_record.json").read_text())
    assert rec["autonmi"]["inner_loop_gpus"] == {"gpus": ["0", "1", "2"], "workers_placed": 2}
    assert against_reference(tmp_path / "autonmi", rec, reference("real_cpu")) == []


def run_both_on_a_model_server(tmp_path, server: FakeVllm, code: int = 0):
    """Both layers with --llm openai on the server: AUTONMI's from the model job's endpoints file, the
    LangGraph layer's from OPENAI_BASE_URL. Their records and their requests."""
    common = (*SERVER_ARGS, "--knobs", knob_file(tmp_path, SERVER_KNOBS))
    (tmp_path / "endpoints").write_text(f"http://127.0.0.1:{server.port}\n")
    a = run("autonmi", tmp_path, *common, "--populations", "2", "--endpoints-file", str(tmp_path / "endpoints"),
            code=code)
    sent_a = server.take()
    b = run("langgraph", tmp_path, *common, code=code,
            env={**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{server.port}/v1"})
    return a, b, sent_a, server.take()


def test_both_layers_send_the_model_the_same_requests(tmp_path):
    """--llm openai: the same requests (messages, the served model's name, max_tokens = thinking budget +
    reply allowance, vLLM's thinking_token_budget; AUTONMI's transports' own temperature withdrawn, so the
    server's default applies to both), the same replies, the same results; and as recorded."""
    with FakeVllm() as server:
        a, b, sent_a, sent_b = run_both_on_a_model_server(tmp_path, server)
    assert agree(tmp_path, a, b) == []
    assert a["llm_check"]["fail"] == [] and a["llm_check"]["model_share"] == 1.0
    assert all(body.get("temperature") is None for body in sent_a)
    assert len(sent_a) == a["counts"]["llm_calls"] > 12 and request_bodies(sent_a) == request_bodies(sent_b)
    ref = reference("model_server")
    assert request_bodies(sent_a) == [json.dumps(body, sort_keys=True) for body in ref["requests"]]
    for layer, rec in (("autonmi", a), ("langgraph", b)):
        assert against_reference(tmp_path / layer, rec, ref) == [], layer
    assert {body["thinking_token_budget"] for body in sent_a} == {64}
    assert {body["max_tokens"] - 64 for body in sent_a} <= {decide.SUMMARY_TOKENS, *map(decide.reply_tokens, range(1, 5))}
    details = [c for r in lines(tmp_path, "autonmi") for c in r["llm_call_details"]]
    assert len(details) == len(sent_a) and {c["finish_reason"] for c in details} == {"stop"}
    # each call made by its own role's agent: the Generator's refills, the Coordinator's summary
    roles = [c["role"] for c in details]
    assert (roles.count("coordinator"), roles.count("generator")) == (12, len(details) - 12)
    assert_agents_per_campaign(tmp_path / "autonmi", a)


def test_a_failed_summary_falls_back_on_the_autonmi_layer_and_fails_its_campaign_on_langgraph(tmp_path):
    """The server rejects some campaigns' summary calls (HTTP 400, which no layer's transport retries). The
    LangGraph layer (frozen) ends each such campaign at its first failed call: a line with its error and that
    call, and the run fails. The AUTONMI layer tries the call 3 times (TRIES), then leaves the summary out (""),
    counted as the Coordinator's fallback, and the campaign completes: its line has the
    refills, then the 3 failed tries; with the fake server's summary put back it is the reference's campaign, so
    the fallback changed nothing else; every other campaign is the reference's; the run passes."""
    def fail(body):
        text = body["messages"][-1]["content"]
        return text.endswith("found.") and int(hashlib.sha256(text.encode()).hexdigest(), 16) % 3 == 0
    common = (*SERVER_ARGS, "--knobs", knob_file(tmp_path, SERVER_KNOBS))
    with FakeVllm(fail) as server:
        (tmp_path / "endpoints").write_text(server.url + "\n")
        a = run("autonmi", tmp_path, *common, "--populations", "2", "--endpoints-file", str(tmp_path / "endpoints"))
        rejected = [b for b in server.take() if fail(b)]
        b = run("langgraph", tmp_path, *common, code=1, env={**os.environ, "OPENAI_BASE_URL": server.url + "/v1"})
    failed = sorted(r["agent_id"] for r in lines(tmp_path, "langgraph") if "error" in r)
    assert failed and b["status"] == "fail" and b["counts"]["agents_failed"] == len(failed)
    assert (a["status"], a["autonmi"]["ok"], a["counts"]["agents"], a["counts"]["agents_failed"]) == (
        "pass", True, 12, 0)
    ours = {r["agent_id"]: r for r in lines(tmp_path, "autonmi")}
    fell = sorted(c for c, r in ours.items() if r["summary"] == "")
    assert fell == failed and len(rejected) == 3 * len(fell)
    for c in fell:   # the refills, then the summary's 3 failed tries
        calls = ours[c]["llm_call_details"]
        assert [x["role"] for x in calls] == ["generator"] * (len(calls) - 3) + ["coordinator"] * 3, calls
        assert all(x["error"] and x["empty"] is None for x in calls[-3:])
        assert not any(x.get("error") for x in calls[:-3])
    co, ge = a["counts"]["by_role"]["coordinator"], a["counts"]["by_role"]["generator"]
    assert (co["model_asks"], co["model_tries"], co["model_fallbacks"], co["model_calls"]) == (
        12, 12 + 2 * len(fell), len(fell), 12 - len(fell))
    assert ge["model_fallbacks"] == 0 and ge["model_asks"] == ge["model_tries"] == ge["model_calls"] > 0
    assert a["llm_service"]["errors"] == 3 * len(fell) and a["llm_check"]["fail"] == []
    ref = reference("model_server")["campaign_sha256"]
    got = campaign_hashes(tmp_path / "autonmi")
    assert {c: h for c, h in got.items() if c not in fell} == {c: h for c, h in ref.items() if c not in fell}
    summaries = {"Summary: " + x["messages"][-1]["content"].splitlines()[0] for x in rejected}
    for c in fell:   # the fake server's summary put back: the reference's campaign
        put_back = [hashlib.sha256(json.dumps(record.comparable({**ours[c], "summary": s}), sort_keys=True,
                                              default=str).encode()).hexdigest() for s in summaries]
        assert ref[c] in put_back, c


def test_a_failed_refill_keeps_the_parents_mutants_on_the_autonmi_layer(tmp_path):
    """The generations mode, --llm openai: the server rejects every refill call (HTTP 400). Each is tried 3 times
    (TRIES), then falls back, counted as its Generator's: the parent's mutants stay, so every campaign completes
    with the results of the same campaign under outer.generator: mutation (the parent's loop, no refill call;
    the stub), its candidates all the parent's. The model filled none of the refill slots it was asked for, so
    the run fails its model check (llm.min_model_share), while AUTONMI's checks pass."""
    def fail(body):
        return "reply with a JSON list of" in body["messages"][-1]["content"].splitlines()[-1]
    with FakeVllm(fail) as server:
        (tmp_path / "endpoints").write_text(server.url + "\n")
        a = run("autonmi", tmp_path, *SERVER_ARGS, "--knobs", knob_file(tmp_path, SERVER_KNOBS), "--populations", "2",
                "--endpoints-file", str(tmp_path / "endpoints"), code=1)
        sent = server.take()
    mutation = tmp_path / "mutation"
    mutation.mkdir()
    knobs = {"outer": {**SERVER_KNOBS["outer"], "generator": "mutation"}}
    m = run("autonmi", mutation, *SERVER_ARGS, "--knobs", knob_file(mutation, knobs), "--populations", "2")
    refills = 12 * (SERVER_KNOBS["outer"]["generations"] - 1)
    assert len([b for b in sent if fail(b)]) == 3 * refills and len(sent) == 3 * refills + 12   # and 12 summaries
    ge = a["counts"]["by_role"]["generator"]
    assert (ge["model_asks"], ge["model_tries"], ge["model_fallbacks"], ge["model_calls"]) == (
        refills, 3 * refills, refills, 0)
    assert a["counts"]["by_role"]["coordinator"]["model_fallbacks"] == 0
    assert a["counts"]["refill_slots"]["filled_by_model"] == 0 < a["counts"]["refill_slots"]["asked"]
    assert a["counts"]["candidates_by_source"]["model"] == 0
    assert (a["status"], a["autonmi"]["ok"], a["counts"]["agents"], a["counts"]["agents_failed"]) == (
        "fail", True, 12, 0)
    assert len(a["llm_check"]["fail"]) == 1 and "refill slots" in a["llm_check"]["fail"][0]
    assert m["status"] == "pass"
    differ = ("refill", "n_llm_calls", "summary")   # the calls asked, and the summary's text (server, stub)
    ours, theirs = ({r["agent_id"]: {k: v for k, v in record.comparable(r).items() if k not in differ}
                     for r in lines_of(d)} for d in (tmp_path / "autonmi", mutation / "autonmi"))
    assert ours == theirs


LIVE_LINE = re.compile(r"^agents live=(\S+) model_requests_open=(\S+) queued=(\S+) decoding_at_servers=(\S+) "
                       r"done=(\S+)$")


def run_logged(tmp_path, *args, code=0) -> tuple[dict, str]:
    """One AUTONMI run in tmp_path/autonmi: its run record and its stderr."""
    proc = subprocess.run([sys.executable, "-m", ENTRY["autonmi"], *args, "--out", str(tmp_path / "autonmi")],
                          cwd=tmp_path, check=False, timeout=900, capture_output=True, text=True)
    assert proc.returncode == code, proc.stderr[-3000:]
    return json.loads((tmp_path / "autonmi" / "run_record.json").read_text()), proc.stderr


def live_lines(stderr: str) -> list[list[str]]:
    return [list(m.groups()) for m in map(LIVE_LINE.match, stderr.splitlines()) if m]


def as_printed(series: list[list]) -> list[list[str]]:
    return [["-" if v is None else str(v) for v in row[1:]] for row in series]


def test_the_live_line_and_the_client_counts_on_the_stub(tmp_path):
    """20 fixture campaigns in one population, each model call 1 s at the stub (emulated) and 4 admitted at once
    (llm.max_in_flight): the client's in-flight maximum is that cap, with calls queued behind it, in the record
    and in the live line; the live line comes about once per interval (record.ticker sleeps an interval after each
    line, so an event-loop stall stretches a gap: up to about 2.2 s at 2,000 campaigns, measured), as the record's
    series, and ends with every hosted agent done (five per campaign); Little's law holds three ways within 10%
    (AUTONMI's agent-seconds in model requests per second,
    fusion's calls per second x their mean time in flight, calls per second x 1 s); and the results are the
    20-campaign reference's (neither knob changes them)."""
    knobs = fixture_knobs("llm refills, one random stream, all live")
    knobs["llm"] = {"stub_latency_s": 1.0, "max_in_flight": 4}
    rec, err = run_logged(tmp_path, *FIXTURE_ARGS, "--knobs", knob_file(tmp_path, knobs), "--populations", "1")
    ref = json.loads((REFERENCES_DIR / "fixture_llm_refills.json").read_text())   # the case's, with two knobs more
    assert rec["results_sha256"] == ref["results_sha256"]
    s = rec["speed"]
    opened, queued = s["model_requests_in_flight"], s["model_requests_queued"]
    assert opened["side"] == "client" and opened["emulated"]
    assert opened["calls_timed"] == rec["counts"]["llm_calls"] == 40
    assert opened["max"] == 4 and opened["time_ge_pct_s"]["hosted_agents"]["of"] == 100
    assert set(opened["time_ge_pct_s"]) == {"hosted_agents"}   # no comparison against the client's caps
    assert queued["max"] > 0 and queued["time_s"]["max"] > 0 and queued["autonmi_built_in"]["agent_seconds"] > 0
    little = opened["littles_law"]
    figures = [little[k] for k in ("autonmi_agent_seconds_per_s", "fusion_calls_per_s_x_mean_time_in_flight",
                                   "calls_per_s_x_stub_latency_s")]
    assert min(figures) > 0 and max(figures) <= 1.1 * min(figures), little
    assert opened["loop_lag"] is not None and isinstance(opened["loop_lag"]["high"], bool)
    # the outer loop's calls are every agent's calls until the new roles make their own (decision 14)
    assert s["rates_per_s"]["model_calls_all_agents"] == s["rates_per_s"]["model_calls"]
    assert s["agents_live"]["max"] == 100 and s["agents_live"]["hosted_agents"] == 100   # five per campaign
    assert {r: x["max"] for r, x in s["agents_live"]["by_role"].items()} == {**dict.fromkeys(ROLES, 20), "monitor": 40}
    assert s["agents_live"]["by_role"]["monitor"]["time_ge_pct_s"]["hosted_agents"]["of"] == 40   # 2 per campaign
    assert s["decoding_at_servers"] is None and s["decoding_at_servers_why_not"].startswith("the stub")
    assert s["headline"]["figure"] == "model_requests_in_flight" and s["headline"]["emulated"]
    assert "emulated" in s["headline"]["label"] and "decoding" not in s["headline"]["label"]
    series = s["live"]["series"]
    assert live_lines(err) == as_printed(series) and len(series) >= 5
    assert series[-1][1:] == [0, 0, 0, None, 100]                   # at the end: none live, every agent done
    gaps = [b[0] - a[0] for a, b in zip(series[:-2], series[1:-1])]  # every interval (the last line: at the end)
    assert all(0.95 <= g <= 2.0 for g in gaps), gaps
    assert max(row[2] for row in series) <= 4 and max(row[3] for row in series) > 0   # never past the cap; queued
    assert all(row[4] is None for row in series)                     # no model servers: "-"
    speed = dict(kv.split("=", 1) for kv in record.speed_line(rec)[len("speed: "):].split())
    assert (speed["requests_open_max"], speed["decoding_max"], speed["emulated"]) == ("4", "None", "True")


def test_the_token_cap_reaches_the_fleet(tmp_path):
    """llm.admission_tokens: 1 admits one model request at a time (each is larger than the cap, which Flowgentic
    admits alone), so the client's in-flight maximum is 1, with the rest queued; the results are unchanged."""
    knobs = fixture_knobs("llm refills, one random stream, all live")
    knobs["llm"] = {"stub_latency_s": 0.2, "admission_tokens": 1}
    rec, _ = run_logged(tmp_path, *FIXTURE_ARGS, "--knobs", knob_file(tmp_path, knobs), "--populations", "1")
    ref = json.loads((REFERENCES_DIR / "fixture_llm_refills.json").read_text())
    assert rec["results_sha256"] == ref["results_sha256"]
    opened, queued = rec["speed"]["model_requests_in_flight"], rec["speed"]["model_requests_queued"]
    assert opened["max"] == 1 and opened["calls_timed"] == 40 and queued["max"] > 0


def test_an_openai_run_over_two_model_servers_records_their_decoding_counts(tmp_path):
    """--llm openai over two fake vLLM servers (FIXTURES, not models; each call 2.2 s, 2 at once per server, so the
    sampler's 2 s readings see every call, and requests wait at the servers), with a key as AUTONMI's jobs/model.sh
    sets one: both servers are sampled into vllm/vllm.json and the record's decoding_at_servers, the headline,
    which names who the servers say serve the model (here the fixture); AUTONMI's check that the fleets' tokens and
    requests equal the servers' passes; the key is in no record; the results and requests are the model-server
    reference's."""
    key = "the-servers-key-7f3a"
    (tmp_path / "api-key").write_text(key + "\n")
    with FakeVllm(latency_s=2.2, batch=2, api_key=key) as one, FakeVllm(latency_s=2.2, batch=2, api_key=key) as two:
        (tmp_path / "endpoints").write_text(f"{one.url}\n{two.url}\n")
        rec, err = run_logged(tmp_path, *SERVER_ARGS, "--knobs", knob_file(tmp_path, SERVER_KNOBS),
                              "--populations", "2", "--endpoints-file", str(tmp_path / "endpoints"),
                              "--api-key-file", str(tmp_path / "api-key"))
        sent = (one.take(), two.take())
        peaks = [(x.running_max, x.waiting_max) for x in (one, two)]
    assert all(sent) and all(r == 2 and w > 0 for r, w in peaks), peaks   # both served, each its batch, with waits
    ref = reference("model_server")
    assert against_reference(tmp_path / "autonmi", rec, ref) == []
    assert request_bodies(sent[0] + sent[1]) == [json.dumps(body, sort_keys=True) for body in ref["requests"]]
    summary = json.loads((tmp_path / "autonmi" / "summary.json").read_text())
    tokens = next(c for c in summary["checks"] if c["name"] == "tokens_match_the_servers")
    assert tokens["ok"] and summary["model_servers"]["servers"] == 2 and rec["autonmi"]["ok"], tokens
    d = rec["speed"]["decoding_at_servers"]
    assert (d["servers"], d["sample_errors"], d["interval_s"]) == (2, 0, 2.0) and d["running"]["samples"] >= 4
    assert 1 <= d["running"]["max"] <= 4 and d["waiting"]["max"] >= 1 and 0 < d["kv_cache_usage"]["max"] < 1
    ge = d["running"]["time_ge_pct_s"]   # against the hosted agents (12 x 5) alone: no client cap
    assert list(ge) == ["hosted_agents"] and ge["hosted_agents"]["of"] == 60
    assert d["stale_s"] == [0.0, 0.0] and d["gauges_missing"] is None   # both answered every reading
    headline = rec["speed"]["headline"]
    assert headline["figure"] == "decoding_at_servers" and not headline["emulated"]
    assert headline["served_by"] == ["fusion-test-fixture"]
    assert rec["llm_service"]["served_models"] == [{"id": "stand-in", "owned_by": "fusion-test-fixture"}]
    assert not [f for f in (tmp_path / "autonmi").rglob("*") if f.is_file() and key.encode() in f.read_bytes()]
    assert not rec["speed"]["model_requests_in_flight"]["emulated"]
    series = rec["speed"]["live"]["series"]
    assert live_lines(err) == as_printed(series) and max(row[4] for row in series) >= 1


@pytest.mark.parametrize("why", ("another model", "a wrong key"))
def test_a_server_that_does_not_serve_llm_model_is_refused_before_any_campaign(tmp_path, why):
    """The start-up check: a server whose /v1/models does not list llm.model, or does not answer it (a key the
    server refuses: 401), refuses the run, with a clear message, before any model request; the record fails with
    no campaign."""
    server, said = ((FakeVllm(model="another-model"), "['another-model'], not llm.model 'stand-in'")
                    if why == "another model" else
                    (FakeVllm(api_key="the-servers-key"), "/v1/models did not answer: HTTPError: HTTP Error 401"))
    (tmp_path / "api-key").write_text("a-wrong-key\n")
    with server:
        (tmp_path / "endpoints").write_text(server.url + "\n")
        rec, err = run_logged(tmp_path, *SERVER_ARGS, "--knobs", knob_file(tmp_path, SERVER_KNOBS),
                              "--endpoints-file", str(tmp_path / "endpoints"), "--api-key-file",
                              str(tmp_path / "api-key"), code=1)
        sent = server.take()
    assert sent == [] and rec["status"] == "fail" and rec["counts"]["agents"] == 0
    refused = [x for x in err.splitlines() if "run refused" in x]
    assert len(refused) == 1 and said in refused[0], err[-2000:]


@pytest.mark.parametrize("how", ("a server without /metrics", "--no-vllm-metrics"))
def test_a_server_without_vllms_metrics_runs_without_decoding_counts(tmp_path, how):
    """An OpenAI-compatible server without vLLM's /metrics (a FIXTURE), or a vLLM one with --no-vllm-metrics: the
    run passes, without the servers' counts (and so without AUTONMI's checks against them), warns once when the
    servers have no /metrics, and its headline is the client-side count (not emulated)."""
    flag = ("--no-vllm-metrics",) if how == "--no-vllm-metrics" else ()
    with FakeVllm(metrics=bool(flag)) as server:   # with the flag, a server that has /metrics
        (tmp_path / "endpoints").write_text(server.url + "\n")
        rec, err = run_logged(tmp_path, *SERVER_ARGS, "--knobs", knob_file(tmp_path, SERVER_KNOBS),
                              "--endpoints-file", str(tmp_path / "endpoints"), *flag)
    assert rec["status"] == "pass" and rec["speed"]["decoding_at_servers"] is None
    why = rec["speed"]["decoding_at_servers_why_not"]
    assert ("--no-vllm-metrics" if flag else "/metrics") in why and not (tmp_path / "autonmi" / "vllm").exists()
    assert len([x for x in err.splitlines() if "decoding counts are not recorded" in x]) == (0 if flag else 1)
    assert rec["speed"]["headline"] == {**rec["speed"]["headline"], "figure": "model_requests_in_flight",
                                        "label": "client side", "emulated": False}


@pytest.mark.parametrize("mode", ("generations", "async"))
def test_sigterm_writes_an_interrupted_record_with_the_campaigns_that_finished(tmp_path, mode):
    """SIGTERM (scancel, a time limit) while campaigns run, as test_lifecycle.py sends it to run.py: the run
    is cancelled through its event loop, its record says interrupted and keeps the campaigns that had
    finished, and it exits 143. 50 live campaigns per population of 16 candidates keep hundreds of
    evaluations queued when the first campaign finishes: the signal cuts them off, and the record gives
    AUTONMI's telemetry verdict (complete: the populations are waited for) with every population's task
    counts, those abandoned included, and AUTONMI's check every_population_ran failed. In both modes (in the
    asynchronous one a campaign's line has no generations, and each evaluation's submission time)."""
    knobs = {"tools": {"latency_ms": 500}, "agents": {"max_live": 50}, "campaign": {"mode": mode},
             "outer": {"population": 16, "generations": 2, "parent_stdout": "drop", "logs": "none"}}
    out = tmp_path / "autonmi"
    with open(tmp_path / "stderr.txt", "w") as err:
        p = subprocess.Popen([sys.executable, "-m", ENTRY["autonmi"], "--preset", "smoke", "--sim", "fixture",
                              "--agents", "400", "--populations", "2", "--knobs", knob_file(tmp_path, knobs),
                              "--out", str(out)], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=err)
        deadline = time.time() + 120
        while not any(f.stat().st_size for f in out.glob("agents-p*.jsonl")):   # a campaign has finished
            assert p.poll() is None and time.time() < deadline, (tmp_path / "stderr.txt").read_text()[-3000:]
            time.sleep(0.2)
        before = {json.loads(x)["agent_id"] for f in out.glob("agents-p*.jsonl")   # finished before SIGTERM
                  for x in f.read_text().splitlines() if x.endswith("}")}
        t = time.time()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=90) == 143, (tmp_path / "stderr.txt").read_text()[-3000:]
    assert time.time() - t < 60
    rec = json.loads((out / "run_record.json").read_text())
    rows = lines(tmp_path, "autonmi")
    assert rec["status"] == "interrupted" and 0 < rec["counts"]["agents"] == len(rows) < 400
    assert rec["counts"]["agents_failed"] == 0   # a cancelled campaign is not a failed one
    assert before <= {r["agent_id"] for r in rows}
    exp = decide.expected(rec["knobs"])
    per = 2 if mode == "generations" else 0
    assert all(r["n_sims"] == exp["sims"] == len(r["sim_s"]) and len(r["generations"]) == per for r in rows)   # whole
    if mode == "async":
        assert all(len(r["t_submit"]) == exp["sims"] for r in rows)
    if mode == "async":   # the finished campaigns' results scored
        assert rec["counts"]["asynchronous"]["results_scored"] == rec["counts"]["sims"]
    # each population reported into the run before it was verified, its telemetry written whole
    tel = rec["autonmi"]["telemetry"]
    assert (tel["complete"], tel["reasons"]) == (True, []), tel["reasons"]
    assert {p: v["complete"] for p, v in tel["populations"].items()} == {"p000": True, "p001": True}
    tasks = tel["tasks"]   # each population's counts add up by construction; this checks the record's sums
    assert tasks["created"] == sum(tasks[k] for k in ("completed", "failed", "canceled", "abandoned", "not_submitted"))
    assert tasks["abandoned"] > 0 and tasks["completed"] >= rec["counts"]["sims"]
    # and AUTONMI's verdict says the run did not end: the check it records for populations it cancels
    summary = json.loads((out / "summary.json").read_text())
    assert (summary["ok"], summary["exit_hint"]) == (False, 4)   # 4: telemetry complete, a check failed
    assert rec["autonmi"]["failed_checks"] == ["every_population_ran"] and not rec["autonmi"]["ok"]
    ran = next(c["detail"] for c in summary["checks"] if c["name"] == "every_population_ran")
    assert (ran["cancelled"], ran["still_running"], ran["interrupted"]) == (["p000", "p001"], [], True)


@pytest.mark.parametrize("mode, llm", (("async", "stub"), ("async", "fake vLLM"), ("generations", "fake vLLM")))
def test_sigterm_keeps_the_model_figures_of_the_calls_made_before_it(tmp_path, mode, llm):
    """SIGTERM in the middle of a run with model calls (on StellarAI, Slurm's signal before the time limit): the
    interrupted record keeps the figures of every call made up to the signal, the unfinished campaigns' included,
    and says so (speed.coverage.model_calls): llm_service, the client-side counts and tokens per second, and each
    role's calls answered, asked, tried and fallen back (Script.cut_short), of which the campaign lines count only
    the finished campaigns' share. Against the fake vLLM server (a FIXTURE) every reply is held 20 s from a moment
    before the signal, so none comes after it, and the record's calls answered, and their tokens, are exactly those
    the server answered. No call sums file (calls-p*.npz) is left behind."""
    knobs = {"tools": {"latency_ms": 500}, "agents": {"max_live": 50}, "campaign": {"mode": mode},
             "outer": {"population": 16, "generations": 2, "parent_stdout": "drop", "logs": "none"}}
    server = None if llm == "stub" else FakeVllm()
    if server is not None:
        knobs["llm"] = {**SERVER_KNOBS["llm"], **({"thinking": "off"} if mode == "async" else {})}
    out = tmp_path / "autonmi"
    with contextlib.ExitStack() as stack, open(tmp_path / "stderr.txt", "w") as err:
        extra = []
        if server is not None:
            stack.enter_context(server)
            (tmp_path / "endpoints").write_text(server.url + "\n")
            extra = ["--endpoints-file", str(tmp_path / "endpoints")]
        p = subprocess.Popen([sys.executable, "-m", ENTRY["autonmi"], "--preset", "smoke", "--sim", "fixture",
                              "--agents", "400", "--populations", "2", "--knobs", knob_file(tmp_path, knobs), *extra,
                              "--out", str(out)], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=err)
        deadline = time.time() + 120
        while not any(f.stat().st_size for f in out.glob("agents-p*.jsonl")):   # a campaign has finished
            assert p.poll() is None and time.time() < deadline, (tmp_path / "stderr.txt").read_text()[-3000:]
            time.sleep(0.2)
        if server is not None:   # each reply from now on held 20 s; once none has come for 3 s, the run has the rest
            server.latency_s = 20.0
            answered, quiet = server.successes, time.time()
            while time.time() - quiet < 3:
                time.sleep(0.1)
                if server.successes != answered:
                    answered, quiet = server.successes, time.time()
        t = time.time()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=90) == 143, (tmp_path / "stderr.txt").read_text()[-3000:]
        assert time.time() - t < 60
        received = len(server.bodies) if server is not None else None
    rec = json.loads((out / "run_record.json").read_text())
    rows = lines(tmp_path, "autonmi")
    s, c, by = rec["llm_service"], rec["counts"], rec["counts"]["by_role"]
    assert rec["status"] == "interrupted" and 0 < c["agents"] == len(rows) < 400
    assert rec["speed"]["coverage"]["model_calls"].startswith("every call up to the signal, the unfinished")
    assert not list(out.glob("calls-p*.npz"))
    replies = s["calls"] - s["errors"]
    assert s["errors"] == 0 and replies == sum(by[role]["model_calls"] for role in ROLES) > 0
    for role in ROLES:   # none failed or fell back; a call in flight at the signal was asked and tried, not answered
        b = by[role]
        assert b["model_fallbacks"] == 0 and b["model_asks"] == b["model_tries"] >= b["model_calls"], role
    callers = ROLES if mode == "async" else ("coordinator", "generator")   # (generations: the refills and summaries)
    assert all(by[role]["model_calls"] > 0 for role in callers)
    assert not any(by[role]["model_asks"] for role in ROLES if role not in callers)
    # the campaign lines count the finished campaigns' calls; the unfinished ones had made calls too
    assert c["llm_calls"] < replies
    if mode == "async":   # each evaluation's note was asked
        assert by["monitor"]["model_asks"] >= c["sims"] > 0
    else:   # the finished campaigns' refills and summaries, in their lines
        assert c["llm_calls"] == len([d for r in rows for d in r["llm_call_details"]])
    flight, tokens = rec["speed"]["model_requests_in_flight"], rec["speed"]["model_tokens_per_s"]
    assert (flight["calls_timed"], flight["calls_untimed"]) == (replies, 0) and flight["max"] > 0
    assert tokens["calls"] > 0 and tokens["prompt_per_s"] > 0 and tokens["emulated"] == (server is None)
    assert rec["speed"]["model_latency_s"]["n"] == s["calls"] == rec["llm_check"]["replies"]["calls"]
    if server is not None:   # exactly the replies it sent before the signal, at the fixture's 100 + 50 tokens each
        assert replies == answered
        assert (s["tokens"]["prompt"], s["tokens"]["completion"]) == (100 * replies, 50 * replies)
        assert answered <= received <= sum(by[role]["model_tries"] for role in ROLES)   # a try may wait for admission
        assert tokens["generated_per_s"] * tokens["window_s"] == pytest.approx(50 * tokens["calls"], rel=1e-3)


def test_the_out_folder_holds_one_run(tmp_path):
    out = tmp_path / "autonmi"
    out.mkdir()
    (out / "run_record.json").write_text("{}")
    proc = subprocess.run([sys.executable, "-m", ENTRY["autonmi"], "--sim", "fixture", "--agents", "2", "--out",
                           str(out)], cwd=tmp_path, check=False, timeout=300, capture_output=True, text=True)
    assert proc.returncode != 0 and "already holds a run" in proc.stderr


def test_populations_and_workers_follow_run_py():
    """By default, a population per agent process of run.py, and its share of the simulation workers."""
    from fusion_miniapp.layer_autonmi import layout
    hundredk = knob_files.load(preset="hundredk")
    assert layout(hundredk, "fixture") == (10, 1)                  # 100,000 campaigns, 10,000 per process
    assert layout(hundredk, "fixture", populations=4) == (4, 2)
    assert layout(hundredk, "real", populations=4, workers=3) == (4, 3)
    with pytest.raises(knob_files.KnobError, match="sim_workers"):
        layout(hundredk, "real", populations=20)                   # fewer workers than populations


def test_populations_divide_the_campaigns():
    """A campaign's agents stay in one population: AUTONMI cuts the agent list into equal contiguous blocks,
    so the populations must divide the campaigns. By default the smallest count that does from run.py's agent
    processes up to twice that; otherwise refused, naming nearby campaign counts that work; a --populations that
    does not divide is refused."""
    from fusion_miniapp.layer_autonmi import layout
    k = knob_files.load(preset="hundredk")
    k["agents"]["count"] = 25_000                                   # 3 agent processes by default: 4 divides
    assert layout(k, "fixture") == (4, 2)
    k["agents"]["count"] = 10_007                                   # a prime: 2 to 4 divide none of it
    with pytest.raises(knob_files.KnobError, match="run 10006 or 10008 campaigns"):
        layout(k, "fixture")
    k["agents"]["count"] = 20
    with pytest.raises(knob_files.KnobError, match="--populations 3 does not divide the 20 campaigns"):
        layout(k, "fixture", populations=3)
    assert layout(k, "fixture", populations=5) == (5, 2) and layout(k, "fixture", populations=50) == (20, 1)
    assert layout(k, "fixture", populations=0) == layout(k, "fixture") and layout(k, "fixture")[0] == 1   # 0: the default


def test_a_cap_below_one_campaign_and_starting_together_with_a_cap_are_refused(tmp_path):
    """agents.max_live counts campaigns, five agents each for AUTONMI's cap: a cap below one campaign is refused
    (the knob's own check, and the layer's), and so is agents.start_together with any cap (the agents past it
    would start only as others end, and every agent waits for all to start), before anything starts."""
    from fusion_miniapp.layer_autonmi import agents_live
    k = knob_files.base()
    assert agents_live(k) is None and agents_live({**k, "agents": {**k["agents"], "max_live": 7}}) == 35
    assert agents_live({**k, "agents": {**k["agents"], "max_live": 7}, "campaign": {**k["campaign"], "slots": 16}}) == 133
    with pytest.raises(knob_files.KnobError, match="must hold one campaign"):
        agents_live({**k, "agents": {**k["agents"], "max_live": 0}})
    for over, said in (({"max_live": 0}, "agents.max_live must be in [1"),
                       ({"max_live": 7, "start_together": True}, "agents.start_together needs agents.max_live: null")):
        out = tmp_path / f"out-{len(said)}"
        proc = subprocess.run([sys.executable, "-m", ENTRY["autonmi"], *FIXTURE_ARGS, "--knobs",
                               knob_file(tmp_path, {"agents": over}), "--out", str(out)], cwd=tmp_path,
                              capture_output=True, text=True, timeout=300, check=False)
        assert proc.returncode != 0 and said in proc.stderr and not out.exists(), proc.stderr[-2000:]


SLOW_START = ("--stagger-batch", "1", "--stagger-interval", "0.2")   # one campaign (5 agents) per batch, 0.2 s apart
LAYOUTS = {   # each gives the 20-campaign reference's results; the cap of one campaign leaves one partly started
    "1 population": ("--populations", "1"), "5 populations": ("--populations", "5"),
    "a cap of 1 campaign": ("--populations", "1", "--knobs", {"agents": {"max_live": 1}}),
    "a cap of 7 campaigns, 2 populations": ("--populations", "2", "--knobs", {"agents": {"max_live": 7}}),
    "1 worker": ("--populations", "1", "--workers", "1"), "4 workers, 2 populations": ("--populations", "2",
                                                                                         "--workers", "4"),
    "starting together, a slow start, 2 populations": ("--populations", "2", *SLOW_START,
                                                       "--knobs", {"agents": {"start_together": True}}),
    "not starting together, a slow start, 2 populations": ("--populations", "2", *SLOW_START,
                                                           "--knobs", {"agents": {"start_together": False}}),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_campaigns_results_do_not_depend_on_their_layout(tmp_path, layout):
    """The 20-campaign llm-refill case in 1, 2 or 5 populations, with a cap of 1 or 7 campaigns or none, with 1 or 4
    simulation workers, and starting together or not: the same results and parent files as the reference, each
    campaign five agents that all end, every message received once.

    With a slow start (one campaign per start-up batch, 0.2 s apart, so about 1.8 s until a population's last agent
    starts). Starting together, every Coordinator begins its campaign after the last agent of its population
    started, so all 100 agents are live at once, and a campaign's five agents start in one batch (the stagger's
    batch is five agents per campaign; without the knob a Coordinator's first step, before its peers' tasks run,
    can stretch that). Without the knob the first campaigns begin, and end, before the last agents start."""
    knobs = next((x for x in LAYOUTS[layout] if isinstance(x, dict)), {})
    args = [knob_file(tmp_path, {**fixture_knobs("llm refills, one random stream, all live"), **x})
            if isinstance(x, dict) else x for x in LAYOUTS[layout]]
    if "--knobs" not in args:
        args += ["--knobs", knob_file(tmp_path, fixture_knobs("llm refills, one random stream, all live"))]
    rec = run("autonmi", tmp_path, *FIXTURE_ARGS, *args)
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    assert against_reference(tmp_path / "autonmi", rec, reference("fixture_llm_refills")) == []
    assert_agents_per_campaign(tmp_path / "autonmi", rec)
    live = rec["speed"]["agents_live"]
    if "--workers" in args:
        assert rec["autonmi"]["workers_per_population"] == int(args[args.index("--workers") + 1])
    if "max_live" in knobs.get("agents", {}):
        cap = 5 * knobs["agents"]["max_live"] * int(args[1])   # per population
        assert live["max"] <= cap and live["all_live_s"] == 0.0, live
    if "--stagger-batch" in args:
        assert live["ramp_up_s"] >= 1.5, live     # one campaign per batch: 9 intervals per population
        together = rec["speed"]["start_together"]
        assert together["on"] is knobs["agents"]["start_together"]
        if together["on"]:
            assert together["begin_margin_s"] >= 0 and together["coordinator_wait_s"]["n"] == 20, together
            rows = app_rows(tmp_path / "autonmi")
            for c in {r["campaign"] for r in rows}:   # a batch is a campaign's five agents, not one
                t = [r["t_start"] for r in rows if r["campaign"] == c]
                assert max(t) - min(t) < 0.1, (c, t)
            assert live["max"] == 100 and live["all_live_s"] > 0, live
        else:
            assert together["begin_margin_s"] < -1 and together["coordinator_wait_s"]["n"] == 0, together
            assert live["max"] < 100 and live["all_live_s"] == 0.0, live


@pytest.mark.parametrize("role", ("generator", "evaluator", "monitor"))
def test_an_agent_that_fails_ends_its_campaign_alone(tmp_path, role):
    """A Generator that fails at its second population, after its refill call, an Evaluator that fails before
    its second results, or a Monitor, the campaign's only slot, that fails before its report of generation 1's
    first candidate, with its other candidates still on the queue (the test hook FUSION_AUTONMI_FAIL; a Monitor
    is named by its call id): its campaign is one error line, written once, with the failing role and every
    role's model calls (by then the Generator's refill, whichever role failed); the others end with an error
    instead of waiting, and the 19 other campaigns end as in the reference. The run ends on its own (no hang)
    and fails, as a run with a failed campaign does."""
    slots, agent, at = (1, "monitor0", "g1c0") if role == "monitor" else (2, role, "1")
    env = {**os.environ, "FUSION_AUTONMI_FAIL": f"a3/{agent}:{at}"}
    knobs = {**fixture_knobs("llm refills, one random stream, all live"), "campaign": {"slots": slots}}
    rec = run("autonmi", tmp_path, *FIXTURE_ARGS, "--populations", "2", "--knobs", knob_file(tmp_path, knobs),
              env=env, code=1)
    out, ref = tmp_path / "autonmi", reference("fixture_llm_refills")
    errors = [r for r in lines_of(out) if "error" in r]
    assert [(r["agent_id"], r["role"]) for r in errors] == [("a3", role)]
    assert "InjectedFailure" in errors[0]["error"] and f"a3/{agent}" in errors[0]["error"]
    assert [c["role"] for c in errors[0]["llm_call_details"]] == ["generator"]   # the refill; no summary yet
    assert (rec["status"], rec["counts"]["agents"], rec["counts"]["agents_failed"]) == ("fail", 20, 1)
    assert campaign_hashes(out) == {**{k: v for k, v in ref["campaign_sha256"].items() if k != "a3"},
                                    "a3": campaign_hashes(out)["a3"]}
    rows = {r["agent_id"]: r for r in app_rows(out)}
    assert [rows[f"a3/{x}"]["status"] for x in names(slots)] == ["failed"] * (3 + slots)
    assert all("CampaignFailed" in rows[f"a3/{x}"]["error"] for x in names(slots) if x != agent)
    assert all(r["finished"] for a, r in rows.items() if not a.startswith("a3/")) and len(rows) == 20 * (3 + slots)
    assert not any(r["messages_left"] for r in rows.values())
    a3 = [rows[f"a3/{x}"] for x in names(slots)]   # each message a3 sent was received or dropped (the queue's too)
    assert sum(r["messages_sent"] for r in a3) == sum(r["messages_received"] + r["messages_dropped"] for r in a3)
    if role == "monitor":   # generation 0's 4 and g1c0 ran; generation 1's 3 others, never taken, are its drops
        assert (rows["a3/monitor0"]["simulations"], rows["a3/monitor0"]["messages_dropped"]) == (5, 3)
    assert rec["autonmi"]["failed_checks"] == ["every_campaign_reports", "every_agent_finishes"]


def test_the_mailbox_delivers_each_message_once():
    """The Mailbox's contract: a receiver that asks first is woken by the send; a second send of a key not yet
    received raises; a send after its receipt stays in the box, reported as left (rows' messages_left, which
    every_message_received fails on); fail(a1) fails a1's waits and sends, now and later, counts a1's messages
    not yet received as dropped, and leaves a10's alone; all_started is set by the last agent's start. A send
    that has woken a waiting receive before its campaign fails is that receive's: received once, not also
    dropped."""
    from fusion_miniapp.layer_autonmi import CampaignFailed, Mailbox

    async def check():
        box = Mailbox(agents=3)
        waiting = asyncio.ensure_future(box.receive("a1/generator", "make", 0))
        await asyncio.sleep(0)
        box.send("a1/coordinator", "generator", "make", 0, {"x": 1})
        assert await waiting == {"x": 1}
        box.send("a1/coordinator", "generator", "make", 1, {"x": 2})
        with pytest.raises(RuntimeError, match="a1/generator/make/1 sent twice"):
            box.send("a1/coordinator", "generator", "make", 1, {"x": 3})
        assert await box.receive("a1/generator", "make", 1) == {"x": 2}
        box.send("a1/coordinator", "generator", "make", 1, {"x": 4})   # after its receipt: left in the box
        assert box.left() == {"a1/generator": 1}
        box.send("a10/coordinator", "evaluator", "evaluate", 0, [1])
        waiting = asyncio.ensure_future(box.receive("a1/evaluator", "evaluate", 0))
        await asyncio.sleep(0)
        box.fail("a1", "a1/generator failed")
        with pytest.raises(CampaignFailed):
            await waiting
        with pytest.raises(CampaignFailed):
            box.send("a1/coordinator", "evaluator", "evaluate", 1, [])
        with pytest.raises(CampaignFailed):
            await box.receive("a1/coordinator", "results", 0)
        assert box.dropped == {"a1/generator": 1} and box.left() == {"a10/evaluator": 1}
        assert await box.receive("a10/evaluator", "evaluate", 0) == [1] and box.left() == {}
        assert box.sent == {"a1/coordinator": 3, "a10/coordinator": 1}
        assert box.received == {"a1/generator": 2, "a10/evaluator": 1}
        for _ in range(3):
            assert not box.all_started.is_set()
            box.start()
        assert box.all_started.is_set()
        waiting = asyncio.ensure_future(box.receive("a2/generator", "make", 0))
        await asyncio.sleep(0)
        box.send("a2/coordinator", "generator", "make", 0, {"x": 5})   # wakes it
        box.fail("a2", "a2/evaluator failed")                          # before it resumes: received, not dropped
        assert await waiting == {"x": 5}
        assert (box.sent["a2/coordinator"], box.received["a2/generator"], box.dropped["a2/generator"]) == (1, 1, 0)
        with pytest.raises(CampaignFailed):
            await box.receive("a2/generator", "make", 1)
    asyncio.run(check())


def test_the_candidate_queue_gives_each_item_to_the_next_taker():
    """The queue beside the exact keys: a taker that asks first is woken by the put; items wait in order for
    takers; each item goes to one taker, whichever asks; a taker cancelled while it waits is passed over; the queue
    is kept only while it holds items or takers; fail(a1) fails a1's takers, now and later, and counts its items not
    taken as dropped under the queue's key, as left() counts them before."""
    from fusion_miniapp.layer_autonmi import CampaignFailed, Mailbox

    async def check():
        box = Mailbox(agents=5)
        first = asyncio.ensure_future(box.take("a1/monitor0", "candidates"))
        cancelled = asyncio.ensure_future(box.take("a1/monitor1", "candidates"))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.sleep(0)
        box.put("a1/generator", "candidates", {"id": "g0c0"})
        assert await first == {"id": "g0c0"}
        for i in (1, 2, 3):
            box.put("a1/generator", "candidates", {"id": f"g0c{i}"})
        assert box.left() == {"a1/candidates": 3}
        assert [await box.take(f"a1/monitor{k}", "candidates") for k in (1, 0)] == [{"id": "g0c1"}, {"id": "g0c2"}]
        box.put("a10/generator", "candidates", None)
        waiting = asyncio.ensure_future(box.take("a1/monitor1", "candidates"))
        await asyncio.sleep(0)
        assert await waiting == {"id": "g0c3"}
        waiting = asyncio.ensure_future(box.take("a1/monitor1", "candidates"))
        await asyncio.sleep(0)
        box.put("a1/generator", "candidates", None)
        assert await waiting is None
        box.put("a1/generator", "candidates", {"id": "g1c0"})
        waiting = asyncio.ensure_future(box.take("a2/monitor0", "candidates"))   # another campaign's queue
        await asyncio.sleep(0)
        box.fail("a1", "a1/evaluator failed")
        box.fail("a2", "a2/evaluator failed")
        with pytest.raises(CampaignFailed):
            await waiting
        with pytest.raises(CampaignFailed):
            await box.take("a1/monitor0", "candidates")
        with pytest.raises(CampaignFailed):
            box.put("a1/generator", "candidates", {})
        assert box.dropped == {"a1/candidates": 1} and box.left() == {"a10/candidates": 1}
        assert await box.take("a10/monitor0", "candidates") is None and box.left() == {} and box.queues == {}
        assert box.sent == {"a1/generator": 6, "a10/generator": 1}
        assert box.received == {"a1/monitor0": 2, "a1/monitor1": 3, "a10/monitor0": 1}
    asyncio.run(check())


def test_an_inbox_gives_everything_that_has_arrived():
    """The asynchronous mode's receives: take_all waits for one message, then gives it and every one delivered
    meanwhile, in the order sent; with messages there it gives them at once; with wait=False it gives what is there,
    or none, at once. emptied returns at once when no item waits on the queue and otherwise when the last is taken;
    `most` counts the items a queue or inbox held at once, not those handed straight to a waiting taker. fail(a1)
    fails a1's inbox and emptied waits, now and later, and counts a1's inbox leftovers as dropped for their agent
    (its queue's under the queue's key); a take_all woken before the failure gets only the message that woke it,
    the rest counted dropped once, so every message sent is received or dropped, once."""
    from fusion_miniapp.layer_autonmi import CampaignFailed, Mailbox

    async def check():
        box = Mailbox(agents=5)
        waiting = asyncio.ensure_future(box.take_all("a1/evaluator", "report"))
        await asyncio.sleep(0)
        box.deliver("a1/monitor0", "evaluator", "report", {"c": 1})   # wakes it, handed straight over
        box.deliver("a1/monitor1", "evaluator", "report", {"c": 2})   # arrives before it resumes
        assert await waiting == [{"c": 1}, {"c": 2}] and box.queues == {}
        assert box.most["a1/evaluator/report"] == 1
        assert await box.take_all("a1/generator", "digest", wait=False) == [] and box.queues == {}
        for n in (1, 2, 3):
            box.deliver("a1/coordinator", "generator", "survivors", {"n": n})
        assert box.left() == {"a1/generator": 3} and box.most["a1/generator/survivors"] == 3
        assert await box.take_all("a1/generator", "survivors", wait=False) == [{"n": 1}, {"n": 2}, {"n": 3}]
        box.deliver("a1/evaluator", "coordinator", "scored", {"results": []})
        assert await box.take_all("a1/coordinator", "scored") == [{"results": []}] and box.left() == {}

        await box.emptied("a1/generator", "candidates")   # nothing waits: at once
        taker = asyncio.ensure_future(box.take("a1/monitor0", "candidates"))
        await asyncio.sleep(0)
        box.put("a1/generator", "candidates", {"id": "c0"})   # to the waiting taker: not held
        assert await taker == {"id": "c0"} and box.most["a1/candidates"] == 0
        for k in (1, 2):
            box.put("a1/generator", "candidates", {"id": f"c{k}"})
        emptied = asyncio.ensure_future(box.emptied("a1/generator", "candidates"))
        await asyncio.sleep(0)
        assert await box.take("a1/monitor0", "candidates") == {"id": "c1"}
        await asyncio.sleep(0)
        assert not emptied.done()
        assert await box.take("a1/monitor1", "candidates") == {"id": "c2"}
        await emptied
        assert box.most["a1/candidates"] == 2 and box.queues == {}

        box.put("a1/generator", "candidates", {"id": "c3"})
        box.deliver("a1/coordinator", "generator", "survivors", {"n": 4})
        box.deliver("a1/evaluator", "generator", "digest", {})
        emptied = asyncio.ensure_future(box.emptied("a1/generator", "candidates"))
        waiting = asyncio.ensure_future(box.take_all("a1/coordinator", "scored"))
        await asyncio.sleep(0)
        box.fail("a1", "a1/monitor1 failed")
        for f in (emptied, waiting):
            with pytest.raises(CampaignFailed):
                await f
        with pytest.raises(CampaignFailed):
            await box.take_all("a1/generator", "survivors", wait=False)
        with pytest.raises(CampaignFailed):
            box.deliver("a1/evaluator", "coordinator", "scored", {})
        assert box.dropped == {"a1/generator": 2, "a1/candidates": 1} and box.left() == {} and box.queues == {}

        waiting = asyncio.ensure_future(box.take_all("a2/evaluator", "report"))
        await asyncio.sleep(0)
        box.deliver("a2/monitor0", "evaluator", "report", {"c": 1})   # wakes it
        box.deliver("a2/monitor1", "evaluator", "report", {"c": 2})
        box.fail("a2", "a2/monitor1 failed")                          # before it resumes
        assert await waiting == [{"c": 1}]
        assert (box.received["a2/evaluator"], box.dropped["a2/evaluator"]) == (1, 1)
        a1 = [a for a in {*box.sent, *box.received, *box.dropped} if a.startswith("a1/")]
        assert sum(box.sent[a] for a in a1) == sum(box.received[a] + box.dropped[a] for a in a1) == 12
    asyncio.run(check())


def test_an_inbox_gives_at_most_its_limit_and_keeps_the_rest_in_order():
    """take_all with a limit (the asynchronous Evaluator's, with its model call: decide.SHOWN_RESULTS): the first
    `limit` messages in the order sent; the rest stay in the inbox and come with the next take at once, without a
    wait; a take that waited counts the message that woke it in its limit. Every message is received once, and on a
    failure the ones left are dropped, counted once."""
    from fusion_miniapp.layer_autonmi import Mailbox

    async def check():
        box = Mailbox(agents=5)
        for c in range(11):
            box.deliver("a1/monitor0", "evaluator", "report", {"c": c})
        assert await box.take_all("a1/evaluator", "report", limit=8) == [{"c": c} for c in range(8)]
        assert box.left() == {"a1/evaluator": 3}
        assert await box.take_all("a1/evaluator", "report", limit=8) == [{"c": c} for c in range(8, 11)]
        assert box.queues == {} and box.left() == {}
        waiting = asyncio.ensure_future(box.take_all("a1/evaluator", "report", limit=2))
        await asyncio.sleep(0)
        for c in (11, 12, 13):
            box.deliver("a1/monitor1", "evaluator", "report", {"c": c})
        assert await waiting == [{"c": 11}, {"c": 12}]
        assert await box.take_all("a1/evaluator", "report", wait=False, limit=8) == [{"c": 13}]
        assert box.received["a1/evaluator"] == 14 and box.queues == {}
        for c in range(3):
            box.deliver("a2/monitor0", "evaluator", "report", {"c": c})
        assert await box.take_all("a2/evaluator", "report", limit=1) == [{"c": 0}]
        box.fail("a2", "a2/monitor1 failed")
        assert (box.received["a2/evaluator"], box.dropped["a2/evaluator"]) == (1, 2) and box.left() == {}
    asyncio.run(check())


def test_every_id_seen_outside_the_agents_is_the_campaigns(tmp_path):
    """Whichever of campaign a4's agents asks, the evaluate task's agent_id (cold starts, work folders), the
    lookups (the abstract differs for a4 and a4/generator) and the store key of its memory are a4's."""
    import types

    from rascal.store import MemoryStore

    from fusion_miniapp import tools
    from fusion_miniapp.layer_autonmi import Fusion, Script

    pop = types.SimpleNamespace(index=0, agent_ids=[f"a4/{name}" for name in names()], state={}, store=MemoryStore())
    script = Script(Fusion(knob_files.base(), "fixture", tmp_path), pop)
    try:
        args = {"params": {}, "candidate_idx": 0, "generation": 0, "seed": 4}
        assert script.evaluate.prepare("a4/monitor1", "g0c0", args)["agent_id"] == "a4"
        script.remember("a4/coordinator", {"generation": 0})
        assert pop.store.keys() == ["campaign/a4"]
        async def lookups():
            return await asyncio.gather(script.lookup("a4/generator", "abstract"), tools.abstract("a4"),
                                        tools.abstract("a4/generator"))
        got, a4, hosted = asyncio.run(lookups())
        assert got == a4 != hosted and script.lookups_made == {"a4/generator": 1}
    finally:
        script.lines.close()


def test_fixture_threads_keep_one_event_loop_each():
    """RHAPSODY's thread backend runs each coroutine task with asyncio.run, whose end walks every task of
    the process (a cost that grows with the live campaigns); the fixture pool the layer gives AUTONMI runs
    them on one loop per thread. Other calls, and its queued work at close, are as run.py's pool."""
    from rhapsody.backends import ConcurrentExecutionBackend

    from fusion_miniapp.layer_autonmi import fixture_pool

    async def loop_of_task():
        return asyncio.get_running_loop()

    pool = fixture_pool(1)
    loops = [pool.submit(ConcurrentExecutionBackend._run_in_thread, loop_of_task, (), {}).result() for _ in range(5)]
    assert len({id(loop) for loop in loops}) == 1 and not loops[0].is_closed()   # the list keeps each alive
    assert pool.submit(sum, (1, 2)).result() == 3
    pool.shutdown()
    assert loops[0].is_closed()


def test_autonmi_layer_has_the_mini_app_shape():
    """The script-shape check that workflow-mini-apps applies to its own mini-apps, from the clone wfMiniAPI
    is installed from."""
    spec = importlib.util.find_spec("wfMiniAPI")
    script = Path(spec.origin).resolve().parents[2] / "tests" / "equivalence" / "contract_check.py"
    if not script.exists():
        pytest.skip(f"needs workflow-mini-apps' tests: {script}")
    sys.path.insert(0, str(script.parent))
    try:
        mod_spec = importlib.util.spec_from_file_location("contract_check", script)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script.parent))
    layer = Path(importlib.util.find_spec("fusion_miniapp.layer_autonmi").origin)
    assert module.check_example(layer) == []

def test_asyncflow_running_set_yields_only_completed_components():
    """AsyncFlow's engine loop checks every running component on each pass; the layer's set makes the loop's
    scan visit only those that completed (membership, size and discard as a plain set)."""
    import types

    from radical.asyncflow import WorkflowEngine

    from fusion_miniapp import layer_autonmi

    assert layer_autonmi.RUNNING_SHIM and WorkflowEngine._fusion_running_shim

    async def check():
        loop = asyncio.get_running_loop()
        engine = types.SimpleNamespace(components={u: {"future": loop.create_future()} for u in "abc"})
        running = layer_autonmi._RunningSet(engine)
        for u in "abc":
            running.add(u)
        assert list(running) == [] and "a" in running and len(running) == 3
        engine.components["b"]["future"].set_result(None)
        await asyncio.sleep(0)                           # the done-callback runs on the loop
        assert list(running) == ["b"]
        running.discard("b")
        assert list(running) == [] and len(running) == 2 and "b" not in running
    asyncio.run(check())


def test_the_asyncflow_shim_can_be_turned_off():
    code = ("from fusion_miniapp import layer_autonmi as m; from radical.asyncflow import WorkflowEngine as W; "
            "print(m.RUNNING_SHIM, getattr(W, '_fusion_running_shim', False))")
    proc = subprocess.run([sys.executable, "-c", code], env={**os.environ, "FUSION_ASYNCFLOW_RUNNING_SHIM": "0"},
                          capture_output=True, text=True, timeout=300, check=False)
    assert proc.stdout.split() == ["False", "False"], proc.stderr[-2000:]


def test_automatic_full_collections_are_off_unless_asked_for(monkeypatch):
    """bound_full_collections raises only CPython's third threshold, and leaves it with FUSION_GC_FULL_AUTO=1."""
    import gc

    from fusion_miniapp import layer_autonmi

    before = gc.get_threshold()
    try:
        gc.set_threshold(700, 10, 10)
        monkeypatch.setenv("FUSION_GC_FULL_AUTO", "1")
        assert layer_autonmi.bound_full_collections() == (700, 10, 10) and layer_autonmi.full_gc_auto()
        monkeypatch.delenv("FUSION_GC_FULL_AUTO")
        assert layer_autonmi.bound_full_collections() == (700, 10, layer_autonmi.FULL_GC_THRESHOLD)
        assert layer_autonmi.bound_full_collections() == (700, 10, layer_autonmi.FULL_GC_THRESHOLD)   # idempotent
        assert not layer_autonmi.full_gc_auto()
    finally:
        gc.set_threshold(*before)


def test_the_autonmi_commit_imported_is_checked_against_the_pin(monkeypatch, capsys):
    """autonmi_pin: the commit of the autonmi package imported against layer_autonmi.AUTONMI_COMMIT; one warning
    on stderr when it differs, is unknown, or has changed files; none when it matches."""
    import autonmi

    from fusion_miniapp import layer_autonmi

    requirements = Path(__file__).resolve().parents[1] / "requirements-autonmi.txt"
    assert layer_autonmi.AUTONMI_COMMIT in requirements.read_text()   # the pin it documents is the one checked
    got = layer_autonmi.autonmi_commit()   # this environment's: an editable clone, or a git install
    if got["from"] == "git":
        src = Path(autonmi.__file__).resolve().parent
        assert got["commit"] == subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                                               text=True, check=True).stdout.strip()
    pin = "a" * 40
    monkeypatch.setattr(layer_autonmi, "AUTONMI_COMMIT", pin)
    for commit, dirty, matches in ((pin, False, True), ("b" * 40, False, False), (pin, True, False),
                                   (None, None, False)):
        monkeypatch.setattr(layer_autonmi, "autonmi_commit",
                            lambda c=commit, d=dirty: {"commit": c, "dirty": d, "from": "git", "path": "clone"})
        rec = layer_autonmi.autonmi_pin()
        err = capsys.readouterr().err.splitlines()
        assert (rec["commit"], rec["pin"], rec["matches_pin"]) == (commit, pin, matches) and "path" not in rec
        assert len(err) == (0 if matches else 1), err
        if not matches:
            assert "warning" in err[0] and "aaaaaaaaaaaa" in err[0] and "(clone)" in err[0], err


def test_a_run_off_the_pin_warns_once_and_records_it(tmp_path):
    """A run whose autonmi commit is not the pin (the pin changed here) warns once and still runs; its record
    names both."""
    out = tmp_path / "autonmi"
    code = ("import sys; from fusion_miniapp import layer_autonmi as m; m.AUTONMI_COMMIT = '0' * 40; "
            f"sys.exit(m.main(['--sim', 'fixture', '--agents', '2', '--out', {str(out)!r}]))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                          check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    warnings = [x for x in proc.stderr.splitlines() if "warning: the autonmi package imported" in x]
    assert len(warnings) == 1 and "pinned 000000000000" in warnings[0], proc.stderr[-3000:]
    rec = json.loads((out / "run_record.json").read_text())["autonmi"]["framework_commit"]
    assert rec["pin"] == "0" * 40 and rec["matches_pin"] is False and "path" not in rec, rec
    assert rec["commit"] or rec["from"] is None, rec   # unknown only when neither git nor the installer says


# --- the asynchronous mode (campaign.mode: async) ------------------------------------------------------------

ASYNC_KNOBS = {"campaign": {"mode": "async"}, "agents": {"start_together": True}, "llm": {"mode": "off"},
               "outer": {"population": 4, "generations": 4, "parent_stdout": "drop"}}   # 16 candidates a campaign
ASYNC_CALLS = {**ASYNC_KNOBS, "llm": {}}   # the stub LLM (FIXTURE): every role makes its own call
ASYNC_ARGS = ("--preset", "smoke", "--sim", "fixture", "--agents", "20", "--populations", "2")
ASYNC_CHECKS = ("every_campaign_reports", "every_agent_finishes", "evaluations_and_model_calls_as_planned",
                "every_message_received", "lookups_match_autonmi", "store_holds_only_the_memories",
                "each_candidate_evaluated_once", "survivors_within_the_population", "inner_loops_within_the_slots",
                "candidates_waiting_within_take", "every_agent_has_a_row")


@pytest.fixture(scope="module")
def async_run(tmp_path_factory):
    """One asynchronous run with no model calls (llm.mode off) and record.extras: 20 fixture campaigns of 16
    candidates (population 4), all agents starting together, in 2 populations, with each campaign's arrivals
    reordered (tests/async_replay.py run as a
    script: every fourth candidate's report and each Coordinator cycle wait on the event loop), so results reach
    the Evaluator out of candidate order and the Coordinator takes several Evaluator cycles at once. Its folder
    and run record."""
    d = tmp_path_factory.mktemp("async")
    proc = subprocess.run([sys.executable, str(Path(__file__).with_name("async_replay.py")), *ASYNC_ARGS, "--knobs",
                           knob_file(d, {**ASYNC_KNOBS, "record": {"extras": True}}), "--out", str(d / "autonmi")],
                          cwd=d, check=False, timeout=900, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return d / "autonmi", json.loads((d / "autonmi" / "run_record.json").read_text())


def reordered_run(d: Path, knobs: dict, *args, code: int = 0) -> tuple[Path, dict]:
    """An asynchronous run of ASYNC_ARGS' 20 campaigns with each campaign's arrivals reordered (async_run); its
    folder and run record."""
    proc = subprocess.run([sys.executable, str(Path(__file__).with_name("async_replay.py")), *ASYNC_ARGS, "--knobs",
                           knob_file(d, knobs), *args, "--out", str(d / "autonmi")],
                          cwd=d, check=False, timeout=900, capture_output=True, text=True)
    assert proc.returncode == code, proc.stderr[-3000:]
    return d / "autonmi", json.loads((d / "autonmi" / "run_record.json").read_text())


@pytest.fixture(scope="module")
def async_calls_run(tmp_path_factory):
    """async_run with the stub LLM (FIXTURE, at 1 ms a call): every role makes its own model call."""
    return reordered_run(tmp_path_factory.mktemp("async_calls"), {**ASYNC_CALLS, "record": {"extras": True}})


def test_the_asynchronous_mode_keeps_its_invariants(async_run):
    """campaign.mode async, no model calls (llm.mode off): each campaign's 16 candidates are evaluated once each by its Monitors,
    every message is received, the survivors stay within the population, at most campaign.slots inner loops run
    and at most decide.TAKE candidates wait at once, and AUTONMI's checks hold them all; each role's counts are
    within messages()' bounds and tied to its cycles (the Generator's 8 takes of 2); the counters behind those
    bounds are live (both slots ran at once, the survivors reached the population); all 100 agents are live at
    once (agents.start_together); the record marks the mode and counts the results scored and kept, the first
    campaign.warmup of each campaign kept whatever their score; a campaign's line has no generations, and its
    evaluations' submission times beside their sim_s and wait_s."""
    out, rec = async_run
    knobs, n, B, S = rec["knobs"], 20, 16, 2
    assert rec["status"] == "pass" and rec["autonmi"]["ok"] and rec["campaign_mode"] == "async", rec["autonmi"]
    checks = {c["name"]: c for c in json.loads((out / "app" / "checks.json").read_text())}
    assert {k: c["ok"] for k, c in checks.items()} == dict.fromkeys(ASYNC_CHECKS, True), checks
    c = rec["counts"]
    assert (c["sims"], c["llm_calls"], c["generations"], c["agents_failed"]) == (n * B, 0, 0, 0)
    assert c["candidates_by_source"] == {"start": n * 4, "survivor": 0, "mutation": n * 12, "model": 0, "restart": 0}
    assert {role: c["by_role"][role]["model_calls"] for role in ROLES} == dict.fromkeys(ROLES, 0)
    assert c["by_role"]["monitor"]["simulations"] == c["by_role"]["monitor"]["reports"] == n * B == c["suspensions"]
    a = c["asynchronous"]
    assert a["results_scored"] == n * B and a["results_scored_after_warmup"] == n * (B - 4)
    assert a["results_kept"] == a["results_kept_after_warmup"] + n * 4 and 0 < a["results_kept_after_warmup"] < n * (B - 4)
    assert a["cycles"]["generator"] == n * B // 2 and a["cycles"]["monitor"] == n * B
    assert n <= a["cycles"]["coordinator"] <= a["cycles"]["evaluator"] <= n * B
    assert c["memory_writes"] == a["cycles"]["coordinator"]
    exp = c["expected_by_role"]
    assert (exp["coordinator"]["sent"], exp["generator"]["received"], exp["evaluator"]["sent"]) == \
        ([3, B + 2], [3, 2 * B + 1], [2, 2 * B])
    rows = app_rows(out)
    assert len(rows) == n * (3 + S) and all(r["finished"] and not r["messages_left"] and not r["messages_dropped"]
                                            for r in rows)
    lines = {r["agent_id"]: r for r in lines_of(out)}
    for campaign in lines:
        x = {r["role"] if r["role"] != "monitor" else r["agent_id"]: r for r in rows if r["campaign"] == campaign}
        co, ge, ev = x["coordinator"], x["generator"], x["evaluator"]
        mons = [x[f"{campaign}/monitor{k}"] for k in range(S)]
        assert sum(r["simulations"] for r in mons) == B and all(r["simulations"] == r["cycles"] for r in mons)
        assert (co["candidates_distinct"], co["candidate_max"], co["candidates_repeated"]) == (B, B - 1, 0)
        assert co["survivors_max"] == 4 and co["running_max"] == S and 1 <= co["waiting_max"] <= 2, co
        assert co["messages_sent"] == co["cycles"] + 2 and ev["messages_sent"] == 2 * ev["cycles"]
        assert ge["messages_received"] == co["cycles"] + ev["cycles"] + 1 and co["messages_received"] == ev["cycles"]
        assert (ge["cycles"], ge["messages_sent"], ev["messages_received"]) == (B // 2, B + S, B + 1)
        assert co["results_scored"] == B and co["results_kept"] >= 4
        line = lines[campaign]
        assert (line["n_sims"], line["n_llm_calls"], line["summary"], line["n_cycles"]) == (B, 0, "", co["cycles"])
        assert line["survival"] == {k: co[k] for k in line["survival"]}
        assert line["generations"] == [] and len(line["t_submit"]) == len(line["sim_s"]) == len(line["wait_s"]) == B
    live = rec["speed"]["agents_live"]
    assert live["max"] == n * (3 + S) and live["all_live_s"] > 0 and rec["speed"]["start_together"]["begin_margin_s"] >= 0
    s = rec["speed"]   # the evaluations' times, from t_submit
    assert s["sim_s"]["n"] == s["queue_wait_s"]["n"] == n * B and s["generation_s"] == {"n": 0}
    assert 1 <= s["sims_in_flight"]["max"] <= n * (2 + S) and 1 <= s["sims_executing"]["max"]   # waiting and running


def test_a_recorded_order_replays_to_the_same_campaign(async_run):
    """With record.extras each campaign's line has the orders its results reached the Evaluator and the
    Coordinator (one list per cycle) and the survivors message its Generator held at each profile: replayed
    through decide.py's pure functions (tests/async_replay.py), with the Monitor's m as recorded and the fixture
    evaluating the same profiles, they give the same campaign: every result's q, m, score, draw and kept flag,
    the survivors at the end and best-so-far. The run's arrivals are reordered (async_run), and the orders it
    recorded show it: results out of candidate order, and Coordinator cycles that took more than one Evaluator
    cycle, so the replay meets both (and the candidate order would not replay)."""
    from async_replay import fixture, replay

    out, rec = async_run
    knobs = rec["knobs"]
    lines = lines_of(out)
    assert len(lines) == 20
    for line in lines:
        x, seed = line["asynchronous"], line["seed"]
        assert sorted(sum(x["evaluator"], [])) == sorted(sum(x["coordinator"], [])) == list(range(16))
        assert len(x["held"]) == 16 and x["held"] == sorted(x["held"])
        m = {r["candidate"]: r["m"] for r in x["results"]}
        cst, rows, _ = replay(seed, knobs, x["evaluator"], x["coordinator"], x["held"], m, fixture(knobs, seed))
        assert rows == x["results"], line["agent_id"]
        assert [s["candidate"] for s in cst["survivors"]] == x["survivors"], line["agent_id"]
        assert json.loads(json.dumps(cst["best"])) == line["best"], line["agent_id"]
    evaluator = [line["asynchronous"]["evaluator"] for line in lines]
    assert sum(sum(x, []) != sorted(sum(x, [])) for x in evaluator) > 0   # out of candidate order
    assert sum(line["asynchronous"]["evaluator"] != line["asynchronous"]["coordinator"] for line in lines) > 0
    assert sum(len(x) < 16 for x in (line["asynchronous"]["coordinator"] for line in lines)) > 0

    def by_candidate(rows):
        return {r["candidate"]: (r["q"], r["score"], r["kept"]) for r in rows}
    changed = 0   # the out-of-order campaigns replayed in candidate order, one result a cycle: other q, scores, kept
    for line in lines:
        x, seed = line["asynchronous"], line["seed"]
        if sum(x["evaluator"], []) != list(range(16)):
            one, m = [[k] for k in range(16)], {r["candidate"]: r["m"] for r in x["results"]}
            _, rows, _ = replay(seed, knobs, one, one, sorted(x["held"]), m, fixture(knobs, seed))
            changed += by_candidate(rows) != by_candidate(x["results"])
    assert changed > 0


def test_the_asynchronous_record_gains_only_counts_without_the_toggle(tmp_path, async_run):
    """record.extras off: no campaign line has per-result detail, and the record gains only the mark of the mode
    and its counts (counts.asynchronous) beside a generations-mode record of the same campaigns; a line gains
    only counts (n_cycles, survival) and its evaluations' submission times (t_submit, beside sim_s and wait_s,
    which replace the generations' t0), and its lists are no longer than P x G (its generations none); with
    every role's model call (the stub) the record and the lines have the same keys and lists as without, the
    calls only counted (no line has a call's details: llm_call_details stays empty, and the record's model
    figures, llm_service and the client-side requests in flight, count every call from the populations' sums,
    whose files are not left beside the record); on: each
    line gains `monitor` (the Monitor's numbers, q and label of each candidate) and `asynchronous` (each result's row:
    arrival, candidate, q, m, score, draw, kept; and the replay's orders), outside results_sha256. Two
    asynchronous records agree in compare_records on their counts and invariants, not their (differing) digests."""
    def paths(d, prefix=""):
        return {p for k, v in d.items() for p in (paths(v, f"{prefix}{k}.") if isinstance(v, dict) and v else
                                                  [f"{prefix}{k}"])}
    on_out, on = async_run
    gen_knobs = {**ASYNC_KNOBS, "campaign": {"mode": "generations"}, "llm": {}}
    for name, knobs in (("off", ASYNC_KNOBS), ("gen", gen_knobs), ("calls", ASYNC_CALLS)):
        (tmp_path / name).mkdir()
        rec = run("autonmi", tmp_path / name, *ASYNC_ARGS, "--knobs", knob_file(tmp_path / name, knobs))
        assert rec["status"] == "pass"
    off, gen, calls = (json.loads((tmp_path / x / "autonmi" / "run_record.json").read_text()) for x in ("off", "gen",
                                                                                                         "calls"))
    calls_lines = lines_of(tmp_path / "calls" / "autonmi")
    keyed = ("speed.", "llm_service.")   # the speed figures and the model calls' figures, keyed by their values
    assert {p for p in paths(calls) if not p.startswith(keyed)} == {p for p in paths(off) if not p.startswith(keyed)}
    assert {k for r in calls_lines for k in r} == {k for r in lines_of(tmp_path / "off" / "autonmi") for k in r}
    assert all(r["llm_call_details"] == [] and r["n_llm_calls"] > 0 for r in calls_lines)
    assert not any("note" in json.dumps(r) or "guidance" in json.dumps(r) for r in calls_lines)
    by = calls["counts"]["by_role"]
    assert calls["llm_service"]["calls"] == sum(by[role]["model_tries"] for role in ROLES) == \
        sum(by[role]["model_calls"] for role in ROLES) == calls["speed"]["model_requests_in_flight"]["calls_timed"] > 0
    assert calls["speed"]["rates_per_s"]["model_calls_all_agents"] > 0 and calls["llm_service"]["tokens"]["completion"] > 0
    assert not list((tmp_path / "calls" / "autonmi").glob("calls-p*"))
    # the speed figures, knob echo, per-role bounds and model calls' figures (none here) of either mode
    skip = ("speed.", "knobs.", "counts.expected_by_role.", "llm_service.")
    added = {p for p in paths(off) - paths(gen) if not p.startswith(skip)}
    assert added == {"campaign_mode", *(f"counts.asynchronous.{k}" for k in (
        "results_scored", "results_kept", "results_scored_after_warmup", "results_kept_after_warmup",
        "cycles.coordinator", "cycles.generator", "cycles.evaluator", "cycles.monitor", "survivor_waits",
        "parameters_filled"))}, added
    assert not {p for p in paths(gen) - paths(off) if not p.startswith(skip)}
    # the toggle's record figures: the request settings sent (none here), the servers (none: the stub)
    # and each role's share of time in a model call (0 here)
    assert paths(on) - paths(off) == {"llm_service.requests", "llm_service.servers", *(
        f"speed.model_call_share.{k}" for k in ("window", "source", "agents_in_a_call_per_campaign")), *(
        f"speed.model_call_share.by_role.{role}" for role in ROLES)} and not paths(off) - paths(on)
    assert on["speed"]["model_call_share"]["by_role"] == dict.fromkeys(ROLES, 0.0)
    assert on["llm_service"]["requests"] == []
    assert not any({"monitor", "asynchronous"} & set(r) for r in lines_of(tmp_path / "off" / "autonmi"))
    off_lines, gen_lines = lines_of(tmp_path / "off" / "autonmi"), lines_of(tmp_path / "gen" / "autonmi")
    assert {k for r in off_lines for k in r} - {k for r in gen_lines for k in r} == {"n_cycles", "survival", "t_submit"}
    assert {k for r in gen_lines for k in r} - {k for r in off_lines for k in r} == set()
    assert {k for r in off_lines for k in r["survival"]} == {f"results_{x}" for x in (
        "scored", "kept", "scored_after_warmup", "kept_after_warmup")}

    def lists(rows):   # each list in a campaign's line, with its length
        return {(k, len(v)) for r in rows for k, v in r.items() if isinstance(v, list)}
    assert lists(gen_lines) == {("generations", 4), ("llm_call_details", 4), ("sim_s", 16), ("wait_s", 16)}
    assert lists(off_lines) == {("generations", 0), ("llm_call_details", 0), ("sim_s", 16), ("wait_s", 16),
                                ("t_submit", 16)} == lists(calls_lines)   # no list grows with the model calls
    for r in lines_of(on_out):
        assert set(r["asynchronous"]) == {"evaluator", "coordinator", "held", "results", "survivors", "guidance"}
        assert r["asynchronous"]["guidance"] == []   # llm.mode off: no guidance
        assert [x["arrival"] for x in r["asynchronous"]["results"]] == list(range(16))
        assert all(set(x) == {"arrival", "candidate", "q", "m", "score", "draw", "kept"} for x in r["asynchronous"]["results"])
        assert [x["candidate_idx"] for x in r["monitor"]] == [x["candidate"] for x in r["asynchronous"]["results"]]
        assert {x["label_from"] for x in r["monitor"]} == {"trace_end_synthetic"}
    assert record.digest((on_out / "agents.jsonl").read_text().splitlines()) == on["results_sha256"]
    assert compare_records.compare(on, off)["results"] == []   # record.extras is what the record writes
    assert compare_records.compare(off, gen)["results"][:1] != []
    worse = {**off, "counts": {**off["counts"], "sims": off["counts"]["sims"] - 1}}
    assert compare_records.compare(on, worse)["results"] == [f"counts.sims: {on['counts']['sims']} != {worse['counts']['sims']}"]


def test_verify_catches_each_broken_invariant(async_run):
    """AUTONMI's checks on the asynchronous run's own rows pass; each invariant broken in a copy of them fails
    its check (and only the checks that read it): evaluations per campaign, each candidate once, every message
    received, the survivors, inner loops and candidates waiting within their bounds, and each role's counts
    within their bounds, tied to its cycles, with no model calls."""
    from rascal.store import MemoryStore

    from fusion_miniapp.layer_autonmi import Fusion

    out, rec = async_run
    rows, store = app_rows(out), MemoryStore()
    for c in {r["campaign"] for r in rows}:
        store.put(f"campaign/{c}", "{}")
    definition = Fusion(rec["knobs"], "fixture", out)

    def failing(edit) -> set:
        broken = copy.deepcopy(rows)
        for r in broken:
            if r["campaign"] == "a5":
                edit(r)
        return {c.name for c in definition.verify(broken, store) if not c.ok}
    assert failing(lambda r: None) == set()

    def one_more_evaluation(r):   # a Monitor that evaluated (and reported) one candidate more
        if r["agent_id"] == "a5/monitor0":
            for k in ("simulations", "reports", "messages_sent", "messages_received", "cycles"):
                r[k] += 1
    planned = {"evaluations_and_model_calls_as_planned"}
    breaks = {
        "evaluations per campaign": (one_more_evaluation, planned),
        "a candidate evaluated twice": (lambda r: r.update(candidates_repeated=1) if r["role"] == "coordinator"
                                        else None, {"each_candidate_evaluated_once"}),
        "a candidate never evaluated": (lambda r: r.update(candidates_distinct=15) if r["role"] == "coordinator"
                                        else None, {"each_candidate_evaluated_once"}),
        "a message left": (lambda r: r.update(messages_left=1) if r["role"] == "evaluator" else None,
                           {"every_message_received"}),
        "survivors over the population": (lambda r: r.update(survivors_max=5) if r["role"] == "coordinator"
                                          else None, {"survivors_within_the_population"}),
        "inner loops over the slots": (lambda r: r.update(running_max=3) if r["role"] == "coordinator" else None,
                                       {"inner_loops_within_the_slots"}),
        "candidates waiting over two": (lambda r: r.update(waiting_max=3) if r["role"] == "coordinator" else None,
                                        {"candidates_waiting_within_take"}),
        "a model call": (lambda r: r.update(model_calls=1) if r["role"] == "evaluator" else None, planned),
        "the Coordinator past its bound": (lambda r: r.update(messages_received=17) if r["role"] == "coordinator"
                                           else None, planned),
        "the Generator's takes": (lambda r: r.update(cycles=9) if r["role"] == "generator" else None, planned),
        "the Evaluator's sends and cycles": (lambda r: r.update(messages_sent=r["messages_sent"] + 1)
                                             if r["role"] == "evaluator" else None, planned),
        "a survivors message not received": (lambda r: r.update(messages_received=r["messages_received"] - 1)
                                             if r["role"] == "generator" else None, planned),
    }
    for what, (edit, want) in breaks.items():
        assert failing(edit) == want, what


@pytest.mark.parametrize("calls", (False, True))
@pytest.mark.parametrize("role, at", (("coordinator", "2"), ("generator", "3"), ("evaluator", "2"),
                                      ("monitor0", "c5")))
def test_an_agent_that_fails_ends_its_campaign_alone_in_the_asynchronous_mode(tmp_path, role, at, calls):
    """A Coordinator before its third survivors message, a Generator before its fourth take, an Evaluator before
    its third cycle's sends, or a Monitor, the campaign's only slot, before its report of candidate 5
    (FUSION_AUTONMI_FAIL): its campaign is one error line with the failing role; the campaign's other agents end
    with an error instead of waiting, each message it sent received or dropped; the other 19 campaigns end whole.
    The run ends on its own and fails. The same with no model calls (llm.mode off) and with every role's own
    call (the stub, FIXTURE), when the other agents may be in a model call as the campaign fails."""
    env = {**os.environ, "FUSION_AUTONMI_FAIL": f"a3/{role}:{at}"}
    slots = 1 if role == "monitor0" else 2
    knobs = {**(ASYNC_CALLS if calls else ASYNC_KNOBS), "campaign": {**ASYNC_KNOBS["campaign"], "slots": slots}}
    rec = run("autonmi", tmp_path, *ASYNC_ARGS, "--knobs", knob_file(tmp_path, knobs), env=env, code=1)
    out, name = tmp_path / "autonmi", role.rstrip("0")
    errors = [r for r in lines_of(out) if "error" in r]
    assert [(r["agent_id"], r["role"]) for r in errors] == [("a3", name)]
    assert "InjectedFailure" in errors[0]["error"] and f"a3/{role}" in errors[0]["error"]
    assert (rec["status"], rec["counts"]["agents"], rec["counts"]["agents_failed"]) == ("fail", 20, 1)
    assert all(r["n_sims"] == 16 for r in lines_of(out) if "error" not in r)
    rows = {r["agent_id"]: r for r in app_rows(out)}
    assert [rows[f"a3/{x}"]["status"] for x in names(slots)] == ["failed"] * (3 + slots)
    assert all("CampaignFailed" in rows[f"a3/{x}"]["error"] for x in names(slots) if x != role)
    assert all(r["finished"] for a, r in rows.items() if not a.startswith("a3/")) and len(rows) == 20 * (3 + slots)
    assert not any(r["messages_left"] for r in rows.values())
    if role == "monitor0":   # candidates 0 to 5 ran
        assert rows["a3/monitor0"]["simulations"] == 6
    a3 = [rows[f"a3/{x}"] for x in names(slots)]
    assert sum(r["messages_sent"] for r in a3) == sum(r["messages_received"] + r["messages_dropped"] for r in a3)
    assert rec["autonmi"]["failed_checks"] == ["every_campaign_reports", "every_agent_finishes"]


@pytest.mark.parametrize("layer", ("langgraph", "academy"))
def test_the_other_layers_refuse_the_asynchronous_mode(tmp_path, layer):
    """run.py's layers run the generations mode alone: campaign.mode async is refused before anything starts."""
    args = ["--use-academy"] if layer == "academy" else []
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", ENTRY["langgraph"], *args, *FIXTURE_ARGS, "--knobs",
                           knob_file(tmp_path, {"campaign": {"mode": "async"}}), "--out", str(out)], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode != 0 and "campaign.mode async is not run by this layer" in proc.stderr, proc.stderr[-2000:]
    assert not out.exists()


@pytest.mark.parametrize("population, generations, slots", ((1, 3, 2), (3, 3, 1), (5, 2, 16)))
def test_odd_shapes_run_in_the_asynchronous_mode(tmp_path, population, generations, slots):
    """A population of 1 (whose second candidate needs a survivor before any result exists), an odd population and
    budget (a take of one), and more Monitor slots than candidates: every campaign completes and AUTONMI's checks
    of the mode pass."""
    knobs = {"campaign": {"mode": "async", "slots": slots},
             "outer": {"population": population, "generations": generations, "parent_stdout": "drop"}}
    rec = run("autonmi", tmp_path, "--preset", "smoke", "--sim", "fixture", "--agents", "6", "--knobs",
              knob_file(tmp_path, knobs))
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    assert rec["counts"]["sims"] == 6 * population * generations
    assert rec["counts"]["asynchronous"]["cycles"]["generator"] >= 6 * -(-population * generations // 2)


STELLAR_LLM = {"thinking": "off", "reply_cap": 4096, "temperature": 1.0, "top_p": 0.95}   # the StellarAI knob file's


def run_on_a_server(tmp_path, knobs: dict, server: FakeVllm, *args, code: int = 0) -> tuple[dict, list[dict]]:
    """An AUTONMI run of SERVER_ARGS with --llm openai on the server: its record and its requests."""
    (tmp_path / "endpoints").write_text(server.url + "\n")
    rec = run("autonmi", tmp_path, *SERVER_ARGS, "--knobs", knob_file(tmp_path, knobs), "--populations", "2",
              "--endpoints-file", str(tmp_path / "endpoints"), *args, code=code)
    return rec, server.take()


@pytest.mark.parametrize("mode, thinking", (("async", "off"), ("generations", "off"), ("async", "on")))
def test_the_stellarai_settings_reach_every_request(tmp_path, mode, thinking):
    """llm.mode openai against the fake vLLM server (a FIXTURE) with the StellarAI knob file's request settings
    (llm.thinking off, llm.reply_cap 4,096, temperature 1.0, top_p 0.95): every request carries them, with
    chat_template_kwargs {enable_thinking: false} from AUTONMI's transport (no per-call extra_body replaced it)
    and no thinking budget. In the asynchronous mode every role calls, and only the Generator's calls carry its
    JSON schema as response_format; in the generations mode (with campaign.notes) the refills, summaries, notes
    and assessments carry the same settings and no schema. With llm.thinking on: no chat_template_kwargs, the
    thinking budget (agents.thinking_budget, 64 here) sent and added to the cap. Without record.extras a line
    holds only the generations mode's refill and summary calls' details; the record counts every call, and its
    tokens per second over the steady window are the fixture's usage counts (100 prompt, 50 generated a reply)."""
    knobs = {**SERVER_KNOBS, "llm": {**SERVER_KNOBS["llm"], **STELLAR_LLM, "thinking": thinking},
             "campaign": {"mode": "async"} if mode == "async" else {"notes": True}}
    with FakeVllm() as server:
        rec, sent = run_on_a_server(tmp_path, knobs, server)
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    by = rec["counts"]["by_role"]
    assert len(sent) == rec["llm_service"]["calls"] == sum(by[role]["model_calls"] for role in ROLES)
    assert all(by[role]["model_calls"] > 0 for role in ROLES)
    for body in sent:
        if thinking == "off":
            assert body["chat_template_kwargs"] == {"enable_thinking": False} and "thinking_token_budget" not in body
        else:
            assert "chat_template_kwargs" not in body and body["thinking_token_budget"] == 64
        assert (body["max_tokens"], body["temperature"], body["top_p"]) == (4096 + 64 * (thinking == "on"), 1.0, 0.95)
        many = re.search(r"JSON list of (\d+)", body["messages"][-1]["content"].splitlines()[-1])
        if mode == "async" and many:
            assert body["response_format"] == decide.candidates_schema(int(many.group(1)))
        else:
            assert "response_format" not in body
    if mode == "async":
        assert sum("response_format" in body for body in sent) == by["generator"]["model_calls"] == 12 * 6
    # tokens per second (on by default; in the asynchronous mode from the calls summed per population, CallSums):
    # the fixture's 100 prompt and 50 generated tokens a reply
    tokens = rec["speed"]["model_tokens_per_s"]
    assert tokens["calls"] > 0 and not tokens["emulated"]
    assert tokens["prompt_per_s"] == pytest.approx(2 * tokens["generated_per_s"], rel=1e-3)
    assert tokens["generated_per_s"] * tokens["window_s"] == pytest.approx(50 * tokens["calls"], rel=1e-3)
    details = [c for r in lines(tmp_path, "autonmi") for c in r["llm_call_details"]]
    if mode == "async":
        assert details == []
    else:
        assert {c["role"] for c in details} == {"generator", "coordinator"}
        assert len(details) == by["generator"]["model_calls"] + by["coordinator"]["model_calls"] < len(sent)


PLANTED = {"key": "planted-servers-key-5d1e9c", "VLLM_API_KEY": "planted-vllm-env-key-77ab", "HF_TOKEN":
           "planted-hf-token-0c3f", "RASCAL_EXTRA_KEY": "planted-rascal-key-91e2"}   # FAKE secrets, for the tests


def fake_fleet(folder: Path, urls: list[str]) -> Path:
    """A fake model job's folder (AUTONMI's FLEET_DIR; FIXTURE files): its endpoints file, its key (api-key, the
    planted key), gpu.csv, a server log of 800 lines that holds the key and the environment's planted secrets (as a
    careless launch script might write them), a job output that holds one, and a token file. Returns the endpoints
    file."""
    folder.mkdir()
    (folder / "endpoints").write_text("".join(u + "\n" for u in urls))
    (folder / "api-key").write_text(PLANTED["key"] + "\n")
    (folder / "gpu.csv").write_text("2026/11/24 00:00:00.000, 0, 97, 150000\n")
    log = [f"INFO line {i}" for i in range(800)]
    log[3] = f"non-default args: {{'api_key': '{PLANTED['key']}', 'port': 8100}}"
    log[5] = f"env VLLM_API_KEY={PLANTED['VLLM_API_KEY']} RASCAL_EXTRA_KEY={PLANTED['RASCAL_EXTRA_KEY']}"
    (folder / "vllm-0.log").write_text("\n".join(log) + "\n")
    (folder / "model-123.out").write_text(f"model job 123, token {PLANTED['HF_TOKEN']}\n")
    (folder / "hf_token").write_text(PLANTED["HF_TOKEN"] + "\n")
    return folder / "endpoints"


# FAKE secrets that only the model job's files hold (not this process's environment), one of them short; and one
# FAKE secret of the environment that holds another (set in the environment too)
IN_FILES_ONLY = {"WANDB_API_KEY": "planted-wandb-file-only-a1", "--hf-token": "planted-flag-b2",
                 "--api-key": "planted-flag-c3", "OTHER_TOKEN": "s3cr7", "hf_token": "planted-json-d4",
                 "url": "planted-url-token-e5", "GITHUB_TOKEN": "planted-after-a-name-f6"}
NESTED = {"OUTER_TOKEN": "planted-outer-NESTEDVAL-tail9", "INNER_KEY": "NESTEDVAL"}


def test_a_planted_secret_never_reaches_a_copied_setup_file(tmp_path, monkeypatch):
    """copy_model_service (record.extras, --llm openai) on a fake model job's folder, with planted FAKE secrets in
    its files and in the environment: the key file and the files named for secrets (also in the plural: secrets.json,
    tokens.txt, api_keys.txt) are not copied; the log is copied as its first 500 lines; environment.txt keeps only
    the allowed variables (not VLLM_API_KEY, HF_TOKEN or a RASCAL_ variable named as a key, nor one of another
    prefix); every planted secret in a copied file is left out (each stretch counted): the environment's wherever
    they are, also one that holds another (no part of it stays, in either order), and those only the files hold
    where a secret-named name sets them (NAME=, "name":, --name, a URL's user; also as another name's value,
    args: NAME=...), a short one too. So no copied file, and nothing the record keeps, holds a planted secret or a
    part of one. A setting named for tokens (--max-num-batched-tokens) is kept, and a secret-named variable's short
    value ("1", under 8 characters) is not looked for wherever it is: gpu.csv is copied as it is."""
    from fusion_miniapp.layer_autonmi import LEFT_OUT, LOG_HEAD_LINES, SECRET_NAME, copy_model_service, leave_out

    endpoints = fake_fleet(tmp_path / "fleet", ["http://node1:8100"])
    f = IN_FILES_ONLY
    (tmp_path / "fleet" / "launch.sh").write_text(
        f"export WANDB_API_KEY={f['WANDB_API_KEY']}\n"
        f"vllm serve stand-in --max-num-batched-tokens 8192 --hf-token {f['--hf-token']} --api-key={f['--api-key']}\n"
        f"OTHER_TOKEN={f['OTHER_TOKEN']}\n"
        f"args: GITHUB_TOKEN={f['GITHUB_TOKEN']}\n"
        f"echo {NESTED['OUTER_TOKEN']}\n")
    (tmp_path / "fleet" / "config.json").write_text(f'{{"hf_token": "{f["hf_token"]}", "port": 8100}}\n')
    (tmp_path / "fleet" / "pip-freeze.txt").write_text(
        f"autonmi @ git+https://{f['url']}@github.com/org/autonmi@e20861a\n")
    for name in ("secrets.json", "tokens.txt", "api_keys.txt"):
        (tmp_path / "fleet" / name).write_text("planted-in-a-secret-named-file\n")
    for k in [k for k in os.environ if SECRET_NAME.search(k)]:   # this machine's own: the planted ones alone
        monkeypatch.delenv(k)
    for k in ("VLLM_API_KEY", "HF_TOKEN", "RASCAL_EXTRA_KEY"):
        monkeypatch.setenv(k, PLANTED[k])
    for k, v in NESTED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SHORT_TOKEN", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("RASCAL_MODEL_NAME", f"stand-in {PLANTED['key']}")   # an allowed variable holding the key
    monkeypatch.setenv("UNLISTED_SETTING", "not in the snapshot")
    got = copy_model_service(endpoints, None, tmp_path / "run")
    copied = tmp_path / "run" / "model_service"
    assert got["files"] == sorted(["endpoints", "gpu.csv", "vllm-0.log", "model-123.out", "environment.txt",
                                   "launch.sh", "config.json", "pip-freeze.txt"])
    assert sorted(p.name for p in copied.iterdir()) == got["files"]
    assert got["not_copied"] == ["api-key", "api_keys.txt", "hf_token", "secrets.json", "tokens.txt"]
    assert got["removed_by_the_check"] == []
    # the log's 3, the job output's 1, RASCAL_MODEL_NAME's 1; launch.sh's 6 (the nested pair as one), config.json's
    # 1, pip-freeze.txt's 1
    assert got["secrets_left_out"] == 13
    planted = [*PLANTED.values(), *IN_FILES_ONLY.values(), *NESTED.values(), "planted-in-a-secret-named-file"]
    for p in copied.iterdir():
        data = p.read_bytes()
        assert not any(s.encode() in data for s in planted), p.name
        assert not any(s[:8].encode() in data or s[-5:].encode() in data for s in planted), p.name   # no part
    assert not any(s in json.dumps(got) for s in planted)
    log = (copied / "vllm-0.log").read_text().splitlines()
    assert len(log) == LOG_HEAD_LINES and log[3].count(LEFT_OUT.decode()) == 1 and log[0] == "INFO line 0"
    env = dict(x.split("=", 1) for x in (copied / "environment.txt").read_text().splitlines())
    assert env["SLURM_JOB_ID"] == "123" and env["CUDA_VISIBLE_DEVICES"] == "0,1" and "PATH" in env
    assert env["RASCAL_MODEL_NAME"] == "stand-in " + LEFT_OUT.decode()
    assert not {"VLLM_API_KEY", "HF_TOKEN", "RASCAL_EXTRA_KEY", "UNLISTED_SETTING"} & set(env)
    assert (copied / "endpoints").read_text() == "http://node1:8100\n"
    assert (copied / "gpu.csv").read_bytes() == (tmp_path / "fleet" / "gpu.csv").read_bytes()   # "1" not looked for
    launch = (copied / "launch.sh").read_text()
    assert "--max-num-batched-tokens 8192 --hf-token " + LEFT_OUT.decode() in launch
    assert (copied / "pip-freeze.txt").read_text() == f"autonmi @ git+https://{LEFT_OUT.decode()}@github.com/org/autonmi@e20861a\n"
    # one secret inside another: one stretch left out whichever is taken first (not hash-seed dependent)
    text = f"x {NESTED['OUTER_TOKEN']} y".encode()
    outer, inner = (v.encode() for v in NESTED.values())
    assert leave_out(text, [outer, inner]) == leave_out(text, [inner, outer]) == (b"x " + LEFT_OUT + b" y", 1)


def test_the_setup_list_copies_each_model_jobs_files_and_nothing_else(tmp_path, monkeypatch):
    """Several model nodes (launch/stellar_ai_autonmi.sbatch): the endpoints file combines the model jobs' own in a
    folder of its own, beside their shared key and the list of files to copy (layer_autonmi.SETUP_LIST: one path or
    glob per line, relative to that folder), each job's folder under it (model-<k>/). The copy takes exactly the
    listed files, at their paths under the folder, from every job's folder (logs as their heads, the key left out
    wherever it is); a file the list does not name is not copied, so a bare FAKE secret under a name that says
    nothing, in a file whose name says nothing, never reaches the run, which the same folder without its list would
    copy; a listed path outside the folder, one that is not there, the key files and a symlink are not copied."""
    from fusion_miniapp.layer_autonmi import LEFT_OUT, LOG_HEAD_LINES, SECRET_NAME, SETUP_LIST, copy_model_service

    for k in [k for k in os.environ if SECRET_NAME.search(k)]:   # this machine's own: none here
        monkeypatch.delenv(k)
    bare = "planted-bare-secret-unnamed-file-4x"   # FAKE
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "api-key").write_text(PLANTED["key"] + "\n")
    (fleet / "endpoints").write_text("http://node0:8100\nhttp://node1:8100\n")
    (fleet / "notes.txt").write_text(bare + "\n")
    (tmp_path / "elsewhere.log").write_text("outside the folder\n")
    for k in range(2):
        job = fleet / f"model-{k}"
        job.mkdir()
        (job / "api-key").write_text(PLANTED["key"] + "\n")
        (job / "endpoints").write_text(f"http://node{k}:8100\n")
        (job / "gpu.csv").write_text(f"2026/11/24 00:00:00.000, {k}, 97, 150000\n")
        log = [f"INFO line {i}" for i in range(700)]
        log[2] = f"serving with key {PLANTED['key']}"
        (job / "vllm-0.log").write_text("\n".join(log) + "\n")
        (job / f"model-{50 + k}.out").write_text(f"model job {50 + k}\n")
        (job / "notes.txt").write_text(bare + "\n")
    (fleet / "model-1" / "vllm-1.log").symlink_to(tmp_path / "elsewhere.log")
    (fleet / SETUP_LIST).write_text("\n".join(["endpoints", "# each model job's folder", *[
        f"model-{k}/{name}" for k in range(2) for name in ("endpoints", "gpu.csv", "vllm-*.log", "model-*.out")],
        "model-1/missing.csv", "../elsewhere.log", str(tmp_path / "elsewhere.log"), "model-0/api-key", ""]))
    got = copy_model_service(fleet / "endpoints", str(fleet / "api-key"), tmp_path / "run")
    jobs = [f"model-{k}/{name}" for k in range(2) for name in ("endpoints", "gpu.csv", "vllm-0.log", f"model-{50 + k}.out")]
    assert got["files"] == sorted(["endpoints", "environment.txt", *jobs])
    assert sorted(got["not_copied"]) == sorted(["model-1/vllm-1.log", "model-1/missing.csv", "../elsewhere.log",
                                                str(tmp_path / "elsewhere.log"), "model-0/api-key"])
    assert got["secrets_left_out"] == 2 and got["removed_by_the_check"] == []
    copied = tmp_path / "run" / "model_service"
    assert sorted(f.relative_to(copied).as_posix() for f in copied.rglob("*") if f.is_file()) == got["files"]
    for f in copied.rglob("*"):
        if f.is_file():
            assert PLANTED["key"].encode() not in f.read_bytes() and bare.encode() not in f.read_bytes(), f
    head = (copied / "model-1" / "vllm-0.log").read_text().splitlines()
    assert len(head) == LOG_HEAD_LINES and head[2] == "serving with key " + LEFT_OUT.decode()
    assert (copied / "model-0" / "gpu.csv").read_text().startswith("2026/11/24 00:00:00.000, 0,")
    # the same folder without its list: every plain file of it (not the jobs' folders), the bare secret with them
    (fleet / SETUP_LIST).unlink()
    plain = copy_model_service(fleet / "endpoints", str(fleet / "api-key"), tmp_path / "run2")
    assert plain["files"] == ["endpoints", "environment.txt", "notes.txt"]
    assert bare in (tmp_path / "run2" / "model_service" / "notes.txt").read_text()


def test_the_record_toggle_adds_the_model_calls_extras(tmp_path):
    """The asynchronous mode, --llm openai with the StellarAI settings over two fake vLLM servers (FIXTURES: each
    reply 100 prompt and 50 generated tokens) with a key, record.extras on, from a fake model job's folder with
    planted FAKE secrets (also in the environment): each call's details name the server that answered, the index
    of llm_service.servers (as many calls for each as it received); llm_service.requests holds, per role, exactly
    the settings of the requests the servers received (their bodies besides the messages), counted; each role's
    share of time in a model call over the steady window is what a recount from the lines gives, and the shares
    add up to the campaign's agents in a call; model_service/
    holds the setup files with every secret left out, and no file of the run holds a planted secret. Tokens per
    second over the steady window: generated half of prompt (the fixture's 50 and 100)."""
    from fusion_miniapp.layer_autonmi import SECRET_NAME

    key = PLANTED["key"]
    with FakeVllm(api_key=key) as one, FakeVllm(api_key=key) as two:
        endpoints = fake_fleet(tmp_path / "fleet", [one.url, two.url])
        knobs = {**SERVER_KNOBS, "llm": {**SERVER_KNOBS["llm"], **STELLAR_LLM}, "campaign": {"mode": "async"},
                 "record": {"extras": True}}
        env = {**{k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)},   # the planted ones alone
               **{k: PLANTED[k] for k in ("VLLM_API_KEY", "HF_TOKEN", "RASCAL_EXTRA_KEY")}}
        rec = run("autonmi", tmp_path, *SERVER_ARGS, "--knobs", knob_file(tmp_path, knobs), "--populations", "2",
                  "--endpoints-file", str(endpoints), "--api-key-file", str(tmp_path / "fleet" / "api-key"), env=env)
        from_one, from_two = one.take(), two.take()
        sent = from_one + from_two
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    svc = rec["llm_service"]
    assert svc["servers"] == [one.url, two.url]
    rows = lines(tmp_path, "autonmi")
    details = [c for r in rows for c in r["llm_call_details"]]
    # call details' server i is the server at llm_service.servers[i]: as many calls as it received
    assert len(details) == len(sent) == svc["calls"] and from_one and from_two
    assert Counter(c["server"] for c in details) == {0: len(from_one), 1: len(from_two)}
    # the settings each role's requests carried, as the servers received them, and as the record counts them
    by_role = Counter()
    for b in sent:
        last = b["messages"][-1]["content"].splitlines()[-1]
        role = {decide.NOTE_ASK: "monitor", decide.SCORES_ASK: "evaluator", decide.GUIDANCE_ASK: "coordinator"}.get(
            last, "generator")
        body = {k: v for k, v in b.items() if k != "messages"}
        by_role[json.dumps({"role": role, "body": body}, sort_keys=True)] += 1
    assert {json.dumps({"role": x["role"], "body": x["body"]}, sort_keys=True): x["requests"]
            for x in svc["requests"]} == dict(by_role)
    assert {x["role"] for x in svc["requests"]} == set(ROLES)
    assert all(x["body"]["chat_template_kwargs"] == {"enable_thinking": False} and x["body"]["model"] == "stand-in"
               for x in svc["requests"])
    share = rec["speed"]["model_call_share"]
    assert share["window"] == [0.15, 0.85] and set(share["by_role"]) == set(ROLES)
    assert all(0 < x <= 1 for x in share["by_role"].values()), share
    assert share["agents_in_a_call_per_campaign"] == pytest.approx(
        sum(x * (2 if role == "monitor" else 1) for role, x in share["by_role"].items()), abs=1e-3)
    # recounted from the lines: each role's seconds in a call within the window, over the window and its agents
    lo, hi = min(r["t_start"] for r in rows), max(r["t_end"] for r in rows)
    w_lo, w_hi = lo + 0.15 * (hi - lo), lo + 0.85 * (hi - lo)
    in_call = Counter()
    for c in details:
        in_call[c["role"]] += max(0.0, min(c["t_end"], w_hi) - max(c["t_end"] - c["service_s"], w_lo))
    assert share["by_role"] == pytest.approx(
        {role: in_call[role] / (w_hi - w_lo) / ((2 if role == "monitor" else 1) * 12) for role in ROLES}, abs=2e-4)
    tokens = rec["speed"]["model_tokens_per_s"]
    assert tokens["calls"] > 0 and not tokens["emulated"]
    assert tokens["prompt_per_s"] == pytest.approx(2 * tokens["generated_per_s"], rel=1e-3)
    assert tokens["generated_per_s"] == pytest.approx(50 * tokens["calls"] / tokens["window_s"], rel=1e-3)
    ms = svc["model_service"]
    assert ms["files"] == sorted(["endpoints", "gpu.csv", "vllm-0.log", "model-123.out", "environment.txt"])
    assert ms["not_copied"] == ["api-key", "hf_token"] and ms["removed_by_the_check"] == []
    assert ms["secrets_left_out"] == 4   # the log's 3 (the key, VLLM_API_KEY's, RASCAL_EXTRA_KEY's), the job output's 1
    leaked = [str(f) for f in (tmp_path / "autonmi").rglob("*") if f.is_file()
              and any(s.encode() in f.read_bytes() for s in PLANTED.values())]
    assert leaked == []


def test_the_record_keeps_tokens_per_second_by_default(tmp_path):
    """record.extras off (the default), the generations mode with the stub (emulated) at 1 s a call: the record
    gains the model calls' prompt and generated tokens per second over the steady window (15% to 85% of the agent
    phase), which a recount from the lines' call details (the refills' and summaries') gives, and nothing of the
    toggle's; the results are the reference's."""
    knobs = {**fixture_knobs("llm refills, one random stream, all live"), "llm": {"stub_latency_s": 1.0}}
    rec = run("autonmi", tmp_path, *FIXTURE_ARGS, "--knobs", knob_file(tmp_path, knobs), "--populations", "2")
    assert rec["results_sha256"] == reference("fixture_llm_refills")["results_sha256"]
    rows = lines(tmp_path, "autonmi")
    lo, hi = min(r["t_start"] for r in rows), max(r["t_end"] for r in rows)
    w_lo, w_hi = lo + 0.15 * (hi - lo), lo + 0.85 * (hi - lo)
    inside = [c for r in rows for c in r["llm_call_details"] if w_lo <= c["t_end"] <= w_hi]
    tokens = rec["speed"]["model_tokens_per_s"]
    assert tokens["emulated"] and tokens["calls"] == len(inside) > 0 and tokens["window"] == [0.15, 0.85]
    assert tokens["prompt_per_s"] == pytest.approx(sum(c["prompt_tokens"] for c in inside) / (w_hi - w_lo), rel=1e-3)
    assert tokens["generated_per_s"] == pytest.approx(sum(c["completion_tokens"] for c in inside) / (w_hi - w_lo),
                                                      rel=1e-3)
    assert "model_call_share" not in rec["speed"]
    assert not {"requests", "servers", "model_service"} & set(rec["llm_service"])
    assert not any("server" in c for r in rows for c in r["llm_call_details"])


def test_a_model_run_in_the_asynchronous_mode_needs_llm_thinking(tmp_path):
    """llm.mode openai in the asynchronous mode with llm.thinking unset is refused before anything starts (unset,
    the server's chat template decides whether the model thinks)."""
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", ENTRY["autonmi"], *SERVER_ARGS, "--knobs",
                           knob_file(tmp_path, {**SERVER_KNOBS, "campaign": {"mode": "async"}}), "--endpoints-file",
                           str(tmp_path / "endpoints"), "--out", str(out)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=300, check=False)
    assert proc.returncode != 0 and "needs llm.thinking set" in proc.stderr and not out.exists(), proc.stderr[-2000:]


def test_each_role_makes_its_own_model_call_in_the_asynchronous_mode(async_calls_run):
    """The stub LLM (FIXTURE): the Generator asks once per take of 2, each Monitor once per candidate, the
    Evaluator and the Coordinator once per cycle, each answered (no try again, no fallback), and AUTONMI's checks
    of the mode pass. Every candidate went through the Generator's call, which filled all 7 parameters (the stub
    draws a full candidate); the Coordinator sends its guidance after each cycle's survivors; a line's model
    calls are its agents'. With record.extras: each candidate's note ("" from the stub), each cycle's guidance
    (""), and every score the score from numbers (the stub's answer), q + 0.25 m clipped."""
    out, rec = async_calls_run
    n, B = 20, 16
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    checks = {c["name"]: c["ok"] for c in json.loads((out / "app" / "checks.json").read_text())}
    assert checks == dict.fromkeys(ASYNC_CHECKS, True), checks
    c = rec["counts"]
    by, a = c["by_role"], c["asynchronous"]
    for role in ROLES:
        assert by[role]["model_asks"] == by[role]["model_tries"] == by[role]["model_calls"] == a["cycles"][role], role
        assert by[role]["model_fallbacks"] == 0
    assert (a["cycles"]["generator"], a["cycles"]["monitor"]) == (n * B // 2, n * B)
    assert c["llm_calls"] == sum(by[role]["model_calls"] for role in ROLES) == rec["llm_service"]["calls"]
    assert c["candidates_by_source"] == {"start": 0, "survivor": 0, "mutation": 0, "model": n * B, "restart": 0}
    assert c["refill_slots"] == {"asked": n * B, "filled_by_model": n * B} and a["parameters_filled"] == [0] * 7 + [n * B]
    assert rec["llm_check"]["fail"] == [] and c["expected_per_agent"]["llm_calls"] == 8 + 3 * B
    rows = app_rows(out)
    for line in lines_of(out):
        x = {r["role"]: r for r in rows if r["campaign"] == line["agent_id"] and r["role"] != "monitor"}
        co, ge, ev = x["coordinator"], x["generator"], x["evaluator"]
        assert co["messages_sent"] == 2 * co["cycles"] + 2 and ge["messages_received"] == 2 * co["cycles"] + ev["cycles"] + 1
        mine = sum(r["model_calls"] for r in rows if r["campaign"] == line["agent_id"])
        assert line["n_llm_calls"] == mine == len(line["llm_call_details"]) <= 8 + 3 * B
        assert line["asynchronous"]["guidance"] == [""] * co["cycles"]
        assert [e["note"] for e in line["monitor"]] == [""] * B
        m = {e["candidate_idx"]: e["m"] for e in line["monitor"]}
        assert all(r["score"] == min(1.0, max(0.0, r["q"] + 0.25 * m[r["candidate"]]))
                   for r in line["asynchronous"]["results"])


def test_a_recorded_order_with_the_stubs_calls_replays_to_the_same_campaign(async_calls_run):
    """The stub's replies replayed: each take's profiles reconciled with the stub's reply for the campaign and the
    request's last line (async_replay.stub_reconcile), the scores the score from numbers (which the stub
    answers): every result's q, m, score, draw and kept flag, the survivors and best-so-far are the run's, in its
    reordered orders. So a value in the Generator's reply replaces the evolutionary loop's, and the stub's three
    other answers change nothing."""
    from async_replay import fixture, replay, stub_reconcile

    out, rec = async_calls_run
    knobs = rec["knobs"]
    for line in lines_of(out):
        x, seed = line["asynchronous"], line["seed"]
        m = {r["candidate"]: r["m"] for r in x["results"]}
        cst, rows, calls = replay(seed, knobs, x["evaluator"], x["coordinator"], x["held"], m, fixture(knobs, seed),
                                  stub_reconcile(knobs))
        assert rows == x["results"], line["agent_id"]
        assert [s["candidate"] for s in cst["survivors"]] == x["survivors"], line["agent_id"]
        assert json.loads(json.dumps(cst["best"])) == line["best"], line["agent_id"]
        assert all(c["source"] == "model" and c["filled"] == 7 for c in calls.values())
    assert sum(sum(line["asynchronous"]["evaluator"], []) != list(range(16)) for line in lines_of(out)) > 0


SHOWN_ONE = """import sys
sys.path.insert(0, sys.argv.pop(1))
from async_replay import reordered
from fusion_miniapp import decide
decide.SHOWN_RESULTS = 1   # (tests only) one result per Evaluator call
sys.exit(reordered(sys.argv[1:]))
"""


def test_an_evaluator_call_shows_at_most_its_limit_of_results(tmp_path):
    """The asynchronous mode with every role's call (the stub, FIXTURE) and decide.SHOWN_RESULTS set to 1 in the run's
    own process (async_run's reordered arrivals, so reports reach the Evaluator while it scores others): each
    Evaluator cycle takes one report, the others wait in its inbox for its next cycle in arrival order; so each
    campaign's Evaluator has as many cycles and calls as results; AUTONMI's checks of the mode pass; and the
    recorded orders replay to the same campaign. With llm.mode off (no call) its cycle takes every report, as
    before (async_run)."""
    from async_replay import fixture, replay, stub_reconcile

    script = tmp_path / "shown_one.py"
    script.write_text(SHOWN_ONE)
    proc = subprocess.run([sys.executable, str(script), str(Path(__file__).parent), *ASYNC_ARGS, "--knobs",
                           knob_file(tmp_path, {**ASYNC_CALLS, "record": {"extras": True}}),
                           "--out", str(tmp_path / "autonmi")],
                          cwd=tmp_path, check=False, timeout=900, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = tmp_path / "autonmi"
    rec = json.loads((out / "run_record.json").read_text())
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    checks = {c["name"]: c["ok"] for c in json.loads((out / "app" / "checks.json").read_text())}
    assert checks == dict.fromkeys(ASYNC_CHECKS, True), checks
    by = rec["counts"]["by_role"]
    assert by["evaluator"]["model_asks"] == rec["counts"]["asynchronous"]["cycles"]["evaluator"] == 20 * 16
    knobs = rec["knobs"]
    for line in lines_of(out):
        x, seed = line["asynchronous"], line["seed"]
        assert [len(b) for b in x["evaluator"]] == [1] * 16, line["agent_id"]
        m = {r["candidate"]: r["m"] for r in x["results"]}
        _, rows, _ = replay(seed, knobs, x["evaluator"], x["coordinator"], x["held"], m, fixture(knobs, seed),
                            stub_reconcile(knobs))
        assert rows == x["results"], line["agent_id"]


def test_a_long_note_or_guidance_is_cut_in_later_prompts_and_kept_whole_in_the_record(tmp_path):
    """The asynchronous mode with every role's call through a fake vLLM server (FIXTURE) whose Monitor notes are
    3,000 characters and Coordinator guidance 2,500, with record.extras: every request shows at most
    decide.NOTE_CHARS of either (the Evaluator's prompts show notes cut to it, the Generator's guidance cut to
    it), the record keeps each whole (agents.jsonl: monitor[i].note and asynchronous.guidance), and the run
    passes."""
    note, guidance = "z" * 3000, "g" * 2500

    def reply(content):
        last = content.splitlines()[-1]
        return note if last == decide.NOTE_ASK else guidance if last == decide.GUIDANCE_ASK else None

    knobs = {"llm": {"mode": "openai", "model": "stand-in", "thinking": "off", "reply_cap": 4096},
             "outer": {"population": 4, "generations": 3, "parent_stdout": "drop"}, "campaign": {"mode": "async"},
             "record": {"extras": True}}
    with FakeVllm(reply=reply) as server:
        (tmp_path / "endpoints").write_text(server.url + "\n")
        rec = run("autonmi", tmp_path, "--preset", "smoke", "--sim", "fixture", "--agents", "6", "--knobs",
                  knob_file(tmp_path, knobs), "--endpoints-file", str(tmp_path / "endpoints"))
        prompts = ["\n".join(m["content"] for m in body["messages"]) for body in server.take()]
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    cut = decide.NOTE_CHARS
    for text in (note, guidance):
        assert any(text[:cut] in p for p in prompts) and not any(text[:cut + 1] in p for p in prompts), text[0]
    lines = lines_of(tmp_path / "autonmi")
    notes = [m["note"] for line in lines for m in line["monitor"]]
    assert notes and set(notes) == {note}
    given = [g for line in lines for g in line["asynchronous"]["guidance"]]
    assert given and set(given) == {guidance}


def test_a_server_that_fails_every_call_leaves_each_role_on_its_fallback(tmp_path):
    """A model server that answers every request with an error (the fake vLLM server, FIXTURE: HTTP 400): each
    role's call is tried 3 times, then its fallback is used and counted (the Generator's profiles unchanged, no
    note, the score from numbers, no guidance); every campaign completes and AUTONMI's checks pass; and replayed
    with no model call (llm.mode off's path) in the run's own reordered orders, every result, the survivors and
    best-so-far are the run's. The run fails only its model check: the model filled none of the candidates."""
    from async_replay import fixture, replay

    knobs = {**ASYNC_CALLS, "llm": {"mode": "openai", "model": "stand-in", "thinking": "off"}, "record": {"extras": True}}
    with FakeVllm(lambda body: True) as server:
        (tmp_path / "endpoints").write_text(server.url + "\n")
        out, rec = reordered_run(tmp_path, knobs, "--endpoints-file", str(tmp_path / "endpoints"), code=1)
        sent = server.take()
    assert rec["autonmi"]["ok"] and (rec["counts"]["agents"], rec["counts"]["agents_failed"]) == (20, 0), rec["autonmi"]
    assert len(rec["llm_check"]["fail"]) == 1 and "filled 0 of the 320" in rec["llm_check"]["fail"][0]
    by, cycles = rec["counts"]["by_role"], rec["counts"]["asynchronous"]["cycles"]
    for role in ROLES:
        assert by[role]["model_asks"] == by[role]["model_fallbacks"] == cycles[role] > 0, role
        assert (by[role]["model_tries"], by[role]["model_calls"]) == (3 * by[role]["model_asks"], 0), role
    assert len(sent) == sum(by[role]["model_tries"] for role in ROLES) == rec["llm_service"]["errors"]
    assert rec["counts"]["candidates_by_source"]["model"] == 0 and rec["counts"]["refill_slots"]["asked"] == 320
    for line in lines_of(out):
        x, seed = line["asynchronous"], line["seed"]
        assert x["guidance"] == [""] * len(x["coordinator"])
        m = {r["candidate"]: r["m"] for r in x["results"]}
        cst, rows, _ = replay(seed, rec["knobs"], x["evaluator"], x["coordinator"], x["held"], m,
                              fixture(rec["knobs"], seed))
        assert rows == x["results"], line["agent_id"]
        assert [s["candidate"] for s in cst["survivors"]] == x["survivors"], line["agent_id"]
        assert json.loads(json.dumps(cst["best"])) == line["best"], line["agent_id"]


def test_a_prompt_too_long_for_the_servers_context_is_not_sent(tmp_path):
    """The prompt-length check: the servers' /v1/models give max_model_len (the fake server's, FIXTURE, set to
    2,000 here); a call whose prompt's characters (at least its tokens) and max_tokens exceed it is not sent and
    not tried again: the role falls back at once, counted, with the error PromptTooLong in its details (in the
    line with record.extras, and in the record's count of errors). The others are sent; every campaign
    completes."""
    knobs = {**ASYNC_CALLS, "llm": {"mode": "openai", "model": "stand-in", "thinking": "off"}, "record": {"extras": True}}
    with FakeVllm(max_model_len=2000) as server:
        rec, sent = run_on_a_server(tmp_path, knobs, server, code=1)
    assert rec["autonmi"]["ok"] and rec["counts"]["agents_failed"] == 0, rec["autonmi"]
    details = [c for r in lines(tmp_path, "autonmi") for c in r["llm_call_details"]]
    too_long = [c for c in details if c.get("error") == "PromptTooLong"]
    by = rec["counts"]["by_role"]
    assert too_long and sent and len(sent) + len(too_long) == len(details) == sum(by[r]["model_tries"] for r in ROLES)
    assert all(c["latency_s"] < 0.01 for c in too_long)
    assert sum(by[r]["model_fallbacks"] for r in ROLES) == len(too_long)   # each once, at once
    assert all(sum(len(str(m["content"])) for m in body["messages"]) + body["max_tokens"] <= 2000 for body in sent)
    assert rec["llm_service"]["errors"] == len(too_long)


def each_roles_reply(content: str) -> str | None:
    """FIXTURE replies that differ from the stub's numbers-only ones, each marked so a test can find it where it
    goes: the Evaluator gives every candidate keep-probability 1.0, a Monitor's note on candidate k is NOTEMARK-k,
    and the Coordinator's guidance after n results is GUIDEMARK-n. The Generator's reply is the fixture's."""
    lines_ = content.splitlines()
    if lines_[-1] == decide.SCORES_ASK:
        return json.dumps({k: 1.0 for k in re.findall(r"^candidate (\d+): q ", content, re.M)})
    if lines_[-1] == decide.NOTE_ASK:
        return "NOTEMARK-" + re.match(r"Candidate (\d+):", lines_[0]).group(1)
    if lines_[-1] == decide.GUIDANCE_ASK:
        return "GUIDEMARK-" + re.search(r"(\d+) of \d+ results in", content).group(1)
    return None


def test_each_roles_reply_reaches_where_it_goes(tmp_path):
    """The asynchronous mode against the fake vLLM server (FIXTURE) whose replies differ from the numbers-only
    values (each_roles_reply): the Evaluator's reply is the score, so every result scores 1.0 and is kept; each
    Monitor's note reaches the Evaluator's prompt on its own candidate's row; the Evaluator's latest scores (its
    digest) and the Coordinator's guidance reach the Generator's later prompts; with record.extras each note and
    guidance is in the campaign's line."""
    knobs = {**SERVER_KNOBS, "llm": {**SERVER_KNOBS["llm"], "thinking": "off"}, "campaign": {"mode": "async"},
             "record": {"extras": True}}
    with FakeVllm(reply=each_roles_reply) as server:
        rec, sent = run_on_a_server(tmp_path, knobs, server)
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    results = [x for r in lines(tmp_path, "autonmi") for x in r["asynchronous"]["results"]]
    assert len(results) == 12 * 12 and all(x["score"] == 1.0 and x["kept"] for x in results)
    content = [body["messages"][-1]["content"] for body in sent]
    generator = [c for c, body in zip(content, sent) if "response_format" in body]
    evaluator = [c for c in content if c.splitlines()[-1] == decide.SCORES_ASK]
    last = [c for c in generator if c.splitlines()[-1].startswith("Candidates 10 and 11 of 12:")]   # each campaign's
    assert len(last) == 12 and all("The Coordinator's guidance: GUIDEMARK-" in c for c in last)
    assert all(re.search(r"keep-probabilities \(candidate: probability\): (\d+: 1\.0(, |$))+$", c, re.M) for c in last)
    rows = [m.groups() for c in evaluator for m in re.finditer(r"^candidate (\d+): q .*note: (.*)$", c, re.M)]
    assert len(rows) == len(results) and all(note == f"NOTEMARK-{k}" for k, note in rows)
    for r in lines(tmp_path, "autonmi"):
        assert [e["note"] for e in r["monitor"]] == [f"NOTEMARK-{e['candidate_idx']}" for e in r["monitor"]]
        assert r["asynchronous"]["guidance"] and all(g.startswith("GUIDEMARK-") for g in r["asynchronous"]["guidance"])


def test_a_reply_that_cannot_be_used_is_tried_again_and_a_late_answer_is_used(tmp_path):
    """Tries at the layer, against the fake vLLM server (FIXTURE), asynchronous mode. Replies the Generator and the
    Evaluator cannot use (plain text): each of their calls is answered 3 times, then falls back, counted (the
    Monitors' and the Coordinator's calls are answered at once). Every request failing twice and answered the third
    time: each call is tried 3 times and answered, no fallback, and the run passes."""
    knobs = {**SERVER_KNOBS, "llm": {**SERVER_KNOBS["llm"], "thinking": "off"}, "campaign": {"mode": "async"}}

    def unusable(content: str) -> str | None:
        last = content.splitlines()[-1]
        return "No JSON today." if last == decide.SCORES_ASK or "JSON list of" in last else None
    (tmp_path / "unusable").mkdir()
    with FakeVllm(reply=unusable) as server:
        rec, sent = run_on_a_server(tmp_path / "unusable", knobs, server, code=1)   # the model filled nothing
    assert rec["autonmi"]["ok"] and rec["counts"]["agents_failed"] == 0, rec["autonmi"]
    by = rec["counts"]["by_role"]
    for role in ("generator", "evaluator"):
        b = by[role]
        assert b["model_tries"] == b["model_calls"] == 3 * b["model_asks"] and b["model_fallbacks"] == b["model_asks"] > 0
    for role in ("coordinator", "monitor"):
        b = by[role]
        assert b["model_tries"] == b["model_calls"] == b["model_asks"] > 0 and b["model_fallbacks"] == 0
    assert len(sent) == sum(by[role]["model_tries"] for role in ROLES)

    seen = Counter()

    def twice(body) -> bool:   # each request fails on its first two sends
        key = json.dumps(body["messages"], sort_keys=True)
        seen[key] += 1
        return seen[key] <= 2
    (tmp_path / "third").mkdir()
    with FakeVllm(twice) as server:
        rec, sent = run_on_a_server(tmp_path / "third", knobs, server)
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    by = rec["counts"]["by_role"]
    for role in ROLES:
        b = by[role]
        assert b["model_tries"] == 3 * b["model_asks"] == 3 * b["model_calls"] > 0 and b["model_fallbacks"] == 0, role
    assert rec["llm_service"]["errors"] == 2 * sum(by[role]["model_asks"] for role in ROLES)


def test_the_notes_falling_back_change_no_result_in_the_generations_mode(tmp_path):
    """campaign.notes against the fake vLLM server (FIXTURE) that fails every note and assessment call (HTTP 400):
    each is tried 3 times and falls back, counted; the refills and summaries are answered; the results and the
    exact counts are the model_server reference's, recorded without the notes."""
    def notes(body) -> bool:
        return body["messages"][-1]["content"].splitlines()[-1] in (decide.NOTE_ASK, decide.SCORES_ASK)
    with FakeVllm(notes) as server:
        rec, _ = run_on_a_server(tmp_path, {**SERVER_KNOBS, "campaign": {"notes": True}}, server)
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    assert against_reference(tmp_path / "autonmi", rec, reference("model_server")) == []
    by = rec["counts"]["by_role"]
    assert (by["monitor"]["model_asks"], by["evaluator"]["model_asks"]) == (12 * 4 * 3, 12 * 3)
    for role in ("monitor", "evaluator"):
        b = by[role]
        assert (b["model_tries"], b["model_calls"], b["model_fallbacks"]) == (3 * b["model_asks"], 0, b["model_asks"])
    for role in ("coordinator", "generator"):
        assert by[role]["model_fallbacks"] == 0 and by[role]["model_calls"] == by[role]["model_asks"] > 0


def test_llm_outage_s_and_thinking_off_reach_the_wait_for_the_model_servers(tmp_path, monkeypatch):
    """llm.outage_s (how long a call waits out a server that went away) and llm.thinking off (AUTONMI's
    THINKING_OFF, which its transports send on every call) are what the layer passes to AUTONMI's
    wait_for_servers."""
    from autonmi import models

    from fusion_miniapp import layer_autonmi

    seen = {}

    class Stop(Exception):
        pass

    async def wait_for_servers(*args, **kwargs):
        seen.update(kwargs)
        raise Stop
    monkeypatch.setattr(models, "wait_for_servers", wait_for_servers)
    a = layer_autonmi.parse(["--llm", "openai", "--endpoints-file", str(tmp_path / "endpoints")])
    knobs = knob_files.validate({"llm": {"mode": "openai", "model": "stand-in", "thinking": "off", "outage_s": 300}},
                                modes=("generations", "async"))
    with pytest.raises(Stop):
        asyncio.run(layer_autonmi.miniapp_workflow(a, knobs, None, tmp_path, None, None, layer_autonmi.LiveLine()))
    assert seen["outage_s"] == 300 and seen["thinking_off"] == models.THINKING_OFF and seen["name"] == "stand-in"


def test_the_notes_change_no_result_in_the_generations_mode(tmp_path):
    """campaign.notes (off by default): in the generations mode each Monitor asks for a note on its candidate and
    the Evaluator for an assessment of each generation (the stub, FIXTURE: no note, the score from numbers); the
    results and parent files are the reference's; the calls are counted by role (a note per candidate, an
    assessment per generation); with record.extras each candidate's entry has its note and assessment."""
    knobs = {**fixture_knobs("llm refills, one random stream, all live"), "campaign": {"notes": True},
             "record": {"extras": True}}
    rec = run("autonmi", tmp_path, *FIXTURE_ARGS, "--populations", "2", "--knobs", knob_file(tmp_path, knobs))
    assert rec["status"] == "pass" and rec["autonmi"]["ok"], rec["autonmi"]
    assert against_reference(tmp_path / "autonmi", rec, reference("fixture_llm_refills")) == []
    by = rec["counts"]["by_role"]
    assert {role: (by[role]["model_asks"], by[role]["model_calls"], by[role]["model_fallbacks"]) for role in ROLES} == {
        "coordinator": (20, 20, 0), "generator": (20, 20, 0), "evaluator": (40, 40, 0), "monitor": (160, 160, 0)}
    assert rec["counts"]["llm_calls"] == 40   # the outer loop's, as without the notes
    for r in lines(tmp_path, "autonmi"):
        assert all(e["note"] == "" and e["assessed"] == pytest.approx(min(1.0, max(0.0, e["q"] + 0.25 * e["m"])),
                                                                       abs=1e-6) for e in r["monitor"])


def test_verify_catches_broken_model_call_counts(async_calls_run):
    """AUTONMI's check of the roles' calls on the stub run's own rows: each break of an agent's counts (a
    Monitor that did not ask for its note, a Generator take without its call, more replies than tries, a
    fallback beyond the calls asked) fails evaluations_and_model_calls_as_planned alone."""
    from rascal.store import MemoryStore

    from fusion_miniapp.layer_autonmi import Fusion

    out, rec = async_calls_run
    rows, store = app_rows(out), MemoryStore()
    for c in {r["campaign"] for r in rows}:
        store.put(f"campaign/{c}", "{}")
    definition = Fusion(rec["knobs"], "fixture", out)

    def failing(agent: str, edit) -> set:
        broken = copy.deepcopy(rows)
        for r in broken:
            if r["agent_id"] == agent:
                r.update(edit(r))
        return {c.name for c in definition.verify(broken, store) if not c.ok}
    assert failing("a5/coordinator", lambda r: {}) == set()
    planned = {"evaluations_and_model_calls_as_planned"}
    one_less = lambda r: {k: r[k] - 1 for k in ("model_asks", "model_tries", "model_calls")}  # noqa: E731
    assert failing("a5/monitor0", one_less) == planned
    assert failing("a5/generator", one_less) == planned
    assert failing("a5/evaluator", lambda r: {"model_calls": r["model_tries"] + 1}) == planned
    assert failing("a5/coordinator", lambda r: {"model_fallbacks": r["model_asks"] + 1}) == planned
    assert failing("a5/coordinator", lambda r: {"model_fallbacks": 1, "model_tries": r["model_tries"] + 2}) == set()


def test_the_asynchronous_mode_labels_each_trace_as_its_result_is_scored():
    """_label_async: as each result is scored, its trace (held by its Monitor) joins the good side when q is 0.5 or
    more, else the bad side (label_from q); a synthetic trace is labelled instead from its own last step among the
    window's synthetic traces, it last (trace_end_synthetic); a result whose Monitor held no trace has no label.
    The window's last steps keep the last outer.population results'. With record.extras each candidate's q, label
    and label_from are kept, in the order scored."""
    import numpy as np

    from fusion_miniapp import monitor
    from fusion_miniapp.layer_autonmi import Script

    sc = Script.__new__(Script)   # only what _label_async reads
    sc.P, sc.extras, sc.numbers, sc.calls_on = 4, True, {}, False
    stats = monitor.Rolling(warmup=1, window=4)
    xs = {k: np.full((3, 4), float(k)) for k in range(4)}
    for k, x in xs.items():   # candidates 2 and 3 synthetic (the wall at the last step: 2 below 3)
        stats.hold(0, k, x, synthetic=k >= 2)
    for k, q in ((1, 0.5), (0, 0.25), (2, 0.0), (3, 1.0), (4, 0.9)):   # in arrival order; 4 holds no trace
        sc._label_async("a0", stats, {"numbers": None}, {"candidate": k, "q": q})
    assert [(x["candidate_idx"], x["q"], x["label"], x["label_from"]) for x in sc.numbers["a0"]] == [
        (1, 0.5, "good", "q"), (0, 0.25, "bad", "q"), (2, 0.0, "good", "trace_end_synthetic"),
        (3, 1.0, "bad", "trace_end_synthetic"), (4, 0.9, None, None)]
    assert (stats.good.traces, stats.bad.traces) == (2, 2) and stats.held == {}
    assert np.allclose(stats.good.mean, 1.5) and np.allclose(stats.bad.mean, 1.5)
    assert [None if e is None else float(e[-1, 1]) for e in stats.ends] == [None, 2.0, 3.0, None]
