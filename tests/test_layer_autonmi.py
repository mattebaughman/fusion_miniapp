"""The AUTONMI layer (layer_autonmi.py) against the LangGraph layer: the same campaigns, the same run record.

Each run is a process of its own, as in production (OpenTelemetry's providers are set once per process),
started in the test's own folder (AsyncFlow leaves a folder for every engine in the working directory).
Two runs agree when compare_records says so, as for two runs on two machines. The model server is a
FIXTURE (not a model): an OpenAI-compatible stand-in on 127.0.0.1, on a thread of the test.

Each AUTONMI run is also compared with a reference recorded from the LangGraph layer
(tests/references/<name>.json, written by tests/references/record_references.py, which runs these
cases): its results_sha256, exact counts, parent files and model requests. The live LangGraph run is
compared with the same reference, so a change to either layer shows.

Needs the AUTONMI framework (the package `autonmi`, installed separately) and Flowgentic 0.6
(requirements-autonmi.txt); skipped without them.
"""
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

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
    from fusion_miniapp.layer_autonmi import AUTONMI_COMMIT

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
    assert 0 < a["counts"]["suspensions"] <= 20 * 2     # at most one per generation: the fan-out's barrier
    if knobs["outer"].get("logs", "per_agent") == "per_agent":
        for f in ("config.json", "evolution_log.csv", "best_result.json"):   # the parent's per-campaign files
            for c in ("a0", "a19"):
                assert (tmp_path / "autonmi" / "campaigns" / c / f).read_bytes() == \
                    (tmp_path / "langgraph" / "campaigns" / c / f).read_bytes(), (c, f)
    assert not list(tmp_path.glob("autonmi/agents-p*.jsonl"))   # merged into agents.jsonl, then removed


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


@pytest.mark.skipif(bool(_missing("torch")), reason="needs torch")
def test_real_evaluate_through_the_autonmi_layer_on_cpu(tmp_path):
    common = (*REAL_ARGS, "--knobs", knob_file(tmp_path, REAL_KNOBS))
    a = run("autonmi", tmp_path, *common)
    b = run("langgraph", tmp_path, *common)
    assert a["touch_check"] == {"evaluations": 12, "fixture": 0, "pass": 12, "fail": {}}
    assert a["device_probe"]["torch"]                     # taken in a spawned process, where simulations run
    assert (a["sim_pool"], a["sim_workers_total"]) == ("process", 2)
    assert agree(tmp_path, a, b) == []
    ref = reference("real_cpu")
    for layer, rec in (("autonmi", a), ("langgraph", b)):
        assert against_reference(tmp_path / layer, rec, ref) == [], layer


class ModelServer:
    """A FIXTURE, not a model: an OpenAI-compatible server on 127.0.0.1 that answers /health, and each
    chat completion from the request's last line only (as the stub LLM does), so both layers get the same
    replies. It keeps every request's body; `fail(body)` true answers that request with HTTP 400."""

    def __init__(self, fail=None) -> None:
        self.fail, self.bodies = fail or (lambda body: False), []
        self._loop = asyncio.new_event_loop()

    async def _chat(self, request):
        from aiohttp import web
        body = await request.json()
        self.bodies.append(body)
        if self.fail(body):
            return web.json_response({"error": {"message": "rejected by the test"}}, status=400)
        last = body["messages"][-1]["content"].splitlines()[-1]
        many = re.search(r"JSON list of (\d+)", last)
        if many:
            rng = random.Random(last)
            text = json.dumps([{k: round(rng.uniform(lo, hi), 6) for k, (lo, hi) in decide.ranges().items()}
                               for _ in range(int(many.group(1)))])
        else:
            text = "Summary: " + body["messages"][-1]["content"].splitlines()[0]
        return web.json_response({"id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": text}}],
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})

    def __enter__(self) -> "ModelServer":
        from aiohttp import web

        async def start():
            app = web.Application()
            app.router.add_post("/v1/chat/completions", self._chat)
            app.router.add_get("/health", lambda request: web.Response(text="ok"))
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            return runner, site._server.sockets[0].getsockname()[1]

        self._runner, self.port = self._loop.run_until_complete(start())
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.run_until_complete(self._runner.cleanup())
        self._loop.close()

    def take(self) -> list[dict]:
        """The requests so far, and forget them."""
        out, self.bodies = self.bodies, []
        return out


def run_both_on_a_model_server(tmp_path, server: ModelServer, code: int = 0):
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
    with ModelServer() as server:
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


def test_a_campaign_whose_model_call_fails_is_an_error_line_in_both_layers(tmp_path):
    """The server rejects some campaigns' summary calls (HTTP 400, which neither layer retries): those
    campaigns fail, each is a line with its error and its failed call, and both runs fail alike."""
    def fail(body):
        text = body["messages"][-1]["content"]
        return text.endswith("found.") and int(hashlib.sha256(text.encode()).hexdigest(), 16) % 3 == 0
    with ModelServer(fail) as server:
        a, b, _, _ = run_both_on_a_model_server(tmp_path, server, code=1)
    failed = {layer: sorted(r["agent_id"] for r in lines(tmp_path, layer) if "error" in r) for layer in ENTRY}
    assert failed["autonmi"] and failed["autonmi"] == failed["langgraph"]
    assert a["status"] == b["status"] == "fail"
    assert a["counts"]["agents"] == 12 and a["counts"]["agents_failed"] == len(failed["autonmi"])
    for r in lines(tmp_path, "autonmi"):
        if "error" in r:
            assert r["llm_call_details"][-1]["error"] and r["llm_call_details"][-1]["empty"] is None
    assert not a["autonmi"]["ok"]   # every_campaign_reports


def test_sigterm_writes_an_interrupted_record_with_the_campaigns_that_finished(tmp_path):
    """SIGTERM (scancel, a time limit) while campaigns run, as test_lifecycle.py sends it to run.py: the run
    is cancelled through its event loop, its record says interrupted and keeps the campaigns that had
    finished, and it exits 143. 50 live campaigns per population of 16 candidates keep hundreds of
    evaluations queued when the first campaign finishes: the signal cuts them off, and the record gives
    AUTONMI's telemetry verdict (complete: the populations are waited for) with every population's task
    counts, those abandoned included, and AUTONMI's check every_population_ran failed."""
    knobs = {"tools": {"latency_ms": 500}, "agents": {"max_live": 50},
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
    assert all(r["n_sims"] == exp["sims"] and len(r["generations"]) == 2 for r in rows)   # each one whole
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
