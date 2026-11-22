"""The AUTONMI layer (layer_autonmi.py) against the LangGraph layer: the same campaigns, the same run record.

Each run is a process of its own, as in production (OpenTelemetry's providers are set once per process),
started in the test's own folder (AsyncFlow leaves a folder for every engine in the working directory).
Two runs agree when compare_records says so, as for two runs on two machines. The model server is a
FIXTURE (not a model): an OpenAI-compatible stand-in on 127.0.0.1, on a thread of the test.

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

from fusion_miniapp import compare_records, decide, knobs as knob_files


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


@pytest.mark.parametrize("case", CASES)
def test_the_autonmi_layer_gives_the_langgraph_layers_results(tmp_path, case):
    """20 fixture campaigns in 2 populations: the same campaigns (results_sha256), counts and knobs as the
    LangGraph layer's run, and the same parent files for each campaign."""
    over = CASES[case]
    knobs = {"outer": {"population": 4, "generations": 2, "parent_stdout": "drop", **over.get("outer", {})},
             "agents": over.get("agents", {})}
    common = ("--preset", "smoke", "--knobs", knob_file(tmp_path, knobs), "--sim", "fixture", "--agents", "20")
    a = run("autonmi", tmp_path, *common, "--populations", "2")
    b = run("langgraph", tmp_path, *common)
    assert (a["agent_layer"], a["status"], a["counts"]["agents"], a["counts"]["sims"]) == ("autonmi", "pass", 20, 160)
    assert agree(tmp_path, a, b) == []
    assert a["autonmi"]["ok"] and a["autonmi"]["populations"] == 2, a["autonmi"]
    assert 0 < a["counts"]["suspensions"] <= 20 * 2     # at most one per generation: the fan-out's barrier
    if knobs["outer"].get("logs", "per_agent") == "per_agent":
        for f in ("config.json", "evolution_log.csv", "best_result.json"):   # the parent's per-campaign files
            for c in ("a0", "a19"):
                assert (tmp_path / "autonmi" / "campaigns" / c / f).read_bytes() == \
                    (tmp_path / "langgraph" / "campaigns" / c / f).read_bytes(), (c, f)
    assert not list(tmp_path.glob("autonmi/agents-p*.jsonl"))   # merged into agents.jsonl, then removed


@pytest.mark.skipif(bool(_missing("torch")), reason="needs torch")
def test_real_evaluate_through_the_autonmi_layer_on_cpu(tmp_path):
    knobs = {"sim": {"device": "cpu"}, "outer": {"population": 3, "generations": 2, "parent_stdout": "drop"},
             "workers": {"sim_workers": 2}}
    common = ("--preset", "smoke", "--knobs", knob_file(tmp_path, knobs), "--sim", "real", "--agents", "2")
    a = run("autonmi", tmp_path, *common)
    b = run("langgraph", tmp_path, *common)
    assert a["touch_check"] == {"evaluations": 12, "fixture": 0, "pass": 12, "fail": {}}
    assert a["device_probe"]["torch"]                     # taken in a spawned process, where simulations run
    assert (a["sim_pool"], a["sim_workers_total"]) == ("process", 2)
    assert agree(tmp_path, a, b) == []


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
    knobs = {"llm": {"mode": "openai", "model": "stand-in"},
             "outer": {"population": 4, "generations": 3, "parent_stdout": "drop"}}
    common = ("--preset", "smoke", "--knobs", knob_file(tmp_path, knobs), "--sim", "fixture", "--agents", "12")
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
    server's default applies to both), the same replies, the same results."""
    with ModelServer() as server:
        a, b, sent_a, sent_b = run_both_on_a_model_server(tmp_path, server)
    assert agree(tmp_path, a, b) == []
    assert a["llm_check"]["fail"] == [] and a["llm_check"]["model_share"] == 1.0
    assert all(body.get("temperature") is None for body in sent_a)

    def requests(sent):
        return sorted(json.dumps({k: v for k, v in body.items() if v is not None}, sort_keys=True) for body in sent)
    assert len(sent_a) == a["counts"]["llm_calls"] > 12 and requests(sent_a) == requests(sent_b)
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
    finished, and it exits 143."""
    knobs = {"tools": {"latency_ms": 500}, "agents": {"max_live": 10},
             "outer": {"population": 4, "generations": 2, "parent_stdout": "drop", "logs": "none"}}
    out = tmp_path / "autonmi"
    with open(tmp_path / "stderr.txt", "w") as err:
        p = subprocess.Popen([sys.executable, "-m", ENTRY["autonmi"], "--preset", "smoke", "--sim", "fixture",
                              "--agents", "400", "--populations", "2", "--knobs", knob_file(tmp_path, knobs),
                              "--out", str(out)], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=err)
        deadline = time.time() + 120
        while not any(f.stat().st_size for f in out.glob("agents-p*.jsonl")):   # a campaign has finished
            assert p.poll() is None and time.time() < deadline, (tmp_path / "stderr.txt").read_text()[-3000:]
            time.sleep(0.2)
        t = time.time()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=90) == 143, (tmp_path / "stderr.txt").read_text()[-3000:]
    assert time.time() - t < 60
    rec = json.loads((out / "run_record.json").read_text())
    assert rec["status"] == "interrupted" and 0 < rec["counts"]["agents"] < 400
    assert rec["counts"]["agents_failed"] == 0   # a cancelled campaign is not a failed one


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
