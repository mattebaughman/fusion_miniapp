"""Agent side: the stub LLM, the OpenAI-compatible client against an in-process stub endpoint, parsing.

The endpoint is a FIXTURE (it is not a model): it listens on 127.0.0.1 on a thread of this test and is
stopped before the test returns.
"""
import asyncio
import threading

from aiohttp import web

from fusion_miniapp import decide, llm
from test_agent_layers import LAYERS, knobs_for, run_layer


def test_stub_is_seeded_and_returns_valid_proposals():
    st = decide.new_state("a7", 7, knobs_for(1))
    msgs = decide.proposal_messages(st)
    stub = llm.StubLLM()
    one, two = (asyncio.run(stub("a7", msgs, 64)) for _ in range(2))
    assert one == two and decide.parse_candidate(one).keys() == decide.ranges().keys()
    assert asyncio.run(stub("a7", decide.summary_messages(st), 64)).startswith("Stub summary: 0 candidates")
    assert stub.seen() == {"mode": "stub", "calls": 3, "budgets": {"64": 3}}


def test_parse_candidate_clips_ignores_and_rejects():
    lo, hi = decide.ranges()["Delta"]
    assert decide.parse_candidate('thinking... {"Delta": 9, "a1_ne": 3.1, "junk": 1}') == {"Delta": hi, "a1_ne": 3.1}
    for bad in ("no json here", '{"a1_ne": "high"}', "[1, 2]", '{"a1_ne": true}', "{broken"):
        assert decide.parse_candidate(bad) is None
    for reply in ('<think>weigh {wall, pedestal}; draft {"a1_ne": 1.0}</think> {"a1_ne": 3.0}',   # reasoning first
                  '{"a1_ne": 3.0} I hope {this} helps', 'Answer: {"candidate": {"a1_ne": 3.0}}',  # prose, wrapped
                  '{"a1_ne": 2.0}\nrevised: {"a1_ne": 3.0}'):                                    # two objects
        assert decide.parse_candidate(reply) == {"a1_ne": 3.0}, reply
    st = decide.proposed(decide.new_state("a1", 1, knobs_for(1)), "no json here")
    assert st["pending"]["source"] == "mutation" and st["n_llm_calls"] == 1


def test_openai_client_sends_the_budget_from_both_layers(tmp_path, monkeypatch):
    seen = []

    async def chat(request):
        body = await request.json()
        seen.append(body)
        return web.json_response({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a1_ne": 3.0}'}}],
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
    try:
        monkeypatch.setenv("MINIAPP_TEST_LLM_URL", f"http://127.0.0.1:{port}/v1")
        knobs = knobs_for(3, sims=2, thinking_budget=96)
        for layer in LAYERS:
            model = llm.make({"mode": "openai", "base_url_env": "MINIAPP_TEST_LLM_URL",
                              "api_key_env": "MINIAPP_TEST_LLM_KEY", "model": "test-model"})
            results, _ = run_layer(layer, knobs, tmp_path / layer, model=model)
            assert not [r for r in results if "error" in r]
            assert {h["source"] for r in results for h in r["history"]} == {"llm"}
            assert {h["candidate"]["a1_ne"] for r in results for h in r["history"]} == {3.0}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.run_until_complete(runner.cleanup())
        loop.close()
    assert len(seen) == 2 * 3 * (2 + 1)                                     # both layers, S + 1 calls per agent
    assert {b["thinking_token_budget"] for b in seen} == {96}
    assert {b["max_tokens"] for b in seen} == {96 + llm.REPLY_TOKENS}
    assert {b["model"] for b in seen} == {"test-model"}
