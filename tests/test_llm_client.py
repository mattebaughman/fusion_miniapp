"""Package 5 tests for llm.py: telemetry, empty/cut-off replies, thinking-budget toggle, retries/timeouts.

Every HTTP test uses a local in-process stub server on 127.0.0.1:0 (not a model, no external network).
"""
from __future__ import annotations

import asyncio
import json
import threading

import openai
import pytest
from aiohttp import web

from fusion_miniapp import decide, llm


class _StubServer:
    """Minimal OpenAI-compatible HTTP stub on a background thread."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self._loop = asyncio.new_event_loop()
        self._runner = None
        self.port = 0
        self._thread = None

    def __enter__(self) -> "_StubServer":
        async def start():
            app = web.Application()
            app.router.add_post("/v1/chat/completions", self._handler)
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

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def test_stub_llm_records_seen_and_handles_list_and_summary():
    async def go():
        stub = llm.StubLLM(seed=3, latency_s=0.0)
        one_msg = [{"role": "user", "content": "Round 1.\nReply with one JSON object."}]
        list_msg = [{"role": "user", "content": "Gen 1.\nReply with a JSON list of 3 objects."}]
        sum_msg = [{"role": "user", "content": "2 candidates evaluated.\nSummarise."}]
        r1 = await stub("a0", one_msg, 64)
        r2 = await stub("a0", list_msg, 128, reply_tokens=448)
        r3, meta3 = await stub.call_with_meta("a0", sum_msg, 64, reply_tokens=256)
        await stub.close()
        return r1, r2, r3, meta3, stub.seen()

    r1, r2, r3, meta3, seen = asyncio.run(go())
    keys = set(decide.ranges())
    assert set(json.loads(r1)) == keys
    items = json.loads(r2)
    assert isinstance(items, list) and len(items) == 3 and all(set(x) == keys for x in items)
    assert r3 == "Stub summary: 2 candidates evaluated."
    assert meta3["finish_reason"] == "stop" and meta3["empty"] is False
    assert seen["mode"] == "stub"
    assert seen["calls"] == 3 and seen["budgets"] == {"64": 2, "128": 1}
    assert seen["empty"] == 0 and seen["errors"] == 0 and seen["retries"] == 0
    assert seen["finish_reasons"] == {"stop": 3}
    assert seen["tokens"] == {"prompt": 0, "completion": 0, "reasoning": 0}
    assert len(seen["latency_s"]) == 3 and all(isinstance(x, float) for x in seen["latency_s"])


def test_openai_records_empty_cutoff_and_empty_choices(monkeypatch):
    calls = 0

    async def handler(request):
        nonlocal calls
        body = await request.json()
        calls += 1
        if calls == 1:
            return web.json_response({
                "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": "",
                                         "reasoning_content": "thinking " * 20}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 320, "total_tokens": 620,
                          "completion_tokens_details": {"reasoning_tokens": 320}},
            })
        return web.json_response({
            "id": "y", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        })

    with _StubServer(handler) as srv:
        monkeypatch.setenv("TEST_LLM_URL", srv.base_url)

        async def go():
            model = llm.make({
                "mode": "openai", "base_url_env": "TEST_LLM_URL", "api_key_env": "TEST_LLM_KEY",
                "model": "reasoner", "send_thinking_budget": True, "timeout_s": 5.0, "max_retries": 0,
            })
            try:
                t1, m1 = await model.call_with_meta("a0", [{"role": "user", "content": "hi"}], 64)
                t2, m2 = await model.call_with_meta("a0", [{"role": "user", "content": "hi"}], 64)
                return t1, m1, t2, m2, model.seen()
            finally:
                await model.close()

        t1, m1, t2, m2, seen = asyncio.run(go())

    assert t1 == "" and m1["empty"] is True and m1["finish_reason"] == "length"
    assert m1["reasoning_tokens"] == 320 and m1["prompt_tokens"] == 300 and m1["completion_tokens"] == 320
    assert t2 == "" and m2["empty"] is True and m2["finish_reason"] == "empty_choices"
    assert seen["mode"] == "openai" and seen["calls"] == 2 and seen["empty"] == 2 and seen["errors"] == 0
    assert seen["finish_reasons"] == {"length": 1, "empty_choices": 1}
    assert seen["tokens"] == {"prompt": 310, "completion": 320, "reasoning": 320}
    assert len(seen["latency_s"]) == 2


def test_openai_vllm_400_on_thinking_budget_then_passes_when_disabled(monkeypatch):
    bodies = []

    async def handler(request):
        body = await request.json()
        bodies.append(body)
        if "thinking_token_budget" in body:
            return web.json_response(
                {"error": {
                    "message": "thinking_token_budget is set but reasoning_config is not configured.",
                    "type": "BadRequestError", "param": None, "code": 400,
                }},
                status=400,
            )
        return web.json_response({
            "id": "ok", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '[{"a1_ne": 3.0}]'}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62},
        })

    with _StubServer(handler) as srv:
        monkeypatch.setenv("TEST_LLM_URL", srv.base_url)

        async def run_one(send_budget: bool):
            model = llm.make({
                "mode": "openai", "base_url_env": "TEST_LLM_URL", "api_key_env": "TEST_LLM_KEY",
                "model": "vllm-model", "send_thinking_budget": send_budget, "timeout_s": 5.0, "max_retries": 2,
            })
            try:
                text = await model("a0", [{"role": "user", "content": "propose"}], 64, reply_tokens=1088)
                return text, None, model.seen()
            except Exception as exc:
                return None, exc, model.seen()
            finally:
                await model.close()

        text_on, err_on, seen_on = asyncio.run(run_one(True))
        text_off, err_off, seen_off = asyncio.run(run_one(False))

    assert text_on is None and isinstance(err_on, openai.BadRequestError)
    assert seen_on["errors"] == 1 and seen_on["retries"] == 0 and seen_on["calls"] == 1
    assert bodies[0]["thinking_token_budget"] == 64
    assert bodies[0]["max_tokens"] == 64 + 1088

    assert err_off is None and text_off == '[{"a1_ne": 3.0}]'
    assert "thinking_token_budget" not in bodies[1]
    assert bodies[1]["max_tokens"] == 64 + 1088
    assert seen_off["errors"] == 0 and seen_off["retries"] == 0 and seen_off["empty"] == 0
    assert seen_off["finish_reasons"] == {"stop": 1}
    assert seen_off["tokens"] == {"prompt": 50, "completion": 12, "reasoning": 0}


def test_openai_counts_retries_on_5xx_and_timeouts(monkeypatch):
    attempts = {"5xx": 0, "slow": 0}

    async def handler(request):
        body = await request.json()
        if body["model"] == "flaky":
            attempts["5xx"] += 1
            if attempts["5xx"] <= 2:
                return web.json_response({"error": {"message": "overloaded"}}, status=503)
            return web.json_response({
                "id": "ok", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "recovered"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
        attempts["slow"] += 1
        await asyncio.sleep(1.0)
        return web.json_response({"choices": []})

    with _StubServer(handler) as srv:
        monkeypatch.setenv("TEST_LLM_URL", srv.base_url)

        async def go():
            flaky = llm.OpenAIChat(
                "TEST_LLM_URL", "TEST_LLM_KEY", "flaky",
                send_thinking_budget=False, timeout_s=2.0, max_retries=2, retry_delay_s=0.02,
            )
            slow = llm.OpenAIChat(
                "TEST_LLM_URL", "TEST_LLM_KEY", "slow",
                send_thinking_budget=False, timeout_s=0.15, max_retries=1, retry_delay_s=0.02,
            )
            try:
                ok_text = await flaky("a0", [{"role": "user", "content": "x"}], 32)
                with pytest.raises(openai.APITimeoutError):
                    await slow("a0", [{"role": "user", "content": "x"}], 32)
                return ok_text, flaky.seen(), slow.seen()
            finally:
                await flaky.close()
                await slow.close()

        ok_text, flaky_seen, slow_seen = asyncio.run(go())

    assert ok_text == "recovered"
    assert attempts["5xx"] == 3 and flaky_seen["calls"] == 1 and flaky_seen["retries"] == 2 and flaky_seen["errors"] == 0
    assert attempts["slow"] == 2 and slow_seen["calls"] == 1 and slow_seen["retries"] == 1 and slow_seen["errors"] == 1


def test_openai_http_limits_and_close(monkeypatch):
    monkeypatch.setenv("TEST_LLM_URL", "http://127.0.0.1:9/v1")

    async def go():
        model = llm.make({
            "mode": "openai", "base_url_env": "TEST_LLM_URL", "api_key_env": "TEST_LLM_KEY",
            "model": "m", "timeout_s": 1.0, "max_retries": 0,
        })
        limits = model._http._transport._pool._max_connections
        keepalive = model._http._transport._pool._max_keepalive_connections
        await model.close()
        return limits, keepalive, model._http.is_closed

    limits, keepalive, closed = asyncio.run(go())
    assert limits >= llm.MAX_IN_FLIGHT
    assert keepalive >= llm.MAX_IN_FLIGHT
    assert closed is True
