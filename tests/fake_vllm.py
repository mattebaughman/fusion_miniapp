"""A FIXTURE, not a model: a stand-in for one vLLM server, for the tests (not a test module).

FakeVllm serves on 127.0.0.1, on a thread of its own, what fusion uses of a vLLM server:
- /health (vLLM answers it without a key);
- /v1/models, the served model's name (`model`);
- /v1/chat/completions, OpenAI-compatible: each reply depends on the request's last line only (as the stub
  LLM's do), so every layer, and every server of several, gives the same replies; usage 100 prompt and 50
  completion tokens per reply;
- /metrics in vLLM's Prometheus format: the gauges vllm:num_requests_running, vllm:num_requests_waiting and
  vllm:kv_cache_usage_perc, from its own requests, and the counters AUTONMI compares with the fleets'
  (vllm:prompt_tokens_total, vllm:generation_tokens_total, vllm:request_success_total).

Each request takes `latency_s` (settable while it runs) in one of `batch` slots (None: no cap); a request
beyond the cap waits, counted as waiting. A request's KV-cache use is its 150 tokens of `kv_tokens`. With
`api_key`, /v1/* need it as a bearer token, as `vllm serve --api-key` does. With `metrics` false it has no
/metrics, as an OpenAI-compatible server that is not vLLM. It keeps every chat request's body; `fail(body)`
true answers that request with HTTP 400.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import threading

from fusion_miniapp import decide

PROMPT_TOKENS, COMPLETION_TOKENS = 100, 50


class FakeVllm:
    def __init__(self, fail=None, *, latency_s: float = 0.0, batch: int | None = None, model: str = "stand-in",
                 api_key: str | None = None, kv_tokens: int = 65_536, metrics: bool = True) -> None:
        self.fail, self.bodies = fail or (lambda body: False), []
        self.latency_s, self.batch, self.model, self.api_key = latency_s, batch, model, api_key
        self.kv_tokens, self.serves_metrics = kv_tokens, metrics
        self.running = self.waiting = self.running_max = self.waiting_max = 0
        self.prompt_tokens = self.generation_tokens = self.successes = 0
        self._slots = asyncio.Semaphore(batch or 10**9)
        self._loop = asyncio.new_event_loop()

    @property
    def url(self) -> str:
        """The base URL, without /v1, as a model job's endpoints file holds it."""
        return f"http://127.0.0.1:{self.port}"

    def _refused(self, request):
        from aiohttp import web
        if self.api_key and request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return web.json_response({"error": {"message": "Unauthorized"}}, status=401)
        return None

    async def _chat(self, request):
        from aiohttp import web
        refused = self._refused(request)
        if refused is not None:
            return refused
        body = await request.json()
        self.bodies.append(body)
        if self.fail(body):
            return web.json_response({"error": {"message": "rejected by the test"}}, status=400)
        self.waiting += 1
        self.waiting_max = max(self.waiting_max, self.waiting)
        try:
            await self._slots.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        self.running_max = max(self.running_max, self.running)
        try:
            await asyncio.sleep(self.latency_s)
            last = body["messages"][-1]["content"].splitlines()[-1]
            many = re.search(r"JSON list of (\d+)", last)
            if many:
                rng = random.Random(last)
                text = json.dumps([{k: round(rng.uniform(lo, hi), 6) for k, (lo, hi) in decide.ranges().items()}
                                   for _ in range(int(many.group(1)))])
            else:
                text = "Summary: " + body["messages"][-1]["content"].splitlines()[0]
            self.prompt_tokens += PROMPT_TOKENS
            self.generation_tokens += COMPLETION_TOKENS
            self.successes += 1
        finally:
            self.running -= 1
            self._slots.release()
        return web.json_response({"id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": text}}],
                                  "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS,
                                            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS}})

    async def _models(self, request):
        from aiohttp import web
        return self._refused(request) or web.json_response(
            {"object": "list", "data": [{"id": self.model, "object": "model", "created": 0,
                                         "owned_by": "fusion-test-fixture", "max_model_len": 32768}]})

    async def _health(self, request):
        from aiohttp import web
        return web.Response(text="")

    def metrics(self) -> str:
        """The /metrics text, in vLLM's Prometheus format."""
        labels = f'engine="0",model_name="{self.model}"'
        kv = self.running * (PROMPT_TOKENS + COMPLETION_TOKENS) / self.kv_tokens
        rows = [("gauge", "num_requests_running", labels, self.running),
                ("gauge", "num_requests_waiting", labels, self.waiting),
                ("gauge", "kv_cache_usage_perc", labels, kv),
                ("counter", "prompt_tokens_total", labels, self.prompt_tokens),
                ("counter", "generation_tokens_total", labels, self.generation_tokens),
                ("counter", "request_success_total", labels + ',finished_reason="stop"', self.successes),
                ("gauge", "cache_config_info", f'block_size="16",num_gpu_blocks="{self.kv_tokens // 16}"', 1)]
        out = []
        for kind, name, lab, value in rows:
            out += [f"# HELP vllm:{name} (fixture)", f"# TYPE vllm:{name} {kind}",
                    f"vllm:{name}{{{lab}}} {float(value)}"]
        return "\n".join(out) + "\n"

    async def _metrics(self, request):
        from aiohttp import web
        return web.Response(text=self.metrics(), content_type="text/plain")

    def __enter__(self) -> "FakeVllm":
        from aiohttp import web

        async def start():
            app = web.Application()
            app.router.add_post("/v1/chat/completions", self._chat)
            app.router.add_get("/v1/models", self._models)
            app.router.add_get("/health", self._health)
            if self.serves_metrics:
                app.router.add_get("/metrics", self._metrics)
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
        """The chat requests so far, and forget them."""
        out, self.bodies = self.bodies, []
        return out
