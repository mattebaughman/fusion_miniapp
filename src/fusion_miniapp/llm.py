"""The LLM service: a seeded stub for laptops and tests, or any OpenAI-compatible endpoint.

Both are called as `await model(agent_id, messages, thinking_budget)` and return the reply text
(or `await model.call_with_meta(...)` for `(text, meta)`). The thinking budget travels with every
call when `send_thinking_budget` is true (vLLM's per-request `thinking_token_budget`, with
`max_tokens` = budget + REPLY_TOKENS). Both record per-call outcomes, tokens and latencies.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from collections import Counter

from fusion_miniapp.decide import GUIDANCE_ASK, NOTE_ASK, SCORES_ASK, numbers_score, ranges

REPLY_TOKENS = 256   # room for the visible reply on top of the thinking budget
MAX_IN_FLIGHT = 512  # requests in flight per model server, over all agent processes (both layers admit
                     # their share in each process); tune per model server (the AUTONMI layer: llm.max_in_flight)


class _Recorder:
    def __init__(self) -> None:
        self.calls = 0
        self.budgets: Counter = Counter()
        self.empty = 0
        self.errors = 0
        self.retries = 0
        self.finish_reasons: Counter = Counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self._latencies: list[float] = []

    def _record_ok(
        self,
        thinking_budget: int,
        latency_s: float,
        finish_reason: str,
        empty: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> dict:
        lat = round(latency_s, 4)
        self.calls += 1
        self.budgets[thinking_budget] += 1
        if empty:
            self.empty += 1
        self.finish_reasons[finish_reason] += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.reasoning_tokens += reasoning_tokens
        self._latencies.append(lat)
        return {
            "latency_s": lat,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "empty": empty,
        }

    def _record_err(self, thinking_budget: int, latency_s: float) -> None:
        self.calls += 1
        self.budgets[thinking_budget] += 1
        self.errors += 1
        self._latencies.append(round(latency_s, 4))

    def seen(self) -> dict:
        return {
            "mode": self.mode,
            "calls": self.calls,
            "budgets": {str(b): n for b, n in self.budgets.items()},
            "empty": self.empty,
            "errors": self.errors,
            "retries": self.retries,
            "finish_reasons": dict(self.finish_reasons),
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "reasoning": self.reasoning_tokens,
            },
            "latency_s": list(self._latencies),
        }

    async def close(self) -> None:
        pass


def numbers_only(content: str, weight: float = 0.25) -> str | None:
    """FIXTURE, not a model: the stub's reply (and tests/fake_vllm.py's) to the calls the AUTONMI layer's roles add,
    known by the request's last line (decide's asks): the Evaluator's gets, for each candidate, the score from the
    numbers alone (decide.numbers_score of the q and m its prompt gives, at monitor.weight `weight`); a Monitor's
    note and the Coordinator's guidance get "" (none). None for any other request."""
    lines = content.splitlines()
    last = lines[-1] if lines else ""
    if last in (NOTE_ASK, GUIDANCE_ASK):
        return ""
    if last == SCORES_ASK:
        found = (re.match(r"candidate (\d+): q ([^;]+); (?:m (\S+) |no trace)", x) for x in lines)
        return json.dumps({m.group(1): numbers_score(float(m.group(2)), m.group(3) and float(m.group(3)), weight)
                           for m in found if m})
    return None


class StubLLM(_Recorder):
    """FIXTURE, not a model. A refill request (its last line asks for "a JSON list of N objects") gets
    N candidates of the seven parameters, drawn in range from a generator seeded by the agent and that
    line; a request for one JSON object gets one; a summary request gets a one-line echo of the best
    candidate; the AUTONMI layer's notes, scores and guidance the value from numbers alone (numbers_only).
    Replies depend on nothing else, so both layers agree."""

    mode = "stub"

    def __init__(self, seed: int = 0, latency_s: float = 0.001, weight: float = 0.25) -> None:
        super().__init__()
        self.seed, self.latency_s, self.weight = seed, latency_s, weight   # weight: monitor.weight (numbers_only)

    async def call_with_meta(
        self,
        agent_id: str,
        messages: list[dict],
        thinking_budget: int,
        reply_tokens: int = REPLY_TOKENS,
    ) -> tuple[str, dict]:
        t0 = time.perf_counter()
        await asyncio.sleep(self.latency_s)
        lines = messages[-1]["content"].splitlines()
        text = numbers_only(messages[-1]["content"], self.weight)
        if text is None and "JSON" in lines[-1]:
            rng = random.Random(f"{self.seed}/{agent_id}/{lines[-1]}")
            draw = lambda: {k: round(rng.uniform(lo, hi), 6) for k, (lo, hi) in ranges().items()}  # noqa: E731
            many = re.search(r"JSON list of (\d+)", lines[-1])
            text = json.dumps([draw() for _ in range(int(many.group(1)))] if many else draw())
        elif text is None:
            text = "Stub summary: " + lines[0]
        meta = self._record_ok(thinking_budget, time.perf_counter() - t0, "stop", not text.strip())
        return text, meta

    async def __call__(
        self,
        agent_id: str,
        messages: list[dict],
        thinking_budget: int,
        reply_tokens: int = REPLY_TOKENS,
    ) -> str:
        text, _ = await self.call_with_meta(agent_id, messages, thinking_budget, reply_tokens=reply_tokens)
        return text


def _is_retryable(exc: Exception) -> bool:
    from openai import APIConnectionError, APIStatusError
    if isinstance(exc, APIConnectionError):   # includes APITimeoutError
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


def _http_limits() -> object:
    try:
        from httpx2 import Limits
    except ImportError:
        from httpx import Limits
    return Limits(max_connections=MAX_IN_FLIGHT + 32, max_keepalive_connections=MAX_IN_FLIGHT)


class OpenAIChat(_Recorder):
    """An OpenAI-compatible chat endpoint (the stack's vLLM servers, or any other). The base URL and
    key are read from the environment variables NAMED in the knob file; one client per process."""

    mode = "openai"

    def __init__(
        self,
        base_url_env: str,
        api_key_env: str,
        model: str,
        *,
        send_thinking_budget: bool = True,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        retry_delay_s: float = 0.25,
    ) -> None:
        super().__init__()
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient
        base_url = os.environ.get(base_url_env)
        if not base_url:
            raise RuntimeError(f"llm.mode is openai but the environment variable {base_url_env} is not set")
        self.model = model
        self.send_thinking_budget = bool(send_thinking_budget)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_delay_s = float(retry_delay_s)
        self._http = DefaultAsyncHttpxClient(limits=_http_limits(), timeout=self.timeout_s)
        # Retries are handled in call_with_meta so retries and errors are counted accurately.
        # A local vLLM server needs no key; the client still wants a non-empty string.
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env) or "none",
            timeout=self.timeout_s,
            max_retries=0,
            http_client=self._http,
        )

    async def call_with_meta(
        self,
        agent_id: str,
        messages: list[dict],
        thinking_budget: int,
        reply_tokens: int = REPLY_TOKENS,
    ) -> tuple[str, dict]:
        t0 = time.perf_counter()
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": thinking_budget + reply_tokens,
        }
        if self.send_thinking_budget:
            kwargs["extra_body"] = {"thinking_token_budget": thinking_budget}
        reply = None
        for attempt in range(self.max_retries + 1):
            try:
                reply = await self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if attempt < self.max_retries and _is_retryable(exc):
                    self.retries += 1
                    await asyncio.sleep(min(self.retry_delay_s * (2 ** attempt), 4.0))
                    continue
                self._record_err(thinking_budget, time.perf_counter() - t0)
                raise
        choices = getattr(reply, "choices", None) or []
        if choices:
            ch = choices[0]
            text = getattr(ch.message, "content", None) or ""
            finish = getattr(ch, "finish_reason", None) or "unknown"
        else:
            text, finish = "", "empty_choices"
        usage = getattr(reply, "usage", None)
        prompt_t = int(getattr(usage, "prompt_tokens", None) or 0)
        comp_t = int(getattr(usage, "completion_tokens", None) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        reason_t = int(getattr(details, "reasoning_tokens", None) or 0)
        meta = self._record_ok(
            thinking_budget,
            time.perf_counter() - t0,
            finish,
            not text.strip(),
            prompt_t,
            comp_t,
            reason_t,
        )
        return text, meta

    async def __call__(
        self,
        agent_id: str,
        messages: list[dict],
        thinking_budget: int,
        reply_tokens: int = REPLY_TOKENS,
    ) -> str:
        text, _ = await self.call_with_meta(agent_id, messages, thinking_budget, reply_tokens=reply_tokens)
        return text

    async def close(self) -> None:
        try:
            await self.client.close()
        finally:
            if not self._http.is_closed:
                await self._http.aclose()


def make(llm_knobs: dict):
    """The LLM service the `llm` knob section asks for."""
    if llm_knobs["mode"] == "stub":
        return StubLLM(latency_s=llm_knobs.get("stub_latency_s", 0.001))
    return OpenAIChat(
        llm_knobs["base_url_env"],
        llm_knobs["api_key_env"],
        llm_knobs["model"],
        send_thinking_budget=llm_knobs.get("send_thinking_budget", True),
        timeout_s=llm_knobs.get("timeout_s", 120.0),
        max_retries=llm_knobs.get("max_retries", 2),
    )
