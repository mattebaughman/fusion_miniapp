"""The LLM service: a seeded stub for laptops and tests, or any OpenAI-compatible endpoint.

Both are called as `await model(agent_id, messages, thinking_budget)` and return the reply text.
The thinking budget travels with every call (vLLM's per-request `thinking_token_budget`, with
`max_tokens` = budget + REPLY_TOKENS), and both record what they were asked to send.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from collections import Counter

from fusion_miniapp.decide import ranges

REPLY_TOKENS = 256   # room for the visible reply on top of the thinking budget
MAX_IN_FLIGHT = 512  # requests in flight per model server, over all agent processes (both layers admit
                     # per_process / count of it in each process); tune per model server


class _Recorder:
    def __init__(self) -> None:
        self.calls, self.budgets = 0, Counter()

    def _record(self, thinking_budget: int) -> None:
        self.calls += 1
        self.budgets[thinking_budget] += 1

    def seen(self) -> dict:
        return {"mode": self.mode, "calls": self.calls, "budgets": {str(b): n for b, n in self.budgets.items()}}


class StubLLM(_Recorder):
    """FIXTURE, not a model. A proposal request (its last line asks for JSON) gets seven parameters
    drawn in range from a generator seeded by the agent and that line; a summary request gets a
    one-line echo of the best candidate. Replies depend on nothing else, so both layers agree."""

    mode = "stub"

    def __init__(self, seed: int = 0, latency_s: float = 0.001) -> None:
        super().__init__()
        self.seed, self.latency_s = seed, latency_s

    async def __call__(self, agent_id: str, messages: list[dict], thinking_budget: int) -> str:
        self._record(thinking_budget)
        await asyncio.sleep(self.latency_s)
        lines = messages[-1]["content"].splitlines()
        if "JSON" in lines[-1]:
            rng = random.Random(f"{self.seed}/{agent_id}/{lines[-1]}")
            return json.dumps({k: round(rng.uniform(lo, hi), 6) for k, (lo, hi) in ranges().items()})
        return "Stub summary: " + lines[0]


class OpenAIChat(_Recorder):
    """An OpenAI-compatible chat endpoint (the stack's vLLM servers, or any other). The base URL and
    key are read from the environment variables NAMED in the knob file; one client per process."""

    mode = "openai"

    def __init__(self, base_url_env: str, api_key_env: str, model: str) -> None:
        super().__init__()
        from openai import AsyncOpenAI
        base_url = os.environ.get(base_url_env)
        if not base_url:
            raise RuntimeError(f"llm.mode is openai but the environment variable {base_url_env} is not set")
        # A local vLLM server needs no key; the client still wants a non-empty string.
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get(api_key_env) or "none")
        self.model = model

    async def __call__(self, agent_id: str, messages: list[dict], thinking_budget: int) -> str:
        self._record(thinking_budget)
        reply = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=thinking_budget + REPLY_TOKENS,
            extra_body={"thinking_token_budget": thinking_budget})
        return reply.choices[0].message.content or ""


def make(llm_knobs: dict):
    """The LLM service the `llm` knob section asks for."""
    if llm_knobs["mode"] == "stub":
        return StubLLM()
    return OpenAIChat(llm_knobs["base_url_env"], llm_knobs["api_key_env"], llm_knobs["model"])
