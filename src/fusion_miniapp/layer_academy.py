"""The Academy agent layer (--use-academy): the same campaigns as Academy agents.

Every agent is hosted in this process, on its event loop: a Manager with no executor and the local
exchange, so no extra processes, servers or ports. A CampaignAgent runs decide.run_campaign; one
EvaluatorAgent per process sits in front of evaluate, so every simulation is an Academy action
(a message from the campaign agent's mailbox to the evaluator's). The evaluator submits each one to
the same RHAPSODY session, and so the same worker processes, as the LangGraph layer.

Academy has no suspend and resume: an agent waiting for a simulation stays resident on an await,
so this layer reports no suspensions. Model calls are admitted as in the LangGraph layer: at most
this process's share of llm.MAX_IN_FLIGHT at once (a semaphore; the fleet's token cap is not copied),
and an agent queued for admission counts as waiting.

The parent's evo_agents.py classes are not reused: they move whole populations through
`agent_run_sync` (a thread per call) into process pools, the wrong shape for one-candidate
campaign agents; these two classes are smaller than adapting them.
"""
from __future__ import annotations

import asyncio

from academy.agent import Agent, action
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from fusion_miniapp import decide, record
from fusion_miniapp.llm import MAX_IN_FLIGHT


class EvaluatorAgent(Agent):
    """The parent's evaluator role, one per process, in front of the simulation service."""

    def __init__(self, svc) -> None:
        super().__init__()
        self.svc = svc

    @action
    async def evaluate(self, agent_id: str, candidate: dict, seed: int) -> dict:
        return await self.svc.simulate(agent_id, candidate, seed)


class CampaignAgent(Agent):
    """One campaign: the parent's coordinator and generator roles for one agent's candidates."""

    def __init__(self, campaign_id: str, seed: int, knobs: dict, svc, on_state) -> None:
        super().__init__()
        self.campaign_id, self.seed, self.knobs, self.svc, self.on_state = campaign_id, seed, knobs, svc, on_state

    @action
    async def run(self) -> dict:
        return await decide.run_campaign(self.campaign_id, self.seed, self.knobs, self.svc, self.on_state)


class _ViaEvaluator:
    """The process's services, with simulate sent to the EvaluatorAgent as an Academy action and
    model calls admitted `cap` at a time."""

    def __init__(self, svc, evaluator, cap: int, live: record.Live) -> None:
        self._svc, self._evaluator, self._gate, self._live = svc, evaluator, asyncio.Semaphore(cap), live

    def __getattr__(self, name):
        return getattr(self._svc, name)

    async def llm(self, agent_id: str, messages: list[dict], thinking_budget: int) -> str:
        self._live(agent_id, "waiting")                          # queued for admission
        async with self._gate:
            self._live(agent_id, "model_request")
            return await self._svc.llm(agent_id, messages, thinking_budget)

    async def simulate(self, agent_id: str, candidate: dict, seed: int) -> dict:
        return await self._evaluator.evaluate(agent_id, candidate, seed)


async def _one(agent_id: str, handle, svc, live: record.Live) -> dict:
    try:
        return await handle.run()
    except Exception as exc:   # one agent's failure ends that agent only
        failed = {"agent_id": agent_id, "error": f"{type(exc).__name__}: {exc}"[:500]}
        await svc.report(agent_id, failed)
        return failed
    finally:
        live.finish(agent_id)


async def run_agents(n_agents: int, knobs: dict, svc_factory, out, *, first: int = 0, emit=None) -> list[dict]:
    svc, live, a = svc_factory(), record.Live(), knobs["agents"]
    cap = max(1, int(MAX_IN_FLIGHT * min(1.0, a["per_process"] / a["count"])))   # the fleet's per-process share
    async with await Manager.from_exchange_factory(factory=LocalExchangeFactory()) as manager:   # no executor
        evaluator = await manager.launch(EvaluatorAgent(svc))
        via = _ViaEvaluator(svc, evaluator, cap, live)
        ids = [f"a{i}" for i in range(first, first + n_agents)]
        handles = [await manager.launch(CampaignAgent(aid, i, knobs, via, live))
                   for i, aid in zip(range(first, first + n_agents), ids)]
        tick = asyncio.create_task(record.ticker(live.counts, emit))
        try:
            results = await asyncio.gather(*(_one(aid, h, svc, live) for aid, h in zip(ids, handles)))
        finally:
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
    svc.layer = {"layer": "academy", "academy_agents": n_agents + 1, "exchange": "local", "executor": None,
                 "suspensions": None, "llm_requests_cap": cap}
    return list(results)
