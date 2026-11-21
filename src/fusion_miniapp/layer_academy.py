"""The Academy agent layer (--use-academy): every campaign is the parent's three Academy agents.

As in parent/evo_agents.py, each campaign has a CoordinatorAgent, a GeneratorAgent and an
EvaluatorAgent, and every hop of the outer loop is an Academy action on a handle: initialize, then per
generation evaluate (the whole population in one message) and, in all but the last, repopulate. The
actions run decide.py's steps, so both layers give the same results.

Every agent of a process is hosted on its event loop: a Manager with no executor and the local exchange,
so no extra processes, servers or ports. The Evaluator submits its P evaluations to the agent process's
RHAPSODY session, the same session and worker processes as the LangGraph layer, and awaits all of them.

Academy has no suspend and resume: a campaign waiting for its generation stays resident on an await, so
this layer reports no suspensions. Model calls are admitted as in the LangGraph layer: at most this
process's share of llm.MAX_IN_FLIGHT at once (a semaphore; the fleet's token cap is not copied), and a
campaign queued for admission counts as waiting.

Differences from evo_agents.py: the Evaluator's action awaits P RHAPSODY tasks instead of running
evaluate_population_parallel in a thread (agent_run_sync); the Coordinator logs after updating the best
result (run_evolutionary_loop's order; the outputs are the same); the random stream is shared unless
outer.rng is per_agent (see decide.py).
"""
from __future__ import annotations

import asyncio

from academy.agent import Agent, action
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from fusion_miniapp import decide, record
from fusion_miniapp.llm import MAX_IN_FLIGHT


class GeneratorAgent(Agent):
    """The parent's generator role: initialize_population and repopulate, and in llm mode the model call
    that replaces mutants."""

    def __init__(self, agent_id: str, seed: int, knobs: dict, ask) -> None:
        super().__init__()
        self.knobs, self.ask = knobs, ask
        self.state = decide.generator_state(agent_id, seed, knobs)

    async def _step(self, msg: dict) -> dict:
        self.state, reply = await decide.generator_step(self.state, msg, self.knobs, self.ask)
        return reply

    @action
    async def initialize(self, msg: dict) -> dict:
        return await self._step(msg)

    @action
    async def repopulate(self, msg: dict) -> dict:
        return await self._step(msg)


class EvaluatorAgent(Agent):
    """The parent's evaluator role: one population in, all its results out."""

    def __init__(self, agent_id: str, seed: int, svc, live: record.Live) -> None:
        super().__init__()
        self.campaign_id, self.seed, self.svc, self.live = agent_id, seed, svc, live

    @action
    async def evaluate(self, population: list, generation: int) -> dict:
        self.live(self.campaign_id, "waiting")
        try:
            return await decide.evaluate_population(self.svc, self.campaign_id, self.seed, population, generation)
        finally:
            self.live(self.campaign_id, "working")


class CoordinatorAgent(Agent):
    """The parent's coordinator role: drives the generations, selects, keeps the best, logs, reports."""

    def __init__(self, agent_id: str, seed: int, knobs: dict, svc, generator, evaluator, ask, out,
                 live: record.Live) -> None:
        super().__init__()
        self.args = (agent_id, seed, knobs, svc, generator, evaluator, ask, out, live)

    @action
    async def run_generations(self) -> dict:
        return await decide.run_campaign(*self.args)


class _Gate:
    """Model calls admitted `cap` at a time in this process; a campaign queued for admission is waiting."""

    def __init__(self, svc, cap: int, live: record.Live) -> None:
        self.svc, self.gate, self.live = svc, asyncio.Semaphore(cap), live

    def ask(self, agent_id: str):
        async def ask(messages: list[dict], thinking_budget: int, reply_tokens: int) -> str:
            self.live(agent_id, "waiting")
            async with self.gate:
                self.live(agent_id, "model_request")
                try:
                    return await self.svc.llm(agent_id, messages, thinking_budget, reply_tokens)
                finally:
                    self.live(agent_id, "working")
        return ask


async def _one(agent_id: str, handle, svc, live: record.Live) -> dict:
    try:
        return await handle.run_generations()
    except Exception as exc:   # one campaign's failure ends that campaign only
        failed = {"agent_id": agent_id, "error": f"{type(exc).__name__}: {exc}"[:500]}
        await svc.report(agent_id, failed)
        return failed
    finally:
        live.finish(agent_id)


async def run_agents(n_agents: int, knobs: dict, svc_factory, out, *, first: int = 0, emit=None) -> list[dict]:
    """Campaigns first .. first+n-1. At most agents.max_live run at once (null: all of them): a campaign
    launches its three agents when it gets a slot and shuts them down when it ends, as FlowGentic's host
    admits and retires the LangGraph layer's campaigns."""
    svc, live, a = svc_factory(), record.Live(), knobs["agents"]
    svc.layer = {"layer": "academy", "campaigns": n_agents, "suspensions": None, "complete": False}
    cap = max(1, int(MAX_IN_FLIGHT * min(1.0, n_agents / a["count"])))   # this process's share, as the fleet's
    live_cap = min(a["max_live"] or n_agents, n_agents)
    gate, slots, peak = _Gate(svc, cap, live), asyncio.Semaphore(live_cap), {"now": 0, "max": 0}
    async with await Manager.from_exchange_factory(factory=LocalExchangeFactory()) as manager:   # no executor

        async def campaign(i: int) -> dict:
            aid = f"a{i}"
            async with slots:
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
                launched = []
                try:
                    launched.append(await manager.launch(GeneratorAgent(aid, i, knobs, gate.ask(aid))))
                    launched.append(await manager.launch(EvaluatorAgent(aid, i, svc, live)))
                    launched.append(await manager.launch(
                        CoordinatorAgent(aid, i, knobs, svc, *launched, gate.ask(aid), out, live)))
                    return await _one(aid, launched[-1], svc, live)
                finally:
                    peak["now"] -= 1
                    me = asyncio.current_task()
                    if not (me and me.cancelling()):   # interrupted: the manager's exit stops them all at once
                        for h in reversed(launched):
                            await manager.shutdown(h, blocking=True, raise_error=False)

        tick = asyncio.create_task(record.ticker(live.counts, emit))
        try:
            results = await asyncio.gather(*(campaign(i) for i in range(first, first + n_agents)))
        finally:
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
            svc.layer.update(academy_agents=3 * n_agents, max_live_agents=live_cap,
                             live_campaigns_max=peak["max"], roles_per_campaign=["coordinator", "generator",
                                                                                 "evaluator"],
                             exchange="local", executor=None, llm_requests_cap=cap)
    svc.layer["complete"] = True
    return list(results)
