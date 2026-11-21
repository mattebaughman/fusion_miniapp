"""The non-Academy agent layer (the default): the PDF's stack, LangGraph agents on FlowGentic and RHAPSODY.

Every campaign is the parent's three agents (parent/evo_agents.py), each a compiled LangGraph subgraph
with its own state (its memory): a Coordinator, a Generator and an Evaluator. They are mounted as nodes
of one campaign graph and talk only through mailbox keys of the campaign state (`to_generator`,
`to_evaluator`, `to_coordinator`), carrying decide.py's messages: initialize, then per generation
evaluate and, in all but the last, repopulate. FlowGentic's AgentHost runs one campaign graph per
campaign.

The Evaluator's first step issues one model-free tool call per candidate (P `evaluate` calls in one
message); `route_tool_calls` sends each to the tool node, and `evaluate` is a FlowGentic task tool with
suspend=True on the process's RHAPSODY session. So the campaign is checkpointed in FlowGentic's key-value
saver and dropped while its generation runs, and the host resumes it when all P tasks have ended: one
suspension per generation, the parent's generation barrier. Model calls (the Generator's refill, the
Coordinator's summary) go through FlowGentic's fleet with the thinking budget per call (`extra_body`,
`max_tokens`, `reserve`, `meta`).

Laptop: RHAPSODY's concurrent backend (built by run.Services: spawned worker processes for the real
evaluate, threads for the fixture), an in-memory store, one fleet replica forwarding to the stub LLM.
Seams for a machine: build Services on the Dragon backend, pass `store` (a RadexStore) and `replicas`
(one ServiceModel per model server, each around its own llm.OpenAIChat), so the fleet routes and admits
calls across the servers.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any, TypedDict

from flowgentic.fleet import Fleet, TokenAdmission
from flowgentic.host import AgentHost, AgentState, Stagger
from flowgentic.langgraph import GraphRunner, KeyValueSaver, bind_tools, route_tool_calls
from flowgentic.store import MemoryStore
from flowgentic.tools import ToolRuntime, task_tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from fusion_miniapp import decide, record
from fusion_miniapp.llm import MAX_IN_FLIGHT, REPLY_TOKENS

ADMISSION = TokenAdmission(tokens=2_000_000, requests=MAX_IN_FLIGHT)   # per replica, all processes
STAGGER = Stagger(batch=500, interval_s=0.01)                 # first-wave starts
ROLE = {"system": "system", "human": "user", "ai": "assistant"}


class ServiceModel(BaseChatModel):
    """A fleet replica that forwards each call to the mini-app's LLM service (stub or OpenAI-compatible)."""

    service: Any

    @property
    def _llm_type(self) -> str:
        return "fusion-miniapp-llm-service"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("asynchronous only")

    async def _agenerate(self, messages, stop=None, run_manager=None, *, caller: str, extra_body: dict, **kwargs):
        b = extra_body["thinking_token_budget"]
        text, meta = await self.service(caller, [{"role": ROLE[m.type], "content": m.content} for m in messages],
                                        b, kwargs.get("max_tokens", b + REPLY_TOKENS) - b)
        usage = None
        if meta.get("prompt_tokens") or meta.get("completion_tokens"):   # the server's counts, for admission
            i, o = int(meta.get("prompt_tokens") or 0), int(meta.get("completion_tokens") or 0)
            usage = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text, usage_metadata=usage))])


class EvaluateArgs(BaseModel):
    params: dict[str, float | str]     # the parent's full candidate, with its "model" and "kind" strings
    candidate_idx: int
    generation: int
    seed: int


class Campaign(TypedDict, total=False):
    """The campaign state: each agent's memory and the three mailboxes."""
    messages: Annotated[list, add_messages]   # the Evaluator's tool calls and their results
    seed: int
    coordinator: dict
    generator: dict
    evaluator: dict
    to_generator: dict | None
    to_evaluator: dict | None
    to_coordinator: dict | None
    result: dict | None


class CoordinatorState(TypedDict, total=False):
    seed: int
    coordinator: dict
    to_generator: dict | None
    to_evaluator: dict | None
    to_coordinator: dict | None
    result: dict | None


class GeneratorState(TypedDict, total=False):
    seed: int
    generator: dict
    to_generator: dict | None
    to_coordinator: dict | None


class EvaluatorState(TypedDict, total=False):
    seed: int
    messages: Annotated[list, add_messages]
    evaluator: dict
    to_evaluator: dict | None
    to_coordinator: dict | None


def _graphs(svc, knobs: dict, fleet: Fleet, tools: list, saver: KeyValueSaver, out):
    def asker(agent_id: str):
        async def ask(messages: list[dict], b: int, r: int) -> str:
            lc = [SystemMessage(m["content"]) if m["role"] == "system" else HumanMessage(m["content"])
                  for m in messages]
            reply = await fleet.ainvoke(agent_id, lc, caller=agent_id, extra_body={"thinking_token_budget": b},
                                        max_tokens=b + r, reserve=b + r, meta={"budget": b, "reply_tokens": r})
            return str(reply.content)
        return ask

    def aid(config: RunnableConfig) -> str:
        return config["configurable"]["thread_id"]

    # --- the Coordinator: one step, which reads whatever reply arrived ---
    async def coordinate(s: CoordinatorState, config: RunnableConfig) -> dict:
        agent_id = aid(config)
        run_dir = decide.campaign_dir(out, knobs, agent_id)
        cst, reply = s.get("coordinator"), s.get("to_coordinator")
        if cst is None:                                                # start: lookups, config.json
            t_start = time.time()
            ctx = await decide.look(svc, agent_id, decide.lookups(knobs))
            cst = decide.coordinator_state(agent_id, s["seed"], knobs, ctx, t_start)
            return {"coordinator": cst, "to_generator": decide.start(cst, knobs, run_dir, getattr(svc, "sim", "real")),
                    "to_coordinator": None}
        if "population" in reply:                                      # from the Generator
            cst, msg = decide.on_population(cst, reply)
            return {"coordinator": cst, "to_evaluator": msg, "to_coordinator": None}
        cst, nxt = decide.on_evaluated(cst, reply, knobs, run_dir)     # from the Evaluator
        await svc.remember(agent_id, decide.memory(cst))
        if nxt is not None:
            return {"coordinator": cst, "to_generator": nxt, "to_coordinator": None}
        result = decide.finish(cst, await asker(agent_id)(decide.summary_messages(cst), cst["budget"],
                                                          decide.SUMMARY_TOKENS), run_dir)
        await svc.report(agent_id, result)
        return {"coordinator": {}, "result": result, "to_coordinator": None}

    # --- the Generator: initialize or repopulate, and in llm mode its model call ---
    async def generate(s: GeneratorState, config: RunnableConfig) -> dict:
        agent_id = aid(config)
        gst = s.get("generator") or decide.generator_state(agent_id, s["seed"], knobs)
        gst, reply = await decide.generator_step(gst, s["to_generator"], knobs, asker(agent_id))
        return {"generator": gst, "to_coordinator": reply, "to_generator": None}

    # --- the Evaluator: fan out P suspending evaluate calls, then collect all of them ---
    async def fan_out(s: EvaluatorState, config: RunnableConfig) -> dict:
        msg, seed = s["to_evaluator"], s["seed"]
        calls = [{"name": "evaluate", "id": c["id"],
                  "args": {"params": c["params"], "candidate_idx": c["candidate_idx"],
                           "generation": c["generation"], "seed": seed}}
                 for c in decide.evaluation_calls(msg["population"], msg["generation"])]
        return {"evaluator": {"t0": time.time()}, "messages": [AIMessage("", tool_calls=calls)]}

    async def collect(s: EvaluatorState) -> dict:
        replies = [m for m in s["messages"] if m.type == "tool"]
        for m in replies:
            if getattr(m, "status", "success") == "error":
                raise RuntimeError(str(m.content))
        t0 = s["evaluator"]["t0"]
        reply = {"results": [json.loads(m.content) for m in replies], "t0": t0, "elapsed_s": time.time() - t0}
        return {"to_coordinator": reply, "to_evaluator": None, "evaluator": {},
                "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    c = StateGraph(CoordinatorState)
    c.add_node("coordinate", coordinate)
    c.add_edge(START, "coordinate")
    c.add_edge("coordinate", END)

    g = StateGraph(GeneratorState)
    g.add_node("generate", generate)
    g.add_edge(START, "generate")
    g.add_edge("generate", END)

    e = StateGraph(EvaluatorState)
    e.add_node("fan_out", fan_out)
    e.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    e.add_node("collect", collect)
    e.add_edge(START, "fan_out")
    e.add_conditional_edges("fan_out", route_tool_calls, ["tools", END])
    e.add_edge("tools", "collect")
    e.add_edge("collect", END)

    def route(s: Campaign) -> str:                 # after the Coordinator: whom it wrote to
        if s.get("result") is not None:
            return END
        return "generator" if s.get("to_generator") else "evaluator"

    top = StateGraph(Campaign)
    top.add_node("coordinator", c.compile())
    top.add_node("generator", g.compile())
    top.add_node("evaluator", e.compile())
    top.add_edge(START, "coordinator")
    top.add_conditional_edges("coordinator", route, ["generator", "evaluator", END])
    top.add_edge("generator", "coordinator")
    top.add_edge("evaluator", "coordinator")
    return top.compile(checkpointer=saver)


def _live(states) -> tuple[int, int, int, int]:
    c = states.counts()
    return (c[AgentState.WORKING] + c[AgentState.TOOL_CALL], c[AgentState.MODEL_QUEUED] + c[AgentState.TOOL_WAIT],
            c[AgentState.MODEL_REQUEST], c[AgentState.DONE])


async def run_agents(n_agents: int, knobs: dict, svc_factory, out, *, first: int = 0, emit=None,
                     store=None, replicas=None) -> list[dict]:
    svc = svc_factory()
    a, o = knobs["agents"], knobs["outer"]
    live_cap = a["max_live"] or n_agents
    # A small cache of ended calls: FlowGentic also keeps every hosted campaign's task outcomes until the
    # campaign ends (_agents.register_pending), so a resume never runs a simulation again (Antigravity
    # 03:45: no repeats at 2,000 campaigns with 64)
    runtime = ToolRuntime(svc.session, retained=1024)
    evaluate = task_tool(svc.sim_fn, suspend=True, name="evaluate", args_schema=EvaluateArgs,
                         description="Evaluate one candidate profile with the turned-down IGNOS inner loop.",
                         prepare=lambda agent_id, call_id, args: {
                             **args, "knobs": svc.sim_knobs(args["seed"]), "agent_id": agent_id})
    # Every agent process admits its share of each replica's caps (FlowGentic's static split).
    fleet = Fleet(replicas or [ServiceModel(service=svc.llm_meta)], admission=ADMISSION,
                  share=min(1.0, n_agents / a["count"]))
    saver = KeyValueSaver(MemoryStore() if store is None else store)
    graph = _graphs(svc, knobs, fleet, bind_tools(runtime, [evaluate]), saver, out)
    results, suspensions = [], [0]

    def on_done(r, output) -> None:
        suspensions[0] += r.suspensions
        fleet.agent_done(r.id)
        # its result is reported: drop its checkpoints from our own store (each generation's Evaluator
        # subgraph leaves a namespace that is never reused); a store the caller passed in is the caller's
        if r.status == "done" and store is None:
            saver.delete_thread(r.id)
        results.append(output["result"] if r.status == "done"
                       else {"agent_id": r.id, "error": r.error or r.status})

    G = o["generations"]
    host = AgentHost(GraphRunner(graph, config={"recursion_limit": 8 * G + 25}),
                     max_live_agents=live_cap, stagger=STAGGER, on_done=on_done)
    tick = asyncio.create_task(record.ticker(lambda: _live(host.states), emit))
    report = None
    try:
        report = await host.run_indexed(range(first, first + n_agents), lambda i: {"seed": i}, id_prefix="a")
    finally:
        tick.cancel()
        await asyncio.gather(tick, return_exceptions=True)
        # also when interrupted: the counts so far (suspensions of the campaigns that ended)
        svc.sim_tasks += runtime.stats()["submitted"]
        f = fleet.stats()
        svc.layer = {"layer": "langgraph", "campaigns": n_agents, "hosted_runs": n_agents,
                     "roles_per_campaign": ["coordinator", "generator", "evaluator"], "max_live_agents": live_cap,
                     "suspensions": suspensions[0], "complete": report is not None,
                     "tasks": runtime.stats(), "checkpoints": saver.stats(),
                     "fleet": {k: f[k] for k in ("replicas", "calls", "in_flight_max", "errors")}}
        if report is not None:
            svc.layer.update(outcomes=report.outcomes, host_wall_s=report.wall_s,
                             share_by_category=report.states["share_by_category"], loop_lag=report.loop_lag)
    for r in results:
        if "error" in r:   # a failed campaign never reached its report step
            await svc.report(r["agent_id"], r)
    return results
