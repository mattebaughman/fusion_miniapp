"""The non-Academy agent layer (the default): the PDF's stack, LangGraph agents on FlowGentic and RHAPSODY.

Each agent is one small LangGraph graph run by FlowGentic's AgentHost. The nodes are decide.py's
steps. Model calls go through FlowGentic's fleet, with the thinking budget per call (`extra_body`,
`max_tokens`, `reserve`, `meta`). `evaluate` is a FlowGentic task tool with suspend=True on the
process's RHAPSODY session: the agent's state is checkpointed in FlowGentic's key-value saver and
dropped while the simulation runs, and the host resumes it when the task ends.

Laptop: RHAPSODY's concurrent backend (built by run.Services: spawned worker processes for the
real evaluate, threads for the fixture), an in-memory store, one fleet replica forwarding to the stub LLM. Seams for a machine: build Services on the Dragon backend, pass
`store` (a RadexStore) and `replicas` (one ServiceModel per model server, each around its own
llm.OpenAIChat), so the fleet routes and admits calls across the servers.
"""
from __future__ import annotations

import asyncio
import json
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
        text = await self.service(caller, [{"role": ROLE[m.type], "content": m.content} for m in messages],
                                  extra_body["thinking_token_budget"])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])


class EvaluateArgs(BaseModel):
    candidate: dict[str, float]
    seed: int


class State(TypedDict):
    messages: Annotated[list, add_messages]
    agent: dict


def _graph(svc, knobs: dict, fleet: Fleet, tools: list, saver: KeyValueSaver):
    async def ask(st: dict, messages: list[dict]) -> str:
        b, aid = st["budget"], st["agent_id"]
        lc = [SystemMessage(m["content"]) if m["role"] == "system" else HumanMessage(m["content"]) for m in messages]
        reply = await fleet.ainvoke(aid, lc, caller=aid, extra_body={"thinking_token_budget": b},
                                    max_tokens=b + REPLY_TOKENS, reserve=b, meta={"budget": b})
        return str(reply.content)

    async def start(s: State, config: RunnableConfig) -> dict:                 # steps 1 and 2
        st = decide.new_state(config["configurable"]["thread_id"], s["agent"]["seed"], knobs)
        return {"agent": await decide.look(st, svc, decide.lookups(knobs))}

    async def propose(s: State) -> dict:                                       # step 3, then step 4's call
        st = s["agent"]
        st = decide.proposed(st, await ask(st, decide.proposal_messages(st)))
        call = {"name": "evaluate", "id": f"sim{st['round']}",
                "args": {"candidate": st["pending"]["candidate"], "seed": st["seed"]}}
        return {"agent": st, "messages": [AIMessage("", tool_calls=[call])]}

    async def absorb(s: State) -> dict:                                        # step 5
        reply = s["messages"][-1]
        if getattr(reply, "status", "success") == "error":
            raise RuntimeError(str(reply.content))
        st = decide.absorb(s["agent"], json.loads(reply.content))
        await svc.remember(st["agent_id"], st)
        return {"agent": st, "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    async def summarise(s: State) -> dict:                                     # steps 7 and 8
        st = s["agent"]
        result = decide.finish(st, await ask(st, decide.summary_messages(st)))
        await svc.report(st["agent_id"], result)
        return {"agent": {"result": result}}

    def more(s: State) -> str:                                                 # step 6
        return "propose" if s["agent"]["round"] < s["agent"]["sims"] else "summarise"

    g = StateGraph(State)
    for name, node in (("start", start), ("propose", propose), ("tools", ToolNode(tools, handle_tool_errors=True)),
                       ("absorb", absorb), ("summarise", summarise)):
        g.add_node(name, node)
    g.add_edge(START, "start")
    g.add_edge("start", "propose")
    g.add_conditional_edges("propose", route_tool_calls, ["tools", END])
    g.add_edge("tools", "absorb")
    g.add_conditional_edges("absorb", more, ["propose", "summarise"])
    g.add_edge("summarise", END)
    return g.compile(checkpointer=saver)


def _live(states) -> tuple[int, int, int, int]:
    c = states.counts()
    return (c[AgentState.WORKING] + c[AgentState.TOOL_CALL], c[AgentState.MODEL_QUEUED] + c[AgentState.TOOL_WAIT],
            c[AgentState.MODEL_REQUEST], c[AgentState.DONE])


async def run_agents(n_agents: int, knobs: dict, svc_factory, out, *, first: int = 0, emit=None,
                     store=None, replicas=None) -> list[dict]:
    svc = svc_factory()
    runtime = ToolRuntime(svc.session)
    evaluate = task_tool(svc.sim_fn, suspend=True, name="evaluate", args_schema=EvaluateArgs,
                         description="Evaluate one candidate profile with the turned-down IGNOS inner loop.",
                         prepare=lambda agent_id, call_id, args: {
                             **args, "knobs": decide.sim_knobs(knobs, args["seed"]), "agent_id": agent_id})
    a = knobs["agents"]
    # Every agent process admits its share of each replica's caps (FlowGentic's static split).
    fleet = Fleet(replicas or [ServiceModel(service=svc.llm)], admission=ADMISSION,
                  share=min(1.0, a["per_process"] / a["count"]))
    saver = KeyValueSaver(MemoryStore() if store is None else store)
    sims = a["sims_per_agent"]
    graph = _graph(svc, knobs, fleet, bind_tools(runtime, [evaluate]), saver)
    results, suspensions = [], [0]

    def on_done(r, output) -> None:
        suspensions[0] += r.suspensions
        fleet.agent_done(r.id)
        results.append(output["agent"]["result"] if r.status == "done"
                       else {"agent_id": r.id, "error": r.error or r.status})

    host = AgentHost(GraphRunner(graph, config={"recursion_limit": 2 * sims + 25}),
                     max_live_agents=n_agents, stagger=STAGGER, on_done=on_done)
    tick = asyncio.create_task(record.ticker(lambda: _live(host.states), emit))
    try:
        report = await host.run_indexed(range(first, first + n_agents), lambda i: {"agent": {"seed": i}},
                                        id_prefix="a")
    finally:
        tick.cancel()
        await asyncio.gather(tick, return_exceptions=True)
    for r in results:
        if "error" in r:   # a failed agent never reached its report step
            await svc.report(r["agent_id"], r)
    svc.sim_tasks += runtime.stats()["submitted"]
    f = fleet.stats()
    svc.layer = {"layer": "langgraph", "suspensions": suspensions[0], "outcomes": report.outcomes,
                 "host_wall_s": report.wall_s, "share_by_category": report.states["share_by_category"],
                 "loop_lag": report.loop_lag, "tasks": runtime.stats(), "checkpoints": saver.stats(),
                 "fleet": {k: f[k] for k in ("replicas", "calls", "in_flight_max", "errors")}}
    return results
