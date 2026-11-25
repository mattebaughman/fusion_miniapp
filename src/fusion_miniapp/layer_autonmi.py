"""The third agent layer: fusion's campaigns on AUTONMI's framework, as a workflow mini-app.

    python -m fusion_miniapp.layer_autonmi --preset smoke --sim fixture --agents 100 --populations 2

    # under Dragon: populations as Dragon processes, simulations on RHAPSODY's native backend, RaDEX
    python -m dragon -s src/fusion_miniapp/layer_autonmi.py --runtime dragon --populations 2 --workers 16 ...

Every campaign is the same three roles as in the other two layers (decide.py's Coordinator, Generator
and Evaluator), with Monitor slots that run the Evaluator's inner loops, and the parent's whole-generation
selection on the inner loop's numbers alone, so its result is the same: `results_sha256` equals the other
layers' for the same knobs, at any campaign.slots. With campaign.mode async (this layer only) there is no
generation barrier, and each agent makes its own model call (unless llm.mode is off): each applies each result as
it arrives (Script), so results differ from run to run and the checks are invariants (verify). A model call of
the roles' own is tried up to TRIES times, then the role decides on numbers alone (its fallback, counted); so are
the generations mode's refill and summary, whose fallbacks keep the parent's mutants and leave the summary out (the
other layers end the campaign at the first failed call). What runs them is AUTONMI's:
- The mini-app is an AsyncFlow application. Its campaigns are split into populations (`--populations`;
  by default one per agent process of run.py, or the next count up that divides the campaigns: layout),
  each an AsyncFlow node with an engine of its own for its simulations, a fleet for its model calls, and
  Flowgentic's agent host. Locally the populations run in this process; under Dragon (`--runtime dragon`)
  each is a Dragon process, its simulations run on RHAPSODY's native Dragon backend, and the store is RaDEX.
- A campaign is 3 + campaign.slots AUTONMI scripted agents (`Script`): a<i>/coordinator, a<i>/generator,
  a<i>/evaluator and the Monitor slots a<i>/monitor0, a<i>/monitor1, ..., each with its own state, model
  calls, lookups and row, talking only through messages (`Mailbox`, one per population). The Generator puts
  each generation's candidates on the campaign's candidate queue; a free Monitor takes the next one, runs
  its inner loop as one suspending `evaluate` task tool (saved and dropped until it ends), scores its trace
  (monitor.py) and reports to the Evaluator, which passes the generation's results on in candidate order:
  the parent's generation barrier, as in the LangGraph layer. The Generator's model call is awaited inside
  its step, as decide.generator_step is written. All stay live until their campaign ends; with
  agents.start_together none begins generation 0 until every agent of its population has started. Every id
  seen outside the agents is the campaign's, a<i>.
- Model calls go through Flowgentic's fleet: fusion's stub LLM as a transport, or (`--llm openai`) one
  Flowgentic `HTTPTransport` per server of the model job (`--endpoints-file`), which waits out a server
  that goes away instead of retrying (`llm.max_retries` does not apply). The fleet admits at most
  `llm.max_in_flight` requests and `llm.admission_tokens` tokens per server at once, split over the
  populations. Before any campaign, each server must list `llm.model` in its /v1/models; its vLLM
  /metrics are sampled through the run (AUTONMI's VllmSampler, into vllm/vllm.json), unless
  --no-vllm-metrics.
- The head process prints one line to stderr per record.ticker interval (LiveLine): agents live, model
  requests open and queued at the client, sequences decoding at the model servers, agents done. The
  record keeps the same values as a series, and the counts' maxima, means and times at or above shares of
  the hosted agents (`speed`; write_record), and the model calls' prompt and generated tokens per second over
  the steady window (record.write).

It writes what run.py writes, with `agent_layer: autonmi`: `run_record.json` (record.write, with an
`autonmi` section), `agents.jsonl` and `campaigns/<id>/`; and AUTONMI's own folders (`launcher/`,
`populations/`, `app/`, `otel/`) and `summary.json`, its checks of the run's telemetry. Each population
streams its campaigns' lines to `agents-p<index>.jsonl`, as each agent process of run.py does, and
record.write merges them. Exit status as run.py's: 0 when the record passes and so do AUTONMI's checks,
1 when not, 143 when SIGTERM or Ctrl-C interrupted the run.

Needs the AUTONMI framework (the package `autonmi`, installed separately) and Flowgentic 0.6
(requirements-autonmi.txt).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import importlib
import json
import multiprocessing as mp
import os
import re
import shutil
import signal
import statistics
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from array import array
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np
from autonmi import AgentSpec, Campaign, CampaignConfig, Check
from autonmi.population import PopulationConfig
from autonmi.scripted import Done, Lookup, Remember, Report, ScriptedAgent, Simulate
from flowgentic.fleet import Reply, Usage
from flowgentic.host import AgentState, activity, current_agent
from flowgentic.tools import task_tool
from pydantic import BaseModel
from radical.asyncflow import WorkflowEngine
from rhapsody.backends import ConcurrentExecutionBackend
from wfMiniAPI.api import MiniApp

from fusion_miniapp import decide, knobs as knob_files, monitor, record, run, tools, trace
from fusion_miniapp.llm import REPLY_TOKENS, StubLLM

# The AUTONMI framework's commit this layer is tested on: origin/MTU-188-framework (its PR #16). An editable
# install runs whatever its clone has checked out (MTU-188-retry, for one, turns on reruns), so the run record
# names the commit imported, and a run on another commit, or on changed files, warns once (autonmi_pin).
AUTONMI_COMMIT = "e20861a8cf5d996c4b1fb332f57d7758bfeb4c75"
# Under `python -m dragon -s .../layer_autonmi.py` (and `python -m`) this file runs as __main__. The
# definition and the simulation pools' factory are taken from the module of this name instead, so the
# populations' processes can unpickle them, and so _POOLS is the list the factory fills.
MODULE = "fusion_miniapp.layer_autonmi"
_POOLS: list = []   # this process's simulation pools, told to stop rebuilding on SIGTERM
_GPUS: list = []    # workers.gpu_per_worker: the GPUs this process's inner-loop workers are placed on (inner_loop_gpus)
_GPU_COUNTER = None   # the workers placed so far, one counter for every population's pool here (sim_pool)
_OWN_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)   # the run's, not its engines' (run_until_signalled)


class _RunningSet(set):
    """AsyncFlow's set of running components, whose iteration yields only the members that have completed.

    AsyncFlow's engine loop finds completions by checking every running component's future on every pass
    (`for comp_uid in list(self.running)`, workflow_manager.py:1679-1684 at radical.asyncflow 3f18cc8): a cost
    that grows with the square of a run's tasks (2,000 campaigns: 209.5 s, against 82 s with this set; 10,000:
    2,039 s against 403 s; the same results). The loop iterates `running` only there, and only to find the
    completed ones, so a set that records completions by done-callback gives it the same answer. Remove once
    AsyncFlow tracks completions itself. FUSION_ASYNCFLOW_RUNNING_SHIM=0 turns it off."""

    def __init__(self, engine, members=()) -> None:
        super().__init__()
        self._engine, self._done = engine, set()
        for uid in members:
            self.add(uid)

    def add(self, uid) -> None:
        super().add(uid)
        self._engine.components[uid]["future"].add_done_callback(lambda _f, uid=uid: self._done.add(uid))

    def discard(self, uid) -> None:
        super().discard(uid)
        self._done.discard(uid)

    def clear(self) -> None:
        super().clear()
        self._done.clear()

    def __iter__(self):
        return iter([uid for uid in self._done if set.__contains__(self, uid)])


def _install_running_shim() -> bool:
    """Give every AsyncFlow engine of this process a _RunningSet (its `running` must be a plain set after
    __init__, or the engine keeps it, with one warning). Idempotent: this file runs as __main__ and as MODULE."""
    if os.environ.get("FUSION_ASYNCFLOW_RUNNING_SHIM", "1") == "0":
        return False
    engine_cls = WorkflowEngine
    if getattr(engine_cls, "_fusion_running_shim", False):
        return True
    init, warned = engine_cls.__init__, []

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if type(getattr(self, "running", None)) is set and isinstance(getattr(self, "components", None), dict):
            self.running = _RunningSet(self, self.running)
        elif not warned:
            warned.append(True)
            print("[fusion/autonmi] AsyncFlow's engine has no plain `running` set: its scan is left as it is",
                  file=sys.stderr, flush=True)

    engine_cls.__init__ = __init__
    engine_cls._fusion_running_shim = True
    return True


RUNNING_SHIM = _install_running_shim()   # recorded in the run record's `autonmi` section

FULL_GC_THRESHOLD = 10**9   # CPython's third threshold: full collections after this many generation-1 collections


def full_gc_auto() -> bool:
    """Whether CPython's automatic full collections stay on: FUSION_GC_FULL_AUTO=1 keeps them, as CPython has them."""
    return os.environ.get("FUSION_GC_FULL_AUTO", "0") == "1"


def bound_full_collections() -> tuple[int, int, int]:
    """Turn off CPython's automatic full (generation 2) collections in this process, unless full_gc_auto().

    While agents run, each full collection scans every live object, so its pause grows with the run: at 10,000
    campaigns four of them stalled the event loop for up to 6.7 s each (measured on a laptop; see the README).
    Flowgentic's host raises only the first threshold and keeps the other two it finds when its first host
    starts, so this runs before that, in each population's process (Fusion.build). Explicit collections still
    run, Flowgentic's freeze included; young objects are still collected as before. main() puts the thresholds
    back after the run."""
    t = gc.get_threshold()
    if not full_gc_auto() and t[2] < FULL_GC_THRESHOLD:
        gc.set_threshold(t[0], t[1], FULL_GC_THRESHOLD)
    return gc.get_threshold()

ROLES = ("coordinator", "generator", "evaluator", "monitor")   # a campaign's roles; agent_names gives its agents
FAIL_ENV = "FUSION_AUTONMI_FAIL"   # tests only: "<agent id>:<g>" fails that agent before its generation-g message
                                   # (a Monitor's g is its candidate's call id, g<g>c<i>: before that report; in
                                   # the asynchronous mode g is the agent's cycle from 0, a Monitor's call id c<k>)
QUEUE = "candidates"   # each campaign's candidate queue: the Generator puts, a free Monitor takes the next
TRIES = 3   # tries of a model call (an error, or for a role's own call a reply that cannot be used) before its fallback


def agent_names(knobs: dict) -> list[str]:
    """A campaign's hosted agents, a<i>/<name>, in this order: its three outer-loop roles, then its Monitor
    slots monitor0 .. monitor<slots - 1>."""
    return [*ROLES[:3], *(f"monitor{k}" for k in range(knobs["campaign"]["slots"]))]


def agents_per_campaign(knobs: dict) -> int:
    return 3 + knobs["campaign"]["slots"]


def campaign_of(agent_id: str) -> str:
    """The campaign of a hosted agent (a3/generator -> a3): the id seen everywhere outside the agents."""
    return agent_id.partition("/")[0]


def role_of(agent_id: str) -> str:
    """A hosted agent's role (a3/monitor1 -> monitor)."""
    return agent_id.rpartition("/")[2].rstrip("0123456789")


def messages(knobs: dict) -> dict:
    """The messages each role sends and receives in one campaign of G generations of P candidates, with S
    Monitor slots (the Monitors' together): each generation the Coordinator sends make/g, the Generator
    population/g and the generation's P candidates on the campaign's queue, each Monitor that takes one a
    report, and the Evaluator results/g; at the end the Coordinator sends done to the Generator and the
    Evaluator, and the Generator puts one stop on the queue for each Monitor. 3G + 2 + 2PG + S in all.

    In the asynchronous mode the counts depend on the order of arrival, so they are bounds [low, high] where
    they vary. With B = P x G candidates: the Generator puts B candidates and S stops; each Monitor reports
    each candidate it takes; each Evaluator cycle (1 to B of them, E) sends the Coordinator its scored results
    and the Generator a digest; each Coordinator cycle (1 to E of them, C) sends the Generator the survivors,
    then (with its model call, decide.role_calls) its guidance, and at the end done to the Generator and the
    Evaluator. So with k = 2 messages a cycle with the call and 1 without, the Coordinator sends kC + 2 (k + 2 to
    kB + 2) and receives E (1 to B), the Generator receives kC + E + 1 (k + 2 to (k + 1)B + 1), the Evaluator
    sends 2E (2 to 2B)."""
    G, P, S = knobs["outer"]["generations"], knobs["outer"]["population"], knobs["campaign"]["slots"]
    if decide.asynchronous(knobs):
        B, k = P * G, 1 + decide.role_calls(knobs)
        return {"coordinator": {"sent": [k + 2, k * B + 2], "received": [1, B]},
                "generator": {"sent": B + S, "received": [k + 2, (k + 1) * B + 1]},
                "evaluator": {"sent": [2, 2 * B], "received": B + 1},
                "monitor": {"sent": B, "received": B + S}}
    return {"coordinator": {"sent": G + 2, "received": 2 * G},
            "generator": {"sent": G + P * G + S, "received": G + 1},
            "evaluator": {"sent": G, "received": P * G + 1},
            "monitor": {"sent": P * G, "received": P * G + S}}


class CampaignFailed(RuntimeError):
    """Another agent of the campaign failed: this one ends with an error instead of waiting for it."""


class InjectedFailure(RuntimeError):
    """FAIL_ENV's failure (tests only)."""


class PromptTooLong(RuntimeError):
    """A prompt and its max_tokens do not fit the model servers' context: the call is not sent."""


class _Queue:
    """A campaign's queue (or an agent's inbox): its items in the order put, its takers waiting, in the order
    they asked, and an agent waiting for it to hold no item (`emptied`)."""

    __slots__ = ("items", "waiting", "emptied")

    def __init__(self) -> None:
        self.items: deque[str] = deque()
        self.waiting: deque[asyncio.Future] = deque()
        self.emptied: asyncio.Future | None = None


def _owner(key: str) -> str:
    """Who a queue's leftovers count for: an inbox <campaign>/<agent>/<kind> its agent's, a campaign's queue
    <campaign>/<kind> its own key."""
    return key.rsplit("/", 1)[0] if key.count("/") == 2 else key


class Mailbox:
    """The messages between one population's agents (locally every population runs in one process, so one
    per population, in pop.state). A message is a JSON copy, as between processes, kept under its exact key
    <campaign>/<to>/<kind>/<g> in an asyncio future: `send` resolves it (making it, when the receiver has not
    asked yet), `receive` awaits it and deletes it. A second send of a key not yet received raises (a send
    after its receipt would stay in the box, which rows() reports). Beside the exact keys, each campaign has
    queues <campaign>/<kind>: `put` adds an item, `take` gives the next one to whichever agent asks (the
    first that waits), so the sender addresses no one, and `emptied` waits until none is left on it (`most`:
    the most items it held at once). Each agent has inboxes <campaign>/<agent>/<kind> (the asynchronous
    mode): `deliver` adds a message, `take_all` takes every one that has arrived (the first `limit` of them
    when given), in the order sent, waiting for one if there is none (or, with wait=False, none). A waiting
    agent awaits inside its own step, counted by Flowgentic as waiting on a tool (TOOL_WAIT), and is woken
    once, never polled. `fail(campaign)` fails that campaign's waits, now and later, and drops its messages
    not yet received (counted per receiver in `dropped`, a campaign queue's under its own key); a message that
    has already woken its receiver is that receiver's, received and not dropped. With agents.start_together,
    `all_started` is set when every agent of the population has started."""

    def __init__(self, agents: int) -> None:
        self.agents, self.started, self.all_started = agents, 0, asyncio.Event()
        self.box: dict[str, asyncio.Future] = {}
        self.asked: set[str] = set()          # exact keys a receive waits on
        self.queues: dict[str, _Queue] = {}   # <campaign>/<kind> and inboxes, while it holds items or waits
        self.most: Counter = Counter()        # per queue: the most items it held at once
        self.failed: dict[str, str] = {}   # campaign -> why
        self.sent, self.received, self.waited = Counter(), Counter(), Counter()   # per agent; waited: seconds
        self.dropped = Counter()   # per receiving agent (or queue): messages not received when its campaign failed

    def start(self) -> None:
        self.started += 1
        if self.started == self.agents:
            self.all_started.set()

    def _future(self, key: str) -> asyncio.Future:
        f = self.box.get(key)
        if f is None:
            f = self.box[key] = asyncio.get_running_loop().create_future()
        return f

    def send(self, sender: str, to: str, kind: str, g: int, value) -> None:
        campaign = campaign_of(sender)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        key = f"{campaign}/{to}/{kind}/{g}"
        f = self._future(key)
        if f.done():
            raise RuntimeError(f"message {key} sent twice")
        f.set_result(json.dumps(value, default=str))   # as the LangGraph layer reads a task's result
        self.sent[sender] += 1

    async def receive(self, me: str, kind: str, g: int):
        campaign = campaign_of(me)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        key, t = f"{me}/{kind}/{g}", time.perf_counter()
        self.asked.add(key)
        try:
            with activity(AgentState.TOOL_WAIT):
                text = await self._future(key)
        finally:
            self.asked.discard(key)
            self.waited[me] += time.perf_counter() - t
            self.box.pop(key, None)
        self.received[me] += 1
        return json.loads(text)

    def put(self, sender: str, kind: str, value) -> None:
        """An item on the sender's campaign's queue <campaign>/<kind>: to its first waiting taker, else kept."""
        campaign = campaign_of(sender)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        key = f"{campaign}/{kind}"
        q = self.queues.get(key) or self.queues.setdefault(key, _Queue())
        text = json.dumps(value, default=str)
        while q.waiting:
            f = q.waiting.popleft()
            if not f.done():   # (one cancelled, as when the run is interrupted, is passed over)
                f.set_result(text)
                break
        else:
            q.items.append(text)
            if len(q.items) > self.most[key]:
                self.most[key] = len(q.items)
        self.sent[sender] += 1

    async def take(self, me: str, kind: str):
        """The next item of my campaign's queue <campaign>/<kind>, waiting for one if there is none."""
        campaign = campaign_of(me)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        key, t = f"{campaign}/{kind}", time.perf_counter()
        q = self.queues.get(key) or self.queues.setdefault(key, _Queue())
        try:
            if q.items:
                text = q.items.popleft()
                if not q.items and q.emptied is not None and not q.emptied.done():
                    q.emptied.set_result(None)
            else:
                f = asyncio.get_running_loop().create_future()
                q.waiting.append(f)
                with activity(AgentState.TOOL_WAIT):
                    text = await f
        finally:
            self.waited[me] += time.perf_counter() - t
        if not q.items and not q.waiting and self.queues.get(key) is q:   # empty: kept only while in use
            del self.queues[key]
        self.received[me] += 1
        return json.loads(text)

    async def emptied(self, me: str, kind: str) -> None:
        """Wait until no item is left on my campaign's queue <campaign>/<kind> (at once if none is)."""
        campaign = campaign_of(me)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        q = self.queues.get(f"{campaign}/{kind}")
        if q is None or not q.items:
            return
        q.emptied, t = asyncio.get_running_loop().create_future(), time.perf_counter()
        try:
            with activity(AgentState.TOOL_WAIT):
                await q.emptied
        finally:
            self.waited[me] += time.perf_counter() - t

    def deliver(self, sender: str, to: str, kind: str, value) -> None:
        """A message for agent <campaign>/<to>'s inbox of `kind` (take_all)."""
        self.put(sender, f"{to}/{kind}", value)

    async def take_all(self, me: str, kind: str, wait: bool = True, limit: int | None = None) -> list:
        """Every message in my inbox of `kind`, in the order sent (at most `limit`: the rest stay for the next
        take); if there is none, wait for one (or, with wait=False, none)."""
        campaign = campaign_of(me)
        if campaign in self.failed:
            raise CampaignFailed(self.failed[campaign])
        key, texts = f"{me}/{kind}", []
        q = self.queues.get(key)
        if (q is None or not q.items) and wait:
            q = q or self.queues.setdefault(key, _Queue())
            f, t = asyncio.get_running_loop().create_future(), time.perf_counter()
            q.waiting.append(f)
            try:
                with activity(AgentState.TOOL_WAIT):
                    texts.append(await f)
            finally:
                self.waited[me] += time.perf_counter() - t
        if q is not None:
            n = len(q.items) if limit is None else max(0, min(len(q.items), limit - len(texts)))
            texts += [q.items.popleft() for _ in range(n)]
            if not q.items and not q.waiting and self.queues.get(key) is q:
                del self.queues[key]
        self.received[me] += len(texts)
        return [json.loads(x) for x in texts]

    def fail(self, campaign: str, why: str) -> None:
        if campaign in self.failed:
            return
        self.failed[campaign] = why
        for key in [k for k in self.box if k.startswith(campaign + "/")]:
            f = self.box.pop(key)
            if not f.done():   # an agent waits on it
                f.set_exception(CampaignFailed(why))
            elif key not in self.asked:   # sent, never received (one that woke its receiver is received)
                self.dropped[key.rsplit("/", 2)[0]] += 1
        for key in [k for k in self.queues if k.startswith(campaign + "/")]:
            q = self.queues.pop(key)
            for f in (*q.waiting, q.emptied):
                if f is not None and not f.done():
                    f.set_exception(CampaignFailed(why))
            if q.items:   # dropped, and cleared: a take_all woken before the failure does not also receive them
                self.dropped[_owner(key)] += len(q.items)
                q.items.clear()

    def left(self) -> Counter:
        """Messages never received, per receiving agent; a campaign queue's items never taken under its key."""
        out = Counter(key.rsplit("/", 2)[0] for key in self.box)
        for key, q in self.queues.items():
            if q.items:
                out[_owner(key)] += len(q.items)
        return out


class EvaluateArgs(BaseModel):
    params: dict[str, float | str]     # the parent's full candidate, with its "model" and "kind" strings
    candidate_idx: int
    generation: int
    seed: int


class StubTransport:
    """Fusion's stub LLM as a Flowgentic transport. Its replies depend on the campaign (of the agent
    Flowgentic's host runs the call for) and the request's last line only, so the three layers agree (the
    Evaluator's on the q and m its prompt gives, at monitor.weight `weight`: llm.numbers_only).
    Each call takes `latency_s` (llm.stub_latency_s): an emulated model server's time, not its decoding."""

    model = "fusion-stub"

    def __init__(self, latency_s: float = 0.001, weight: float = 0.25) -> None:
        self.stub = StubLLM(latency_s=latency_s, weight=weight)

    async def complete(self, messages, tools=None, *, extra_body: dict | None = None, max_tokens: int | None = None,
                       **kwargs) -> Reply:
        b = int((extra_body or {}).get("thinking_token_budget", 0))
        r = max_tokens - b if max_tokens is not None else REPLY_TOKENS
        text, meta = await self.stub.call_with_meta(campaign_of(current_agent() or ""), list(messages), b,
                                                    reply_tokens=r)
        prompt = sum(len(str(m.get("content") or "")) for m in messages) // 4   # an estimate, for admission
        return Reply(content=text, finish_reason=meta["finish_reason"],
                     usage=Usage(prompt_tokens=prompt, completion_tokens=len(text) // 4))


def transport_settings(model: dict | None) -> dict:
    """What a fleet's transport sends with every call besides the call's own fields: for the model job's servers,
    AUTONMI's settings (autonmi.models.transport: the served model's name, temperature 0, max_tokens, and
    thinking_off as chat-template arguments in its extra_body); for the stub, its name alone (model None)."""
    if model is None:
        return {"model": StubTransport.model}
    out = {"model": model["name"], "temperature": 0, "max_tokens": int(model.get("max_tokens", 512))}
    if model.get("thinking_off"):
        out["extra_body"] = {"chat_template_kwargs": dict(model["thinking_off"])}
    return out


def request_fields(knobs: dict, budget: int, reply_tokens: int) -> dict:
    """A model call's request fields (Script._request; launch/probe_model.py sends the same): by default run.py's
    OpenAIChat's, max_tokens = thinking budget + reply allowance, vLLM's thinking_token_budget (unless
    llm.send_thinking_budget is false), and no temperature (AUTONMI's transports set 0; None leaves the server's
    default). The llm knobs change them: with llm.reply_cap, max_tokens is the cap plus the budget (the cap alone
    with llm.thinking off); llm.thinking off sends no budget (the transports send chat_template_kwargs,
    miniapp_workflow); llm.temperature and llm.top_p are sent as set. No per-call extra_body otherwise: one would
    replace the transports' own (flowgentic/fleet/_transport.py:555-556). Admission reserves the budget and the
    allowance (`reserve`)."""
    k = knobs["llm"]
    cap = budget + reply_tokens if k["reply_cap"] is None else k["reply_cap"] + (0 if k["thinking"] == "off"
                                                                                  else budget)
    fields = {"max_tokens": cap, "reserve": budget + reply_tokens, "temperature": k["temperature"]}
    if k["top_p"] is not None:
        fields["top_p"] = k["top_p"]
    if k.get("send_thinking_budget", True) and k["thinking"] != "off":
        fields["extra_body"] = {"thinking_token_budget": budget}
    return fields


def sent(transport: dict, kwargs: dict) -> dict:
    """A request's settings, its body's fields besides the messages, as Flowgentic's HTTPTransport.complete makes
    them: the transport's own (transport_settings), then the call's (`kwargs`, without the fleet's reserve and
    meta), a call's extra_body replacing the transport's; then the extra_body's keys, put into the body."""
    body = {**transport, **{k: v for k, v in kwargs.items() if k not in ("reserve", "meta")}}
    body.update(body.pop("extra_body", None) or {})
    return body


class CallSums:
    """One population's model calls whose details are not in a campaign's line (llm_call_details): the roles' own
    calls without record.extras (counts only by default), and a call that ends after its campaign's line was
    written (it failed). Summed here instead, for the record's model figures (llm_service, the client-side
    counts of requests in flight and queued, and the tokens per second, that record.write makes from each call's
    times and tokens), and written at the population's end beside its campaign lines (`write`, calls-p<index>.npz),
    where part() reads them (`read`); write_record removes the file once the record is written. With
    record.extras it also counts the requests sent, by role and settings (`requests`: Script._ask)."""

    def __init__(self) -> None:
        self.n = Counter()   # calls, errors, empty, retried
        self.budgets, self.finish_reasons, self.tokens = Counter(), Counter(), Counter()
        self.latency = array("d")   # every call's latency_s
        self.times = array("d")     # each answered call's t_end, service_s, recovery_s, wait_s, prompt and
                                    # completion tokens
        self.requests = Counter()   # record.extras: JSON of {"role", "body": the request's settings} -> requests

    def add(self, d: dict) -> None:
        """One call's details, as on_call and _ask make them."""
        self.n["calls"] += 1
        self.budgets[str(d["budget"])] += 1
        self.latency.append(d["latency_s"])
        if d.get("error"):
            self.n["errors"] += 1
            return
        self.n["empty"] += bool(d["empty"])
        self.n["retried"] += bool(d["recovery_s"])
        self.finish_reasons[d["finish_reason"]] += 1
        for k in ("prompt", "completion", "reasoning"):
            self.tokens[k] += d[f"{k}_tokens"] or 0
        self.times.extend((d["t_end"], d["service_s"], d["recovery_s"], d["wait_s"], d["prompt_tokens"] or 0,
                           d["completion_tokens"] or 0))

    def write(self, path: Path) -> None:
        """To `path` (none without a call or a request counted)."""
        if self.n["calls"] or self.requests:
            counts = {"n": self.n, "budgets": self.budgets, "finish_reasons": self.finish_reasons,
                      "tokens": self.tokens, "requests": self.requests}
            np.savez(path, counts=json.dumps(counts), latency=np.array(self.latency, dtype=float),
                     times=np.array(self.times, dtype=float).reshape(-1, 6))

    @staticmethod
    def read(path: Path) -> dict | None:
        """What `write` wrote: the counts, `latency_s` (a list) and `times` (an array of rows t_end, service_s,
        recovery_s, wait_s, prompt and completion tokens); None without the file."""
        if not path.exists():
            return None
        with np.load(path) as z:
            return {**json.loads(str(z["counts"])), "latency_s": z["latency"].tolist(), "times": z["times"]}


def sums_file(lines_file: Path, index: int) -> Path:
    """The CallSums file of population `index`, beside its campaign lines."""
    return lines_file.with_name(f"calls-p{index:03d}.npz")


def sim_pool(workers: int):
    """The executor of a population's real evaluations (AUTONMI's PopulationConfig.executor): spawned
    worker processes, as run.py's, which rebuild once when a worker dies and stop within run.SHUTDOWN_S. With
    GPUs to place them on (`_GPUS`, inner_loop_gpus), each worker takes the next number i from one counter that
    every population's pool in this process shares, and sees GPU i mod their count alone (pool.on_gpu)."""
    global _GPU_COUNTER
    from fusion_miniapp.pool import ResilientProcessPool, on_gpu

    spawn, init, args = mp.get_context("spawn"), run._worker_init, ()
    if _GPUS:
        if _GPU_COUNTER is None:
            _GPU_COUNTER = spawn.Value("i", 0)
        init, args = on_gpu, (_GPU_COUNTER, list(_GPUS), run._worker_init)
    pool = ResilientProcessPool(workers, mp_context=spawn, initializer=init, initargs=args,
                                shutdown_timeout=run.SHUTDOWN_S)
    _POOLS.append(pool)
    return pool


def inner_loop_gpus(knobs: dict, device: dict | None) -> list[str]:
    """workers.gpu_per_worker: the GPUs to place the inner-loop workers on, one each: the n GPUs the device probe
    saw where the inner loops run (CUDA_VISIBLE_DEVICES's first n entries when it is set, else 0 .. n - 1); none
    without the knob or without a GPU (also the fixture and Dragon, which have no probe here)."""
    n = len((device or {}).get("gpus") or [])
    if not knobs["workers"]["gpu_per_worker"] or not n:
        return []
    seen = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    return (seen or [str(k) for k in range(n)])[:n]


class FixturePool(run.FixturePool):
    """run.py's fixture threads, each running the simulations' coroutines on one event loop of its own.

    Flowgentic makes every simulation a coroutine (AsyncFlow runs async tasks only), and RHAPSODY's thread
    backend runs each one with asyncio.run (`_run_in_thread`). On Python 3.12 asyncio.run ends by collecting
    every task of the process (asyncio.all_tasks walks one process-wide set), the live campaigns' included:
    a cost per evaluation that grows with the campaigns, so a run grew about with their square (2,000
    campaigns in the hundredk shape: 2,053 s, against 180 s on one loop per thread). RHAPSODY's own fix
    (reuse_event_loop, radical-cybertools/rhapsody#163) is not released; once it is, PopulationConfig(
    reuse_event_loop=True) replaces this class. Every other call is submitted unchanged, and so is this one
    if RHAPSODY stops making it. The fixture's coroutines leave no tasks behind on the loop."""

    def __init__(self, workers: int) -> None:
        super().__init__(workers)
        self._local = threading.local()
        self._loops: list = []
        self._loops_lock = threading.Lock()

    def submit(self, fn, /, *args, **kwargs):
        if fn is ConcurrentExecutionBackend._run_in_thread and len(args) == 3 and not kwargs \
                and asyncio.iscoroutinefunction(args[0]):   # RHAPSODY's own test, so both agree
            return super().submit(self._on_thread_loop, *args)
        return super().submit(fn, *args, **kwargs)

    def _on_thread_loop(self, func, args, kwargs):
        loop = getattr(self._local, "loop", None)
        if loop is None:
            loop = self._local.loop = asyncio.new_event_loop()
            with self._loops_lock:
                self._loops.append(loop)
        return loop.run_until_complete(func(*args, **kwargs))

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait:   # the threads have ended: their loops can close
            with self._loops_lock:
                loops, self._loops = self._loops, []
            for loop in loops:
                loop.close()


def fixture_pool(workers: int):
    """The executor of a population's fixture evaluations: run.py's threads, which drop their queued work
    when they close (an interrupted run can hold tens of thousands of queued evaluations), each with an
    event loop of its own (FixturePool)."""
    return FixturePool(workers)


class Fusion:
    """Fusion as an AUTONMI agent definition (autonmi.AgentDefinition): its campaigns, how a population
    runs them (Script), their rows and the checks on them."""

    name = "fusion"

    def __init__(self, knobs: dict, sim: str, out: Path, work_root: str | None = None) -> None:
        self.knobs, self.sim, self.out, self.work_root = knobs, sim, Path(out), work_root
        self.populations: dict[str, asyncio.Task] = {}   # p000...: the task running each population here (build)
        self.live: Counter = Counter()   # agents started and ended in this process (LiveLine)
        self.context: int | None = None   # the model servers' context (max_model_len), for the prompt-length check
        self.transport = transport_settings(None)   # what the transports send with every call (record.extras)

    def agents(self):
        """Each campaign's agents (agent_names), one after another (AUTONMI cuts this list into contiguous
        populations, which layout() keeps on campaign boundaries)."""
        first, names = self.knobs["agents"]["first"], agent_names(self.knobs)   # as the other layers: a<i>, seed i
        for i in range(first, first + self.knobs["agents"]["count"]):
            for name in names:
                yield AgentSpec(f"a{i}/{name}", {"seed": i})

    def services(self, run_dir: Path, servers: int = 1) -> tuple[list, list[str]]:
        return [], []   # no MCP servers: the lookups run in the campaign's process (tools.lookup)

    def stub(self, replica: int) -> StubTransport:
        return StubTransport(self.knobs["llm"]["stub_latency_s"], self.knobs["monitor"]["weight"])

    def sim_knobs(self, seed: int) -> dict:
        """The `sim` knobs one campaign's evaluations get: its cost scale, and this run's own work folder."""
        s = decide.sim_knobs(self.knobs, seed)
        if self.work_root is not None:
            s["work_root"] = self.work_root
        return s

    def build(self, pop) -> ScriptedAgent:
        """Called in the task that runs the population (locally, a block of the run's engine), which is
        kept so that an interrupted run can wait for the population to end (miniapp_workflow). Before the
        population's host starts: the full collections' setting (bound_full_collections)."""
        bound_full_collections()
        self.populations[f"p{pop.index:03d}"] = asyncio.current_task()
        return Script(self, pop).agent()

    def rows(self, pop) -> list[dict]:
        """At the population's end, in its process: one row per hosted agent, with its role and campaign,
        whether it ended normally (`finished`; a Coordinator only once its campaign reported), the evaluations
        it submitted (a Monitor's, each of which ends before it goes on: AUTONMI checks them against the
        tasks its engine completed), its lookups, its messages sent, received, left unreceived and dropped
        when its campaign failed, its seconds waiting for messages, and its own model calls asked, tried and
        fallen back (`model_asks`, `model_tries`, `model_fallbacks`; AUTONMI's `model_calls` beside them counts
        the replies); the asynchronous Generator's also its proposals by the parameters its replies filled
        (`parameters_filled`, 0 to 7); the Coordinator's also its wait for
        every agent to start and when it began its campaign (`t_begin`, after that wait); a Monitor's also
        its slot, the inner-loop traces it got (monitor.capture) and the reports it sent. The campaign's
        candidate queue is its Monitors' inbox: what is left on it, or dropped, counts as its first slot's. The
        population's calls not in its campaigns' lines are written beside them (CallSums)."""
        sc = pop.state.get("script")
        if sc is None:   # the population ended before its agents were built
            return [{"agent_id": a, "finished": False, "simulations": 0, "population": pop.index}
                    for a in pop.agent_ids]
        sc.lines.close()
        sc.sums.write(sums_file(Path(sc.lines.name), pop.index))
        mail, left, out = sc.mail, sc.mail.left(), []
        for a in pop.agent_ids:
            campaign, role = campaign_of(a), role_of(a)
            row = {"agent_id": a, "campaign": campaign, "role": role, "finished": a in sc.finished,
                   "simulations": sc.sims.get(a, 0), "population": pop.index, "lookups": sc.lookups_made[a],
                   "messages_sent": mail.sent[a], "messages_received": mail.received[a], "messages_left": left[a],
                   "messages_dropped": mail.dropped[a], "message_wait_s": round(mail.waited[a], 6),
                   "model_asks": sc.asks[a], "model_tries": sc.tries[a], "model_fallbacks": sc.fallbacks[a]}
            if role == "coordinator":   # the campaign's model calls by its own count (decide.finish)
                row.update(llm_calls=sc.llm_calls.get(campaign), start_wait_s=sc.start_wait.get(a),
                           t_begin=sc.began.get(a))
            elif role == "monitor":
                slot = int(a.rpartition("/")[2][len(role):])
                row.update(slot=slot, traces=sc.traced[a], reports=sc.reports[a])
                if slot == 0:
                    queue = f"{campaign}/{QUEUE}"
                    row["messages_left"] += left[queue]
                    row["messages_dropped"] += mail.dropped[queue]
            if sc.asynchronous:   # each agent's cycles; the campaign's counts and invariants on its Coordinator's row
                row["cycles"] = sc.cycles[a]
                if role == "coordinator":
                    seen = sc.seen.get(campaign, 0)
                    row.update(sc.tally.get(campaign, {}), survivors_max=sc.survivors_max[campaign],
                               waiting_max=mail.most[f"{campaign}/{QUEUE}"], running_max=sc.running_max[campaign],
                               candidates_distinct=seen.bit_count(), candidate_max=seen.bit_length() - 1,
                               candidates_repeated=sc.repeated[campaign])
                elif role == "generator":
                    row.update(survivor_waits=sc.survivor_waits[a], parameters_filled=sc.filled.get(a, [0] * 8))
            out.append(row)
        return out

    def verify(self, rows: list[dict], store) -> list[Check]:
        """AUTONMI's checks of the campaigns' agents; the run record checks their results (touches, the model).
        Per role, exactly: evaluations, model calls (each agent's asked: the Coordinator's summary, the
        Generator's refills as many as its campaign counted, and with campaign.notes the Evaluator's assessment
        per generation and a Monitor's note per candidate, else none; each answered, or fallen back after at most
        TRIES tries, which AUTONMI's count of replies must match: calls_as_planned), lookups and messages, the
        Monitors' together per campaign (which slot takes which candidate
        is the queue's), and each Monitor one evaluation, one report and one message sent per candidate it
        took; every message received (none left); the lookups against AUTONMI's own count; and the store at
        the end: no agent's saved state (ScriptRunner's script/p<index>/<agent>, kept while it is
        suspended), and each reported campaign's memory under its campaign id, campaign/a<i>. In the
        asynchronous mode the counts are bounds, with that mode's invariants (_verify_async)."""
        n, exp, msgs = self.knobs["agents"]["count"], decide.expected_by_role(self.knobs), messages(self.knobs)
        coordinators = {r["campaign"]: r for r in rows if r.get("role") == "coordinator"}
        slots = self.knobs["campaign"]["slots"]
        if decide.asynchronous(self.knobs):
            return self._verify_async(rows, store, exp, msgs)

        notes = decide.role_calls(self.knobs)

        def planned(r: dict) -> bool:
            e, m = exp[r["role"]], msgs[r["role"]]
            if r["role"] == "monitor":   # its share, whichever candidates it took
                return (r["simulations"] == r["reports"] == r["messages_sent"] == r["messages_received"] - 1
                        and r["lookups"] == 0 and calls_as_planned(r, r["simulations"] if notes else 0))
            calls = {"coordinator": 1, "evaluator": e["llm_calls"],
                     "generator": (coordinators[r["campaign"]].get("llm_calls") or 0) - 1}[r["role"]]
            return (r["simulations"] == e["sims"] and calls_as_planned(r, calls) and calls <= e["llm_calls"]
                    and r["lookups"] == len(e["lookups"]) and (r["messages_sent"], r["messages_received"]) ==
                    (m["sent"], m["received"]))
        done = [r for r in rows if r["finished"]]
        off = [r["agent_id"] for r in done if not planned(r)]
        monitors: dict[str, list] = {}
        for r in done:
            if r["role"] == "monitor":
                monitors.setdefault(r["campaign"], []).append(r)
        e, m = exp["monitor"], msgs["monitor"]
        for rs in monitors.values():   # a campaign whose Monitors all finished: together, its P x G candidates
            total = [sum(r[k] for r in rs) for k in ("simulations", "messages_sent", "messages_received")]
            if len(rs) == slots and total != [e["sims"], m["sent"], m["received"]]:
                off += [r["agent_id"] for r in rs]
        return self._common_checks(rows, store, off, exp, msgs)

    def _common_checks(self, rows: list[dict], store, off: list[str], exp: dict, msgs: dict) -> list[Check]:
        """The checks of both modes, with `off`: the agents whose counts are not as planned."""
        n, done = self.knobs["agents"]["count"], [r for r in rows if r["finished"]]
        finished = {role: sum(r["finished"] for r in rows if r.get("role") == role) for role in ROLES}
        runners = [Path(self.out) / "populations" / f"p{i:03d}" / "population.json"
                   for i in sorted({r["population"] for r in rows})]
        autonmi = sum(json.loads(f.read_text())["runner"]["lookups"] for f in runners) \
            if all(f.exists() for f in runners) else None
        ours = sum(r.get("lookups", 0) for r in rows)
        saved = sum(store.contains(f"script/p{r['population']:03d}/{r['agent_id']}") for r in rows)
        reported = [r["campaign"] for r in done if r.get("role") == "coordinator"]
        memories = sum(store.contains(f"campaign/{c}") for c in reported)
        return [Check("every_campaign_reports", finished["coordinator"] == n,
                      {"reported": finished["coordinator"], "campaigns": n}),
                Check("every_agent_finishes", sum(finished.values()) == agents_per_campaign(self.knobs) * n,
                      {"finished": finished, "campaigns": n, "agents_per_campaign": agents_per_campaign(self.knobs)}),
                Check("evaluations_and_model_calls_as_planned", not off,
                      {"per_campaign": exp, "messages": msgs, "not_as_planned": off[:10]}),
                Check("every_message_received", not any(r.get("messages_left") for r in rows),
                      {"left": sum(r.get("messages_left", 0) for r in rows)}),
                Check("lookups_match_autonmi", ours == autonmi, {"rows": ours, "autonmi": autonmi}),
                Check("store_holds_only_the_memories", saved == 0 and memories == len(reported),
                      {"saved_states_left": saved, "memories": memories, "campaigns_reported": len(reported)})]

    def _verify_async(self, rows: list[dict], store, exp: dict, msgs: dict) -> list[Check]:
        """verify() in the asynchronous mode, where the counts depend on the order of arrival. Per agent that
        ended normally: its model calls (with decide.role_calls one asked per cycle, a Monitor's per candidate,
        each answered or fallen back: calls_as_planned; none with llm.mode off), its lookups, and its counts
        within messages()' bounds and tied to its cycles (the Coordinator sends 2 per cycle with its call, else
        1, + 2 and receives at least cycles; the Generator takes
        ceil(B / decide.TAKE) times, or once more when a take stops at candidate P for want of a survivor (an odd
        P), and puts B candidates and S stops; the Evaluator sends 2 per cycle and
        receives B reports and done; each Monitor one evaluation, report and message sent per candidate
        taken, one cycle each). Per campaign whose agents all finished: the Monitors' B evaluations, B reports
        and B + S messages received; the Generator received every survivors message and digest sent it and
        done, the Coordinator every scored message; each candidate 0 .. B - 1 evaluated exactly once; at most
        outer.population survivors; at most campaign.slots inner loops running; at most decide.TAKE candidates
        waiting on the queue. Then the checks of both modes."""
        o, S, take = self.knobs["outer"], self.knobs["campaign"]["slots"], decide.TAKE
        P, B, on = o["population"], o["population"] * o["generations"], decide.role_calls(self.knobs)
        takes = (-(-B // take), -(-P // take) + -(-(B - P) // take))   # the Generator's cycles: in one take, or split at P

        def within(x: int, bound) -> bool:
            return bound[0] <= x <= bound[1] if isinstance(bound, list) else x == bound

        def planned(r: dict) -> bool:
            role, m, c = r["role"], msgs[r["role"]], r.get("cycles", 0)
            if not calls_as_planned(r, c if on else 0) or r["lookups"] != len(exp[role]["lookups"]):
                return False
            if role == "monitor":
                return r["simulations"] == r["reports"] == r["messages_sent"] == r["messages_received"] - 1 == c
            if r["simulations"] != 0 or not (within(r["messages_sent"], m["sent"])
                                             and within(r["messages_received"], m["received"])):
                return False
            return {"coordinator": r["messages_sent"] == (1 + on) * c + 2 and 1 <= c <= r["messages_received"],
                    "generator": takes[0] <= c <= takes[1],
                    "evaluator": r["messages_sent"] == 2 * c and 1 <= c <= B}[role]
        done = [r for r in rows if r["finished"]]
        off = [r["agent_id"] for r in done if not planned(r)]
        by: dict[str, dict] = {}
        for r in rows:
            if "campaign" in r:   # (not a population that ended before its agents were built)
                by.setdefault(r["campaign"], {}).setdefault(r["role"], []).append(r)
        whole = {c: x for c, x in by.items() if all(r["finished"] for rs in x.values() for r in rs)
                 and all(role in x for role in ROLES) and len(x["monitor"]) == S}
        e, m = exp["monitor"], msgs["monitor"]
        once, survivors, running, waiting = [], [], [], []
        for c, x in whole.items():
            (co,), (ge,), (ev,), mons = x["coordinator"], x["generator"], x["evaluator"], x["monitor"]
            total = [sum(r[k] for r in mons) for k in ("simulations", "messages_sent", "messages_received")]
            if total != [e["sims"], m["sent"], m["received"]] or \
                    ge["messages_received"] != (co["messages_sent"] - 2) + ev["messages_sent"] // 2 + 1 or \
                    co["messages_received"] != ev["messages_sent"] // 2:
                off += [r["agent_id"] for rs in x.values() for r in rs]
            if not (co["candidates_distinct"] == B and co["candidate_max"] == B - 1 and co["candidates_repeated"] == 0):
                once.append(c)
            if co["survivors_max"] > o["population"]:
                survivors.append(c)
            if co["running_max"] > S:
                running.append(c)
            if co["waiting_max"] > take:
                waiting.append(c)
        return [*self._common_checks(rows, store, off, exp, msgs),
                Check("each_candidate_evaluated_once", not once, {"candidates": B, "campaigns_checked": len(whole),
                                                                  "not_once": once[:10]}),
                Check("survivors_within_the_population", not survivors,
                      {"most": o["population"], "over": survivors[:10]}),
                Check("inner_loops_within_the_slots", not running, {"most": S, "over": running[:10]}),
                Check("candidates_waiting_within_take", not waiting, {"most": take, "over": waiting[:10]})]


def calls_as_planned(r: dict, asks: int) -> bool:
    """An agent's model calls (its row): `asks` asked, each answered by a reply it could use or fallen back after
    at most TRIES tries, so AUTONMI's count of replies (model_calls) lies between the calls answered and the
    tries."""
    a, t, f = (r.get(k) or 0 for k in ("model_asks", "model_tries", "model_fallbacks"))
    return a == asks and f <= a and a - f <= r["model_calls"] <= t <= TRIES * a


class Script:
    """One population's campaigns as AUTONMI scripted agents (autonmi.scripted), 3 + campaign.slots per
    campaign: decide.py's Coordinator, Generator and Evaluator, and the Monitor slots, each a hosted agent of
    its own (a<i>/<name>: agent_names) with its own state, model calls (under its own id) and lookups, which
    talks to the others only through the population's Mailbox. One generation g (campaign.mode generations):

        Coordinator -> Generator    make/g        initialize (g = 0, with the constraints lookup) or repopulate
        Generator -> Coordinator    population/g  decide.generator_step: in llm mode, its refill call
        Generator -> the queue      P candidates  decide.evaluation_calls, on the campaign's candidate queue
        Monitor (each free slot)    the next candidate: its inner loop as one evaluation, suspended until it
                                    ends; its trace scored (monitor.py)
        Monitor -> Evaluator        report/<call> the result and the Monitor's numbers
        Evaluator -> Coordinator    results/g     the P results in candidate order, once all have arrived (the
                                    barrier): the Coordinator's selection, best-so-far and log; its memory

    then make/g+1, or after the last generation the Coordinator's summary call, done/G to the Generator and
    the Evaluator, and its Report; the Generator then puts one stop on the queue for each Monitor. A refill or
    summary call that fails is tried up to TRIES times, then falls back, counted (`_ask`): the refill keeps the
    parent's mutants, and the summary is left out (""); the campaign goes on. The
    Generator makes the abstract lookup when it starts (decide.LOOKUP_ROLE); the Coordinator makes the
    constraints lookup, after waiting (agents.start_together) for every agent of the population to start.
    The others end on done or their stop, so all stay live until their campaign ends. The Generator never
    addresses a Monitor, and no candidate passes through the Coordinator.

    Every id seen outside the agents is the campaign's, a<i>: the stub's key, the evaluations' agent_id
    (their cold starts and work folders), the lookups, the store key, the campaign's folder and its line
    (its result, or its error), which goes to the population's agents-p<index>.jsonl.

    With monitor.capture each evaluation's record carries its inner-loop trace (trace.py). The Monitor
    takes it out of the result before it reports, so decide.py (whose collect refuses a trace), the parent's
    files and the campaign's line never see one. `stats` holds each live campaign's trace statistics
    (monitor.Rolling), which its Monitors score against and hold their traces in; at the barrier the
    Evaluator labels the generation's traces good or bad and adds them (`_label`). The Monitor's numbers
    change no result in this mode: selection is the parent's, on the inner loop's numbers alone. With
    campaign.notes each Monitor also asks the model for a note on its candidate, and the Evaluator at the
    barrier for an assessment of the generation (a keep-probability for each candidate), both recorded with
    record.extras and read by nothing. AsyncFlow and RHAPSODY keep each evaluate
    task's returned record, trace included, and RHAPSODY a text copy of it (its `stdout`), until the
    population ends, so the traces are held per population.

    In the asynchronous mode (campaign.mode: async) there is no generation and no barrier: each agent repeats
    one cycle on what has arrived, its decisions decide.async_*'s pure functions, its messages in the inboxes of
    the population's Mailbox (take_all), and each cycle makes the role's own model call (`_ask`; none with
    llm.mode off):

        Generator                   once no candidate waits on the queue: the next decide.TAKE profiles onto
                                    it (candidates c0 .. c<B-1>), each reconciled with its model call's reply;
                                    it waits only for its first survivors
        Monitor (each free slot)    the next candidate, as in the generations mode; then its call: the note
        Monitor -> Evaluator        report        the result, the Monitor's numbers and its note
        Evaluator -> Coordinator    scored        each result that has arrived, with its q and score (its
                                                  call's keep-probability)
        Evaluator -> Generator      digest        their candidates and scores
        Coordinator -> Generator    survivors     after each cycle: survival, survivors, best-so-far and log
        Coordinator -> Generator    guidance      then its call's guidance on what to explore next

    then, once the budget's last result is in, done/G to the Generator and the Evaluator and the Coordinator's
    Report; the Generator takes what is left for it and puts the Monitors' stops. The campaign's statistics
    object also holds its window of the last outer.population scored results (the Evaluator's), and each trace
    is labelled as its result is scored (`_label_async`). A role's call that fails (an error, a reply its
    decide parse cannot use, or a prompt too long for the servers' context) is tried up to TRIES times; then
    the role decides as with llm.mode off, counted: the Generator's profiles unchanged, no note, the score from
    numbers, no guidance. The roles' own calls (those tried TRIES times) are in their campaign's line
    (llm_call_details) with record.extras only; otherwise each is summed in the population's `sums` (CallSums),
    as is any call that ends after its campaign's line was written."""

    def __init__(self, definition: Fusion, pop) -> None:
        self.knobs, self.sim, self.out, self.pop = definition.knobs, definition.sim, definition.out, pop
        self.live = definition.live
        self.G, self.roles = self.knobs["outer"]["generations"], decide.expected_by_role(self.knobs)
        self.P, self.slots = self.knobs["outer"]["population"], self.knobs["campaign"]["slots"]
        self.extras = self.knobs["record"]["extras"]
        self.together = self.knobs["agents"]["start_together"]
        self.mail = Mailbox(len(pop.agent_ids))
        self.calls: dict[str, list] = {}    # each campaign's model calls, until its line is written
        self.sims: Counter = Counter()      # the evaluations each Monitor submitted
        self.lookups_made: Counter = Counter()
        self.reported: set[str] = set()     # campaigns
        self.finished: set[str] = set()     # agents that ended normally
        self.llm_calls: dict[str, int] = {}   # each reported campaign's n_llm_calls
        self.start_wait: dict[str, float] = {}   # each Coordinator's wait for every agent to start
        self.began: dict[str, float] = {}        # when each Coordinator began its campaign (time.time())
        self.stats: dict[str, monitor.Rolling] = {}   # each live campaign's trace statistics
        self.numbers: dict[str, list] = {}       # each campaign's per-candidate Monitor numbers (record.extras)
        self.traced: Counter = Counter()         # the traces each Monitor got
        self.reports: Counter = Counter()        # the reports each Monitor sent
        # the asynchronous mode: each agent's cycles, and per campaign its Coordinator's counts, the most
        # survivors and inner loops running at once, the candidates evaluated (a bit each) and repeated, the
        # Generator's waits for survivors, and with record.extras what a replay needs (`replay`)
        self.asynchronous = decide.asynchronous(self.knobs)
        self.B = decide.async_budget(self.knobs)
        self.cycles: Counter = Counter()
        self.tally: dict[str, dict] = {}
        self.survivors_max: Counter = Counter()
        self.running: Counter = Counter()
        self.running_max: Counter = Counter()
        self.seen: dict[str, int] = {}
        self.repeated: Counter = Counter()
        self.survivor_waits: Counter = Counter()
        self.replay: dict[str, dict] = {}
        # the roles' own model calls (decide.role_calls): per agent, those asked, tried and fallen back; each
        # Generator's proposals by parameters filled; each campaign's calls answered; the servers' context
        self.calls_on = decide.role_calls(self.knobs)
        self.asks, self.tries, self.fallbacks = Counter(), Counter(), Counter()
        self.filled: dict[str, list] = {}
        self.answered: Counter = Counter()
        self.sums = CallSums()
        self.context, self.transport = definition.context, definition.transport
        self.fail_at = os.environ.get(FAIL_ENV)
        self.lines = open(self.out / f"agents-p{pop.index:03d}.jsonl", "w", buffering=1)  # noqa: SIM115 (rows())
        pop.state.update(script=self, mailbox=self.mail)
        self.evaluate = task_tool(
            run.sim_function(definition.sim), suspend=True, name="evaluate", args_schema=EvaluateArgs,
            description="Evaluate one candidate profile with the turned-down IGNOS inner loop.",
            prepare=lambda agent_id, call_id, args: {**args, "knobs": definition.sim_knobs(args["seed"]),
                                                     "monitor": self.knobs["monitor"],
                                                     "agent_id": campaign_of(agent_id)})

    def agent(self) -> ScriptedAgent:
        """One definition for all the roles (AUTONMI has one per population), so one model-call limit: the
        largest role's, each call tried at most TRIES times. verify() checks each role's count."""
        return ScriptedAgent(start=self.start, step=self.step, simulate=self.evaluate, lookup=self.lookup,
                             remember=self.remember, report=self.report, on_call=self.on_call, on_done=self.on_done,
                             model_calls_limit=TRIES * max(r["llm_calls"] for r in self.roles.values()))

    # --- the steps --------------------------------------------------------------------------------------

    def start(self, agent_id: str, spec: dict) -> dict:
        """An agent's first state, when the host starts it."""
        self.live["started"] += 1
        self.mail.start()
        role = role_of(agent_id)
        s = {"agent_id": agent_id, "campaign": campaign_of(agent_id), "role": role, "seed": spec["seed"]}
        if role in ("evaluator", "monitor"):
            return {**s, "phase": "waiting"}
        return {**s, "phase": "start" if role == "coordinator" else "look", "t_start": time.time(),
                "todo": self.roles[role]["lookups"], "ctx": {}}

    async def step(self, s: dict, event, ask):
        if self.asynchronous:
            return await {"coordinator": self._coordinator_async, "generator": self._generator_async,
                          "evaluator": self._evaluator_async, "monitor": self._monitor}[s["role"]](s, event, ask)
        return await {"coordinator": self._coordinator, "generator": self._generator,
                      "evaluator": self._evaluator, "monitor": self._monitor}[s["role"]](s, event, ask)

    async def _begin(self, me: str) -> None:
        """With agents.start_together, wait until every agent of the population has started, so every one is
        live at once (the Coordinator's wait is recorded); then the campaign begins (`began`)."""
        if self.together:
            t = time.perf_counter()
            with activity(AgentState.TOOL_WAIT):
                await self.mail.all_started.wait()
            self.start_wait[me] = round(time.perf_counter() - t, 6)
        self.began[me] = time.time()

    async def _coordinator(self, s: dict, event, ask):
        me, campaign, phase = s["agent_id"], s["campaign"], s["phase"]
        run_dir = decide.campaign_dir(self.out, self.knobs, campaign)
        if phase == "start":
            await self._begin(me)
            s, phase = {**s, "phase": "look"}, "look"
        if phase == "look":
            todo, ctx = s["todo"], s["ctx"]
            if event is not None:                               # the previous lookup's result
                todo, ctx = todo[1:], {**ctx, todo[0]: event}
            if todo:
                return {**s, "todo": todo, "ctx": ctx}, Lookup(todo[0])
            cst = decide.coordinator_state(campaign, s["seed"], self.knobs, ctx, s["t_start"])
            return await self._generation(s, cst, decide.start(cst, self.knobs, run_dir, self.sim), run_dir)
        if phase == "remembering":
            cst = s["cst"]
            if s["next"] is not None:
                return await self._generation(s, cst, s["next"], run_dir)
            summary = await self._ask(me, ask, decide.summary_messages(cst), decide.SUMMARY_TOKENS, cst["budget"])
            for to in ("generator", "evaluator"):
                self._send(me, to, "done", self.G, None)
            # after TRIES failed tries (None) the summary is left out: ""
            return _state(s, "reporting"), Report(decide.finish(cst, summary or "", run_dir))
        if phase == "reporting":
            return s, Done()
        raise ValueError(f"unknown phase {phase!r}")

    async def _generation(self, s: dict, cst: dict, msg: dict, run_dir):
        """The Coordinator's generation: the Generator's population (at g = 0 with the count of its own
        lookups), then the Evaluator's results, then selection, best-so-far and log, and the memory."""
        me, g = s["agent_id"], cst["generation"]
        self._send(me, "generator", "make", g, msg)
        reply = await self.mail.receive(me, "population", g)
        cst = {**cst, "n_tool_calls": cst["n_tool_calls"] + reply.get("lookups", 0)}
        cst, _ = decide.on_population(cst, reply)   # its candidates go to the Monitors from the Generator
        results = await self.mail.receive(me, "results", g)
        cst, nxt = decide.on_evaluated(cst, results, self.knobs, run_dir)
        return _state(s, "remembering", cst=cst, next=nxt), Remember(decide.memory(cst))

    async def _generator(self, s: dict, event, ask):
        """Its lookups, then each generation's population on make/g (decide.generator_step), to the
        Coordinator, and its candidates on the campaign's queue (`t0`: when they were put there); on done,
        one stop on the queue for each Monitor slot."""
        me, campaign, todo, ctx = s["agent_id"], s["campaign"], s["todo"], s["ctx"]
        if event is not None:
            todo, ctx = todo[1:], {**ctx, todo[0]: event}
        if todo:
            return {**s, "todo": todo, "ctx": ctx}, Lookup(todo[0])
        gst = decide.generator_state(campaign, s["seed"], self.knobs)
        ask_model = lambda m, b, r: self._ask(me, ask, m, r, b)  # noqa: E731
        for g in range(self.G):
            msg = await self.mail.receive(me, "make", g)
            if g == 0:   # the Coordinator's lookups, and its own: the refill prompt reads both
                msg = {**msg, "context": {**msg.get("context", {}), **ctx}}
            gst, reply = await decide.generator_step(gst, msg, self.knobs, ask_model)
            self._send(me, "coordinator", "population", g, {**reply, "lookups": len(ctx)} if g == 0 else reply)
            t0 = time.time()
            for call in decide.evaluation_calls(reply["population"], g):
                self.mail.put(me, QUEUE, {**call, "t0": t0})
        await self.mail.receive(me, "done", self.G)
        for _ in range(self.slots):
            self.mail.put(me, QUEUE, None)
        return s, Done()

    async def _monitor(self, s: dict, event, ask):
        """A Monitor slot: the next candidate on its campaign's queue, its inner loop as one evaluation (the
        call decide.evaluation_calls names, as the Evaluator submitted it before: the same results),
        suspended until it ends; then its trace, taken out of the result, scored against the campaign's
        statistics and held there for the barrier's label; its model call (decide.role_calls), the note on its
        candidate ("" when the call falls back: none); its report (the result, `t0`, the numbers and the note)
        to the Evaluator; then the next candidate, until its stop. In the asynchronous mode the report goes to
        the Evaluator's inbox (take_all), and its trace is labelled when the Evaluator scores its result."""
        me, campaign = s["agent_id"], s["campaign"]
        if s["phase"] == "running":   # resumed: its candidate's inner loop ended
            if campaign in self.mail.failed:
                raise CampaignFailed(self.mail.failed[campaign])
            call = s["call"]
            (result,), (tr,) = trace.split([event])
            numbers = None
            if tr is not None:
                self.traced[me] += 1
                stats = self._stats(campaign)
                x = monitor.steps(tr)
                stats.hold(call["generation"], call["candidate_idx"], x, bool(tr["synthetic"]))
                numbers = stats.score(x)
            if self.asynchronous:   # its inner loop has ended
                self.running[campaign] -= 1
            note = None
            if self.calls_on:
                note = await self._ask(me, ask, decide.note_messages(result, numbers), decide.NOTE_TOKENS,
                                       decide.budget(self.knobs, s["seed"]), parse=decide.parse_text) or ""
            if self.asynchronous:   # to the Evaluator's inbox, which takes every report that has arrived
                self._hook(me, call["id"])
                self.mail.deliver(me, "evaluator", "report", {
                    "candidate": call["candidate_idx"], "result": result, "t0": call["t0"], "source": call["source"],
                    "filled": call["filled"], "m": numbers["m"] if numbers else None, "numbers": numbers,
                    "note": note})
            else:
                self._send(me, "evaluator", "report", call["id"],
                           {"result": result, "t0": call["t0"], "numbers": numbers, "note": note})
            self.reports[me] += 1
        call = await self.mail.take(me, QUEUE)
        if campaign in self.mail.failed:   # it failed after the item reached me: start no inner loop for it
            raise CampaignFailed(self.mail.failed[campaign])
        if call is None:   # its stop: the campaign has ended
            return _state(s, "done"), Done()
        self.sims[me] += 1
        keep = ("id", "generation", "candidate_idx", "t0")
        if self.asynchronous:   # its cycle; each candidate's evaluations and the inner loops running, counted
            keep += ("source", "filled")
            self.cycles[me] += 1
            bit, seen = 1 << call["candidate_idx"], self.seen.get(campaign, 0)
            self.repeated[campaign] += bool(seen & bit)
            self.seen[campaign] = seen | bit
            self.running[campaign] += 1
            self.running_max[campaign] = max(self.running_max[campaign], self.running[campaign])
        return _state(s, "running", call={k: call[k] for k in keep}), \
            Simulate({"params": call["params"], "candidate_idx": call["candidate_idx"],
                      "generation": call["generation"], "seed": s["seed"]}, call_id=call["id"])

    async def _evaluator(self, s: dict, event, ask):
        """Each generation's P reports, gathered in candidate order (the barrier); the labels of their traces
        (`_label`); with campaign.notes its assessment (`_assess`, before the results go on, so that its call is
        in the campaign's line); their results, in candidate order, to the Coordinator (`t0`: when the Generator
        put them on the queue); after the last generation, done."""
        me, campaign = s["agent_id"], s["campaign"]
        for g in range(self.G):
            reports = [await self.mail.receive(me, "report", f"g{g}c{i}") for i in range(self.P)]
            results, t0 = [r["result"] for r in reports], min(r["t0"] for r in reports)
            self._label(campaign, g, results, [r["numbers"] for r in reports])
            if self.calls_on:
                await self._assess(me, ask, s["seed"], reports)
            self._send(me, "coordinator", "results", g, {"results": results, "t0": t0, "elapsed_s": time.time() - t0})
        await self.mail.receive(me, "done", self.G)
        return s, Done()

    async def _assess(self, me: str, ask, seed: int, reports: list[dict]) -> None:
        """campaign.notes, after a generation's barrier: the Evaluator's call, as in the asynchronous mode (a
        keep-probability for each candidate; the score from numbers, q + monitor.weight x m clipped to 0..1, when
        it falls back), which changes no result; with record.extras each candidate's note and assessment
        (`assessed`) beside its numbers (the entries _label has just written)."""
        q = decide.keep_probabilities([decide.native(r["result"]) for r in reports], self.knobs["outer"]["wall_weight"])
        w = self.knobs["monitor"]["weight"]
        items = [{"candidate": i, "result": r["result"], "q": q[i], "numbers": r["numbers"], "note": r["note"],
                  "score": decide.numbers_score(q[i], (r["numbers"] or {}).get("m"), w)} for i, r in enumerate(reports)]
        got = await self._ask(me, ask, decide.scores_messages(items), decide.scores_tokens(len(items)),
                              decide.budget(self.knobs, seed),
                              parse=lambda x: decide.parse_scores(x, list(range(len(items)))))
        if self.extras:
            for entry, it in zip(self.numbers.get(campaign_of(me), [])[-len(items):], items):
                entry.update(note=it["note"], assessed=it["score"] if got is None else got[it["candidate"]])

    def _label(self, campaign: str, g: int, results: list[dict], numbers: list) -> None:
        """At generation g's barrier: each held trace joins the campaign's good side when its keep-probability
        q is 0.5 or more, else the bad side, in candidate order. q is the parent's (decide.keep_probabilities,
        the generation as select_survivors takes it); for a synthetic trace (the fixture, whose results follow
        a hash unrelated to the trace) it comes from the trace's own end values (monitor.end_keep), marked
        synthetic. With record.extras, each candidate's numbers, q and label, for its campaign's line."""
        stats = self.stats.get(campaign)
        held = stats.take(g) if stats is not None else {}
        if not held and not self.extras:
            return
        q = decide.keep_probabilities([decide.native(r) for r in results], self.knobs["outer"]["wall_weight"])
        order = sorted(held)
        synthetic = [i for i in order if held[i][1]]
        label_q = dict(zip(synthetic, monitor.end_keep([held[i][0] for i in synthetic])))
        label = {}
        for i in order:
            label[i] = label_q.get(i, q[i]) >= 0.5
            stats.add(held[i][0], label[i])
        if self.extras:
            self.numbers.setdefault(campaign, []).extend(
                {"generation": g, "candidate_idx": i, "q": round(q[i], 6), **(numbers[i] or {}),
                 "label": None if i not in label else "good" if label[i] else "bad",
                 "label_from": None if i not in label else "trace_end_synthetic" if i in label_q else "q"}
                for i in range(len(results)))

    # --- the asynchronous mode (campaign.mode: async): each agent's cycle, its decisions decide.async_* ------

    async def _coordinator_async(self, s: dict, event, ask):
        """Its start and lookups as in the generations mode; then its cycle until the budget's last result is
        in: every scored message that has arrived (waiting for one), their results in arrival order
        (decide.async_on_scored: survival, the survivors, best-so-far, the log), the newest survivors to the
        Generator, then its model call and its guidance to the Generator ("" when the call falls back: none;
        survival is already decided), and its memory; at the end done to the Generator and the Evaluator, and
        its Report (no summary call in this mode), with its campaign's model calls answered."""
        me, campaign, phase = s["agent_id"], s["campaign"], s["phase"]
        run_dir = decide.campaign_dir(self.out, self.knobs, campaign)
        if phase == "start":
            await self._begin(me)
            s, phase = {**s, "phase": "look"}, "look"
        if phase == "look":
            todo, ctx = s["todo"], s["ctx"]
            if event is not None:                               # the previous lookup's result
                todo, ctx = todo[1:], {**ctx, todo[0]: event}
            if todo:
                return {**s, "todo": todo, "ctx": ctx}, Lookup(todo[0])
            cst = decide.async_coordinator_state(campaign, s["seed"], self.knobs, ctx, s["t_start"])
            decide.start(cst, self.knobs, run_dir, self.sim)   # config.json (outer.logs)
            s, phase = _state(s, "cycle", cst=cst), "cycle"
        if phase == "cycle":
            scored = [x for msg in await self.mail.take_all(me, "scored") for x in msg["results"]]
            cst, rows = decide.async_on_scored(s["cst"], scored, self.knobs, run_dir)
            self.cycles[me] += 1
            self.tally[campaign] = {"results_scored": cst["arrived"], "results_kept": cst["kept"],
                                    "results_scored_after_warmup": cst["scored_after_warmup"],
                                    "results_kept_after_warmup": cst["kept_after_warmup"]}
            self.survivors_max[campaign] = max(self.survivors_max[campaign], len(cst["survivors"]))
            replay = self._replay(campaign)
            if replay is not None:
                replay["coordinator"].append([x["candidate"] for x in scored])
                replay["results"] += rows
                replay["survivors"] = [x["candidate"] for x in cst["survivors"]]
            self._hook(me, self.cycles[me] - 1)
            self.mail.deliver(me, "generator", "survivors", decide.async_survivors(cst))
            if self.calls_on:
                text = await self._ask(me, ask, decide.guidance_messages(cst, rows, self.knobs),
                                       decide.GUIDANCE_TOKENS, cst["budget"], parse=decide.parse_text) or ""
                self.mail.deliver(me, "generator", "guidance", {"n": cst["cycles"], "text": text})
                if replay is not None:
                    replay["guidance"].append(text)
            return _state(s, "cycle" if cst["arrived"] < self.B else "ending", cst=cst), \
                Remember(decide.async_memory(cst))
        if phase == "ending":
            for to in ("generator", "evaluator"):
                self._send(me, to, "done", self.G, None)
            return _state(s, "reporting"), Report(decide.async_finish(s["cst"], run_dir, self.answered[campaign]))
        if phase == "reporting":
            return s, Done()
        raise ValueError(f"unknown phase {phase!r}")

    async def _generator_async(self, s: dict, event, ask):
        """Its lookups; then (with agents.start_together, once every agent of its population has started) its
        cycle until the budget's candidates are made: once no candidate waits on the campaign's queue, the
        survivors, digests and guidance that have arrived (taken without waiting), its first survivors if the
        take needs one and none has come (its only wait), and the next decide.TAKE profiles, reconciled by its
        model call (`_reconcile`), onto the queue (`t0`: when they were put there). Then done, every message
        still left for it, and one stop on the queue per Monitor."""
        me, campaign, todo, ctx = s["agent_id"], s["campaign"], s["todo"], s["ctx"]
        if event is not None:
            todo, ctx = todo[1:], {**ctx, todo[0]: event}
        if todo:
            return {**s, "todo": todo, "ctx": ctx}, Lookup(todo[0])
        if self.together:
            with activity(AgentState.TOOL_WAIT):
                await self.mail.all_started.wait()
        gst, replay = decide.async_generator_state(campaign, s["seed"], self.knobs, ctx), self._replay(campaign)
        while gst["made"] < self.B:
            await self.mail.emptied(me, QUEUE)
            gst = decide.async_on_survivors(gst, await self.mail.take_all(me, "survivors", wait=False))
            gst = decide.async_on_digests(gst, await self.mail.take_all(me, "digest", wait=False), self.knobs)
            gst = decide.async_on_guidance(gst, await self.mail.take_all(me, "guidance", wait=False))
            if decide.async_waits(gst, self.knobs):
                self.survivor_waits[me] += 1
                gst = decide.async_on_survivors(gst, await self.mail.take_all(me, "survivors"))
            gst, calls = decide.async_profiles(gst, self.knobs)
            if self.calls_on and calls:
                calls = await self._reconcile(me, ask, gst, calls, s["seed"])
            self._hook(me, self.cycles[me])
            t0 = time.time()
            for call in calls:
                self.mail.put(me, QUEUE, {**call, "t0": t0})
            if replay is not None:
                replay["held"] += [gst["held"]] * len(calls)
            self.cycles[me] += 1
        await self.mail.receive(me, "done", self.G)
        for kind in ("survivors", "digest", "guidance"):   # all sent before done
            await self.mail.take_all(me, kind, wait=False)
        for _ in range(self.slots):
            self.mail.put(me, QUEUE, None)
        return s, Done()

    async def _reconcile(self, me: str, ask, gst: dict, calls: list[dict], seed: int) -> list[dict]:
        """The Generator's model call on a take (its JSON schema as response_format): each profile with the
        reply's values in place of the evolutionary loop's (decide.async_reconcile), or unchanged when the call
        falls back; each candidate's `filled` (the parameters its reply filled, 0 on the fallback), and this
        Generator's proposals counted by the parameters filled."""
        n = len(calls)
        got = await self._ask(me, ask, decide.async_generator_messages(gst, calls, self.knobs), decide.reply_tokens(n),
                              decide.budget(self.knobs, seed), parse=lambda r: decide.parse_reconcile(r, n),
                              response_format=decide.candidates_schema(n))
        if got is None:
            return [{**c, "filled": 0} for c in calls]
        calls, filled = decide.async_reconcile(calls, got)
        counts = self.filled.setdefault(me, [0] * 8)
        for k in filled:
            counts[k] += 1
        return calls

    async def _evaluator_async(self, s: dict, event, ask):
        """Its cycle until the budget's results are scored: every report that has arrived (waiting for one; with
        its model call at most decide.SHOWN_RESULTS, the rest left for its next cycle in arrival order), scored in
        arrival order against the campaign's window, which its statistics hold (decide.async_evaluate), each trace
        labelled (`_label_async`); its model call, whose keep-probabilities are the scores (from numbers when it
        falls back); the scored results to the Coordinator and a digest to the Generator. Then done."""
        me, campaign = s["agent_id"], s["campaign"]
        scored_n, replay = 0, self._replay(campaign)
        limit = decide.SHOWN_RESULTS if self.calls_on else None
        while scored_n < self.B:
            reports = await self.mail.take_all(me, "report", limit=limit)
            stats = self._stats(campaign)
            stats.recent, scored, digest = decide.async_evaluate(stats.recent, reports, self.knobs)
            for r, sc in zip(reports, scored):
                self._label_async(campaign, stats, r, sc)
            if self.calls_on:
                items = [{**sc, "numbers": r["numbers"], "note": r["note"]} for r, sc in zip(reports, scored)]
                got = await self._ask(me, ask, decide.scores_messages(items),
                                      decide.scores_tokens(len(items)), decide.budget(self.knobs, s["seed"]),
                                      parse=lambda x: decide.parse_scores(x, [sc["candidate"] for sc in scored]))
                if got is not None:
                    scored, digest = decide.async_model_scores(scored, got)
            scored_n += len(scored)
            self._hook(me, self.cycles[me])
            self.mail.deliver(me, "coordinator", "scored", {"results": scored})
            self.mail.deliver(me, "generator", "digest", digest)
            if replay is not None:
                replay["evaluator"].append([sc["candidate"] for sc in scored])
            self.cycles[me] += 1
        await self.mail.receive(me, "done", self.G)
        return s, Done()

    def _label_async(self, campaign: str, stats: monitor.Rolling, report: dict, sc: dict) -> None:
        """As a result is scored, its trace (held by its Monitor) joins the campaign's good side when q is 0.5
        or more, else the bad side; a synthetic trace (the fixture's) is labelled instead from its own end
        values among the window's synthetic traces, it last (monitor.end_keep), marked synthetic. With
        record.extras, the candidate's numbers, q, label and (with the roles' calls) the Monitor's note, for its
        campaign's line."""
        k = sc["candidate"]
        held, label, label_from = stats.take_one(k // self.P, k), None, None
        if held is not None:
            x, synthetic = held
            if synthetic:
                label = monitor.end_keep([e for e in stats.ends if e is not None] + [x[-1:]])[-1] >= 0.5
                label_from = "trace_end_synthetic"
            else:
                label, label_from = sc["q"] >= 0.5, "q"
            stats.add(x, label)
        stats.ends = (stats.ends + [held[0][-1:] if held is not None and held[1] else None])[-self.P:]
        if self.extras:
            self.numbers.setdefault(campaign, []).append(
                {"generation": k // self.P, "candidate_idx": k, "q": round(sc["q"], 6), **(report["numbers"] or {}),
                 "label": None if label is None else "good" if label else "bad", "label_from": label_from,
                 **({"note": report["note"]} if self.calls_on else {})})

    def _stats(self, campaign: str) -> monitor.Rolling:
        """The campaign's statistics object, made by whichever agent needs it first."""
        return self.stats.get(campaign) or self.stats.setdefault(
            campaign, monitor.Rolling(self.knobs["campaign"]["warmup"], self.P))

    def _replay(self, campaign: str) -> dict | None:
        """With record.extras, what a replay of the campaign needs, for its line: the order results reached the
        Evaluator and the Coordinator (one list per cycle), the survivors message held at each profile, each
        result's row (decide.async_on_scored) and the survivors at the end; and each Coordinator cycle's
        guidance (with the roles' calls)."""
        if not self.extras:
            return None
        return self.replay.get(campaign) or self.replay.setdefault(
            campaign, {"evaluator": [], "coordinator": [], "held": [], "results": [], "survivors": [], "guidance": []})

    def _hook(self, me: str, tag) -> None:
        if self.fail_at == f"{me}:{tag}":   # the tests' injected failure (FAIL_ENV)
            raise InjectedFailure(f"{me} failed before its sends of {tag} ({FAIL_ENV})")

    def _send(self, me: str, to: str, kind: str, g: int, value) -> None:
        if self.fail_at == f"{me}:{g}":   # the tests' injected failure (FAIL_ENV)
            raise InjectedFailure(f"{me} failed before sending {kind}/{g} ({FAIL_ENV})")
        self.mail.send(me, to, kind, g, value)

    def _request(self, budget: int, reply_tokens: int, role: str, own: bool) -> dict:
        """A call's request fields (request_fields), with its meta for on_call. `own`: a role's own call (tried
        TRIES times)."""
        return {**request_fields(self.knobs, budget, reply_tokens),
                "meta": {"budget": budget, "reply_tokens": reply_tokens, "role": role, "own": own}}

    async def _ask(self, me: str, ask, messages: list[dict], reply_tokens: int, budget: int, parse=None,
                   response_format: dict | None = None):
        """A model call of agent `me` (decide.Ask through the population's fleet, `_request`'s fields;
        `response_format`, the Generator's JSON schema, as a plain keyword). Each try's details (`_detail`): one
        answered by on_call, one that failed here, as run.Services records it. Counted per agent: asked, tried,
        fallen back. No try is made once another agent has failed the campaign (CampaignFailed).

        Without `parse` (the generations mode's refill and summary): the reply's text, tried up to TRIES times
        while the call fails (an error); then None, the step's fallback (the refill keeps the parent's mutants, the
        summary is left out), counted. The other layers end the campaign at the first failed call. With it (the
        roles' own calls): parse(reply), tried up to TRIES times while the call fails or parse gives None (a reply
        it cannot use); then None, the role's fallback.
        A role's call whose prompt is too long for the servers' context (the prompt-length check: its characters,
        at least its tokens, plus max_tokens over the context) is not sent and not tried again: it fails at once,
        with the error PromptTooLong."""
        role, campaign, own = role_of(me), campaign_of(me), parse is not None
        kwargs = self._request(budget, reply_tokens, role, own)
        if response_format is not None:
            kwargs["response_format"] = response_format
        chars = sum(len(str(m.get("content") or "")) for m in messages) if own and self.context is not None else 0
        too_long = own and self.context is not None and chars + kwargs["max_tokens"] > self.context
        # record.extras: the request's settings, counted per request sent (CallSums.requests)
        settings = json.dumps({"role": role, "body": sent(self.transport, kwargs)}, sort_keys=True) \
            if self.extras else None
        self.asks[me] += 1
        for _ in range(TRIES):
            if campaign in self.mail.failed:
                raise CampaignFailed(self.mail.failed[campaign])
            self.tries[me] += 1
            t0 = time.perf_counter()
            try:
                if too_long:
                    raise PromptTooLong(f"{chars} prompt characters + max_tokens {kwargs['max_tokens']} > the "
                                        f"servers' context {self.context}")
                if settings is not None:
                    self.sums.requests[settings] += 1
                reply = await ask(messages, **kwargs)
            except Exception as exc:
                self._detail(campaign, own, {"role": role, "reply_tokens": reply_tokens, "budget": budget,
                                             "latency_s": round(time.perf_counter() - t0, 4),
                                             "error": type(exc).__name__, "finish_reason": None, "empty": None})
                if too_long:
                    break
                continue
            if not own:
                return reply
            out = parse(reply)
            if out is not None:
                return out
        self.fallbacks[me] += 1
        return None

    def _detail(self, campaign: str, own: bool, d: dict) -> None:
        """A call's details: in its campaign's line (llm_call_details) for the generations mode's refill and
        summary, as before, and for the roles' own calls with record.extras; otherwise, and for a call that ends
        after its campaign's line was written, summed for the record (`sums`)."""
        if (self.extras or not own) and self._open(campaign):
            self.calls.setdefault(campaign, []).append(d)
        else:
            self.sums.add(d)

    def _open(self, campaign: str) -> bool:
        """Whether the campaign's line is still to be written (it has neither reported nor failed)."""
        return campaign not in self.reported and campaign not in self.mail.failed

    # --- the actions' services ----------------------------------------------------------------------------

    async def lookup(self, agent_id: str, what: str) -> dict:
        """Counted once it returns, as AUTONMI's runner counts it (lookups_match_autonmi)."""
        out = await tools.lookup(what, campaign_of(agent_id), self.knobs["tools"]["latency_ms"])
        self.lookups_made[agent_id] += 1
        return out

    def remember(self, agent_id: str, state: dict) -> None:
        """The Coordinator's memory after each generation, in the population's store (RaDEX under Dragon)."""
        self.pop.store.put(f"campaign/{campaign_of(agent_id)}", json.dumps(state, default=str))

    def report(self, agent_id: str, result: dict) -> None:
        """A campaign's result to the campaign manager: its line, as run.Services.report writes it."""
        campaign = campaign_of(agent_id)
        self.reported.add(campaign)
        self.llm_calls[campaign] = result["n_llm_calls"]
        self.answered.pop(campaign, None)
        extras = {"monitor": self.numbers.pop(campaign, [])} if self.extras else {}   # record.extras: outside the digest
        if self.extras and self.asynchronous:   # each result's row, and what a replay needs
            extras["asynchronous"] = self.replay.pop(campaign, {})
        self._line({**result, "llm_call_details": self.calls.pop(campaign, []), **extras})

    def on_call(self, rec, reply: Reply) -> None:
        """A model call that returned: counted for its campaign's line (n_llm_calls in the asynchronous mode) while
        the line is still to be written, and its details (`_detail`), as run.Services keeps them, with its times
        (Flowgentic's CallRecord): it was in flight at the server from t_end - service_s to t_end, after
        recovery_s of failed tries (latency_s is their sum) and, before those, wait_s for admission. With
        record.extras also the server that answered it (`server`: the fleet's replica, the index of the model
        job's endpoints, llm_service.servers; 0 for the stub)."""
        m, campaign = rec.meta or {}, campaign_of(rec.agent_id)
        if self._open(campaign):
            self.answered[campaign] += 1
        self._detail(campaign, m.get("own", False), {
            "role": m.get("role"), "reply_tokens": m.get("reply_tokens"), "budget": m.get("budget"),
            "latency_s": round(rec.service_s + rec.recovery_s, 4), "wait_s": round(rec.wait_s, 4),
            "t_end": round(rec.t_end, 6), "service_s": round(rec.service_s, 6), "recovery_s": round(rec.recovery_s, 6),
            "finish_reason": reply.finish_reason, "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens, "reasoning_tokens": rec.reasoning_tokens or 0,
            "empty": not str(reply.content or "").strip(), **({"server": rec.replica} if self.extras else {})})

    def on_done(self, result, output) -> None:
        """An agent that ended normally is finished. The first of a campaign's agents to fail writes the
        campaign's line, with its error, its role and every role's model calls, as the other layers write a
        failed campaign; the others then end with an error (CampaignFailed) instead of waiting. One
        cancelled (the run was interrupted) writes nothing."""
        self.live["ended"] += 1
        campaign = campaign_of(result.id)
        if role_of(result.id) == "evaluator":   # its campaign's statistics end with it: no trace comes after
            self.stats.pop(campaign, None)
        if result.status == "done":
            self.finished.add(result.id)
        elif result.status == "failed":
            if campaign not in self.reported and campaign not in self.mail.failed:
                self._line({"agent_id": campaign, "error": result.error or result.status,
                            "role": role_of(result.id), "llm_call_details": self.calls.pop(campaign, [])})
                self.numbers.pop(campaign, None)
                self.replay.pop(campaign, None)
                self.answered.pop(campaign, None)
            self.mail.fail(campaign, f"{result.id} failed: {result.error}")

    def _line(self, line: dict) -> None:
        self.lines.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def _state(s: dict, phase: str, **memory) -> dict:
    """An agent's state: its id, campaign, role, seed and phase, and what the phase needs."""
    return {k: s[k] for k in ("agent_id", "campaign", "role", "seed")} | {"phase": phase, **memory}


# --- the live line and the model service -----------------------------------------------------------------

def _model_states() -> tuple[int, int] | None:
    """Agents with a model request open (admitted and sent) and agents queued for admission, summed over
    this process's agent hosts: Flowgentic's own counts (its AgentStates), read as Flowgentic's
    OpenTelemetry gauge reads them (flowgentic._telemetry._observe_agents). None if Flowgentic stops
    keeping them there."""
    try:
        from flowgentic._agents import AgentState, _all_states

        opened = queued = 0
        for states in list(_all_states):
            counts = states._live_counts()   # None once that host's run has ended
            if counts is not None:
                opened += counts[AgentState.MODEL_REQUEST]
                queued += counts[AgentState.MODEL_QUEUED]
        return opened, queued
    except (ImportError, AttributeError, IndexError):
        return None


class LiveLine:
    """The line the head process prints to stderr once per record.ticker interval, and its series for the
    run record:

        agents live=<n> model_requests_open=<n> queued=<n> decoding_at_servers=<n> done=<n>

    live and done: hosted agents started and not ended, and ended (Script's counts); model_requests_open
    and queued: client side, Flowgentic's agent states (_model_states); decoding_at_servers: the sampler's
    latest reading of the model servers' running batches, summed ("-" without a sampler, as with the stub).
    Under Dragon the populations run in processes of their own, and only the servers' count is seen here."""

    FIELDS = ("t_s", "agents_live", "model_requests_open", "model_requests_queued", "decoding_at_servers", "done")

    def __init__(self, local: bool = True) -> None:
        self.local, self.counts, self.sampler, self.servers, self.served = local, Counter(), None, None, None
        self.t0, self.series, self.why_no_sampler = time.time(), [], None

    def read(self) -> list:
        states = _model_states() if self.local else None
        started, ended = self.counts["started"], self.counts["ended"]
        return [round(time.time() - self.t0, 2), started - ended if self.local else None,
                *(states or (None, None)), self._decoding(), ended if self.local else None]

    def emit(self, row: list) -> None:
        self.series.append(row)
        print("agents live={} model_requests_open={} queued={} decoding_at_servers={} done={}".format(
            *("-" if v is None else v for v in row[1:])), file=sys.stderr, flush=True)

    def _decoding(self) -> int | None:
        """The servers' latest running counts, summed, of those that answered within hold_s (a server that
        stopped answering is left out, as the record counts it, not held at its last reading)."""
        if self.sampler is None:
            return None
        series, now = list(self.sampler.series.values()), time.time()
        hold = hold_s(series, self.sampler.interval, recent=10)
        last = [s["num_requests_running"][-1] for s in series
                if s["t"] and s["num_requests_running"] and now < s["t"][-1] + hold]
        return int(sum(v or 0 for v in last)) if last else None

    @contextlib.asynccontextmanager
    async def during(self, out: Path):
        """The live line, and the sampler (if any) with a first reading before the run, around the run; at
        the end the sampler's last reading (which the last line shows), and its series in
        out/vllm/vllm.json, where AUTONMI's campaign finds it (its checks then compare the servers' token
        counts with the fleets')."""
        if self.sampler is not None:
            await asyncio.to_thread(self.sampler.sample)
            self.sampler.start()
        tick = asyncio.ensure_future(record.ticker(self.read, self.emit))
        interrupted = False
        try:
            yield
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            try:
                if self.sampler is not None:
                    if interrupted:   # SIGTERM: no last sweep (unreachable servers would take 5 s each), what
                        self.sampler._halt.set()   # was sampled is written; the run's TERM grace is short
                        await asyncio.to_thread(self.sampler.join, 5.0)
                    else:
                        await asyncio.to_thread(self.sampler.stop)
                    (out / "vllm").mkdir(parents=True, exist_ok=True)
                    (out / "vllm" / "vllm.json").write_text(json.dumps(self.sampler.raw()))
            finally:
                tick.cancel()
                await asyncio.gather(tick, return_exceptions=True)

    def section(self) -> dict:
        return {"interval_s": record.LIVE_INTERVAL_S, "t0": round(self.t0, 3), "fields": list(self.FIELDS),
                "series": self.series}


class ModelServiceError(RuntimeError):
    """The model servers cannot serve this run's model: the run is refused before any campaign starts."""


def _get(url: str, key: str | None, timeout: float = 10.0) -> bytes:
    headers = {"Authorization": f"Bearer {key}"} if key and key != "EMPTY" else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read()


def check_served_model(model: dict, name: str, context: list | None = None) -> list[dict]:
    """Every server of the model job lists `name` (llm.model) in its /v1/models, or ModelServiceError. The
    distinct entries they list for it (id and owned_by: who the server says serves it), for the record; and
    into `context` each server's max_model_len, when it gives one (vLLM does)."""
    from autonmi.models import api_key

    key, served = api_key(model), []
    for url in model["endpoints"]:
        try:
            data = json.loads(_get(url.rstrip("/") + "/v1/models", key)).get("data", [])
        except Exception as exc:   # noqa: BLE001 - any failure refuses the run, with its reason
            raise ModelServiceError(f"{url}/v1/models did not answer: {type(exc).__name__}: {exc}") from None
        ids = [m.get("id") for m in data]
        if name not in ids:
            raise ModelServiceError(f"{url} serves {ids}, not llm.model {name!r}: set llm.model to the name the "
                                    "server serves (vLLM's --served-model-name)")
        entry = next({"id": m.get("id"), "owned_by": m.get("owned_by")} for m in data if m.get("id") == name)
        length = next(m.get("max_model_len") for m in data if m.get("id") == name)
        if context is not None and isinstance(length, int):
            context.append(length)
        if entry not in served:
            served.append(entry)
    return served


def vllm_sampler(model: dict, tries: int = 3, pause_s: float = 2.0):
    """AUTONMI's VllmSampler over the model job's servers, every 2 s (5 s past 50 servers: it reads them one
    after another), when every server answers /metrics with vLLM's running count, each given `tries` tries
    `pause_s` apart; else (None, why not). All or none: AUTONMI compares the servers' token counts with all
    the fleets' calls."""
    from autonmi.models import api_key
    from rascal.telemetry.vllm import VllmSampler, fetch, parse

    key, urls = api_key(model), model["endpoints"]
    for url in urls:
        for attempt in range(tries):
            try:
                values, _ = parse(fetch(url, key))
                break
            except Exception as exc:   # noqa: BLE001 - no sampler, with the reason
                why = f"{url}/metrics: {type(exc).__name__}: {exc}"
                if attempt + 1 == tries:
                    return None, why
                time.sleep(pause_s)
        if "num_requests_running" not in values:
            return None, f"{url}/metrics has no vllm:num_requests_running"
    return VllmSampler(urls, key, 2.0 if len(urls) <= 50 else 5.0), None


VLLM_GAUGES = ("t", "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")   # what the record reads
HOLD_INTERVALS = 2.5   # a server's reading counts this many sampling intervals at most: one missed reading is bridged


def hold_s(series: list[dict], interval: float, recent: int | None = None) -> float:
    """How long a model server's reading of /metrics counts (record.summed's hold_s): 2.5 sampling intervals,
    or 2.5 times the servers' median gap between readings (of their last `recent`) when a sweep over them
    takes longer (the sampler reads them one after another). Past it, the server has stopped answering."""
    gaps = []
    for s in series:
        t = s["t"][-(recent + 1):] if recent else s["t"]
        gaps += [b - a for a, b in zip(t, t[1:])]
    return HOLD_INTERVALS * max(interval, statistics.median(gaps) if gaps else 0.0)


# a name that says it holds a secret (api-key, VLLM_API_KEY, HF_TOKEN, api_keys, ...): such a variable is not in
# the snapshot, and its value is left out of every copied file, where it is set by that name and wherever it is
SECRET_WORDS = r"API_?KEYS?|KEYS?|TOKEN|SECRETS?|PASSWORDS?|PASSWD|CREDENTIALS?"
SECRET_NAME = re.compile(rf"(^|[_.-])({SECRET_WORDS})($|[_.-])", re.IGNORECASE)
# a file so named is not copied: also tokens.txt (a setting named for tokens, max_num_batched_tokens, is no secret)
SECRET_FILE = re.compile(rf"(^|[_.-])({SECRET_WORDS}|TOKENS)($|[_.-])", re.IGNORECASE)
# a value set by a name in a copied file: NAME=value, NAME: value, "name": "value", 'name': ['value'], and a
# command line's --name value or --name=value (left out when SECRET_NAME says the name holds a secret)
_VALUE = rb"""("[^"\n]*"|'[^'\n]*'|[^\s"',;)}\]]+)"""
SET_BY_NAME = (re.compile(rb"""(?<![\w.-])([A-Za-z_][\w.-]*)["']?[ \t]*[:=][ \t]*\[?[ \t]*""" + _VALUE),
               re.compile(rb"""(?<![\w-])--([A-Za-z][\w-]*)(?:=|[ \t]+)""" + _VALUE))
URL_USER = re.compile(rb"://([^/\s@'\"]+)@")   # a URL's user and password (pip freeze's git+https://<token>@...)
# the environment snapshot's variables: these prefixes and names (LLM plan, question 11)
ENV_PREFIXES = ("RASCAL_", "SLURM_", "CUDA_", "VLLM_", "FLASHINFER_", "TRITON_")
ENV_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PATH", "TMPDIR")
LOG_HEAD_LINES = 500   # a log is copied as its head: version, non-default arguments, KV-cache memory
LEFT_OUT = b"[secret left out]"
SECRET_MIN = 8   # a secret-named variable's value is left out wherever it is only from this length: "1" is not
# beside the endpoints file, the files to copy (launch/stellar_ai_autonmi.sbatch writes it): one path or glob per
# line, relative to that folder, such as model-0/vllm-*.log for the first of several model jobs' folders
SETUP_LIST = "setup-files"


def leave_out(data: bytes, secrets: set[bytes]) -> tuple[bytes, int]:
    """`data` with LEFT_OUT for each stretch that holds a secret: each occurrence of a value in `secrets`, each
    value set by a secret-named name (SET_BY_NAME) and each URL's user and password. Stretches that overlap or
    hold one another are left out as one, so no part of a secret stays whatever the order. Returns the data and
    the count of stretches left out."""
    spans = []
    for s in secrets:
        i = data.find(s)
        while i >= 0:
            spans.append((i, i + len(s)))
            i = data.find(s, i + 1)
    for pattern in SET_BY_NAME:
        at = 0
        while m := pattern.search(data, at):
            if not SECRET_NAME.search(m.group(1).decode()):
                at = m.start(2)   # its value may set a name itself (args: api_key=..., --port --api-key ...)
                continue
            (a, b), at = m.span(2), m.end()
            if data[a:a + 1] in (b'"', b"'"):   # a quoted value: its inside
                a, b = a + 1, b - 1
            if b > a:
                spans.append((a, b))
    spans += [m.span(1) for m in URL_USER.finditer(data)]
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    parts, at = [], 0
    for a, b in merged:
        parts += [data[at:a], LEFT_OUT]
        at = b
    return b"".join(parts) + data[at:], len(merged)


def copy_model_service(endpoints_file: Path, api_key_file: str | None, out: Path) -> dict:
    """record.extras with --llm openai: the model servers' setup files, from the folder of the model job's endpoints
    file (AUTONMI's FLEET_DIR: endpoints, gpu.csv, each server's vllm-<i>.log, and what a launch script puts
    there), copied into out/model_service/ with every secret left out. When that folder holds a SETUP_LIST (the
    StellarAI launch script's, beside the endpoints file it combines from one or several model jobs' folders),
    only the files it names are copied, at their paths relative to the folder; else every file of the folder:
    - not copied: the servers' key (api-key, and the --api-key-file), any file whose name says it holds a secret
      (SECRET_FILE), anything that is not a plain file, and a listed path outside the folder;
    - a log (.log, .out, .err) is copied as its first LOG_HEAD_LINES lines;
    - environment.txt, this process's environment: only the variables of ENV_PREFIXES and ENV_NAMES, and none
      whose name says it holds a secret (VLLM_API_KEY among them);
    - in every copied file each secret is replaced by LEFT_OUT (leave_out): the servers' key and the value
      (SECRET_MIN characters or more) of every environment variable whose name says it holds a secret, wherever
      they are; any value set by a secret-named name (HF_TOKEN=..., "api_key": "...", --api-key ...), whatever
      its length; and a URL's user and password. A secret under a name that does not say so, in a file whose
      name does not say so, and not in this process's environment, cannot be known, and is copied when the folder
      has no SETUP_LIST (with one, only the files it names).
    Then every copied file is checked for the key and those variables' values, and one that still holds one is
    removed. No secret is printed or returned. Returns, for the record: the files copied, those not copied, the
    secrets left out (a count of stretches) and the files the check removed (none expected)."""
    src, dest = endpoints_file.parent, out / "model_service"
    dest.mkdir(parents=True, exist_ok=True)
    key_files = {src / "api-key", *([Path(api_key_file)] if api_key_file else [])}
    secrets = {os.fsencode(v) for k, v in os.environ.items() if SECRET_NAME.search(k) and len(v) >= SECRET_MIN}
    for f in key_files:
        with contextlib.suppress(OSError):
            secrets.add(f.read_bytes().strip())
    secrets.discard(b"")
    copied, skipped, left_out = [], [], 0

    def put(name: str, data: bytes) -> None:
        nonlocal left_out
        data, n = leave_out(data, secrets)
        left_out += n
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        (dest / name).write_bytes(data)
        copied.append(name)
    files, listed, root = [], src / SETUP_LIST, src.resolve()
    if listed.is_file():
        for line in listed.read_text().splitlines():
            pattern = line.strip()
            if not pattern or pattern.startswith("#"):
                continue
            inside = not Path(pattern).is_absolute() and ".." not in Path(pattern).parts
            found = sorted(src.glob(pattern)) if inside else []
            if not found:
                skipped.append(pattern)   # none such, or outside the folder
            files += [f for f in found if f not in files]
    else:
        files = sorted(src.iterdir())
    for f in files:
        name = f.relative_to(src).as_posix()
        if f.is_symlink() or not f.is_file() or SECRET_FILE.search(f.name) or not f.resolve().is_relative_to(root) \
                or any(f.resolve() == k.resolve() for k in key_files):
            skipped.append(name)
            continue
        data = f.read_bytes()
        if f.suffix in (".log", ".out", ".err"):
            data = b"".join(data.splitlines(keepends=True)[:LOG_HEAD_LINES])
        put(name, data)
    env = sorted(f"{k}={v}" for k, v in os.environ.items()
                 if (k.startswith(ENV_PREFIXES) or k in ENV_NAMES) and not SECRET_NAME.search(k))
    put("environment.txt", os.fsencode("\n".join(env) + "\n"))
    removed = [f.relative_to(dest).as_posix() for f in sorted(dest.rglob("*"))
               if f.is_file() and any(s in f.read_bytes() for s in secrets)]
    for name in removed:
        (dest / name).unlink()
    return {"folder": "model_service", "files": sorted(set(copied) - set(removed)), "not_copied": skipped,
            "secrets_left_out": left_out, "removed_by_the_check": removed}


# --- the mini-app ----------------------------------------------------------------------------------------

def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=int, help="campaigns (default: the knob file's agents.count)")
    p.add_argument("--preset", choices=knob_files.PRESETS, help="merged over the smoke preset")
    p.add_argument("--knobs", action="append", default=[], metavar="FILE",
                   help="a knob file (YAML) merged over the preset; repeatable, applied in order")
    p.add_argument("--sim", choices=("real", "fixture"), default="real")
    p.add_argument("--llm", choices=("stub", "openai"), help="overrides llm.mode")
    p.add_argument("--out", help="output directory (default: results/<time>-autonmi-<agents>-<pid>)")
    p.add_argument("--populations", type=int, help="populations of campaigns; must divide the campaigns "
                                                   "(default: the smallest count that does from run.py's agent "
                                                   "processes, ceil(agents.count / agents.per_process), up to "
                                                   "twice that; else refused)")
    p.add_argument("--workers", type=int, help="simulation workers per population "
                                               "(default: workers.sim_workers / populations)")
    p.add_argument("--runtime", choices=("local", "dragon"), default="local",
                   help="dragon: populations as Dragon processes; run under `python -m dragon -s`")
    p.add_argument("--endpoints-file", help="--llm openai: the model job's servers, one base URL per line "
                                            "(without /v1); each must answer /health")
    p.add_argument("--api-key-file", help="--llm openai: a file holding the servers' key, if they need one")
    p.add_argument("--model-wait", type=float, default=1800.0, help="seconds to wait for every server to answer")
    p.add_argument("--no-vllm-metrics", action="store_true",
                   help="--llm openai: do not sample the servers' /metrics (no decoding counts, and none of "
                        "AUTONMI's checks against the servers' counters, which a server restart, a requeue, a "
                        "request the client gave up on, or servers shared with another run fail)")
    p.add_argument("--stagger-batch", type=int, default=500, help="first-wave campaign starts per batch")
    p.add_argument("--stagger-interval", type=float, default=0.01, help="seconds between those batches")
    return p.parse_args(argv)


def load_knobs(a: argparse.Namespace) -> dict:
    """As run.py loads them."""
    knobs = knob_files.load(preset=a.preset, paths=a.knobs)
    over = {"agents": {"count": a.agents} if a.agents is not None else {}, "llm": {"mode": a.llm} if a.llm else {}}
    knobs = knob_files.validate({**knobs, **{s: {**knobs[s], **v} for s, v in over.items()}},
                                modes=("generations", "async"))   # the asynchronous mode runs on this layer only
    if knobs["llm"]["mode"] == "openai" and knobs["llm"]["model"] == "stub":
        raise knob_files.KnobError('llm.mode is openai but llm.model is still "stub"; set llm.model in a knob file')
    if knobs["llm"]["mode"] == "openai" and decide.asynchronous(knobs) and knobs["llm"]["thinking"] is None:
        raise knob_files.KnobError("llm.mode openai in the asynchronous mode needs llm.thinking set (off or on): "
                                   "unset, the server's chat template decides whether the model thinks")
    return knobs


def _divides(n: int, lo: int, hi: int) -> int | None:
    """The smallest population count from lo to hi that divides n campaigns, or None."""
    return next((p for p in range(lo, min(hi, n) + 1) if n % p == 0), None)


def layout(knobs: dict, sim: str, populations: int | None = None, workers: int | None = None) -> tuple[int, int]:
    """Populations, and simulation workers in each. AUTONMI cuts the agent list into contiguous equal blocks
    (divmod), so a campaign's agents stay in one population only when the populations divide the
    campaigns: by default (None or 0) the smallest count that does from run.py's agent processes
    (agents.per_process campaigns each) up to twice that, else refused with nearby campaign counts that have
    one; a --populations that does not divide is refused. Workers: by default the population's share of
    workers.sim_workers, which with --sim real must give each one, as run.py requires."""
    n, w = knobs["agents"]["count"], knobs["workers"]["sim_workers"]
    whole = "a campaign's agents must stay in one population, and AUTONMI splits the agents into equal blocks"
    if populations:
        populations = min(populations, n)
        if populations < 1 or n % populations:
            raise knob_files.KnobError(f"--populations {populations} does not divide the {n} campaigns: {whole}; "
                                       f"give a count that divides {n}")
    else:
        d = min(-(-n // knobs["agents"]["per_process"]), n)
        populations = _divides(n, d, 2 * d)
        if populations is None:
            below = next(m for m in range(n - 1, 0, -1) if _divides(m, d, 2 * d))
            above = next(m for m in range(n + 1, 2 * n + 2) if _divides(m, d, 2 * d))
            raise knob_files.KnobError(
                f"{n} campaigns divide by no population count from {d} (agents.per_process) to {2 * d}: {whole}; "
                f"run {below} or {above} campaigns, or give --populations")
    if workers is None:
        if sim == "real" and populations > w:
            raise knob_files.KnobError(
                f"{populations} populations each need a simulation worker, but workers.sim_workers is {w}; raise "
                "workers.sim_workers or agents.per_process, or give --workers")
        workers = max(1, w // populations)
    return populations, workers


def agents_live(knobs: dict) -> int | None:
    """AUTONMI's cap on the hosted agents live at once per population (None: all): agents.max_live keeps its
    meaning, campaigns, so 3 + campaign.slots agents each. The host starts agents in order and the next only when one ends,
    so a cap below one campaign would leave a campaign's first agents waiting for good on peers that cannot
    start, and agents.start_together on any cap would leave every agent waiting for those past it: both are
    refused. With whole campaigns at most one is partly started, and its other agents start as others end."""
    a = knobs["agents"]
    if a["max_live"] is None:
        return None
    if a["start_together"]:
        raise knob_files.KnobError(f"agents.start_together needs agents.max_live: null, not {a['max_live']}: every "
                                   "agent waits until all have started, and with a cap those past it start only as "
                                   "others end")
    if a["max_live"] < 1:
        raise knob_files.KnobError(f"agents.max_live must hold one campaign ({agents_per_campaign(knobs)} agents) at "
                                   f"least, not {a['max_live']}")
    return agents_per_campaign(knobs) * a["max_live"]


def own(out: Path) -> None:
    """--out as run.py takes it: not a folder that holds a run (the parent's logs append), and held by
    run.lock, also against a second run started at the same moment."""
    used = [p.name for p in out.glob("*") if p.name in ("run_record.json", "agents.jsonl", "campaigns", "run.lock")
            or p.name.startswith(("agents-", "part-"))] if out.is_dir() else []
    if used:
        raise knob_files.KnobError(f"--out {out} already holds a run ({', '.join(sorted(used))}); "
                                   "give a new directory")
    out.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(out / "run.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise knob_files.KnobError(f"--out {out} is in use by another run (run.lock); give a new directory") from None


async def make_backend(a: argparse.Namespace):
    """The mini-app's own backend, which runs the populations: RHAPSODY's native one under Dragon (each
    population a process of its own), a thread pool locally (the populations are blocks of this process)."""
    if a.runtime == "dragon":
        from rhapsody.backends import DragonNativeExecutionBackend

        return await DragonNativeExecutionBackend(workers_per_node=2)
    from rhapsody.backends import ConcurrentExecutionBackend

    return await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=4))


async def run_miniapp(a: argparse.Namespace, knobs: dict, out: Path, work_root: str | None, live: LiveLine) -> int:
    backend = await make_backend(a)
    flow = await WorkflowEngine.create(backend=backend, work_dir=str(out))
    telemetry = await flow.start_telemetry(checkpoint_path=str(out / "launcher"))
    app = MiniApp(flow, telemetry)
    ok = False
    try:
        async with app, flow.workflow_scope("fusion"):
            ok = await miniapp_workflow(a, knobs, flow, out, app, work_root, live)
    finally:
        try:
            await flow.shutdown()
        finally:
            await telemetry.stop()
            app.verify()
    report = app.report
    print(f"[fusion/autonmi] checks {'pass' if ok and report.checks_pass else 'FAIL'}; telemetry "
          f"{'complete' if report.telemetry_complete else 'INCOMPLETE'}; {out}", file=sys.stderr, flush=True)
    return 0 if ok and report.exit_hint == 0 else 1


async def miniapp_workflow(a, knobs: dict, flow, out: Path, app, work_root: str | None, live: LiveLine) -> bool:
    k = knobs["llm"]
    model, context = None, []
    if k["mode"] == "openai":   # the model job's servers, once all answer; the thinking budget is per call
        from autonmi.models import THINKING_OFF, wait_for_servers

        # llm.thinking off: chat_template_kwargs {enable_thinking: false} on every call (the transports' own)
        model = await wait_for_servers(a.endpoints_file, a.api_key_file, wait_s=a.model_wait, name=k["model"],
                                       timeout_s=k["timeout_s"], outage_s=k["outage_s"],
                                       thinking_off=THINKING_OFF if k["thinking"] == "off" else {})
        live.served = await asyncio.to_thread(check_served_model, model, k["model"], context)   # before any campaign
        if a.no_vllm_metrics:
            live.why_no_sampler = "--no-vllm-metrics"
        else:
            live.sampler, live.why_no_sampler = await asyncio.to_thread(vllm_sampler, model)
        if live.sampler is None and not a.no_vllm_metrics:
            print(f"[fusion/autonmi] warning: the model servers' decoding counts are not recorded "
                  f"({live.why_no_sampler})", file=sys.stderr, flush=True)
    live.servers = list(model["endpoints"]) if model else None   # each fleet's replicas, in order (None: the stub)
    definition = importlib.import_module(MODULE).Fusion(knobs, a.sim, out, work_root)
    definition.live = live.counts
    definition.context = min(context) if context else None   # the prompt-length check (Script._ask)
    definition.transport = transport_settings(model)   # record.extras: the request settings sent
    # locally, the populations' simulations run on run.py's pools; under Dragon, on RHAPSODY's native backend
    executor = f"{MODULE}:{'sim_pool' if a.sim == 'real' else 'fixture_pool'}" if a.runtime == "local" else None
    campaign = Campaign(
        flow,
        definition,
        CampaignConfig(
            run_dir=str(out), populations=a.populations, runtime=a.runtime,
            population=PopulationConfig(
                # a campaign's agents start together, so campaigns start as fast as one agent each did
                workers=a.workers, stagger_batch=agents_per_campaign(knobs) * a.stagger_batch,
                stagger_interval_s=a.stagger_interval, cap_tokens=k["admission_tokens"],
                cap_requests=k["max_in_flight"], max_live=agents_live(knobs),
                executor=executor, model=model),
        ),
        app=app,
    )
    try:
        await campaign.start_services()
        try:
            async with live.during(out):
                await campaign.run()
        except asyncio.CancelledError:
            # SIGTERM: the campaign cancels its populations, but neither waits for them nor records why, as it
            # does for populations it cancels itself. Each still shuts its engine down and writes its
            # telemetry and its report into the run's; wait for that (at most cancel_grace_s), so that the
            # run's verify() sees every population, and asyncio.run does not cancel one again mid-write. A
            # population cancelled before build (while its engine starts) is not waited for: it reports
            # nothing, and the run's telemetry says so. Under Dragon, build runs in the populations' own
            # processes and nothing is waited for here. Then the check the campaign records for its own
            # cancels, every_population_ran, fails with the populations the signal cancelled (under Dragon,
            # all of them: this process cannot tell which had ended).
            pops, grace, t = definition.populations, campaign.config.cancel_grace_s, time.perf_counter()
            names = [f"p{i:03d}" for i in range(campaign.config.populations)]
            cancelled = [n for n in names if n not in pops or not pops[n].done()]
            pending = [pops[n] for n in cancelled if n in pops]
            if pending:
                await asyncio.wait(pending, timeout=grace)
            running = [n for n in cancelled if n in pops and not pops[n].done()]
            if running:
                print(f"[fusion/autonmi] warning: {running} still running {grace} s (cancel_grace_s) after the "
                      "signal: their telemetry may be cut short", file=sys.stderr, flush=True)
            app.check("every_population_ran", ok=False, detail={
                "failed": {}, "cancelled": cancelled, "timed_out": [], "interrupted": True,
                "waited_s": round(time.perf_counter() - t, 1), "cancel_grace_s": grace, "still_running": running})
            raise
        finally:
            await campaign.stop_services()
        return campaign.check()
    finally:
        await campaign.close()


async def run_until_signalled(a, knobs: dict, out: Path, work_root: str | None,
                              live: LiveLine) -> tuple[int, bool, dict]:
    """run_miniapp, which SIGTERM or SIGINT cancels through the event loop, as run.run_slice is cancelled:
    the populations, their engines and pools close, and the run is recorded as interrupted. Returns
    run_miniapp's exit hint, whether the run was interrupted, and the event loop's lag (record.LoopLag)."""
    loop, me, stopped = asyncio.get_running_loop(), asyncio.current_task(), []
    pools = importlib.import_module(MODULE)._POOLS

    def on_term() -> None:
        if not stopped:
            stopped.append(time.time())
            for pool in pools:   # their workers are about to get the same signal: do not replace them
                pool.stop_rebuilds()
            me.cancel()
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in before:   # a plain handler that queues on_term, as run.run_slice installs
        with contextlib.suppress(ValueError):   # not the main thread: skip
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(on_term))
    # Every AsyncFlow engine registers SIGTERM, SIGINT and SIGHUP on its event loop for a shutdown of its
    # own, the last engine created winning (here, one population's): the run would end as that one
    # population's failure. This run owns its signals, as run.py does, so its loop declines them.
    register = loop.add_signal_handler
    loop.add_signal_handler = lambda sig, *args: None if sig in _OWN_SIGNALS else register(sig, *args)
    lag = record.LoopLag()
    lag.start()
    rc = 1
    try:
        rc = await run_miniapp(a, knobs, out, work_root, live)
    except asyncio.CancelledError:
        if not stopped:
            raise
    finally:
        if stopped and me.cancelling():   # the cancellation came from on_term (and may have been absorbed
            me.uncancel()                 # below, where a host ends its agents as cancelled)
        for sig, old in before.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(sig, signal.SIG_IGN if stopped else old)
        loop_lag = await lag.stop()
    return rc, bool(stopped), loop_lag


def probe(a: argparse.Namespace) -> dict | None:
    """record.device_probe where the simulations run, as run.Services.probe takes it: in a spawned process
    with --sim real, so that a broken torch or device stops the run before any campaign starts. None for
    the fixture, and under Dragon, whose simulation workers this process cannot reach."""
    if a.sim != "real" or a.runtime != "local":
        return None
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(record.device_probe, True).result()


def llm_service(knobs: dict, lines: list[dict], sums: dict | None = None) -> dict:
    """These campaigns' model calls, as llm.py's recorder reports an agent process's (seen()): those in their
    lines' details, and those the population summed instead (`sums`, CallSums.read)."""
    details = [c for r in lines for c in r.get("llm_call_details", [])]
    ok = [c for c in details if not c.get("error")]
    n, sums = Counter(), sums or {}
    n.update(sums.get("n", {}))
    return {"mode": knobs["llm"]["mode"], "calls": len(details) + n["calls"],
            "budgets": dict(Counter(str(c["budget"]) for c in details) + Counter(sums.get("budgets", {}))),
            "empty": sum(bool(c["empty"]) for c in ok) + n["empty"], "errors": len(details) - len(ok) + n["errors"],
            "retries": 0,   # AUTONMI's fleet waits out a server instead (layer.model_outages)
            "finish_reasons": dict(Counter(c["finish_reason"] for c in ok) + Counter(sums.get("finish_reasons", {}))),
            "tokens": {k: sum(c.get(f"{k}_tokens") or 0 for c in ok) + sums.get("tokens", {}).get(k, 0)
                       for k in ("prompt", "completion", "reasoning")},
            "latency_s": [c["latency_s"] for c in details] + sums.get("latency_s", [])}


def part(a: argparse.Namespace, knobs: dict, index: int, lines_file: Path, *, loop_lag, device, threads) -> dict:
    """Population `index` as record.write takes an agent process of run.py (run.Services.part): its campaign
    lines, and its counts, from them and from AUTONMI's population.json (written when it ends; an
    interrupted population has none). The counts from the lines are those of the campaigns that reported. The
    model calls not in the lines (CallSums) count in llm_service, and their times and tokens (`call_times`) in
    record.write's client-side counts and tokens per second; with record.extras its requests' settings
    (`requests`)."""
    lines = [json.loads(x) for x in lines_file.read_text().splitlines() if x]
    sums = CallSums.read(sums_file(lines_file, index))   # the calls not in the lines
    done = [r for r in lines if "error" not in r]
    stats_file = Path(a.out) / "populations" / f"p{index:03d}" / "population.json"
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
    runner, local = stats.get("runner") or {}, a.runtime == "local"
    survival = {"survival": {k: sum(r["survival"][k] for r in done) for k in ASYNC_COUNTS}} \
        if decide.asynchronous(knobs) else {}   # the reported campaigns' results scored and kept
    return {**survival,
        "agents_file": str(lines_file), "llm_service": llm_service(knobs, lines, sums),
        "call_times": None if sums is None else {"calls": sums["n"].get("calls", 0),
                                                 "retried": sums["n"].get("retried", 0), "times": sums["times"]},
        "requests": (sums or {}).get("requests"),   # record.extras: the request settings sent (write_record)
        "tool_calls": dict(Counter(what for _ in done for what in decide.lookups(knobs))),
        "sim_tasks": runner.get("simulations", sum(r["n_sims"] for r in done)),
        # one memory write per generation, or per Coordinator cycle in the asynchronous mode
        "memory_writes": sum(r["n_cycles"] if "n_cycles" in r else len(r["generations"]) for r in done),
        "reports": len(done),
        "layer": {"layer": "autonmi", "population": index, "runtime": a.runtime, "complete": bool(stats),
                  "suspensions": runner.get("suspensions"), "runner": runner, "tasks": stats.get("tasks"),
                  "outcomes": stats.get("outcomes"), "wall_s": stats.get("wall_s"),
                  "model_outages": stats.get("model_outages")},
        "sim_workers": a.workers, "sim_pool": ("process" if a.sim == "real" else "thread") if local else "dragon",
        "threads": threads, "pool": None, "pruned_tasks": 0,
        "backend": "ConcurrentExecutionBackend" if local else "DragonNativeExecutionBackend",
        # locally every population runs on this process's event loop; under Dragon, in processes of their own
        "loop_lag": loop_lag if local and index == 0 else None,
        "peak_rss_mb": record.peak_rss_mb() if local and index == 0 else None,
        "device": device or record.device_probe(), "frameworks": record.frameworks(),
    }


def framework_versions() -> dict:
    """AUTONMI's framework, AsyncFlow and wfMiniAPI, which record.versions() leaves out: their versions,
    and their commits when installed from git."""
    out = {}
    for dist in (*metadata.packages_distributions().get("autonmi", []), "radical.asyncflow", "wfMiniAPI"):
        try:
            d = metadata.distribution(dist)
        except metadata.PackageNotFoundError:
            continue
        vcs = json.loads(d.read_text("direct_url.json") or "{}").get("vcs_info", {})
        out[d.metadata["Name"]] = {"version": d.version, "commit": vcs.get("commit_id")}
    return out


def autonmi_commit() -> dict:
    """The commit of the autonmi package this process imported: git's, in the clone it is imported from
    (an editable install), with whether tracked files differ; else its installer's record (direct_url.json,
    when installed from git). None when neither says."""
    import subprocess

    import autonmi

    src = Path(autonmi.__file__).resolve().parent

    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    try:
        git("ls-files", "--error-unmatch", "__init__.py")   # the package's own clone, not a repository above it
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "from": "git", "path": str(src)}
    except (OSError, subprocess.SubprocessError):
        pass
    for dist in metadata.packages_distributions().get("autonmi", []):
        vcs = json.loads(metadata.distribution(dist).read_text("direct_url.json") or "{}").get("vcs_info", {})
        if vcs.get("commit_id"):
            return {"commit": vcs["commit_id"], "dirty": None, "from": "direct_url.json", "path": str(src)}
    return {"commit": None, "dirty": None, "from": None, "path": str(src)}


def autonmi_pin() -> dict:
    """autonmi_commit() against AUTONMI_COMMIT, for the run record; one warning on stderr when the commit
    differs (or is unknown) or its tracked files are changed. The clone's path is in the warning only: run
    records name no local path."""
    got = autonmi_commit()
    path = got.pop("path")
    got.update(pin=AUTONMI_COMMIT, matches_pin=got["commit"] == AUTONMI_COMMIT and not got["dirty"])
    if not got["matches_pin"]:
        what = "an unknown commit" if got["commit"] is None else got["commit"][:12] + (
            " with changed files" if got["dirty"] else "")
        print(f"[fusion/autonmi] warning: the autonmi package imported ({path}) is at {what}, not at the "
              f"pinned {AUTONMI_COMMIT[:12]} (layer_autonmi.AUTONMI_COMMIT)", file=sys.stderr, flush=True)
    return got


TASK_COUNTS = ("created", "completed", "failed", "canceled", "abandoned", "not_submitted")


def telemetry_verdict(summary: dict) -> dict:
    """AUTONMI's verdict on the run's telemetry (its check telemetry_complete, in summary.json) and the
    task counts of the populations that reported into the run, each and summed. `abandoned`: tasks
    submitted that never ended, as when SIGTERM stops a run; workflow-mini-apps reports them, and they do
    not by themselves make the telemetry incomplete. `tasks` is None when no population reported."""
    check = next((c for c in summary.get("checks", []) if c["name"] == "telemetry_complete"),
                 {"ok": False, "detail": ["no telemetry_complete check in summary.json"]})
    pops = {p["population"]: {"complete": p.get("telemetry_complete"), "tasks": p.get("tasks")}
            for p in summary.get("per_population", [])}
    tasks = {k: sum((p["tasks"] or {}).get(k, 0) for p in pops.values()) for k in TASK_COUNTS} if pops else None
    return {"complete": bool(check.get("ok")), "reasons": check.get("detail", []), "tasks": tasks,
            "populations": pops}


def references(knobs: dict) -> dict:
    """What the counts of agents and model requests are measured against: the hosted agents (3 +
    campaign.slots per campaign). (No comparison against the client's admission caps: Matt's round-4 answer 11.)"""
    return {"hosted_agents": agents_per_campaign(knobs) * knobs["agents"]["count"]}


def _rows(out: Path) -> list[dict]:
    """AUTONMI's rows, one per hosted agent (app/agents-p*.jsonl; a population the run cancelled wrote none)."""
    return [json.loads(x) for f in sorted((out / "app").glob("agents-p*.jsonl"))
            for x in f.read_text().splitlines() if x.strip()]


def by_role(rows: list[dict]) -> dict:
    """counts.by_role, from AUTONMI's rows: each role's agents, those that ended normally, their model calls
    (the fleets' count of replies; and the layer's of calls asked, tries and fallbacks: Script._ask), lookups,
    messages sent, received and dropped (a failed campaign's), suspensions
    (AUTONMI's: the Monitors', one per evaluation unless it had ended when the Monitor asked), the
    evaluations the Monitors submitted, the inner-loop traces they got (monitor.capture) and the reports they
    sent, and each agent's seconds waiting for messages (their sum, and n, mean, median, 95th percentile and
    maximum over the agents)."""
    out = {}
    for role in ROLES:
        rs = [r for r in rows if r.get("role") == role]
        waits = [r.get("message_wait_s") or 0.0 for r in rs]
        out[role] = {"agents": len(rs), "finished": sum(bool(r.get("finished")) for r in rs),
                     **{k: sum(r.get(k) or 0 for r in rs) for k in ("model_calls", "model_asks", "model_tries",
                                                                     "model_fallbacks", "lookups", "messages_sent",
                                                                     "messages_received", "messages_dropped",
                                                                     "suspensions", "simulations", "traces",
                                                                     "reports")},
                     "message_wait_s": {"sum": round(sum(waits), 3), **record._pct(waits)}}
    return out


ASYNC_COUNTS = ("results_scored", "results_kept", "results_scored_after_warmup", "results_kept_after_warmup")


def asynchronous_counts(parts: list[dict], rows: list[dict]) -> dict:
    """counts.asynchronous: the results scored and kept, in all and after warm-up (campaign.warmup: the first
    results are kept whatever their score), of the campaigns that reported (their lines, as counts.sims);
    and from AUTONMI's rows each role's cycles, the Generators' waits for their first survivors, and their
    proposals by the parameters (0 to 7) their model calls filled (None without rows: a population the run's
    interruption cancelled writes none)."""
    out = {k: sum(p["survival"][k] for p in parts) for k in ASYNC_COUNTS}
    if not rows:
        return {**out, "cycles": None, "survivor_waits": None, "parameters_filled": None}
    generators = [r for r in rows if r.get("role") == "generator"]
    return {**out, "cycles": {role: sum(r.get("cycles") or 0 for r in rows if r.get("role") == role) for role in ROLES},
            "survivor_waits": sum(r.get("survivor_waits") or 0 for r in generators),
            "parameters_filled": [sum((r.get("parameters_filled") or [0] * 8)[k] for r in generators) for k in range(8)]}


def begin_margin_s(rows: list[dict]) -> float | None:
    """The earliest Coordinator's t_begin (when it began its campaign) minus the last t_start of its
    population's hosted agents, the smallest over the populations: at least 0 when every campaign began
    after every agent of its population had started, as agents.start_together promises; below 0 when one
    began before. None without the times."""
    pops: dict = {}
    for r in rows:
        pops.setdefault(r.get("population"), []).append(r)
    margins = []
    for rs in pops.values():
        starts, begins = [r.get("t_start") for r in rs], [r["t_begin"] for r in rs if r.get("t_begin")]
        if not begins or not all(starts):
            return None
        margins.append(min(begins) - max(starts))
    return round(min(margins), 3) if margins else None


def decoding_metric(knobs: dict, out: Path, rec: dict, summary: dict, refs: dict, live: LiveLine | None,
                    rows: list[dict] | None = None) -> dict:
    """The record's `speed` counts besides the client's exact ones (record.write): agents live, from AUTONMI's
    rows (`rows`, else read from out/app), in all and per role, with how long all were live at once
    (`all_live_s`); AUTONMI's own client-side figures (summary.json), with Little's law across the sources; the event
    loop's lag beside the client count, flagged when high; the model servers' running batches, waiting
    requests and KV-cache use (vllm/vllm.json; null with the stub); the headline (decision 17: the servers'
    running count where there are servers, else the client-side count, emulated with the stub); and the live
    line's series."""
    s, stub = rec["speed"], knobs["llm"]["mode"] == "stub"
    timed = [r for r in (_rows(out) if rows is None else rows) if r.get("t_start") and r.get("t_end")]
    starts, ends = [r["t_start"] for r in timed], [r["t_end"] for r in timed]
    lo, hi = (min(starts), max(ends)) if starts else (None, None)
    agents = None
    if starts:
        n = refs["hosted_agents"]
        agents = {"source": "AUTONMI's rows (app/agents-p*.jsonl): each hosted agent's t_start to t_end",
                  "hosted_agents": n, "timed": len(starts),
                  **record.occupancy(starts, ends, lo, hi, refs={"hosted_agents": n}),
                  "ramp_up_s": round(max(starts) - lo, 3),
                  # every hosted agent live at once: from the last start to the first end
                  "all_live_s": round(max(0.0, min(ends) - max(starts)), 3) if len(timed) == n else 0.0}
        roles = [role for role in ROLES if any(r.get("role") == role for r in timed)]
        if roles:   # per role, against its own agents (one per campaign; the Monitors campaign.slots)
            agents["by_role"] = {}
            for role in roles:
                rs = [r for r in timed if r.get("role") == role]
                b, e = [r["t_start"] for r in rs], [r["t_end"] for r in rs]
                own = knobs["agents"]["count"] * (knobs["campaign"]["slots"] if role == "monitor" else 1)
                agents["by_role"][role] = {
                    "timed": len(rs), **record.occupancy(b, e, lo, hi, refs={"hosted_agents": own}),
                    "ramp_up_s": round(max(b) - min(b), 3)}
    phase, secs = s["agent_phase_s"], summary.get("agent_seconds_by_state") or {}

    def built_in(state: str) -> dict:
        x = secs.get(state)
        return {"agent_seconds": x, "mean": round(x / phase, 4) if x is not None and phase else None}
    opened, queued = s.get("model_requests_in_flight"), s.get("model_requests_queued")
    if opened is not None:
        sampled = (summary.get("agent_states_over_time") or {}).get("model_request") or []
        opened["autonmi_sampled"] = {
            "source": "AUTONMI's summary.json: agent_states_over_time.model_request (Flowgentic's agents in "
                      "MODEL_REQUEST, sampled at each export, about every 10 s) and agent_seconds_by_state."
                      "model_request (its agent-seconds) / agent_phase_s",
            "samples": len(sampled), "max_sampled": max(sampled) if sampled else None, **built_in("model_request")}
        rate = opened["calls_timed"] / phase if phase else None
        mean_s = opened["time_s"].get("mean")
        latency = knobs["llm"]["stub_latency_s"] if stub else None
        agent_s, wall = opened["autonmi_sampled"]["agent_seconds"], rec["wall_seconds"]
        opened["littles_law"] = {   # mean in flight = calls per second x time in flight, three ways
            "autonmi_agent_seconds_per_s": opened["autonmi_sampled"]["mean"],
            "fusion_calls_per_s_x_mean_time_in_flight": round(rate * mean_s, 4) if rate and mean_s else None,
            "calls_per_s_x_stub_latency_s": round(rate * latency, 4) if rate and stub else None,
            "autonmi_agent_seconds_per_wall_s": round(agent_s / wall, 4) if agent_s is not None and wall else None,
            "per": "per second of the agent phase (agent_phase_s), the calls' rate too; the last figure per "
                   "second of the whole run (wall_seconds: start-up and shut-down included)"}
        lag = next((x for x in s["loop_lag"] if x and x.get("samples")), None)
        opened["loop_lag"] = None if lag is None or not mean_s else {
            **{k: lag[k] for k in ("p95_ms", "max_ms", "share_over_100ms")},
            "high": lag["p95_ms"] / 1e3 >= 0.1 * mean_s,
            "rule": "high: the loop's 95th-percentile lag is at least 10% of the mean time in flight, which it "
                    "lands inside (the count then measures the loop as well as the model)"}
    if queued is not None:
        queued["autonmi_built_in"] = built_in("model_queued")
    decoding, why = None, ("the stub LLM: no model servers" if stub else
                           live.why_no_sampler if live is not None and live.why_no_sampler else "no vllm/vllm.json")
    vf = out / "vllm" / "vllm.json"
    if vf.exists():
        raw, series = json.loads(vf.read_text()), []
        for x in raw.get("series", {}).values():   # a sweep cut off by SIGTERM: its whole readings only
            n = min(len(x[k]) for k in VLLM_GAUGES)
            series.append({k: x[k][:n] for k in VLLM_GAUGES})
        hold, missing = hold_s(series, raw.get("interval_s") or 2.0), {}

        def gauge(name: str, scale: float = 1.0, references: dict | None = None) -> dict | None:
            """One gauge summed over the servers (record.summed, each reading counted for at most `hold`), or
            None when a server that answered never gave it (as an older vLLM's other names)."""
            lacking = [i for i, x in enumerate(series) if x["t"] and all(v is None for v in x[name])]
            if lacking:
                missing[name] = f"the servers {lacking} (in vllm.json's order) gave no vllm:{name}"
                return None
            t, v = record.summed([(x["t"], x[name]) for x in series], hold)
            return record.sampled(t, v / scale, lo, hi, references)
        times = sorted(t for x in series for t in x["t"])
        if lo is None and times:   # no rows: the sampler's own time
            lo, hi = times[0], times[-1]
        if times and hi > lo:
            running = gauge("num_requests_running", references=refs)
            decoding, why = None if running is None else {
                "source": "the model servers' /metrics, vllm:num_requests_running summed over the servers, sampled "
                          "by AUTONMI's VllmSampler (vllm/vllm.json)",
                "note": "the servers' running batches: prompt processing included (vLLM has no decode-only "
                        "count); total only (a server does not know which agent sent a request)",
                "servers": len(series), "interval_s": raw.get("interval_s"), "sample_errors": raw.get("errors"),
                "hold_s": round(hold, 3),
                "stale_s": [record.unheld_s(x["t"], lo, hi, hold) for x in series],
                "stale_rule": "a server's reading counts until its next one, at most hold_s (2.5 sampling intervals, "
                              "or 2.5 median gaps); stale_s: each server's seconds of the agent phase with no reading "
                              "counted (it stopped answering /metrics, or before its first), counted as 0",
                "running": running,
                "waiting": gauge("num_requests_waiting"),
                "kv_cache_usage": gauge("kv_cache_usage_perc", scale=len(series)),   # the servers' mean, a fraction
                "gauges_missing": missing or None}, missing.get("num_requests_running")
    if decoding is not None:
        headline = {"figure": "decoding_at_servers", "label": "the servers' running count",
                    "max": decoding["running"]["max"], "mean": decoding["running"]["mean"],
                    # who the servers' /v1/models say serves the model (llm_service.served_models)
                    "served_by": sorted({m["owned_by"] for m in (live.served if live is not None else None) or []
                                         if m["owned_by"]})}
    else:
        label = "client side" + (f", emulated (stub LLM, llm.stub_latency_s = {knobs['llm']['stub_latency_s']} s "
                                 "per call)" if stub else "")
        headline = {"figure": "model_requests_in_flight", "label": label,
                    "max": (opened or {}).get("max"), "mean": (opened or {}).get("mean")}
    return {"agents_live": agents, "decoding_at_servers": decoding, "decoding_at_servers_why_not": why,
            "headline": {**headline, "emulated": stub},
            "live": live.section() if live is not None else None}


def write_record(a, knobs: dict, out: Path, *, wall_s: float, t_run: float, interrupted: bool, crashed: bool,
                 rc: int, loop_lag, device, threads, pin: dict | None = None, live: LiveLine | None = None) -> dict:
    """run_record.json and agents.jsonl by record.write, one part per population; then AUTONMI's verdict
    (summary.json, by autonmi.assemble), its telemetry check with the populations' task counts
    (`telemetry_verdict`), and the framework's commit against its pin (`pin`, autonmi_pin) in its `autonmi`
    section; and the decoding metric's other counts in `speed` (decoding_metric). With record.extras also: each
    role's share of time in a model call (record.write's `model_call_share`), the request settings sent, by role
    and counted (`llm_service.requests`), the model servers in the order of a call's `server` (`llm_service.servers`)
    and a copy of their setup files without their secrets (`model_service/`: copy_model_service). With
    workers.gpu_per_worker, the GPUs the inner-loop workers were placed on (`autonmi.inner_loop_gpus`). A run
    whose AUTONMI checks fail fails."""
    from autonmi.assemble import assemble

    parts, failed = [], []
    for i in range(a.populations):
        f = out / f"agents-p{i:03d}.jsonl"
        if f.exists():
            parts.append(part(a, knobs, i, f, loop_lag=loop_lag, device=device, threads=threads))
        else:   # the population never started
            failed.append(i)
    refs = references(knobs)
    roles = {**dict.fromkeys(ROLES[:3], 1), "monitor": knobs["campaign"]["slots"]}   # agents per campaign
    rec = record.write(out, knobs=knobs, layer="autonmi", sim=a.sim, parts=parts, failed=failed, wall_s=wall_s,
                       interrupted=interrupted, t_run=t_run, model_refs=refs, roles=roles)
    for i in range(a.populations):   # summed into the record (part): not kept beside it
        sums_file(out / f"agents-p{i:03d}.jsonl", i).unlink(missing_ok=True)
    try:
        summary = assemble(out)
    except Exception as exc:   # an interrupted or broken run may lack what AUTONMI assembles
        summary = {"ok": False, "checks": [{"name": "assembled", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}]}
    ok = rc == 0 and bool(summary.get("ok")) and not crashed
    rec["autonmi"] = {"ok": ok, "failed_checks": [c["name"] for c in summary.get("checks", []) if not c["ok"]],
                      "summary": "summary.json", "runtime": a.runtime, "populations": a.populations,
                      "workers_per_population": a.workers, "crashed": crashed, "versions": framework_versions(),
                      "framework_commit": pin, "asyncflow_running_shim": RUNNING_SHIM,
                      "gc": {"automatic_full_collections": full_gc_auto(), "thresholds": list(gc.get_threshold())},
                      "telemetry": telemetry_verdict(summary)}
    rec["counts"]["suspensions"] = sum(p["layer"]["suspensions"] or 0 for p in parts)
    rows, n = _rows(out), knobs["agents"]["count"]
    # counts.agents keeps meaning campaigns (as in the other layers' records); the hosted agents are 3 + slots each
    per = agents_per_campaign(knobs)
    rec["counts"].update(campaigns=n, hosted_agents=per * n, agents_per_campaign=per,
                         roles=list(ROLES), by_role=by_role(rows),
                         expected_by_role={role: {**e, "lookups": len(e["lookups"]), **messages(knobs)[role]}
                                           for role, e in decide.expected_by_role(knobs).items()})
    if decide.asynchronous(knobs):   # the mode, and its counts
        rec["campaign_mode"] = "async"
        rec["counts"]["asynchronous"] = asynchronous_counts(parts, rows)
    rec["speed"].update(decoding_metric(knobs, out, rec, summary, refs, live, rows))
    waits = [r["start_wait_s"] for r in rows if r.get("start_wait_s") is not None]
    rec["speed"]["start_together"] = {"on": knobs["agents"]["start_together"], "coordinator_wait_s": record._pct(waits),
                                      "begin_margin_s": begin_margin_s(rows)}
    if live is not None and live.served is not None:   # what each server's /v1/models said serves llm.model
        rec["llm_service"]["served_models"] = live.served
    if knobs["record"]["extras"]:
        sent_n = Counter()
        for p in parts:
            sent_n.update(p.get("requests") or {})
        rec["llm_service"]["requests"] = [{**json.loads(k), "requests": v} for k, v in
                                          sorted(sent_n.items(), key=lambda kv: (json.loads(kv[0])["role"], -kv[1]))]
        rec["llm_service"]["servers"] = live.servers if live is not None else None
        if knobs["llm"]["mode"] == "openai" and a.endpoints_file:
            try:
                rec["llm_service"]["model_service"] = copy_model_service(Path(a.endpoints_file), a.api_key_file, out)
            except Exception as exc:   # a run refused or cut short before its servers' folder was there, or a file
                # that cannot be read: the record is still written, and names no path
                rec["llm_service"]["model_service"] = {"folder": None, "error": type(exc).__name__}
    if knobs["workers"]["gpu_per_worker"]:   # the GPUs the inner-loop workers were placed on (sim_pool)
        mod = importlib.import_module(MODULE)
        rec["autonmi"]["inner_loop_gpus"] = {"gpus": list(mod._GPUS), "workers_placed":
                                             mod._GPU_COUNTER.value if mod._GPU_COUNTER is not None else 0}
    if rec["status"] == "pass" and not ok:
        rec["status"] = "fail"
    tmp = out / "run_record.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, out / "run_record.json")
    return rec


def main(argv=None) -> int:
    a = parse(argv)
    knobs = load_knobs(a)
    if knobs["llm"]["mode"] == "openai" and not a.endpoints_file:
        raise SystemExit("--llm openai needs --endpoints-file: the model job's servers")
    a.populations, a.workers = layout(knobs, a.sim, a.populations, a.workers)
    agents_live(knobs)   # refuses a cap that could leave agents waiting for good, before anything starts
    pin = autonmi_pin()   # before the run, so that its warning comes first
    n = knobs["agents"]["count"]
    out = Path(a.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}-autonmi-{n}-{os.getpid()}").resolve()
    a.out = str(out)
    own(out)
    t0, t_run = time.perf_counter(), time.time()
    thresholds = gc.get_threshold()   # put back after the run (bound_full_collections)
    work_root, threads = None, None
    if a.sim == "real":
        from fusion_miniapp.sim import work_root as sim_work_root

        # before any worker starts: spawned workers inherit these, as run.Services sets them
        t = str(run.thread_limit(knobs))
        os.environ.update({v: t for v in run.THREAD_VARS}, FUSION_MINIAPP_THREADS=t)
        threads = {"per_worker": int(t), "cpus": run.cpus(), "knob": knobs["sim"]["threads_per_worker"]}
        if not knobs["sim"]["keep_artifacts"]:   # one folder for all evaluations' work, removed at the end
            work_root = tempfile.mkdtemp(prefix="fusion_miniapp_autonmi_", dir=sim_work_root(knobs["sim"]))
    rc, interrupted, crashed, loop_lag, device = 1, False, False, None, None
    live = LiveLine(local=a.runtime == "local")
    try:   # the campaign configures the OpenTelemetry export (CampaignConfig.telemetry's defaults)
        device = probe(a)   # first: a broken torch or device stops the run before any campaign starts
        importlib.import_module(MODULE)._GPUS[:] = inner_loop_gpus(knobs, device)   # sim_pool's (gpu_per_worker)
        rc, interrupted, loop_lag = asyncio.run(run_until_signalled(a, knobs, out, work_root, live))
    except KeyboardInterrupt:   # SIGTERM or Ctrl-C before the event loop took the signals over
        interrupted = True
    except ModelServiceError as exc:   # before any campaign: the reason alone
        print(f"[fusion/autonmi] run refused: {exc}", file=sys.stderr, flush=True)
        crashed = True
    except Exception:           # recorded as a failed run, with what finished
        traceback.print_exc()
        crashed = True
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
    rec = write_record(a, knobs, out, wall_s=time.perf_counter() - t0, t_run=t_run, interrupted=interrupted,
                       crashed=crashed, rc=rc, loop_lag=loop_lag, device=device, threads=threads, pin=pin,
                       live=live)
    gc.set_threshold(*thresholds)
    print(json.dumps({"run_record": str(out / "run_record.json"), "status": rec["status"], "agent_layer": "autonmi",
                      **rec["counts"], "touch_check": rec["touch_check"], "devices": rec["devices"],
                      "wall_seconds": rec["wall_seconds"], "peak_rss_mb": rec["peak_rss_mb"]}), flush=True)
    print(record.speed_line(rec), flush=True)
    for reason in rec["llm_check"]["fail"]:
        print(f"run failed: {reason}", file=sys.stderr)
    for name in rec["autonmi"]["failed_checks"]:
        print(f"run failed: AUTONMI's check {name}", file=sys.stderr)
    return {"pass": 0, "interrupted": 143}.get(rec["status"], 1)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, run._interrupt)   # until the event loop takes the signals over
    sys.exit(main())
