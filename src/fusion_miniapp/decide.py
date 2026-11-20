"""What one agent does: one small IGNOS design campaign (the plan, section 1).

Both agent layers run these functions; no agent framework is imported here. The candidate
sampler, the mutation, the feasibility flags and the best-so-far rule are the parent's own
(parent/evolutionary_loop.py), unchanged. `run_campaign` is the whole workflow; the LangGraph
layer runs the same steps as graph nodes so that the simulation wait can suspend.
"""
from __future__ import annotations

import functools
import json
import math
import re
import sys
from typing import Callable, Protocol

import numpy as np

from fusion_miniapp import PARENT_DIR

MODEL, KIND, NOISE = "three_region", "H", 0.1   # the parent campaign's profile model and mutation amplitude
PROMPT_HISTORY = 8                              # results shown to the model per proposal
KEEP = ("te_core_metric", "te_wall_eV", "ti_wall_eV", "te_ped_eV", "wall_ok", "pedestal_ok", "is_good",
        "touch_check", "device", "fixture")     # what an agent keeps of each simulation result


class Services(Protocol):
    async def llm(self, agent_id: str, messages: list[dict], thinking_budget: int) -> str: ...
    async def simulate(self, agent_id: str, candidate: dict, seed: int) -> dict: ...   # awaits one evaluate
    async def lookup(self, agent_id: str, what: str) -> dict: ...                       # tool call
    async def remember(self, agent_id: str, state: dict) -> None: ...                   # memory / checkpoint
    async def report(self, agent_id: str, result: dict) -> None: ...                    # to the campaign manager


def parent():
    """The parent's evolutionary_loop module (numpy and scipy only; it never imports torch here)."""
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))
    import evolutionary_loop
    return evolutionary_loop


@functools.lru_cache(maxsize=1)
def ranges() -> dict:
    """The seven mutable parameters and their bounds, as the parent campaign clips them."""
    evo = parent()
    r = {k: tuple(evo.get_param_ranges(MODEL, KIND)[k]) for k in evo.MUTABLE_PARAMS[MODEL]}
    r["Delta"] = tuple(evo.DELTA_CLIP)
    return r


def cost_scale(knobs: dict, seed: int) -> float:
    """This agent's cost scale: the knob, or entry seed % len of it when it is a list."""
    c = knobs["agents"]["cost_scale"]
    return c[seed % len(c)] if isinstance(c, list) else c


def budget(knobs: dict, seed: int) -> int:
    """Thinking tokens per LLM call: uniform, or scaled with this agent's simulation cost."""
    a = knobs["agents"]
    return round(a["thinking_budget"] * (cost_scale(knobs, seed) if a["budget_scales_with_cost"] else 1))


def sim_knobs(knobs: dict, seed: int) -> dict:
    """The `sim` section this agent's evaluate() gets: its cost scale multiplies graph size and fixed-point cap."""
    s, c = dict(knobs["sim"]), cost_scale(knobs, seed)
    s["nodes_per_plane"] = max(16, round(s["nodes_per_plane"] * c))
    s["fixed_point_cap"] = max(1, round(s["fixed_point_cap"] * c))
    return s


def new_state(agent_id: str, seed: int, knobs: dict) -> dict:
    """Step 1: the agent's id, seed, starting point (the parent's Latin-hypercube sampler) and budget."""
    evo = parent()
    start = evo.initialize_population(1, MODEL, KIND, seed, delta_bounds=evo.DELTA_CLIP)[0]
    return {"agent_id": agent_id, "seed": seed, "round": 0, "sims": knobs["agents"]["sims_per_agent"],
            "budget": budget(knobs, seed), "start": {k: start[k] for k in ranges()}, "context": {},
            "best": evo.empty_best_result(), "history": [], "touches": {}, "n_llm_calls": 0, "n_tool_calls": 0}


def lookups(knobs: dict) -> list[str]:
    """Step 2: the constraints and an abstract, as the tools knobs choose."""
    t = knobs["tools"]
    return [w for w, on in (("constraints", t["constraints_lookup"]), ("abstract", t["abstract_lookup"])) if on]


async def look(st: dict, svc: Services, names: list[str]) -> dict:
    ctx = dict(st["context"])
    for what in names:
        ctx[what] = await svc.lookup(st["agent_id"], what)
    return {**st, "context": ctx, "n_tool_calls": st["n_tool_calls"] + len(names)}


def _f(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.1f}"


def proposal_messages(st: dict) -> list[dict]:
    """Step 3's prompt. The last line is the request; everything the model may use comes before it."""
    ctx = st["context"]
    system = ("You are the experiment generator of one IGNOS fusion design campaign. You propose H-mode "
              "plasma profile parameters for the three-region profile model. The goal is the highest "
              "te_core_metric among feasible candidates (wall_ok and pedestal_ok both true).")
    if "constraints" in ctx:
        system += " Constraints and parameter ranges: " + json.dumps(ctx["constraints"], sort_keys=True)
    if "abstract" in ctx:
        system += " Background: " + ctx["abstract"]["text"]
    seen = [f"round {h['round'] + 1}: {json.dumps(h['candidate'])} -> te_core_metric={_f(h['te_core_metric'])}, "
            f"te_wall_eV={_f(h['te_wall_eV'])}, te_ped_eV={_f(h['te_ped_eV'])}, feasible={h['is_good']}"
            for h in st["history"][-PROMPT_HISTORY:]]
    notes = "Your results so far:\n" + ("\n".join(seen) or "none yet")
    ask =(f"Round {st['round'] + 1} of {st['sims']}. Reply with one JSON object with numeric values for "
           f"exactly these keys: {', '.join(ranges())}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": notes + "\n\n" + ask}]


def parse_candidate(reply: str) -> dict | None:
    """The mutable parameters of the last JSON object in the reply that has any, clipped to range; else None.
    Taking the last such object skips reasoning text, braces in prose and earlier drafts."""
    reply, decoder = reply or "", json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", reply)]):
        try:
            obj = decoder.raw_decode(reply, i)[0]
        except ValueError:
            continue
        out = {}
        for k, (lo, hi) in ranges().items():
            v = obj.get(k) if isinstance(obj, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out[k] = float(min(max(v, lo), hi))
        if out:
            return out
    return None


def proposed(st: dict, reply: str) -> dict:
    """Step 3: the next candidate from the reply, or the parent's mutation of the best so far."""
    evo = parent()
    base = {k: v for k, v in (st["best"]["params"] or st["start"]).items() if k in ranges()}
    parsed = parse_candidate(reply)
    if parsed:
        candidate, source = {**base, **parsed}, "llm"
    else:
        rng = np.random.default_rng([st["seed"], st["round"]])
        candidate = evo.mutate_params(base, MODEL, KIND, NOISE, rng, delta_bounds=evo.DELTA_CLIP)
        source = "mutation"
    return {**st, "n_llm_calls": st["n_llm_calls"] + 1, "pending": {"candidate": candidate, "source": source}}


def absorb(st: dict, result: dict) -> dict:
    """Step 5: the parent's feasibility flags and best-so-far rule, the history, the touch counts.

    The parent's update_best_result keeps the best feasible core; with nothing feasible it takes the
    latest generation's winner, and here a generation is one candidate. So an infeasible candidate
    replaces an infeasible best only if it wins the parent's own core/wall ranking against it."""
    evo, k, pending = parent(), st["round"], st["pending"]
    rec = {"params": pending["candidate"], **result, "candidate_idx": k}
    old = st["history"][st["best"]["candidate_idx"]] if st["best"]["params"] else None
    wins = old is None or rec["is_good"] or evo._rank_infeasible([old, rec], evo.DEFAULT_WALL_WEIGHT)[0] is rec
    best = evo.update_best_result(st["best"], [rec], [rec], k) if wins else st["best"]
    touches = dict(st["touches"])
    for kind, n in (result.get("touches") or {}).items():
        touches[kind] = touches.get(kind, 0) + n
    entry = {"round": k, **pending, **{key: result.get(key) for key in KEEP}}
    rest = {key: v for key, v in st.items() if key != "pending"}
    return {**rest, "round": k + 1, "best": best, "history": st["history"] + [entry], "touches": touches}


def summary_messages(st: dict) -> list[dict]:
    """Step 7's prompt; its first line states the best candidate."""
    b = st["best"]
    line = (f"{len(st['history'])} candidates evaluated; best: te_core_metric={_f(b['te_core_metric'])}, "
            f"te_wall_eV={_f(b.get('te_wall_eV'))}, te_ped_eV={_f(b.get('te_ped_eV'))}, "
            f"feasible={b['is_good']}, basis={b['selection_basis']}.")
    return [{"role": "system", "content": "You summarise the evidence of one IGNOS fusion design campaign."},
            {"role": "user", "content": line + "\nSummarise in one or two sentences what this campaign found."}]


def finish(st: dict, summary: str) -> dict:
    """Step 8's message to the campaign manager, and the agent's result."""
    return {"agent_id": st["agent_id"], "best": st["best"], "history": st["history"], "summary": summary.strip(),
            "n_llm_calls": st["n_llm_calls"] + 1, "n_sims": len(st["history"]), "n_tool_calls": st["n_tool_calls"],
            "budget": st["budget"], "touches": st["touches"]}


async def run_campaign(agent_id: str, seed: int, knobs: dict, svc: Services,
                       on_state: Callable[[str, str], None] | None = None) -> dict:
    """The eight steps of the plan's section 1, in one coroutine."""
    say = on_state or (lambda a, s: None)
    say(agent_id, "working")
    st = await look(new_state(agent_id, seed, knobs), svc, lookups(knobs))
    while st["round"] < st["sims"]:
        say(agent_id, "model_request")
        reply = await svc.llm(agent_id, proposal_messages(st), st["budget"])
        st = proposed(st, reply)
        say(agent_id, "waiting")
        result = await svc.simulate(agent_id, st["pending"]["candidate"], st["seed"])
        say(agent_id, "working")
        st = absorb(st, result)
        await svc.remember(agent_id, st)
    say(agent_id, "model_request")
    summary = await svc.llm(agent_id, summary_messages(st), st["budget"])
    say(agent_id, "working")
    result = finish(st, summary)
    await svc.report(agent_id, result)
    return result
