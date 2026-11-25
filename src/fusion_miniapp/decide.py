"""What one campaign does: the IGNOS outer loop of the parent's run_evolutionary_loop, split into the
parent's three roles (parent/evo_agents.py): a Coordinator, a Generator and an Evaluator.

Every agent layer gives every campaign these three agents, carries the same messages between them and
calls these functions; no agent framework is imported here. (The AUTONMI layer hosts the three as
separate agents, with Monitor agents that run the Evaluator's inner loops, and splits the two lookups
between the Coordinator and the Generator: LOOKUP_ROLE.)
Every step of the loop is the parent's own
function, unchanged: initialize_population, select_survivors, update_best_result, log_generation,
repopulate, empty_best_result (parent/evolutionary_loop.py).

Messages, the same in both layers (one campaign, P candidates, G generations):

    Coordinator -> Generator  initialize   {"op": "initialize", "context"}             -> population
    for g in 0 .. G-1:
      Coordinator -> Evaluator  evaluate   {"population", "generation": g}             -> P results
      Coordinator: select_survivors, update_best_result, log_generation, in one step
      if g < G-1:
        Coordinator -> Generator  repopulate  {"op": "repopulate", "survivors", "generation", "rng",
                                                "results"}                              -> population
    Coordinator: the summary model call, best_result.json, the report to the campaign manager

A population reply is {"population": [P full parameter dicts], "sources": [...], "rng", "refill"}.

The random generator (knob outer.rng):
- shared: one np.random.default_rng(seed) drives selection and refill, exactly as run_evolutionary_loop.
  The Coordinator owns it and sends its state with each repopulate message; the Generator draws from it
  and sends the state back.
- per_agent: as evo_agents.py, the Generator owns stream 0 and the Coordinator stream 1 of
  SeedSequence(seed).spawn(2).
The state travels as a JSON string (two 128-bit integers, which msgpack checkpoints cannot carry).

The Generator (knob outer.generator; the asynchronous mode refuses mutation unless llm.mode is off):
- mutation: the parent's repopulate, nothing else. The campaign is the parent's loop exactly.
- llm: the parent's repopulate runs first, so the random stream is unchanged, then one model call
  proposes candidates for the slots repopulate filled with mutants (or with a restart sample). Each parsed
  proposal overwrites the mutable values of one of those, in order; the rest of its parameters stay. This
  is the seam evo_agents.py names: "a directed proposer ... can replace random repopulation".

The AUTONMI layer's asynchronous mode (campaign.mode: async, the end of this file) has no generations: each
agent's cycle on the results as they arrive, with the same parent functions, and each agent's own model call
(its prompt and how its reply is read are here; `role_calls`).
"""
from __future__ import annotations

import contextlib
import functools
import io
import json
import math
import re
import sys
import time
from typing import Awaitable, Callable, Protocol

import numpy as np

from fusion_miniapp import PARENT_DIR

MODEL, KIND = "three_region", "H"   # the parent campaign's profile model (validated mode requires it)
# Keys the parent's evaluators return as floats. Results cross processes as JSON, where NaN becomes
# None; the Coordinator puts NaN back before the parent's selection and logging see them.
FLOAT_KEYS = ("te_core_metric", "te_core_keV", "ti_core_keV", "te_wall_eV", "ti_wall_eV", "input_te_core_keV",
              "puff_best", "te_ped_eV", "ne_ped_m3", "p_ped_Pa", "q_peak_proxy", "te_wall_div_eV", "ti_wall_div_eV")
SHOWN = ("te_core_metric", "te_wall_eV", "ti_wall_eV", "te_ped_eV", "wall_ok", "pedestal_ok", "is_good")
SOURCES = ("start", "survivor", "mutation", "model", "restart")

Ask = Callable[[list, int, int], Awaitable[str]]   # ask(messages, thinking_budget, reply_tokens) -> reply text
SUMMARY_TOKENS = 256                                # visible reply allowance of the summary call


def reply_tokens(n: int) -> int:
    """Visible reply allowance of a refill call for n candidates (about 105 tokens per candidate object
    with a Qwen tokenizer; exact counts are tokenizer-specific)."""
    return 64 + 128 * n


class Services(Protocol):
    async def llm(self, agent_id: str, messages: list[dict], thinking_budget: int, reply_tokens: int) -> str: ...
    async def simulate(self, agent_id: str, params: dict, candidate_idx: int, generation: int,
                       seed: int) -> dict: ...                                    # awaits one evaluate
    async def lookup(self, agent_id: str, what: str) -> dict: ...                 # tool call
    async def remember(self, agent_id: str, state: dict) -> None: ...             # memory / checkpoint
    async def report(self, agent_id: str, result: dict) -> None: ...              # to the campaign manager


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
    """This campaign's cost scale: the knob, or entry seed % len of it when it is a list."""
    c = knobs["agents"]["cost_scale"]
    return c[seed % len(c)] if isinstance(c, list) else c


def budget(knobs: dict, seed: int) -> int:
    """Thinking tokens per model call: uniform, or scaled with this campaign's simulation cost."""
    a = knobs["agents"]
    return round(a["thinking_budget"] * (cost_scale(knobs, seed) if a["budget_scales_with_cost"] else 1))


def sim_knobs(knobs: dict, seed: int) -> dict:
    """The `sim` section this campaign's evaluate() gets: its cost scale multiplies graph size and fixed-point cap."""
    s, c = dict(knobs["sim"]), cost_scale(knobs, seed)
    s["nodes_per_plane"] = max(16, round(s["nodes_per_plane"] * c))
    s["fixed_point_cap"] = max(1, round(s["fixed_point_cap"] * c))
    return s


def expected(knobs: dict) -> dict:
    """Per campaign: evaluations, and model calls when every refill has a slot to fill; in the asynchronous mode
    the most its agents ask the model (async_calls), none with llm.mode off."""
    o = knobs["outer"]
    if asynchronous(knobs):
        return {"sims": async_budget(knobs), "llm_calls": sum(async_calls(knobs).values())}
    refills = o["generations"] - 1 if o["generator"] == "llm" and o["population"] > 1 else 0
    return {"sims": o["population"] * o["generations"], "llm_calls": refills + 1}


def role_calls(knobs: dict) -> bool:
    """Whether the roles make their own model calls (the AUTONMI layer): in the asynchronous mode unless llm.mode
    is off; in the generations mode the Monitors' notes and the Evaluator's assessment, with campaign.notes."""
    return knobs["llm"]["mode"] != "off" and (asynchronous(knobs) or knobs["campaign"]["notes"])


# Who makes each lookup where the three roles are separate agents (the AUTONMI layer): the Coordinator keeps
# the constraints, which it sends in its first message as today; the Generator, whose refill prompt reads
# both, makes the abstract lookup itself. The other layers' Coordinator makes both.
LOOKUP_ROLE = {"constraints": "coordinator", "abstract": "generator"}


def expected_by_role(knobs: dict) -> dict:
    """Per campaign, where the roles are separate agents (the AUTONMI layer, whose Monitor slots run the
    inner loops): each role's evaluations (the Monitors' together), model calls (the Generator's at most: a
    refill with no slot to fill is not called) and lookups (their names, in order)."""
    exp, mine = expected(knobs), lookups(knobs)
    by = {role: [w for w in mine if LOOKUP_ROLE[w] == role] for role in ("coordinator", "generator")}
    if asynchronous(knobs):   # the most each role asks the model (no summary call in this mode)
        calls = async_calls(knobs)
        return {"coordinator": {"sims": 0, "llm_calls": calls["coordinator"], "lookups": by["coordinator"]},
                "generator": {"sims": 0, "llm_calls": calls["generator"], "lookups": by["generator"]},
                "evaluator": {"sims": 0, "llm_calls": calls["evaluator"], "lookups": []},
                "monitor": {"sims": exp["sims"], "llm_calls": calls["monitor"], "lookups": []}}
    notes = role_calls(knobs)   # campaign.notes: a note per candidate, an assessment per generation
    return {"coordinator": {"sims": 0, "llm_calls": 1, "lookups": by["coordinator"]},
            "generator": {"sims": 0, "llm_calls": exp["llm_calls"] - 1, "lookups": by["generator"]},
            "evaluator": {"sims": 0, "llm_calls": knobs["outer"]["generations"] if notes else 0, "lookups": []},
            "monitor": {"sims": exp["sims"], "llm_calls": exp["sims"] if notes else 0, "lookups": []}}


# --- the random generator's state as a message ------------------------------------------------------

def rng_dump(rng: np.random.Generator) -> str:
    return json.dumps(rng.bit_generator.state)


def rng_load(state: str) -> np.random.Generator:
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = json.loads(state)
    return rng


def _stream(seed: int, which: int) -> str:
    """Stream `which` of evo_agents.py's SeedSequence(seed).spawn(2) (0: Generator, 1: Coordinator)."""
    return rng_dump(np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[which]))


class _Discard(io.TextIOBase):
    def write(self, s: str) -> int:
        return len(s)


def parent_stdout(knobs: dict):
    """Around the parent's sync calls: its prints kept (keep) or discarded (drop). No await runs inside,
    so swapping sys.stdout here cannot catch another agent's output."""
    return contextlib.nullcontext() if knobs["outer"]["parent_stdout"] == "keep" \
        else contextlib.redirect_stdout(_Discard())


def _plain(params: dict) -> dict:
    return {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in params.items()}


# --- lookups (the Coordinator's tool calls) ----------------------------------------------------------

def lookups(knobs: dict) -> list[str]:
    """The constraints and an abstract, as the tools knobs choose."""
    t = knobs["tools"]
    return [w for w, on in (("constraints", t["constraints_lookup"]), ("abstract", t["abstract_lookup"])) if on]


async def look(svc: Services, agent_id: str, names: list[str]) -> dict:
    return {what: await svc.lookup(agent_id, what) for what in names}


# --- the Generator -----------------------------------------------------------------------------------

def generator_state(agent_id: str, seed: int, knobs: dict) -> dict:
    """The Generator's own memory."""
    return {"agent_id": agent_id, "seed": seed, "context": {}, "n_llm_calls": 0,
            "rng": _stream(seed, 0) if knobs["outer"]["rng"] == "per_agent" else None}


def _f(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.1f}"


def _shown(r: dict) -> str:
    """A result's SHOWN values, as the refill prompt shows them."""
    return ", ".join(f"{k}={_f(r.get(k))}" if isinstance(r.get(k), float) or r.get(k) is None else f"{k}={r[k]}"
                     for k in SHOWN)


def _seven(params: dict) -> dict:
    """A profile's seven mutable parameters, rounded to 6 places."""
    return {k: round(float(params[k]), 6) for k in ranges() if k in params}


def refill_messages(gst: dict, msg: dict, n: int, knobs: dict) -> list[dict]:
    """The refill prompt: this generation's results, then (last line) the request for n candidates."""
    ctx, o = gst["context"], knobs["outer"]
    g, G = msg["generation"], o["generations"]
    system = ("You are the experiment generator of one IGNOS fusion design campaign. You propose H-mode "
              "plasma profile parameters for the three-region profile model. The goal is the highest "
              "te_core_metric among feasible candidates (wall_ok and pedestal_ok both true).")
    if "constraints" in ctx:
        system += " Constraints and parameter ranges: " + json.dumps(ctx["constraints"], sort_keys=True)
    if "abstract" in ctx:
        system += " Background: " + ctx["abstract"]["text"]
    rows = []
    for r in msg["results"]:
        vals = ", ".join(f"{k}={_f(r[k])}" if isinstance(r[k], float) or r[k] is None else f"{k}={r[k]}"
                         for k in SHOWN)
        rows.append(f"candidate {r['candidate_idx']}: {json.dumps(r['params'])} -> {vals}, survived={r['survived']}")
    kept = sum(r["survived"] for r in msg["results"])
    notes = (f"Generation {g + 1} of {G} results (survivors go into the next generation unchanged):\n"
             + "\n".join(rows) + f"\n\n{kept} candidates survive; propose the other {n}.")
    ask = (f"Generation {g + 2} of {G}: reply with a JSON list of {n} objects, each with numeric values "
           f"for exactly these keys: {', '.join(ranges())}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": notes + "\n\n" + ask}]


def _number(v) -> bool:
    """A finite JSON number: not a bool, NaN or infinite, nor an integer too large for a float (on which
    math.isfinite raises OverflowError)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) <= sys.float_info.max


def _clip(obj) -> dict:
    out = {}
    for k, (lo, hi) in ranges().items():
        v = obj.get(k) if isinstance(obj, dict) else None
        if _number(v):
            out[k] = float(min(max(v, lo), hi))
    return out


def parse_candidate(reply: str) -> dict | None:
    """The mutable parameters of the last JSON object in the reply that has any, clipped to range; else None.
    Taking the last such object skips reasoning text, braces in prose and earlier drafts."""
    reply, decoder = reply or "", json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", reply)]):
        try:
            obj = decoder.raw_decode(reply, i)[0]
        except ValueError:
            continue
        out = _clip(obj)
        if out:
            return out
    return None


def parse_candidates(reply: str, n: int) -> list[dict]:
    """Up to n proposals: the usable objects of the last JSON list in the reply that has any, clipped to
    range; else the last single object (parse_candidate); else none."""
    reply, decoder = reply or "", json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\[", reply)]):
        try:
            obj = decoder.raw_decode(reply, i)[0]
        except ValueError:
            continue
        if isinstance(obj, list):
            found = [c for c in (_clip(o) for o in obj) if c]
            if found:
                return found[:n]
    one = parse_candidate(reply)
    return [one][:n] if one else []


async def generator_step(gst: dict, msg: dict, knobs: dict, ask: Ask) -> tuple[dict, dict]:
    """The Generator's two actions, initialize and repopulate: (its new memory, its reply)."""
    evo, o = parent(), knobs["outer"]
    P = o["population"]
    if msg["op"] == "initialize":
        pop = [_plain(p) for p in evo.initialize_population(P, MODEL, KIND, gst["seed"],
                                                             delta_bounds=evo.DELTA_CLIP)]
        return ({**gst, "context": msg.get("context", {})},
                {"population": pop, "sources": ["start"] * len(pop), "rng": None, "refill": None})
    shared = o["rng"] == "shared"
    rng = rng_load(msg["rng"] if shared else gst["rng"])
    survivors = [{"params": p} for p in msg["survivors"]]
    with parent_stdout(knobs):
        pop = [_plain(p) for p in evo.repopulate(survivors, P, MODEL, KIND, o["noise"], rng, msg["generation"],
                                                  delta_bounds=evo.DELTA_CLIP)]
    kept = len(survivors)
    sources = ["survivor"] * kept + ["mutation"] * (len(pop) - kept) if survivors else ["restart"] * len(pop)
    state = rng_dump(rng)                      # after the parent's draws; the model draws nothing from it
    refill = {"asked": 0, "filled": 0, "called": False}
    if o["generator"] == "llm":
        slots = [i for i, s in enumerate(sources) if s != "survivor"]
        if slots:
            reply = await ask(refill_messages(gst, msg, len(slots), knobs), budget(knobs, gst["seed"]),
                              reply_tokens(len(slots)))
            proposals = parse_candidates(reply, len(slots))
            for i, prop in zip(slots, proposals):
                pop[i] = evo.clip_delta({**pop[i], **prop}, evo.DELTA_CLIP)
                sources[i] = "model"
            refill = {"asked": len(slots), "filled": len(proposals), "called": True}
            gst = {**gst, "n_llm_calls": gst["n_llm_calls"] + 1}
    if not shared:
        gst = {**gst, "rng": state}
    return gst, {"population": pop, "sources": sources, "rng": state if shared else None, "refill": refill}


# --- the Evaluator -----------------------------------------------------------------------------------

def evaluation_calls(population: list[dict], generation: int) -> list[dict]:
    """One evaluate call per candidate. The call id carries the generation: FlowGentic returns a call
    made again with the same id from its first run."""
    return [{"id": f"g{generation}c{i}", "params": p, "candidate_idx": i, "generation": generation}
            for i, p in enumerate(population)]


async def evaluate_population(svc: Services, agent_id: str, seed: int, population: list[dict],
                              generation: int) -> dict:
    """The Evaluator's action: all P evaluations submitted at once, then a wait for all of them."""
    import asyncio
    t0 = time.time()
    results = await asyncio.gather(*(svc.simulate(agent_id, c["params"], c["candidate_idx"], generation, seed)
                                     for c in evaluation_calls(population, generation)))
    return {"results": list(results), "t0": t0, "elapsed_s": time.time() - t0}


# --- the Coordinator ---------------------------------------------------------------------------------

def coordinator_state(agent_id: str, seed: int, knobs: dict, context: dict, t_start: float | None = None) -> dict:
    """The Coordinator's own memory. t_start: when the campaign began (before its first lookup)."""
    evo = parent()
    rng = rng_dump(np.random.default_rng(seed)) if knobs["outer"]["rng"] == "shared" else _stream(seed, 1)
    return {"agent_id": agent_id, "seed": seed, "generation": 0, "rng": rng, "best": evo.empty_best_result(),
            "population": None, "sources": None, "context": context, "budget": budget(knobs, seed),
            "gens": [], "counts": {s: 0 for s in SOURCES}, "refill": {"asked": 0, "filled": 0},
            "n_llm_calls": 0, "n_tool_calls": len(context), "touches": {}, "devices": {}, "touch_checks": {},
            "fixture": 0, "sim_s": [], "wait_s": [], "worker_peak_rss_mb": None,
            "t_start": time.time() if t_start is None else t_start}


@functools.lru_cache(maxsize=1)
def _t_core():
    import yaml
    return (yaml.safe_load((PARENT_DIR / "config" / "config.yaml").read_text()) or {}).get("target_core_temperature_keV")


@functools.lru_cache(maxsize=1)
def _provenance() -> dict:
    """The parent config.json's two provenance fields, once per agent process (a declared difference:
    the parent computes them once per campaign, which here would be three git calls per campaign)."""
    evo = parent()
    return {"runtime_source_sha256": evo._runtime_source_sha256(), "git": evo._git_provenance()}


def campaign_config(cst: dict, knobs: dict, sim: str) -> dict:
    """The parent's config.json keys, with this campaign's values (its own cost-scaled sim knobs); `sim`
    says what evaluated it: "real" (the parent's inner loop over the stand-ins) or "fixture"."""
    evo, o, s = parent(), knobs["outer"], sim_knobs(knobs, cst["seed"])
    stand_in = "stand-in" if sim == "real" else "none (fixture: no inner loop, no models)"
    constraints = PARENT_DIR / "config" / "config.yaml"
    method = dict(evo.VALIDATED_METHOD, xgc_ar_steps=s["rollout_steps"], solps_tolerance=0.0,
                  solps_cap=s["fixed_point_cap"],
                  puffs_m3_s=[float(v) for v in np.linspace(0.5e21, 4.0e21, s["puffs"])], name="miniapp_standin")
    return {"population_size": o["population"], "n_generations": o["generations"], "noise_amplitude": o["noise"],
            "wall_weight": o["wall_weight"], "n_gpus": None, "model": MODEL, "kind": KIND, "seed": cst["seed"],
            "inner_loop": "validated" if sim == "real" else "fixture", "profile_mode": "raw_campaign",
            "step_ids": None, "n_core_points": 5, "T_core_keV": _t_core(),
            "n_ar_steps": s["rollout_steps"], "dispatch": "rhapsody", "gpus_per_node": None, "gpu_offset": 0,
            "run_tag": cst["agent_id"], "constraints_path": str(constraints), "matey_model_dir": stand_in,
            "helios_ckpt_path": None, "profile_shaping": None,
            "validated_method": method if sim == "real" else None,
            "conditioned_ckpt": stand_in, "conditioned_ckpt_sha256": None, "conditioned_yaml": stand_in,
            "conditioned_yaml_sha256": None, "conditioned_config_name": stand_in, **_provenance(),
            "driver": "fusion_miniapp", "sim": sim, "sim_knobs": s, "rng": o["rng"], "generator": o["generator"],
            "emulated": True, "note": "all physics output is synthetic (random-weight stand-in models)",
            # the asynchronous mode always has the two random streams, and its own Generator: the parent's mutation,
            # reconciled by its model call unless llm.mode is off
            **({"rng": "per_agent", "generator": "llm" if role_calls(knobs) else "mutation", "campaign_mode": "async"}
               if asynchronous(knobs) else {})}


def start(cst: dict, knobs: dict, run_dir, sim: str = "real") -> dict:
    """The Coordinator's first message, after the parent's config.json (outer.logs: per_agent)."""
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(campaign_config(cst, knobs, sim), indent=2))
    return {"op": "initialize", "context": cst["context"]}


def on_population(cst: dict, reply: dict) -> tuple[dict, dict]:
    """A population from the Generator: remember it, and send it to the Evaluator."""
    counts = dict(cst["counts"])
    for s in reply["sources"]:
        counts[s] += 1
    cst = {**cst, "population": reply["population"], "sources": reply["sources"], "counts": counts}
    if reply["rng"] is not None:              # shared stream: the Generator's draws come back
        cst["rng"] = reply["rng"]
    if reply["refill"]:
        cst["refill"] = {k: cst["refill"][k] + reply["refill"][k] for k in ("asked", "filled")}
        cst["n_llm_calls"] += reply["refill"]["called"]
    return cst, {"population": cst["population"], "generation": cst["generation"]}


def native(result: dict) -> dict:
    """A result as the parent's evaluator returns it: NaN where the parent has a float and the JSON round
    trip left None."""
    return {k: float("nan") if v is None and k in FLOAT_KEYS else v for k, v in result.items()}


def collect(population: list[dict], results: list[dict]) -> list[dict]:
    """The generation as the parent's evaluator returns it: in candidate order, each result's params the
    submitted candidate, and NaN where the parent has a float and the JSON round trip left None. A result
    that still holds its inner-loop trace (trace.py) is refused: the outer loop and its logs never see one."""
    results = sorted(results, key=lambda r: r["candidate_idx"])
    if [r["candidate_idx"] for r in results] != list(range(len(population))):
        raise RuntimeError(f"expected candidates 0..{len(population) - 1}, got "
                           f"{[r['candidate_idx'] for r in results]}")
    out = []
    for r, p in zip(results, population):
        if r["params"] != p:
            raise RuntimeError(f"candidate {r['candidate_idx']}: the result's params are not the submitted ones")
        if "trace" in r:
            raise RuntimeError(f"candidate {r['candidate_idx']}: the result holds its inner-loop trace")
        out.append(native(r))
    return out


def ranked(evaluated: list[dict], wall_weight: float) -> list[dict]:
    """A set of results in the order the parent's select_survivors ranks them: when any meets the constraints
    (is_good), those alone by te_core_metric (its own sort), else all by its _rank_infeasible."""
    good = [e for e in evaluated if e["is_good"]]
    if good:
        return sorted(good, key=lambda e: e["te_core_metric"], reverse=True)
    return parent()._rank_infeasible(evaluated, wall_weight) if evaluated else []


def keep_probabilities(evaluated: list[dict], wall_weight: float) -> list[float]:
    """For each result of a set (as the parent's selection takes a generation: native values, in order), the
    probability that the parent's select_survivors keeps it, ranked as it ranks them (`ranked`). The best 1,
    the worst 0, the others 1 - rank / (n - 1); one it does not rank 0. It draws nothing."""
    ranked_ = ranked(evaluated, wall_weight)
    n, at, q = len(ranked_), {id(e): i for i, e in enumerate(evaluated)}, [0.0] * len(evaluated)
    for rank, e in enumerate(ranked_):
        q[at[id(e)]] = 1.0 if rank == 0 or n <= 1 else 1.0 - rank / (n - 1)
    return q


def on_evaluated(cst: dict, reply: dict, knobs: dict, run_dir) -> tuple[dict, dict | None]:
    """A generation's results from the Evaluator: the parent's selection, best-so-far and logging, in one
    step on the same objects (log_generation marks survivors by object identity). Returns the new memory
    and the repopulate message, or None after the last generation."""
    evo, o = parent(), knobs["outer"]
    g = cst["generation"]
    evaluated = collect(cst["population"], reply["results"])
    rng = rng_load(cst["rng"])
    with parent_stdout(knobs):
        survivors = evo.select_survivors(evaluated, rng, o["wall_weight"])
    best = evo.update_best_result(cst["best"], evaluated, survivors, g)
    if run_dir is not None:
        evo.log_generation(run_dir, g, evaluated, survivors)
    kept = {id(s) for s in survivors}
    touches, devices, checks = dict(cst["touches"]), dict(cst["devices"]), dict(cst["touch_checks"])
    for e in evaluated:
        for kind, n in (e.get("touches") or {}).items():
            touches[kind] = touches.get(kind, 0) + n
        devices[str(e.get("device"))] = devices.get(str(e.get("device")), 0) + 1
        if not e.get("fixture"):
            checks[str(e.get("touch_check"))] = checks.get(str(e.get("touch_check")), 0) + 1
    finite = [e["te_core_metric"] for e in evaluated if math.isfinite(e["te_core_metric"])]
    gen = {"generation": g, "n_evaluated": len(evaluated), "n_good": sum(bool(e["is_good"]) for e in evaluated),
           "n_survivors": len(survivors), "best_te_core_metric": max(finite, default=None),
           "t0": round(reply["t0"], 6), "elapsed_s": round(reply["elapsed_s"], 6)}   # submit, then last result
    t0 = reply["t0"]
    took = [(round(float(e.get("elapsed_seconds") or 0.0), 6), e["candidate_idx"]) for e in evaluated]
    gen["fastest"], gen["slowest"] = ({"candidate_idx": i, "sim_s": t} for t, i in (min(took), max(took)))
    rss = [e["worker_peak_rss_mb"] for e in evaluated if e.get("worker_peak_rss_mb") is not None]
    rss = max(rss + ([cst["worker_peak_rss_mb"]] if cst["worker_peak_rss_mb"] is not None else []), default=None)
    cst = {**cst, "generation": g + 1, "rng": rng_dump(rng), "best": best, "gens": cst["gens"] + [gen],
           "touches": touches, "devices": devices, "touch_checks": checks,
           "fixture": cst["fixture"] + sum(bool(e.get("fixture")) for e in evaluated),
           "sim_s": cst["sim_s"] + [round(float(e.get("elapsed_seconds") or 0.0), 6) for e in evaluated],
           # queue wait: from the generation's submission (on the AUTONMI layer, when the Generator put it on the
           # candidate queue, so with the wait for a free Monitor slot) to the evaluation's start
           "wait_s": cst["wait_s"] + [round(max(0.0, float(e.get("t_start") or t0) - t0), 6) for e in evaluated],
           "worker_peak_rss_mb": rss}
    if g == o["generations"] - 1:
        # the parent writes best_result.json when its loop ends; here too, before the extra summary call,
        # so a model-server outage at the summary does not lose the campaign's result
        if run_dir is not None:
            (run_dir / "best_result.json").write_text(json.dumps(best, indent=2, default=str))
        return {**cst, "population": None, "sources": None}, None
    results = [{"candidate_idx": e["candidate_idx"], "survived": id(e) in kept,
                "params": {k: round(e["params"][k], 6) for k in ranges()},
                **{k: (None if isinstance(e[k], float) and not math.isfinite(e[k]) else e[k]) for k in SHOWN}}
               for e in evaluated]
    msg = {"op": "repopulate", "survivors": [s["params"] for s in survivors], "generation": g,
           "rng": cst["rng"] if o["rng"] == "shared" else None, "results": results}
    return cst, msg


def memory(cst: dict) -> dict:
    """What the Coordinator writes to the agent memory after each generation."""
    return {k: cst[k] for k in ("generation", "best", "gens", "rng")}


def summary_messages(cst: dict) -> list[dict]:
    """The summary prompt; its first line states the best candidate."""
    b, n = cst["best"], sum(g["n_evaluated"] for g in cst["gens"])
    line = (f"{n} candidates evaluated over {len(cst['gens'])} generations; best: "
            f"te_core_metric={_f(b['te_core_metric'])}, te_wall_eV={_f(b.get('te_wall_eV'))}, "
            f"te_ped_eV={_f(b.get('te_ped_eV'))}, feasible={b['is_good']}, basis={b['selection_basis']}.")
    return [{"role": "system", "content": "You summarise the evidence of one IGNOS fusion design campaign."},
            {"role": "user", "content": line + "\nSummarise in one or two sentences what this campaign found."}]


def finish(cst: dict, summary: str, run_dir=None) -> dict:
    """The campaign's result for the campaign manager (best_result.json is already written)."""
    return {"agent_id": cst["agent_id"], "seed": cst["seed"], "best": cst["best"], "generations": cst["gens"],
            "summary": summary.strip(), "sources": cst["counts"], "refill": cst["refill"],
            "n_llm_calls": cst["n_llm_calls"] + 1, "n_sims": sum(g["n_evaluated"] for g in cst["gens"]),
            "n_tool_calls": cst["n_tool_calls"], "budget": cst["budget"], "touches": cst["touches"],
            "devices": cst["devices"], "touch_checks": cst["touch_checks"], "fixture": cst["fixture"],
            "sim_s": cst["sim_s"], "wait_s": cst["wait_s"], "worker_peak_rss_mb": cst["worker_peak_rss_mb"],
            "t_start": round(cst["t_start"], 6), "t_end": round(time.time(), 6)}


def campaign_dir(out, knobs: dict, agent_id: str):
    """Where the parent's per-campaign files go, or None (outer.logs: none)."""
    return None if out is None or knobs["outer"]["logs"] == "none" else out / "campaigns" / agent_id


async def run_campaign(agent_id: str, seed: int, knobs: dict, svc: Services, generator, evaluator, ask: Ask,
                       out=None, on_state: Callable[[str, str], None] | None = None) -> dict:
    """The Coordinator's whole run, for a layer whose agents are called through handles (Academy):
    `generator.initialize(msg)`, `generator.repopulate(msg)`, `evaluator.evaluate(population, generation)`
    are the other two agents' actions. The LangGraph layer runs the same steps as graph nodes."""
    say = on_state or (lambda a, s: None)
    run_dir = campaign_dir(out, knobs, agent_id)
    say(agent_id, "working")
    t_start = time.time()
    cst = coordinator_state(agent_id, seed, knobs, await look(svc, agent_id, lookups(knobs)), t_start)
    cst, msg = on_population(cst, await generator.initialize(start(cst, knobs, run_dir, getattr(svc, "sim", "real"))))
    while True:
        cst, nxt = on_evaluated(cst, await evaluator.evaluate(msg["population"], msg["generation"]), knobs, run_dir)
        await svc.remember(agent_id, memory(cst))
        if nxt is None:
            break
        cst, msg = on_population(cst, await generator.repopulate(nxt))
    say(agent_id, "working")
    result = finish(cst, await ask(summary_messages(cst), cst["budget"], SUMMARY_TOKENS), run_dir)
    await svc.report(agent_id, result)
    return result


# --- the asynchronous mode (campaign.mode: async; the AUTONMI layer) -----------------------------------
#
# No generation barrier: the Generator keeps the campaign's candidate queue fed, a free Monitor slot takes the
# next candidate, and the Evaluator and the Coordinator each apply every result as it arrives. Each agent's
# decision is a pure function here, of its state and what arrived, so one order of arrival, one seed and the
# Monitors' m give one campaign (at monitor.weight 0, the order and the seed alone: m changes no score) with
# llm.mode off, or with the stub when no model call fell back: which Generator takes fell back is a further input,
# and a real model's replies are not recorded. The AUTONMI layer records the orders and each m with record.extras.
# Without model calls (llm.mode off) every agent decides on numbers alone; that path is also each role's fallback when its model call fails (the end of this
# file), so a campaign whose calls all fail is the campaign with llm.mode off.

TAKE = 2   # profiles the Generator takes at a time; it takes them once no candidate waits on its queue
WINDOW_KEYS = ("candidate_idx", "params", "is_good", "te_core_metric", "wall_ok", "te_wall_eV", "ti_wall_eV",
               "te_ped_eV", "pedestal_ok")   # what the parent's ranking and update_best_result read of a result in
                                             # the window, and the Coordinator's prompt shows (SHOWN)


def asynchronous(knobs: dict) -> bool:
    return knobs["campaign"]["mode"] == "async"


def async_budget(knobs: dict) -> int:
    """A campaign's candidates, outer.population x outer.generations, numbered 0 .. budget - 1 as they are made."""
    o = knobs["outer"]
    return o["population"] * o["generations"]


def _window(window: list[dict], result: dict, size: int) -> list[dict]:
    """The window with a result added last, its oldest left out past `size`."""
    return (window + [{k: result.get(k) for k in WINDOW_KEYS}])[-size:]


def async_generator_state(agent_id: str, seed: int, knobs: dict, context: dict | None = None) -> dict:
    """The Generator's memory: its random stream (stream 0 of outer.rng per_agent, whatever outer.rng says), the
    profiles made, the parent's initial population while it is drawn from, the newest survivors message it
    holds (`held`: the message's number, 0 for none), and for its model call its lookups (`context`), the
    Evaluator's latest scores (`scores`: [candidate, score], the last outer.population) and the Coordinator's
    newest guidance."""
    return {"agent_id": agent_id, "seed": seed, "rng": _stream(seed, 0), "made": 0, "initial": None,
            "survivors": None, "held": 0, "context": context or {}, "scores": [], "guidance": ""}


def async_on_digests(gst: dict, msgs: list[dict], knobs: dict) -> dict:
    """Digests that arrived (the Evaluator's): the Generator keeps the latest outer.population scores."""
    if not msgs:
        return gst
    scores = gst["scores"] + [[c, x] for d in msgs for c, x in zip(d["candidates"], d["scores"])]
    return {**gst, "scores": scores[-knobs["outer"]["population"]:]}


def async_on_guidance(gst: dict, msgs: list[dict]) -> dict:
    """Guidance that arrived (the Coordinator's): the Generator keeps the newest ("" when that call fell back)."""
    return {**gst, "guidance": msgs[-1]["text"]} if msgs else gst


def async_on_survivors(gst: dict, msgs: list[dict]) -> dict:
    """Survivors messages that arrived (numbered by the Coordinator's cycle): the Generator keeps the newest."""
    newest = max(msgs, key=lambda m: m["n"], default=None)
    if newest is None or newest["n"] <= gst["held"]:
        return gst
    return {**gst, "survivors": newest["survivors"], "held": newest["n"]}


def async_waits(gst: dict, knobs: dict) -> bool:
    """Whether the next profile needs a survivor and none has arrived: the Generator's only wait, for its first
    survivors message, when profile outer.population + 1 is due."""
    return gst["survivors"] is None and knobs["outer"]["population"] <= gst["made"] < async_budget(knobs)


def async_profiles(gst: dict, knobs: dict) -> tuple[dict, list[dict]]:
    """The next TAKE profiles from the evolutionary loop, as evaluate calls (evaluation_calls' keys, call id c<k>,
    `source`, and `filled`: None until the Generator's model call fills any, async_reconcile): fewer at the end
    of the budget, or when the next needs a survivor and none is held (then
    async_waits). Candidate k < P is the k-th of the parent's initialize_population, drawn once; each later one
    the parent's mutate_params of a survivor chosen at random from the newest survivors held, drawn from stream 0
    as the parent's repopulate chooses and mutates."""
    evo, o = parent(), knobs["outer"]
    P = o["population"]
    initial, rng, out = gst["initial"], rng_load(gst["rng"]), []
    for k in range(gst["made"], min(gst["made"] + TAKE, async_budget(knobs))):
        if k < P:
            if initial is None:
                initial = [_plain(p) for p in evo.initialize_population(P, MODEL, KIND, gst["seed"],
                                                                         delta_bounds=evo.DELTA_CLIP)]
            params, source = initial[k], "start"
        elif not gst["survivors"]:
            break
        else:
            chosen = gst["survivors"][int(rng.integers(0, len(gst["survivors"])))]
            params, source = _plain(evo.mutate_params(chosen, MODEL, KIND, o["noise"], rng,
                                                      delta_bounds=evo.DELTA_CLIP)), "mutation"
        out.append({"id": f"c{k}", "params": params, "candidate_idx": k, "generation": k // P, "source": source,
                    "filled": None})
    made = gst["made"] + len(out)
    return {**gst, "made": made, "initial": initial if made < P else None, "rng": rng_dump(rng)}, out


def async_score(window: list[dict], result: dict, m: float | None, knobs: dict) -> tuple[list[dict], float, float]:
    """One result as the Evaluator takes it (native values): q, the parent's keep-probability of it in the set of
    the window (the campaign's last outer.population scored results, in arrival order) and itself, last
    (keep_probabilities); its score, q + monitor.weight x m clipped to 0..1 (m 0 when the Monitor has none); and
    the window with it, its oldest left out."""
    q = keep_probabilities(window + [result], knobs["outer"]["wall_weight"])[-1]
    return _window(window, result, knobs["outer"]["population"]), q, numbers_score(q, m, knobs["monitor"]["weight"])


def numbers_score(q: float, m: float | None, weight: float) -> float:
    """The score from numbers alone: q + monitor.weight x m, clipped to 0..1 (m 0 when the Monitor has none)."""
    return min(1.0, max(0.0, q + weight * (m or 0.0)))


def async_evaluate(window: list[dict], reports: list[dict], knobs: dict) -> tuple[list[dict], list[dict], dict]:
    """The Evaluator's cycle on the Monitors' reports that arrived (each: `candidate`, `result`, `t0`, `source`,
    the parameters the Generator's reply filled (`filled`, None when not asked) and the Monitor's `m`), in arrival
    order, each scored against the window as it stands when it comes (async_score): the score from numbers,
    which the Evaluator's model reply then replaces (async_model_scores). Returns the new window, the scored
    results for the Coordinator and a digest for the Generator."""
    scored = []
    for r in reports:
        window, q, score = async_score(window, native(r["result"]), r["m"], knobs)
        scored.append({"candidate": r["candidate"], "result": r["result"], "q": q, "m": r["m"], "score": score,
                       "t0": r["t0"], "source": r["source"], "filled": r.get("filled")})
    return window, scored, _digest(scored)


def _digest(scored: list[dict]) -> dict:
    return {"candidates": [s["candidate"] for s in scored], "scores": [round(s["score"], 4) for s in scored]}


def async_model_scores(scored: list[dict], scores: dict) -> tuple[list[dict], dict]:
    """The Evaluator's model reply (parse_scores: a keep-probability per candidate) as the scores, and their
    digest."""
    scored = [{**s, "score": scores[s["candidate"]]} for s in scored]
    return scored, _digest(scored)


def async_coordinator_state(agent_id: str, seed: int, knobs: dict, context: dict, t_start: float | None = None) -> dict:
    """The Coordinator's memory: coordinator_state's, with stream 1 of outer.rng per_agent (survival's draws),
    the results arrived and kept, its window (as the Evaluator's), the survivors (each kept result's candidate
    and profile, the oldest first), its cycles, and when each result's candidate was put on the queue
    (`t_submit`, as `sim_s` and `wait_s`: this mode has no generation to time)."""
    return {**coordinator_state(agent_id, seed, knobs, context, t_start), "rng": _stream(seed, 1), "arrived": 0,
            "kept": 0, "scored_after_warmup": 0, "kept_after_warmup": 0, "window": [], "survivors": [], "cycles": 0,
            "t_submit": []}


def async_on_scored(cst: dict, scored: list[dict], knobs: dict, run_dir) -> tuple[dict, list[dict]]:
    """The Coordinator's cycle on the scored results that arrived, each in arrival order, n its arrival number
    (from 0): kept when its draw from stream 1 (one for every result) is below its score, and always while
    n < campaign.warmup; the survivors, at most outer.population, the oldest leaving; best-so-far by the
    parent's update_best_result, given the window and the result as the generation and their top-ranked as
    the survivor; one parent log_generation call with n as the generation (outer.logs); the result's counts.
    Returns the new memory and one row per result: n, candidate, q, m, score, draw, kept."""
    evo, o, warmup = parent(), knobs["outer"], knobs["campaign"]["warmup"]
    rng, rows = rng_load(cst["rng"]), []
    c = {k: cst[k] for k in ("window", "survivors", "best", "kept", "scored_after_warmup", "kept_after_warmup",
                             "fixture", "worker_peak_rss_mb")}
    counts, touches, devices, checks, refill = (dict(cst[k]) for k in ("counts", "touches", "devices", "touch_checks",
                                                                        "refill"))
    t_submit, sim_s, wait_s = list(cst["t_submit"]), list(cst["sim_s"]), list(cst["wait_s"])
    for s in scored:
        n, e = cst["arrived"] + len(rows), native(s["result"])
        draw = float(rng.random())
        kept = n < warmup or draw < s["score"]
        generation = [*c["window"], e]
        c["best"] = evo.update_best_result(c["best"], generation, ranked(generation, o["wall_weight"])[:1], n)
        if run_dir is not None:
            evo.log_generation(run_dir, n, [e], [e] if kept else [])
        if kept:
            c["survivors"] = (c["survivors"] + [{"candidate": s["candidate"], "params": e["params"]}])[-o["population"]:]
        c["window"] = _window(c["window"], e, o["population"])
        c["kept"] += kept
        if n >= warmup:
            c["scored_after_warmup"] += 1
            c["kept_after_warmup"] += kept
        rows.append({"arrival": n, "candidate": s["candidate"], "q": s["q"], "m": s["m"], "score": s["score"],
                     "draw": draw, "kept": kept})
        counts[s["source"]] += 1
        if s.get("filled") is not None:   # it went through the Generator's model call: asked, and filled by it
            refill["asked"] += 1
            refill["filled"] += s["filled"] > 0
        for kind, k in (e.get("touches") or {}).items():
            touches[kind] = touches.get(kind, 0) + k
        devices[str(e.get("device"))] = devices.get(str(e.get("device")), 0) + 1
        if not e.get("fixture"):
            checks[str(e.get("touch_check"))] = checks.get(str(e.get("touch_check")), 0) + 1
        c["fixture"] += bool(e.get("fixture"))
        if e.get("worker_peak_rss_mb") is not None:
            c["worker_peak_rss_mb"] = max(e["worker_peak_rss_mb"], c["worker_peak_rss_mb"] or 0)
        t_submit.append(round(s["t0"], 6))   # when the candidate was put on the queue
        sim_s.append(round(float(e.get("elapsed_seconds") or 0.0), 6))
        wait_s.append(round(max(0.0, float(e.get("t_start") or s["t0"]) - s["t0"]), 6))   # with the queue's wait
    return {**cst, **c, "rng": rng_dump(rng), "arrived": cst["arrived"] + len(rows), "cycles": cst["cycles"] + 1,
            "counts": counts, "touches": touches, "devices": devices, "touch_checks": checks, "t_submit": t_submit,
            "sim_s": sim_s, "wait_s": wait_s, "refill": refill}, rows


def async_survivors(cst: dict) -> dict:
    """The survivors message after a Coordinator cycle: its number (the cycle) and the survivors' profiles."""
    return {"n": cst["cycles"], "survivors": [s["params"] for s in cst["survivors"]]}


def async_memory(cst: dict) -> dict:
    """What the Coordinator writes to the agent memory after each cycle."""
    return {"arrived": cst["arrived"], "kept": cst["kept"], "best": cst["best"], "rng": cst["rng"],
            "survivors": [s["candidate"] for s in cst["survivors"]]}


def async_finish(cst: dict, run_dir=None, n_llm_calls: int = 0) -> dict:
    """The campaign's result once its budget is spent and its last result is in (best_result.json first), as
    finish() gives it, with no summary call (none in this mode) and no generations, its agents' model calls
    answered (`n_llm_calls`, the layer's count), the evaluations' times (`t_submit` beside `sim_s` and `wait_s`),
    the Coordinator's cycles, and its results scored and kept, in all and after warm-up (`survival`)."""
    if run_dir is not None:
        (run_dir / "best_result.json").write_text(json.dumps(cst["best"], indent=2, default=str))
    return {**finish(cst, ""), "n_llm_calls": n_llm_calls, "n_sims": cst["arrived"],
            "t_submit": cst["t_submit"], "n_cycles": cst["cycles"],
            "survival": {"results_scored": cst["arrived"], "results_kept": cst["kept"],
                         "results_scored_after_warmup": cst["scored_after_warmup"],
                         "results_kept_after_warmup": cst["kept_after_warmup"]}}


# --- the asynchronous mode's model calls (and campaign.notes' in the generations mode) --------------------------
#
# Each role's prompt ends with a line that says what reply it wants; the stub LLM and tests/fake_vllm.py answer
# by that line (llm.numbers_only). A reply that cannot be used is None from its parse function; the layer then
# tries again, and after its last try uses the role's fallback, which is the path with no model call above.

NOTE_ASK = "Reply with a note of one or two sentences on what this candidate's inner loop shows."
SCORES_ASK = ("Reply with one JSON object that maps each candidate number above to its keep-probability, a number "
              "from 0 to 1.")
GUIDANCE_ASK = "Reply with guidance for the Generator in two or three sentences: what to explore next."
NOTE_TOKENS, GUIDANCE_TOKENS = 128, 256   # visible reply allowances (as reply_tokens and SUMMARY_TOKENS)
# Bounds that keep every prompt of the asynchronous mode within the context by construction (a model's note or
# guidance is free text up to the reply cap): a Monitor's note or the Coordinator's guidance shown inside a later
# prompt is cut to NOTE_CHARS characters (the record keeps the whole text, with record.extras), and one Evaluator
# call shows at most SHOWN_RESULTS results (the rest wait for its next cycle, in arrival order: the layer's).
NOTE_CHARS, SHOWN_RESULTS = 1200, 8


def _cut(text: str) -> str:
    """A note or guidance as a later prompt shows it: its first NOTE_CHARS characters."""
    return (text or "")[:NOTE_CHARS]


def scores_tokens(n: int) -> int:
    """Visible reply allowance of the Evaluator's call for n results."""
    return 32 + 16 * n


def async_calls(knobs: dict) -> dict:
    """The most model calls each role asks in one campaign of the asynchronous mode (a call tried again counts
    once): the Generator one per take (async_profiles; one more when a take stops at candidate P for want of a
    survivor), the Monitors one per candidate, the Evaluator and the Coordinator one per cycle, at most one per
    result. None with llm.mode off."""
    if knobs["llm"]["mode"] == "off":
        return dict.fromkeys(("coordinator", "generator", "evaluator", "monitor"), 0)
    P, B = knobs["outer"]["population"], async_budget(knobs)
    return {"coordinator": B, "generator": -(-P // TAKE) + -(-(B - P) // TAKE), "evaluator": B, "monitor": B}


def candidates_schema(n: int) -> dict:
    """The Generator's structured reply, sent as the request's response_format: a JSON list of n objects, each with
    numeric values for any of the seven parameters."""
    item = {"type": "object", "properties": {k: {"type": "number"} for k in ranges()}, "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": "candidates", "schema": {
        "type": "array", "items": item, "minItems": n, "maxItems": n}}}


def _numbered(ks: list[int]) -> str:
    return ks and ("Candidate " if len(ks) == 1 else "Candidates ") + (
        str(ks[0]) if len(ks) == 1 else ", ".join(map(str, ks[:-1])) + f" and {ks[-1]}")


def async_generator_messages(gst: dict, calls: list[dict], knobs: dict) -> list[dict]:
    """The Generator's prompt: the take's profiles from the evolutionary loop, the newest survivors, the
    Evaluator's latest scores and the Coordinator's guidance (cut to NOTE_CHARS). Its last line names the
    candidates, by their numbers (from 0), of the campaign's budget: the stub seeds its reply by the campaign and
    that line."""
    system = ("You are the experiment generator of one IGNOS fusion design campaign. The campaign's evolutionary "
              "loop proposes H-mode plasma profile parameters for the three-region profile model; you reconcile "
              "each proposal with what the campaign has learned. The goal is the highest te_core_metric among "
              "feasible candidates (wall_ok and pedestal_ok both true). Parameter ranges: "
              + json.dumps(ranges()))
    if "abstract" in gst["context"]:
        system += " Background: " + gst["context"]["abstract"]["text"]
    survivors = gst["survivors"] or []
    parts = [f"The campaign's newest survivors ({len(survivors)}):\n" + "\n".join(json.dumps(_seven(p)) for p in survivors)
             if survivors else "No survivors yet."]
    if gst["scores"]:
        parts.append("The Evaluator's latest keep-probabilities (candidate: probability): "
                     + ", ".join(f"{c}: {x}" for c, x in gst["scores"]))
    if gst["guidance"]:
        parts.append("The Coordinator's guidance: " + _cut(gst["guidance"]))
    parts.append("The evolutionary loop proposes:\n" + "\n".join(
        f"candidate {c['candidate_idx']} ({c['source']}): {json.dumps(_seven(c['params']))}" for c in calls)
                 + "\nGive each candidate the values you would change; a key you leave out keeps the evolutionary "
                 "loop's value.")
    ask = (f"{_numbered([c['candidate_idx'] for c in calls])} of {async_budget(knobs)}: reply with a JSON list of "
           f"{len(calls)} objects, one per candidate in that order, each with numeric values for any of these keys: "
           f"{', '.join(ranges())}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts) + "\n\n" + ask}]


def parse_reconcile(reply: str, n: int) -> list[dict] | None:
    """The Generator's reply: the last JSON list in it of n objects, each object's values for the seven parameters
    clipped to range (_clip: a key left out, or not a number, is not filled); else None (a reply that cannot be
    used)."""
    reply, decoder = reply or "", json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\[", reply)]):
        try:
            obj = decoder.raw_decode(reply, i)[0]
        except ValueError:
            continue
        if isinstance(obj, list) and len(obj) == n and all(isinstance(o, dict) for o in obj):
            return [_clip(o) for o in obj]
    return None


def async_reconcile(calls: list[dict], proposals: list[dict]) -> tuple[list[dict], list[int]]:
    """Each candidate with the reply's values in place of the evolutionary loop's (then the parent's clip_delta,
    as the refill applies a proposal), its source `model` when the reply filled any; and how many each filled."""
    evo, out = parent(), []
    for c, prop in zip(calls, proposals):
        if prop:
            c = {**c, "params": _plain(evo.clip_delta({**c["params"], **prop}, evo.DELTA_CLIP)), "source": "model"}
        out.append({**c, "filled": len(prop)})
    return out, [len(p) for p in proposals]


def _trace(numbers: dict | None) -> str:
    if numbers is None:
        return "no trace"
    return (f"m {numbers['m']} (+1: like the traces of high-ranked candidates, -1: like low-ranked ones, 0: cannot "
            f"tell); left the typical band at step {numbers['left_at']}, back in it at step {numbers['back_at']}; "
            f"the good and the bad sides split at step {numbers['split_at']}")


def note_messages(result: dict, numbers: dict | None) -> list[dict]:
    """A Monitor's prompt: its candidate's profile, its inner loop's numbers and its trace against the campaign's
    rolling statistics (the Monitor's numbers, monitor.py); the last line asks for the note."""
    system = ("You are a Monitor of one IGNOS fusion design campaign. You watched one candidate's inner loop (the "
              "turned-down IGNOS evaluation) and compared its trace, step by step, with the campaign's rolling "
              "statistics of the traces of high-ranked (good) and low-ranked (bad) candidates.")
    user = (f"Candidate {result['candidate_idx']}: {json.dumps(_seven(result['params']))}\n"
            f"Its inner loop: {_shown(result)}\nIts trace: {_trace(numbers)}\n\n{NOTE_ASK}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_text(reply: str) -> str:
    """A note or guidance: the reply on one line; any reply can be used ("" is none)."""
    return " ".join((reply or "").split())


def scores_messages(items: list[dict]) -> list[dict]:
    """The Evaluator's prompt: each result to score (items: candidate, result, q, numbers, note, the note cut to
    NOTE_CHARS), q and m at full precision (so that the stub's numbers-only answer, llm.numbers_only, is exact); the
    last line asks for each one's keep-probability. Neither the score from numbers nor its formula is shown: the
    reply is the model's own."""
    system = ("You are the evaluator of one IGNOS fusion design campaign. For each result below you give the "
              "probability, from 0 to 1, that the campaign keeps it as a survivor. q is the probability the "
              "campaign's evolutionary loop gives it by its rank among the campaign's recent results (the best 1, "
              "the worst 0). The Monitor's m says whether its inner loop's trace looked like those of high-ranked "
              "(+1) or low-ranked (-1) candidates. Weigh q, the Monitor's numbers and note, and the result's own "
              "numbers.")
    rows = [f"candidate {it['candidate']}: q {it['q']!r}; {_trace(it['numbers'])}; {_shown(it['result'])}; "
            f"profile {json.dumps(_seven(it['result']['params']))}; note: {_cut(it['note']) or 'none'}" for it in items]
    return [{"role": "system", "content": system},
            {"role": "user", "content": "Results to score:\n" + "\n".join(rows) + "\n\n" + SCORES_ASK}]


def parse_scores(reply: str, candidates: list[int]) -> dict | None:
    """The Evaluator's reply: the last JSON object in it that gives each candidate (its number as a key) a number,
    each clipped to 0..1; else None (a reply that cannot be used)."""
    reply, decoder = reply or "", json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", reply)]):
        try:
            obj = decoder.raw_decode(reply, i)[0]
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        out = {}
        for k in candidates:
            v = obj.get(str(k))
            if not _number(v):
                break
            out[k] = min(1.0, max(0.0, float(v)))
        else:
            return out
    return None


def guidance_messages(cst: dict, rows: list[dict], knobs: dict) -> list[dict]:
    """The Coordinator's prompt after a cycle: the results it has just judged (kept or dropped, with their numbers
    from its window), the campaign so far (results in and kept, the survivors held, best-so-far) and its
    constraints lookup; the last line asks for guidance to the Generator."""
    system = ("You are the coordinator of one IGNOS fusion design campaign. Survival is decided by the campaign's "
              "rule; you guide the campaign's Generator on which H-mode profile parameters to explore next. The goal "
              "is the highest te_core_metric among feasible candidates (wall_ok and pedestal_ok both true).")
    if "constraints" in cst["context"]:
        system += " Constraints and parameter ranges: " + json.dumps(cst["context"]["constraints"], sort_keys=True)
    seen = {e["candidate_idx"]: e for e in cst["window"]}
    judged = [f"candidate {r['candidate']}: {'kept' if r['kept'] else 'dropped'} (score {round(r['score'], 4)}); "
              + (f"{_shown(seen[r['candidate']])}; profile {json.dumps(_seven(seen[r['candidate']]['params']))}"
                 if r["candidate"] in seen else "") for r in rows]
    b = cst["best"]
    user = ("Results just judged:\n" + "\n".join(judged) + f"\n\n{cst['arrived']} of {async_budget(knobs)} results "
            f"in, {cst['kept']} kept; {len(cst['survivors'])} survivors held. Best so far: te_core_metric="
            f"{_f(b['te_core_metric'])}, te_wall_eV={_f(b.get('te_wall_eV'))}, te_ped_eV={_f(b.get('te_ped_eV'))}, "
            f"feasible={b['is_good']}.\n\n{GUIDANCE_ASK}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
