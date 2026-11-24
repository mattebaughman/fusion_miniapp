"""A campaign of the asynchronous mode replayed through decide.py's pure functions (not a test module).

Given the orders a run records with record.extras (each campaign's line, `asynchronous`): the candidates each
Evaluator cycle and each Coordinator cycle took, in arrival order, and the survivors message the Generator held
at each profile, with the Monitor's m of each candidate, `replay` drives the Generator, the Evaluator and the
Coordinator as the AUTONMI layer drives them, each as soon as what it needs exists, and evaluates each candidate
with the fixture (whose values are a function of the profile alone). One order, one seed and the Monitors' m
give one campaign.

Run as a script, `python tests/async_replay.py <layer_autonmi's arguments>` is the AUTONMI layer with each
campaign's arrivals reordered (`reordered`), so that a replay meets results that arrive out of candidate order
and a Coordinator that takes several Evaluator cycles at once.
"""
import asyncio
import sys

from fusion_miniapp import decide, sim

DELAY_S = 0.05   # reordered: the event loop's wait before every fourth candidate's report, and each Coordinator cycle


def fixture(knobs: dict, seed: int):
    """The fixture's evaluate for campaign a<seed>, as a Monitor submits it (its cost-scaled sim knobs)."""
    s = decide.sim_knobs(knobs, seed)
    return lambda call: sim.evaluate_fixture(knobs=s, agent_id=f"a{seed}", seed=seed, params=call["params"],
                                             candidate_idx=call["candidate_idx"], generation=call["generation"])


def replay(seed: int, knobs: dict, evaluator: list[list[int]], coordinator: list[list[int]], held: list[int],
           m: dict, evaluate) -> tuple[dict, list[dict], dict]:
    """The Coordinator's memory at the end and its rows (decide.async_on_scored's, in arrival order), and the
    candidates made (their evaluate calls, by number). Raises when the orders cannot have happened."""
    B = decide.async_budget(knobs)
    gst = decide.async_generator_state(f"a{seed}", seed, knobs)
    cst = decide.async_coordinator_state(f"a{seed}", seed, knobs, {}, 0.0)
    window, calls, scored, msgs, rows, e, c = [], {}, {}, {}, [], 0, 0
    while c < len(coordinator):
        moved = False
        if gst["made"] < B and (held[gst["made"]] == 0 or held[gst["made"]] in msgs):   # the Generator's next take
            n = held[gst["made"]]
            gst = decide.async_on_survivors(gst, [msgs[n]] if n else [])
            gst, made = decide.async_profiles(gst, knobs)
            calls.update((x["candidate_idx"], x) for x in made)
            moved = True
        if e < len(evaluator) and all(k in calls for k in evaluator[e]):   # the Evaluator's next cycle
            reports = [{"candidate": k, "result": evaluate(calls[k]), "t0": 0.0, "source": calls[k]["source"],
                        "m": m[k]} for k in evaluator[e]]
            window, out, _ = decide.async_evaluate(window, reports, knobs)
            scored.update((x["candidate"], x) for x in out)
            e, moved = e + 1, True
        if all(k in scored for k in coordinator[c]):   # the Coordinator's next cycle, and its survivors message
            cst, r = decide.async_on_scored(cst, [scored[k] for k in coordinator[c]], knobs, None)
            rows += r
            msgs[cst["cycles"]] = decide.async_survivors(cst)
            c, moved = c + 1, True
        if not moved:
            raise RuntimeError(f"the orders cannot be replayed: stuck at Evaluator cycle {e}, Coordinator cycle {c}, "
                               f"profile {gst['made']}")
    return cst, rows, calls


def reordered(argv: list[str]) -> int:
    """layer_autonmi.main(argv) with two waits on the event loop (asyncio.sleep, so no thread is held): a Monitor
    whose candidate's number is a multiple of 4 waits DELAY_S before it reports, so the other slot's later
    candidate can reach the Evaluator first; and each Coordinator cycle waits DELAY_S before it takes, so several
    Evaluator cycles can reach it at once. The agents' decisions are unchanged."""
    from fusion_miniapp import layer_autonmi

    script = layer_autonmi.Script
    monitor, coordinator = script._monitor, script._coordinator_async

    async def slow_monitor(self, s, event, ask):
        if s["phase"] == "running" and s["call"]["candidate_idx"] % 4 == 0:
            await asyncio.sleep(DELAY_S)
        return await monitor(self, s, event, ask)

    async def slow_coordinator(self, s, event, ask):
        if s["phase"] == "cycle":
            await asyncio.sleep(DELAY_S)
        return await coordinator(self, s, event, ask)

    script._monitor, script._coordinator_async = slow_monitor, slow_coordinator
    return layer_autonmi.main(argv)


if __name__ == "__main__":
    sys.exit(reordered(sys.argv[1:]))
