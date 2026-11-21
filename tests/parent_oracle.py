"""The parent's own outer loop as the reference for tests/test_parent_oracle.py (not a test module).

    python tests/parent_oracle.py OUT {fixture,real} SEED {shared,per_agent} KNOBS_JSON

Runs, in this fresh process, one campaign of the unmodified parent driver: run_evolutionary_loop
(shared: one random generator) or evo_agents' three Academy agents (per_agent: two streams). Only the
parent's population evaluator is replaced, here and nowhere else, by the mini-app's evaluate (or its
fixture) called in candidate order, as evaluate_population_parallel returns them. The validated mode
checks that a checkpoint and a YAML file exist before it hashes them; two small files stand in, which
evaluate never reads. Thread variables are set before numpy is imported, so floating-point results
match a mini-app worker started with sim.threads_per_worker: 1. Writes OUT/runs/<run>/{config.json,
evolution_log.csv, evolution_log.jsonl, best_result.json} and OUT/run_dir.txt.
"""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", FUSION_MINIAPP_THREADS="1")

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def main(out: str, mode: str, seed: str, rng: str, knobs_json: str) -> None:
    from fusion_miniapp import PARENT_DIR, decide, sim

    out, seed, knobs = Path(out), int(seed), json.loads(knobs_json)
    evo, o = decide.parent(), knobs["outer"]
    evaluate = sim.evaluate if mode == "real" else sim.evaluate_fixture
    sim_knobs = decide.sim_knobs(knobs, seed)

    def native(record: dict) -> dict:
        # evaluate returns JSON-safe values (NaN -> None); the parent's worker returns its native NaN floats.
        # Only elm_ok is None natively.
        return {k: float("nan") if v is None and k != "elm_ok" else v for k, v in record.items()}

    def evaluate_population(population, n_gpus, shared_kwargs, **dispatch):
        g = shared_kwargs["generation"]
        return [native(evaluate(knobs=sim_knobs, agent_id=f"a{seed}", seed=seed, params=p, candidate_idx=i,
                                generation=g))
                for i, p in enumerate(population)]

    evo.evaluate_population_parallel = evaluate_population
    sys.modules["evolutionary_loop"].evaluate_population_parallel = evaluate_population
    stand_in = out / "stand_in_asset"
    stand_in.write_text("stand-in: never read")
    os.environ["DEMO_SCRATCH"] = str(out)
    options = dict(population_size=o["population"], n_generations=o["generations"], seed=seed,
                   noise_amplitude=o["noise"], wall_weight=o["wall_weight"],
                   constraints_path=str(PARENT_DIR / "config" / "config.yaml"),
                   conditioned_ckpt=str(stand_in), conditioned_yaml=str(stand_in), inner_loop="validated",
                   run_tag=f"{mode}_{seed}")
    with open(out / "driver.log", "w") as log, contextlib.redirect_stdout(log):
        if rng == "shared":
            evo.run_evolutionary_loop(**options)
        else:
            import evo_agents

            async def campaign():
                run_dir, shared = evo_agents.build_run_context(**options)
                async with evo_agents.build_runtime(shared, run_dir, population_size=o["population"], seed=seed,
                                                    noise_amplitude=o["noise"],
                                                    wall_weight=o["wall_weight"]) as (coordinator, _, _):
                    result = await coordinator.run_generations(n_generations=o["generations"])
                # evo_agents returns the best but writes no best_result.json; write it as run_evolutionary_loop does
                (run_dir / "best_result.json").write_text(json.dumps(result["best_ever"], indent=2, default=str))
            asyncio.run(campaign())
    (run_dir,) = (out / "runs").iterdir()
    (out / "run_dir.txt").write_text(str(run_dir))


if __name__ == "__main__":
    main(*sys.argv[1:])
