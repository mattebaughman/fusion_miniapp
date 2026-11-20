"""
Academy agent wrappers around the evolutionary outer loop.

The outer loop in evolutionary_loop.py:669-735 is already three roles in a
trench coat:

    population = initialize_population(...)              # generator
    for gen in range(n_generations):
        evaluated  = evaluate_population_parallel(...)    # evaluator
        survivors  = select_survivors(evaluated, rng)     # coordinator
        log_generation(...)                               # coordinator
        population = repopulate(survivors, ..., rng, gen) # generator

This module makes those roles explicit as Academy agents without moving any
logic: every action delegates straight to the function that already does the
work. The point is the seam, not new behaviour -- once the generator sits behind
a handle, a directed proposer (skopt, the ProposalAgent pattern in
stubbed_ui/ignos/agents/proposal.py) can replace random repopulation without the
coordinator or evaluator noticing.

RNG NOTE: the monolithic loop drives selection *and* repopulation from one
np.random.Generator (evolutionary_loop.py:599). Agents cannot share a live
Generator, so each owns a stream derived from the run seed via SeedSequence.
That is a deliberate divergence: results will not reproduce bit-for-bit against
runs made by evolutionary_loop.py with the same --seed.

Usage:
    from evo_agents import build_runtime
    async with build_runtime(cfg) as coordinator:
        result = await coordinator.run_generations(n_generations=20)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from academy.agent import Agent, action
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.manager import Manager

import evolutionary_loop as evo


# =============================================================================
# Generator: owns population construction (LHS init + mutation-based refill)
# =============================================================================
class GeneratorAgent(Agent):
    """Produces candidate populations. Wraps initialize_population/repopulate."""

    def __init__(
        self,
        model: str = "three_region",
        kind: str = "H",
        noise_amplitude: float = 0.1,
        seed: int = 42,
        delta_bounds: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.kind = kind
        self.noise_amplitude = noise_amplitude
        self.seed = seed
        self.delta_bounds = delta_bounds
        # Stream 0 of the run seed. See RNG NOTE in the module docstring.
        self._rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])

    @action
    async def initialize(self, population_size: int) -> List[Dict]:
        """First generation: Latin-hypercube sample of the parameter space."""
        return evo.initialize_population(
            population_size, self.model, self.kind, self.seed,
            delta_bounds=self.delta_bounds,
        )

    @action
    async def repopulate(
        self,
        survivors: List[dict],
        population_size: int,
        generation: int,
    ) -> List[Dict]:
        """Refill to population_size from survivors, or LHS-restart if none."""
        return evo.repopulate(
            survivors,
            population_size,
            self.model,
            self.kind,
            self.noise_amplitude,
            self._rng,
            generation,
            delta_bounds=self.delta_bounds,
        )


# =============================================================================
# Evaluator: owns the physics pipeline dispatch
# =============================================================================
class EvaluatorAgent(Agent):
    """Evaluates a population. Wraps evaluate_population_parallel.

    evaluate_population_parallel blocks -- with dispatch="srun" it shells out via
    subprocess.Popen (evolutionary_loop.py:335), with dispatch="pool" it builds a
    spawn-context ProcessPoolExecutor (:402). Either way it must not run on the
    event loop, so it goes through Agent.agent_run_sync, which hands it to a
    separate thread.

    Prefer dispatch="srun" here: a spawn-context process pool started from a
    non-main thread is the one part of this wiring that is not obviously safe.
    """

    def __init__(
        self,
        shared_kwargs: Dict[str, Any],
        n_gpus: int = 4,
        dispatch: str = "srun",
        gpus_per_node: int = 4,
        gpu_offset: int = 0,
    ) -> None:
        super().__init__()
        self.shared_kwargs = shared_kwargs
        self.n_gpus = n_gpus
        self.dispatch = dispatch
        self.gpus_per_node = gpus_per_node
        self.gpu_offset = gpu_offset

    @action
    async def evaluate(self, population: List[Dict], generation: int) -> List[dict]:
        """Run the full pipeline on every candidate; return per-candidate dicts."""
        kwargs = dict(self.shared_kwargs)
        kwargs["generation"] = generation
        return await self.agent_run_sync(
            evo.evaluate_population_parallel,
            population,
            self.n_gpus,
            kwargs,
            dispatch=self.dispatch,
            gpus_per_node=self.gpus_per_node,
            gpu_offset=self.gpu_offset,
        )


# =============================================================================
# Coordinator: owns the generation loop, selection, and logging
# =============================================================================
class CoordinatorAgent(Agent):
    """Drives generations across the generator and evaluator handles."""

    def __init__(
        self,
        generator: Handle,
        evaluator: Handle,
        run_dir: Path,
        population_size: int,
        seed: int = 42,
        wall_weight: float = evo.DEFAULT_WALL_WEIGHT,
    ) -> None:
        super().__init__()
        self._generator = generator
        self._evaluator = evaluator
        self.run_dir = Path(run_dir)
        self.population_size = population_size
        self.wall_weight = wall_weight
        # Stream 1 of the run seed. See RNG NOTE in the module docstring.
        self._rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        self.best_ever = evo.empty_best_result()
        self.history: List[dict] = []

    @action
    async def run_generations(self, n_generations: int) -> Dict[str, Any]:
        """The outer loop, one hop per role."""
        population = await self._generator.initialize(self.population_size)

        for gen in range(n_generations):
            t0 = time.perf_counter()

            evaluated = await self._evaluator.evaluate(population, gen)
            survivors = evo.select_survivors(evaluated, self._rng, self.wall_weight)
            evo.log_generation(self.run_dir, gen, evaluated, survivors)

            self.best_ever = evo.update_best_result(
                self.best_ever, evaluated, survivors, gen
            )

            n_good = sum(1 for e in evaluated if e["is_good"])
            elapsed = time.perf_counter() - t0
            self.history.append({
                "generation": gen,
                "n_evaluated": len(evaluated),
                "n_good": n_good,
                "n_survivors": len(survivors),
                "elapsed_s": elapsed,
            })
            print(f"  gen {gen + 1}/{n_generations}: {len(evaluated)} evaluated, "
                  f"{n_good} good, {len(survivors)} survivors ({elapsed:.1f}s)")

            if gen < n_generations - 1:
                population = await self._generator.repopulate(
                    survivors, self.population_size, gen
                )

        return {"best_ever": self.best_ever, "history": self.history}

    @action
    async def smoke_check(self) -> str:
        """Liveness probe -- used by the smoke test, touches no physics.

        Not named ping(): Handle defines its own ping() (a latency probe), and
        an action of that name is shadowed by it when called through a handle.
        Academy warns about the clash at decoration time.
        """
        return f"coordinator alive, population_size={self.population_size}"


# =============================================================================
# Run context
# =============================================================================
def build_run_context(
    population_size: int,
    n_generations: int,
    run_tag: Optional[str] = None,
    model: str = "three_region",
    kind: str = "H",
    n_core_points: int = 5,
    seed: int = 42,
    noise_amplitude: float = 0.1,
    n_gpus: int = 4,
    constraints_path: Optional[str] = None,
    helios_ckpt_path: Optional[str] = None,
    matey_model_dir: Optional[str] = None,
    step_ids: Optional[List[int]] = None,
    save_viz: bool = False,
    verbose: int = 1,
    flash: bool = False,
    elmimpute: bool = True,
    n_ar_steps: Optional[int] = None,
    wall_weight: float = evo.DEFAULT_WALL_WEIGHT,
    inner_loop: str = evo.DEFAULT_INNER_LOOP,
    conditioned_ckpt: Optional[str] = None,
    conditioned_yaml: Optional[str] = None,
    conditioned_config_name: str = evo.DEFAULT_CONDITIONED_CONFIG_NAME,
    dispatch: str = "srun",
    gpus_per_node: int = 4,
    gpu_offset: int = 0,
) -> tuple:
    """Build (run_dir, shared_kwargs) the way run_evolutionary_loop does.

    This mirrors the setup block inside run_evolutionary_loop rather than calling
    into it, because that setup is welded to the middle of the function and the
    working sweep path should not be refactored just to host an agent driver.
    The duplication is deliberate and lives here, not there -- if the worker
    contract changes, this is the function to update.

    The two pieces most likely to drift (the default step list and the step-0
    profile shaping) are shared with the sweep driver via evo.DEFAULT_STEP_IDS and
    evo.resolve_profile_shaping, so they cannot fall out of sync silently.

    Returns:
        (evo_run_dir: Path, shared_kwargs: dict)
    """
    import yaml

    if not 0.0 <= wall_weight <= 1.0:
        raise ValueError(f"wall_weight must be in [0, 1], got {wall_weight}")
    if constraints_path is None:
        constraints_path = str(evo.DEMO_ROOT / "config" / "config.yaml")
    if helios_ckpt_path is None:
        helios_ckpt_path = evo.DEFAULT_HELIOS_CKPT
    if matey_model_dir is None:
        matey_model_dir = evo.DEFAULT_MATEY_MODEL_DIR

    settings = evo.resolve_inner_loop_settings(
        inner_loop=inner_loop,
        step_ids=step_ids,
        n_ar_steps=n_ar_steps,
        conditioned_ckpt=conditioned_ckpt,
        conditioned_yaml=conditioned_yaml,
        conditioned_config_name=conditioned_config_name,
        model=model,
        kind=kind,
    )
    step_ids = settings["step_ids"]
    n_ar_steps = settings["n_ar_steps"]

    cp = Path(constraints_path)
    if not cp.is_absolute():
        cp = evo.DEMO_ROOT / cp
    constraints_path_abs = str(cp)

    with open(constraints_path_abs) as f:
        constraints = yaml.safe_load(f) or {}
    T_core_keV = constraints.get("target_core_temperature_keV")
    profile_shaping = (
        evo.resolve_profile_shaping(constraints)
        if settings["inner_loop"] == "legacy"
        else None
    )

    # Same DEMO_SCRATCH / timestamp / tag / jobid naming as the sweep, so agent
    # runs land beside the others instead of in $HOME.
    output_base = Path(os.environ.get("DEMO_SCRATCH", str(evo.DEMO_ROOT)))
    name = f"evo_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    if run_tag:
        name += f"_{run_tag}"
    jobid = os.environ.get("SLURM_JOB_ID")
    if jobid:
        name += f"_j{jobid}"
    evo_run_dir = output_base / "runs" / name
    evo_run_dir.mkdir(parents=True, exist_ok=True)

    with open(evo_run_dir / "config.json", "w") as f:
        json.dump({
            "population_size": population_size,
            "n_generations": n_generations,
            "noise_amplitude": noise_amplitude,
            "wall_weight": wall_weight,
            "n_gpus": n_gpus,
            "model": model,
            "kind": kind,
            "seed": seed,
            "inner_loop": settings["inner_loop"],
            "profile_mode": settings["profile_mode"],
            "step_ids": step_ids,
            "n_core_points": n_core_points,
            "T_core_keV": T_core_keV,
            "n_ar_steps": n_ar_steps,
            "dispatch": dispatch,
            "gpus_per_node": gpus_per_node,
            "gpu_offset": gpu_offset,
            "run_tag": run_tag,
            "constraints_path": constraints_path_abs,
            "matey_model_dir": str(Path(matey_model_dir).expanduser().resolve()),
            "helios_ckpt_path": (
                str(Path(helios_ckpt_path).expanduser().resolve())
                if settings["inner_loop"] == "legacy" else None
            ),
            "profile_shaping": profile_shaping,
            "validated_method": (
                evo.VALIDATED_METHOD if settings["inner_loop"] == "validated" else None
            ),
            "conditioned_ckpt": settings["conditioned_ckpt"],
            "conditioned_ckpt_sha256": (
                evo._sha256_file(settings["conditioned_ckpt"])
                if settings["conditioned_ckpt"] else None
            ),
            "conditioned_yaml": settings["conditioned_yaml"],
            "conditioned_yaml_sha256": (
                evo._sha256_file(settings["conditioned_yaml"])
                if settings["conditioned_yaml"] else None
            ),
            "conditioned_config_name": settings["conditioned_config_name"],
            "runtime_source_sha256": evo._runtime_source_sha256(),
            "git": evo._git_provenance(),
            "driver": "evo_agents",
        }, f, indent=2)

    shared_kwargs: Dict[str, Any] = {
        "evo_run_dir": str(evo_run_dir),
        "model": model,
        "kind": kind,
        "n_core_points": n_core_points,
        "T_core_keV": T_core_keV,
        "inner_loop": settings["inner_loop"],
        "constraints_path": constraints_path_abs,
        "helios_ckpt_path": helios_ckpt_path,
        "matey_model_dir": matey_model_dir,
        "use_solps_inference": True,
        "solps_data_dir": None,
        "use_xgc_inference": True,
        "xgc_data_dir": None,
        "step_ids": step_ids,
        "verbose": verbose,
        "flash": flash,
        "elmimpute": elmimpute,
        "save_viz": save_viz,
        "n_ar_steps": n_ar_steps,
        "profile_shaping": profile_shaping,
        "conditioned_ckpt": settings["conditioned_ckpt"],
        "conditioned_yaml": settings["conditioned_yaml"],
        "conditioned_config_name": settings["conditioned_config_name"],
    }
    return evo_run_dir, shared_kwargs


# =============================================================================
# Runtime bootstrap
# =============================================================================
@asynccontextmanager
async def build_runtime(
    shared_kwargs: Dict[str, Any],
    run_dir: Path,
    population_size: int = 4,
    model: str = "three_region",
    kind: str = "H",
    noise_amplitude: float = 0.1,
    seed: int = 42,
    n_gpus: int = 4,
    dispatch: str = "srun",
    gpus_per_node: int = 4,
    gpu_offset: int = 0,
    wall_weight: float = evo.DEFAULT_WALL_WEIGHT,
):
    """Launch the three agents on a local exchange and yield handles.

    Yields (coordinator, generator, evaluator) handles. The exchange is
    in-process: this buys the seam, not distribution -- the srun dispatch inside
    the evaluator is still what spans nodes.
    """
    factory = LocalExchangeFactory()
    async with await Manager.from_exchange_factory(
        factory=factory,
        executors=ThreadPoolExecutor(max_workers=4),
    ) as manager:
        generator = await manager.launch(
            GeneratorAgent(model=model, kind=kind,
                           noise_amplitude=noise_amplitude, seed=seed,
                           delta_bounds=(
                               evo.DELTA_CLIP
                               if shared_kwargs.get("inner_loop") == "validated"
                               else None
                           )),
            name="generator",
        )
        evaluator = await manager.launch(
            EvaluatorAgent(shared_kwargs=shared_kwargs, n_gpus=n_gpus,
                           dispatch=dispatch, gpus_per_node=gpus_per_node,
                           gpu_offset=gpu_offset),
            name="evaluator",
        )
        coordinator = await manager.launch(
            CoordinatorAgent(generator=generator, evaluator=evaluator,
                             run_dir=run_dir, population_size=population_size,
                             seed=seed, wall_weight=wall_weight),
            name="coordinator",
        )
        yield coordinator, generator, evaluator
