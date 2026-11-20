# Interface contract (fixed; change only by agreement recorded here)

Package: `fusion_miniapp`, src layout under `src/fusion_miniapp/`. Python 3.12 preferred (3.11 to 3.13 accepted).

## Files and owners
- SIM SIDE: `parent/` (five files, byte-identical), `standins.py`, `sim.py`, `touches.py`, `knobs.py`, `tests/test_sim_*.py`, `knobs/*.yaml`.
- AGENT SIDE: `decide.py`, `llm.py`, `tools.py`, `layer_langgraph.py`, `layer_academy.py`, `record.py`, `run.py`, `tests/test_agent_*.py`.
- INTEGRATION: `launch/` (three sbatch files, `run.sh`, `probe.py`), `README.md`, `tests/test_end_to_end.py`, `results/`.

## The simulation task
```python
# fusion_miniapp/sim.py  -- top-level, importable by name, picklable arguments and result
def evaluate(candidate: dict, knobs: dict | None = None, agent_id: str = "", seed: int = 0) -> dict: ...
```
- `candidate`: any subset of the seven mutable profile parameters `a1_ne, gamma_ne, Delta, a2_ne, a1_Te, gamma_Te, a2_Te`; the rest are filled deterministically from `seed` with the parent's own sampler.
- `knobs`: the `sim` section of the knob file as a dict (None means the smoke preset).
- Returns a JSON-serialisable dict: the parent worker's scalar result record, plus `emulated: true`, `device`, `elapsed_seconds`, `touches` (a dict of touch-kind to count) and `touch_check` (`"pass"` or `"fail: <which>"`).
- It runs the parent's real `_evaluate_single_candidate` / `evaluate_final_innerloop` over stand-ins injected through the adapter's `context` (keys `core_edge_optimization`, `run_matey_inference`, `puff_optimizer`). The context is built once per worker process and set of context knobs, and cached. `evaluate` does not read `cold_start_fraction`; run.py makes cold starts real by retiring each worker process after round(1 / fraction) tasks.
- A fixture twin `evaluate_fixture(candidate, knobs=None, agent_id="", seed=0) -> dict` returns the same keys instantly with `emulated: true, fixture: true` and no torch import, for large agent-count tests.

## Knob file (YAML; `knobs.py` loads, validates, applies presets `smoke`, `hundredk`, `ignos_shape`)
```yaml
sim:      {nodes_per_plane, planes, rollout_steps, puffs, fixed_point_cap, model_width, checkpoint_mb, device, cold_start_fraction, keep_artifacts, work_root}
agents:   {count, per_process, sims_per_agent, cost_scale (number or list), thinking_budget, budget_scales_with_cost}
tools:    {constraints_lookup, abstract_lookup, latency_ms}
llm:      {mode: stub|openai, base_url_env, api_key_env, model}
workers:  {sim_workers}
```

## The decision logic (no agent-framework import in this file)
```python
# fusion_miniapp/decide.py
class Services(Protocol):
    async def llm(self, agent_id: str, messages: list[dict], thinking_budget: int) -> str: ...
    async def simulate(self, agent_id: str, candidate: dict, seed: int) -> dict: ...   # awaits one evaluate
    async def lookup(self, agent_id: str, what: str) -> dict: ...                       # tool call
    async def remember(self, agent_id: str, state: dict) -> None: ...                   # memory / checkpoint
    async def report(self, agent_id: str, result: dict) -> None: ...                    # to the campaign manager
async def run_campaign(agent_id: str, seed: int, knobs: dict, svc: Services, on_state=None) -> dict: ...
```
- `run_campaign` is the whole workflow of the plan's section 1: lookup, then `sims_per_agent` rounds of (LLM proposal, simulate, feasibility and best-so-far update, remember), then an LLM summary, then report. It returns `{agent_id, best, history, summary, n_llm_calls, n_sims, n_tool_calls}`.
- `on_state(agent_id, state)` is called with `"working"`, `"waiting"` or `"model_request"` at each transition; each layer maps it to its own counters.
- With the stub LLM and a fixed seed, the result must be identical whichever agent layer runs it.

## The two layers and the flag
- `layer_langgraph.py`: `async def run_agents(n_agents, knobs, svc_factory, out) -> list[dict]` using a LangGraph graph hosted by FlowGentic's agent host, with `evaluate` declared as a suspending task tool on a RHAPSODY session. On a laptop: RHAPSODY's concurrent backend on spawned worker processes for `--sim real` (threads for the fixture), in-memory store, stub model.
- `layer_academy.py`: the same signature, using Academy agents hosted in-process (local exchange, no executor): a `CampaignAgent` running `run_campaign` and one `EvaluatorAgent` per process in front of the simulation, which submits to the same RHAPSODY session and worker processes.
- Both accept extra keywords (`first`, `emit`; the LangGraph layer also `store`, `replicas`) used by `run.py` and the machine seams.
- `run.py` entry point: `python -m fusion_miniapp.run --agents N [--use-academy] [--preset NAME | --knobs FILE] [--sim real|fixture] [--llm stub|openai] [--out DIR]`. `--use-academy` is read once; the chosen layer is imported lazily.
- While running, both layers print one line per interval to stderr: `agents working=<n> waiting=<n> model_requests=<n> done=<n>`.

## The run record
`record.py` writes `<out>/run_record.json`: all knobs, `agent_layer`, `emulated`/`method`/`note` (all physics output synthetic), package versions, the parent commit, the device probe, counts (agents requested and reported, failed agent processes, sims, LLM calls, proposals by source, tool calls, suspensions where the layer has them), `lookups`, `sim_workers_total`, wall time, and the touch-check verdict summed over all evaluations. One JSON line per finished agent goes to `<out>/agents.jsonl`. Only the agent files of this run's agent processes are read.

## Changes recorded at integration (2026-11-19)
1. The laptop backend for `--sim real` was a one-thread pool inside the agent process. It is now a pool of `workers.sim_workers` spawned processes (at least one per agent process), so simulations never share the agents' process or event loop, as the plan requires; this also ended the HDF5 diagnostics a second thread printed. Spawn, not fork, so each worker's exit hook removes its assets directory. The fixture stays on threads (it simulates nothing). `run.Services.open` gained `processes=`; RHAPSODY needs `cloudpickle` for this, added to both extras.
2. The device probe in the run record is taken inside a simulation worker (a RHAPSODY task), since torch no longer loads in the agent process. The record also names the pool (`sim_pool`) and the largest worker's peak memory.
3. `knobs/smoke.yaml`: `agents.per_process` 4 -> 1000, so laptop runs of up to 1,000 agents use one agent process (each agent process starts its own torch workers).
4. (Superseded below: `workers.gpus_for_sim` was removed.)

## Changes recorded at the review fixes (2026-11-19)
1. Knobs removed: `sim.extra_imports` (it imported nothing in the documented environment), `tools.shared_best` (the board was always empty when read), `workers.gpus_for_sim` (recorded, never acted on). The batch files request one GPU, which every simulation worker uses; how GPUs are split with model servers stays open.
2. `agents.cost_scale` takes a number or a list; agent i uses entry `i % len`. `decide.cost_scale(knobs, seed)`, `decide.budget(knobs, seed)` and `decide.sim_knobs(knobs, seed)` replace the per-run versions; `run.Services.sim_knobs` is gone. `decide.lookups(knobs)` lost its `first` argument; `tools.lookup(what, agent_id, latency_ms)` lost `board`.
3. Best-so-far: an infeasible candidate replaces an infeasible best only if it wins the parent's `_rank_infeasible` against it (before, the latest candidate always replaced it). Feasible candidates follow the parent's `update_best_result` unchanged.
4. `cold_start_fraction` is enforced at the pool (`max_tasks_per_child`), not by rebuilding the context in a warm process. `library_import` is touched once per worker process. `sim.py` sets the parent's `FP_TOL` to 0, so `small_forward` is exactly `puffs x fixed_point_cap`.
5. Both layers admit at most `per_process / count` of `llm.MAX_IN_FLIGHT` model calls at once per process (LangGraph: the fleet; Academy: a semaphore in `_ViaEvaluator.llm`).
6. `run.py`: the device probe runs before the agents; a failed agent process no longer discards the others' results (`record.write(..., failed=)`); with `--sim real` more agent processes than `workers.sim_workers` is refused; `--llm openai` with `llm.model: stub` is refused; `--agents 0` is rejected; the default output folder name ends in the process id; the exit code also fails when fewer agents report than were requested; SIGTERM unwinds like Ctrl-C, and spawned processes end when their parent ends. The record key `tool_service` is now `lookups`.
