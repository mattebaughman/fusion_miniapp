# Fusion mini-app workflow

This is the "Fusion mini-app workflow" box at the top of the AUTONMI stack: LangGraph agents on
FlowGentic and RHAPSODY, each agent running a small fusion design campaign that calls an LLM, uses
tools, and runs a PyTorch simulation. RaDEX and Dragon are not wired in (see Open items).

**All physics output is synthetic.** The simulation runs the real IGNOS inner-loop code over small
stand-in models with random weights. It makes the same kinds of data flow, library loads and
resource touches as the real loop, at a fraction of the size. No result is calibrated, and none
says anything about a plasma. Every run record says so (`emulated`, `method`, `note`).

## Two parts and one flag

1. **`evaluate`, the IGNOS inner loop turned down** (`sim.py`, `standins.py`). It runs the parent's
   own worker and inner-loop functions, unmodified, with stand-ins injected through their `context`:
   it loads checkpoints from disk to the device, builds the models, runs graph forwards on the device,
   hands each stage's output to the next (in memory and through a netCDF file), runs the fixed-point
   loop, and writes the real artifact layout. Every call checks its own touches (below).
2. **The agents** (`decide.py`). One agent is one small campaign: look up constraints and an abstract,
   then S times: ask the LLM for the next candidate, run `evaluate`, update best-so-far; then ask the
   LLM for a summary and report it. The thinking budget travels with every LLM call. Two of the
   PDF's agent behaviours are in-process stand-ins: the lookups are plain async functions with a
   latency knob (not MCP; in the LangGraph layer they are awaited inside graph nodes, not bound as
   graph tools), and the report to the campaign manager is a call on the agent process's `Services`
   object, not a message.

**`--use-academy`** picks the agent layer, once, at invocation:

- off (default): the PDF's stack. A LangGraph graph per agent, hosted by FlowGentic's agent host,
  model calls through FlowGentic's fleet, `evaluate` as a suspending task tool on a RHAPSODY session
  (`layer_langgraph.py`).
- on: Academy agents in the same process (local exchange, no executor): a `CampaignAgent` per agent
  and one `EvaluatorAgent` per process in front of the same RHAPSODY session (`layer_academy.py`).

Both run the same decision code against the same services, so with the stub LLM they give identical
per-agent results. Both admit the same number of model calls at once: each agent process's share
(`per_process / count`) of `llm.MAX_IN_FLIGHT` (512 per model server); the LangGraph layer through
the fleet, the Academy layer through a semaphore (without the fleet's token cap). With `--sim real`
the simulations run in spawned worker processes, never on the agents' event loop; `--sim fixture`
replaces `evaluate` with an instant, torch-free stand-in for large agent counts. While running, both
print `agents working=<n> waiting=<n> model_requests=<n> done=<n>` to stderr once a second.

## Install

Python 3.12. From this folder:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[stack,academy,dev]"
source .venv/bin/activate        # in every new shell (or call .venv/bin/python directly)
```

`stack` is the PDF's stack (FlowGentic 0.5.0 comes from GitHub at a pinned commit, as it is not on
PyPI yet), `academy` is academy-py. torch is not pinned: on a GPU machine install the CUDA or ROCm
build first (each batch file in `launch/` names one). `ENVIRONMENT.md` lists the exact laptop set.

## Run

```sh
python -m fusion_miniapp.run --agents 100 --preset smoke
python -m fusion_miniapp.run --agents 100 --preset smoke --use-academy
```

Other options: `--knobs FILE` (instead of `--preset`), `--sim real|fixture` (default `real`),
`--llm stub|openai`, `--out DIR` (default `results/<time>-<layer>-<agents>-<pid>`). `--llm openai`
needs `llm.model` set in a knob file to the served model's name (the presets say `stub`, which the
run refuses), the endpoint's URL and key in the environment variables the knob file names (default
`OPENAI_BASE_URL`, `OPENAI_API_KEY`), and an endpoint that accepts vLLM's per-request
`thinking_token_budget` field, which every call sends (other servers may reject it; not tried).
With `--sim real` the run stops at once if there are more agent processes than `workers.sim_workers`.
The exit code is non-zero if an agent or an agent process failed, fewer agents reported than were
requested, or an evaluation missed a touch. SIGTERM (scancel, a time limit) unwinds like Ctrl-C:
pools close and temporary folders are removed; a killed agent process takes its workers with it.

Each run writes `run_record.json` (all knobs, the layer, package versions, the parent commit, a
device probe taken in a simulation worker for `--sim real`, the counts, proposals by source (LLM or
the fallback mutation), the summed touches and their verdict, wall time, peak memory; agent processes
that failed are listed and their agents are missing) and `agents.jsonl` (one line per agent). Only
this run's files are read, so a reused `--out` folder does not mix runs. The one-line summary on
stdout includes the devices the evaluations ran on.

## The knob file

One YAML file with five sections; every file is merged over `knobs/smoke.yaml`, so a file needs only
the keys it changes. The turn-down is the `sim` section.

| Section | Knobs |
|---|---|
| `sim` | `nodes_per_plane`, `planes`, `rollout_steps`, `puffs`, `fixed_point_cap` (every fixed point runs to it), `model_width`, `checkpoint_mb`, `device` (auto = cuda if available, else cpu), `cold_start_fraction` (share of evaluations run in a fresh worker process: each worker is retired after round(1 / fraction) tasks), `keep_artifacts`, `work_root` |
| `agents` | `count`, `per_process`, `sims_per_agent` (S), `cost_scale` (a number, or a short list: agent i uses entry i % length; it multiplies `nodes_per_plane` and `fixed_point_cap`), `thinking_budget`, `budget_scales_with_cost` (each agent's budget times its cost scale) |
| `tools` | `constraints_lookup`, `abstract_lookup`, `latency_ms` |
| `llm` | `mode` (stub or openai), `base_url_env`, `api_key_env`, `model` |
| `workers` | `sim_workers` (simulation processes in all; each agent process gets `sim_workers // agent processes`, at least one; the record gives `sim_workers_total`) |

Presets: `smoke` (seconds on a laptop), `hundredk` (the shape of the 100,000-agent run: S = 2, 10 agent
processes of 10,000 agents and 10 simulation workers, heavily turned down), `ignos_shape` (few agents,
S = 320, larger simulations). None is calibrated. Each distinct cost-scale value costs every worker one
more context (models and generated files), so keep a cost-scale list short.

## The touch checklist: what "validated" means here

"Validated" means only that each evaluation made the same kinds of touches as the real loop, in the
numbers the knobs imply. Every evaluation counts its touches and checks them (`touches.py`); a run
with any failed check exits non-zero. Per evaluation, with k = `rollout_steps`, p = `puffs`,
c = `fixed_point_cap`, b = 1 if the context was built in this call else 0, s = small forwards seen:

| Touch | Count |
|---|---|
| library imports (torch, numpy, scipy.io, netCDF4) | 4 on a worker process's first context build, 0 on a later build there (another cost scale) or without one; the check accepts 0 to 4 when b = 1 |
| mesh reads; `.mat` geometry reads by `loadmat` | b + 2 + k; 2b |
| checkpoint loads to the device; models built and moved | b + 2 each (the real loop: 3) |
| graph forwards on the device | 1 + k |
| host-to-device; device-to-host copies | 12 + 2k + s; 1 + 3k + s |
| stage output becomes the next stage's input in memory | 1 + k |
| netCDF hand-off written, rewritten in place, read by two readers | 1, 1, 2 |
| fixed-point small forwards, each with a host round trip | p x c (the parent's tolerance is set to 0, so the cap sets duration) |
| artifact files written | 7 + 2k (11 at k = 2, as in the real loop) |
| context build; result record returned | 0 or 1; 1 |

The real loop starts a fresh process and loads everything for every evaluation; here workers stay
warm unless `cold_start_fraction` retires them. The fixture makes no touches; its evaluations are
counted apart and never pass. The parent's result record still says `"inner_loop": "validated"` (its
own variant name); here that word means only the checklist above. The method label is set to
`miniapp_standin`, and every record has `emulated: true`.

## Files from FusionFM/Demo

Five files under `src/fusion_miniapp/parent/` are copied unchanged from FusionFM/Demo (branch
`evo-innerloop-upgrade`, commit `9768bcb`): `steps/final_inner_loop.py`, `evolutionary_loop.py`,
`steps/step0/profile_generator.py`, `evo_agents.py` and `config/config.yaml`. `PARENT.lock` holds their
SHA-256 hashes and a test checks them. They are configured only through module attributes and the
injected context. `evo_agents.py` is carried but not imported: its agents move whole populations
through process pools, the wrong shape for one-candidate campaign agents. Two of the files
(`evolutionary_loop.py` and `final_inner_loop.py`) contain default paths on a site file system, used
only by the parent's own command line and context builder, which this package never calls. Those
paths include a project directory name and the user-directory names of two other people (see Open
items). Nothing non-public is imported (no MATEY, no xgc_reader, no site paths).

The agents use the parent's sampler, mutation, feasibility flags, and its best-so-far rule: the best
feasible core metric, or, while nothing is feasible, the winner of the parent's core/wall ranking.
The parent applies that ranking to a generation; an agent here evaluates one candidate at a time, so
a new infeasible candidate replaces the best only if it wins the ranking against it.

## Machines

All three targets use Slurm. `launch/` has the device probe (`probe.py`, the first job on each
machine), the shared run script (`run.sh`, which activates the environment named by `FUSION_ENV`),
and `perlmutter.sbatch`, `stellar_ai.sbatch`, `lux.sbatch`, which differ only in their header and
module lines. Account, QOS and partition come from `SBATCH_ACCOUNT`, `SBATCH_QOS` and
`SBATCH_PARTITION`, with no defaults. Each requests one GPU, which every simulation worker uses, and
stops after the probe if torch sees no GPU (so a CPU-only torch build cannot pass on a GPU node).
None of them has been run on its machine. Device rule: `cuda` if available, else `cpu` (ROCm builds
answer through `torch.cuda`); the workers never set a visible-device variable.

## What has run, and what has not

Run on one laptop only (Apple silicon, 12 cores, 24 GB; Python 3.12.13; torch 2.15.0, a CPU/MPS
build), with the stub LLM, on 2026-11-19. These are functional checks, not performance claims.

After the review fixes (CPU):
- the test suite (`.venv/bin/python -m pytest -q`, about 41 s): 48 passed, 1 skipped (CUDA). It runs
  the real `evaluate` on the CPU through both layers end to end, a run with `cold_start_fraction` 1
  (every evaluation in a fresh worker process), and checks that both layers admit the same number of
  model calls at once;
- 100 agents x 2 real evaluations on the smoke preset, flag off and on: exit 0, 200 of 200 touch checks
  pass, identical per-agent results (touch counts aside), about 7 s each;
- failure paths: a simulation worker killed with SIGKILL, an agent process killed with SIGKILL (in a
  run of two), and SIGTERM to the main process or the whole process group. Each run ended, wrote its
  record where an agent process survived, and left no process behind; temporary folders were removed
  except the killed worker's own.

A ladder was run on this build with the commands in `results/laptop_2026-11-19_final/ladder.sh`
(run from this folder). Each run's record is beside it.

| Rung | Layer | Wall time | Peak memory, whole process tree | Touch check |
|---|---|---|---|---|
| (a) 1,000 agents x 1 real `evaluate`, smoke preset, CPU, 1 agent process + 2 workers | LangGraph | 25.3 s | 0.85 GB | 1,000 of 1,000 pass |
| | Academy | 24.2 s | 0.82 GB | 1,000 of 1,000 pass |
| (b) 10,000 agents x 2, fixture, 1 agent process | LangGraph | 77.6 s | 0.87 GB | fixture (no touches) |
| | Academy | 32.6 s | 0.93 GB | fixture |
| (c) 100,000 agents x 2, fixture, 10 agent processes | LangGraph | 131.6 s | 6.0 GB | fixture |
| | Academy | 60.7 s | 7.4 GB | fixture |

Peak memory was sampled once a second. In every rung all agents finished with exit code 0, and the
two layers' results were identical (same `results_sha256` in the two run records).
(b) and (c) used 10,000 agents per agent process, as `hundredk` does.
Earlier build steps also ran `evaluate` and small runs of both layers on Apple MPS.

Not run: CUDA and ROCm (no such GPU here), Perlmutter, Stellar-AI and Lux, the batch files, Dragon,
RaDEX, vLLM or any real model server, FlowGentic's MCP tool route, and the `ignos_shape` preset.

## Open items

- RaDEX and Dragon are not wired in. The stack's Dragon route needs RHAPSODY's native Dragon backend,
  which is on an unreleased branch, and RaDEX, which is distributed as source only. The places they
  would go: `run.Services.open(backend=, store=, model=)` and `layer_langgraph.run_agents(store=,
  replicas=)`; nothing calls these arguments today.
- How GPUs are split between the model servers and `evaluate` on a node (the plan's question 3).
  Today a job requests one GPU and every simulation worker uses it. To use several, start one run
  per GPU.
- The plan's step 1 gives each agent a region of parameter space; here each agent starts from the
  parent's sampler at its own seed and may propose anywhere in range.
- The five parent files contain site paths with a project directory name and two other people's
  user-directory names. They must stay byte-identical here; decide before publishing whether they ship
  as they are.
- A licence file is needed before publishing (the source repository has none).
- Memory grows by tens of KB per agent (RHAPSODY keeps every task, FlowGentic keeps ended tasks and
  final checkpoints); each agent process leaves an empty `rhapsody.session.*` folder in the output.
