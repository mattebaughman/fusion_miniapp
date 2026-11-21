# Fusion mini-app workflow

This is the "Fusion mini-app workflow" box at the top of the AUTONMI stack: LangGraph agents on
FlowGentic and RHAPSODY, where every agent campaign is the IGNOS design loop, which calls an LLM, uses
tools and runs a PyTorch simulation. It is for scaling to 100,000 campaigns running at once. Dragon and
RaDEX are not wired in yet (see Open items).

**All physics output is synthetic.** The simulation runs the real IGNOS inner-loop code over small
stand-in models with random weights. It makes the same kinds of data flow, library loads and
resource touches as the real loop, at a fraction of the size. No result is calibrated, and none
says anything about a plasma. Every run record says so (`emulated`, `method`, `note`).

## What one campaign is

Each campaign is the parent's outer loop, `run_evolutionary_loop` in FusionFM/Demo's
`evolutionary_loop.py`, carried by the parent's three agents from its `evo_agents.py`: a
**Coordinator**, a **Generator** and an **Evaluator**. The parent's unmodified functions do the
sampling, selection, best-result tracking, logging and repopulation:

1. The Generator draws the starting population, `initialize_population(P, ...)`: P full parameter
   sets (Latin hypercube).
2. For each of G generations:
   - The Coordinator sends the whole population to the Evaluator. The Evaluator submits all P
     simulations at once and waits for all of them. This is the parent's generation barrier, and the
     fan-out: at P 16, 100,000 live campaigns can request up to 1.6 million simulations in one
     generation. Each campaign has its own barrier, so the observed peak is in `speed.sims_in_flight`.
   - The Coordinator runs `select_survivors`, `update_best_result` and `log_generation` (the parent's
     `evolution_log.csv` and `evolution_log.jsonl`).
   - Before every generation but the last, the Coordinator sends the survivors to the Generator, which
     runs the parent's `repopulate`. With `outer.generator: llm` (the default) it then makes one model
     call with the generation's results, and each parsed proposal replaces one of the parent's random
     mutants. That is the place `evo_agents.py` names for "a directed proposer". With
     `outer.generator: mutation` there is no model call, and the campaign is the parent's loop exactly.
3. The Coordinator writes `best_result.json`, asks the model for a one-paragraph summary, and reports
   to the campaign manager.

Per campaign: P x G simulations. In llm mode, one model call per refill that has slots to fill, plus
the summary: G calls in all. In mutation mode, only the summary call.

One random generator drives selection and refill, in the parent's order (`outer.rng: shared`, as
`run_evolutionary_loop`). Its state travels with the Coordinator's message to the Generator and back.
`outer.rng: per_agent` gives the Generator and the Coordinator their own streams, as `evo_agents.py`
does.

**Exact match.** `tests/test_parent_oracle.py` runs the unmodified parent driver in a fresh process,
with only its population evaluator replaced by this package's `evaluate`, and compares it with the
mini-app campaign in both agent layers (generator: mutation). It does this for P 4 and G 3:
- fixture seeds 0, 1, 2 and 99 (99 reaches a feasible candidate), one random stream;
- fixture seeds 0 and 99 with two streams, against `evo_agents`;
- real stand-in seeds 0, 1 and 2, one stream, and seed 0 with two streams.

In every case `evolution_log.csv`, `evolution_log.jsonl` (timestamps aside) and `best_result.json` are
identical. The CSV holds every generation's candidates, results and survivors.

## The simulation: `evaluate`, the IGNOS inner loop turned down

`sim.py` and `standins.py` run the parent's own worker (`_evaluate_single_candidate`) and inner loop
(`evaluate_final_innerloop`), unmodified, with stand-ins injected through their `context`. Each
evaluation:
- loads checkpoints from disk to the device and builds the models;
- runs graph forwards on the device;
- hands each stage's output to the next, in memory and through a netCDF file;
- runs the fixed-point loop;
- writes the real artifact layout under `gen{g}_candidate{i}/final_innerloop`.

Every call checks its own touches (below). `--sim fixture` replaces `evaluate` with an instant,
torch-free stand-in for large agent counts. It makes no touches.

## Two agent layers, one flag

`--use-academy` picks the agent layer, once, at invocation. Both layers give every campaign the same
three agents, carry the same messages and call the same code (`decide.py`), so they give identical
results. `results_sha256` in the run record is the same for both.

| | Default (the PDF's stack, `layer_langgraph.py`) | `--use-academy` (`layer_academy.py`) |
|---|---|---|
| The three agents | three compiled LangGraph subgraphs in one campaign graph, talking through mailbox keys of its state | three Academy agents per campaign on the process's local exchange, talking through action calls |
| Hosting | FlowGentic's `AgentHost`, one hosted graph run per campaign | Academy's `Manager`, no executor; a campaign launches its agents when it gets a slot and shuts them down when it ends |
| The generation's P simulations | P suspending `evaluate` task tools in one step: FlowGentic saves and drops the campaign until all P end (one suspension per generation) | the Evaluator's action awaits all P RHAPSODY tasks |
| Model calls | FlowGentic's fleet | a semaphore with the same per-process share |

Both layers:
- admit at most each agent process's share (campaigns in the process ÷ all campaigns) of
  `llm.MAX_IN_FLIGHT` (512 per model server) model calls at once;
- cap campaigns live at once per process with `agents.max_live` (null: all of them);
- run real simulations in spawned worker processes, never on the agents' event loop;
- print `agents working=<n> waiting=<n> model_requests=<n> done=<n>` to stderr once a second.

## Install

Python 3.12. From this folder:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[stack,academy,dev]"
source .venv/bin/activate        # in every new shell (or call .venv/bin/python directly)
```

- `stack` is the PDF's stack. FlowGentic 0.5.0 comes from GitHub at a pinned commit, as it is not on
  PyPI yet.
- `academy` is academy-py.
- torch is not pinned: on a GPU machine install the CUDA or ROCm build first (each batch file in
  `launch/` names one).
- `ENVIRONMENT.md` lists the exact laptop set.

## Run

```sh
python -m fusion_miniapp.run --preset smoke                 # 100 campaigns, P 4, G 2
python -m fusion_miniapp.run --preset smoke --use-academy
```

**Options:**
- `--agents N` (campaigns; default the knob file's `agents.count`);
- `--preset smoke|hundredk|ignos_shape`;
- `--knobs FILE`, repeatable, merged in order over the preset (for example a machine file from
  `knobs/machines/`);
- `--sim real|fixture` (default `real`);
- `--llm stub|openai`;
- `--out DIR`. It must be a new or empty folder: the parent's logs append, so a folder that already
  holds a run is refused.

**`--llm openai`** needs:
- `llm.model` set in a knob file to the served model's name (the run refuses `stub`);
- the endpoint's URL in the environment variable the knob file names (default `OPENAI_BASE_URL`), and
  its key in `OPENAI_API_KEY` if it needs one (a local server without authentication does not).

Every call sends vLLM's per-request `thinking_token_budget`; set `llm.send_thinking_budget: false` for
a server that rejects it. `llm.timeout_s` and `llm.max_retries` bound each call. The run fails (see
`llm_check` below) if the model filled fewer than `llm.min_model_share` of the refill slots it was
asked for, or if more than `llm.max_empty_share` of its replies were empty. Without this check, a
model that never answers would leave every campaign running on the parent's random mutants, and the
run would still pass.

**Exit codes:**
- 0 (`status: pass`): every campaign finished, every real evaluation passed its touch check, and the
  model check passed;
- 1 (`fail`);
- 143 (`interrupted`).

**How a run ends badly:**
- SIGTERM or Ctrl-C (scancel, a Slurm time limit): the agents are cancelled through the event loop,
  the simulation workers are stopped within 20 s (leftovers killed), and a record with `status:
  interrupted` is still written.
- A simulation worker that dies: its pool rebuilds once and resubmits the lost simulations.
- A cold-start child (below) exits when its worker dies.

## What a run writes

- `run_record.json`:
  - `status`;
  - `run`: time, host, Slurm job id, this package's git commit and whether it had uncommitted changes,
    the execution backend;
  - all knobs, the layer, package versions (with FlowGentic's and RHAPSODY's commits), the parent
    commit, a device probe taken in a simulation worker;
  - `counts`: campaigns, simulations, model calls, generations, candidates by source (start, survivor,
    mutation, model, restart), refill slots asked and filled, suspensions;
  - `llm_service` (calls, empty replies, errors, retries, finish reasons, tokens) and `llm_check`;
  - the summed touches and their verdict;
  - `speed` (below);
  - peak memory (agent processes, and simulation workers by their own high-water mark);
  - `threads`, pool rebuilds, `results_sha256`.
- `agents.jsonl`: one line per campaign:
  - its best result, per-generation summaries, candidate sources and refill counts;
  - its evaluations' simulation and queue-wait times;
  - every model call's latency, finish reason, token counts and empty flag (`llm_call_details`).
- `campaigns/<id>/` (`outer.logs: per_agent`): the parent's `config.json`, `evolution_log.csv`,
  `evolution_log.jsonl` and `best_result.json` for each campaign. At 100,000 campaigns that is 400,000
  small files, so `hundredk` sets `outer.logs: none`.
- stdout: a one-line JSON summary, then one parsable `speed:` line.

**The `speed` section**, for the goal of 100,000 campaigns working at once:
- campaigns live at once: the maximum, the time-weighted mean, the seconds with at least 95% of the
  requested campaigns live, and the ramp-up to the last start;
- the agent phase (first campaign start to last finish), with start-up and shutdown reported apart;
- evaluations, model calls, generations and campaigns per second;
- simulations in flight (submitted, not returned) and executing (on a worker): the maximum and the
  mean;
- the mean, 50th and 95th percentiles and the maximum of simulation time, queue wait (from the
  generation's submission to the evaluation's start), generation time (submission to last result) and
  model-call latency;
- event-loop lag (the same 100 ms timer in both layers);
- `sim_bound_ratio`: the simulations' own time spread over the workers, against the agent phase
  (1 means the run took no longer than its simulations had to).

An interrupted run's speed numbers cover only the campaigns that finished (`speed.coverage`).

`python -m fusion_miniapp.compare_records A/run_record.json B/run_record.json` checks two runs (two
machines, or both layers) against each other:
- statuses, counts and non-machine knobs must agree;
- the campaigns must agree too: exactly by digest, or campaign by campaign within `--rtol`/`--atol`
  when last digits differ between machines;
- speed numbers are reported, never failed on.

## The knob file

One YAML file with six sections. A preset and any number of knob files are merged in order over
`knobs/smoke.yaml`, so a file needs only the keys it changes. The turn-down is the `sim` section.

| Section | Knobs |
|---|---|
| `sim` | `nodes_per_plane`, `planes`, `rollout_steps`, `puffs`, `fixed_point_cap` (every fixed point runs to it), `model_width`, `checkpoint_mb`, `device` (auto = cuda if available, else cpu), `cold_start_fraction` (share of evaluations run in a fresh single-use process, chosen from the campaign, generation and candidate number, so it repeats), `keep_artifacts`, `work_root` (`$VARS` expanded; an unset one is an error), `threads_per_worker` (auto = the CPUs this process may use ÷ `workers.sim_workers`; sets OMP, OpenBLAS, MKL and torch threads) |
| `agents` | `count` (campaigns), `per_process`, `max_live` (null = all of a process's campaigns), `cost_scale` (a number, or a short list: campaign i uses entry i % length; it multiplies `nodes_per_plane` and `fixed_point_cap`), `thinking_budget`, `budget_scales_with_cost` |
| `outer` | `population` (P), `generations` (G), `generator` (llm or mutation), `rng` (shared or per_agent), `logs` (per_agent or none), `wall_weight` (the parent's 0.7), `noise` (the parent's 0.1), `parent_stdout` (keep or drop the parent's per-generation prints) |
| `tools` | `constraints_lookup`, `abstract_lookup`, `latency_ms` |
| `llm` | `mode` (stub or openai), `base_url_env`, `api_key_env`, `model`, `send_thinking_budget`, `timeout_s`, `max_retries`, `min_model_share`, `max_empty_share` |
| `workers` | `sim_workers` (simulation processes in all; each agent process gets `sim_workers // agent processes`, at least one) |

| Preset | Campaigns | P | G | Simulations per campaign | Simulations in all | Notes |
|---|---|---|---|---|---|---|
| `smoke` | 100 | 4 | 2 | 8 | 800 | seconds with `--sim fixture`, a few minutes with real simulations on a laptop CPU |
| `ignos_shape` | 4 | 16 | 20 | 320 | 1,280 | the parent campaign's shape; larger simulations |
| `hundredk` | 100,000 | 16 | 2 | 32 | 3.2 million | 10 agent processes of 10,000; `logs: none`, `parent_stdout: drop` |

None is calibrated. With real simulations, `hundredk` is set by simulation throughput (about 0.16 s
each, so roughly 14 hours on one laptop's 10 workers). The agent-count ladder uses the fixture.

## The touch checklist: what "validated" means here

"Validated" means only that each evaluation made the same kinds of touches as the real loop, in the
numbers the knobs imply. Every evaluation counts its touches and checks them (`touches.py`); a run
with any failed check exits non-zero. Per evaluation, with k = `rollout_steps`, p = `puffs`,
c = `fixed_point_cap`, b = 1 if the context was built in this call else 0, s = small forwards seen:

| Touch | Count |
|---|---|
| library imports (torch, numpy, scipy.io, netCDF4) | 4 on a process's first context build, 0 on a later build there (another cost scale) or without one; the check accepts 0 to 4 when b = 1 |
| mesh reads; `.mat` geometry reads by `loadmat` | b + 2 + k; 2b |
| checkpoint loads to the device; models built and moved | b + 2 each (the real loop: 3) |
| graph forwards on the device | 1 + k |
| host-to-device; device-to-host copies | 12 + 2k + s; 1 + 3k + s |
| stage output becomes the next stage's input in memory | 1 + k |
| netCDF hand-off written, rewritten in place, read by two readers | 1, 1, 2 |
| fixed-point small forwards, each with a host round trip | p x c (the parent's tolerance is set to 0, so the cap sets duration) |
| artifact files written | 7 + 2k (11 at k = 2, as in the real loop) |
| context build; result record returned | 0 or 1; 1 |

The real loop starts a fresh process and loads everything for every evaluation. Here workers stay warm
unless `cold_start_fraction` sends an evaluation to a fresh process (3 checkpoint loads, 4 library
imports).

The fixture makes no touches; its evaluations are counted apart and never pass. The parent's result
record still says `"inner_loop": "validated"`, which is the parent's name for its variant; here that
word means only the checklist above. The method label is `miniapp_standin`, and every record has
`emulated: true`.

## Files from FusionFM/Demo, and the declared differences

Five files under `src/fusion_miniapp/parent/` are copied unchanged from FusionFM/Demo (branch
`evo-innerloop-upgrade`, commit `9768bcb`):
- `steps/final_inner_loop.py`;
- `evolutionary_loop.py`;
- `steps/step0/profile_generator.py`;
- `evo_agents.py`;
- `config/config.yaml`.

`PARENT.lock` holds their SHA-256 hashes, and a test checks them. They are configured only through
module attributes and the injected context. `evo_agents.py` is the reference for `outer.rng:
per_agent` in the exact-match test; the two layers re-create its three agents rather than import it.
Two of the files (`evolutionary_loop.py` and `final_inner_loop.py`) contain default paths on a site
file system. Only the parent's own command line and context builder use them, and this package never
calls either. Those paths include a project directory name and the user-directory names of two other
people (see Open items).

**Where a campaign differs from the parent's loop, on purpose:**
- The population evaluator: the parent runs batches of `n_gpus` in a process pool, or `srun`; here the
  P simulations are RHAPSODY tasks on the agent process's worker pool.
- `outer.generator: llm`: the model's proposals replace mutants. They replace them only after the
  parent's `repopulate` has run, so selection and mutation draw the same random numbers. Proposals can
  still change later generations' survivors, so only `mutation` mode is the parent's trajectory.
- `config.json`: the parent's keys, with this package's values. The checkpoints are stand-ins (or none,
  for the fixture), and each campaign records its own cost-scaled simulation knobs.
  `runtime_source_sha256` and `git` are computed once per agent process, not once per campaign (300,000
  `git` calls at 100,000 campaigns).
- `best_result.json` is written after the last generation, before the summary model call (the parent
  has no summary).
- The Coordinator logs after updating the best result, as `run_evolutionary_loop` does; `evo_agents`
  logs first. The outputs are the same.
- `outer.parent_stdout: drop` discards the parent's prints. `select_survivors` prints a line for every
  generation with no feasible candidate.
- Results cross processes as JSON, so NaN arrives as None. The Coordinator puts NaN back for the
  parent's float keys before selection and logging.

## Machines

All three targets use Slurm. **The app's simulations need GPUs, and a run without one fails.** Each
batch file:
- requests one GPU;
- writes under the machine's scratch, never the package folder;
- applies the machine's knob file (`knobs/machines/<machine>.yaml`, `sim.device: cuda`) after the
  user's presets and files;
- runs the device probe first (`launch/probe.py`, which needs the GPU and a finite convolution) and
  stops if it fails;
- asks Slurm for SIGTERM 120 s before the time limit, so the run writes an interrupted record.

The batch files:
- `perlmutter.sbatch`: shared QOS, 1 A100, 32 CPUs, output under `$PSCRATCH`.
- `stellar_ai.sbatch`: output under `$FUSION_SCRATCH`.
- `lux.sbatch`: output under the member work area, MIOpen caches on node-local storage.

They share `launch/batch.sh` and `launch/run.sh` (which activates `FUSION_ENV`). The account comes from
`SBATCH_ACCOUNT`. For example:

```sh
sbatch --output="$PSCRATCH/%x-%j.out" "$FUSION_MINIAPP_DIR/launch/perlmutter.sbatch" --preset smoke
```

How GPUs are split between the simulations and model servers on a node is open. Device rule: `cuda` if
available, else `cpu` (ROCm builds answer through `torch.cuda`). The workers never set a
visible-device variable.

## What has run, and what has not

As of 2026-11-21, on one laptop (Apple silicon, 12 cores, 24 GB; Python 3.12.13; torch 2.15.0, a
CPU/MPS build), with the stub LLM. These are functional checks, not performance claims.

- **Tests:** `.venv/bin/python -m pytest -q`, about 90 s: 88 tests, 1 skipped (CUDA). They cover:
  - the exact match with the parent (above);
  - both layers agreeing;
  - the real `evaluate` on the CPU through both layers end to end;
  - cold starts;
  - a dead worker rebuilt, and a cold child ending with its worker;
  - the model check against a local endpoint that returns empty replies;
  - the speed section;
  - the record comparison;
  - the batch files against a stub `srun`.
- **10,000 campaigns** in the `hundredk` shape (P 16, G 2), fixture on 10 threads, one agent process:
  320,000 simulations in each layer, all 10,000 campaigns live at once, and the same `results_sha256`
  (Codex also paired the campaigns one by one: no differences).

  | Layer | Time | Simulations/s | Agent-process peak | Per campaign |
  |---|---|---|---|---|
  | LangGraph | 462 s | 694 | 6,142 MiB | about 629 KiB |
  | Academy (30,000 Academy agents) | 172 s | 1,922 | 3,553 MiB | about 364 KiB |

  The LangGraph run overlapped a Docker image build, and the Academy run did not, so the times are not a
  clean comparison. The per-campaign figures include the interpreter and the frameworks. The fixture runs
  on threads in
  the agent process, so LangGraph's heavier per-campaign work also slowed the fixture (median 10 ms
  against 4.5 ms).
- **Codex's 80 fault trials** on macOS, 40 per layer:
  - 40 cold-start runs at fractions 0.25 and 0.5: all identical, with no retries;
  - 20 runs with a worker killed mid-simulation: all recovered, ending 4.9 to 8.0 s after the kill;
  - 20 runs with SIGTERM to the whole job during a cold simulation: all wrote an interrupted record
    within 0.24 s.

  None left scratch files behind.

- **Ladder rung (a): 1,000 campaigns** in the smoke shape (P 4, G 2) with the real stand-ins on the CPU,
  one agent process and 4 workers. Both layers ran 8,000 real simulations, all passing their touch checks,
  with the same `results_sha256`:
  - LangGraph: 77.7 s, agent process 487 MiB at peak;
  - Academy: 75.4 s, 348 MiB.

  Each worker peaked at about 300 MiB. The simulation-bound ratio was 0.95 and 0.98, so at this size the
  simulations, not the agents, set the time. Rung (b) is the 10,000-campaign fixture run above. The two
  rungs' records and commands are in `results/laptop_2026-11-21/`.
- **The reference problem, `ignos_shape`** (4 campaigns x P 16 x G 20, real stand-ins on the CPU):
  - both layers pass, with 1,280 of 1,280 touch checks;
  - the same `results_sha256`;
  - byte-identical `evolution_log.csv` and `best_result.json` (the JSONL logs differ only in timestamps).

  Its records are in `results/ignos_shape_reference/`, with how to check another machine against them.
- **Linux** (native arm64 container: Debian trixie, Python 3.12.16, CPU torch 2.15.0, the frozen package
  set):
  - the test suite: 86 passed, 2 skipped;
  - the same 80 fault trials as on macOS, all passed;
  - the CPU-thread limit: with `auto` (3 threads each for 4 workers on 12 CPUs), 40 real campaigns took
    18.4 s; with 12 threads per worker they took 76.2 s (46 threads each, 4.1 times slower). Each
    worker's own peak memory is recorded (360 MiB).
- **Slurm** (a one-node Slurm 24.11 in that container, `perlmutter.sbatch` unchanged, the package folder
  mounted read-only). Two labelled test substitutions: a `module` command that knows only `load conda`,
  and a wrapper that drops only the probe's GPU requirement, since that torch is CPU-only.
  - A normal job completed, writing only under the scratch path.
  - A job with a 3-minute limit got Slurm's early TERM (`--signal=TERM@120`) and wrote an interrupted
    record, keeping both agent processes' counts and exiting 143.
- **A fresh install from this README** (a clean `python:3.12.16-trixie` container, `git archive` of the
  commit): the smoke preset passed in both layers, and `compare_records` reported "results agree".

**Not run:**
- CUDA and ROCm (no such GPU here);
- Perlmutter, Stellar-AI and Lux;
- Dragon, RaDEX, vLLM or any real model server;
- FlowGentic's MCP tool route.

The 2026-11-19 ladder in `results/laptop_2026-11-19_final/` predates the outer loop. Its campaigns
evaluated one candidate at a time, and its memory column is MiB ÷ 1000.

## Open items

- **Dragon and RaDEX are not wired in.** RHAPSODY's released Dragon backend (rhapsody-py 0.7.0) ran this
  workload correctly when passed in by hand, and the same goes for the RaDEX store. Choosing them from
  the command line and batch files is the next round. The places they go:
  `run.Services.open(backend=, store=, model=)` and `layer_langgraph.run_agents(store=, replicas=)`.
- **How GPUs are split between the model servers and the simulations** on a node. Today a job requests
  one GPU and every simulation worker uses it.
- **The model server and model** for the first Perlmutter job.
- **Memory per live campaign at P 16:** about 629 KiB (LangGraph) and 364 KiB (Academy) of host memory
  per agent process, measured with fixture campaigns. That puts 100,000 campaigns in 10 processes at
  about 60 GiB and 35 GiB: more than this 24 GB laptop. Not measured: a full 100k run, real GPU
  simulations, and the model servers' memory, all of which a node's capacity plan needs.
  - A finished LangGraph campaign now releases its checkpoints from the private store, and the cache of
    finished simulation results is bounded (1,024). Each finished campaign's result and statistics are
    still kept until the run ends.
  - The peak while every campaign is live is held by the libraries: RHAPSODY's and FlowGentic's copies of
    each result, and a RHAPSODY task cycle that only Python's cyclic collector frees. Reducing it is
    upstream work.
- **The five parent files** contain site paths with a project directory name and two other people's
  user-directory names. They must stay byte-identical here; decide before publishing whether they ship
  as they are.
- **Publishing:** a lockfile, a release tag, and the multi-node path.
