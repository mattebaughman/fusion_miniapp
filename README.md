# Fusion mini-app workflow

This is the "Fusion mini-app workflow" box at the top of the AUTONMI stack. Every agent campaign is the IGNOS
fusion design loop: the parent's evolutionary outer loop, whose candidates are proposed, run, watched and judged
by agents that make their own model calls, with the IGNOS inner loop (turned down) evaluating each candidate.
It is built to run 100,000 hosted agents at once: on the AUTONMI layer a campaign is 5 agents, so 20,000 campaigns.

- **The final shape** is the AUTONMI layer (`layer_autonmi.py`): per campaign a Coordinator, a Generator, an
  Evaluator and two Monitor slots, each a hosted agent with its own model calls, in one of two modes:
  - `campaign.mode: generations`, the parent's generation barrier, whose results equal IGNOS's exactly;
  - `campaign.mode: async`, one result at a time, with no barrier.
- **Two more agent layers**, LangGraph on FlowGentic and RHAPSODY (`layer_langgraph.py`, the default of
  `fusion_miniapp.run`) and Academy (`--use-academy`), run the same campaigns with three agents each, in the
  generations mode only. They are kept unchanged as the comparison the AUTONMI layer is checked against.
- **The run path on StellarAI** (`launch/stellar_ai_autonmi.sbatch`) brings up NVIDIA Nemotron 3.5 Lightning on
  vLLM, checks every server, and runs the campaigns against it. It is configured and checked on a laptop only.

**All physics output is synthetic.** The inner loop runs the real IGNOS inner-loop code over small stand-in
models with random weights. It makes the same kinds of data flow, library loads and resource touches as the
real loop, at a fraction of the size. No result is calibrated, and none says anything about a plasma. Every run
record says so (`emulated`, `method`, `note`).

The per-candidate evaluation is called **the inner loop** in this README; code names such as `--sim`, `sim.py`
and the task `Simulate` keep the older word.

## Quick start on a laptop

The AUTONMI layer needs the AUTONMI framework, which is not public yet (the RADICAL team can provide access),
installed into an environment made from `requirements-autonmi.txt` (its header gives the steps; see Install).
With that environment active (`source .venv-autonmi/bin/activate`), from this folder:

```sh
python -m pytest -q          # the tests: about 9 minutes on a laptop on battery in Low Power Mode, less on AC
python -m fusion_miniapp.layer_autonmi --preset smoke --sim fixture --agents 100 --out results/smoke-autonmi
python -m fusion_miniapp.run --preset smoke --sim fixture --agents 100 --out results/smoke-langgraph
python -m fusion_miniapp.compare_records results/smoke-autonmi/run_record.json results/smoke-langgraph/run_record.json
printf 'campaign: {mode: async}\n' > results/async.yaml
python -m fusion_miniapp.layer_autonmi --preset smoke --sim fixture --agents 100 --knobs results/async.yaml \
    --out results/smoke-async
```

- The first run is the generations mode on the AUTONMI layer: 100 campaigns of the smoke shape (population 4,
  2 generations), the fixture in place of the inner loop, the stub LLM. The second is the same campaigns on the
  LangGraph layer, and `compare_records` finds their results equal ("results agree").
- The last is the asynchronous mode: every agent makes its own call to the stub LLM.
- Each run writes `run_record.json`, `agents.jsonl` and AUTONMI's folders under its `--out`, prints a one-line
  JSON summary and a `speed:` line, and exits 0 when it passes.
- The StellarAI launch script's dry run, which prints what it would do and needs no Slurm:

```sh
FUSION_DRY_RUN=1 FUSION_MINIAPP_DIR="$PWD" FUSION_ENV="$VIRTUAL_ENV" FUSION_SCRATCH=/tmp/fusion-dry \
    bash launch/stellar_ai_autonmi.sbatch --agents 100 --sim fixture
```

## What one campaign is

Each campaign is the parent's outer loop, `run_evolutionary_loop` in FusionFM/Demo's `evolutionary_loop.py`,
carried by the parent's three agents from its `evo_agents.py`, a **Coordinator**, a **Generator** and an
**Evaluator**, with **Monitor slots** on the AUTONMI layer. The parent's unmodified functions do the sampling,
selection, best-result tracking, logging and mutation (`initialize_population`, `select_survivors`,
`update_best_result`, `log_generation`, `repopulate`, `mutate_params`).

| Agent | How many | What it does |
|---|---|---|
| Coordinator | 1 | the outer loop: survival, the survivors, best-so-far, the parent's log, the budget; guidance to the Generator |
| Generator | 1 | the candidates: the evolutionary loop's profiles, reconciled with its model call, onto the campaign's queue |
| Monitor | `campaign.slots` (2) | each takes the next candidate from the queue, runs its inner loop, scores its trace and writes a note |
| Evaluator | 1 | gathers the Monitors' reports and scores each result |

- **On the AUTONMI layer** each agent is a hosted agent of its own: `a<i>/coordinator`, `a<i>/generator`,
  `a<i>/evaluator`, `a<i>/monitor0`, `a<i>/monitor1`, ... Each has its own state, start and end, row in AUTONMI's
  records (`app/agents-p*.jsonl`), model calls (under its own id) and lookups. So a campaign is 3 +
  `campaign.slots` agents: 5 by default.
- **They talk only through messages**: JSON copies in a mailbox per population (`layer_autonmi.Mailbox`), and
  each campaign's **candidate queue** (`a<i>/candidates`). The Generator puts candidates on the queue, and a free
  Monitor takes the next one. The Generator never addresses a Monitor, and no candidate passes through the
  Coordinator.
- A waiting agent awaits its message inside its step, counted as waiting on a tool, and is never polled. Only a
  Monitor's wait for its inner loop is AUTONMI's real suspension (its state goes to the store).
- **Lookups:** the Coordinator makes the constraints lookup, the Generator the abstract lookup
  (`decide.LOOKUP_ROLE`; `tools.py`, plain in-process functions).
- **Every id seen outside the agents is the campaign's**, `a<i>`: the stub's key, the evaluations' `agent_id`
  (cold starts, work folders), the lookups, the store key `campaign/a<i>`, the campaign's folder and `run_tag`.
- **On the LangGraph and Academy layers** a campaign is the three outer-loop agents (no Monitor slots), and the
  Evaluator runs the generation's P inner loops itself.

### The generations mode (`campaign.mode: generations`, the default)

The parent's loop with its generation barrier, and the check against IGNOS:

1. The Generator draws the starting population, `initialize_population(P, ...)`: P parameter sets (Latin
   hypercube).
2. For each of G generations:
   - the Generator puts the generation's P candidates on the queue, and the Monitor slots run them
     `campaign.slots` at a time (`campaign.slots: 16` runs a generation of 16 at once);
   - the Evaluator gathers the P reports and passes the results on in candidate order (the barrier);
   - the Coordinator runs `select_survivors`, `update_best_result` and `log_generation` (the parent's
     `evolution_log.csv` and `evolution_log.jsonl`), on the inner loop's numbers alone;
   - before every generation but the last, the Generator runs the parent's `repopulate`. With
     `outer.generator: llm` (the default) it then makes one model call, the refill, with the generation's
     results, and each parsed proposal replaces one of the parent's random mutants. That is the place
     `evo_agents.py` names for "a directed proposer". With `outer.generator: mutation` there is no refill call,
     and the campaign is the parent's loop exactly.
3. The Coordinator writes `best_result.json`, makes the summary call (a one-paragraph summary) and reports.

- Per campaign: P x G inner loops, and G model calls in llm mode (each refill with slots to fill, plus the
  summary).
- One random generator drives selection and refill in the parent's order (`outer.rng: shared`, as
  `run_evolutionary_loop`). `outer.rng: per_agent` gives the Generator and the Coordinator their own streams, as
  `evo_agents.py` does.
- The Monitors' numbers change no result in this mode, so the results equal IGNOS's at any `campaign.slots`.
  The Evaluator labels each generation's traces at the barrier (below).
- With `campaign.notes: true` (off by default) each Monitor also asks the model for a note on its candidate, and
  the Evaluator for an assessment of each generation (a keep-probability for each candidate). They change no
  result and are recorded with `record.extras`.

**Exact match.** `tests/test_parent_oracle.py` runs the unmodified parent driver in a fresh process, with only
its population evaluator replaced by this package's `evaluate`, and compares it with the campaign in every layer
(generator: mutation; on the AUTONMI layer through its entry point, one campaign, `agents.first` = the seed). It
does this for P 4 and G 3:
- fixture seeds 0, 1, 2 and 99 (99 reaches a feasible candidate), one random stream;
- fixture seeds 0 and 99 with two streams, against `evo_agents`;
- real stand-in seeds 0, 1 and 2, one stream, and seed 0 with two streams.

In every case `evolution_log.csv`, `evolution_log.jsonl` (timestamps aside) and `best_result.json` are
identical. The CSV holds every generation's candidates, results and survivors. The layers' runs are also
compared with references recorded from the LangGraph layer (`tests/references/*.json`: results_sha256, each
campaign's hash, counts, parent files and model requests, with the commit, command and date that made them;
`python tests/references/record_references.py` records them again from a commit, default HEAD).

### The asynchronous mode (`campaign.mode: async`, the AUTONMI layer only)

No generations and no barrier: each agent repeats one cycle on what has arrived, and a result is applied as it
arrives. The decisions are pure functions of each agent's state and what arrived (`decide.async_*`).

| Agent | Each cycle | Its model call | After 3 failed tries |
|---|---|---|---|
| **Generator** | Once no candidate waits on the queue and budget is left: take the next 2 profiles from the evolutionary loop, reconcile them in one call, put both on the queue | One per 2 candidates: a JSON list of 2 objects, up to 7 parameter values each | The evolutionary loop's profiles, unchanged |
| **Monitor slot** (2) | Take the next candidate; run its inner loop as one task; score its trace against the campaign's rolling trace statistics; send the result, its numbers and its note to the Evaluator | One per candidate: the note | Numbers, no note |
| **Evaluator** | Take the reports that have arrived (with its model call at most 8, the rest left for its next cycle); score each one; send the scores to the Coordinator and a digest to the Generator | One per cycle: a keep-probability for each result | The score from numbers |
| **Coordinator** | Take the scored results that have arrived; decide survival; update the survivors, best-so-far, the log and the budget; send the survivors to the Generator; only then make its call | One per cycle: guidance to the Generator on what to explore next. It decides nothing about survival | No guidance; survival is unaffected |

- **Where profiles come from.** A campaign's P x G candidates are numbered 0 to P x G - 1 as they are made. The
  first P are the parent's `initialize_population`, drawn once. Each later one is the parent's `mutate_params` of
  a survivor chosen at random from the newest survivors the Generator holds. If no survivors have arrived when
  candidate P is due, the Generator waits for the first survivors message: its only wait.
- **Reconciling.** The Generator's prompt shows the 2 profiles, the newest survivors, the Evaluator's latest
  scores and the Coordinator's guidance. A value in the reply replaces the profile's value for that parameter,
  clipped to range; a parameter the reply leaves out keeps the evolutionary loop's value. The request's last line
  names its candidates ("Candidates 5 and 6 of 32: reply with a JSON list of 2 objects ..."), and its JSON schema
  goes as `response_format`.
- **The score is a keep-probability between 0 and 1.**
  - `q`, from numbers: the keep-probability the parent's `select_survivors` would give this result if it and the
    campaign's last P scored results were one generation (best always kept, worst never, the rest by rank;
    `decide.keep_probabilities`, the parent's own ranking).
  - `m`, the Monitor's number: from -1 (the trace looked like low-ranked candidates') to +1 (like high-ranked
    ones'), 0 when it cannot tell.
  - With the model, the Evaluator's call reads each result's numbers, q, the Monitor's numbers and its note, and
    its reply is the score. The prompt shows neither the score from numbers nor its formula.
  - Without the model (`llm.mode: off`, or after 3 failed tries): score = q + `monitor.weight` x m, clipped to
    0..1. `monitor.weight` defaults to 0.25; at 0 this is IGNOS's rule over the last P results.
- **Survival.** The Coordinator keeps a result when its draw from its own random stream is below the score, and
  every result during `campaign.warmup` (4). A kept result joins the survivors, at most P, the oldest leaving.
- **Random streams.** This mode always has the two streams of `outer.rng: per_agent`: the Generator draws its
  parents and mutations from stream 0, the Coordinator its survival draws from stream 1.
- **Best-so-far and the log.** After each result, the parent's `update_best_result` gets the window and the
  result as the generation and their top-ranked as the survivor. One parent `log_generation` row per result,
  its arrival number as the generation.
- **Good and bad, for the Monitors' trace statistics:** a trace is good when its result's q is 0.5 or more. The
  fixture's results follow a hash unrelated to its trace, so there the label comes from the trace's own end
  values, marked synthetic.
- **Results differ from run to run** with the order of arrival. One order, one seed and the Monitors' m give one
  campaign; the tests replay recorded orders through the pure functions (`tests/async_replay.py`). AUTONMI's
  checks are this mode's invariants: P x G inner loops per campaign, each candidate once, every message
  received, at most P survivors, at most `campaign.slots` inner loops running and 2 candidates waiting, and each
  role's messages and calls within bounds tied to its cycles.
- A campaign's line has no generations (`counts.generations` is 0). Beside each evaluation's `sim_s` and
  `wait_s` it has its submission time `t_submit`, and the counts `n_cycles` and `survival`.

## Model calls

Every call is one request with a whole reply: no token streaming, and no tools chosen by a model. Each agent
has at most one call open. Calls go through Flowgentic's fleet: to fusion's stub LLM, or with `--llm openai` to
the model servers (The model service, below).

**The four calls of the asynchronous mode**, and their fallbacks:

| Role | When | Its prompt (`decide.py`) | Its reply | Reply allowance | Fallback |
|---|---|---|---|---|---|
| Generator | per take of 2 | `async_generator_messages` | a JSON list of 2 objects (`parse_reconcile`) | 320 tokens (`reply_tokens(2)`) | the profiles unchanged |
| Monitor | per candidate | `note_messages` | any text: the note (`parse_text`) | 128 | no note |
| Evaluator | per cycle | `scores_messages` | a JSON object of each candidate's keep-probability (`parse_scores`) | 32 + 16 per result | the score from numbers |
| Coordinator | per cycle, after it sends the survivors | `guidance_messages` | any text: the guidance (`parse_text`) | 256 | no guidance |

- **Tries.** A call that fails is tried up to 3 times (`layer_autonmi.TRIES`), then the role's fallback is used
  and counted. A failure is an error, or a reply the role's parse function cannot use. A try is one request; it
  includes Flowgentic's own wait for a server that went away (`llm.outage_s`). No try is made once another agent
  has failed the campaign. The fallbacks are the `llm.mode: off` path, so a campaign whose calls all fail is the
  campaign with no model calls.
- **Every prompt of the asynchronous mode fits the context by construction.** A Monitor's note or the
  Coordinator's guidance shown inside a later prompt is cut to `decide.NOTE_CHARS` (1,200) characters; the record
  keeps the whole text with `record.extras`. One Evaluator call shows at most `decide.SHOWN_RESULTS` (8) results;
  the rest wait in its inbox for its next cycle, in arrival order. At the StellarAI shape (population 16, 32
  candidates) each role's longest prompt is under 21,504 characters, three quarters of a 32,768-token context less
  the 4,096-token reply cap (`tests/test_async.py`). The bounds grow with `outer.population` (survivors and
  scores the Generator shows) and the campaign's budget (the results the Coordinator judges in one cycle).
- **The prompt-length check.** Before a role's call is sent, its prompt's characters (at least its tokens) plus
  `max_tokens` are compared with the smallest `max_model_len` the servers report. A prompt too long is not sent
  and not tried again: the fallback at once. With the stub there is no check.
- **The generations mode's calls** are the refill and the summary, and with `campaign.notes` a note per
  candidate and an assessment per generation. The refill and summary are also tried up to 3 times while they
  fail (an error); then the refill keeps the parent's mutants and the summary is left out (`""`), counted, and
  the campaign goes on. The LangGraph and Academy layers end the campaign at the first failed call, so the
  layers differ on this failure path only. An assessment shows a generation's P results; at P 16 its longest
  prompt (about 28,400 characters) is above the bound, and it changes no result.
- **Request settings** (each null by default, so that every request is as before; `run.py`'s layers refuse them
  set): `llm.thinking` (off: `chat_template_kwargs {enable_thinking: false}` on every call through AUTONMI's
  transports, and no thinking budget; on: the budget as `llm.send_thinking_budget` says), `llm.reply_cap`
  (`max_tokens`: the cap, plus `agents.thinking_budget` unless thinking is off), `llm.temperature`, `llm.top_p`.
  The Generator's JSON schema goes as a plain `response_format` keyword, never in a per-call `extra_body`,
  which would replace the transports' thinking switch. A model run (`--llm openai`) in the asynchronous mode
  needs `llm.thinking` set. By default a request is `run.py`'s: `max_tokens` = thinking budget + reply allowance,
  vLLM's per-request `thinking_token_budget`, no temperature of its own.
- **The stub LLM** (`llm.StubLLM`, a FIXTURE, not a model) is called like a model, so its latency is felt:
  `llm.stub_latency_s` per call (emulated). Its replies depend only on the campaign and the request's last line,
  so every layer agrees. It answers a refill or the Generator's call with parameters drawn in range, and the three
  other calls with the value from numbers alone (the score from numbers, computed from the q and m the
  Evaluator's prompt gives; no note; no guidance). So stub runs do not show reconciling.
- `llm.mode: off` (the asynchronous mode only): no agent calls a model, and each decides on numbers alone.

## The Monitor and the inner-loop trace

**The trace** is added instrumentation, not in IGNOS. Each evaluation's record can carry `trace`, a step-by-step
account of its inner loop (`trace.py`). The values are read, never changed, so results, touches and the
parent's files are the same with or without it. In the hundredk shape it holds 232 numbers:
- what the parent computes and then drops: the summary's per-puff scan rows (puff, the six wall and pedestal
  metrics, iterations, residual, converged) and its four stage timings. `sim.py` wraps the parent's
  `evaluate_final_innerloop`, as it wraps `_clean_pfr`; the parent's files are untouched.
- values recorded in the stand-ins (`standins.py`): for the XGC stage's priming state and each rollout step, the
  relative change and the mean density, electron and ion temperature over the nodes with psi > 0.80 (outside the
  anchored core); for each iteration of each puff's fixed point, the residual and the new state's te_wall,
  ti_wall and q_peak_proxy (the cells `wall_pedestal_metrics` reads); and the hand-off file's normalisation
  ranges.

The field is `{"kind", "synthetic", "complete", "sections", "f32"}`: the values as float32 in base64, with a
one-line header of section shapes (`xgc:3x4 fixed_point:3x15x4 handoff:6 scan:3x10 timings_s:4`), about 1.4 KB.
`trace.unpack` returns the arrays. Every fixed point runs to its cap (the tolerance is 0), so `converged` is
always false; the residual series is the only convergence signal.

The fixture has no inner loop, so its trace is synthetic, of the same shape, and labelled (`"kind":
"synthetic_smooth"` or `"synthetic_hash"`, `"synthetic": true`). It uses none of the fixture's own draws.
- `smooth` (the default) is a smooth function of the normalised parameters with one planted split: candidates on
  either side of the middle of `a1_Te`'s range follow each other until fixed-point iteration
  `fixed_point_cap // 2` of each puff, and then part in te_wall, ti_wall and q_peak_proxy.
- `hash` is random, from a second generator seeded by the parameters' digest.

Knobs `monitor.capture` (default true) and `monitor.fixture_trace` (`smooth` or `hash`). Only the AUTONMI layer
asks for a trace; the Monitors take it out of the result before they report, so `decide.py`, the parent's files
and `agents.jsonl` never see one.

**The Monitor's numbers** (`monitor.py`). Each campaign keeps one statistics object in its population's process,
which both slots read: for every step of the inner loop an exponentially weighted mean and variance, kept apart
for good and bad candidates (in the asynchronous mode it also holds the window of the last P scored results).
For one trace:
- `left_at`: the step where the trace left the typical band (more than 2 pooled spreads from both sides);
- `back_at`: where it came back;
- `m`: from -1 (like low-ranked candidates) to +1 (like high-ranked ones), weighted toward the steps where the
  two sides part;
- `split_at`: where the two sides part.

`m` is 0 until each side holds `campaign.warmup` traces (4). Memory stays constant per campaign. The traces are
labelled at each generation's barrier (the generations mode) or as each result is scored (the asynchronous
mode). AsyncFlow and RHAPSODY keep each evaluation's returned record, trace included, until the population ends:
about 3.6-3.8 KiB per evaluation at 2,000 campaigns (fixture, hundredk, one population; measured), and the run
was about 4.6% slower; `capture: false` removes both.

## The inner loop: `evaluate`, the IGNOS inner loop turned down

`sim.py` and `standins.py` run the parent's own worker (`_evaluate_single_candidate`) and inner loop
(`evaluate_final_innerloop`), unmodified, with stand-ins injected through their `context`. Each evaluation:
- loads checkpoints from disk to the device and builds the models;
- runs graph forwards on the device;
- hands each stage's output to the next, in memory and through a netCDF file;
- runs the fixed-point loop;
- writes the real artifact layout under `gen{g}_candidate{i}/final_innerloop`.

Every call checks its own touches (The touch checklist, below). `--sim fixture` replaces `evaluate` with an
instant, torch-free stand-in for large agent counts. It makes no touches.

Real inner loops run in spawned worker processes (`pool.ResilientProcessPool`), never on the agents' event loop.
With `workers.gpu_per_worker` (the AUTONMI layer, `--sim real`) each worker takes the next number i from one
counter that every population's pool shares, and sees only GPU i mod the number of GPUs the device probe saw
(`pool.on_gpu`, through `CUDA_VISIBLE_DEVICES`; the record's `autonmi.inner_loop_gpus`). It is tested with a
faked GPU count only; real placement is unverified until a run on GPU nodes. Device rule: `cuda` if available,
else `cpu` (ROCm builds answer through `torch.cuda`).

## The agent layers

### AUTONMI (`layer_autonmi.py`)

```sh
python -m fusion_miniapp.layer_autonmi --preset smoke --sim fixture --agents 100 --populations 2
python -m dragon -s src/fusion_miniapp/layer_autonmi.py --runtime dragon --populations 2 ...   # under Dragon; not run by the tests
```

- **What runs the campaigns is AUTONMI's.** The campaigns are split into populations (`--populations`; by
  default one per agent process of `run.py`), each an AsyncFlow node with an engine of its own for its inner
  loops, a fleet for its model calls and Flowgentic's agent host. They share one AUTONMI agent definition, so
  one list of actions and one model-call limit (the largest role's, times 3 tries); fusion checks each role's
  counts (exact in the generations mode, bounds in the asynchronous mode).
- **Layout.** A campaign's agents must stay in one population, and AUTONMI splits its agent list into equal
  contiguous blocks, so the populations must divide the campaigns. By default the layer takes the smallest count
  that does, from `run.py`'s agent processes up to twice that, and otherwise refuses with nearby campaign counts
  that work.
- `agents.start_together` (on in `hundredk`, off in `smoke`): no campaign begins until every agent of its
  population has started, so all are live at once. It needs `agents.max_live: null`. `agents.max_live` counts
  campaigns (3 + `campaign.slots` agents each); a cap below one campaign is refused.
- **When one agent fails**, the others of its campaign end with an error instead of waiting, and the campaign is
  one line with that error, the failing `role` and every role's model calls. The other campaigns go on.
- **Locally** the populations run in one process. Fixture inner loops run on threads that each keep one event
  loop (`layer_autonmi.FixturePool`): RHAPSODY's `asyncio.run` per task would walk every task of the process each
  time, and a run's time grew about with the square of its campaigns.
- **AsyncFlow's engine** finds finished tasks by checking every running one on every pass
  (`workflow_manager.py:1679-1684` at radical.asyncflow `3f18cc8`), which also grows with the square of a run.
  The layer gives each engine a set of running tasks that records completions as they happen
  (`layer_autonmi._RunningSet`; `FUSION_ASYNCFLOW_RUNNING_SHIM=0` turns it off; the record's
  `autonmi.asyncflow_running_shim` says whether it was on). Measured on a laptop, fixture, one population: 10,000
  campaigns in 403 s instead of 2,039 s, the same results. Remove it once AsyncFlow tracks completions itself.
- **CPython's automatic full collections are off while agents run** (`layer_autonmi.bound_full_collections`;
  `FUSION_GC_FULL_AUTO=1` keeps them; the record's `autonmi.gc`). Each full collection scans every live object:
  at 10,000 campaigns four of them stalled the event loop for up to 6.7 s each (measured).
- **Under Dragon** (`--runtime dragon`) each population is a Dragon process, its inner loops run on RHAPSODY's
  native Dragon backend, and the store is RaDEX. A run uses one node: AUTONMI does not place populations on nodes
  yet. Untested in this package's tests.
- **What it writes** is what `run.py` writes, with `agent_layer: autonmi` and an `autonmi` section (AUTONMI's own
  verdict), plus AUTONMI's folders (`launcher/`, `populations/`, `app/`, `otel/`) and `summary.json`, checked by
  workflow-mini-apps' `MiniApp`. So `compare_records` compares it with the other layers' runs.
- **The live line.** About once a second the head process prints to stderr
  `agents live=<n> model_requests_open=<n> queued=<n> decoding_at_servers=<n> done=<n>`: hosted agents started
  and not ended; model requests sent whose reply is not back, and requests waiting for the client's admission
  caps; sequences in the model servers' running batches, summed (`-` with the stub); agents ended.
- **Its environment.** The AUTONMI framework (the package `autonmi`) installed separately, and
  `requirements-autonmi.txt` for the rest (Flowgentic 0.6.0 and the commits the framework runs on). The layer is
  tested on the framework's commit `layer_autonmi.AUTONMI_COMMIT` (`e20861a`); each run records the commit it
  imported in `autonmi.framework_commit` and warns once on stderr when it is another one.

### LangGraph and Academy (`fusion_miniapp.run`)

`--use-academy` picks the layer, once, at invocation. Both give every campaign the same three agents, carry the
same messages and call the same code (`decide.py`), so they give identical results: `results_sha256` is the same.

| | Default (FlowGentic with LangGraph, `layer_langgraph.py`) | `--use-academy` (`layer_academy.py`) |
|---|---|---|
| The three agents | three compiled LangGraph subgraphs in one campaign graph, talking through mailbox keys of its state | three Academy agents per campaign on the process's local exchange, talking through action calls |
| Hosting | FlowGentic's `AgentHost`, one hosted graph run per campaign | Academy's `Manager`, no executor; a campaign launches its agents when it gets a slot |
| The generation's P inner loops | P suspending `evaluate` task tools in one step (one suspension per generation) | the Evaluator's action awaits all P RHAPSODY tasks |
| Model calls | FlowGentic's fleet | a semaphore with the same per-process share |

Both admit at most each agent process's share of `llm.MAX_IN_FLIGHT` (512 per model server) model calls at once,
cap campaigns live at once per process with `agents.max_live`, and print `agents working=<n> waiting=<n>
model_requests=<n> done=<n>` to stderr once a second. They refuse the asynchronous mode and the AUTONMI-only
knobs set (`llm.thinking`, `reply_cap`, `temperature`, `top_p`, `campaign.notes`, `workers.gpu_per_worker`).
With `--llm openai` they read the endpoint's URL from the environment variable the knob file names (default
`OPENAI_BASE_URL`) and its key from `OPENAI_API_KEY`; `llm.timeout_s` and `llm.max_retries` bound each call.

## Install

Python 3.12. From this folder, for the LangGraph and Academy layers:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[stack,academy,dev]"
```

- These two commands download packages; they were not rerun for this version of the README.
- `stack` is FlowGentic with LangGraph and RHAPSODY (FlowGentic from GitHub at a pinned commit, as it is not on
  PyPI); `academy` is academy-py. `ENVIRONMENT.md` lists the exact laptop set.
- **The AUTONMI layer's environment** (the other two layers run in it too): the steps at the top of
  `requirements-autonmi.txt` (a workflow-mini-apps checkout beside this one, then
  `uv venv --python 3.12 .venv-autonmi` and one `uv pip install`), and the AUTONMI framework installed into it.
- torch is not pinned: on a GPU machine install the CUDA or ROCm build first (each batch file in `launch/` names
  one).

## Running

```sh
python -m fusion_miniapp.layer_autonmi [options] [--populations N] [--workers N] [--runtime local|dragon] \
    [--endpoints-file F] [--api-key-file F] [--model-wait S] [--no-vllm-metrics]
python -m fusion_miniapp.run [options] [--use-academy]
```

**Options of both:**
- `--agents N` (campaigns; default the knob file's `agents.count`);
- `--preset smoke|hundredk|ignos_shape`;
- `--knobs FILE`, repeatable, merged in order over the preset (for example a machine file from `knobs/machines/`);
- `--sim real|fixture` (default `real`);
- `--llm stub|openai` (over `llm.mode`); `llm.model` must then be the served model's name (`stub` is refused);
- `--out DIR`: a new or empty folder (the parent's logs append, so a folder that already holds a run is refused).

The AUTONMI layer's `--llm openai` reads the servers from `--endpoints-file` and their key from
`--api-key-file` (The model service, below); `--model-wait` (1,800 s) bounds the wait for them.

**Exit codes:** 0 (`status: pass`: every campaign finished, every real evaluation passed its touch check, the model
check passed, and on the AUTONMI layer AUTONMI's checks passed); 1 (`fail`); 143 (`interrupted`).

**The model check** (`llm_check`, with `--llm openai`): the run fails if the model filled fewer than
`llm.min_model_share` (0.5) of the candidate slots it was asked for (the refills' slots; in the asynchronous mode
the Generator's candidates), or if more than `llm.max_empty_share` (0.1) of its replies were empty. Without it, a
model that never answers would leave every campaign on the parent's mutants, and the run would still pass.

**How a run ends badly:**
- SIGTERM or Ctrl-C (scancel, a Slurm time limit): the campaigns are cancelled through the event loop and a
  record with `status: interrupted` is written, with the campaigns that finished (exit 143). On the AUTONMI layer
  the populations report into the run first (at most AUTONMI's `cancel_grace_s`, 60 s), AUTONMI's check
  `every_population_ran` then fails, and `autonmi.telemetry` gives each population's task counts. At 10,000
  campaigns on a laptop the run exited about 55 s after SIGTERM (measured). The AUTONMI layer's interrupted record
  keeps the model figures of every call made up to the signal, the unfinished campaigns' included, and says so
  (`speed.coverage.model_calls`): each population that has not ended writes its call sums and its roles' model
  counts then (`Script.cut_short`). `counts.by_role`'s other counts come from AUTONMI's rows, which a population
  writes only when it ends.
- An inner-loop worker that dies: its pool rebuilds once and resubmits the lost evaluations. A cold-start child
  exits when its worker dies.

## The knob file

One YAML file with nine sections. A preset and any number of knob files are merged in order over
`knobs/smoke.yaml`, so a file needs only the keys it changes (`hundredk` and `ignos_shape` take the newer knobs
from it). Unknown keys and out-of-range values are refused with the knob's name. The turn-down is the `sim`
section.

| Section | Knobs |
|---|---|
| `sim` | `nodes_per_plane`, `planes`, `rollout_steps`, `puffs`, `fixed_point_cap` (every fixed point runs to it), `model_width`, `checkpoint_mb`, `device` (auto = cuda if available, else cpu), `cold_start_fraction` (share of evaluations run in a fresh single-use process, chosen from the campaign, generation and candidate number, so it repeats), `keep_artifacts`, `work_root` (`$VARS` expanded; an unset one is an error), `threads_per_worker` (auto = the CPUs this process may use ÷ `workers.sim_workers`) |
| `agents` | `count` (campaigns), `first` (the first campaign's index and seed: a run hosts `a<first>` to `a<first+count-1>`), `per_process`, `max_live` (null = all of a process's campaigns), `start_together` (the AUTONMI layer; needs `max_live: null`), `cost_scale` (a number, or a short list: campaign i uses entry i % length; it multiplies `nodes_per_plane` and `fixed_point_cap`), `thinking_budget`, `budget_scales_with_cost` |
| `outer` | `population` (P), `generations` (G), `generator` (llm or mutation), `rng` (shared or per_agent; the asynchronous mode always has per_agent's two streams), `logs` (per_agent or none), `wall_weight` (the parent's 0.7), `noise` (the parent's 0.1), `parent_stdout` (keep or drop the parent's prints) |
| `tools` | `constraints_lookup`, `abstract_lookup`, `latency_ms` |
| `llm` | `mode` (stub, openai, or off: no model calls, the asynchronous mode only), `base_url_env`, `api_key_env`, `model`, `send_thinking_budget`, `timeout_s`, `max_retries`, `min_model_share`, `max_empty_share`, `stub_latency_s` (emulated), `max_in_flight` and `admission_tokens` (the AUTONMI layer's admission caps per model server; 0 tokens: no token cap); the AUTONMI layer's request settings, null by default: `thinking` (off or on; YAML's bare `off` and `on` are read as these), `reply_cap`, `temperature`, `top_p` (above 0, at most 1); and `outage_s` (how long a call waits out a server that went away; 1,800, AUTONMI's) |
| `workers` | `sim_workers` (inner-loop processes in all; each agent process or population gets its share, at least one); `gpu_per_worker` (the AUTONMI layer with `--sim real`; default false) |
| `monitor` | `capture` (the inner-loop trace; default true), `fixture_trace` (`smooth` or `hash`), `weight` (the Monitor's m in a result's score in the asynchronous mode; default 0.25) |
| `campaign` | the AUTONMI layer: `slots` (Monitor agents per campaign; default 2), `mode` (`generations`, the default, or `async`), `warmup` (traces each side holds before m is not 0, and in the asynchronous mode the first results kept whatever their score; default 4), `notes` (the generations mode's notes and assessments; default false) |
| `record` | `extras` (the record toggle, below; default false) |

| Preset | Campaigns | P | G | Inner loops per campaign | Inner loops in all | Notes |
|---|---|---|---|---|---|---|
| `smoke` | 100 | 4 | 2 | 8 | 800 | seconds with `--sim fixture`, a few minutes with real inner loops on a laptop CPU |
| `ignos_shape` | 4 | 16 | 20 | 320 | 1,280 | the parent campaign's shape; larger inner loops |
| `hundredk` | 100,000 | 16 | 2 | 32 | 3.2 million | 10 agent processes of 10,000; `logs: none`, `parent_stdout: drop` |

None is calibrated. With real inner loops, `hundredk` is set by inner-loop throughput (about 0.16 s each on a
laptop CPU). `knobs/machines/` holds the machine files: `sim.device: cuda` for each of the three batch files, and
`stellar_ai_nemotron.yaml` for the StellarAI run path (below).

## What a run writes

- `run_record.json`: `status`; `run` (time, host, Slurm job id, this package's git commit and whether it had
  uncommitted changes, the backend); all knobs; the layer; package versions; the parent commit; a device probe
  taken in an inner-loop worker; `counts`; `llm_service` and `llm_check`; the summed touches and their verdict;
  `speed`; peak memory; `threads`, pool rebuilds, `results_sha256`.
- `agents.jsonl`: one line per campaign: its best result, its generations' summaries (the generations mode),
  candidate sources and refill counts, its evaluations' inner-loop and queue-wait times, and the generations
  mode's refill and summary calls' details (`llm_call_details`: latency, finish reason, tokens, empty flag).
- `campaigns/<id>/` (`outer.logs: per_agent`): the parent's `config.json`, `evolution_log.csv`,
  `evolution_log.jsonl` and `best_result.json`. At 100,000 campaigns that is 400,000 small files, so `hundredk`
  sets `outer.logs: none`.
- stdout: a one-line JSON summary, then one parsable `speed:` line.

**The AUTONMI layer's counts** (`counts`): campaigns (`agents`), hosted agents (`hosted_agents`,
`agents_per_campaign`), and per role (`by_role`): agents, those that ended normally, model calls (the replies; and
the calls asked, tried and fallen back: `model_asks`, `model_tries`, `model_fallbacks`), lookups, messages sent,
received and dropped, suspensions, evaluations, traces, Monitor reports and seconds waiting for messages. In the
asynchronous mode `campaign_mode: async` and `counts.asynchronous`: results scored and kept (in all and after
warm-up), each role's cycles, the Generator's waits for survivors and its proposals by the parameters their
replies filled (0 to 7). The roles' own calls are counted, not listed: each population sums them
(`layer_autonmi.CallSums`) for the record's model figures.

**The record toggle, `record.extras`** (off by default). Off, the record and the lines hold counts only, and the
tokens per second (`speed.model_tokens_per_s`). On, they also hold:
- in each campaign's line: each candidate's Monitor numbers, q, label, note and (the generations mode with
  `campaign.notes`) assessment, as `monitor`; in the asynchronous mode each result's arrival number, q, m, score,
  draw and kept flag, each Coordinator cycle's guidance, and the orders a replay needs, as `asynchronous`; and
  every call's details in `llm_call_details`, each with the `server` that answered it. All are outside
  `results_sha256`;
- in the record: each role's share of time in a model call over the steady window (`speed.model_call_share`), the
  request settings sent (`llm_service.requests`: each distinct body besides the messages, as the transport makes
  it, by role, with its count of requests), the model servers in the order of a call's `server`
  (`llm_service.servers`), and with `--llm openai` a copy of the model servers' setup files without their secrets
  (`<run>/model_service/`, `llm_service.model_service`; The model service, below).

**The `speed` section**, for the goal of 100,000 agents working at once:
- campaigns live at once (maximum, time-weighted mean, seconds with at least 95% of the requested campaigns
  live, ramp-up); the agent phase (first campaign start to last finish), with start-up and shutdown apart;
- evaluations, model calls, generations and campaigns per second; inner loops in flight and executing;
- the mean, 50th and 95th percentiles and the maximum of inner-loop time, queue wait (on the AUTONMI layer from
  when the candidate was put on the queue, so with its wait for a free Monitor slot), generation time and
  model-call latency; event-loop lag;
- `model_tokens_per_s` (on by default): prompt and generated tokens per second over the steady window, 15% to
  85% of the agent phase (`record.STEADY`), from the usage counts of the calls answered in it. With the stub they
  are its estimates, marked `emulated`. Null in `run.py`'s layers, whose call details carry no times;
- `sim_bound_ratio`: the inner loops' own time spread over the workers, against the agent phase.

**The decoding metric** (the AUTONMI layer's `speed`). For each count, the maximum, the time-weighted mean, and
the seconds at or above 50, 90 and 95% of the hosted agents (3 + `campaign.slots` per campaign); no comparison
against the client's admission caps:
- `agents_live`, in all and per role (from AUTONMI's rows), with `all_live_s` (how long every hosted agent was
  live at once) and, with `start_together`, `begin_margin_s` (at least 0 when every campaign began after every
  agent had started);
- `model_requests_in_flight` (client side, exact from each call's `[t_end - service_s, t_end]`), beside AUTONMI's
  own sampled figures, Little's law three ways and the event loop's lag (flagged `high` when its 95th percentile
  is at least 10% of the mean time in flight);
- `model_requests_queued`: each call's wait for admission;
- `decoding_at_servers`: the vLLM servers' `num_requests_running`, summed, with their waiting requests and
  KV-cache use (`vllm/vllm.json`); null with the stub. It includes prompt processing: vLLM has no decode-only
  count. A server's reading counts until its next one, for at most 2.5 sampling intervals;
- `headline`: the servers' running count where there are servers, else the client-side count (labelled emulated
  with the stub);
- `rates_per_s.model_calls_all_agents` counts every agent's calls (every try); `counts.llm_calls` the outer
  loop's (the refills and summaries; in the asynchronous mode every call answered).

An interrupted run's speed numbers cover only the campaigns that finished (`speed.coverage`); on the AUTONMI layer
its model figures cover every call made up to the signal (`speed.coverage.model_calls`).

`python -m fusion_miniapp.compare_records A/run_record.json B/run_record.json` checks two runs (two machines, or
two layers): statuses, counts and non-machine knobs must agree; the campaigns too, exactly by digest or campaign by
campaign within `--rtol`/`--atol`; speed numbers are reported, never failed on. Two asynchronous records are
compared on their counts and invariants, not their digests.

## The model service

What the AUTONMI layer's `--llm openai` needs of the model servers. Tested against `tests/fake_vllm.py` only (a
FIXTURE standing in for a vLLM server); real vLLM is unverified until a run on the machine.
- **Endpoints file** (`--endpoints-file`): one base URL per line, without `/v1` (for example
  `http://nid001234:8000`). Each server must answer `/health` without a key; the run waits up to `--model-wait`
  seconds for all of them, re-reading the file. The key, if the servers have one, is in the file
  `--api-key-file`, never in a knob file or the record.
- **The served model's name is `llm.model`.** Before any campaign, every server's `/v1/models` must list it
  (vLLM's `--served-model-name`); otherwise the run is refused with `run refused: <url> serves [...], not
  llm.model '<name>'` and exits 1. Its `max_model_len` sets the prompt-length check.
- **Chat completions** (`/v1/chat/completions`, OpenAI-compatible), with the request settings above. The thinking
  budget field (`thinking_token_budget`) needs the server started with a reasoning parser; without one, set
  `llm.send_thinking_budget: false` (or `llm.thinking: off`, which sends none). The replies' `usage` token counts
  are recorded.
- **`/metrics`**, reachable from the head process in vLLM's Prometheus format: each server's running and waiting
  requests, KV-cache use and token and request counters are sampled every 2 s (5 s past 50 servers) into
  `vllm/vllm.json`. AUTONMI then checks that the counters never went down and that the fleets' tokens and
  successful requests equal the servers' exactly. These checks fail the run when a server restarts during it,
  when a requeued model job moves the servers, when the client gave up on a request the server still finished,
  or when the servers also serve another run; `--no-vllm-metrics` drops the sampling and those checks. A server
  without vLLM's `/metrics` still serves the run, with one warning and no decoding counts.
- **Admission:** `llm.max_in_flight` requests and `llm.admission_tokens` tokens per server at once, split over
  the populations. They are the client's caps: above what a server can run, the extra requests wait in the
  server's queue. The StellarAI knob file puts them out of reach, so vLLM's `--max-num-seqs` is the only cap.
- **Outages:** a server that goes away is waited out for up to `llm.outage_s` (AUTONMI's `ServerRecovery`), which
  re-reads the endpoints and key files, as after a requeued model job; then the call fails (and is tried again,
  up to 3 tries in all).
- **The servers' setup files**, with `record.extras`: copied into `<run>/model_service/` with every secret left
  out (`layer_autonmi.copy_model_service`).
  - **What:** when the endpoints file's folder holds `setup-files` (`layer_autonmi.SETUP_LIST`, which the
    StellarAI launch script writes), exactly the files it names (one path or glob per line, relative to that
    folder; for several model jobs `model-<k>/vllm-*.log` and so on), at those paths; else every plain file of the
    folder. Logs (`.log`, `.out`, `.err`) as their first 500 lines. And `environment.txt`, this process's variables
    of the prefixes `RASCAL_`, `SLURM_`, `CUDA_`, `VLLM_`, `FLASHINFER_`, `TRITON_` and `HF_HUB_OFFLINE`,
    `TRANSFORMERS_OFFLINE`, `PATH`, `TMPDIR`.
  - **Not copied:** the key file (`api-key`, `--api-key-file`), any file whose name says it holds a secret
    (`secrets.json`, `tokens.txt`, `hf_token`, ...), anything that is not a plain file, a listed path outside the
    folder.
  - **Left out of every copied file:** the key, and the value of each secret-named variable of this process with
    8 characters or more, wherever they appear; any value set by a secret-named name in the file (`HF_TOKEN=...`,
    `"api_key": "..."`, `--api-key ...`) whatever its length; a URL's user and password. A copied file that still
    holds the key or one of those values is removed.
  - **The limit:** a secret under a name that does not say so, in a file whose name does not say so and not in
    this process's environment, cannot be recognised. With `setup-files` such a file is copied only if the list
    names it; without the list, the folder must not hold one.
- **Not built:** per-role models or servers (every role's calls go to one fleet); token streaming; time to first
  token.

## The StellarAI run path

The campaigns against NVIDIA Nemotron 3.5 Lightning (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`) on
vLLM 0.27.1, on StellarAI's 8 x B200 nodes. **Configured and checked on a laptop only; nothing here has run on the
machine.**

**The files:**
- `knobs/machines/stellar_ai_nemotron.yaml`, applied over the `hundredk` preset and `stellar_ai.yaml`
  (`sim.device: cuda`):
  - `agents.count: 20000` for the scale run (100,000 hosted agents); `campaign.mode: async`, `campaign.slots: 2`;
  - `llm.mode: openai`, `llm.model` the full name above;
  - `llm.thinking: off`, `llm.send_thinking_budget: false`, `llm.reply_cap: 4096`, `agents.thinking_budget: 4096`
    (sent only with thinking on);
  - `llm.temperature: 1.0`, `llm.top_p: 0.95` (NVIDIA's recommended sampling for this model);
  - `llm.timeout_s: 86400` (no per-call limit in practice: the longest job is 24 h), `llm.outage_s: 300`;
  - `llm.max_in_flight: 1000000`, `llm.admission_tokens: 0` (the client's caps out of reach);
  - `workers.sim_workers: 80`, `workers.gpu_per_worker: true` (the real inner loop, 10 workers per GPU),
    `sim.work_root: $TMPDIR` (node-local).
- `launch/stellar_ai_model.env`: the model servers' settings for AUTONMI's `jobs/model.sh` (the weights path,
  `RASCAL_MODEL_NAME` = `llm.model`, the tool parser, `RASCAL_VLLM_RESTARTS=3`), the serving flags
  (`RASCAL_VLLM_ARGS`: TRITON attention, an FP8 KV cache, no prefix caching, 92% of GPU memory, a 16-bit Mamba
  state, `--max-num-seqs 4096`, `--max-num-batched-tokens 32768`, asynchronous scheduling, CUDA graphs for batches
  up to 4,096), the variables for running without internet, and the caches on scratch. It holds no secret.
- `launch/stellar_ai_autonmi.sbatch`: the launch script (its header lists every step and variable).
- `launch/probe_model.py`: the check before each run. With the run's own arguments it sends one request of each
  role (built by `decide.py` from fixture values) to every server with the run's settings. It stops the launch on
  a server that does not list `llm.model`, a prompt that does not fit `max_model_len` with the reply cap, an HTTP
  error, a time-out (300 s, its own) or an empty reply, a Traceback or EngineDeadError in a server log, and with
  `--idle` a server that is not idle. It reports each reply's finish reason, whether the role could use it,
  usage and latency. It never prints the key.

**How the launch script gets the servers:**

| `FUSION_MODEL_NODES` | The servers | The campaigns | For |
|---|---|---|---|
| 0 (default) | `jobs/model.sh` in the background on this job's 8 GPUs | this node, `--sim fixture` | the wiring check: one node, servers and campaigns together |
| N | N model jobs through AUTONMI's `jobs/submit model`, one 8-GPU node each | this node, the real inner loop, one GPU per worker under an MPS daemon | two nodes (N = 1), and the scale run (N = 4) |

With `FUSION_KEEP_SERVERS=1` (and `FUSION_MODEL_NODES`), a launch whose servers passed the check leaves them
running at its end: no stop file, nothing cancelled. They end at `FUSION_MODEL_TIME`, or after
`touch <fleet folder>/model-*/stop`. A later launch with `FUSION_MODEL_FLEET=<that fleet folder>` reuses them
instead of starting servers, only if every one shows no request running or waiting; it neither starts nor stops
them, and keeps `fleet/endpoints` current after a requeue. The servers of a launch without `FUSION_KEEP_SERVERS=1`
have stopped by its end, and the check refuses them.

What it does, in order (all of it printed by `FUSION_DRY_RUN=1`):
1. A run folder `$FUSION_SCRATCH/fusion-miniapp/<job name>-<job id>`, its fleet folder `fleet/`, and each model
   job's folder `fleet/model-<k>/`; node-local temporary files (`TMPDIR`) named for the run; the open-files limit
   raised to the hard limit (each open call holds a socket); the campaign venv activated.
2. One key for every server (`fleet/api-key`, owner-only, copied into each model job's folder before it starts,
   since `model.sh` keeps a key it finds). The key is never printed.
3. The servers started, and waited for (`FUSION_MODEL_WAIT`, 3,600 s; it stops early if `model.sh` or a model
   job ends first). Then `fleet/endpoints`, the model jobs' own endpoints in order, rewritten whenever one
   changes (a requeued job; looked at every 10 s), and `fleet/setup-files`, the files a run with `record.extras`
   copies (the endpoints, `gpu.csv`, each `vllm-<i>.log` and each job's output) and nothing else.
4. `launch/probe_model.py`; its failure stops the launch.
5. With the servers on other nodes, an MPS daemon for this node's GPUs (`FUSION_MPS=0`: none).
6. The run, `python -m fusion_miniapp.layer_autonmi --preset hundredk --knobs knobs/machines/stellar_ai.yaml
   --knobs knobs/machines/stellar_ai_nemotron.yaml <the script's arguments> --llm openai --endpoints-file
   fleet/endpoints --api-key-file fleet/api-key --model-wait 3600 --out <run folder>/run`, started directly, so
   that its PID is the Python process's. The script's arguments come after the knob files, so a `--knobs` file
   given there wins; `--agents` sets a smaller run.
7. It stops the run only by SIGTERM to that PID: on Slurm's signal 15 minutes before the time limit
   (`--signal=B:TERM@900`; the run needs about a minute to write its interrupted record), or once the node's
   available memory falls below `FUSION_MEM_FLOOR_GIB` (100). It never signals an `srun` client.
8. At the end, whatever happened: when this launch started the servers (and `FUSION_KEEP_SERVERS=1` did not keep
   them), a stop file in each model job's folder (`model.sh` then stops its servers and the job ends), and a model
   job not yet serving cancelled (not running, or without its endpoints: it would remove its stop file when it
   starts); the MPS daemon quit; the GPUs checked for processes left; the node-local folder removed. It exits with
   the run's status.

**Once, on StellarAI (on the machine; not run):**

```sh
# the campaign venv: this package, the AUTONMI framework and requirements-autonmi.txt, and torch's CUDA build
export FUSION_MINIAPP_DIR=/path/to/fusion_miniapp FUSION_ENV=/path/to/campaign-venv
export FUSION_SCRATCH=/scratch/gpfs/<group>/<you> RASCAL_REPO=/path/to/autonmi
export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=gpu
# the vLLM 0.27.1 environment at $FUSION_SCRATCH/envs/vllm (or set FUSION_VLLM_ENV), and the model's weights
# at $FUSION_SCRATCH/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (or set FUSION_MODEL_PATH)
cp "$RASCAL_REPO/jobs/local.env.example" "$RASCAL_REPO/jobs/local.env"
# in local.env: SLURM_ACCOUNT, SLURM_GPU_PARTITION=gpu, SLURM_GPU_GRES=gpu:8, and as its last line
#   source "$FUSION_MINIAPP_DIR/launch/stellar_ai_model.env"
# (that file sets RASCAL_VENV to $FUSION_ENV and RASCAL_MODEL to the weights, over the example's placeholders)
```

**The runs, in order (on the machine; not run):**

```sh
cd "$FUSION_SCRATCH"
# one node: the servers and 100 campaigns together, fixture inner loop (the wiring check)
sbatch --time=01:00:00 --output="$FUSION_SCRATCH/%x-%j.out" "$FUSION_MINIAPP_DIR/launch/stellar_ai_autonmi.sbatch" \
    --agents 100 --sim fixture
# two nodes: one model node, and the campaigns with the real inner loop under MPS
FUSION_MODEL_NODES=1 FUSION_MODEL_TIME=01:30:00 sbatch --time=01:00:00 --output="$FUSION_SCRATCH/%x-%j.out" \
    "$FUSION_MINIAPP_DIR/launch/stellar_ai_autonmi.sbatch" --agents 300
FUSION_MODEL_NODES=1 FUSION_MODEL_TIME=03:30:00 sbatch --time=03:00:00 --output="$FUSION_SCRATCH/%x-%j.out" \
    "$FUSION_MINIAPP_DIR/launch/stellar_ai_autonmi.sbatch" --agents 5000
# the scale run: four model nodes, 20,000 campaigns (100,000 agents)
FUSION_MODEL_NODES=4 FUSION_MODEL_TIME=05:30:00 sbatch --time=05:00:00 --output="$FUSION_SCRATCH/%x-%j.out" \
    "$FUSION_MINIAPP_DIR/launch/stellar_ai_autonmi.sbatch"
```

- The times are placeholders, to be set from each step's measured campaign length (StellarAI's QOS follows the
  walltime: 30 minutes or less gpu-test, 4 h gpu-short, 24 h gpu-medium). A model job's time is the campaign
  job's plus a margin, so the servers outlive the run.
- The wiring check needs more than 30 minutes: `model.sh` starts its 8 servers one at a time (about 2 minutes each
  in an earlier deployment on StellarAI, so about 16 minutes; inferred for 8), and Slurm's warning comes 15
  minutes before the limit. A signal before the run starts ends the launch with no run (exit 143).
- **Unverified until these runs:** vLLM 0.27.1 starting with these flags (repeated flags taking their last
  value); the JSON schema together with thinking off; real prompt and reply sizes; the model's replies and how
  often each role can use them; GPU placement of the inner-loop workers and the MPS daemon; submitting model jobs
  from inside a job; calls between nodes; `/tmp` being private to each job; memory and file limits at 100,000
  open calls. Also never run: the memory guard (it reads `/proc/meminfo`, which a laptop does not have); the
  check's scan of real vLLM logs (any line with `Traceback` stops the launch, a harmless one too); keeping and
  reusing servers, the endpoints' refresh after a real requeue, and the cancelling of model jobs, under Slurm
  (each is tested on a laptop with fake Slurm tools and fake model jobs).

## Other machines

Three single-GPU batch files run `fusion_miniapp.run` (the LangGraph or Academy layer). **The app's inner loop
needs a GPU there, and a run without one fails.** Each batch file requests one GPU, writes under the machine's
scratch, applies the machine's knob file (`knobs/machines/<machine>.yaml`, `sim.device: cuda`) after the user's
presets and files, runs the device probe first (`launch/probe.py`, which needs the GPU and a finite convolution)
and stops if it fails, and asks Slurm for SIGTERM 120 s before the time limit.
- `perlmutter.sbatch`: shared QOS, 1 A100, 32 CPUs, output under `$PSCRATCH`.
- `stellar_ai.sbatch`: output under `$FUSION_SCRATCH`.
- `lux.sbatch`: output under the member work area, MIOpen caches on node-local storage.

They share `launch/batch.sh` and `launch/run.sh` (which activates `FUSION_ENV`). The account comes from
`SBATCH_ACCOUNT`. For example (on the machine; not run):

```sh
sbatch --output="$PSCRATCH/%x-%j.out" "$FUSION_MINIAPP_DIR/launch/perlmutter.sbatch" --preset smoke
```

## The touch checklist: what "validated" means here

"Validated" means only that each evaluation made the same kinds of touches as the real loop, in the numbers the
knobs imply. Every evaluation counts its touches and checks them (`touches.py`); a run with any failed check exits
non-zero. Per evaluation, with k = `rollout_steps`, p = `puffs`, c = `fixed_point_cap`, b = 1 if the context was
built in this call else 0, s = small forwards seen:

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

The real loop starts a fresh process and loads everything for every evaluation. Here workers stay warm unless
`cold_start_fraction` sends an evaluation to a fresh process (3 checkpoint loads, 4 library imports). The fixture
makes no touches; its evaluations are counted apart and never pass. The parent's result record still says
`"inner_loop": "validated"`, which is the parent's name for its variant; here that word means only the checklist
above. The method label is `miniapp_standin`, and every record has `emulated: true`.

## Files from FusionFM/Demo, and the declared differences

Five files under `src/fusion_miniapp/parent/` are copied unchanged from FusionFM/Demo (branch
`evo-innerloop-upgrade`, commit `9768bcb`): `steps/final_inner_loop.py`, `evolutionary_loop.py`,
`steps/step0/profile_generator.py`, `evo_agents.py` and `config/config.yaml`. `PARENT.lock` holds their SHA-256
hashes, and a test checks them. They are configured only through module attributes and the injected context.
`evo_agents.py` is the reference for `outer.rng: per_agent` in the exact-match test; the layers re-create its
three agents rather than import it. Two of the files (`evolutionary_loop.py` and `final_inner_loop.py`) contain
default paths on a site file system, which only the parent's own command line and context builder use; this
package never calls either. Those paths include a project directory name and the user-directory names of two
other people (see Open items).

**Where a campaign differs from the parent's loop, on purpose:**
- The population evaluator: the parent runs batches of `n_gpus` in a process pool, or `srun`; here each inner
  loop is a RHAPSODY task (on the AUTONMI layer one per Monitor's candidate).
- `outer.generator: llm`: the model's proposals replace mutants, only after the parent's `repopulate` has run, so
  selection and mutation draw the same random numbers. Proposals can still change later generations' survivors,
  so only `mutation` mode is the parent's trajectory.
- The asynchronous mode is not the parent's loop: it applies the parent's ranking and keep-probability to each
  result against the campaign's last P results, as it arrives (above).
- `config.json`: the parent's keys, with this package's values. `runtime_source_sha256` and `git` are computed
  once per agent process, not once per campaign.
- `best_result.json` is written after the last generation, before the summary call (the parent has no summary).
- `outer.parent_stdout: drop` discards the parent's prints.
- Results cross processes as JSON, so NaN arrives as None. The Coordinator puts NaN back for the parent's float
  keys before selection and logging.
- The inner-loop trace (above) is added instrumentation, not in IGNOS. Nothing it records reaches selection in
  the generations mode, or the parent's files.

## Tests

`python -m pytest -q` in the AUTONMI layer's environment (about 9 minutes on a laptop on battery in Low Power
Mode, less on AC power; without the AUTONMI framework its tests are skipped). The model server in the tests is `tests/fake_vllm.py`, a FIXTURE (not a model).
- **The campaigns:** the exact match with the parent (above); the layers agreeing, and each against the
  references; the real `evaluate` on the CPU end to end; cold starts; a dead worker rebuilt.
- **The AUTONMI layer** (`tests/test_layer_autonmi.py`): the agents per campaign (their rows, messages, per-role
  counts, one Monitor report per inner loop, the store at the end); the same results at 1, 2 or 16 Monitor slots,
  in 1, 2 or 5 populations, with caps and workers; the Monitors' numbers and the planted split; the mailbox,
  candidate queue and inboxes; an injected failure of each role ending its campaign alone; SIGTERM, and the model
  figures it keeps (with the stub and the fake server); the record toggle. In the asynchronous mode: the invariants, a replay of recorded orders through the pure functions, each
  role's call counted and its reply reaching where it goes, the request settings against the fake server, a
  server that fails every call (3 tries, each role's fallback, the same campaign as with no model calls), the
  prompt-length check, the Evaluator's limit of results per call, verify() catching each broken invariant.
  Planted fake secrets never reach a copied setup file, with one model job's folder or several.
- **The rule itself** (`tests/test_async.py`): q against the parent's `select_survivors`, survival, warm-up, the
  Generator's profiles, reconciling, one order and one seed giving one campaign, each role's longest prompt within
  the bound.
- **The launchers** (`tests/test_launch.py`): the three batch files against a stub `srun`; the StellarAI knob file
  and env file (also sourced last in a `local.env` made from AUTONMI's example); the StellarAI launch script's
  `bash -n`, its dry run in each of its three ways, a whole launch against two fake model nodes (fake Slurm and
  GPU tools, fake vLLM servers on localhost), SIGTERM reaching the run, the wiring check with a fake `model.sh`
  (late endpoints, the endpoints' refresh, the stop file), launches that stop before the run (a `model.sh` that
  ends, model jobs that never start, a server down, a failure under `set -u`), and servers kept by one launch and
  reused by the next; `launch/probe_model.py` against two fake servers, and each reason it stops a launch.
- `tests/test_decode_metric.py` checks the decoding metric's arithmetic against brute-force recounts.

## What has run, and what has not

Measured on one laptop (Apple silicon, 12 cores, 24 GB). These are functional checks and laptop timings, not
performance claims for a machine. Every number below is measured unless marked otherwise.

- **The AUTONMI layer, 2026-11-24** (hundredk shape, fixture, stub at 1 ms, one population, agents starting
  together, AC power):
  - the generations mode at 500, 2,000 and 10,000 campaigns: 17.5 s, 68.9 s and 359.4 s, 1.18 to 1.22 times the
    earlier shape whose Evaluator ran a generation's 16 inner loops itself (the cost of one Monitor round trip
    per inner loop on the event loop); peak memory 716, 2,419 and 8,872 MiB;
  - the asynchronous mode with no model calls at the same sizes: 21.8 s, 86.6 s and 445.8 s (760 to 795
    evaluations per second), peak memory 621, 2,096 and 8,428 MiB; about half of the results scored after
    warm-up were kept (49.8% at `monitor.weight` 0, 50.0 to 50.7% at 0.25).
  - With the stub at 2 s a call (emulated) and 500 campaigns, about 4.2 of a campaign's 5 agents were in a model
    call over the middle of the run, on battery in Low Power Mode, where the agents' one process saturated a core.
    On AC power the ladder below gives about 4.7 to 4.8 (4.82 in its run, 4.74 in four repeats).
- **The laptop ladder of 2026-11-25** (the final shape, fixture inner loop, one population, AC power; every model
  call to the stub or to fake servers, and in the generations mode, with `campaign.notes` off by default, only the
  refill and the summary call the model; records, figures and the script in `results/laptop_2026-11-25/`):
  - every hosted agent live at once at 2,500, 10,000 and 50,000 agents (500, 2,000 and 10,000 campaigns, the
    asynchronous mode; the generations mode at 500 too);
  - 4.82, 4.81 and 4.79 of a campaign's 5 agents in a model call with the stub at 2 s, 8 s and 40 s a call
    (emulated; one run each, and 4.74 in four repeats at 500), against 4.9 modelled; in the asynchronous mode
    with the stub at 1 ms, 542 to 584 evaluations per second; peak memory 9,169 MiB at 10,000 campaigns;
  - SIGTERM at 10,000 campaigns: exit 143 and a record of the 5,060 campaigns that had finished, without their
    model-call figures (the code then wrote none for an interrupted run; it now keeps them, see "How a run ends
    badly");
  - the StellarAI knob file against two fake servers: every request carried its settings, and both servers
    answered every role; and 12,000 model calls open at once from one process against two fake servers, none
    failed.
- **The StellarAI run path:** its launch script and check ran only on the laptop, against fake Slurm tools, fake
  model jobs and fake vLLM servers (FIXTURES). No real model server, GPU or MPS daemon has run.
- **10,000 campaigns on the LangGraph and Academy layers** (by 2026-11-21, hundredk shape, fixture on 10 threads,
  one agent process): 320,000 evaluations in each layer, all 10,000 campaigns live at once, the same
  `results_sha256`. LangGraph 462 s (694 evaluations per second, 6,142 MiB, about 629 KiB per campaign; the run
  overlapped a Docker image build); Academy, 30,000 Academy agents, 172 s (1,922 per second, 3,553 MiB, about
  364 KiB per campaign).
- **80 fault trials** on macOS (by 2026-11-21), 40 per layer: 40 cold-start runs, all identical; 20 runs with a
  worker killed mid-evaluation, all recovered within 4.9 to 8.0 s; 20 runs with SIGTERM to the whole job during a
  cold evaluation, all with an interrupted record within 0.24 s. None left scratch files behind.
- **1,000 campaigns** in the smoke shape with the real stand-ins on the CPU (2026-11-21, one agent process, 4
  workers): 8,000 real evaluations per layer, all passing their touch checks, the same `results_sha256`;
  LangGraph 77.7 s, Academy 75.4 s. Records and commands in `results/laptop_2026-11-21/`.
- **The reference problem, `ignos_shape`** (4 campaigns x P 16 x G 20, real stand-ins on the CPU): both layers
  pass with 1,280 of 1,280 touch checks, the same `results_sha256`, byte-identical `evolution_log.csv` and
  `best_result.json`. Records in `results/ignos_shape_reference/`, with how to check another machine against them.
- **Linux** (by 2026-11-21; native arm64 container, Debian trixie, Python 3.12.16, CPU torch 2.15.0): the test
  suite of that date (86 passed, 2 skipped) and the same 80 fault trials passed. **Slurm** (a one-node Slurm 24.11
  in that container, `perlmutter.sbatch` unchanged, with two labelled test substitutions: a `module` command that
  knows only `load conda`, and a wrapper that drops only the probe's GPU requirement): a normal job wrote only
  under the scratch path, and a job with a 3-minute limit got Slurm's early TERM and wrote an interrupted record
  (exit 143).

**Not run:** CUDA and ROCm (no such GPU here); Perlmutter, StellarAI and Lux; Dragon and RaDEX from this
package's tests; vLLM or any real model server; FlowGentic's MCP tool route. The 2026-11-19 ladder in
`results/laptop_2026-11-19_final/` predates the outer loop.

## Open items

- **Several nodes for the campaigns' own process.** The scale run puts 20,000 campaigns (100,000 agents) in one
  process on one node. Whether one process can drive them, or the populations must be placed on several nodes
  (Dragon), is decided by a load check on the machine.
- **Dragon and RaDEX from `run.py`.** The AUTONMI layer runs under Dragon with RaDEX through its own entry point;
  the other two layers take them only when passed in by hand (`run.Services.open(backend=, store=, model=)`,
  `layer_langgraph.run_agents(store=, replicas=)`).
- **Memory per live campaign** at P 16 was about 629 KiB (LangGraph) and 364 KiB (Academy) of host memory per
  agent process with fixture campaigns (measured); on the AUTONMI layer the 8,872 MiB peak at 10,000 campaigns
  is about 0.9 MiB per campaign (inferred: the peak divided by the campaigns, the interpreter included).
  Not measured: a full 100,000-agent run, real GPU inner loops at scale, the model servers' memory.
- **The five parent files** contain site paths with a project directory name and two other people's
  user-directory names. They must stay byte-identical here; decide before publishing whether they ship as they
  are.
- **Publishing:** a lockfile, a release tag, and the AUTONMI framework's own release.
