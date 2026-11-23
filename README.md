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
mini-app campaign in both agent layers (generator: mutation), and, where the AUTONMI framework is
installed, in the AUTONMI layer through its entry point (one campaign, `agents.first` = the seed). It
does this for P 4 and G 3:
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

### The inner-loop trace: added instrumentation, not in IGNOS

For the Monitor (planned: plan `separate agents and Monitor on AUTONMI`, package P6), each evaluation's record
can carry `trace`, a step-by-step account of its inner loop (`trace.py`). IGNOS records none of it; the
values are read, never changed, so results, touches and the parent's files are the same with or without it.
In the `hundredk` shape it holds 232 numbers:
- what the parent computes and then drops: the summary's per-puff scan rows (puff, the six wall and pedestal
  metrics, iterations, residual, converged) and its four stage timings. `sim.py` wraps the parent's
  `evaluate_final_innerloop`, as it wraps `_clean_pfr`; the parent's files are untouched.
- values recorded in the stand-ins (`standins.py`): for the XGC stage's priming state and each rollout step,
  the relative change and the mean density, electron and ion temperature over the nodes with psi > 0.80
  (outside the anchored core); for each iteration of each puff's fixed point, the residual and the new
  state's te_wall, ti_wall and q_peak_proxy (the cells `wall_pedestal_metrics` reads); and the hand-off
  file's normalisation ranges.

The field is `{"kind", "synthetic", "complete", "sections", "f32"}`: the values as float32 in base64, with a
one-line header of section shapes (`xgc:3x4 fixed_point:3x15x4 handoff:6 scan:3x10 timings_s:4`), about
1.4 KB. `trace.unpack` returns the arrays. Each puff's series ends at that puff's scan row, and the selected
puff's at the record's own values. Every fixed point runs to its cap (the tolerance is 0), so `converged` is
always false; the residual series is the only convergence signal.

The fixture has no inner loop, so its trace is synthetic, of the same shape, and labelled
(`"kind": "synthetic_smooth"` or `"synthetic_hash"`, `"synthetic": true`). It uses none of the fixture's own
draws: every other field of the fixture's record is byte-identical with or without it.
- `smooth` (the default) is a smooth function of the normalised parameters with one planted split.
  Candidates on either side of the middle of `a1_Te`'s range follow each other until fixed-point iteration
  `fixed_point_cap // 2` of each puff, and then part in te_wall, ti_wall and q_peak_proxy. Its series end at
  their own values, not the record's (the record's come from a hash of the parameters).
- `hash` is random, from a second generator seeded by the parameters' digest. The selected puff ends at the
  record's te_wall, ti_wall and q_peak_proxy.

Knobs `monitor.capture` (default true) and `monitor.fixture_trace` (`smooth` or `hash`). They are not under
`sim`, so no campaign's `config.json` changes. Only the AUTONMI layer asks for a trace. Its Evaluator takes
each trace out of the results before it sends them to the Coordinator, so `decide.py` (whose `collect`
refuses a result that holds one), the parent's files and `agents.jsonl` never see one; it indexes them for
the Monitor until its campaign ends (`counts.by_role.evaluator.traces` counts them). The memory stays held
until the population ends: AsyncFlow and RHAPSODY keep each evaluation's returned record, and RHAPSODY a
text copy of it, until then. Measured at 2,000 campaigns (fixture, hundredk, one population) that is about
3.6-3.8 KiB per evaluation, and the run was about 4.6% slower; `capture: false` removes both. The LangGraph
and Academy layers never ask for a trace, so in their run records the `monitor` knobs are recorded but do
nothing.

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

## The third layer: AUTONMI

`layer_autonmi.py` runs the same campaigns on AUTONMI's framework, as a workflow mini-app with an entry
point of its own:

```sh
python -m fusion_miniapp.layer_autonmi --preset smoke --sim fixture --agents 100 --populations 2
python -m dragon -s src/fusion_miniapp/layer_autonmi.py --runtime dragon --populations 2 ...   # under Dragon
```

- **The same campaigns.** Each campaign is `decide.py`'s three roles with the same messages, in the same
  order. Its results are the other layers' results: `results_sha256` is the same, and so are the parent's
  per-campaign files.
- **Three agents per campaign.** On this layer each role is a hosted AUTONMI agent of its own:
  `a<i>/coordinator`, `a<i>/generator` and `a<i>/evaluator`, one after another in the agent list. Each has
  its own state, start and end, row in AUTONMI's records (`app/agents-p*.jsonl`), model calls (under its own
  id, so its own routing and call records) and lookups. They share one AUTONMI agent definition, so they
  share its list of actions and one model-call limit (the largest role's); fusion checks each role's exact
  counts. Model calls are whole replies: no token streaming, and no tools chosen by a model.
  - They talk only through messages, JSON copies kept in a mailbox per population
    (`layer_autonmi.Mailbox`) under the key `a<i>/<to>/<kind>/<g>`. Each generation: Coordinator ->
    Generator `make/g`; Generator -> Coordinator `population/g`; Coordinator -> Evaluator `evaluate/g`;
    Evaluator -> Coordinator `results/g`. After the last one the Coordinator makes the summary call, sends
    `done` to the other two, reports and ends; the others end on `done`, so all three stay live until their
    campaign ends. That is 4G + 2 messages per campaign. A waiting agent awaits its message inside its step,
    counted as waiting on a tool, and is never polled; only the Evaluator's wait for its evaluations is
    AUTONMI's real suspension (its state goes to the store). The other waits keep the paused step in memory.
  - The Coordinator makes the constraints lookup and sends it in `make/0`; the Generator makes the abstract
    lookup itself (`decide.LOOKUP_ROLE`). Every id seen outside the agents stays the campaign's, `a<i>`: the
    stub's key, the evaluations' `agent_id` (cold starts, work folders), the lookups, the store key
    `campaign/a<i>`, the campaign's folder and its `run_tag`.
  - `agents.start_together` (on in `hundredk`, off in `smoke`): no Coordinator begins until every agent of
    its population has started, so all are live at once. It needs `agents.max_live: null`; with a cap the
    run is refused.
  - `agents.max_live` still counts campaigns: AUTONMI's cap is three agents per campaign, and a cap of one
    campaign runs (a cap below that is refused). The start-up stagger (`--stagger-batch`, campaigns) is
    likewise three agents per campaign.
  - A campaign's agents must stay in one population, and AUTONMI splits its agent list into equal
    contiguous blocks, so the populations must divide the campaigns. By default the layer takes the
    smallest count that does, from `run.py`'s agent processes up to twice that, and otherwise refuses with
    nearby campaign counts that work. A `--populations` that does not divide is refused.
  - When one of a campaign's agents fails, the other two end with an error instead of waiting, and the
    campaign is one line with that error, the failing `role` and every role's model calls. The other
    campaigns go on, as before.
  - The record counts campaigns (`counts.agents`, as in the other layers) and the hosted agents
    (`counts.hosted_agents`, `agents_per_campaign`), and per role (`counts.by_role`): agents, those that
    ended normally, model calls, lookups, messages sent, received and dropped (a failed campaign's messages
    not yet received), suspensions and each agent's seconds waiting for messages. `counts.llm_calls` keeps
    counting the outer loop's calls. AUTONMI's checks include each role's exact counts, every message
    received once with none left, the lookups against AUTONMI's own count, and the store at the end
    (`store_holds_only_the_memories`: no agent's saved state, each reported campaign's memory under
    `campaign/a<i>`).
- **A generation is one fan-out.** The Evaluator's P `evaluate` calls are submitted at once as
  suspending task tools. The Evaluator is saved and dropped until all P end, as the campaign is in the
  LangGraph layer.
- **What runs the campaigns is AUTONMI's.** The campaigns are split into populations
  (`--populations`; by default one per agent process of `run.py`), each an AsyncFlow node with an engine
  of its own for its simulations.
  - Locally the populations run in one process. Real simulations run in spawned workers
    (`pool.ResilientProcessPool`, `--workers` per population; by default its share of
    `workers.sim_workers`). Fixture simulations run on threads that each keep one event loop
    (`layer_autonmi.FixturePool`): RHAPSODY's `asyncio.run` per task would walk every task of the
    process each time, and a run's time grew about with the square of its campaigns.
  - AsyncFlow's engine loop finds finished tasks by checking every running one on every pass
    (`workflow_manager.py:1679-1684` at radical.asyncflow `3f18cc8`), which also grows with the square of a run.
    The layer gives each engine a set of running tasks that records completions as they happen
    (`layer_autonmi._RunningSet`; `FUSION_ASYNCFLOW_RUNNING_SHIM=0` turns it off; the record's
    `autonmi.asyncflow_running_shim` says whether it was on). Measured on a laptop, fixture simulations, one
    population: 10,000 campaigns in 403 s instead of 2,039 s, the same results, throughput flat from 2,000 to
    10,000 campaigns. 100,000 is not measured, and it has not run under Dragon yet. Remove it once AsyncFlow tracks
    completions itself.
  - Under Dragon (`--runtime dragon`), each population is a Dragon process, its simulations run on
    RHAPSODY's native Dragon backend, and the store is RaDEX. A run uses one node: AUTONMI does not
    place populations on nodes yet, and refuses more.
- **Model calls** go through Flowgentic's fleet, with the request `llm.py` sends: the thinking budget per
  call, `max_tokens` as budget plus reply, no temperature of its own. The Generator awaits its call
  inside its step, as `decide.generator_step` is written. `--llm openai` reads the model job's servers
  from `--endpoints-file` (base URLs without `/v1`, each answering `/health`) and its key from
  `--api-key-file`, not from `llm.base_url_env`. A server that goes away is waited out for up to 30
  minutes, instead of `llm.max_retries` retries.
- **What it writes** is what `run.py` writes, with `agent_layer: autonmi`: `run_record.json` (its
  status, counts, `llm_check`, touch check, `speed` and `results_sha256`, by `record.write`),
  `agents.jsonl` and `campaigns/`. So `compare_records` compares it with the other layers' runs. The
  record's `autonmi` section gives AUTONMI's own verdict; its run folders (`launcher/`, `populations/`,
  `app/`, `otel/`) and `summary.json` hold the run's telemetry, checked by workflow-mini-apps' `MiniApp`.
- **How a run ends badly,** as with `run.py`: SIGTERM or Ctrl-C cancels the campaigns through the event
  loop and writes an `interrupted` record with those that finished (exit 143); a failed campaign is a
  line with its error. Locally, the populations shut down and report into the run before it is verified
  (at most AUTONMI's `cancel_grace_s`, 60 s; untested under Dragon, where nothing waits for them).
  AUTONMI's check `every_population_ran` then fails with the populations the signal cancelled, and the
  record's `autonmi.telemetry` gives AUTONMI's `telemetry_complete` verdict with each population's task
  counts: those the signal cut off are `abandoned`, which does not by itself make the telemetry
  incomplete. At 10,000 campaigns on a laptop the run exited about 55 s after SIGTERM (measured; larger
  runs not yet).
- **The live line.** About once a second the head process prints to stderr
  `agents live=<n> model_requests_open=<n> queued=<n> decoding_at_servers=<n> done=<n>`: hosted agents
  started and not ended; model requests admitted and sent whose reply is not back, and requests waiting for
  the client's admission caps (both client side, Flowgentic's agent states); sequences in the model
  servers' running batches, summed (the vLLM sampler's latest reading of each server that answered within
  2.5 sampling intervals; `-` with the stub); agents ended. The record keeps the same rows
  (`speed.live.series`). The line waits a second after the last one, so an event-loop stall stretches a
  gap (up to about 2.2 s at 2,000 campaigns on a laptop, measured). Under Dragon only the servers' count is seen in
  the head process (the populations run elsewhere; untested).
- **The decoding metric** (`speed` in the record). For each count: the maximum, the time-weighted mean,
  and the seconds at or above 50, 90 and 95% of each reference: the hosted agents (three per campaign) and
  the admission capacity (`llm.max_in_flight` x servers, split over the populations).
  - `agents_live`: from AUTONMI's rows (`app/agents-p*.jsonl`, each agent's `t_start` to `t_end`), in all
    and per role (`by_role`, against one agent per campaign), with `all_live_s`: how long every hosted agent
    was live at once (from the last start to the first end). `start_together` gives each Coordinator's wait
    for every agent to start, and `begin_margin_s`: the earliest Coordinator's start of its campaign
    (`t_begin` in its row) minus the last start of its population's agents, the smallest over populations;
    at least 0 when every campaign began after every agent had started, as the knob promises. With the
    default stagger every agent is soon live at once with or without the knob (`all_live_s` > 0 at 20 and
    at 2,000 campaigns), so `begin_margin_s` is what shows the knob at work.
  - `model_requests_in_flight` (client side): exact, from each call's `[t_end - service_s, t_end]`
    (kept in `llm_call_details` with the call's `role`, `wait_s` and `recovery_s`); beside it AUTONMI's
    own figures (`autonmi_sampled`: `summary.json`'s `agent_states_over_time.model_request`, sampled about
    every 10 s, and `agent_seconds_by_state.model_request` per second of the agent phase), Little's law three
    ways (each per second of the agent phase; AUTONMI's agent-seconds per second of wall time beside them),
    and the event loop's lag, flagged `high` when its 95th percentile is at least 10% of the mean time in
    flight (the lag lands inside the call intervals).
  - `model_requests_queued`: each call's wait for admission (Flowgentic's MODEL_QUEUED).
  - `decoding_at_servers`: the vLLM servers' `num_requests_running`, summed, with their waiting requests
    and KV-cache use (`vllm/vllm.json`); null with the stub. It includes prompt processing: vLLM has no
    decode-only count. A server's reading counts until its next one, for at most 2.5 sampling intervals: a
    server that stopped answering `/metrics` counts 0 from then on, and its seconds without a reading are in
    `stale_s`. A gauge a server never gave (an older vLLM's name) is null, named in `gauges_missing`.
  - `headline`: the servers' running count where there are servers, else the client-side count. With the
    stub it is labelled emulated (`llm.stub_latency_s` per call), never decoding. With servers it names who
    their `/v1/models` say serve the model (`served_by`, from `llm_service.served_models`): `vllm` for
    vLLM, `fusion-test-fixture` for the tests' fake server.
  - `rates_per_s.model_calls_all_agents` counts every agent's calls; `counts.llm_calls` and
    `rates_per_s.model_calls` keep counting the outer loop's (today the same calls: the Generator's refills
    and the Coordinator's summary, each call's `role` in `llm_call_details`).
  Measured on a laptop (2026-11-23, hundredk shape, fixture, one population, emulated): at 2,000 campaigns
  with 2 s per call, 160.78 requests in flight on average by AUTONMI's agent-seconds, 160.78 by fusion's
  calls per second x time in flight, and 160.52 by calls per second x 2 s (each per second of the agent
  phase; AUTONMI's per second of wall time, 154.1). With 30 s per call the client's count reached the 512
  cap: 94 s at it, in the summary wave, and 210 s at or above 95% of it. In the refill wave (about 10 s to
  124 s) the 2,000,000-token cap held it near 498. Up to 1,502 requests were queued.
- **Its environment.** It needs the AUTONMI framework (the package `autonmi`) installed: it is not
  public yet, and the RADICAL team can provide access. `requirements-autonmi.txt` installs the rest:
  Flowgentic 0.6.0 and the commits the framework runs on. The LangGraph and Academy layers also run in
  that environment. The layer is tested on the framework's commit `layer_autonmi.AUTONMI_COMMIT`
  (`e20861a`, origin/MTU-188-framework). Each run records the commit it imported (git in an editable
  clone, else the installer's record) in `autonmi.framework_commit`, and warns once on stderr when it is
  another commit, unknown, or has changed files.
- **Tests.** `tests/test_layer_autonmi.py` compares it with the LangGraph layer by `compare_records`:
  fixture campaigns in two knob cases, the real stand-in on CPU, and a model server stand-in (the same
  requests, and a failed campaign). It also checks SIGTERM and the shape workflow-mini-apps requires, the
  live line and the client's counts on the stub, and the model-service checks below. It checks the three
  agents per campaign (their rows, messages, exact per-role counts and the store at the end), the same
  results in 1, 2 or 5 populations, with a cap of 1 or 7 campaigns or none, 1 or 4 workers, and with a slow
  start-up with and without starting together (with it, every campaign begins after its population's last
  agent starts), the refusals (populations that do not divide, a cap below one campaign, starting together
  with a cap), the mailbox's own contract, that the ids seen outside the agents are the campaign's, and that
  an injected Generator or Evaluator failure (`FUSION_AUTONMI_FAIL=<agent id>:<g>`, tests only) ends its
  campaign alone, as one error line with every role's model calls. The model server is
  `tests/fake_vllm.py`, a FIXTURE (not a model) standing in for a vLLM server; `tests/test_decode_metric.py`
  checks the metric's arithmetic against brute-force recounts.
  Both layers' runs are also compared with references recorded from the LangGraph layer
  (`tests/references/*.json`: results_sha256, each campaign's hash, counts, parent files and model
  requests, with the commit, command and date that made them). `python tests/references/record_references.py`
  records them again from a commit (default HEAD, exported with `git archive`). The parent oracle
  (above) runs through it too. Without the AUTONMI framework these are skipped.

## The model service

What the AUTONMI layer's `--llm openai` needs of the model servers. Tested against `tests/fake_vllm.py`
only (a FIXTURE); real vLLM is unverified until a machine.
- **Endpoints file** (`--endpoints-file`): one base URL per line, without `/v1` (for example
  `http://nid001234:8000`). Each server must answer `/health` without a key; the run waits up to
  `--model-wait` seconds (default 1,800) for all of them, re-reading the file. The key, if the servers have
  one, is in the file `--api-key-file`, never in a knob file or the record.
- **The served model's name is `llm.model`.** Before any campaign, every server's `/v1/models` must list it
  (vLLM's `--served-model-name`); otherwise the run is refused with `run refused: <url> serves [...], not
  llm.model '<name>'` and exits 1.
- **Chat completions** (`/v1/chat/completions`, OpenAI-compatible): each request has the messages, the
  model, `max_tokens` = thinking budget + reply allowance, no temperature (the server's default), and
  vLLM's per-request `thinking_token_budget`. That field needs the server started with a reasoning parser
  (`vllm serve --reasoning-parser`, for example `qwen3`); without one, set `llm.send_thinking_budget:
  false`. The replies' `usage` token counts are recorded.
- **`/metrics` reachable from the head process**, in vLLM's Prometheus format. The layer samples each
  server's `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` and token and
  request counters every 2 s (every 5 s past 50 servers), with AUTONMI's `VllmSampler`, into
  `vllm/vllm.json`. AUTONMI then checks that the servers' counters never went down (`verify_vllm`) and that
  the fleets' tokens and successful requests equal the servers' exactly (`tokens_match_the_servers`). These
  checks fail the run (status fail, exit 1) when a server restarts during it (its counters reset), when a
  requeued model job moves the servers (the sampler keeps reading the old addresses), when the client gave
  up on a request the server still finished (a timeout), or when the servers also serve another run. Where
  any of these is expected, give `--no-vllm-metrics`: no sampling, no decoding counts, none of those checks.
  A server without vLLM's `/metrics` still serves the run: one warning, and `decoding_at_servers` is null.
  The first reading of each server's `/metrics` gets three tries, 2 s apart; if any server fails them, no
  server is sampled (AUTONMI compares the servers' counters with all of the fleets' calls).
- **Admission knobs:** `llm.max_in_flight` (512) requests and `llm.admission_tokens` (2,000,000; 0 = no token
  cap) per server, at once, split over the populations. They are the client's caps: above what a server can
  run, they raise the client count, not decoding, and the extra requests wait in the server's queue
  (`num_requests_waiting`). Whichever cap is reached first binds. At the hundredk shape a refill reserves
  about 4,000 tokens (inferred: 2,000,000 / 498, the count the token cap held at 30 s per call), so the
  token cap binds near 500 refills per server, below 512. Raising `llm.max_in_flight` alone does not raise
  refill concurrency; raise `llm.admission_tokens` with it. AUTONMI's own defaults are 96 requests and
  235,000 tokens per server (`PopulationConfig`). Measure them on each machine. `run.py`'s layers keep
  `llm.MAX_IN_FLIGHT`.
- **Outages:** a server that goes away is waited out for up to 30 minutes (AUTONMI's `ServerRecovery`),
  which re-reads the endpoints and key files, as after a requeued model job. The campaigns survive it,
  but with `/metrics` sampled the run then fails AUTONMI's checks against the servers' counters (above);
  give `--no-vllm-metrics` where outages are expected.
- **Reference launcher:** AUTONMI's `jobs/model.sh` starts one `vllm serve` per GPU (or per `RASCAL_TP`
  GPUs), served as `RASCAL_MODEL_NAME`, with the reasoning parser `RASCAL_REASONING`, and writes
  `endpoints` and `api-key` into its folder once every server answers `/health`; it restarts a server that
  dies. Wiring it into `launch/` is not done yet.
- **Not built (asks):** per-role models or servers (every role's calls go to one fleet); token streaming;
  time to first token (the record has each call's whole time only).

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

One YAML file with seven sections. A preset and any number of knob files are merged in order over
`knobs/smoke.yaml`, so a file needs only the keys it changes. The turn-down is the `sim` section.

| Section | Knobs |
|---|---|
| `sim` | `nodes_per_plane`, `planes`, `rollout_steps`, `puffs`, `fixed_point_cap` (every fixed point runs to it), `model_width`, `checkpoint_mb`, `device` (auto = cuda if available, else cpu), `cold_start_fraction` (share of evaluations run in a fresh single-use process, chosen from the campaign, generation and candidate number, so it repeats), `keep_artifacts`, `work_root` (`$VARS` expanded; an unset one is an error), `threads_per_worker` (auto = the CPUs this process may use ÷ `workers.sim_workers`; sets OMP, OpenBLAS, MKL and torch threads) |
| `agents` | `count` (campaigns), `first` (the first campaign's index and seed: a run hosts campaigns `a<first>` to `a<first+count-1>`, as they are in a run from 0; default 0), `per_process`, `max_live` (null = all of a process's campaigns), `start_together` (the AUTONMI layer: generation 0 waits until every hosted agent of the population has started; needs `max_live: null`), `cost_scale` (a number, or a short list: campaign i uses entry i % length; it multiplies `nodes_per_plane` and `fixed_point_cap`), `thinking_budget`, `budget_scales_with_cost` |
| `outer` | `population` (P), `generations` (G), `generator` (llm or mutation), `rng` (shared or per_agent), `logs` (per_agent or none), `wall_weight` (the parent's 0.7), `noise` (the parent's 0.1), `parent_stdout` (keep or drop the parent's per-generation prints) |
| `tools` | `constraints_lookup`, `abstract_lookup`, `latency_ms` |
| `llm` | `mode` (stub or openai), `base_url_env`, `api_key_env`, `model`, `send_thinking_budget`, `timeout_s`, `max_retries`, `min_model_share`, `max_empty_share`, `stub_latency_s` (the stub's seconds per call: emulated), `max_in_flight` and `admission_tokens` (the AUTONMI layer's admission caps per model server) |
| `workers` | `sim_workers` (simulation processes in all; each agent process gets `sim_workers // agent processes`, at least one) |
| `monitor` | `capture` (each evaluation's record carries the inner-loop trace; the AUTONMI layer only; default true), `fixture_trace` (`smooth` or `hash`: the fixture's synthetic trace). Added instrumentation, not in IGNOS; no result reads it |

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
- The inner-loop trace (above) is added instrumentation, not in IGNOS: `evaluate_final_innerloop` is
  wrapped to keep its summary's scan rows and timings, and the stand-ins record per-step values. Nothing it
  records reaches selection or the parent's files.

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
  The AUTONMI layer (above) already runs under Dragon with RaDEX, through its own entry point.
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
