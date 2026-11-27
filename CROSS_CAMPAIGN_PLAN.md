# Cross-campaign plan: a live ledger, a progress plane and one entry point for a run of many campaigns

**Nothing in this document is implemented.** It is a plan. The only trace of it in the code is short comments,
each with a tag `XC-...`, at the places where the plan would attach (section 12). Every component, knob, file,
message, record field and test named below is planned unless the text says it exists.

Code references are to the `dev` branch at `000300c`. Paths without a folder are under `src/fusion_miniapp/`.
Each fact is labelled:
- **code**: read in the source at the line given;
- **laptop**: measured on a laptop on 2026-11-26, outside the test suite;
- **StellarAI**: read from the run records of 2026-11-25 (see "The StellarAI run path" and "What has run, and
  what has not" in the README);
- **configured**: set in a knob or launch file, not run;
- **inferred**: arithmetic or reasoning, not run.

## 1. Purpose, and the plan in short

**Purpose.** The mini-app reproduces a plausible compute and communication pattern of a 100,000-agent run. The
cross-campaign plane exists to exercise communication between campaigns at that scale. Its ledger needs no
scientific meaning or validity: whether a hit is a true repeat does not matter. What matters is that:
- Generators really consult and claim against a shared structure during the run;
- Coordinators really report to one entry point and are steered from it;
- the volumes, the fan-in and the latencies are realistic, and they are recorded.

Saving real work would need validity. That belongs to the real application (IGNOS), not to the mini-app, and this
plan does not design it. No phase here waits on, or is judged by, work saved.

In short:
1. **Two planes and one entry point per run, all off by default.** No campaign ever waits on either plane. If a
   plane or the entry point is slow, broken or gone, every campaign goes on as today.
2. **The Coordinators' side (P0).** Each Coordinator's memory already passes `Script.remember` at every cycle in
   both modes. A copy of a few of its fields is the digest. It goes into a progress table per population. The
   plane of each process is the aggregator: a tick replaces one small file per population in the run's plane
   folder.
3. **One entry point per run, a real service in a process of its own** (`python -m fusion_miniapp.entry_point
   serve`). It is never on a population's event loop. It reads every progress file, holds the roll-ups, answers an
   end user's queries over HTTP on `127.0.0.1`, and is the only writer of directives. The launch script starts it,
   or the head when no launch script did.
4. **The Generators' side is a live ledger (P1).** At every take a Generator consults the ledger for each candidate
   and claims it. Every finished evaluation publishes a result row. Reads come from a cache in the process. A
   background thread synchronises the caches across processes through one append-only log per process, so real
   traffic crosses processes while no campaign waits on it.
5. **The key is coarse and campaign-independent on purpose:** the 7 mutable parameters on a grid whose step is a
   knob (`grid_step`). The six frozen shape values are left out, so that consults hit at a plausible rate that the
   knob tunes (section 3.1).
6. **A hit does nothing to the campaign's computation by default.** It is counted and reported. Serving a hit at
   the Generator would break the campaign's own checks (section 3.2). Acting on hits, by reuse inside the worker,
   is an optional later knob (P2) with no gate.
7. **Steering (P3) is guidance notes only.** A note is one line before the last line of the Coordinator's prompt
   (async) or of the refill prompt (generations), read at the Coordinator's next cycle. Stop stays the launch
   script's SIGTERM.
8. **Files carry everything between processes**, each with one writer: progress files and directives in the
   run's plane folder, ledger logs on node-local disk (the shared run folder when a run spans nodes, P4).
9. **Every hook is guarded.** An exception inside the plane turns the plane off in that process, is counted, and
   never reaches an agent. A failed file write is only counted.
10. **Each phase's exit check is about the communication pattern:** messages per second, fan-in, latency and
    loss, recorded in the run record and in the entry point's summary, and the campaigns unchanged.

## 2. The shape, and why

### 2.1 What ran that decides it

| Fact | Label | Consequence for the plan |
|---|---|---|
| One local-runtime process did not drive 10,000 agents against 8 servers (event-loop lag p95 1,545 ms, no campaign finished in 6 minutes). 8 launches side by side ran 8 x 800 campaigns to the end, lag p95 6-7 ms (README, "Open items" and "The StellarAI run path") | StellarAI | Nothing may add a remote round trip to a campaign's event loop. A consult is a read of the process's own cache; every exchange between processes runs in the plane's thread |
| The inner-loop workers were busy 47% of the agent phase (`sim_bound_ratio` 0.47 in each of the 8 launches) | StellarAI | Workers were busy 47% on average and saturated at peaks (`sims_executing.max` 8 of 8, `queue_wait_s.p95` 61.4 s in g0). Acting on hits (P2) would change the compute pattern (fewer inner loops); what that does to the pace is not known, and P2's exit check reports it |
| 106.6 model calls and 58,400 prompt tokens per campaign (548 per call); 89% of all tokens are prompt tokens | StellarAI (8 x 800 records) | Text added to prompts is the cost that lands on the pace-setting servers |
| 26.9 Coordinator cycles per campaign (172,134 for 6,400) | StellarAI | About 0.27 digests per model call: the servers bound the control plane's rate |
| Each seed draws 16 individual initial profiles. Each profile carries a six-value shape signature inherited by its descendants. All 320,000 initial signatures for campaigns a0 to a19999 are distinct | code, laptop | No two campaigns evaluate the same full profile. The ledger's key leaves the six shape values out on purpose (section 2.5). It is a fact about the experiment, not a gate |
| An async Coordinator sleeps in `take_all(me, "scored")` until a result is scored (`layer_autonmi.py:1343`) | code | A stuck campaign has no cycle and emits nothing. Liveness must come from the process, not the cycle |
| Nothing reads mid-campaign progress during the run, and nothing outside one launch can: the store holds each Coordinator's last memory, per launch (`layer_autonmi.py:1596`), and campaign lines are written only at a campaign's end (`:1598-1607`, `:1640-1642`) | code | P0 is new information across launches, not a copy of something that exists |

### 2.2 The two planes

```
per campaign (5 hosted agents)        per process (the plane: the aggregator)       across processes and the run
Coordinator: _begin, remember,   -->  progress table per population            -->  xc/progress-a<first>-<count>-
  report, on_done (XC-C1..C4)         tick on the loop; encode and write in the         <pid>-<start>.json, replaced
                                      plane's own thread                                whole every tick_s
                                                                                     --> entry point service (reads all,
                                                                                         every tick_s / 2)
Coordinator at its cycle         <--  directive set in memory (one assignment)  <--  xc/directives-v<N>.json
  (XC-C5, XC-C7)                      the thread lists and reads the newest file     <-- entry point (only writer)
Generator at each take           <->  ledger cache (consult, claim)             <->  ledger-<writer>.log, one per
  (XC-G1, XC-G2)                      sync task on the loop every sync_s;            process, in $TMPDIR/xc-<run id>/
Monitor's resume (XC-W1)         -->  result row                                     (node-local); each process reads
                                      the thread appends and reads the logs          every other process's log
launch script (XC-E1)            ------------------------------------------------>  xc/manifest.json; starts the
                                                                                     entry point service
```

- **Control plane (the Coordinators' side).** Digest, table, file, entry point, directives. P0 and P3.
- **Data plane (the Generators' side).** Consult, claim, publish, cache, logs. P1, and the optional P2.

### 2.3 The entry point: one service per run, in a process of its own

Whether one entry-point agent or a decentralised control plane should hold the run's state was left open. This
plan takes one entry point per run, and makes it a service of the run:
1. **A resident service** (P0): `python -m fusion_miniapp.entry_point serve --dir <plane folder>`. Its own process,
   with its own event loop. It reads every progress file every `tick_s / 2`, holds the roll-ups in memory, and
   answers the end user over HTTP on `127.0.0.1`. It writes a summary of its own traffic (section 4.8).
2. **The only writer of directives** (P3): `guide`, `revoke`. One writer, so directives never conflict.
3. **A fallback and an option.** The same commands read the files directly when the service does not answer, so a
   user can still see the run if the service is gone. `ask` (off unless used) sends one question and the status
   summary to one of the run's own model servers in one call. A model may draft a directive. It is never issued
   without a `guide` command from the user.

Why not an agent hosted in a population: it would run on a campaign event loop, which is the bound
(section 2.1). Why not a decentralised plane where Coordinators tell each other: no edge between campaigns exists
today (code), and every such message would land on that same loop. The plane is decentralised where the hot path
is: each process keeps its own campaigns' table and ledger cache and resolves directive targets itself, and nothing
waits on the entry point.

### 2.4 Where this plan departs from the first sketch of the shape

| First sketch | This plan | Evidence |
|---|---|---|
| A shared ledger consulted by the Generator before it queues; a hit reuses the result or moves the candidate; a miss claims, then evaluates | Consult and claim at the queue, as sketched, for every candidate. A hit changes nothing by default: it is counted and reported. Reuse is an optional knob (P2), served in the worker. No "move" | A hit served at the Generator or the Monitor changes `Monitor simulations == reports == messages_sent` (`layer_autonmi.py:864-865`, `:934`), the campaign totals (`:878-881`, `:953-957`), AUTONMI's `simulations_match` and `decide.collect`'s params check (`decide.py:445-452`). A new draw shifts stream 0 (async) or the shared stream (`decide.py:321`, `:664-676`) |
| A stale read costs a repeated evaluation, never a wrong result | A stale read costs a missed hit. Two claims of one key that crossed in flight are counted as a race | Nothing acts on a claim or a hit by default |
| Phases, measure first: count would-be repeats before anything depends on the ledger | P1 is the live ledger itself. No phase is gated on repeats or work saved | The mini-app needs the pattern, not a payoff. The fact of section 2.5 stays a fact |
| An entry-point agent | A service of the run in a process of its own, started by the launch script or the head | The event-loop bound (section 2.1) |
| A backend per way of running; RaDEX under Dragon | One transport: files, each with one writer | The store has no put-if-absent, no compare-and-set and no key scan, and its calls are synchronous on the population's loop (AUTONMI's `rascal` store). Dragon runs on one node only. The `live-p<idx>.json` files already cross processes under Dragon (`layer_autonmi.py:766-780`) |
| Directives: guidance, priorities, budgets, stop | Guidance notes only | B bounds three roles (`layer_autonmi.py:1363`, `:1389`, `:1440`) and `verify` (`:958`). No priority lever exists in the client or the servers. A lost stop sentinel would wedge a campaign |
| A stall is seen at the Coordinator's cycle | Liveness comes from the process's plane; stall from cumulative fields | A stuck Coordinator has no cycle (`:1343`). The parent replaces the best on every result while nothing is feasible (`parent/evolutionary_loop.py:1016-1022`) |
| Off means unchanged | Unchanged, and an off record keeps today's schema: with every plane knob at its default the section is left out of `rec["knobs"]`, `campaign.json` and the app's parameters | Every knob is echoed in `rec["knobs"]` (`record.py:519`) and AUTONMI's `Campaign` (built at `layer_autonmi.py:2229`) writes `definition.knobs` into `campaign.json` and the app's parameters |
| (not stated) | Every hook is guarded | A raise in `Script.remember`, a Generator or Monitor hook or a tick would fail that role and with it the campaign (README, "When one agent fails") |

### 2.5 A fact about the experiment: campaigns cannot repeat each other

This is a fact about the configured experiment. It is not a gate and it decides nothing in this plan.

- Campaign a<i> uses seed i (`layer_autonmi.py:729-735`). The seed initializes a population. The parent asks for
  P Latin-hypercube points at that seed (`parent/evolutionary_loop.py:435-454`), and the sampler builds one profile
  per point (`parent/steps/step0/profile_generator.py:429-445`).
- So each of the 16 individuals of a campaign's initial population has its own six frozen shape values: `a3`,
  `a4`, `bcore_Te`, `bcore_ne`, `core_linear_frac_Te` and `core_linear_frac_ne`. They belong to the individual,
  not to the campaign.
- A descendant carries the six values of its parent. Mutation alters only the seven mutable values
  (`parent/evolutionary_loop.py:1046-1063`); model proposals replace only those seven (`decide.py:263-269`,
  `:323-333`, `:890-905`). The restart branch of `repopulate` (`:1088-1089` in the parent) needs an empty survivor
  list, which a completed generation never gives: `select_survivors` keeps rank 0 whenever the population is not
  empty (`:959`, `:970`, `:977`), and `decide.collect` requires all P results (`decide.py:445-448`).
- Laptop: over seeds 0 to 19,999, all 320,000 initial six-tuples were distinct. Two complete campaigns in each mode,
  run through the normal AUTONMI launch path, gave disjoint sets of 16 initial tuples, and every later candidate
  carried a tuple of its own campaign's set.
- So no two campaigns ever evaluate the same full profile at the configured shape, in either mode. Repeats of a
  full profile happen only inside a campaign in the generations mode: survivors go into the next population
  unchanged (`decide.py:313-320`) and every member is queued again (`layer_autonmi.py:1199-1200`). Laptop: about
  24.7% of 10,240 candidates in 320 fixture campaigns.

This is why the ledger's key leaves the six values out (section 3.1). With them in, no consult would ever hit
another campaign.

## 3. Answers to the design questions

### 3.1 What makes two candidates "the same work" here: the grid key

In this plan two candidates are the same work when their grid keys are equal. That is a definition for the
pattern. It makes no claim that their inner loops agree, and they generally do not. The plan does not need them to.

- **The key** (XC-G3): for each of the 7 mutable parameters, in the order of `decide.ranges()` (`decide.py:96-101`:
  `a1_ne`, `gamma_ne`, `Delta`, `a2_ne`, `a1_Te`, `gamma_Te`, `a2_Te`, with their bounds), its cell on a grid of
  `cells = round(1 / grid_step)` cells over its range, clipped to `0 .. cells - 1`. Each cell index takes 8 bits;
  the key is the 56-bit integer of the seven. `grid_step` is at least 0.004, so `cells` is at most 250.
- **Campaign-independent.** The key holds no seed, campaign, generation, candidate number, shape value, model, kind
  or knob. It is the same integer in every process and on every node.
- **The hit rate is a knob.** Laptop, from the saved candidate captures of three fixture runs (320 campaigns each,
  10,240 candidates, every earlier claim visible at once), the share of consults that found another campaign's
  claim:

  | `grid_step` (cells per parameter) | async, stub | generations, stub | async, model off |
  |---|---|---|---|
  | 0.5 (2) | 98.8% | 98.7% | 98.7% |
  | 0.333 (3) | 78.7% | 78.0% | 78.2% |
  | 0.25 (4) | 25.4% | 22.7% | 24.4% |
  | 0.2 (5) | 6.1% | 5.9% | 6.3% |
  | 0.125 (8) | 0.30% | 0.16% | 0.29% |
  | 0.1 (10) | 0.05% | 0.05% | 0.02% |
  | 0.05 (20) | 0 | 0 | 0 |

  Own-campaign hits come on top in the generations mode: 24.7% at a fine grid (the survivors queued again).
- **The rate grows with the campaigns.** At a step of 0.2, the first 20, 40, 80, 160 and 320 campaigns gave 0.31%,
  1.0%, 1.6%, 3.3% and 6.1% (laptop, async stub): about double for double the campaigns while the rate is low.
- **At the configured 20,000 campaigns** (inferred, assuming the same spread of candidates): a step of 0.125 gives
  about 15 to 20% other-campaign hits (with 62.5 = 20,000 / 320: 0.30% x 62.5 = 19%, or from the step of 0.2,
  6.1% x (5/8)^7 x 62.5 = 14%), and a step of 0.1 about 3%. The default `grid_step` is 0.125, a choice (open decision 1). Tests use 0.25 or 1.0 to make
  hits certain on a few campaigns.

### 3.2 Where a Generator creates and queues a candidate, and where it consults and claims

- **Generations mode**, `Script._generator` (`layer_autonmi.py:1181-1204`). `decide.generator_step`
  (`decide.py:304-336`) draws the initial population (`:308-312`) or repopulates (`:313-321`) and, with
  `outer.generator: llm`, fills the non-survivor slots by the refill call (`:323-333`; prompt
  `decide.refill_messages`, `:233-254`, its ask on the last line at `:252-253`). The population goes to the
  Coordinator at `layer_autonmi.py:1197` and P items go on the queue at `:1199-1200`, each
  `{id: g<g>c<i>, params, candidate_idx, generation, t0}` (`decide.py:341-345`). Known there: the campaign, its
  seed, the survivors' full params, the rng state, the previous generation's results and the sources.
- **Asynchronous mode**, `Script._generator_async` (`layer_autonmi.py:1373-1412`). It waits for an empty queue
  (`:1390`), takes survivors, digests and guidance without waiting (`:1391-1393`), and waits only for its first
  survivors (`:1394-1396`). `decide.async_profiles` (`decide.py:655-680`) makes TAKE = 2 profiles: the k-th
  initial profile for k < P, else `mutate_params` of a survivor drawn on stream 0. `Script._reconcile`
  (`:1414-1429`) applies the model's values. The final params exist only after `:1399`; they go on the queue at
  `:1402-1403` as `{id: c<k>, params, candidate_idx, generation, source, filled, t0}`.
- **The consult and the claim** (P1): in both modes, in the loop that puts the take's items on the queue (XC-G1
  above `:1199`, XC-G2 above `:1402`), before each `put`: `plane.guard("consult", ledger.consult_claim, campaign,
  call["candidate_idx"], call["params"], t0)`. It computes the key, reads the process's cache, classifies the
  consult (`miss`, `own`, `other_claimed`, `other_done`), records the claim in the cache when the key is new, and
  appends one claim record to the outbound buffer. Every candidate is claimed, hit or not. No I/O, no message, no
  wait.
- **What a hit does: nothing to the campaign.** The item, the queue, the campaign's random streams and its results
  stay as they are. The hit is counted per campaign (`hits` in the progress table) and per process. Serving it
  here would break the campaign's own checks (section 2.4). `Mailbox.put` makes a JSON copy of each item
  (`layer_autonmi.py:357`), and the consult only reads `params`, so the queued dict is never touched.
- **The result row** (P1): at the Monitor's resume (`layer_autonmi.py:1215-1220`, XC-W1), where both modes pass
  with the result. `result["params"]` holds the params (the inner loop echoes them). One result row goes to the
  buffer and the key is marked done in the cache.

### 3.3 Where a Coordinator cycles, what it knows there, and where a directive enters

| | Async (`Script._coordinator_async`, `layer_autonmi.py:1321-1371`) | Generations (`Script._generation`, `:1169-1179`) |
|---|---|---|
| Cycle | wake on scored results `:1343`; `decide.async_on_scored` `:1344` (`decide.py:732-780`); tallies `:1345-1349`; survivors `:1356`; guidance call `:1357-1362`; `Remember` and the B test `:1363-1364` | `make/g` `:1173`; population `:1174`; results `:1177`; `decide.on_evaluated` `:1178`; `Remember` `:1179` |
| Cycles per campaign | 1 to B = 32; 26.9 on average (StellarAI) | G = 2 (configured) |
| Known there | `cst`: `arrived`, `kept`, `cycles`, `best` (with `selection_basis`, and `generation` = the arrival number of its last change, `decide.py:751`), `survivors`, `window`; `self.B`; `Script.running[campaign]`; `Script.answered[campaign]` | `cst`: `generation`, `best`, `gens` (`n_evaluated`, `n_good`, `n_survivors`, `best_te_core_metric`) |
| Memory written | `decide.async_memory`: `{arrived, kept, best, rng, survivors}` (`decide.py:788-791`) | `decide.memory`: `{generation, best, gens, rng}` (`decide.py:532-534`) |
| Digest | `Script.remember` (`:1596`, XC-C2), called by AUTONMI's runner after every cycle and generation | same |
| Directive read | above `:1356` (XC-C5), after survival is decided, before the guidance call | at `:1173` (XC-C7), into the repopulate message |
| Directive in the prompt | `decide.guidance_messages`, one line before `GUIDANCE_ASK` (`decide.py:996-999`, XC-C6) | `decide.refill_messages`, one line before the ask (`decide.py:252`, XC-C8); only with `outer.generator: llm` and slots to fill |
| Directive in decisions | none | none |
| "Next cycle" means | after the next scored result: about every 56 s per campaign on the 8 x 800 real run (1,506 s / 26.9, inferred average) | after the next generation's barrier |

A laptop probe (a print in `Script.remember` and `Script._begin` on a scratch copy) confirmed the digest hook:
with G 2, `remember` ran 2 times per campaign; with B 8 in the async mode, 7 to 8 times; only the Coordinator
called it in both modes; `_begin` ran once per campaign.

The stub and `tests/fake_vllm.py` answer by the request's last line (`llm.py:94-108`, `:132-138`), so the last
line must stay the ask. The stub's JSON replies are drawn from a generator keyed by that line
(`llm.py:135`), so with the stub a note changes the requests and no reply (code; the P3 test confirms it).

### 3.4 What already aggregates across campaigns, and what the planes reuse

| Mechanism (code) | Reused? | How, or why not |
|---|---|---|
| Coordinator memory into the store, `Script.remember` (`layer_autonmi.py:1594-1596`) | the call, yes; the store, no | The call is the digest hook (XC-C2). The store is not the medium: it has no put-if-absent and no key scan, its calls are synchronous on the population's loop, and k launches have k separate stores |
| The head's live line, `LiveLine` (`:1694-1781`) | no | It shows model and agent load once a second. Progress columns would change every record's `speed.live` |
| Handed files `live-p<idx>.json`, `proc-p<idx>.json` (`:598-611`, `:766-780`, `:807-809`) | the pattern | Replaced whole by one writer (tmp, then `os.replace`), read by a tolerant reader. The progress file is its sibling |
| Campaign lines `agents-p<idx>.jsonl` (`:1093`, `:1598-1607`, `:1640-1642`) | no change | They stay the record's source. A test checks that the last progress file agrees with them |
| Call sums `CallSums` (`:535-595`), written at the population's end and on SIGTERM (`:802`, `:1665`) | the pattern and the places | The plane's counters per population, `xc-p<index>.json`, are written beside them (XC-A2, XC-A3) and summed into the record (XC-W2) |
| `record.write` and `write_record` (`record.py:350-565`; `layer_autonmi.py:2699-2794`) | both | The `cross_campaign` section is added in `write_record` (XC-K4). Observe and ledger add no scientific field to campaign rows |
| `merge_records` (`merge_records.py:159-239`) | yes | Sums the sections of the launches it reads and, with `--xc-dir`, adds the entry point's summary (XC-K5). Its `_validate` already refuses launches whose non-machine knobs or commits differ (`:92-118`). It stops on a launch folder with no `run_record.json` (`:163`, `_read_launch` `:75`), so coverage against the manifest is the entry point's |
| The launch script's folders (`run-<i>/`, `launch/stellar_ai_autonmi.sbatch:289`) and node-local `TMPDIR` (`:407`) | yes | It writes the manifest, starts the entry point and passes `--xc-dir` (XC-E1). The ledger logs live under `TMPDIR` (XC-R7) |
| Flowgentic's task key (`ToolRuntime.submit`) | no | Its key is agent, scope and call: an idempotence guard for one agent's resume, not a shared structure |

### 3.5 What each plane costs at 20,000 campaigns (inferred unless marked)

Inputs. Configured: N = 20,000 campaigns (the StellarAI scale knob file, README "The StellarAI run path"),
B = 32 (P 16 x G 2), so E = 640,000 evaluations, 2 Monitor slots, 100,000 hosted agents. The layout for this
arithmetic is the README's start for 20,000: k = 32 launches of 625 campaigns on one campaign node (inferred
there); Dragon has P = 40 population processes. StellarAI (8 x 800, real inner loop): 26.9 Coordinator cycles,
106.6 model calls and 58,400 prompt tokens per campaign; 453 model calls/s summed on one node. Run length T: the
README's line 100 s + 0.22 s x N = 4,500 s, which the README calls a guess past 6,400 campaigns.

Unit costs (laptop). Control plane (on AC power): a digest's list writes 0.08 us; building a digest dict 0.64 us;
a progress document for 10,000 campaigns 3.8 ms to encode (310 KB, 7 columns), 0.47 ms to write, 1.8 ms to
parse. Ledger (on battery, high power mode; pure Python): the grid key 0.44 us; consult and claim with its 64-byte
record 0.60 us; publish a result row 0.64 us; parse and insert one remote record 0.07 us; one cache entry
(56-bit key to an integer) 97 bytes.

**Control plane (P0, P3):**

| Quantity | Arithmetic | Value |
|---|---|---|
| Digests (in the process) | N x (26.9 cycles + start + end) | 578,000 per run (at most N x 34 = 680,000); 128/s run-wide; 4/s per launch |
| Bound by the servers | 28.9 digests / 106.6 calls per campaign = 0.27 per call; x 453 calls/s | about 123/s on one model node |
| Loop work for digests | 578,000 x 0.08 to 0.64 us | 0.05 to 0.37 s per run, all processes together |
| Progress files written | one per population per `tick_s` (10 s) | 0.2/s (one launch of 2 populations), 3.2/s (k = 32), 4/s (Dragon) |
| Bytes per tick | about 50 B per campaign (11 columns; 31 B measured for 7) x N | about 1.0 MB per 10 s run-wide, 100 kB/s, in every layout |
| Encoding, in the plane's thread | about 6 ms per 10,000 campaigns at 11 columns | one launch of 20,000: about 12 ms per 10 s (0.1%); k = 32: about 0.4 ms per launch per 10 s |
| Entry point fan-in | every progress file, every `tick_s / 2` (5 s) | 32 files per pass at k = 32 (6.4 reads/s), 40 under Dragon (8/s), 2 for one launch |
| Entry point parse | about 1.0 MB per pass, from 1.8 ms per 310 KB | about 6 ms per pass, 0.12% of its own process |
| Digest to entry point | up to `tick_s` to the next write, up to `tick_s / 2` to the next read | about 7.5 s on average, at most about 15 s, plus write and parse |
| Memory | about 100 B per campaign in the tables; the entry point holds the latest file of each population | 2 MB run-wide; about 1 MB in the entry point |
| Directive check per cycle (P3) | one lookup and a time compare, under 1 us | under 0.6 s per run in all |
| Directive file listings (P3) | one listing of `xc/` per process per tick, in the thread; a read only on a new version | 3.2/s at k = 32 |
| Issue to apply (P3) | the next listing (up to `tick_s`), then the campaign's next cycle (about 56 s apart, async) | about 33 s on average |
| **A live note in prompts (P3)** | Coordinator calls only: 26.9 x 100 tokens = 2,690 per campaign, over 58,400 | **4.6% more prompt tokens** while the note is live (53.8 M of 1.17 G per run) |
| (not planned) The note also in the Generator's prompts, 150 tokens | (26.9 + 16) x 150 = 6,435 per campaign | 11.0% |

The note is the only cost that lands on the model servers, which set the pace. Its effect on run time is not
measured: prefill and decode cost differ per token, and a note can change reply lengths and later work. P3's exit
check measures it. That is why `note_chars` defaults to 400 characters (about 100 tokens at 4 characters per
token, inferred) and v1 shows notes to one prompt per cycle: the Coordinator's guidance prompt in the async mode,
the Generator's refill prompt in the generations mode.

**Data plane, P1 (the live ledger; `sync_s` 2 s; 64-byte records):**

| Quantity | Arithmetic | k = 32 (Dragon, P = 40) |
|---|---|---|
| Consults and claims | one each per candidate: E | 640,000 per run; 142/s run-wide; 4.4/s per launch. Local reads |
| Result rows | one per evaluation: E | 640,000 per run; 142/s run-wide |
| Records written | 2 E x 64 B | 1.28 M, 82 MB per run; 284/s and 18 kB/s run-wide; one append per process per sync |
| Records read | each process reads every other process's records: (k - 1) / k x 82 MB per process | 79 MB per process, 2.5 GB per run, 0.56 MB/s run-wide (Dragon: 3.2 GB) |
| File reads | one `pread` of new bytes per other log per sync: k x (k - 1) / `sync_s` | 496/s per node (Dragon: 780/s), on node-local disk |
| Loop work per evaluation | consult and claim 0.60 us + publish 0.64 us | 0.79 s per run in all processes; 0.025 s per launch (under 0.001% of 4,500 s) |
| Loop work for remote records | 1.24 M records per process x 0.07 us (parse and insert, an upper bound for the insert alone) | 0.09 s per process per run; about 0.04 ms per sync |
| Cache memory | at most E keys x 97 B per process; fewer with hits | 62 MB per process; 2.0 GB per node (0.2% of the 950,000 MiB the job asks for, `launch/stellar_ai_autonmi.sbatch:97`; Dragon 2.5 GB) |
| Backlog if writes stall | this process's records for the rest of the run, held in memory | at most 40,000 x 64 B = 2.6 MB per launch |
| Visibility (claim in one process to the cache of another) | up to `sync_s` to the writer's append, up to `sync_s` to the reader's poll, plus the loop handoff | about 2 s on average, at most about 4 s |
| Hits at `grid_step` 0.125 | section 3.1 | about 15 to 20% of consults find another campaign's claim |

**Data plane, P2 (reuse in the worker, only if switched on):**

| Quantity | Arithmetic | k = 32 |
|---|---|---|
| Result files | the first result per key, about 3.0 kB each (record 1.6 kB and trace 1.4 kB, laptop) | at most 640,000 files, 1.9 GB per run on node-local disk |
| Reuses | `other_done` consults at take time; at 15% of E | about 96,000 per run, 21/s run-wide: one file read in a worker instead of one inner loop |

### 3.6 Failure and consistency

| Case | What happens | Why it is safe |
|---|---|---|
| A plane hook raises | The guard catches it, counts it under its site, turns the plane off in that process (`disabled_at`), writes one warning to stderr | No exception reaches an agent. The campaigns go on as with the plane off |
| A ledger log cannot be written or read (folder missing, disk full, a log deleted) | `write_errors` or `read_errors` is counted in the thread; the plane stays on. Consults go on against the local cache, so cross-process records stop flowing and hits fall | No campaign reads anything from the ledger by default. With reuse (P2) a missing result file is a refused reuse and the inner loop runs |
| A claim not yet seen (two campaigns in different processes claim one key within one visibility window) | Both go on. When the other's record arrives, the cache keeps the earlier claim (time, then writer) and the later claimant counts a `race` | Every process converges on the same first claimant. Nothing acts on a claim |
| A torn record (a reader reads while a writer appends) | A reader takes whole 64-byte records only and keeps the rest for the next sync. Each record carries its writer's sequence number; a gap is counted as `gaps` | Expected 0: each writer appends with one `os.write` on an append-only descriptor |
| Two writers of one result file (P2 only) | Results are published with `os.link` on the final name, which refuses an existing name (laptop: `FileExistsError`; `os.rename` and `os.replace` keep the last writer instead). The later one is counted and removed | The first published result per key stays |
| Dead or slow entry point | Campaigns never read it. Planes keep writing progress files; directives already issued apply until `expires_t`, then lapse. The CLI falls back to reading the files. A slow service reads late; its own summary shows the lag | Nothing in a campaign waits on it. The user may restart it; it then registers again (section 4.8) |
| A process dies or blocks | Its files stop changing. After 3 x `tick_s` without an `end` mark the entry point shows its population `lost` and its open campaigns `unknown`, and the gap against the manifest. Its ledger log stops growing; the others keep what they read | Each file has one writer; no other file depends on it |
| A stuck campaign | No cycle, so no digest. The entry point flags it `quiet` after `quiet_s` and shows its liveness detail: inner loops or Monitor calls in flight, model calls answered (section 4.4) | Read from the process, which still ticks |
| A directive arrives late | It is prospective: a campaign shows it from its next cycle on | A note cannot change a result already scored, the survival rule, the budget, or stop a campaign. It can change the candidates the Generator's model proposes from then on (async: `decide.async_on_guidance`, `decide.py:636-638`, then `Script._reconcile`; generations: the refill reply, `decide.py:323-333`), and through them later scores and survival |
| Two launches disagree | Each launch is meant to own a disjoint campaign range; nothing enforces it during the run. Each progress file's name carries its writer's pid and start time. The entry point checks every file's `run_id` and range against the manifest and shows overlapping ranges and duplicate writers instead of summing them (`merge_records` refuses overlaps after the run, `merge_records.py:170-172`). The only shared inputs are the directive set (one writer, a version that only grows) and the ledger logs (one writer each, append only) | No shared mutable state is reconciled |
| Half-written or corrupt progress file | Writers use tmp, then `os.replace`. Readers skip a file they cannot parse and keep the last good one, as the head does with `live-p<idx>.json` | As the handed files today |
| Shared file system slow or full | A progress write takes long or fails. A tick whose previous write has not finished is skipped and counted. An `OSError` is caught in the thread and counted in `write_errors`; the plane stays on. A ledger batch is never skipped: it waits in memory (at most 2.6 MB per launch). A write that hangs holds only the plane's daemon thread, which is never joined at exit | The loop never waits on I/O, and the process exits as today |
| SIGTERM | Local runtime only: `Script.cut_short` (XC-A3) queues the final progress files, ledger batch and counters; the process gives all such writes one shared 2 s deadline. The record is `interrupted`, exit 143, as today. Under Dragon no flush runs (the head calls `cut_short` only for the Scripts in its own process, `layer_autonmi.py:2254-2274`): the last progress file is the last tick's, and the entry point shows the population `lost` | Same path as the call sums (StellarAI ran it locally: interrupted, exit 143) |
| Clocks on several nodes (P4) | Ages and visibility times are taken against each record's or file's own time and the reader's clock. Skew between nodes shifts them | Not measured; P4's exit check measures it |

### 3.7 What the record and the README would say, and the tests that pin "off means unchanged"

Section 9.

## 4. The design

### 4.1 Components and cardinalities (configured shape: 20,000 campaigns, async, P 16, G 2, B 32, 2 slots)

| Component | Module (planned) | Lives in | Count | Talks to |
|---|---|---|---|---|
| Hooks | `layer_autonmi.Script` (XC-C1 to C5, C7, G1, G2, W1) | the campaign's event loop | 20,000 campaigns | its process's plane, by a guarded method call |
| Plane, the aggregator | `plane.Plane` | one per process: per local launch (all its populations), per population process under Dragon | 1, k (`FUSION_LAUNCHES=k`) or P (Dragon) | its tables, its ledger cache, its thread, the folders |
| Progress table | `plane.Table` | inside the plane | one per population | written by the hooks on the loop, copied at the tick |
| Ledger cache | `plane.Ledger` | inside the plane | one per process | the Generator and Monitor hooks; the sync task; the thread |
| Plane thread | one daemon `threading.Thread(name="xc", daemon=True)`, never joined at exit (a `ThreadPoolExecutor` is joined at interpreter exit, so a hung write would hold the process: laptop, 3.0 s for a 3 s job) | one per plane | 1 per process | writes progress files and ledger batches; reads other processes' logs and the directive files |
| Plane folder | `<out>/xc`, or `--xc-dir` | the run folder (shared scratch on the machine, `launch/stellar_ai_autonmi.sbatch:335`) | 1 per run | the planes, the launch script, the entry point |
| Ledger folder | `$TMPDIR/xc-<run id>/` (node-local); `xc/ledger/` when a run spans nodes (P4) | the campaign node | 1 per node | the planes of the node |
| Entry point service | `entry_point` (`serve`) | a process of its own on the campaign node | 1 per run | the plane folder; the user over HTTP on `127.0.0.1`; optionally one model server |

### 4.2 The plane in one process

- **Created** by the first `Fusion.build` in a process when any plane knob is on (XC-A1); `Fusion.plane` is
  `None` otherwise, and every hook is then a single `if self.plane is not None:` test.
- **The guard.** Every hook calls `plane.guard(site, fn, *args)`. It catches `Exception`, adds 1 to
  `errors[site]`, sets `plane.off = True` and `disabled_at`, and writes one warning to stderr. Once `off`, every
  hook returns at once. Sites: `begin`, `remember`, `report`, `failed`, `consult`, `publish`, `merge`,
  `directive`, `tick`, `sync`, `write`. In the thread an `OSError` is caught first, counted in `write_errors` or
  `read_errors`, and the plane stays on; only other exceptions reach the guard.
- **On the event loop**, and only there: the hooks' list writes and cache reads and inserts; the tick's snapshot;
  the sync task's swap of the outbound buffer; the merge of remote records handed back by the thread; reading the
  directive set (an attribute the thread replaces by one assignment, read without a lock).
- **The tick** runs every `tick_s` on the loop, as a task started once per process. It records how late it woke
  (`tick_late_ms`, a coarse loop-lag reading per process). For each table it copies the columns and the roll-up
  counters (with the ledger's counters), and reads the liveness detail of the campaigns the previous snapshot found
  quiet (at most 50 per table, section 4.4). It hands the copies to the thread. If the previous progress write has
  not finished, it skips this write and counts it. The tick writes every table every time, so a file's age is its
  process's heartbeat.
- **The sync** runs every `sync_s` on the loop when the ledger is on. It swaps the outbound buffer for an empty
  one (one assignment) and hands the full one to the thread. A ledger batch is never skipped.
- **In the plane's thread**: encode each progress column with its own `json.dumps` call (so the thread never holds
  the interpreter lock for long: about 0.5 ms per column of 10,000, inferred), write `progress-...json.tmp`,
  `os.replace` it; append the ledger batch to this process's log with one `os.write`; read the new bytes of every
  other log; parse whole records and hand them to the loop with `loop.call_soon_threadsafe(ledger.merge, ...)`;
  list `xc/` at each tick for new logs and new directive versions, and parse a directive file only when its N is
  higher than the last one that parsed. The thread never touches an object the loop owns.
- **Ending**: `Fusion.rows` (XC-A2) writes the population's last file with `"end": true`, the last ledger batch,
  and the plane's counters `xc-p<index>.json` beside the call sums; the process's last population stops the tick,
  the sync and the thread. `Script.cut_short` (XC-A3) does a bounded flush on SIGTERM, in the local runtime only
  (section 3.6). The head's `miniapp_workflow` closes the plane in its `finally` (XC-A4). `flush(timeout_s)` hands
  the last writes to the thread and waits on an event with that timeout; on timeout it abandons them, counts it,
  and returns.
- **Off**: no plane, no thread, no folder, no file, no entry point process, no new record key.

### 4.3 The digest

`Table.remember(campaign, memory, now)` reads (XC-C2):

| Column | Async | Generations |
|---|---|---|
| `n` (results in) | `arrived` | `generation` x P |
| `kept` | `kept` | `gens[-1]["n_survivors"]` |
| `best` | `best["te_core_metric"]`, rounded to 6 places; null if not finite | same |
| `feas` | `best["selection_basis"] == "feasible_core"` | same |
| `best_k` | `best["candidate_idx"]` | same |
| `gain_n` (results in at the last feasible gain) | `best["generation"] + 1` when `feas`, else -1 | `(best["generation"] + 1) x P` when `feas`, else -1 |
| `cycles` | +1 per call | `generation` |
| `t_last` | `now - t0`, 0.1 s (`t0` the file's wall-clock epoch) | same |

`Table.begin` (XC-C1) sets the state `r` and `t_last`. `Table.end` (XC-C3, XC-C4) sets `d` or `f` (with the error
cut to 200 characters in a small dict of failures). The state column is one character per campaign: `w` waiting, `r`
running, `d` done, `f` failed, `i` interrupted (set by the SIGTERM flush). With the ledger on, `hits` counts the
campaign's consults that found another campaign's claim (written by the consult hook, XC-G1 and XC-G2). With steer
on, `dir_v` (P3) is the newest directive version the campaign applied.

### 4.4 Stall and liveness (judged by the entry point)

`S_R` is `--stall-results` (default P, from the manifest). `Q` is `--quiet-s` (default 600 s, about 10 times the
56 s mean cycle interval of the real run, inferred).

| Flag | Rule | Why this rule |
|---|---|---|
| `stalled` | `feas` and `n - gain_n >= S_R` | once a feasible result exists, the parent replaces the best only on a feasible gain (`parent/evolutionary_loop.py:1007-1015`) |
| `infeasible` | not `feas` and `n >= S_R` | before any feasible result the best changes on every result (`:1016-1022`), so "unchanged best" means nothing there |
| `quiet` | state `r` and `now - (t0 + t_last) >= Q`, with `t0` the file's wall-clock epoch | a Coordinator emits only after a scored result (`layer_autonmi.py:1343`) |
| population `lost` | its file is older than 3 x `tick_s` and has no `end` mark | its process is gone or blocked; its open campaigns become `unknown` |
| population `late` | `tick_late_ms` above 1,000 ms | the event-loop bind the README's "Open items" describes |
| run `no_progress` | results in unchanged over 300 s while campaigns run | no score advance observed over the stated coverage and interval |

`counts.feasible_agents` is 0 in all 8 launches of the 8 x 800 real run (StellarAI), so `feas` is false everywhere
and `infeasible` flags every campaign after `S_R` results in that run. In the async mode the survivors are kept
oldest first, not ranked (`decide.async_on_scored`), so no column of the digest shows an infeasible campaign
improving. A rule for that phase is open decision 5.

Liveness detail for a quiet campaign, read on the loop at the next tick (at most 50 per table): in flight =
the sum over its Monitors of `Script.sims[m] - Script.reports[m]` (both modes: `sims` counts at the take,
`layer_autonmi.py:1248`, `reports` after the report, `:1242`; it includes the Monitor's own note call); in the
async mode also `Script.running[campaign]` (`:1228`, `:1256`), the inner loops alone; and
`Script.answered[campaign]` (`:1618`), its model calls answered. A flat answer count cannot distinguish an
outstanding first model call from a blocked role; the entry point reports that cause as unknown.

### 4.5 Directives (P3)

- **Shape and writer.** Version N contains the full directive set. The entry point service is the only writer.
  Each `guide` or `revoke` computes its update against the latest version, writes it into a fresh private inode
  (exclusive creation), closes it and links it to the N+1 final name, then unlinks only that private name.
  Committed versions stay immutable. Readers can miss intermediate versions; each version holds the full set.
- **Kinds.** `note` only: text of at most `note_chars` characters. Every note has `expires_t` (default
  `--ttl-s 1800`); the user renews a note by issuing it again. A dead entry point therefore lapses to today's
  behaviour.
- **Targets.** `{"all": true}`, `{"ranges": [[first, last_exclusive], ...]}` or `{"ids": ["a12", ...]}`. The
  entry point resolves a selector such as "the stalled ones" to ids when it issues the note, so no process needs
  run-wide state.
- **Reading.** The plane's thread parses a new highest version and replaces the in-memory set by one assignment.
  `plane.note_for(campaign, now)` returns the newest note that targets the campaign and has not expired.
- **Applying, async** (XC-C5, XC-C6): at the cycle, after `async_on_scored` and before the survivors go out, the
  Coordinator takes `note_for`. With a note it calls `decide.guidance_messages(cst, rows, knobs, note=text)`,
  which adds the line `"Run-wide guidance from the operator: " + text` before `GUIDANCE_ASK`. The first time a
  campaign dispatches a request carrying the version, it records one `applied` ack, or `not_applicable` when
  `llm.mode` is off (no prompt exists). `async_memory` and `async_finish` pick named keys (`decide.py:788-791`,
  `:794-803`), so nothing else sees the note.
- **Applying, generations** (XC-C7, XC-C8): `:1173` sends both `initialize` (g = 0) and `repopulate`. A live
  note rides only on a repopulate message, as `msg["note"]`; `refill_messages` shows it as one line before the
  ask. The ack is `applied` only when a request carrying the note was sent; otherwise `not_applicable` (with
  `outer.generator: mutation`, or with no non-survivor slot to fill, `decide.py:325-326`).
- **Acks travel back** in the progress files (`directives.acks`), so the entry point sees which targets applied a
  version and when.
- **Provenance.** With `steer` on, a campaign line gains `directives: [{"id", "v", "at_n", "cycle"}]` only when
  the campaign applied one (added at XC-C3). That key changes `results_sha256` on purpose: a steered campaign is a
  different run.
- **Prompt room.** The longest prompts at the StellarAI shape are about 6,700 characters (Coordinator) and 5,200
  (Generator) against the bound of 21,504 characters in `tests/test_async.py:336` (laptop). A 400-character note
  fits.

### 4.6 The live ledger (P1)

- **Cache.** One `plane.Ledger` per process: a `dict` from the 56-bit grid key to one integer that packs the first
  claim's time (milliseconds from the run's epoch), its campaign and a done bit. Touched only on the loop. Counters
  beside it.
- **Consult and claim** (XC-G1, XC-G2; section 3.2): `consult_claim(campaign, candidate, params, now)` returns one
  of `miss`, `own`, `other_claimed`, `other_done`, inserts the claim when the key is new, adds `other_*` to the
  campaign's `hits` column, and appends one claim record to the outbound buffer.
- **Publish** (XC-W1): `publish(campaign, candidate, params, score, now)` marks the key done and appends one result
  record. `score` is the result's `te_core_metric` (null as NaN).
- **Record** (64 bytes, little-endian): `key` u8, `t` f8 (wall-clock seconds), `campaign` u4 (i of a<i>),
  `candidate` u2, `kind` u1 (1 claim, 2 result), `flags` u1, `seq` u4 (this writer's record number), the seven
  values f4 x 7, `score` f4, 4 bytes of padding.
- **Logs.** One append-only `ledger-<writer>.log` per process in the ledger folder, `writer` being
  `<pid>-<start>` (a host-unique ID in P4). The first 64 bytes are a header: version, run id, writer, `grid_step`.
  A reader refuses a log whose run id or `grid_step` differs from its own and counts it.
- **Sync.** Every `sync_s` the loop hands the outbound buffer to the thread (section 4.2). The thread appends it,
  then reads the new whole records of every other log from the last offset, and hands them to the loop.
  `merge(records, t_read)` inserts each new key, keeps the earlier claim of a key (time, then writer), sets the
  done bit from result records, counts a `race` when an earlier remote claim arrives for a key this process
  claimed, and adds each record's `t_read - t` to a fixed-bin histogram (`visible_ms`).
- **Counters per process**, in each progress file's roll-up and in the record: `consults`, `claims`, `published`,
  `hits` (`own`, `other_claimed`, `other_done`), `races`, `records_out`, `bytes_out`, `records_in`, `bytes_in`,
  `reads`, `syncs`, `sync_late` (a batch handed over while the previous one was still being written), `gaps`,
  `writers_seen`, `refused_logs`, `cache_keys`, `backlog_max_bytes`, `visible_ms` (p50, p95, max).
- **With one local launch** there is one process: the log is written and nobody else reads it. Cross-process
  traffic needs `FUSION_LAUNCHES=k` or Dragon.

### 4.7 Acting on a hit (P2, optional)

Off by default (`reuse` false), with no gate. It changes the compute pattern: a reused candidate skips its inner
loop. The campaign's numbers after a reuse are another candidate's and carry no scientific meaning; the record says
how many results were reused.

- **Publish** at XC-W1: each result and its trace also go to a fresh `<key>.<writer>.<n>.tmp` in the ledger
  folder (opened with `O_EXCL`), which is closed, then linked to `<key>.json` with `os.link`, which refuses an
  existing name. The temporary name is removed whether the link succeeded or not. The first result per key stays.
- **Lookup at take time** in the task tool's `prepare` (XC-R3, `layer_autonmi.py:1098-1103`), when the runner
  submits the Monitor's `Simulate`: the key from `args["params"]`, a read of the process's cache. On a done key
  first claimed by another campaign, `reuse={"key", "path"}` joins the worker's kwargs beside `knobs` and
  `monitor`.
- **In the worker** (XC-R1 `sim.evaluate`, XC-R2 `sim.evaluate_fixture`; `evaluate_placed` calls `evaluate`):
  read the file; rebind `params`, `candidate_idx`, `t_start`, `elapsed_seconds`, `touches` (zero), `touch_check`
  (`"reused"`), `cold_start`, `worker_peak_rss_mb` and `summary_path` to this candidate, and add `reused: {"key",
  "campaign", "candidate", "sim_s"}`. If the file is missing or unreadable, or has no trace while
  `monitor.capture` asks for one, run the inner loop and count `reuse_refused`. The fixture needs its own site:
  with `--sim fixture` the tool runs `sim.evaluate_fixture` (`run.py:289-292`, used at `layer_autonmi.py:1095`).
- **Counts.** Each candidate stays one engine task, so `simulations_match`, the Monitor invariant and
  `each_candidate_evaluated_once` hold, and `decide.collect`'s params check passes on the rebound params.
  `decide.on_evaluated` (XC-R5) and `decide.async_on_scored` (XC-R6) leave reused results out of the touch verdicts
  and count them. `record.write` (XC-R4) gives `touch_check.reused`, so a reuse is never a touch-check failure
  (`record.py:537-538`, `:558-559`). The record gives `logical` (`counts.sims` as today), `physical` (inner loops
  run) and `reused` apart. The campaign line gains `n_reused`.

### 4.8 The entry point service (P0)

- **Started** by the launch script with `FUSION_XC=1` (XC-E1), once per run, before the launches; or, with no
  launch script, by the head that wrote the manifest (XC-A4), as a child process of its own. Stopped at teardown
  (SIGTERM), after the launches have ended.
- **Registers** in `xc/entry_point.json` (`{v, run_id, pid, host: "127.0.0.1", port, started_t}`) by a private
  file linked to the final name. A second `serve` for the run refuses. A registration whose pid no longer runs on
  the same host is replaced.
- **Every `tick_s / 2`**: lists `xc/`, reads each progress file whose modification time changed, parses it, and
  keeps the latest per population. It computes the run's roll-up, the flags (section 4.4) and coverage against the
  manifest, and appends one point to a history ring (results, digests, hits and ledger records per second).
- **Serves** HTTP on `127.0.0.1` from memory: `GET /status`, `/campaigns`, `/campaign/<id>`, `/best`,
  `/populations`, `/ledger`, `/directives`; `POST /guide`, `/revoke` (P3). Every answer carries `as_of`, coverage
  and age.
- **Records its own traffic** in `xc/entry_point_summary.json`, every 60 s and at exit: passes, files and bytes
  read per pass (fan-in), parse ms, the age of each file at read (the digest-to-entry-point latency, p50, p95,
  max), sequence gaps per population, queries and their time, directives issued.
- **The CLI**: `python -m fusion_miniapp.entry_point <command> --dir <plane folder> [--json]` asks the service.
  When none answers within 2 s, the readers read the files themselves and say so. `guide` and `revoke` need the
  service, the only writer. On the machine the user runs the CLI on the campaign node (for example with
  `srun --overlap` into the job); the file fallback works from any node that mounts the run folder.

## 5. Interface (planned)

```python
# src/fusion_miniapp/plane.py: standard library only
VERSION = 1

class Plane:                                        # one per process (Fusion.plane), made by the first build
    def __init__(self, folder: Path, ledger_dir: Path | None, knobs: dict, run_id: str, mode: str) -> None
    off: bool                                       # set by guard on the first error in this process
    def guard(self, site: str, fn, *args)           # every hook: an exception is counted and turns the plane off
    def attach(self, script) -> "Table"             # XC-A1: one table per population
    def start(self, loop) -> None                   # once per process: the tick, the sync and the xc thread
    def note_for(self, campaign: str, now: float) -> "Note | None"   # XC-C5, XC-C7: from memory, no I/O
    def ack(self, campaign: str, version: int, outcome: str) -> None # applied | not_applicable
    def population_ended(self, table: "Table") -> None   # XC-A2: last file, last batch, counters
    def flush(self, timeout_s: float = 2.0) -> None      # XC-A3 queues local writes; the 2 s deadline is shared at XC-A4
    def section(self) -> dict                            # XC-K4: the record's cross_campaign section

class Table:                                        # one population's campaigns; index int(c[1:]) - first
    def __init__(self, first: int, count: int, budget: int, P: int, mode: str, t0: float) -> None
    def begin(self, campaign: str, now: float) -> None                       # XC-C1
    def remember(self, campaign: str, memory: dict, now: float) -> None      # XC-C2
    def end(self, campaign: str, state: str, now: float, error: str = "") -> None   # XC-C3 "d", XC-C4 "f"
    def snapshot(self, script, quiet: list[str]) -> dict                     # on the loop, at the tick

class Ledger:                                       # P1: one per process, inside the plane
    def __init__(self, folder: Path, writer: str, run_id: str, grid_step: float, bounds: dict) -> None
    def consult_claim(self, campaign: int, candidate: int, params: dict, now: float) -> str   # XC-G1, XC-G2
    def publish(self, campaign: int, candidate: int, params: dict, score: float | None, now: float) -> None  # XC-W1
    def lookup(self, params: dict) -> Path | None   # P2, XC-R3: another campaign's published result file
    def take_batch(self) -> bytearray               # the sync task, on the loop: the records since the last sync
    def merge(self, records: bytes, t_read: float) -> None   # on the loop, handed by the thread
    def counters(self) -> dict

def grid_key(params: dict, bounds: dict, grid_step: float) -> int   # XC-G3: 8 bits per parameter, 7 parameters

# src/fusion_miniapp/entry_point.py
def serve(folder: Path, bind: str = "127.0.0.1", port: int = 0) -> None   # P0: the service
def gather(folder: Path) -> "Run"                   # manifest, every progress file, directives; tolerant reads
def status(run, now: float) -> dict                 # every answer carries as_of, coverage and age
def campaigns(run, flag: str | None = None, order: str = "best", limit: int = 100) -> list[dict]
def campaign(run, cid: str) -> dict
def guide(run, text: str, targets: dict, ttl_s: float = 1800.0) -> dict   # P3: in the service only
def revoke(run, directive_id: str) -> dict
def ask(run, question: str, endpoints_file: Path, api_key_file: Path | None) -> dict   # optional
def main(argv=None) -> int   # serve | status | campaigns | campaign | best | populations | ledger | directives
                             # | guide | revoke | ask; all take --dir; --json on every reader
```

Changed signatures (planned): `decide.guidance_messages(cst, rows, knobs, note="")` and
`decide.refill_messages` reading `msg.get("note", "")`; `sim.evaluate(..., reuse=None)` and
`sim.evaluate_fixture(..., reuse=None)` (P2). With the defaults, every request body and result is unchanged.

## 6. Data shapes (planned)

**`xc/manifest.json`.** Before launch, the launch script (or a lone head) writes a fresh run ID and the complete set
of expected slice and population ranges. A creator closes a fresh private temporary file before linking it to the
final name, then unlinks its private name. A losing creator validates the winner's ID and layout. Hand launches
that join one run receive its run ID in `--xc-run-id`.

```json
{"v": 1, "run_id": "<16 hex>", "mode": "async", "budget": 32, "population": 16, "generations": 2,
 "campaigns": {"first": 0, "count": 20000}, "launches": [{"slice": [0, 32], "first": 0, "count": 625, "population_ranges": [[0, 625]]}, "..."],
 "runtime": "local", "nodes": 1, "written_t": "<float>", "by": "launch_script"}
```

**`xc/progress-a<first>-<count>-<pid>-<start>.json`**, one per population, replaced whole every `tick_s`. The name
carries the campaign range and its writer's pid and start time, so a slice run again or a hand launch with the
same range writes its own file:

```json
{"v": 1, "run_id": "<16 hex>", "seq": "<int>", "t": "<float>", "t0": "<float>", "first": 0, "count": 625,
 "mode": "async", "budget": 32, "pid": "<int>", "end": false,
 "tick_late_ms": "<float>", "write_ms": "<float>", "skipped": 0, "write_errors": 0, "errors": {}, "off": false,
 "rollup": {"waiting": 0, "running": 0, "done": 0, "failed": 0, "interrupted": 0, "results": 0, "kept": 0,
            "feasible": 0, "digests": 0, "best": {"campaign": "a17", "best": "<float>", "n": "<int>"}},
 "ledger": {"consults": 0, "claims": 0, "published": 0, "hits": {"own": 0, "other_claimed": 0, "other_done": 0},
            "races": 0, "records_out": 0, "records_in": 0, "bytes_out": 0, "bytes_in": 0, "gaps": 0,
            "writers_seen": 0, "cache_keys": 0, "visible_ms": {"p50": null, "p95": null, "max": null}},
 "table": {"state": "rrd...", "n": [], "kept": [], "best": [], "feas": [], "best_k": [], "gain_n": [],
           "cycles": [], "t_last": [], "hits": [], "dir_v": []},
 "failed": {"a42": {"role": "monitor", "error": "<at most 200 characters>"}},
 "quiet": [["a7", "<age_s>", "<in_flight>", "<running or null>", "<answered>"]],
 "directives": {"version_read": 0, "read_t": "<epoch seconds>", "acks": {"3": {"applied": 0, "not_applicable": 0}}}}
```

`ledger` and `hits` exist only with the ledger on (the process's counters, repeated in each of its populations'
files); `dir_v` and `directives` only with `steer`. The values above are placeholders, not measurements.

**`xc/directives-v<N>.json`** (P3; written only by the entry point, one immutable file per version):

```json
{"v": 1, "run_id": "<16 hex>", "version": 7, "written_t": "<float>",
 "directives": [{"id": "d3", "version": 3, "kind": "note", "targets": {"all": true},
                 "text": "<at most note_chars characters>", "issued_t": "<float>", "expires_t": "<float>"}]}
```

**`ledger-<writer>.log`** (P1): a 64-byte header, then 64-byte records (section 4.6). **`<key>.json`** (P2): one
result with its trace.

**`xc/entry_point.json`** and **`xc/entry_point_summary.json`** (P0): section 4.8.

**The record's section** (XC-K4; only when a plane knob is on; summed by `merge_records`, XC-K5):

```json
"cross_campaign": {"v": 1, "run_id": "<16 hex>", "folder": "xc", "observe": true, "ledger": true, "steer": false,
  "reuse": false, "populations": "<int>", "ticks": "<int>", "files_written": "<int>", "skipped": "<int>",
  "write_errors": "<int>", "write_ms": {"p50": null, "max": null}, "tick_late_ms": {"p50": null, "p95": null, "max": null},
  "errors": {"<site>": "<int>"}, "disabled_at": null, "digests": "<int>",
  "last": {"done": "<int>", "failed": "<int>", "interrupted": "<int>", "results": "<int>"},
  "ledger": {"grid_step": 0.125, "sync_s": 2.0, "consults": "<int>", "claims": "<int>", "published": "<int>",
             "hits": {"own": 0, "other_claimed": 0, "other_done": 0}, "races": 0, "records_out": "<int>",
             "records_in": "<int>", "bytes_out": "<int>", "bytes_in": "<int>", "reads": "<int>", "syncs": "<int>",
             "sync_late": 0, "gaps": 0, "read_errors": 0, "refused_logs": 0, "writers_seen": "<int>",
             "cache_keys": "<int>", "visible_ms": {"p50": null, "p95": null, "max": null}},
  "directives": [{"id": "d3", "version": 3, "text": "...", "issued_t": "<float>", "expires_t": "<float>",
                  "targets": {"all": true}, "applied": 0, "not_applicable": 0}],
  "reuse": {"logical": "<int>", "physical": "<int>", "reused": "<int>", "reuse_refused": "<int>", "saved_sim_s": 0.0}}
```

The folder is named relative to the run, never as a path. `reuse` exists only with reuse on. The merged record sums
the launches' counters, keeps the spread of the latency figures, and with `--xc-dir` adds the entry point's summary
as `entry_point`. Coverage against the manifest is the entry point's, not the record's.

## 7. Transports per way of running

| Way of running | Progress files (writer) | Ledger logs | Entry point service | Who reads directives | Status |
|---|---|---|---|---|---|
| One local launch (laptop, or by hand) | the head's one plane, one file per population, on the head's one loop and its thread, in `<out>/xc` (`--xc-dir` overrides) | one log in `$TMPDIR/xc-<run id>/`; one process, so no other reader | started by the head, which writes the manifest (XC-A4) | the head's plane | the layer ran on StellarAI (2,000 campaigns, bound by its loop); the plane: not built |
| `FUSION_LAUNCHES=k` (start here: what ran to the end on StellarAI, by hand) | each launch's plane, in its own process, in `$run_dir/xc`, one for all k launches (XC-E1 passes `--xc-dir`) | k logs in the node's `$TMPDIR/xc-<run id>/` (XC-R7); each launch reads the other k - 1 | started once by the launch script (XC-E1) | each launch's plane | 8 x 800 real ran by hand on StellarAI; the script's `FUSION_LAUNCHES` checked on a laptop only; the plane: not built |
| `FUSION_RUNTIME=dragon` | each population process's plane (`Fusion.build` runs there, precedent `layer_autonmi.py:760-763`), in `$run_dir/xc` | P logs in the node's `$TMPDIR/xc-<run id>/`; each population process reads the other P - 1 | started once by the launch script (XC-E1) | each population process's plane | the layer ran on StellarAI with the fixture inner loop (2,000 campaigns); the plane: not built. No SIGTERM flush: the head does not call `cut_short` for Scripts in other processes (`layer_autonmi.py:2254-2274`) |
| Several campaign nodes (P4) | unchanged, in the run folder on the shared scratch file system | in `xc/ledger/` on the shared file system, so that every node reads every log; `sync_s` raised if the file system needs it | one, on the first campaign node | unchanged | not built: no campaign side spans nodes yet (Dragon refuses more than one node; k launches run on one node) |

In P4 the reads grow with the square of the writers: at 4 nodes x 32 launches and `sync_s` 2 s, 128 x 127 / 2 =
8,128 reads per second on the shared file system (inferred). P4 starts with `sync_s` 10 s (1,626 per second) and
measures. During the run nothing else crosses a process: no socket between campaign processes, no broker, no store
key. The user's queries reach the entry point over HTTP on the campaign node.

## 8. Knobs, flags and defaults (planned)

A new knob section, `cross_campaign`, read by the AUTONMI layer only (XC-K1). Its keys are not added to `record`
or `campaign`: `tests/test_sim_knobs.py:89-90` pins those two sections exactly. A laptop experiment on a scratch
copy confirmed it: a key added to `record` broke that test, a new section did not.

| Knob | Rule | Default | Phase | `MACHINE_KNOBS` | `ADDED_KNOBS` value | `run.py`'s layers |
|---|---|---|---|---|---|---|
| `cross_campaign.observe` | `("bool",)` | false | P0 | yes: it changes no result | false | refused when true |
| `cross_campaign.tick_s` | `("num", 1, None)` | 10.0 | P0 | yes | 10.0 | refused when changed |
| `cross_campaign.ledger` | `("bool",)`; needs `observe` | false | P1 | yes | false | refused when true |
| `cross_campaign.grid_step` | `("num", 0.004, 1.0)` | 0.125 | P1 | yes | 0.125 | refused when changed |
| `cross_campaign.sync_s` | `("num", 0.1, None)` | 2.0 | P1 | yes | 2.0 | refused when changed |
| `cross_campaign.steer` | `("bool",)`; needs `observe` | false | P3 | no: it changes prompts | false | refused when true |
| `cross_campaign.note_chars` | `("int", 0, 2000)` | 400 | P3 | no | 400 | refused when changed |
| `cross_campaign.reuse` | `("bool",)`; needs `ledger` | false | P2 | no: it changes results | false | refused when true |

- Each knob goes into `knobs.SCHEMA` (XC-K1), `knobs/smoke.yaml` (every knob must be set there,
  `knobs.py:200-208`), `tests/test_sim_knobs.py`'s `NEWER` (`:9-11`), `compare_records.ADDED_KNOBS` and, for the
  machine knobs, `MACHINE_KNOBS` (XC-K3), and `knobs.validate`'s refusals (XC-K2). `knobs/hundredk.yaml` stays as
  it is and takes the defaults from smoke. A machine knob file turns the plane on.
- `merge_records._validate` then requires launches to agree on `steer`, `note_chars` and `reuse`
  (`merge_records.py:111-118`).
- **Layer flags**, not knobs (paths and identities, so not in the record): `--xc-dir DIR` (XC-A5), default
  `<out>/xc`, refused without `cross_campaign.observe`; `--xc-run-id ID` for a hand launch joining a run.
  `merge_records --xc-dir DIR` adds the entry point's summary.
- **Launch script**: `FUSION_XC=1` (XC-E1), default unset. Unset, the script behaves as today, and its dry-run
  lines are unchanged.
- **Entry point options** (they change no campaign): `--dir`, `--json`, `--bind`, `--port`, `--stall-results`
  (default P), `--quiet-s` (600), `--ttl-s` (1800), `--to FIRST-LAST`, `--ids`, `--flag`.

## 9. Record, README and tests

**Off.** With every plane knob at its default, the `cross_campaign` section is left out of `rec["knobs"]`
at the shared `record.write` serializer (`record.py:519`; LangGraph/Academy call it directly at `run.py:438`), of
`campaign.json` and of the app's parameters (both written by AUTONMI's `Campaign` from `definition.knobs`, so the
`Fusion` definition drops it there). An off record is today's schema. Nothing else changes: no folder, no thread, no
file, no process, no request, no campaign line, no `results_sha256`. `ADDED_KNOBS` still lets `compare_records`
and `merge_records` read a missing section as the defaults when an on record meets an off or older one
(`merge_records.py:118`).

**On.** Section 6 says what each feature adds. With `observe` and `ledger` only, the campaigns' computation and the
campaign schema stay unchanged: the generations reference cases match. For independent async runs
`results_sha256` can differ between identical runs even with no plane (laptop: two identical fixture runs gave
`c4be6240...` and `0d94b6b1...`; the docstring of `tests/test_layer_autonmi.py:1849`, `compare_records.ASYNC_VARY`),
so an async run with the plane on is compared as any two async runs are: through `compare_records` and the replay.
With `steer`, campaign lines carry `directives` when applied, by design. With `reuse`, `n_reused`,
`touch_check.reused` and results that are not science.

**README.** Now: one line under "Open items" pointing to this file. When P0 is built: a section "Watching and
steering a run" (the knobs, the plane folder, the entry point's commands per way of running); "What a run
writes" gains `xc/` and the ledger folder; "What has run, and what has not" gains one row per phase.

**Existing tests that must pass unchanged with the knobs at their defaults** (code): `tests/test_layer_autonmi.py`
`:197` (results, counts and parent files against the references), `:300`, `:315`, `:330`, `:342`, `:438`,
`:462` (the real inner loop on CPU), `:701` (the model requests of both layers), `:725`, `:768`, `:1281`, `:1809`
and `:2389` (a recorded async order replays), `:1849` (the async record gains only counts without the toggle),
`:2249`, `:2320`, `:2626`, `:2670`; `tests/test_merge_records.py:50`; `tests/test_agent_layers.py:182`;
`tests/test_async.py:381` (the prompt bound); `tests/test_sim_evaluate.py:125`, `:166`;
`tests/test_sim_knobs.py:89-90`. One existing test changes: `tests/test_sim_knobs.py`'s `NEWER` gains the new
knobs, or `:15` fails.

**New tests (planned):**

| Phase | Test | Asserts |
|---|---|---|
| P0 | `test_off_makes_no_plane` | knobs at defaults: no `xc/` folder, no ledger folder, no thread named `xc`, no entry point process; no `cross_campaign` key at the record's top level, in raw `rec["knobs"]` for all three layers, or in AUTONMI campaign lines, `campaign.json`, the launcher manifest/report parameters and the Parameters checkpoint event; `compare_records` finds the record equal to one written before the section existed |
| P0 | `test_observe_changes_no_result` | the generations-mode reference cases with `observe`: the same `results_sha256`, counts, parent files and request bodies as off |
| P0 | `test_observe_replays_the_async_order` | an async run with `observe`: every campaign replays (`tests/async_replay.py`) |
| P0 | `test_progress_ends_as_the_lines` | the last file per population: done equals the lines without error, failed the error lines; `n` and `best` equal each line's `n_sims` and `best.te_core_metric` |
| P0 | `test_plane_errors_never_reach_an_agent` | a raise injected at each guard site in turn: the generations reference results stay equal; `errors` counts the site; `disabled_at` is set |
| P0 | `test_unwritable_plane_folder` | the run passes; `write_errors` > 0; the plane stays on (`off` false) |
| P0 | `test_sigterm_writes_the_last_files` | local runtime only: exit 143; the unfinished campaigns are `i` in the last files |
| P0 | `test_unflushable_write_never_holds_exit` | a write that hangs: the process still exits within the flush timeout, and the abandoned write is counted |
| P0 | `test_entry_point_serves` | the service on written files: `/status` from memory with as_of, coverage and age; the flags of section 4.4; overlapping ranges and duplicate writers shown and not summed; its summary counts passes, files, bytes, ages and gaps; a second `serve` refuses |
| P0 | `test_entry_point_gone` | the service killed mid-run: the run passes with the same results; the CLI falls back to the files and says so |
| P0 | `test_run_py_layers_refuse_the_plane` | LangGraph and Academy refuse each plane knob with `KnobError`; `ledger` or `steer` without `observe` and `reuse` without `ledger` are refused |
| P0 | `test_two_slices_one_plane_folder` | two `--slice` launches with one `--xc-dir`: one entry point; its totals equal the merged record's; a slice that never started shows as missing; two creators of the manifest keep the winner's run ID |
| P0 | the launch script's dry run (as `tests/test_launch.py`) | with `FUSION_XC=1` the manifest is written, the entry point started and `--xc-dir` passed; without it the dry-run lines are unchanged |
| P1 | `test_ledger_changes_no_result` | the generations reference cases with `ledger` on: the same `results_sha256`, counts and request bodies as off; an async run replays |
| P1 | `test_ledger_counts_every_candidate` | consults = claims = candidates, published = evaluations; with `grid_step` 1.0 every consult after a campaign's first finds a claim (`own + other = consults - 1` in one process); the generations mode at a fine grid has `own` equal to the survivors queued again |
| P1 | `test_ledger_crosses_processes` | two slices on one ledger folder: each reads the other's records (`records_in` of one equals `records_out` of the other after the final sync), `gaps` 0, `visible_ms` recorded; two claims of one key inside one sync give one race; a log with another run id or `grid_step` is refused and counted |
| P1 | `test_ledger_torn_and_stalled` | a partial trailing record is read only once whole; an unwritable ledger folder: the run passes, `write_errors` counted, the control plane goes on |
| P3 | `test_note_reaches_the_prompts` | with `tests/fake_vllm.py`: the note's line just before `GUIDANCE_ASK` (async) and before the refill ask (generations), cut to `note_chars`; with `steer` on and no file, request bodies byte-equal to off; the prompt bound holds at `note_chars`; the stub's replies are unchanged |
| P3 | `test_note_expires_and_is_revoked` | guide and revoke keep the immutable prior version; a skipped version and an exited target; one dispatch ack per campaign and version; `not_applicable` with `llm.mode: off` and with `outer.generator: mutation` |
| P3 | `test_steer_off_ignores_directives` | a directives file with `steer` off changes nothing |
| P2 | `test_reuse_keeps_every_count` | generations fixture, `reuse` on, `grid_step` 1.0: `verify`, AUTONMI's checks and `decide.collect` pass; `logical = physical + reused`; reused results carry `touch_check: "reused"` |
| P2 | `test_reuse_refused_runs_the_inner_loop` | a missing or unreadable file, or no trace while `monitor.capture` is on: the inner loop runs and `reuse_refused` counts it; a second publisher of one key leaves the first file |
| P2 | async reuse | every campaign replays (`tests/async_replay.py`) |

## 10. Phases, each with its exit check

Every exit check is about the communication pattern, never about work saved. "Within noise" means within the
spread of three runs of the same shape with the feature off.

| Phase | Builds (tags) | Exit check |
|---|---|---|
| **P0** digests, aggregator and entry point (laptop, then StellarAI) | `plane.Plane`, `Table`, the guard, the tick and thread, the progress file, the manifest, `--xc-dir`, the entry point service and CLI, knobs, the record section, the merged sections (XC-C1 to C4, A1 to A5, E1, K1 to K5, W2) | Laptop: the suite at 357 passed and 1 skipped plus the P0 tests. StellarAI: `FUSION_LAUNCHES=8`, 8 x 800 campaigns, real inner loop, `observe` on: the run passes as on 2026-11-25; agent phase within noise of those records; event-loop lag p95 within 1 ms of the 6-7 ms measured then; the entry point up and answering throughout. Recorded and set against section 3.5 scaled to the run: digests per second (about 6,400 x 28.9 / 1,500 s = 123/s, inferred), progress files per second, fan-in per pass, bytes per pass, digest-to-entry-point age p50 and p95 (at most about 1.5 x `tick_s`), sequence gaps (0 apart from counted skips). The entry point's final totals equal the merged record's campaigns and `counts.sims` |
| **P1** the live ledger (laptop, then StellarAI) | `Ledger`, `grid_key`, the consult and claim, the result row, the sync, the logs, the counters (XC-G1 to G3, W1, R7) | Laptop: the P1 tests. StellarAI: the P0 run with `ledger` on, at `grid_step` 0.125 and at 0.25: consults = claims = candidates and published = evaluations; `gaps` 0; records, bytes and reads per second recorded and set against section 3.5; `visible_ms` p50 and p95 (about `sync_s` and at most about 2 x `sync_s`); hits by class and races at each step; event-loop lag p95 within 1 ms and the agent phase within noise of the P0 run |
| **P3** steer with notes (after P0; beside P1) | `guide`, `revoke`, the directive set, acks (XC-C5 to C8) | Laptop: the P3 tests. StellarAI: a note to every campaign at t = 300 s is shown by at least 95% of targets within two eligible prompt opportunities after their first observed version; the issue-to-dispatch latency p50 and p95; dead, ineligible and unconfirmed counts; event-loop lag as in P0; added prompt tokens, changed reply lengths and elapsed time at matched load |
| **P2** acting on hits, optional | the result files, `prepare`'s lookup, the worker's rebinding, the counts (XC-R1 to R6) | Laptop: the P2 tests. StellarAI: the P1 run with `reuse` on: `logical = physical + reused`; result files written and read per second, their bytes and the publish-to-reuse age, set against section 3.5; the ledger's traffic as in P1; inner loops skipped, `reuse_refused`, the workers' busy share (`sim_bound_ratio`) and the agent phase reported against the P1 run. No threshold on savings |
| P3b stop, budget, priority | not in this plan | A separate design with a handshake over the B-tied bounds (XC-C9 to C12) |
| **P4** several campaign nodes | the ledger logs in `xc/ledger/`, host-unique writer IDs; nothing else new until the campaign side spans nodes | The same entry point on the shared run folder; coverage complete; reads per second on the shared file system and `visible_ms` across nodes recorded; `lost` and `quiet` behave when one node's launch is killed; clock skew measured |

## 11. Open decisions

1. **The default hit rate.** The default `grid_step` 0.125 gives, by inference, about 15 to 20% of consults a hit on
   another campaign's claim at 20,000 campaigns (0.3% on the laptop's 320). A step of 0.1 gives about 3%; 0.25
   gives most consults a hit at that scale. Which rate should the default aim at?
2. **Build P2 at all?** Reuse changes the compute pattern (fewer inner loops) and makes a campaign's numbers after
   a reuse meaningless. It is off by default and optional.
3. **The cost of notes.** A live 100-token note adds about 4.6% to the run's prompt tokens on the servers that
   set the pace. Is that acceptable, and should notes also reach the Generators' prompts (about 11% at 150
   tokens)?
4. **Stop, budget and priority directives.** A later design (P3b, sites XC-C9 to C12), or never, with whole-run
   stop left to SIGTERM?
5. **A stall rule while nothing is feasible.** All 6,400 campaigns in the measured eight-launch run were
   infeasible, so the feasible-only `stalled` rule did not fire (section 4.4). Should the digest gain a column for
   that phase (for example, results since the window's best wall-blend rank last improved), or is "infeasible
   after `S_R` results" enough for the entry point?

## 12. Where the plan attaches: the tags

Each tag below is a comment in the code, `# CROSS-CAMPAIGN [<tag>] (see CROSS_CAMPAIGN_PLAN.md): ...`, one to
four lines at the indentation of the line it sits above. The comments are the only change in the code. Every tag
in this table is in the code and every tag in the code is in this table: `grep -rn "CROSS-CAMPAIGN \[" src launch`
lists them. 36 tags in 8 files: `layer_autonmi.py` 21, `decide.py` 6, `sim.py` 2, `knobs.py` 2,
`merge_records.py` 1, `record.py` 1, `compare_records.py` 1, `launch/stellar_ai_autonmi.sbatch` 2. Line numbers
are those of `000300c`, before the comments went in.

| Tag | Phase | File | Function | Above line (at `000300c`) | What attaches there |
|---|---|---|---|---|---|
| XC-C1 | P0 | `layer_autonmi.py` | `Script._begin` | 1140: `self.began[me] = time.time()` | start digest: the campaign is marked running |
| XC-C2 | P0 | `layer_autonmi.py` | `Script.remember` | 1596: `self.pop.store.put(f"campaign/` | the one Coordinator digest hook, both modes |
| XC-C3 | P0 | `layer_autonmi.py` | `Script.report` | 1601: `self.reported.add(campaign)` | done digest; with steer, the line's applied notes |
| XC-C4 | P0 | `layer_autonmi.py` | `Script.on_done` | 1641: `self._line({"agent_id": campaign, "error":` | failed digest (first failure of a campaign) |
| XC-C5 | P3 | `layer_autonmi.py` | `Script._coordinator_async` | 1356: `self.mail.deliver(me, "generator", "survivors"` | directive read, async: the newest live note for this campaign |
| XC-C6 | P3 | `decide.py` | `guidance_messages` | 996: `user = ("Results just judged:` | the note as one line before GUIDANCE_ASK |
| XC-C7 | P3 | `layer_autonmi.py` | `Script._generation` | 1173: `self._send(me, "generator", "make", g, msg)` | directive, generations mode: the note rides on the repopulate message |
| XC-C8 | P3 | `decide.py` | `refill_messages` | 252: `ask = (f"Generation {g + 2} of {G}` | the message's note as one line before the ask |
| XC-C9 | P3b, not planned | `layer_autonmi.py` | `Script._coordinator_async` | 1363: `return _state(s, "cycle" if cst["arrived"] < ...` | stop or budget would change this bound on B (with C10 to C12) |
| XC-C10 | P3b, not planned | `layer_autonmi.py` | `Script._generator_async` | 1389: `while gst["made"] < self.B:` | a stop would end the Generator's loop |
| XC-C11 | P3b, not planned | `layer_autonmi.py` | `Script._evaluator_async` | 1440: `while scored_n < self.B:` | a stop would lower the Evaluator's bound |
| XC-C12 | P3b, not planned | `layer_autonmi.py` | `Fusion._verify_async` | 958: `if not (co["candidates_distinct"] == B` | a stopped campaign would be checked against its stopped_at |
| XC-C13 | none (refused) | `decide.py` | `run_campaign` | 576: `await svc.remember(agent_id, memory(cst))` | the Academy layer's generation boundary (LangGraph's is `layer_langgraph.coordinate`, `:147-148`, locked) |
| XC-A1 | P0, P1 | `layer_autonmi.py` | `Fusion.build` | 760: `if os.getpid() != self.head_pid:` | attach the population's table to the process's plane (and its ledger cache); start the tick and sync once |
| XC-A2 | P0, P1 | `layer_autonmi.py` | `Fusion.rows` | 802: `sc.sums.write(sums_file(Path(sc.lines.name), ...` | population end: last progress file, last ledger batch, the plane's counters beside the call sums |
| XC-A3 | P0, P1 | `layer_autonmi.py` | `Script.cut_short` | 1665: `self.sums.write(sums_file(Path(self.lines.nam...` | SIGTERM, local runtime only: one bounded flush (no flush under Dragon) |
| XC-A4 | P0 | `layer_autonmi.py` | `miniapp_workflow` | 2244: `await campaign.start_services()` | open the plane folder, the manifest and the entry point if no launch script did, the run id; close in the finally |
| XC-A5 | P0 | `layer_autonmi.py` | `parse` | 2027: `p.add_argument("--stagger-batch"` | --xc-dir DIR, the folder every launch of a run shares |
| XC-E1 | P0 | `launch/stellar_ai_autonmi.sbatch` | `main` | 503: `if (( launches > 1 )); then several "${args[@...` | FUSION_XC=1: write the manifest, start the entry point service, pass --xc-dir to every launch and to merge_records |
| XC-G1 | P1 | `layer_autonmi.py` | `Script._generator` | 1199: `for call in decide.evaluation_calls(reply["po...` | consult and claim each candidate before its put (generations); a hit is only counted |
| XC-G2 | P1 | `layer_autonmi.py` | `Script._generator_async` | 1402: `for call in calls:` | consult and claim each final candidate before its put (async); a hit is only counted |
| XC-G3 | P1 | `decide.py` | `ranges` | 99: `r = {k: tuple(evo.get_param_ranges(MODEL, KI...` | the grid key's seven parameters and bounds; the six shape values left out |
| XC-W1 | P1 (P2 publishes files here) | `layer_autonmi.py` | `Script._monitor` | 1220: `numbers = None` | publish one result row per evaluation, on the loop |
| XC-W2 | P0, P1 | `layer_autonmi.py` | `write_record` | 2733: `for i in range(a.populations):   # summed int...` | sum the populations' plane counters into the record; remove their files |
| XC-R1 | P2 (optional) | `sim.py` | `evaluate` | 249: `if not os.environ.get(COLD_CHILD) and is_cold(` | reuse inside the worker, real inner loop |
| XC-R2 | P2 (optional) | `sim.py` | `evaluate_fixture` | 321: `digest = hashlib.sha256(json.dumps(params, so...` | reuse in the fixture |
| XC-R3 | P2 (optional) | `layer_autonmi.py` | `Script.__init__` | 1098: `self.evaluate = task_tool(` | prepare: the take-time lookup, another campaign's result file to the worker |
| XC-R4 | P2 (optional) | `record.py` | `write` | 537: `"touch_check": {"evaluations": sum(devices.va...` | reused counted apart in the touch check; logical and physical counts |
| XC-R5 | P2 (optional) | `decide.py` | `on_evaluated` | 495: `for kind, n in (e.get("touches") or {}).items():` | a reused result adds no touches or verdict (generations) |
| XC-R6 | P2 (optional) | `decide.py` | `async_on_scored` | 767: `for kind, k in (e.get("touches") or {}).items():` | a reused result adds no touches or verdict (async) |
| XC-R7 | P1 (P2 files too) | `launch/stellar_ai_autonmi.sbatch` | `main` | 407: `export TMPDIR=$node_tmp` | the ledger's node-local folder |
| XC-K1 | P0 (each phase adds its knobs) | `knobs.py` | `SCHEMA` | 109: `"record": {` | the new cross_campaign section |
| XC-K2 | P0 | `knobs.py` | `validate` | 234: `if "async" not in modes:` | run.py's layers refuse the plane; the knobs' dependencies |
| XC-K3 | P0 | `compare_records.py` | `module (ADDED_KNOBS)` | 45: `# knobs newer than some records, with the val...` | ADDED_KNOBS and MACHINE_KNOBS entries |
| XC-K4 | P0 to P3 | `layer_autonmi.py` | `write_record` | 2766: `if knobs["record"]["extras"]:` | the record's cross_campaign section, only when on |
| XC-K5 | P0 | `merge_records.py` | `merge` | 237: `rec["wall_seconds"] = _range(` | sum the launches' sections; add the entry point's summary |

**Locked places (no note in the code).** These files may not be edited, so the plan names the place here only.

| File | Place | Why it matters |
|---|---|---|
| `parent/evolutionary_loop.py` | `initialize_population` (`:435-454`), `mutate_params` (`:1046-1063`), `repopulate` and its restart seed (`:1072-1104`, `:1089`) | where each initial individual receives its six frozen values and each descendant inherits its parent's (section 2.5) |
| `parent/evolutionary_loop.py` | `run_evolutionary_loop`'s evaluation of the whole population each generation (`:1403`) | why the generations mode has own-campaign hits (about 24.7% at a fine grid) |
| `parent/evolutionary_loop.py` | `update_best_result` (`:1000-1041`), `select_survivors` (`:929`) | the feasible-gain rule that `gain_n` reads; survival, which nothing in this plan steers |
| `parent/evo_agents.py` | the directed-proposer seam its docstring names (`:17-18`) | the parent's own place for a guided Generator |
| `run.py` | `Services.simulate` (`:255`), `Services.remember` (`:268-270`), `Services.report` (`:272-275`), `_fan_out`'s live array (`:351-388`); `sim_function` (`:289-292`) | the other layers' evaluation, digest, report and live counts (the plane is refused there, XC-K2); `sim_function` picks `evaluate` or `evaluate_fixture`, which is why P2 needs XC-R2 |
| `layer_langgraph.py`, `layer_academy.py` | `coordinate` (`:134-155`); `CoordinatorAgent.run_generations` (`:80-83`) | those layers' Coordinator cycle: Academy's through `decide.run_campaign` (XC-C13); LangGraph's in `coordinate` itself (`:147-148`) |
| `knobs/hundredk.yaml` | the preset | stays off; it takes the new knobs' defaults from smoke, and a machine knob file turns the plane on |
| `launch/run.sh`, `batch.sh`, `stellar_ai.sbatch`, `lux.sbatch`, `perlmutter.sbatch` | the `run.py` launches | no plane for those layers |

## 13. What this plan rests on

- **Code**: every line cited, read at `000300c`. Every tag's line was checked against its text. The comments are
  checked against the commit before them: each Python file's syntax tree is equal, and the launch script's lines
  other than comments are unchanged.
- **Laptop (2026-11-26)**, on scratch copies and outside the test suite: the six-tuple count over 20,000 seeds and
  the two-mode check of section 2.5; the grid-key hit rates of section 3.1, from saved candidate captures of three
  fixture runs (320 campaigns each; every earlier claim taken as visible at once); the unit costs of section 3.5
  (the control plane's on AC power, the ledger's on battery in high power mode); `os.rename` onto an existing name
  keeps the second writer, `os.link` refuses it; the knob placement (a key in `record` breaks
  `tests/test_sim_knobs.py:89-90`, a new section does not, and `compare_records` needs the `ADDED_KNOBS` entries);
  the digest-hook probe of section 3.3; an off-mode generations fixture run with two default-off knobs added: the
  same `results_sha256`; two identical async fixture runs (16 campaigns, stub, no plane): different
  `results_sha256`; a `ThreadPoolExecutor` job of 3 s after `shutdown(wait=False, cancel_futures=True)` held
  process exit 3.0 s.
- **StellarAI (2026-11-25 records)**: 8 x 800 campaigns with the real inner loop: 26.9 Coordinator cycles,
  106.6 model calls and 548 prompt tokens per call per campaign, `sim_bound_ratio` 0.47 (with
  `sims_executing.max` 8 of 8), `counts.feasible_agents` 0 in every launch, 453 model calls/s
  summed, agent phase 1,494-1,511 s, event-loop lag p95 6-7 ms; one process at 2,000 campaigns: lag p95 1.5 s.
- **Inferred**: every cost at 20,000 campaigns; the run length T (the README's line, a guess past 6,400
  campaigns); the k = 32 layout; the 50 bytes per campaign of a progress file; the hit rates at 20,000 campaigns;
  the latencies; the note's share.
- **Not measured anywhere**: anything of the planes themselves; the hit rate with the real model or at scale; file
  replace, `stat` and `pread` latency on the shared scratch file system under load; clock skew between nodes.
