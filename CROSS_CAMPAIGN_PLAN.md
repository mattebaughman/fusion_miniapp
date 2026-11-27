# Cross-campaign plan: a progress plane, a work tally and one entry point for a run of many campaigns

**Nothing in this document is implemented.** It is a plan. The only trace of it in the code is short comments,
each with a tag `XC-...`, at the places where the plan would attach (section 12). Every component, knob, file,
message, record field and test named below is planned unless the text says it exists.

Code references are to the `dev` branch at `000300c`. Paths without a folder are under `src/fusion_miniapp/`.
Each fact is labelled:
- **code**: read in the source at the line given;
- **laptop**: measured on a laptop on 2026-11-26 (AC power), outside the test suite;
- **StellarAI**: read from the run records of 2026-11-25 (see "The StellarAI run path" and "What has run, and
  what has not" in the README);
- **configured**: set in a knob or launch file, not run;
- **inferred**: arithmetic or reasoning, not run.

## 1. In short

1. **Two planes, one entry point per run, off by default.** No campaign ever waits on either plane. If a plane
   is slow, broken or gone, every campaign goes on as today.
2. **The Coordinators' side comes first (P0, observe).** Each Coordinator's memory already passes
   `Script.remember` at every cycle in both modes. A copy of a few of its fields goes into a per-population
   progress table. A tick replaces one small file per population in one plane folder per run. This gives what
   nothing in the code gives today: the state of every campaign while the run goes on.
3. **One entry point per run** (`python -m fusion_miniapp.entry_point`) reads those files, answers a user's
   questions about the whole run, and is the only writer of directives. It starts as a reader of files. A
   resident server and a model-backed `ask` are options on top. It is not an agent hosted in a population.
4. **The Generators' side starts as a measurement (P1), not a ledger.** For the configured 20,000 seeds, each campaign
   starts with 16 individual profiles whose six-value shape signatures are disjoint across campaigns. Descendants
   inherit a local signature, ruling out exact cross-campaign profile repeats in these search paths (section 3.1). P1 keeps one 64-byte row per evaluation in memory,
   counts repeats after the run, and checks on every repeat that equal keys gave equal results.
5. **Reuse (P2) is built only if P1's gate passes**, and is served inside the worker. A hit served at the
   Generator or the Monitor would break `verify`'s counts, and a new draw would shift the campaign's random
   streams.
6. **Steering (P3) is guidance notes only.** A note is one line before the last line of the Coordinator's
   prompt (async) or of the refill prompt (generations). Stop, budget and priority are not in this plan. Whole-run
   stop stays the launch script's SIGTERM.
7. **One transport for every way of running:** small files, each with one writer, replaced whole in the run's
   plane folder. Only files carry plane information between processes. RaDEX is not used.
8. **Every hook is guarded.** An exception inside the plane turns the plane off in that process, is counted, and
   never reaches an agent. A failed file write is only counted.
9. **The one plane cost on the resource that sets the pace** is a live note's text in prompts: about 4.6% more
   prompt tokens while a 100-token note is live (inferred from the StellarAI records, section 3.5).
10. **The initial distribution determines which cross-campaign hits are possible.** Sharing shapes or initial
    profiles changes the experiment; P1 would measure the resulting exact hits before P2 (section 11).

## 2. The shape, and why

### 2.1 What ran that decides it

| Fact | Label | Consequence for the plan |
|---|---|---|
| One local-runtime process did not drive 10,000 agents against 8 servers (event-loop lag p95 1,545 ms, no campaign finished in 6 minutes). 8 launches side by side ran 8 x 800 campaigns to the end, lag p95 6-7 ms (README, "Open items" and "The StellarAI run path") | StellarAI | Nothing may add a remote round trip, or a message per candidate or per cycle, to a campaign's event loop |
| The inner-loop workers were busy 47% of the agent phase (`sim_bound_ratio` 0.47 in each of the 8 launches) | StellarAI | Workers were busy 47% on average and saturated at peaks (`sims_executing.max` 8 of 8, `queue_wait_s.p95` 61.4 s in g0). How much time a saved inner loop saves is not measured |
| 106.6 model calls and 58,400 prompt tokens per campaign (548 per call); 89% of all tokens are prompt tokens | StellarAI (8 x 800 records) | Text added to prompts is the cost that lands on the pace-setting servers |
| 26.9 Coordinator cycles per campaign (172,134 for 6,400) | StellarAI | About 0.27 digests per model call: the servers bound the control plane |
| Each seed draws 16 individual initial profiles. Each profile carries a six-value shape signature inherited by its descendants. All 320,000 initial signatures for campaigns a0 to a19999 are distinct | code, laptop | Exact cross-campaign repeats cannot happen in the async mode at the configured shape |
| An async Coordinator sleeps in `take_all(me, "scored")` until a result is scored (`layer_autonmi.py:1343`) | code | A stuck campaign has no cycle and emits nothing. Liveness must come from the process, not the cycle |
| Nothing reads mid-campaign progress during the run, and nothing outside one launch can: the store holds each Coordinator's last memory, per launch (`layer_autonmi.py:1596`), and campaign lines are written only at a campaign's end (`:1598-1607`, `:1640-1642`) | code | P0 is new information across launches, not a copy of something that exists |

### 2.2 The two planes

```
per campaign (5 hosted agents)       per process (one plane)                       per run (the plane folder xc/)
Coordinator: _begin, remember,  -->  Table per population: columns per campaign --> progress-a<first>-<count>-
  report, on_done (XC-C1..C4)        tick on the loop; encode and write in the         <pid>-<start>.json
                                     plane's own thread                              (one per population, replaced
                                                                                     whole every tick_s)
Coordinator at its cycle        <--  directive set in memory (one assignment)   <-- directives-v<N>.json
  (XC-C5, XC-C7)                     the thread lists and reads the newest file      <-- entry point (only writer)
Monitor's resume (XC-W1)        -->  WorkKeys per population (P1 rows)          --> keys-p<i>.npz -> work_keys.npz
                                                                                     -> merge_records (after the run)
launch script (XC-E1)           ----------------------------------------------> manifest.json (expected ranges)
```

- **Control plane (the Coordinators' side).** Digest, table, file, entry point, directives. Built in P0 and P3.
- **Data plane (the Generators' side).** P1 is an offline work-key tally: no live ledger, no claims and no cache.
  A live ledger exists only in P2, which is gated on P1's numbers.

### 2.3 The entry point: one per run, a reader first

Whether one entry-point agent or a decentralised control plane should hold the run's state was left open. This
plan takes one entry point per run, built in this order:
1. **A reader** (P0a): `python -m fusion_miniapp.entry_point status --dir <plane folder>` reads the manifest and
   every progress file, and answers. Nothing resident can die. It works during the run and after it, from any
   node that mounts the run folder.
2. **The only writer of directives** (P3): `guide`, `revoke`. One writer, so directives never conflict.
3. **Options** (P0b, off unless started): `serve` keeps the same answers resident over HTTP on `127.0.0.1`, with
   a history of results per second; `ask` sends one question and the status summary to one of the run's own
   model servers in one call and returns the answer with its coverage and age. A model may draft a directive. It
   is never issued without a `guide` command from the user.

Why not an agent hosted in a population: it would run on a campaign event loop, which is the bound
(section 2.1). Why not a decentralised plane where Coordinators tell each other: no edge between campaigns exists
today (code), and every such message would land on that same loop. The plane is decentralised where the hot path
is: each process keeps its own campaigns' table and resolves directive targets itself, and nothing waits on the
entry point.

### 2.4 Where this plan departs from the first sketch of the shape

| First sketch | This plan | Evidence |
|---|---|---|
| A shared ledger consulted by the Generator before it queues; a hit reuses the result or moves the candidate; claim, then evaluate | P1 is an offline tally. Reuse (P2, gated) is served in the worker, from finished results only. No "move", no claims | A hit at the Generator or the Monitor changes `Monitor simulations == reports == messages_sent` (`layer_autonmi.py:864-865`, `:934`), the campaign totals (`:878-881`, `:953-957`), AUTONMI's `simulations_match` and `decide.collect`'s params check (`decide.py:445-452`). A new draw shifts stream 0 (async) or the shared stream (`decide.py:321`, `:664-676`) |
| A backend per way of running; RaDEX under Dragon | One transport: files in the plane folder | The store has no put-if-absent, no compare-and-set and no key scan, and its calls are synchronous on the population's loop (AUTONMI's `rascal` store). Dragon runs on one node only. The `live-p<idx>.json` files already cross processes under Dragon (`layer_autonmi.py:766-780`) |
| Directives: guidance, priorities, budgets, stop | Guidance notes only | B bounds three roles (`layer_autonmi.py:1363`, `:1389`, `:1440`) and `verify` (`:958`). No priority lever exists in the client or the servers. A lost stop sentinel would wedge a campaign |
| A stall is seen at the Coordinator's cycle | Liveness comes from the process's plane; stall from cumulative fields | A stuck Coordinator has no cycle (`:1343`). The parent replaces the best on every result while nothing is feasible (`parent/evolutionary_loop.py:1016-1022`) |
| Off means unchanged | Unchanged, and an off record keeps today's schema: with every plane knob at its default the section is left out of `rec["knobs"]`, `campaign.json` and the app's parameters | Every knob is echoed in `rec["knobs"]` (`record.py:519`) and AUTONMI's `Campaign` (built at `layer_autonmi.py:2229`) writes `definition.knobs` into `campaign.json` and the app's parameters. So shared `record.write` and the `Fusion` definition drop the all-default section; the plane reads its knobs before that |
| (not stated) | Every hook is guarded | A raise in `Script.remember`, a Monitor hook or a tick would fail that role and with it the campaign (README, "When one agent fails") |

## 3. Answers to the design questions

### 3.1 What makes two evaluations "the same work"

- **CPU stand-in comparisons (laptop).** Across the tested processes, thread counts, seeds, candidate identities
  and cold starts, the scalar science fields agreed for a fixed complete profile and effective sim knobs. A
  candidate index, trace presence, physical touches, stage times and paths belong to the individual execution
  (`sim.py:249-251`, `:260-289`). A supplied full `params` dict passes through at `sim.py:160-166`. The seed fills a
  partial candidate at `:167-180`. The stand-ins are seeded with 0 (`standins.py:122,137-138`).
- **Conservative key inputs.** Keep the 15 typed params (seven mutable, six inherited shape values, `model` and
  `kind`). The fixture hashes `3` and `3.0` differently (`sim.py:321`). Distinct keys can produce equal outputs:
  the real worker sets model and kind at `sim.py:267`, and the parent applies them at
  `parent/steps/step0/profile_generator.py:533-534`. Other key inputs include the sim
  knobs after the cost scale `nodes_per_plane, planes, rollout_steps, puffs, fixed_point_cap, model_width`
  (`decide.py:116-121`); for the trace the Monitor scores, `monitor.capture` and, with the fixture,
  `monitor.fixture_trace` (smooth and hash traces differ with equal scalars); the device (mps against cpu differ at
  about 5e-8 relative; cuda not measured); real against fixture; the code.
- **The work key** (P1): `key = blake2b(ns + json.dumps(params, sort_keys=True).encode(), digest_size=8)` as an
  unsigned 64-bit integer. The params bytes are the ones the fixture already hashes (`sim.py:321`, XC-R2). `ns =
  blake2b(json.dumps({"v": 1, "inner_loop": "real"|"fixture", "sim": <those six values after the cost scale>,
  "monitor": {"capture", "fixture_trace"}, "device": "cpu"|"cuda"|"mps"|"none", "miniapp": <commit>, "parent": <lock
  commit>}, sort_keys=True), digest_size=16)`, per effective cost scale and runtime contract (XC-W4). The two `monitor` knobs
  are in it because P2 returns the stored trace and the Monitor scores it (`layer_autonmi.py:1223-1226`; async score
  q + `monitor.weight` x m, `decide.numbers_score`). The P1 scalar tally omits seed, campaign id,
  generation, candidate number, cold start, `work_root`, `threads_per_worker`, `checkpoint_mb`, `keep_artifacts`,
  `campaign.slots`, and every `agents`, `llm` and `outer` knob. Collision odds among 640,000 keys of 64 bits: about
  n^2 / 2^65 = 1.1e-8 (inferred). P2 uses 128-bit keys and validates the consumer's typed input and execution
  contract at lookup (section 4.7).
- **The result digest (P1).** Feed the sorted JSON bytes of the scalar result without EXEC_FIELDS to blake2b-8.
  XC-W1 receives the scalar `result` and the separate `tr` from `trace.split([event])` (`trace.py:238-242`).
  Append a presence marker. For a trace, also digest its kind, completeness, each non-timing section's name,
  dimensions and little-endian float32 values in declared order. This covers the Monitor's `xgc` and
  `fixed_point`, and the `handoff` and `scan` that P2 returns. Timing bytes describe the producer.
  Equal-key rows with differing digests are failed twins.
- **Tolerance for reuse is zero.** On the laptop, two observed cross-campaign pairs from the async fixture run were
  reevaluated through the real CPU stand-in at the hundredk shape. Wall electron temperatures differed by 7.73 eV
  for the nearest eligible pair on all 13 numbers and 9.63 eV for the nearest on the seven tunables. These observations support using exact typed matches; they establish no wider tolerance.
- **Why the configured campaigns cannot repeat each other's exact profiles.** Campaign a<i> uses seed i
  (`layer_autonmi.py:729-735`). The parent asks for P Latin-hypercube points at that seed
  (`parent/evolutionary_loop.py:435-454`). The sampler creates one profile dictionary per point
  (`parent/steps/step0/profile_generator.py:429-445`). Each initial profile carries six inherited coordinates:
  `a3`, `a4`, `bcore_Te`, `bcore_ne`, `core_linear_frac_Te` and `core_linear_frac_ne`.
  A local run of `initialize_population(16, ...)` at seeds 0 and 1 returned 16 distinct six-tuples per seed,
  with 32 distinct across their union. Over the configured seeds 0 through 19,999, all 320,000 initial
  six-tuples were distinct (laptop enumeration). Slice offsets preserve those campaign IDs.
  A survivor retains its profile. Mutation copies a survivor and alters only seven mutable coordinates
  (`parent/evolutionary_loop.py:1046-1063`); model proposals can replace only those seven
  (`decide.py:263-269`, `:323-333`, `:890-905`). Thus every descendant inherits one signature from its
  campaign's initial set. Full typed inputs from two campaigns cannot match on this configuration in either
  mode, even when the seven tunables are close. The restart branch of `repopulate` (`:1088-1089` in the parent) needs
  an empty survivor list, which a completed generation never gives: `select_survivors` keeps rank 0 whenever the
  population is not empty (`:959`, `:970`, `:977`), and `decide.collect` requires all P results
  (`decide.py:445-448`).
- **Where repeats are real:** inside a campaign in the generations mode. Survivors go into the next population
  unchanged (`decide.py:313-320`) and every member is queued again (`layer_autonmi.py:1199-1200`). Laptop: 311 of
  1,280 proposals (24.3%) in 40 fixture campaigns. Three further fixture runs each had 320 campaigns and
  10,240 proposals. Generations/stub had 2,531 prior own-campaign exact matches, async/stub and async/model-off
  zero. All three had zero exact matches to a different campaign completed before proposal. With equal categories
  and all seven tunables within 10% of their allowed ranges, the prior-other counts were 204, 379 and 291. With
  all 13 numeric fields at 10% they were zero; at 20% they were 50, 52 and 57. These are laptop measurements.
  The StellarAI aggregates record how many values the model filled but cannot reconstruct candidate-level proximity.
- **A reused record would need rebinding** (P2): `candidate_idx` (checked by `decide.collect`,
  `decide.py:445-448`), and `t_start`, `elapsed_seconds`, `touches`, `touch_check`, `cold_start`,
  `worker_peak_rss_mb`, `summary_path`, which describe the original run. `device` stays the producer's, because
  the numbers came from there.

### 3.2 Where a Generator creates and queues a candidate, and what it knows there

- **Generations mode**, `Script._generator` (`layer_autonmi.py:1181-1204`). `decide.generator_step`
  (`decide.py:304-336`) draws the initial population (`:308-312`) or repopulates (`:313-321`) and, with
  `outer.generator: llm`, fills the non-survivor slots by the refill call (`:323-333`; prompt
  `decide.refill_messages`, `:233-254`, its ask on the last line at `:252-253`). The population goes to the
  Coordinator at `layer_autonmi.py:1197` (XC-G1) and P items go on the queue at `:1198-1200`, each
  `{id: g<g>c<i>, params, candidate_idx, generation, t0}` (`decide.py:341-345`). Known there: the campaign, its
  seed, the survivors' full params, the rng state, the previous generation's results and the sources. The seed
  and the source are not on the item.
- **Asynchronous mode**, `Script._generator_async` (`layer_autonmi.py:1373-1412`). It waits for an empty queue
  (`:1390`), takes survivors, digests and guidance without waiting (`:1391-1393`), and waits only for its first
  survivors (`:1394-1396`). `decide.async_profiles` (`decide.py:655-680`) makes TAKE = 2 profiles: the k-th
  initial profile for k < P, else `mutate_params` of a survivor drawn on stream 0. `Script._reconcile`
  (`:1414-1429`) applies the model's values. The final params exist only after `:1399`; they go on the queue at
  `:1402-1403` (XC-G2) as `{id: c<k>, params, candidate_idx, generation, source, filled, t0}`.
- **What the plan does there: nothing but a note.** XC-G1 and XC-G2 say why the plane does not act at the
  Generator (section 2.4). The one place both modes pass with the final params and the result is the Monitor's
  resume (`layer_autonmi.py:1215-1220`): `result["params"]` holds the params (the inner loop echoes them), and
  `call["t0"]` the queue time. P1 counts there (XC-W1). `Mailbox.put` makes a JSON copy of each item
  (`layer_autonmi.py:357`), so the queued dict is never shared.

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
| Call sums `CallSums` (`:535-595`), written at the population's end and on SIGTERM (`:802`, `:1665`) | the pattern and the places | P1 rows are written beside them (XC-A2, XC-A3) |
| `record.write` and `write_record` (`record.py:350-565`; `layer_autonmi.py:2699-2794`) | both: shared raw knob serialization in `record.write`, AUTONMI report sections in `write_record` | The `cross_campaign` section is added in `write_record` (XC-K4, XC-W2). Observe-only adds no scientific field to campaign rows; scheduling can affect free-running async outputs |
| `merge_records` (`merge_records.py:159-239`) | yes | Sums the sections of the launches it reads (XC-K5) and counts repeats across launches within equal namespaces (XC-W3). Its `_validate` already refuses launches whose non-machine knobs or commits differ (`:92-118`). It stops on a launch folder with no `run_record.json` (`:163`, `_read_launch` `:75`), so coverage against the manifest is the entry point's (`status`, `populations`) |
| The launch script's folders (`run-<i>/`, `launch/stellar_ai_autonmi.sbatch:289`) and node-local `TMPDIR` (`:407`) | yes | It writes the manifest and passes `--xc-dir` (XC-E1). P2's ledger would live under `TMPDIR` (XC-R7) |
| Flowgentic's task key (`ToolRuntime.submit`) | no | Its key is agent, scope and call: an idempotence guard for one agent's resume, not a shared memo |

### 3.5 What each plane costs at 20,000 campaigns (inferred unless marked)

Inputs. Configured: N = 20,000 campaigns (the StellarAI scale knob file, README "The StellarAI run path"),
B = 32 (P 16 x G 2), so E = 640,000 evaluations, 2 Monitor slots, 100,000 hosted agents. StellarAI (8 x 800, real
inner loop): 26.9 Coordinator cycles, 106.6 model calls and 58,400 prompt tokens per campaign; 453 model calls/s
summed on one node. Run length T: the README's line 100 s + 0.22 s x N = 4,500 s, which the README calls a guess
past 6,400 campaigns. Unit costs (laptop): a digest's list writes 0.08 us; building a digest dict 0.64 us; a work
key 5.0 us; a result digest 5.9 us; a row append 0.07 us; a progress document for 10,000 campaigns 3.8 ms to
encode (310 KB, 7 columns), 0.47 ms to write, 1.8 ms to parse; for 625 campaigns 0.24 ms; an offline repeat count
over 640,000 rows 128 ms; merging one remote 64-byte ledger record 0.38 us.

**Control plane (P0, P3):**

| Quantity | Arithmetic | Value |
|---|---|---|
| Digests per run | N x (26.9 cycles + start + end) | 578,000 (at most N x 34 = 680,000) |
| Digest rate | 578,000 / 4,500 s | 128/s run-wide |
| Bound by the servers | 28.9 digests / 106.6 calls per campaign = 0.27 per call; x 453 calls/s | about 123/s on one model node |
| Loop work for digests | 578,000 x 0.08 to 0.64 us | 0.05 to 0.37 s per run, all processes together |
| Files replaced | one per population per `tick_s` (10 s) | 0.2/s (one launch of 2 populations), 3.2/s (k = 32), 4/s (Dragon, 40 populations) |
| Bytes written | about 45 B per campaign per file (10 columns; 31 B measured for 7) x N per 10 s | about 0.9 MB per 10 s run-wide, 90 kB/s, in every layout |
| Encoding, in the plane's thread | 3.8 ms per 10,000 campaigns at 7 columns, about 5.5 ms at 10 | one launch of 20,000: about 11 ms per 10 s (0.1%); k = 32: about 0.35 ms per launch per 10 s |
| Snapshot on the loop, per tick | copies of 10 columns of a table's lists | about 0.3 ms per 10,000 campaigns (a guess, not measured) |
| Memory | about 100 B per campaign | 2 MB run-wide |
| Directive check per cycle (P3) | one lookup and a time compare, under 1 us | under 0.6 s per run in all |
| Directive file reads (P3) | one listing of `directives-v*.json` per process per tick, in the thread; a read only on a new version | 3.2/s at k = 32 |
| Entry point, `status` | parse every file: about 0.9 MB | about 5 ms per answer (from 1.8 ms per 310 KB) |
| **A live note in prompts (P3)** | Coordinator calls only: 26.9 x 100 tokens = 2,690 per campaign, over 58,400 | **4.6% more prompt tokens** while the note is live (53.8 M of 1.17 G per run) |
| (not planned) The note also in the Generator's prompts, 150 tokens | (26.9 + 16) x 150 = 6,435 per campaign | 11.0% |

The note is the only cost that lands on the model servers, which set the pace. While a 100-token note is live it
adds 4.6% more prompt tokens (inferred). Its effect on run time is not measured: prefill and decode cost differ
per token, and a note can change reply lengths and later work. P3's exit check measures it. That is why
`note_chars` defaults to 400 characters (about 100 tokens at 4 characters per token, inferred) and v1 shows notes
to one prompt per cycle: the Coordinator's guidance prompt in the async mode, the Generator's refill prompt in
the generations mode.

**Data plane, P1 (the tally):**

| Quantity | Arithmetic | Value |
|---|---|---|
| Messages between processes during the run | | 0 |
| Rows | E x 64 B | 41 MB per run; 1.3 MB per launch at k = 32 |
| Loop work per evaluation | key 5.0 + result digest 5.9 + append 0.07 us | about 11 us, excluding the trace digest: 7 s per run in all; 0.22 s per launch at k = 32 (0.005% of 4,500 s); 0.16% of one local launch's loop |
| Count after the run | 128 ms per 640,000 rows | under 1 s in `write_record` and `merge_records` |

**Data plane, P2 (only if built; one log per writer on node-local disk, synced every 2 s):**

| Quantity | Arithmetic | k = 32 |
|---|---|---|
| Index per process | 142 B per entry (laptop) x 640,000 | 91 MB per process; 2.9 GB per node (0.3% of the 950,000 MiB the job asks for, `launch/stellar_ai_autonmi.sbatch:97`) |
| Log bytes | 1 DONE record x 64 B x E written; read by each other writer | 41 MB written, 1.3 GB read per run |
| File reads | writers x (writers - 1) / 2 s | 496/s |
| Merge work on each loop | each process merges every other writer's records: 620,000 x 0.38 us | 0.24 s per process per run, about 12 us per own candidate |
| Result files | about 3.0 kB per evaluation (record 1.6 kB and trace 1.4 kB, laptop) | 1.9 GB per run on node-local disk |

### 3.6 Failure and consistency

| Case | What happens | Why it is safe |
|---|---|---|
| A plane hook raises | The guard catches it, counts it under its site, turns the plane off in that process (`disabled_at`), writes one warning to stderr | No exception reaches an agent. The campaigns go on as with the plane off |
| Lost ledger | P0, P1 and P3 have none. A crashed population's P1 rows are lost; `work_keys.coverage.missing` names it. In P2 a lost or unreadable ledger gives misses only | No campaign reads P1. A P2 miss runs the inner loop, as today |
| Two writers of one key (P2 only; there are no claims) | P2 acts only on a finished (DONE) result, so a candidate still in flight in another campaign is evaluated again. Results are published with `os.link` on the final name, which refuses an existing name (laptop: `FileExistsError`; `os.rename` and `os.replace` keep the last writer instead). A late writer's result is counted as a twin, equal or differing, and never replaces the first | The first published result stays. A stale read can repeat work. A differing twin disqualifies its key/runtime and flags any earlier consumption for review |
| Dead entry point | In P0a there is none to die: the reader runs on demand. A dead `serve` stops answers only. Directives already issued apply until `expires_t`, then lapse | Campaigns never read the entry point; they read their process's in-memory copy of the newest `directives-v<N>.json` |
| A process dies or blocks | Its files stop changing. After 3 x `tick_s` without an `end` mark the reader shows its population `lost` and its open campaigns `unknown`. The entry point shows the gap against the manifest | Each population's file has one writer; no other file depends on it |
| A stuck campaign | No cycle, so no digest. The reader flags it `quiet` after `quiet_s` and shows its liveness detail: inner loops or Monitor calls in flight, model calls answered (section 4.4) | Read from the process, which still ticks |
| A directive arrives late | It is prospective: a campaign shows it from its next cycle on. The reader can derive `closed` from a durable terminal observation before issue; an exited population cannot send a later ack | A note cannot change a result already scored, the survival rule, the budget, or stop a campaign. It can change the candidates the Generator's model proposes from then on (async: `decide.async_on_guidance`, `decide.py:636-638`, then `Script._reconcile`; generations: the refill reply, `decide.py:323-333`), and through them later scores and survival. Late means less steering, never a wrong result |
| Two launches disagree | Each launch is meant to own a disjoint campaign range; nothing enforces it during the run. Each progress file's name carries its writer's pid and start time, so one node can identify distinct process attempts; P4 also requires a host-unique writer ID. The reader checks every file's `run_id` and range against the manifest and shows overlapping ranges and duplicate writers instead of summing them (`merge_records` refuses overlaps after the run, `merge_records.py:170-172`). The only shared input is the directive set, one writer, a version that only grows, set semantics. A delayed reader may skip multiple versions. Show each population's `version_read`, snapshot time and pending target count. `merge_records` counts cross-launch repeats within equal P1 namespaces only and reports the namespaces it could not compare; it never refuses a merge for them (a launch on another device merges today, `merge_records.py:112`) | No shared mutable state is reconciled |
| Half-written or corrupt file | Writers use tmp, then `os.replace`. Readers skip a file they cannot parse and keep the last good one, as the head does with `live-p<idx>.json` | As the handed files today |
| Shared file system slow or full | The thread's write takes long or fails. A tick whose previous write has not finished is skipped and counted. An `OSError` from a write is caught in the thread and counted in `write_errors`; the plane stays on. The table keeps the latest state, so the next write catches up. A write that hangs holds only the plane's daemon thread, which is never joined at exit | The loop never waits on I/O, and the process exits as today |
| SIGTERM | Local runtime only: `Script.cut_short` (XC-A3) queues the final partial snapshots; the process gives all such writes one shared 2 s monotonic deadline. Without a successful final write, preserve last-observed facts and show unresolved outcomes as unknown. The record is `interrupted`, exit 143, as today. Under Dragon no flush runs (the head calls `cut_short` only for the Scripts in its own process, `layer_autonmi.py:2254-2274`): the last progress file is the last tick's, the reader shows the population `lost` and its open campaigns `unknown`, and that population's P1 rows are lost | Same path as the call sums (StellarAI ran it locally: interrupted, exit 143) |
| Clocks on several nodes (P4) | Ages are taken against each file's own `t`, written by its process, and the reader's clock. Skew between nodes shifts ages | Not measured; P4's exit check measures it |

### 3.7 What the record and the README would say, and the tests that pin "off means unchanged"

Section 9.

## 4. The design

### 4.1 Components and cardinalities (configured shape: 20,000 campaigns, async, P 16, G 2, B 32, 2 slots)

| Component | Module (planned) | Lives in | Count | Talks to |
|---|---|---|---|---|
| Hooks | `layer_autonmi.Script` (XC-C1 to C5, C7, W1) | the campaign's event loop | 20,000 campaigns | its process's plane, by a guarded method call |
| Plane | `plane.Plane` | one per process: per local launch (all its populations), per population process under Dragon | 1, k (`FUSION_LAUNCHES=k`) or P (Dragon) | its tables, its thread, the plane folder |
| Progress table | `plane.Table` | inside the plane | one per population | written by the hooks on the loop, copied at the tick |
| Plane thread | one daemon `threading.Thread(name="xc", daemon=True)` fed by a one-slot `queue.Queue(maxsize=1)`, never joined at exit (a `ThreadPoolExecutor` is joined at interpreter exit, so a hung write would hold the process: laptop, 3.0 s for a 3 s job) | one per plane | 1 per process | encodes and replaces files; lists and reads the directive files |
| Work-key rows (P1) | `plane.WorkKeys` | inside the plane | one per population | written at XC-W1 on the loop; saved at XC-A2 |
| Plane folder | `<out>/xc`, or `--xc-dir` | the run folder (shared scratch on the machine, `launch/stellar_ai_autonmi.sbatch:335`) | 1 per run | written by the planes, the launch script and the entry point |
| Entry point | `entry_point` | a process of the user's, on demand; `serve` optional | 1 per run (any number of readers) | the plane folder; optionally one model server |

### 4.2 The plane in one process

- **Created** by the first `Fusion.build` in a process when any plane knob is on (XC-A1); `Fusion.plane` is
  `None` otherwise, and every hook is then a single `if self.plane is not None:` test.
- **The guard.** Every hook calls `plane.guard(site, fn, *args)`. It catches `Exception`, adds 1 to
  `errors[site]`, sets `plane.off = True` and `disabled_at`, and writes one warning to stderr. Once `off`, every
  hook returns at once. Sites: `begin`, `remember`, `report`, `failed`, `row`, `directive`, `tick`, `write`. At
  `write`, an `OSError` is caught in the thread first, counted in `write_errors`, and the plane stays on; only
  other exceptions reach the guard.
- **On the event loop**, and only there: the hooks' list writes; the tick's snapshot; reading the directive set
  (an attribute the thread replaces by one assignment, read without a lock).
- **The tick** runs every `tick_s` on the loop, as a task started once per process. It records how late it woke
  (`tick_late_ms`, a coarse loop-lag reading per process). For each table it copies the columns and the roll-up
  counters, and reads the liveness detail of the campaigns the previous snapshot found quiet (at most 50 per table,
  section 4.4). It hands the copies to the plane's thread through the one-slot queue (`put_nowait`). If the slot is
  still full, the previous write has not finished: it skips this write and counts it. The tick writes every table
  every time, so a file's age is its process's heartbeat.
- **In the plane's thread**: encode each column with its own `json.dumps` call (so the thread never holds the
  interpreter lock for long: about 0.5 ms per column of 10,000, inferred), write `progress-...json.tmp`,
  `os.replace` it, list `directives-v*.json`, and read and parse a file only when its N is higher than the last
  one that parsed. The thread also finds the next quiet list from the copies. It never touches an object the loop
  owns.
- **Ending**: `Fusion.rows` (XC-A2) writes the population's last file with `"end": true` and its P1 rows; the
  process's last population stops the tick and the thread. `Script.cut_short` (XC-A3) does a bounded flush on
  SIGTERM, in the local runtime only (section 3.6). The head's `miniapp_workflow` closes the plane in its
  `finally` (XC-A4). `flush(timeout_s)` hands the last write to the thread and waits on an event with that
  timeout; on timeout it abandons the write, counts it, and returns.
- **Off**: no plane, no thread, no folder, no file, no new record key.

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
running, `d` done, `f` failed, `i` interrupted (set by the SIGTERM flush). `dir_v` (P3) is the newest directive
version the campaign applied.

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

**None of the 6,400 measured campaigns triggered this feasible-only stall rule.** `counts.feasible_agents` is 0 in all 8 launches of the
8 x 800 real run (StellarAI), so `feas` is false everywhere and `infeasible` flags every campaign after `S_R`
results in that run. The entry point cannot detect an improvement plateau during this infeasible phase. In the async mode the
survivors are kept oldest first, not ranked (`decide.async_on_scored`), so no column of the digest shows an
infeasible campaign improving. A rule for that phase (for example, results since the window's best wall-blend
rank last improved, a new column computed at the cycle) is open decision 7.

Liveness detail for a quiet campaign, read on the loop at the next tick (at most 50 per table): in flight =
the sum over its Monitors of `Script.sims[m] - Script.reports[m]` (both modes: `sims` counts at the take,
`layer_autonmi.py:1248`, `reports` after the report, `:1242`; it includes the Monitor's own note call); in the
async mode also `Script.running[campaign]` (`:1228`, `:1256`), the inner loops alone; and
`Script.answered[campaign]` (`:1618`), its model calls answered. These are observations at the sample time.
A flat answer count cannot distinguish an outstanding first model call from a blocked role. Report that cause
as unknown. An unchanged result count over a fresh observed interval supports `no_progress`. A one-shot
reader lacking comparable history reports insufficient history.

### 4.5 Directives (P3)

- **Shape and writer.** Version N contains the full directive set. Each `guide` or `revoke` computes its
  update against the latest validated version for this run, writes it into a fresh private inode (exclusive
  creation), closes it and links it to the N+1 final name. Unlink only that private name. On collision,
  reread the winner and recompute the update. Committed versions stay immutable. Report an unreadable final
  version; set a bounded retry deadline. Readers can miss intermediate versions while their I/O is delayed.
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
  campaign actually dispatches a request carrying the version, record one `applied` ack, or `not_applicable` when `llm.mode` is off
  (no prompt exists). `async_memory` and `async_finish` pick named keys (`decide.py:788-791`, `:794-803`), so
  nothing else sees the note.
- **Applying, generations** (XC-C7, XC-C8): `:1173` sends both `initialize` (g = 0) and `repopulate`. A live
  note rides only on a repopulate message, as `msg["note"]`; `refill_messages` shows it as one line before the
  ask. The ack is `applied` only when a request carrying the note was sent. Otherwise it is `not_applicable`:
  with `outer.generator: mutation`, with no non-survivor slot to fill (`decide.py:325-326`), or when the call
  was refused before dispatch. A sent request that later fails still counts as dispatched; record its fallback.
- **Closed and missing targets.** Derive `closed` only from terminal evidence predating this issue. Targets
  that expired, were revoked before their next poll, or lack a fresh writer stay unconfirmed unless their
  disposition was observed. Keep the last-seen version and sample time. A later version never proves that an
  earlier version was applied.
- **Provenance.** With `steer` on, a campaign line gains `directives: [{"id", "v", "at_n", "cycle"}]` only when
  the campaign applied one (added at XC-C3). That key changes `results_sha256` on purpose: a steered campaign is a
  different experiment.
- **What a note cannot do:** change a result already scored, the survival rule, the budget, or stop a campaign.
  It can change the candidates the Generator's model proposes from then on, and through them later scores and
  survival (section 3.6).
- **Prompt room.** The longest prompts at the StellarAI shape are about 6,700 characters (Coordinator) and 5,200
  (Generator) against the bound of 21,504 characters in `tests/test_async.py:336` (laptop). A 400-character note
  fits.

### 4.6 The work-key tally (P1)

- **At XC-W1** (the Monitor's resume, both modes): `WorkKeys.add(ns, campaign, call, result, tr, now)` receives
  both outputs of `trace.split`. It computes the key and section 3.1's scalar and trace digest and appends one row.
  The section 3.5 estimate of 11 us uses the measured scalar-only parts. Measure the additional trace work with
  capture on at the intended shape before accepting the loop-time budget.
- **Row (64 B)**: `key` u8, `digest` u8, `campaign` u4 (i of a<i>), `candidate` u2 (`candidate_idx`), `generation`
  u2, `t_queued` f4 (`call["t0"]` minus the table's `t0`), `t_back` f4 (now minus `t0`), `sim_s` f4
  (`result["elapsed_seconds"]`), `seven` f4 x 7. Columns are `array` objects in memory, grouped by namespace. The file header maps each group to its namespace.
- **Saved** at the population's end (XC-A2) or on SIGTERM in the local runtime (XC-A3; under Dragon the interrupted
  populations' rows are lost) as `keys-p<index>.npz` beside the call sums, with a header: `{"v": 1, "t0", "first",
  "count", "namespaces": [{"ns": "<32 hex>", "inner_loop", "sim", "monitor", "device", "miniapp", "parent"}]}`.
- **Counted** in `write_record` (XC-W2): within each namespace, rebase each population's elapsed timestamps to a common epoch before sorting by key and completion time. Each row after the first of its
  key is a **repeat** and gets two flags: an earlier row of the key from its own campaign (`same_campaign`), and an
  earlier row of the key from another campaign (`other_campaign`). A row can carry both, and each scope counts the
  rows with its flag. An **ideal hit** is the same with "earlier" meaning back before this row was queued (`t_back`
  of that row < `t_queued` of this one): what a ledger with no delay, consulted at the queue, could have served.
  A worker-dispatch lookup can see a completion between queue and dispatch, or miss a DONE still waiting for
  synchronization. Measure actual P2 visibility separately.
  `would_save_sim_s` sums the ideal hits' own inner-loop seconds. The **twin check** compares each repeat's digest
  with the first row's: `twins.equal` or `twins.differ`. The rows are kept as one `work_keys.npz` and the
  per-population files are removed with the call sums.
- **Across launches** (XC-W3): `merge_records` concatenates the launches' `work_keys.npz` (times made absolute by
  each header's `t0`) and counts repeats within equal namespaces only; it reports the namespaces it could not
  compare and never refuses the merge for them. It reports `other_launch`, a subset of `other_campaign`: rows
  that have at least one qualifying predecessor from another launch. At finite timestamp precision, a tie does not establish strict-before order. Cross-node clock uncertainty can leave additional hits unresolved.
- **The gate to P2.** Build P2 only if, in a run of the shape someone intends to run, the merged
  `ideal_hits.other_campaign` (which includes `other_launch`) is at least 1% of the evaluations **and** the lowest
  launch's `speed.sim_bound_ratio` is at least 0.8 (the merged record gives that figure as a spread,
  `merge_records.py:365`). 0.8 is a chosen threshold, not a measured one: below it the workers are idle much of the
  time, and how much time a saved inner loop would save is not measured. On the 2026-11-25 records the second half
  fails (0.47 in every launch, StellarAI) and the first is expected to be 0 (section 3.1). At 20,000 campaigns on 80
  workers the README infers that the workers may set the pace ("The StellarAI run path"), so the second half is
  measured on that shape, not taken from 8 x 800. The repeats within a campaign in the generations mode (24%) are
  IGNOS's own behaviour; serving them is decision 2 of section 11.

### 4.7 Reuse inside the worker (P2, only if the gate passes)

- **Ledger.** One `plane.Ledger` per process: an index `dict[bytes, int]` from a 128-bit key to a row of a numpy
  table (state, device, writer, campaign, candidate, time, inner-loop seconds, result digest), touched only on
  the loop. One append-only log per writer (`ledger-<writer>.log`, 64-byte DONE records only) and
  one folder of result files, under `$TMPDIR/xc-<manifest run id>/` on node-local disk (XC-R7). The plane's thread
  appends its own records and reads the others' every `sync_s` (2 s), and hands them to the loop with
  `loop.call_soon_threadsafe`.
- **Publish** at XC-W1: each result and its trace go to a fresh `<key>.<writer>.<n>.tmp` (opened with
  `O_EXCL`), which is closed, then linked to `<key>.json` with `os.link`, which refuses an existing name. The
  temporary name is removed whether the link succeeded or not, so no later write can truncate a published
  result through a shared inode. Compare a losing attempt with the winner. A conflicting scientific twin
  disqualifies the key/runtime from further reuse when observed. Append DONE only after a verified winner
  exists. Record its producing attempt and run id; a crash before DONE leaves a cache miss.
- **No claims.** The lookup acts only on a DONE record, so a claim would change nothing. A candidate in flight in
  another campaign is evaluated again and counted as a twin (section 3.6). No compare-and-set is needed, so plain
  files work.
- **Lookup at take time** in the task tool's `prepare` (XC-R3, `layer_autonmi.py:1098-1103`): a fresh index read
  when the runner submits the Monitor's `Simulate`. With opt-in `reuse: other`, a DONE from a different campaign
  adds `reuse={"key", "path"}` beside `knobs` and `monitor`. The configured default remains `off`.
- **In the worker** (XC-R1 `sim.evaluate`, XC-R2 `sim.evaluate_fixture`; `evaluate_placed` calls `evaluate`):
  compare the sorted JSON bytes of the full typed params, effective sim and trace configuration, and versioned
  evaluator, input assets, numerical runtime and observed device class. Validate the stored payload checksum
  and required trace sections/completeness. Recompute on missing or mismatched fields or if `keep_artifacts`
  requests files that cannot be supplied under this candidate's identity. Otherwise return the rebound result
  with `touch_check: "reused"`, zero touches and `reused: {"key", "campaign", "candidate", "sim_s"}`. The consuming
  Monitor scores its trace locally. Only the producer reports physical stage times, touches and cold starts.
  On a refused hit evaluate normally and record `reuse_refused`. The fixture needs its own site: with
  `--sim fixture` the tool runs `sim.evaluate_fixture` (`run.py:289-292`, used at `layer_autonmi.py:1095`).
- **Counts.** Each candidate stays one engine task, so `simulations_match`, the Monitor invariant and
  `each_candidate_evaluated_once` hold. `decide.on_evaluated` (XC-R5) and `decide.async_on_scored` (XC-R6) leave
  reused results out of the touch verdicts and count them. `record.write` (XC-R4) gives `touch_check.reused`, so a
  reuse is never a touch-check failure (`record.py:537-538`, `:558-559`). The record gives `logical` (the
  evaluations the campaigns consumed, `counts.sims` as today), `physical` (inner loops run) and `reused` apart.
  The campaign line gains `n_reused`.
- **The async mode** can change order when a reused result returns sooner. Check budgets and each run's own
  replay. For exact paired scientific comparisons, replay identical arrivals, model replies and fallbacks.

## 5. Interface (planned)

```python
# src/fusion_miniapp/plane.py: standard library and numpy only
VERSION = 1
EXEC_FIELDS = ("candidate_idx", "params", "t_start", "elapsed_seconds", "worker_peak_rss_mb", "touches",
               "touch_check", "cold_start", "device", "summary_path", "trace")

class Plane:                                        # one per process (Fusion.plane), made by the first build
    def __init__(self, folder: Path, knobs: dict, run_id: str, mode: str, code: dict, device_kind: str) -> None
    off: bool                                       # set by guard on the first error in this process
    def guard(self, site: str, fn, *args)           # every hook: an exception is counted and turns the plane off
    def attach(self, script) -> "Table"             # XC-A1: one table (and WorkKeys) per population
    def start(self, loop) -> None                   # once per process: the tick task and the xc thread
    def note_for(self, campaign: str, now: float) -> "Note | None"   # XC-C5, XC-C7: from memory, no I/O
    def ack(self, campaign: str, version: int, outcome: str) -> None # actual dispatch | known inapplicability; reader derives closed
    def population_ended(self, table: "Table") -> None   # XC-A2: last file; the last one stops the plane
    def flush(self, timeout_s: float = 2.0) -> None      # XC-A3 queues local rows; the 2 s deadline is shared at XC-A4
    def section(self) -> dict                            # XC-K4: the record's cross_campaign section

class Table:                                        # one population's campaigns; index int(c[1:]) - first
    def __init__(self, first: int, count: int, budget: int, P: int, mode: str, t0: float) -> None
    def begin(self, campaign: str, now: float) -> None                       # XC-C1
    def remember(self, campaign: str, memory: dict, now: float) -> None      # XC-C2
    def end(self, campaign: str, state: str, now: float, error: str = "") -> None   # XC-C3 "d", XC-C4 "f"
    def snapshot(self, script, quiet: list[str]) -> dict                     # on the loop, at the tick

class WorkKeys:                                     # P1: one population's rows
    def add(self, ns: bytes, campaign: int, call: dict, result: dict, tr: dict | None, now: float) -> None   # XC-W1
    def write(self, path: Path) -> None                                      # XC-A2, XC-A3

def namespace(sim: str, sim_knobs: dict, monitor: dict, device_kind: str, code: dict) -> bytes   # 16 B, each scale
def work_key(ns: bytes, params: dict) -> int        # blake2b-8 of ns + json.dumps(params, sort_keys=True)
def result_digest(result: dict, tr: dict | None) -> int  # blake2b-8 of scalar projection and non-timing trace
def repeats(files: list[Path], launches: list[dict] | None = None) -> dict   # XC-W2, XC-W3

# src/fusion_miniapp/entry_point.py
def gather(folder: Path) -> "Run"                   # manifest, every progress file, directives; tolerant reads
def status(run, now: float) -> dict                 # every answer carries as_of, coverage and age
def campaigns(run, flag: str | None = None, order: str = "best", limit: int = 100) -> list[dict]
def campaign(run, cid: str) -> dict
def guide(folder: Path, text: str, targets: dict, ttl_s: float = 1800.0) -> dict   # P3: the only writer
def revoke(folder: Path, directive_id: str) -> dict
def ask(run, question: str, endpoints_file: Path, api_key_file: Path | None) -> dict   # optional
def serve(folder: Path, bind: str = "127.0.0.1", port: int = 0) -> None               # optional
def main(argv=None) -> int   # status | campaigns | campaign | best | populations | repeats | directives
                             # | guide | revoke | ask | serve; all take --dir; --json on every reader
```

Changed signatures (planned): `decide.guidance_messages(cst, rows, knobs, note="")` and
`decide.refill_messages` reading `msg.get("note", "")`; `sim.evaluate(..., reuse=None)` and
`sim.evaluate_fixture(..., reuse=None)` (P2). With the defaults, every request body and result is unchanged.

## 6. Data shapes (planned)

**`xc/manifest.json`.** Before launch, the supervisor seeds a fresh run ID and the complete set of expected
slice and population ranges. A single hand launch can seed a new folder. Hand launches sharing a folder
must receive the same expected run ID and layout (an existing folder with the same shape can be stale).
A creator closes a fresh private temporary inode before linking to the final name, then unlinks its private
name. A losing creator validates the winner's ID and complete layout. A corrupt winner produces an explicit
coverage error after a bounded read; no creator can overwrite it. Multi-slice hand launches supply this run ID in
`--xc-run-id`; a single new run creates one and the supervisor passes it to each ordinary slice.

```json
{"v": 1, "run_id": "<16 hex>", "mode": "async", "budget": 32, "population": 16, "generations": 2,
 "campaigns": {"first": 0, "count": 20000}, "launches": [{"slice": [0, 8], "first": 0, "count": 2500, "population_ranges": [[0, 1250], [1250, 2500]]}, "..."],
 "runtime": "local", "written_t": "<float>", "by": "launch_script"}
```

**`xc/progress-a<first>-<count>-<pid>-<start>.json`**, one per population, replaced whole every `tick_s`. The name
carries the campaign range and its writer's pid and start time, so a slice run again or a hand launch with the
same range writes its own file. The reader checks each file's `run_id` and range against the manifest and shows
overlapping ranges and duplicate writers instead of summing them. A conflicting range stays unknown until an operator designates an owner or starts a fresh run. P4 requires host-unique writer IDs across nodes:

```json
{"v": 1, "run_id": "<16 hex>", "seq": "<int>", "t": "<float>", "t0": "<float>", "first": 0, "count": 625,
 "mode": "async", "budget": 32, "pid": "<int>", "end": false,
 "tick_late_ms": "<float>", "write_ms": "<float>", "skipped": 0, "write_errors": 0, "errors": {}, "off": false,
 "rollup": {"waiting": 0, "running": 0, "done": 0, "failed": 0, "interrupted": 0, "results": 0, "kept": 0,
            "feasible": 0, "best": {"campaign": "a17", "best": "<float>", "n": "<int>"}},
 "table": {"state": "rrd...", "n": [], "kept": [], "best": [], "feas": [], "best_k": [], "gain_n": [],
           "cycles": [], "t_last": [], "dir_v": []},
 "failed": {"a42": {"role": "monitor", "error": "<at most 200 characters>"}},
 "quiet": [["a7", "<age_s>", "<in_flight>", "<running or null>", "<answered>"]],
 "directives": {"version_read": 0, "read_t": "<epoch seconds>", "acks": {"3": {"applied": 0, "not_applicable": 0}}}}
```

`dir_v` and `directives` exist only with `steer`. The values above are placeholders, not measurements.

**`xc/directives-v<N>.json`** (P3; written only by the entry point, one immutable file per version; readers verify `run_id` and version before using it):

```json
{"v": 1, "run_id": "<16 hex>", "version": 7, "written_t": "<float>",
 "directives": [{"id": "d3", "version": 3, "kind": "note", "targets": {"all": true},
                 "text": "<at most note_chars characters>", "issued_t": "<float>", "expires_t": "<float>"}]}
```

**`keys-p<index>.npz` and `work_keys.npz`** (P1): the row of section 4.6, column by column, and the header.

**The record's section** (XC-K4; only when a plane knob is on; summed by `merge_records`, XC-K5, XC-W3):

```json
"cross_campaign": {"v": 1, "run_id": "<16 hex>", "folder": "xc", "observe": true, "steer": false,
  "populations": "<int>", "ticks": "<int>", "files_written": "<int>", "skipped": "<int>", "write_errors": "<int>",
  "write_ms": {"p50": null, "max": null}, "tick_late_ms": {"p50": null, "p95": null, "max": null},
  "errors": {"<site>": "<int>"}, "disabled_at": null,
  "last": {"done": "<int>", "failed": "<int>", "interrupted": "<int>", "results": "<int>"},
  "work_keys": {"rows": "<int>", "distinct": "<int>", "namespaces": ["<32 hex>"],
                "repeats": {"same_campaign": 0, "other_campaign": 0},
                "ideal_hits": {"same_campaign": 0, "other_campaign": 0}, "would_save_sim_s": {"same_campaign": 0.0,
                "other_campaign": 0.0}, "twins": {"equal": 0, "differ": 0},
                "coverage": {"populations": "<int>", "missing": []}, "file": "work_keys.npz"},
  "directives": [{"id": "d3", "version": 3, "text": "...", "issued_t": "<float>", "expires_t": "<float>",
                  "targets": {"all": true}, "applied": 0, "not_applicable": 0, "closed": 0, "unconfirmed": 0}]}
```

The folder is named relative to the run, never as a path. Each scope of `repeats`, `ideal_hits` and
`would_save_sim_s` counts the rows with that flag; a row can be in both (section 4.6). The merged record adds
`other_launch` (a subset of `other_campaign`) to the three, and the namespaces it could not compare. Coverage
against the manifest is the entry point's, not the record's. P2 adds `reuse: {"logical",
"physical", "reused", "reuse_refused", "saved_sim_s", "twins"}`.

## 7. Transports per way of running

| Way of running | Who writes the progress files | Plane folder | P1 rows | Who reads directives | Status |
|---|---|---|---|---|---|
| One local launch (laptop, or by hand) | the head's one plane, one file per population, on the head's one loop and its thread | `<out>/xc` (`--xc-dir` overrides); the head writes the manifest | per population, `work_keys.npz` in `<out>` | the head's plane | the layer ran on StellarAI (2,000 campaigns, bound by its loop); the plane: not built |
| `FUSION_LAUNCHES=k` (start here: what ran to the end on StellarAI, by hand) | each launch's plane, in its own process | `$run_dir/xc`, one for all k launches (XC-E1 passes `--xc-dir`) | per launch, in `run-<i>/`; across launches by `merge_records` | each launch's plane | 8 x 800 real ran by hand on StellarAI; the script's `FUSION_LAUNCHES` checked on a laptop only; the plane: not built |
| `FUSION_RUNTIME=dragon` | each population process's plane (`Fusion.build` runs there, precedent `layer_autonmi.py:760-763`) | `$run_dir/xc` | per population process; lost for a population cut short (no SIGTERM flush) | each population process's plane | the layer ran on StellarAI with the fixture inner loop (2,000 campaigns); the plane: not built. No SIGTERM flush: the head does not call `cut_short` for Scripts in other processes (`layer_autonmi.py:2254-2274`) |
| Several campaign nodes (P4) | unchanged | the run folder on the shared scratch file system | unchanged | unchanged, `tick_s` raised if the file system needs it | not built: no campaign side spans nodes yet (Dragon refuses more than one node; k launches run on one node) |

During the run nothing else crosses a process. P2's ledger (gated) is the one exception: per-writer logs on the
node-local `TMPDIR`, shared by the launches and the population processes of one node.

## 8. Knobs, flags and defaults (planned)

A new knob section, `cross_campaign`, read by the AUTONMI layer only (XC-K1). Its keys are not added to `record`
or `campaign`: `tests/test_sim_knobs.py:89-90` pins those two sections exactly. A laptop experiment on a scratch
copy confirmed it: a key added to `record` broke that test, a new section did not.

| Knob | Rule | Default | Phase | `MACHINE_KNOBS` | `ADDED_KNOBS` value | `run.py`'s layers |
|---|---|---|---|---|---|---|
| `cross_campaign.observe` | `("bool",)` | false | P0 | yes: it changes no result | false | refused when true |
| `cross_campaign.tick_s` | `("num", 1, None)` | 10.0 | P0 | yes | 10.0 | refused when changed |
| `cross_campaign.work_keys` | `("bool",)` | false | P1 | yes | false | refused when true |
| `cross_campaign.steer` | `("bool",)`; needs `observe` | false | P3 | no: it changes prompts | false | refused when true |
| `cross_campaign.note_chars` | `("int", 0, 2000)` | 400 | P3 | no | 400 | refused when changed |
| `cross_campaign.reuse` | `("choice", ("off", "other", "campaign", "run"))` | off | P2 | no | off | refused unless off |

- Each knob goes into `knobs.SCHEMA` (XC-K1), `knobs/smoke.yaml` (every knob must be set there,
  `knobs.py:200-208`), `tests/test_sim_knobs.py`'s `NEWER` (`:9-11`), `compare_records.ADDED_KNOBS` and, for the
  machine knobs, `MACHINE_KNOBS` (XC-K3), and `knobs.validate`'s refusal for `run.py`'s layers (XC-K2).
  `knobs/hundredk.yaml` stays as it is and takes the defaults from smoke. A machine knob file turns the plane on.
- `merge_records._validate` then requires launches to agree on `steer`, `note_chars` and `reuse`
  (`merge_records.py:111-118`).
- **Layer flag**, not a knob (it is a path, so it is not in the record): `--xc-dir DIR` (XC-A5), default
  `<out>/xc`, refused without `cross_campaign.observe`. `--xc-run-id ID` pins the expected run identity for
  a hand launch joining a group. The regular supervisor supplies it with `--xc-dir`.
- **Launch script**: `FUSION_XC=1` (XC-E1), default unset. Unset, the script behaves as today, and its dry-run
  lines are unchanged.
- **Entry point options** (they change no campaign): `--dir`, `--json`, `--stall-results` (default P),
  `--quiet-s` (600), `--ttl-s` (1800), `--to FIRST-LAST`, `--ids`, `--flag`, and for `serve` `--bind`, `--port`.

## 9. Record, README and tests

**Off.** With every plane knob at its default, the `cross_campaign` section is left out of `rec["knobs"]`
at the shared `record.write` serializer (`record.py:519`; LangGraph/Academy call it directly at `run.py:438`), of `campaign.json` and of the app's parameters (both written by
AUTONMI's `Campaign` from `definition.knobs`, so the `Fusion` definition drops it there). An off record is
today's schema. Nothing else changes: no folder, no thread, no file, no request, no campaign line, no
`results_sha256`. `ADDED_KNOBS` still lets `compare_records` and `merge_records` read a missing section as the
defaults when an on record meets an off or older one (`merge_records.py:118`).

**On.** Section 6 reports enabled features. With observe and work_keys only, the scientific path and campaign schema
stay unchanged. With the same profile inputs, model replies and fallbacks, the generations reference cases match.
For exact async equality also fix the order of arrivals. For independent async runs `results_sha256` can differ between identical
runs even with no plane (laptop: two identical fixture runs gave `c4be6240...` and `0d94b6b1...`; the docstring of
`tests/test_layer_autonmi.py:1849`, `compare_records.ASYNC_VARY`), so an async run with the plane on is compared as
any two async runs are: through `compare_records` and the replay. With `steer`, campaign lines carry `directives`
when applied, by design. With `reuse`, `n_reused` and `touch_check.reused`.

**README.** Now: one line under "Open items" pointing to this file. When P0 is built: a section "Watching and
steering a run" (the knobs, the plane folder, the entry point's commands per way of running); "What a run
writes" gains `xc/` and `work_keys.npz`; "What has run, and what has not" gains one row per phase.

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
| P0a | `test_off_makes_no_plane` | knobs at defaults: no `xc/` folder, no thread named `xc`; no `cross_campaign` key at the record's top level, in raw `rec["knobs"]` for all three layers, or in AUTONMI campaign lines, `campaign.json`, the launcher manifest/report parameters and the Parameters checkpoint event; `compare_records` finds the record equal to one written before the section existed |
| P0a | `test_observe_changes_no_result` | the generations-mode reference cases with `observe`: the same `results_sha256`, counts, parent files and request bodies as off |
| P0a | `test_observe_replays_the_async_order` | an async run with `observe`: every campaign replays (`tests/async_replay.py`) |
| P0a | `test_progress_ends_as_the_lines` | the last file per population: done equals the lines without error, failed the error lines; `n` and `best` equal each line's `n_sims` and `best.te_core_metric` |
| P0a | `test_plane_errors_never_reach_an_agent` | a raise injected at each guard site in turn: the matched-arrival, matched-reply control retains the same scientific results; `errors` counts the site; `disabled_at` is set |
| P0a | `test_unwritable_plane_folder` | the run passes; `write_errors` > 0; the plane stays on (`off` false) |
| P0a | `test_sigterm_writes_the_last_files` | local runtime only: exit 143; the unfinished campaigns are `i` in the last files |
| P0a | `test_unflushable_write_never_holds_exit` | a write that hangs: the process still exits within the flush timeout, and the abandoned write is counted |
| P0a | `test_entry_point_flags` | on written files: the rules of section 4.4, coverage against the manifest, overlapping ranges and duplicate writers shown and not summed, age on every answer |
| P0a | `test_run_py_layers_refuse_the_plane` | LangGraph and Academy refuse each plane knob with `KnobError` |
| P0b | `test_two_slices_one_plane_folder` | two `--slice` launches with one `--xc-dir`: `status` totals equal the merged record's; in `status`, a slice that never started shows as missing; two heads racing on private inodes preserve the winning run ID and bytes; a stale same-shape folder is rejected |
| P0b | the launch script's dry run (as `tests/test_launch.py`) | with `FUSION_XC=1` the manifest is written and `--xc-dir` passed; without it the dry-run lines are unchanged |
| P1 | `test_work_keys_count_the_survivors` | generations fixture, P 4, G 2: `repeats.same_campaign` and `ideal_hits.same_campaign` equal the survivors queued again; `other_campaign` 0; `twins.differ` 0; `results_sha256` equal to off. Async: 0 and 0 |
| P1 | `test_work_key_tracks_the_numbers` | The key preserves the fixture's typed params bytes. With fixed scalars, changing only trace content changes the digest through the real XC-W1 path; changing only trace timings preserves it. |
| P1 | `test_work_keys_merge` | two slices merged: `other_launch` 0; rows equal the evaluations; two slices with different namespaces still merge, and their namespaces are reported as not compared |
| P3 | `test_note_reaches_the_prompts` | with `tests/fake_vllm.py`: the note's line just before `GUIDANCE_ASK` (async) and before the refill ask (generations), cut to `note_chars`; with `steer` on and no file, request bodies byte-equal to off; the prompt bound holds at `note_chars`; the stub's replies are unchanged |
| P3 | `test_note_expires_and_is_revoked` | concurrent guide/revoke updates preserve both operations and the immutable prior version; include a killed writer, a skipped version and an exited target; one dispatch ack per campaign/version; `not_applicable` with `llm.mode: off` and with `outer.generator: mutation`; `closed` for ended campaigns |
| P3 | `test_steer_off_ignores_directives` | a directives file with `steer` off changes nothing |
| P2 | `test_reuse_keeps_every_count` | Generations fixture, `reuse: campaign`: dispatch sees only published completions visible then. Test a completion between queue and dispatch and a DONE delayed past dispatch. `verify`, AUTONMI checks, per-generation science and `logical = physical + reused` hold. |
| P2 | `test_reuse_refuses_a_bad_result`, `test_publication_never_replaces` | Wrong typed params, runtime, payload, trace or missing requested artifacts cause local evaluation. Same-writer retry and competing-writer publication preserve the first payload. A differing twin disqualifies the key when observed. |
| P2 | async reuse | compared through `tests/async_replay.py` |

## 10. Phases, each with its exit check

| Phase | Builds (tags) | Exit check |
|---|---|---|
| **P0a** observe, one process (laptop) | `plane.Plane`, `Table`, the guard, the tick and thread, the progress file, `entry_point` as a reader; knobs; record section (XC-C1 to C4, A1 to A4, K1 to K4) | The suite at 357 passed and 1 skipped plus the P0a tests (`results_sha256` equal to off in the generations-mode reference cases). On a 2,000-campaign async fixture run with the stub, 3 runs each with and without `observe`: `compare_records` finds no result difference (counts and invariants), every campaign replays (`tests/async_replay.py`), and the head's event-loop lag p95 and the agent phase are within run-to-run noise. Async `results_sha256` is not compared: it differs between identical runs with no plane (section 9) |
| **P0b** observe, several launches (laptop, then StellarAI) | the manifest and `--xc-dir` (XC-E1, XC-A5), the sections summed in `merge_records` (XC-K5), coverage against the manifest in the entry point; `serve` and `ask` as options | Laptop: the P0b tests. StellarAI: `FUSION_LAUNCHES=8`, 8 x 800 campaigns, real inner loop, `observe` on: passes as on 2026-11-25; agent phase within noise of those records; event-loop lag p95 within 1 ms of the 6-7 ms measured then; `status` answered throughout; its final totals equal the merged record's campaigns and `counts.sims` |
| **P1** measure (laptop, then StellarAI) | `WorkKeys`, `repeats` (XC-W1 to W4, XC-R2's encoding, XC-A2, XC-A3) | Laptop: the P1 tests. StellarAI: the P0b run with `work_keys` on: rows equal the evaluations, full coverage, every counter recorded; one generations-mode real run on the GPUs (400 campaigns) for the CUDA twin check (`twins.differ`). The gate of section 4.6 is evaluated and its result written here and in the README. Expected (inferred): `other_campaign` and `other_launch` 0 in the async mode |
| **P3** steer with notes (after P0b; beside P1; it does not wait for P2) | `guide`, `revoke`, the directive set, acks (XC-C5 to C8) | Laptop: the P3 tests. StellarAI: a note to every campaign at t = 300 s is shown by at least 95% of targets with two eligible prompt opportunities after their first observed version, within those opportunities; separately report all targeted, dead, ineligible and unconfirmed counts and issue-to-dispatch latency; event-loop lag as in P0b; measure added prompt tokens, changed reply lengths and elapsed time at matched load |
| **P2** reuse, gated | `Ledger`, `prepare`'s lookup, the worker's check and rebinding, the counts (XC-R1 to R7, XC-W1's publish) | Only if P1's gate passes. Laptop: the P2 tests. StellarAI: the CUDA twin check passes first (or P2 stays on CPU and fixture); then a run whose saved inner-loop seconds exceed the plane's added work |
| P3b stop, budget, priority | not in this plan | A separate design with a handshake over the B-tied bounds (XC-C9 to C12) and revised checks |
| **P4** several campaign nodes | nothing new until the campaign side spans nodes | The same reader on the shared run folder, `tick_s` 30 if the file system needs it; coverage complete; `lost` and `quiet` behave when one node's launch is killed; clock skew measured |

The P1/P2 CUDA exit requires at least one controlled same-input, complete-trace twin for each eligible actual
GPU/runtime class. Require equal scalar and required non-timing trace outputs and zero differing twins. Missing
coverage leaves that class ineligible for reuse.

## 11. Open decisions

1. **Should independent campaigns share initial conditions?** Today each starts with its own 16 full profiles
   and 16 shape signatures. Fixing one six-value signature everywhere, sharing a pool of signatures while
   drawing each campaign's seven tunables independently, and sharing 16 complete initial profiles are three
   different designs. Only the last guarantees repeated full initial inputs. Each changes the sampling
   distribution and requires an explicit experiment version and fresh measurements of exact matches.
2. **May reuse skip re-evaluating survivors in the generations mode?** IGNOS re-evaluates survivors
   (`parent/evolutionary_loop.py:1403`). The corresponding repeated inputs represented about a quarter of the
   local generations fixture evaluations. The knob defaults to `reuse: off` in every phase. If the P2 gate
   passes, `reuse: other` is the first opt-in policy. It excludes a campaign's own prior results;
   `reuse: campaign` or `run` would permit them. Each reused result still counts as one logical evaluation.
3. **On GPUs, is a reused result acceptable if two evaluations of one profile differ at float noise?**
   `GraphNet.forward` uses `index_add_` (`standins.py:68`) with no deterministic setting, so CUDA results may
   differ between runs (not measured). P1's twin check measures it. If the answer is no, P2 runs on CPU and the
   fixture only.
4. **The P2 gate.** This plan's gate is ideal cross-campaign hits of at least 1% of the evaluations and the
   lowest launch's `sim_bound_ratio` of at least 0.8, both chosen thresholds. Another threshold, or none, is a
   choice.
5. **The cost of notes.** A live 100-token note adds about 4.6% to the run's prompt tokens on the servers that
   set the pace. Is that acceptable, and should notes also reach the Generators' prompts (about 11% at 150
   tokens)?
6. **Stop, budget and priority directives.** A later design (P3b, sites XC-C9 to C12), or never, with whole-run
   stop left to SIGTERM?
7. **A stall rule while nothing is feasible.** All 6,400 campaigns in the measured eight-launch run were
   infeasible, so the feasible-only `stalled` rule did not fire (section 4.4). Should the digest gain a column for that phase (for example, results since the
   window's best wall-blend rank last improved, computed at the cycle), or is "infeasible after `S_R` results"
   enough for the entry point?

## 12. Where the plan attaches: the tags

Each tag below is a comment in the code, `# CROSS-CAMPAIGN [<tag>] (see CROSS_CAMPAIGN_PLAN.md): ...`, one to
four lines at the indentation of the line it sits above. The comments are the only change in the code. Every tag
in this table is in the code and every tag in the code is in this table: `grep -rn "CROSS-CAMPAIGN \[" src launch`
lists them. 37 tags in 8 files: `layer_autonmi.py` 21, `decide.py` 6, `sim.py` 2, `knobs.py` 2,
`merge_records.py` 2, `record.py` 1, `compare_records.py` 1, `launch/stellar_ai_autonmi.sbatch` 2. Line numbers
are those of `000300c`, before the comments went in.

| Tag | Phase | File | Function | Above line (at `000300c`) | What attaches there |
|---|---|---|---|---|---|
| XC-C1 | P0a | `layer_autonmi.py` | `Script._begin` | 1140: `self.began[me] = time.time()` | start digest: the campaign is marked running |
| XC-C2 | P0a | `layer_autonmi.py` | `Script.remember` | 1596: `self.pop.store.put(f"campaign/` | the one Coordinator digest hook, both modes |
| XC-C3 | P0a | `layer_autonmi.py` | `Script.report` | 1601: `self.reported.add(campaign)` | done digest; with steer, the line's applied notes |
| XC-C4 | P0a | `layer_autonmi.py` | `Script.on_done` | 1641: `self._line({"agent_id": campaign, "error":` | failed digest (first failure of a campaign) |
| XC-C5 | P3 | `layer_autonmi.py` | `Script._coordinator_async` | 1356: `self.mail.deliver(me, "generator", "survivors"` | directive read, async: the newest live note for this campaign |
| XC-C6 | P3 | `decide.py` | `guidance_messages` | 996: `user = ("Results just judged:` | the note as one line before GUIDANCE_ASK |
| XC-C7 | P3 | `layer_autonmi.py` | `Script._generation` | 1173: `self._send(me, "generator", "make", g, msg)` | directive, generations mode: the note rides on the repopulate message |
| XC-C8 | P3 | `decide.py` | `refill_messages` | 252: `ask = (f"Generation {g + 2} of {G}` | the message's note as one line before the ask |
| XC-C9 | P3b, not planned | `layer_autonmi.py` | `Script._coordinator_async` | 1363: `return _state(s, "cycle" if cst["arrived"] < ...` | stop or budget would change this bound on B (with C10 to C12) |
| XC-C10 | P3b, not planned | `layer_autonmi.py` | `Script._generator_async` | 1389: `while gst["made"] < self.B:` | a stop would end the Generator's loop |
| XC-C11 | P3b, not planned | `layer_autonmi.py` | `Script._evaluator_async` | 1440: `while scored_n < self.B:` | a stop would lower the Evaluator's bound |
| XC-C12 | P3b, not planned | `layer_autonmi.py` | `Fusion._verify_async` | 958: `if not (co["candidates_distinct"] == B` | a stopped campaign would be checked against its stopped_at |
| XC-C13 | none (refused) | `decide.py` | `run_campaign` | 576: `await svc.remember(agent_id, memory(cst))` | the Academy layer's generation boundary (LangGraph's is `layer_langgraph.coordinate`, `:147-148`, locked) |
| XC-A1 | P0a | `layer_autonmi.py` | `Fusion.build` | 760: `if os.getpid() != self.head_pid:` | attach the population's table to the process's plane; start the tick once |
| XC-A2 | P0a, P1 | `layer_autonmi.py` | `Fusion.rows` | 802: `sc.sums.write(sums_file(Path(sc.lines.name), ...` | population end: last progress file; P1 rows beside the call sums |
| XC-A3 | P0a, P1 | `layer_autonmi.py` | `Script.cut_short` | 1665: `self.sums.write(sums_file(Path(self.lines.nam...` | SIGTERM, local runtime only: one bounded flush, rows so far (no flush under Dragon) |
| XC-A4 | P0a | `layer_autonmi.py` | `miniapp_workflow` | 2244: `await campaign.start_services()` | open the plane folder, manifest if absent (tmp, then `os.link`), run id; close in the finally |
| XC-A5 | P0b | `layer_autonmi.py` | `parse` | 2027: `p.add_argument("--stagger-batch"` | --xc-dir DIR, the folder every launch of a run shares |
| XC-E1 | P0b | `launch/stellar_ai_autonmi.sbatch` | `main` | 503: `if (( launches > 1 )); then several "${args[@...` | FUSION_XC=1: write the run manifest, pass --xc-dir to every launch |
| XC-G1 | none (notes only) | `layer_autonmi.py` | `Script._generator` | 1197: `self._send(me, "coordinator", "population", g,` | where the Generator makes candidates (generations); the plane does not act here |
| XC-G2 | none (notes only) | `layer_autonmi.py` | `Script._generator_async` | 1402: `for call in calls:` | where the Generator queues final candidates (async); the plane does not act here |
| XC-W1 | P1 (P2 publishes here) | `layer_autonmi.py` | `Script._monitor` | 1220: `numbers = None` | one work-key row per evaluation, on the loop |
| XC-W2 | P1 | `layer_autonmi.py` | `write_record` | 2733: `for i in range(a.populations):   # summed int...` | count the rows into the record; keep work_keys.npz |
| XC-W3 | P1 | `merge_records.py` | `merge` | 238: `rec["speed"] = _speed(launches)` | count repeats and ideal hits across launches, within equal namespaces |
| XC-W4 | P1 | `decide.py` | `sim_knobs` | 118: `s, c = dict(knobs["sim"]), cost_scale(knobs, ...` | the sim values after the cost scale that the work key's namespace holds |
| XC-R1 | P2 (gated) | `sim.py` | `evaluate` | 249: `if not os.environ.get(COLD_CHILD) and is_cold(` | reuse inside the worker, real inner loop |
| XC-R2 | P1, P2 (gated) | `sim.py` | `evaluate_fixture` | 321: `digest = hashlib.sha256(json.dumps(params, so...` | the key's encoding; reuse in the fixture |
| XC-R3 | P2 (gated) | `layer_autonmi.py` | `Script.__init__` | 1098: `self.evaluate = task_tool(` | prepare: the take-time lookup, the stored result to the worker |
| XC-R4 | P2 (gated) | `record.py` | `write` | 537: `"touch_check": {"evaluations": sum(devices.va...` | reused counted apart in the touch check; logical and physical counts |
| XC-R5 | P2 (gated) | `decide.py` | `on_evaluated` | 495: `for kind, n in (e.get("touches") or {}).items():` | a reused result adds no touches or verdict (generations) |
| XC-R6 | P2 (gated) | `decide.py` | `async_on_scored` | 767: `for kind, k in (e.get("touches") or {}).items():` | a reused result adds no touches or verdict (async) |
| XC-R7 | P2 (gated) | `launch/stellar_ai_autonmi.sbatch` | `main` | 407: `export TMPDIR=$node_tmp` | the ledger's node-local folder |
| XC-K1 | P0a (each phase adds its knobs) | `knobs.py` | `SCHEMA` | 109: `"record": {` | the new cross_campaign section |
| XC-K2 | P0a | `knobs.py` | `validate` | 234: `if "async" not in modes:` | run.py's layers refuse the plane; steer needs observe |
| XC-K3 | P0a | `compare_records.py` | `module (ADDED_KNOBS)` | 45: `# knobs newer than some records, with the val...` | ADDED_KNOBS and MACHINE_KNOBS entries |
| XC-K4 | P0a to P3 | `layer_autonmi.py` | `write_record` | 2766: `if knobs["record"]["extras"]:` | the record's cross_campaign section, only when on |
| XC-K5 | P0b | `merge_records.py` | `merge` | 237: `rec["wall_seconds"] = _range(` | sum the sections of the launches read (coverage against the manifest is the entry point's) |

**Locked places (no note in the code).** These files may not be edited, so the plan names the place here only.

| File | Place | Why it matters |
|---|---|---|
| `parent/evolutionary_loop.py` | `initialize_population` (`:435-454`), `mutate_params` (`:1046-1063`), `repopulate` and its restart seed (`:1072-1104`, `:1089`) | where each initial profile receives its six frozen values and each descendant inherits one local profile's values. The restart branch is unreachable after a completed generation (section 3.1). A "move" or a diversity draw would belong here, in IGNOS itself |
| `parent/evolutionary_loop.py` | `run_evolutionary_loop`'s evaluation of the whole population each generation (`:1403`) | why the generations mode repeats about 24% (decision 2) |
| `parent/evolutionary_loop.py` | `_evaluate_single_candidate` (`:546`), `_evaluate_validated_candidate` (`:459`) | where a memo would sit in IGNOS; the mini-app's equivalent is XC-R1 |
| `parent/evolutionary_loop.py` | `update_best_result` (`:1000-1041`), `select_survivors` (`:929`) | the feasible-gain rule that `gain_n` reads; survival, which nothing in this plan steers |
| `parent/evo_agents.py` | the directed-proposer seam its docstring names (`:17-18`) | the parent's own place for a guided Generator |
| `run.py` | `Services.simulate` (`:255`), `Services.remember` (`:268-270`), `Services.report` (`:272-275`), `_fan_out`'s live array (`:351-388`); `sim_function` (`:289-292`) | the other layers' evaluation, digest, report and live counts (the plane is refused there, XC-K2); `sim_function` picks `evaluate` or `evaluate_fixture`, which is why P2 needs XC-R2 |
| `layer_langgraph.py`, `layer_academy.py` | `coordinate` (`:134-155`); `CoordinatorAgent.run_generations` (`:80-83`) | those layers' Coordinator cycle: Academy's through `decide.run_campaign` (XC-C13); LangGraph's in `coordinate` itself (`:147-148`) |
| `knobs/hundredk.yaml` | the preset | stays off; it takes the new knobs' defaults from smoke, and a machine knob file turns the plane on |
| `launch/run.sh`, `batch.sh`, `stellar_ai.sbatch`, `lux.sbatch`, `perlmutter.sbatch` | the `run.py` launches | no plane for those layers |

## 13. What this plan rests on

- **Code**: every line cited, read at `000300c`. Every tag's line was checked against its text, and the comments
  were inserted into a scratch copy outside the repository: each Python file's syntax tree stayed equal to the
  original's, and `bash -n` passed on the launch script.
- **Laptop (2026-11-26)**, on scratch copies and outside the test suite: the same-work cases of section 3.1; the
  six-tuple count over 20,000 seeds; the repeat rates (24.3% over 40 campaigns, 24.7% over 320; 0 across
  campaigns); near repeats; the unit costs of section 3.5; `os.rename` onto an existing name keeps the second
  writer, `os.link` refuses it; the knob placement (a key in `record` breaks `tests/test_sim_knobs.py:89-90`, a
  new section does not, and `compare_records` needs the `ADDED_KNOBS` entries); the digest-hook probe of
  section 3.3; an off-mode generations fixture run with two default-off knobs added: the same `results_sha256`;
  two identical async fixture runs (16 campaigns, stub, no plane): different `results_sha256`; a
  `ThreadPoolExecutor` job of 3 s after `shutdown(wait=False, cancel_futures=True)` held process exit 3.0 s.
- **StellarAI (2026-11-25 records)**: 8 x 800 campaigns with the real inner loop: 26.9 Coordinator cycles,
  106.6 model calls and 548 prompt tokens per call per campaign, `sim_bound_ratio` 0.47 (with
  `sims_executing.max` 8 of 8), `counts.feasible_agents` 0 in every launch, 453 model calls/s
  summed, agent phase 1,494-1,511 s, event-loop lag p95 6-7 ms; one process at 2,000 campaigns: lag p95 1.5 s.
- **Inferred**: every cost at 20,000 campaigns; the run length T (the README's line, a guess past 6,400
  campaigns); the 45 bytes per campaign of a progress file; the note's share.
- **Not measured anywhere**: anything of the planes themselves; CUDA determinism of the inner loop; file
  replace and `stat` latency on the shared scratch file system under load; clock skew between nodes.
