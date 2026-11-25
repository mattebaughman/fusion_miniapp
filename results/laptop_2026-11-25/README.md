# Laptop ladder, 2026-11-25 (the AUTONMI layer in its final shape)

The AUTONMI layer at 500, 2,000 and 10,000 campaigns (2,500, 10,000 and 50,000 hosted agents: a Coordinator, a
Generator, an Evaluator and two Monitor slots per campaign), a SIGTERM at 10,000, the StellarAI knob file against two
fake model servers, and a client-load check to 12,000 open model calls. These are functional checks and laptop
timings, not performance claims for a machine. No real model and no real inner loop ran: the inner loop is the
fixture and every model call goes to the stub LLM or to fake servers. All physics output is synthetic.

**Labels.** Every number below carries one:
- **[configured]**: set by a knob file or the command (each rung's `command.txt`, the knob files here);
- **[run]**: what ran and how it ended (exit code, the record's `status`);
- **[measured]**: read on this laptop from the run's record (`<rung>/run_record.json`) or from this folder's
  read-only probes (`<rung>/ladder.json`);
- **[emulated]**: a stand-in's own setting: the stub LLM's latency per call (`llm.stub_latency_s`; its replies are
  computed from the numbers, not a model's) and the fake model servers' latency (`tests/fake_vllm.py`, a FIXTURE
  that answers every request after a set time);
- **[inferred]**: computed from measured numbers; the text says how.

**The laptop and the code.** Apple silicon, 12 cores, 24 GB, macOS 26.7. Every rung ran on AC power with Low Power
Mode off (pmset's powermode 2), one at a time, starting once the 1-minute load average was under 3; each
`ladder.json` holds pmset's and uptime's readings before and after its rung [measured]. Code: `dev` at `ea5d0bf`;
the records say `dirty: true` because this folder and its `.gitignore` line were not yet committed.

## How the rungs were run

```sh
python results/laptop_2026-11-25/ladder.py <runs folder> all     # from the repository's root; about an hour [inferred]
python results/laptop_2026-11-25/ladder.py --list                # the rungs
```

`ladder.py` runs `python -m fusion_miniapp.layer_autonmi` once per rung through its own `--child`, which adds two
read-only probes in the run's process (each model call's role and prompt length; each answered call's role and
times). The record toggle (`record.extras`) stays off, and nothing is added to the run's record or lines. It
writes each rung's large run folder to the runs folder and keeps here the record (host name and local paths
replaced by `laptop`, `<runs>`, `<repo>`, `~` and `$TMPDIR`) and `ladder.json`, the figures below. A 10,000-campaign
rung starts only when an extrapolation from the 500- and 2,000-campaign rungs in the same runs folder puts it under
45 minutes; here it predicted 592 s at 1 ms and 856 s at 40 s [inferred], and both ran. A skipped rung leaves its
files here as they were; a rung that runs replaces them.

- **Open files.** The ladder raises its open-files limit to 65,536 [configured] when the shell's is lower (a macOS
  shell starts at 256), and the rungs inherit it. It refuses a rung with fake servers when the limit stays under 2
  per hosted agent: in `client_load` the run's process holds about 12,000 sockets and each fake server about 6,000
  [measured].
- **Load and power.** Before each rung it waits up to 15 minutes for the 1-minute load average to fall under 3
  [configured]. It does not start a rung when the load is still higher, or when the laptop is on battery or in Low
  Power Mode. A rung that ends on battery or in Low Power Mode is not kept, and the ladder stops.
- **The probes are in the timings.** They run in the run's own process, on its event loop. Per call, the prompt
  probe sums the messages' lengths and splits the last one into lines, and the call probe stores two numbers. Their
  cost was not measured on its own. It is a few microseconds a call against about 530 µs of the run's own time per
  call in the 10,000-campaign 1 ms rung (1,117,784 calls asked in a 590 s agent phase), so under 1% [inferred].
- **This `ladder.py` and the runs.** The probes, the rungs' commands and the fake servers' replies are as run.
  After the runs, the file changed only in how it summarizes and guards a rung: check 1's figure for an interrupted
  record, the power fields kept here, the open-files limit, the load and power refusals, the prediction's source
  and the fake servers' start. Each `ladder.json` here equals what this file's summary code gives from its run's
  folder [measured].

Every rung [configured]: the `hundredk` preset (population 16, 2 generations: 32 inner loops per campaign), 2
Monitor slots, all agents starting together, `--sim fixture` on 10 threads (80 in the two StellarAI rungs, the
file's `workers.sim_workers`), one agent process and one population, the record toggle off, `campaign.notes` off
(the default). The rungs [configured; the stub's and the fake servers' times per call are emulated]:

| Rung | Campaigns (hosted agents) | Mode | Model calls | Client admission caps |
|---|---|---|---|---|
| `r500_generations_1ms` | 500 (2,500) | generations | stub, 1 ms a call; the refill and summary only | `hundredk`'s: 512 calls, 2,000,000 tokens |
| `r500_async_1ms` | 500 (2,500) | async | stub, 1 ms | `hundredk`'s |
| `r500_async_2s` | 500 (2,500) | async | stub, 2 s | lifted: 1,000,000 calls, no token cap |
| `r2000_async_1ms` | 2,000 (10,000) | async | stub, 1 ms | `hundredk`'s |
| `r2000_async_8s` | 2,000 (10,000) | async | stub, 8 s | lifted |
| `r10000_async_1ms` | 10,000 (50,000) | async | stub, 1 ms | `hundredk`'s |
| `r10000_async_40s` | 10,000 (50,000) | async | stub, 40 s | lifted |
| `r10000_async_sigterm` | 10,000 (50,000) | async | stub, 1 ms | `hundredk`'s; SIGTERM once 5,000 campaigns' lines were written |
| `stellar_two_fakes` | 20 (100) | async | 2 fake servers, 0.05 s a reply | the StellarAI knob file's (lifted) |
| `client_load` | 2,400 (12,000) | async | 2 fake servers, 30 s a reply | the StellarAI knob file's (lifted) |

The stub's latency at 2 s, 8 s and 40 s grows with the campaigns so that the inner loop keeps up: each rung asks
about the same number of model calls and inner loops per second [inferred; the rates below agree]. The last two
rungs use `knobs/machines/stellar_ai.yaml` and `knobs/machines/stellar_ai_nemotron.yaml` with `--llm openai`, the
fake servers listed in an endpoints file and a key made for the rung.

In the generations mode with `campaign.notes` off, only the Coordinator's summary and the Generator's refill call
the model: 2 calls per campaign [configured; measured], where the asynchronous rungs make 82 (stub at 2 s to 40 s)
to 112 (stub at 1 ms) [measured]. The Evaluator and the Monitors of that rung make no model call.

## Results

**Every rung passed its checks** (exit 0 and `status: pass`, AUTONMI's checks passed; the SIGTERM rung exit 143 and
`interrupted`, as it should) [run]. In every rung that finished, no model call failed, was tried again or fell
back [measured].

| Rung | Wall s | Hosted agents live at once; all live for s | Evaluations per s | Peak MiB | MiB per campaign |
|---|---|---|---|---|---|
| `r500_generations_1ms` | 16.1 | 2,500 of 2,500; 14.4 | 1,023 | 710 | 1.42 |
| `r500_async_1ms` | 28.2 | 2,500 of 2,500; 25.8 | 584 | 659 | 1.32 |
| `r500_async_2s` | 42.6 | 2,500 of 2,500; 40.6 | 382 | 683 | 1.37 |
| `r2000_async_1ms` | 115.4 | 10,000 of 10,000; 105.6 | 571 | 2,190 | 1.09 |
| `r2000_async_8s` | 171.1 | 10,000 of 10,000; 160.6 | 380 | 2,287 | 1.14 |
| `r10000_async_1ms` | 607.4 | 50,000 of 50,000; 555.7 | 542 | 9,169 | 0.92 |
| `r10000_async_40s` | 858.9 | 50,000 of 50,000; 801.8 | 379 | 9,153 | 0.92 |
| `r10000_async_sigterm` | 566.4 | 50,000 of 50,000 (the live line) | not comparable | 8,962 | not comparable |
| `stellar_two_fakes` | 2.9 | 100 of 100; 2.6 | 231 | 180 | not comparable |
| `client_load` | 683.3 | 12,000 of 12,000; 613.2 | 113 | 3,059 | 1.27 |

- Wall time, agents live, peak memory (the run's own peak resident memory) [measured]. "All live for" is how long
  every hosted agent was live at once (AUTONMI's rows). The interrupted record has no rows, so for the SIGTERM rung
  the count is the live line's agents started and not ended, read once a second.
- Evaluations per second over the agent phase (the first campaign's start to the last one's end) [measured]. The
  generations rung's 1,023 is not comparable with the asynchronous rungs: its campaigns make 2 model calls, not 82
  to 112.
- Each figure is from one run. Two repeats of `r500_async_2s` with this ladder about 85 minutes later, in the same
  power state, gave 367 and 368 evaluations per second and 44.1 and 44.3 s, against 382 and 42.6 s here
  [measured]: rungs that differ by a few percent are within the spread between runs.
- MiB per campaign is the peak divided by the campaigns, the interpreter included [inferred]. The added memory per
  added campaign is about 0.87 MiB from 2,000 to 10,000 campaigns at 1 ms (9,169 − 2,190 MiB over 8,000) and
  0.86 MiB at the stub's longer latencies [inferred].
- In the asynchronous rungs 49.25% to 49.67% of the results scored after warm-up were kept with the stub, and 48.3%
  in `client_load` [measured]: IGNOS's keep rule gives about half.

**Agents of a campaign in a model call**, over the middle of the run (15% to 85% of the agent phase), beside the
modelled 4.9 of 5 for identical call times [measured; the call times are emulated]. A call counts from when it
was sent to its reply.

| Rung | Coordinator | Generator | Evaluator | Monitors (2) | All, of 5 | Per Monitor slot |
|---|---|---|---|---|---|---|
| `r500_async_2s` | 0.975 | 0.909 | 0.997 | 1.942 | 4.82 | 0.971 |
| `r2000_async_8s` | 0.981 | 0.897 | 0.995 | 1.939 | 4.81 | 0.970 |
| `r10000_async_40s` | 0.982 | 0.893 | 0.994 | 1.925 | 4.79 | 0.962 |
| `client_load` (HTTP) | 0.914 | 0.858 | 0.939 | 1.828 | 4.54 | 0.914 |

- Each row is one run. Four repeats of `r500_async_2s` 65 to 90 minutes later, in the same power state and with
  the same call probe (two with this ladder, two with that probe alone), gave 4.738 to 4.744 of 5 (0.945 to 0.946
  per Monitor slot; the Generator 0.87) [measured]. Over those five runs, 4.74 to 4.82 at 500 campaigns.
- With the stub, 4.79 to 4.82 of 5 at every size in these runs, against the model's 4.9. The Generator is the role
  most often out of a call; it takes its next 2 candidates only once none waits on the queue [inferred from the code].
- Model calls per campaign in these three rungs: Generator 16, Monitors 32, Evaluator 17, Coordinator 17 (82 in
  all) [measured]; at 1 ms the Evaluator and Coordinator make about 32 each, taking one result per cycle.
- An earlier measurement of the 500-campaign rung, on battery in Low Power Mode, gave 4.2 of 5 (the repository's
  README, "What has run"); this one is on AC power.

**Each role's largest prompt**, in characters, in the asynchronous rungs [measured], against the bound of 21,504
characters that every prompt of this mode is built to stay under (three quarters of a 32,768-token context less
the 4,096-token reply cap). The stub's notes and guidance are empty, so later prompts that show them are shorter
than a real model's would be; the fake servers' are short fixture texts.

| Rung | Generator | Monitor | Evaluator (results shown) | Coordinator (results shown) |
|---|---|---|---|---|
| `r2000_async_1ms` | 3,928 | 873 | 1,677 (2) | 1,456 (2) |
| `r2000_async_8s` | 3,924 | 873 | 1,683 (2) | 1,459 (2) |
| `r10000_async_1ms` | 3,931 | 874 | 1,680 (2) | 1,457 (2) |
| `r10000_async_40s` | 3,924 | 874 | 1,683 (2) | 1,460 (2) |
| `client_load` | 3,989 | 873 | 3,367 (4) | 2,069 (4) |

The largest is 18.6% of the bound. No Evaluator call held more than 4 results, so its limit of 8 per call never
applied [measured]. Built from the code's constants with every note at its 1,200-character cut and 8 results, the
longest Evaluator prompt is 14,517 characters [inferred].

**SIGTERM at 10,000 campaigns** [run, measured]: the ladder sent the signal 553.6 s after it started the run's
process, once 5,058 campaigns had written their lines. On that clock the full 1 ms rung took 623.3 s, so the signal
came at 89% of it. The run exited 143, 37.8 s after the signal, with `status: interrupted` and a record of the 5,060
campaigns that finished (161,920 evaluations). The record's own wall time, which starts after the process's imports
and ends before it exits, is 566.4 s. AUTONMI's check `every_population_ran` fails in an interrupted run, as
expected.
- The interrupted record has no per-role counts and no model-call figures: `counts.by_role` and `llm_service` show
  0 model calls, and the tokens per second and requests in flight are empty. The 5,060 campaigns did make model
  calls: their lines show all 161,920 Generator slots filled by a reply. AUTONMI's rows and the layer's sums of
  each population's calls are written when the population ends, and the signal cancelled this run's one population
  first [measured]. A run stopped by SIGTERM, as at a batch job's
  time limit, keeps its campaigns' results but loses their model figures. The code has changed since this run: an
  interrupted record now keeps the model figures of the calls made up to the signal (the repository README, "How a
  run ends badly").

**The StellarAI knob file against two fake servers** (20 campaigns) [run, measured]: all 1,864 request bodies the
two servers received carry the file's settings: the model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`,
`chat_template_kwargs {enable_thinking: false}`, `max_tokens` 4,096, temperature 1.0, top_p 0.95, and no thinking
budget. Only the Generator's carry the JSON schema (`response_format`). Both servers answered every role (983 and 881
requests). The run's model check passed (every Generator candidate filled by a reply, none empty).

**The client-load check** (2,400 campaigns, 12,000 hosted agents, two fake servers each in its own process, 30 s a
reply) [run, measured]:
- **12,000 open model calls were reached**, by four counts: the client's calls in flight from each call's times,
  the live line's open requests, the two servers' running requests summed (sampled once a second), and 12,000
  established connections to the servers (netstat, every 10 s). Over the agent phase the client averaged 8,948.
- 196,118 calls, none failed or tried again, every campaign passed; 288 calls and 113 evaluations per second.
- The one process kept up, with less margin than with the stub: 4.54 of 5 agents in a call, the event loop's lag
  74 ms at the 95th percentile and 2.8 s at most, against 40 ms and 6.5 s in the 40 s stub rung. HTTP is not the
  only difference: this rung has 80 inner-loop threads against 10, and 2,400 campaigns against 10,000
  [configured].
- Sockets [measured]: the open-files limit was 1,048,576 for the shell and 92,160 per process (macOS's
  `kern.maxfilesperproc`); local ports for outgoing connections are 49,152 to 65,535 (16,384); TIME_WAIT sockets on
  the whole laptop peaked at 6,104. A laptop check much past 12,000 open calls to one address would be near that
  port range [inferred; not tried].
- A second run of this rung, started from a shell open-files limit of 256, had its limit raised to 65,536 by the
  ladder and again reached 12,000 open calls by all four counts, with no call failed and status `pass`
  [run, measured]. The laptop went onto battery in its last minute, so the ladder did not keep it here.
- Not explained: Flowgentic's own count of agents waiting for admission (the live line's `queued`) read up to 2,201
  for 18 s about 160 s into the run, while each call's recorded wait for admission was at most 0.4 ms and no call
  failed [measured; not investigated].

## What this does not show

No real model server, GPU, real inner loop, MPS, Slurm or StellarAI node ran. The stub's calls are timed waits in
the run's own process; the fake servers run on this laptop and answer after a set time. Tokens per second in
these records are the stub's estimates (marked `emulated`) or the fake servers' fixed counts.

## Files

- `ladder.py`: the ladder (rungs, probes, fake servers, the summaries); `async.yaml`, `async_2s.yaml`,
  `async_8s.yaml`, `async_40s.yaml`: the rungs' knob files.
- `<rung>/command.txt`: the command run; `<rung>/run_record.json`: the run's record; `<rung>/ladder.json`: the
  figures above with the power readings. Campaign lines and AUTONMI's folders are left out for their size.
