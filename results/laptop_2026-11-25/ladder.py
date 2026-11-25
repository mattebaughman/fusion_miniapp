"""The laptop ladder of 2026-11-25: the AUTONMI layer in its final shape (5 hosted agents per campaign).

    python results/laptop_2026-11-25/ladder.py RUNS RUNG [RUNG ...]   # from the repository's root; RUNG or "all"
    python results/laptop_2026-11-25/ladder.py --list

Run in the AUTONMI layer's environment. Each rung runs `python -m fusion_miniapp.layer_autonmi` once, alone, through
this file's `--child` (below), with its run folder RUNS/<rung>/ (large: not kept here). Into this folder it writes
<rung>/run_record.json (the record, with the host name and local paths replaced) and <rung>/ladder.json: the
checks' figures, the power state before and after, and the laptop's load; a re-run replaces them. Before each rung
it waits for the 1-minute load average to fall under LOAD_MAX, and it stops before a rung when the load is still at
or over LOAD_MAX after LOAD_WAIT_S, or the laptop is on battery or in Low Power Mode (pmset's powermode 1); a rung
that ends on battery or in Low Power Mode is not kept here, and the ladder stops. A 10,000-campaign rung starts
only when the extrapolation from the 500- and 2,000-campaign rungs in the same RUNS (`predict`) puts it under
LIMIT_S; otherwise it is skipped, says so on stdout and in RUNS, and leaves this folder as it was. The ladder
raises its open-files limit to OPEN_FILES (`open_files`; a macOS shell starts at 256), which its rungs inherit, and
refuses a rung with fake servers when the limit stays under 2 per hosted agent.

Read-only probes, in the run's own process (`--child`). Nothing is added to the run's record or lines, and the
record toggle (record.extras) stays off:
- every model call asked (Script._ask): its role, its prompt's characters and its "candidate ..." rows, for each
  role's largest prompt;
- with `calls`: every answered call's role and times (Script.on_call), for the agents in a model call by role over
  the record's steady window, 15% to 85% of the agent phase (the first campaign's start to the last one's end).

Fake model servers (`fakes`): tests/fake_vllm.py, a FIXTURE and not a model, one per process (`--fake-server`),
serving the StellarAI knob file's model name with a key made for the rung (kept under RUNS, removed after it). Each
keeps a summary of the requests it received (by role: count, the largest prompt, each distinct body besides the
messages) and samples its running requests once a second.
"""
from __future__ import annotations

import json
import math
import os
import re
import resource
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from array import array
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"   # llm.model of knobs/machines/stellar_ai_nemotron.yaml
STELLAR = ["--knobs", "knobs/machines/stellar_ai.yaml", "--knobs", "knobs/machines/stellar_ai_nemotron.yaml"]
ROLES = ("coordinator", "generator", "evaluator", "monitor")
BOUND = (32768 - 4096) * 3 // 4   # 21,504: each asynchronous role's longest prompt, by construction (tests/test_async.py)
LOAD_MAX, LOAD_WAIT_S, LIMIT_S = 3.0, 900, 45 * 60
OPEN_FILES = 65536   # each open call to a fake server holds a socket in the run's process and one in a server's
MODELLED = 4.9   # agents of 5 in a model call when every call takes the same time (the plan's model, not measured)

# rung: campaigns; knob files of this folder; the stub's latency it sets (for the record); `calls`: the call probe;
# `sigterm_at`: SIGTERM once that many campaigns' lines are written; `stellar`: the StellarAI knob files with
# --llm openai against `fakes` fake servers whose replies each take `fake_latency_s`.
RUNGS = {
    "r500_generations_1ms": {"agents": 500, "knobs": []},
    "r500_async_1ms": {"agents": 500, "knobs": ["async.yaml"]},
    "r500_async_2s": {"agents": 500, "knobs": ["async_2s.yaml"], "calls": True},
    "r2000_async_1ms": {"agents": 2000, "knobs": ["async.yaml"]},
    "r2000_async_8s": {"agents": 2000, "knobs": ["async_8s.yaml"], "calls": True},
    "r10000_async_1ms": {"agents": 10000, "knobs": ["async.yaml"]},
    "r10000_async_40s": {"agents": 10000, "knobs": ["async_40s.yaml"], "calls": True},
    "r10000_async_sigterm": {"agents": 10000, "knobs": ["async.yaml"], "sigterm_at": 5000},
    "stellar_two_fakes": {"agents": 20, "knobs": [], "stellar": True, "fakes": 2, "fake_latency_s": 0.05,
                          "calls": True},
    "client_load": {"agents": 2400, "knobs": [], "stellar": True, "fakes": 2, "fake_latency_s": 30.0,
                    "calls": True},
}


# --- the run's own process: the layer with the probes ----------------------------------------------------------

def child(probe_out: Path, calls: bool, argv: list[str]) -> int:
    """layer_autonmi's main(argv) with the read-only probes, as `python -m fusion_miniapp.layer_autonmi` runs it."""
    from fusion_miniapp import layer_autonmi, run

    S, prompts, times = layer_autonmi.Script, {}, {}
    ask = S._ask

    async def _ask(self, me, a, messages, *rest, **kw):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        rows = sum(x.startswith("candidate ") for x in str(messages[-1].get("content") or "").splitlines())
        p = prompts.setdefault(layer_autonmi.role_of(me), {"asked": 0, "max_chars": 0, "sum_chars": 0, "max_rows": 0})
        p["asked"] += 1
        p["sum_chars"] += chars
        p["max_chars"], p["max_rows"] = max(p["max_chars"], chars), max(p["max_rows"], rows)
        return await ask(self, me, a, messages, *rest, **kw)

    S._ask = _ask
    if calls:
        on_call = S.on_call

        def probed(self, rec, reply):
            times.setdefault((rec.meta or {}).get("role"), array("d")).extend((rec.t_end - rec.service_s, rec.t_end))
            return on_call(self, rec, reply)

        S.on_call = probed
    signal.signal(signal.SIGTERM, run._interrupt)   # as layer_autonmi's __main__, until the event loop takes it over
    rc = layer_autonmi.main(argv)
    probe_out.write_text(json.dumps({"prompts": prompts, "calls": {r: list(v) for r, v in times.items()}}))
    return rc


# --- a fake model server's own process ---------------------------------------------------------------------------

class Received:
    """In place of FakeVllm.bodies: a summary of each request as it arrives, in constant memory."""

    def __init__(self) -> None:
        from fusion_miniapp import decide
        self.by_role, self.settings, self.lock = {}, Counter(), threading.Lock()
        self.asks = {decide.NOTE_ASK: "monitor", decide.SCORES_ASK: "evaluator", decide.GUIDANCE_ASK: "coordinator"}

    def append(self, body: dict) -> None:
        last = str(body["messages"][-1]["content"]).splitlines()[-1]
        role = self.asks.get(last, "generator" if "reply with a JSON list of" in last else "other")
        chars = sum(len(str(m.get("content") or "")) for m in body["messages"])
        rest = json.dumps({"role": role, "body": {k: v for k, v in body.items() if k != "messages"}}, sort_keys=True)
        with self.lock:
            r = self.by_role.setdefault(role, {"requests": 0, "max_chars": 0})
            r["requests"] += 1
            r["max_chars"] = max(r["max_chars"], chars)
            self.settings[rest] += 1


def fake_server(url_file: Path, stats_file: Path, latency_s: float, key_file: Path) -> None:
    """One FakeVllm until SIGTERM; then its summary in stats_file."""
    sys.path.insert(0, str(ROOT / "tests"))
    from fake_vllm import FakeVllm

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    with FakeVllm(latency_s=latency_s, model=NEMOTRON, api_key=key_file.read_text().strip()) as server:
        server.bodies = got = Received()
        tmp = url_file.with_name(url_file.name + ".tmp")
        tmp.write_text(server.url + "\n")
        os.replace(tmp, url_file)   # whole when the ladder sees it
        series, t0 = [], time.time()
        while not stop.wait(1.0):
            series.append([round(time.time() - t0, 1), server.running, server.waiting, server.successes])
        stats_file.write_text(json.dumps({
            "url": server.url, "latency_s": latency_s, "successes": server.successes,
            "running_max": server.running_max, "waiting_max": server.waiting_max, "by_role": got.by_role,
            "settings": [{"count": n, **json.loads(s)} for s, n in got.settings.most_common()],
            "series_t_running_waiting_successes": series}))


# --- the ladder ----------------------------------------------------------------------------------------------------

def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def power() -> dict:
    batt, settings, up = sh("pmset", "-g", "batt"), sh("pmset", "-g"), sh("uptime")
    mode = re.search(r"powermode\s+(\d+)", settings)
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "pmset_batt": batt,
            "powermode": int(mode.group(1)) if mode else None, "ac_power": "'AC Power'" in batt,
            "uptime": up, "load_1min": float(re.search(r"load averages?: ([\d.]+)", up).group(1)),
            "top_cpu": [" ".join((x.split(None, 1) + [""])[:1] + [Path((x.split(None, 1) + [""])[1]).name])
                        for x in sh("ps", "-Ao", "pcpu,comm", "-r").splitlines()[1:4]]}


def wait_for_quiet() -> dict:
    """The power state once the 1-minute load average is under LOAD_MAX (or after LOAD_WAIT_S)."""
    t0 = time.time()
    while (p := power())["load_1min"] >= LOAD_MAX and time.time() - t0 < LOAD_WAIT_S:
        time.sleep(15)
    p["waited_s"] = round(time.time() - t0)
    return p


def open_files() -> int:
    """The open-files limit of this process (and of the processes it starts), its soft value raised to OPEN_FILES
    when it is lower and the hard limit allows."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < OPEN_FILES:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(OPEN_FILES, hard), hard))
        except (ValueError, OSError):
            pass
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def rss_mb(pid: int) -> float:
    out = sh("ps", "-o", "rss=", "-p", str(pid))
    return int(out) / 1024 if out.isdigit() else 0.0


def tcp_counts(ports: list[int]) -> dict:
    """Established client connections to each fake server, and TIME_WAIT sockets, from netstat."""
    rows = [r.split() for r in sh("netstat", "-an", "-p", "tcp").splitlines()]
    est = {p: sum(1 for r in rows if len(r) > 5 and r[5] == "ESTABLISHED" and r[4] == f"127.0.0.1.{p}") for p in ports}
    return {"established_to_servers": est, "time_wait": sum(1 for r in rows if len(r) > 5 and r[5] == "TIME_WAIT")}


def lines(out: Path) -> list[dict]:
    """The campaigns' lines of a run that has ended (agents.jsonl)."""
    return [json.loads(x) for x in (out / "agents.jsonl").read_text().splitlines() if x.strip()]


def count_lines(out: Path) -> int:
    """The campaigns' lines written so far by a running run (each population's agents-p<index>.jsonl)."""
    return sum(f.read_bytes().count(b"\n") for f in out.glob("agents-p*.jsonl"))


def scrub(text: str, runs: Path) -> str:
    """The host name and local paths out of a file to be kept here."""
    for path, name in ((str(runs), "<runs>"), (str(ROOT), "<repo>"), (os.environ.get("TMPDIR", "").rstrip("/"),
                                                                       "$TMPDIR"), (str(Path.home()), "~")):
        if path:
            text = text.replace(path, name)
    for host in {socket.gethostname(), socket.gethostname().split(".")[0]}:
        text = text.replace(host, "laptop")
    return text.replace(Path.home().name, "user")


def public(m: dict) -> dict:
    """The summary as kept here: the power readings without the laptop's other processes or its battery's id
    (the whole summary stays under RUNS)."""
    m = json.loads(json.dumps(m))
    for k in ("power_before", "power_after"):
        m[k].pop("top_cpu", None)
        m[k]["pmset_batt"] = re.sub(r" \(id=\d+\)", "", m[k]["pmset_batt"])
    return m


def occupancy(rows: list[dict], calls: dict, slots: int) -> dict:
    """Agents of a campaign in a model call, by role and in all, over 15% to 85% of the agent phase: each answered
    call counts from when it was sent (t_end - service_s) to its reply (t_end)."""
    lo0, hi0 = min(r["t_start"] for r in rows), max(r["t_end"] for r in rows)
    lo, hi = lo0 + 0.15 * (hi0 - lo0), lo0 + 0.85 * (hi0 - lo0)
    span = (hi - lo) * len(rows)
    by = {}
    for role in ROLES:
        v = calls.get(role, [])
        by[role] = round(sum(max(0.0, min(v[i + 1], hi) - max(v[i], lo)) for i in range(0, len(v), 2)) / span, 4)
    return {"window_s": round(hi - lo, 3), "by_role": by, "all_of_5": round(sum(by.values()), 4),
            "monitor_per_slot": round(by["monitor"] / slots, 4), "modelled_all_of_5": MODELLED,
            "calls_seen": sum(len(v) // 2 for v in calls.values())}


def check1(hosted: int, live: dict, s: dict) -> dict:
    """All hosted agents live at once: from AUTONMI's rows (speed.agents_live); an interrupted record has none, so
    there from the live line's once-a-second count of agents started and not ended."""
    line = max((r[1] for r in (s.get("live") or {}).get("series", []) if isinstance(r[1], int)), default=None)
    out = {"hosted_agents": hosted, "agents_live_max": live.get("max"), "all_live_s": live.get("all_live_s"),
           "by_role_max": {r: v.get("max") for r, v in (live.get("by_role") or {}).items()},
           "live_line_agents_live_max": line}
    out["pass"] = (live.get("max") == hosted and (live.get("all_live_s") or 0) > 0) if live else line == hosted
    return out


def marks(name: str, spec: dict, out: Path, outer: dict) -> dict:
    rec = json.loads((out / "run_record.json").read_text())
    probe = json.loads((out.parent / f"{name}.probe.json").read_text()) if (out.parent / f"{name}.probe.json").exists() \
        else {"prompts": {}, "calls": {}}
    rows, s, c, k = lines(out), rec["speed"], rec["counts"], rec["knobs"]
    n = len(rows)
    live = s.get("agents_live") or {}
    m = {
        "rung": name,
        "configured": {"campaigns": spec["agents"], "hosted_agents_per_campaign": 3 + k["campaign"]["slots"],
                       "mode": k["campaign"]["mode"], "llm_mode": k["llm"]["mode"],
                       "stub_latency_s": k["llm"]["stub_latency_s"] if k["llm"]["mode"] == "stub" else None,
                       "max_in_flight": k["llm"]["max_in_flight"], "admission_tokens": k["llm"]["admission_tokens"],
                       "fake_servers": spec.get("fakes", 0), "fake_latency_s": spec.get("fake_latency_s"),
                       "record_extras": k["record"]["extras"], "population": k["outer"]["population"],
                       "generations": k["outer"]["generations"], "slots": k["campaign"]["slots"],
                       "knob_files": spec["knobs"] + (["stellar_ai.yaml", "stellar_ai_nemotron.yaml"]
                                                      if spec.get("stellar") else [])},
        "run": {**outer, "status": rec["status"], "campaign_lines": n, "campaigns_recorded": c["agents"],
                "wall_seconds": rec["wall_seconds"], "agent_phase_s": s.get("agent_phase_s"),
                "startup_s": s.get("startup_s"), "shutdown_s": s.get("shutdown_s"),
                "autonmi_ok": rec["autonmi"]["ok"], "autonmi_failed_checks": rec["autonmi"]["failed_checks"]},
        "check1_all_live": check1(c["hosted_agents"], live, s),
        "check3_rates_and_memory": {
            "evaluations": c["sims"], "evaluations_per_s_agent_phase": s["rates_per_s"]["evaluations"],
            "evaluations_per_s_wall": round(c["sims"] / rec["wall_seconds"], 1),
            "model_calls_per_s_agent_phase": s["rates_per_s"]["model_calls_all_agents"],
            "peak_rss_mb_record": rec["peak_rss_mb"]["main_process"], "rss_peak_mb_sampled_1s": outer["rss_peak_mb"],
            "mib_per_campaign": round(rec["peak_rss_mb"]["main_process"] / max(n, 1), 3)},
        "model_calls_per_campaign": {r: round(c["by_role"][r]["model_calls"] / max(n, 1), 2) for r in ROLES},
        "model_fallbacks": {r: c["by_role"][r]["model_fallbacks"] for r in ROLES},
        "model_requests_in_flight": {x: (s.get("model_requests_in_flight") or {}).get(x) for x in ("max", "mean")},
        "decoding_at_servers": {x: (s.get("decoding_at_servers") or {}).get(x) for x in ("max", "mean")},
        "live_line_model_requests_open_max": max((r[2] for r in (s.get("live") or {}).get("series", [])
                                                  if isinstance(r[2], int)), default=None),
        "queue_wait_s": s["queue_wait_s"], "sim_s_mean": s["sim_s"].get("mean"), "loop_lag": s["loop_lag"],
        "prompts_chars": {r: {"asked": p["asked"], "max_chars": p["max_chars"],
                              "mean_chars": round(p["sum_chars"] / max(p["asked"], 1)), "max_rows": p["max_rows"],
                              "of_bound": round(p["max_chars"] / BOUND, 4)} for r, p in sorted(probe["prompts"].items())},
        "prompt_bound_chars": BOUND,
    }
    if probe["calls"] and n:
        m["check2_in_a_model_call"] = occupancy(rows, probe["calls"], k["campaign"]["slots"])
    if rec["status"] == "interrupted":
        m["check4_sigterm"] = {"exit": outer["exit"], "status": rec["status"], "campaigns_recorded": c["agents"],
                               "coverage": s.get("coverage"),
                               "pass": outer["exit"] == 143 and c["agents"] > 0 and c["agents"] == n}
    return m


def settings_check(servers: list[dict]) -> dict:
    """Every request body the fake servers received carries the StellarAI knob file's settings: its model, thinking
    off, the reply cap, NVIDIA's sampling, no thinking budget; the Generator's alone its JSON schema. And every server
    answered requests of every role."""
    want = {"model": NEMOTRON, "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4096,
            "temperature": 1.0, "top_p": 0.95}
    bad = [s for sv in servers for s in sv["settings"]
           if {k: v for k, v in s["body"].items() if k != "response_format"} != want
           or ("response_format" in s["body"]) != (s["role"] == "generator")]
    used = {sv["url"]: {r: x["requests"] for r, x in sv["by_role"].items()} for sv in servers}
    return {"expected": want, "bodies_checked": sum(s["count"] for sv in servers for s in sv["settings"]),
            "bodies_not_as_expected": sum(s["count"] for s in bad), "requests_by_server_and_role": used,
            "pass": not bad and all(set(u) == set(ROLES) for u in used.values())}


def predict(runs: Path, name: str) -> float | None:
    """A 10,000-campaign rung's wall time from the 500 and 2,000 rungs in the same RUNS (None: not yet known).
    1 ms: a power law through the two asynchronous 1 ms rungs. 40 s: the 2,000-campaign 8 s rung's wall time in
    call times (the campaign's length is set by its call times), times 40 s, and at least the 1 ms rung's time
    (the event loop's work is the same)."""
    def wall(r: str):
        f = runs / f"{r}.ladder.json"
        return json.loads(f.read_text())["run"]["wall_seconds"] if f.exists() else None
    a, b = wall("r500_async_1ms"), wall("r2000_async_1ms")
    if a is None or b is None:
        return None
    ms1 = b * (5 ** (math.log(b / a) / math.log(4)))
    if "40s" not in name:
        return ms1
    c = wall("r2000_async_8s")
    return None if c is None else max(c / 8 * 40, ms1)


def rung(runs: Path, name: str) -> int:
    spec = RUNGS[name]
    out, keep = runs / name, HERE / name
    if out.exists():
        print(f"{name}: {out} exists; remove it to run again", file=sys.stderr)
        return 2
    limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if spec.get("fakes") and limit < 2 * 5 * spec["agents"]:   # 5 hosted agents per campaign
        print(f"{name}: not run: the open-files limit ({limit}) is under 2 per hosted agent; raise it (ulimit -n)",
              file=sys.stderr)
        return 3
    before = wait_for_quiet()
    if not before["ac_power"] or before["powermode"] == 1 or before["load_1min"] >= LOAD_MAX:
        print(f"{name}: not run: on battery, in Low Power Mode or loaded ({before['pmset_batt']!r}, powermode "
              f"{before['powermode']}, load {before['load_1min']} after {before['waited_s']} s)", file=sys.stderr)
        return 3
    runs.mkdir(parents=True, exist_ok=True)
    if spec["agents"] >= 10000:
        p = predict(runs, name)
        if p is None or p > LIMIT_S:
            why = {"rung": name, "skipped": True, "predicted_wall_s": p, "limit_s": LIMIT_S,
                   "why": "no prediction yet" if p is None else "the extrapolation is over the limit"}
            (runs / f"{name}.ladder.json").write_text(json.dumps(why, indent=1))
            print(json.dumps(why))
            return 0
        before["predicted_wall_s"] = round(p)
    argv = ["--preset", "hundredk", "--sim", "fixture", "--agents", str(spec["agents"])]
    argv += STELLAR if spec.get("stellar") else []
    for f in spec["knobs"]:
        argv += ["--knobs", str(HERE / f)]
    fakes, ports = [], []
    if spec.get("fakes"):
        fleet = runs / f"{name}.fleet"
        fleet.mkdir()
        key = fleet / "api-key"
        key.write_text(secrets.token_hex(24))   # a key for fake servers on this laptop only, removed below
        key.chmod(0o600)
        for i in range(spec["fakes"]):
            fakes.append(subprocess.Popen([sys.executable, __file__, "--fake-server", str(fleet / f"url-{i}"),
                                           str(fleet / f"server-{i}.json"), str(spec["fake_latency_s"]), str(key)],
                                          stdout=subprocess.DEVNULL, stderr=open(fleet / f"server-{i}.stderr", "w")))
        started = time.time()
        while not all((fleet / f"url-{i}").exists() for i in range(spec["fakes"])):
            if time.time() - started > 60 or any(f.poll() is not None for f in fakes):
                for f in fakes:
                    f.kill()
                key.unlink()
                print(f"{name}: not run: a fake server did not start ({fleet}/server-*.stderr)", file=sys.stderr)
                return 3
            time.sleep(0.2)
        urls = [(fleet / f"url-{i}").read_text().strip() for i in range(spec["fakes"])]
        ports = [int(u.rsplit(":", 1)[1]) for u in urls]
        (fleet / "endpoints").write_text("\n".join(urls) + "\n")
        argv += ["--llm", "openai", "--endpoints-file", str(fleet / "endpoints"), "--api-key-file", str(key),
                 "--model-wait", "120"]
    argv += ["--out", str(out)]
    cmd = [sys.executable, __file__, "--child", str(runs / f"{name}.probe.json"),
           "calls" if spec.get("calls") else "-", "--", *argv]
    command = scrub(" ".join(["python", "results/laptop_2026-11-25/ladder.py", "--child", "<probe>", *cmd[4:]]) + "\n",
                    runs)
    t0, peak, tcp, signalled, seen = time.time(), 0.0, [], None, 0
    with open(runs / f"{name}.stdout", "w") as so, open(runs / f"{name}.stderr", "w") as se:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=so, stderr=se)
        tick = 0
        while proc.poll() is None:
            time.sleep(1.0)
            tick += 1
            peak = max(peak, rss_mb(proc.pid))
            if ports and tick % 10 == 0:
                tcp.append({"t_s": round(time.time() - t0), **tcp_counts(ports)})
            if spec.get("sigterm_at") and signalled is None and out.exists():
                seen = count_lines(out)
                if seen >= spec["sigterm_at"]:
                    proc.send_signal(signal.SIGTERM)
                    signalled = {"t_s": round(time.time() - t0, 1), "campaign_lines_at_signal": seen}
        rc = proc.wait()
    t1 = time.time()
    for f in fakes:
        f.send_signal(signal.SIGTERM)
    for f in fakes:
        f.wait(timeout=60)
    outer = {"exit": rc, "outer_wall_s": round(t1 - t0, 1), "rss_peak_mb": round(peak, 1)}
    if signalled:
        outer["sigterm"] = {**signalled, "exit_after_signal_s": round(t1 - t0 - signalled["t_s"], 1)}
    m = marks(name, spec, out, outer)
    m["power_before"], m["power_after"] = before, power()
    if spec.get("fakes"):
        fleet = runs / f"{name}.fleet"
        (fleet / "api-key").unlink()
        servers = [json.loads((fleet / f"server-{i}.json").read_text()) for i in range(spec["fakes"])]
        m["fake_servers"] = [{k: v for k, v in sv.items() if k != "series_t_running_waiting_successes"}
                             for sv in servers]
        m["servers_running_max_sum"] = max((sum(x[i][1] for x in (sv["series_t_running_waiting_successes"]
                                                                    for sv in servers))
                                            for i in range(min(len(sv["series_t_running_waiting_successes"])
                                                               for sv in servers))), default=None)
        m["tcp_every_10s"] = tcp
        m["check5_request_settings"] = settings_check(servers)
    (runs / f"{name}.ladder.json").write_text(json.dumps(m, indent=1))
    if not m["power_after"]["ac_power"] or m["power_after"]["powermode"] == 1:
        print(f"{name}: not kept: the laptop was on battery or in Low Power Mode at the rung's end "
              f"({m['power_after']['pmset_batt']!r}, powermode {m['power_after']['powermode']})", file=sys.stderr)
        return 3
    keep.mkdir(parents=True, exist_ok=True)
    (keep / "command.txt").write_text(command)
    (keep / "run_record.json").write_text(scrub((out / "run_record.json").read_text(), runs))
    (keep / "ladder.json").write_text(scrub(json.dumps(public(m), indent=1), runs))
    print(json.dumps({"rung": name, "exit": rc, "status": m["run"]["status"], "wall_s": m["run"]["wall_seconds"]}))
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--child"]:
        return child(Path(argv[1]), argv[2] == "calls", argv[4:])
    if argv[:1] == ["--fake-server"]:
        fake_server(Path(argv[1]), Path(argv[2]), float(argv[3]), Path(argv[4]))
        return 0
    if argv[:1] == ["--list"] or len(argv) < 2:
        print("\n".join(f"{k}: {json.dumps(v)}" for k, v in RUNGS.items()))
        return 0
    runs = Path(argv[0]).resolve()
    print(f"open-files limit: {open_files()}", file=sys.stderr)
    for name in (list(RUNGS) if argv[1:] == ["all"] else argv[1:]):
        rc = rung(runs, name)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
