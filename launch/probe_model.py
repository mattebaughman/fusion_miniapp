"""Check the model servers before a run: one request of each agent role to every server, with the run's settings.

launch/stellar_ai_autonmi.sbatch runs it once the model servers answer and before the run starts, with the run's
own arguments (the AUTONMI layer's: --endpoints-file, --api-key-file, --preset, --knobs, ...), so that it loads the
run's knobs and sends what the run sends: the request fields of layer_autonmi.request_fields over the transports'
settings (thinking off, the reply cap, temperature, top_p; the Generator's JSON schema as response_format), and each
role's prompt as decide.py builds it in the asynchronous mode, for a campaign's first two candidates evaluated with
the fixture inner loop (synthetic values: a check of the servers, not of a campaign).

    python launch/probe_model.py [--logs DIR] [--timeout S] [--idle] <the run's layer_autonmi arguments>

It stops the launch (exit 1) on:
- a server whose /v1/models does not answer, or does not list llm.model;
- with --idle, a server whose /metrics shows requests running or waiting (another run's would land in this one's);
- a prompt whose characters (at least its tokens) plus max_tokens exceed the max_model_len a server reports;
- a request that ends in an HTTP error, a time-out (--timeout, 300 s: this check's own, not the run's) or an empty
  reply;
- a Traceback or EngineDeadError in a vllm-<i>.log under DIR (--logs: the model jobs' folders), except in what a
  server prints once it is stopping: from a line with [shutdown] until the next line that starts with "=== "
  (model.sh's line before each start).
It reports each server's replies without judging them: the finish reason, whether the role's parse could use the
reply (for the Generator, how many of the 7 parameters each candidate's object filled), token usage and latency.
One JSON line per server on stdout, the verdict on stderr. The servers' key is read from its file and never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fusion_miniapp import decide, knobs as knob_files, layer_autonmi, sim, tools

LOG_PROBLEMS = ("Traceback", "EngineDeadError")
LOG_START, LOG_STOPPING = "=== ", "[shutdown]"   # model.sh's line before each start; vLLM's lines once it is stopping


def role_requests(knobs: dict) -> list[dict]:
    """One request of each role of the asynchronous mode, as its agents make them for campaign a0: the
    Generator's on its first take, a Monitor's note on its first candidate, the Evaluator's scores and the
    Coordinator's guidance on both. Each: role, messages, reply_tokens, response_format, parse."""
    agent, seed = "a0", 0
    abstract = {"abstract": asyncio.run(tools.abstract(agent))} if knobs["tools"]["abstract_lookup"] else {}
    gst, calls = decide.async_profiles(decide.async_generator_state(agent, seed, knobs, abstract), knobs)
    s, n = decide.sim_knobs(knobs, seed), len(calls)
    results = [sim.evaluate_fixture(knobs=s, agent_id=agent, seed=seed, params=c["params"],
                                    candidate_idx=c["candidate_idx"], generation=c["generation"]) for c in calls]
    reports = [{"candidate": c["candidate_idx"], "result": r, "t0": 0.0, "source": c["source"], "filled": None,
                "m": None} for c, r in zip(calls, results)]
    _, scored, _ = decide.async_evaluate([], reports, knobs)
    constraints = {"constraints": asyncio.run(tools.constraints())} if knobs["tools"]["constraints_lookup"] else {}
    with decide.parent_stdout(knobs):
        cst, rows = decide.async_on_scored(decide.async_coordinator_state(agent, seed, knobs, constraints, 0.0),
                                           scored, knobs, None)
    ks = [x["candidate"] for x in scored]
    return [
        {"role": "generator", "messages": decide.async_generator_messages(gst, calls, knobs),
         "reply_tokens": decide.reply_tokens(n), "response_format": decide.candidates_schema(n),
         "parse": lambda r: decide.parse_reconcile(r, n)},
        {"role": "monitor", "messages": decide.note_messages(results[0], None), "reply_tokens": decide.NOTE_TOKENS,
         "parse": decide.parse_text},
        {"role": "evaluator", "messages": decide.scores_messages([{**x, "numbers": None, "note": ""} for x in scored]),
         "reply_tokens": decide.scores_tokens(len(scored)), "parse": lambda r: decide.parse_scores(r, ks)},
        {"role": "coordinator", "messages": decide.guidance_messages(cst, rows, knobs),
         "reply_tokens": decide.GUIDANCE_TOKENS, "parse": decide.parse_text},
    ]


def request_body(knobs: dict, req: dict) -> dict:
    """The request's body as the run's fleet sends it: AUTONMI's transport settings for the served model, then the
    call's fields (layer_autonmi.sent), the Generator's schema, and the messages."""
    from autonmi.models import THINKING_OFF

    k = knobs["llm"]
    model = {"name": k["model"], "thinking_off": THINKING_OFF if k["thinking"] == "off" else {}}
    body = layer_autonmi.sent(layer_autonmi.transport_settings(model),
                              layer_autonmi.request_fields(knobs, decide.budget(knobs, 0), req["reply_tokens"]))
    if req.get("response_format"):
        body["response_format"] = req["response_format"]
    return {**body, "messages": req["messages"]}


def _open(url: str, key: str | None, timeout: float, data: bytes | None = None) -> bytes:
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if key and key != "EMPTY":
        headers["Authorization"] = f"Bearer {key}"
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as r:
        return r.read()


def _gauges(text: str) -> dict:
    """vLLM's requests running and waiting in a /metrics text, summed over its engines."""
    out = {"num_requests_running": 0.0, "num_requests_waiting": 0.0}
    for line in text.splitlines():
        name = line.split("{")[0].split(" ")[0].removeprefix("vllm:")
        if name in out and not line.startswith("#"):
            out[name] += float(line.rsplit(" ", 1)[1])
    return out


def check_server(url: str, key: str | None, knobs: dict, reqs: list[dict], timeout: float, idle: bool) -> dict:
    """One server: its model, its context, with `idle` its load, then each role's request in turn."""
    base, name = url.rstrip("/"), knobs["llm"]["model"]
    out = {"server": url, "max_model_len": None, "calls": [], "problems": []}
    try:
        data = json.loads(_open(base + "/v1/models", key, timeout)).get("data", [])
    except Exception as exc:   # noqa: BLE001 - any failure stops the launch, with its reason
        out["problems"].append(f"/v1/models did not answer: {type(exc).__name__}: {exc}")
        return out
    entry = next((m for m in data if m.get("id") == name), None)
    if entry is None:
        out["problems"].append(f"serves {[m.get('id') for m in data]}, not llm.model {name!r}")
        return out
    context = out["max_model_len"] = entry.get("max_model_len")
    if idle:
        try:
            load = _gauges(_open(base + "/metrics", key, timeout).decode())
        except Exception as exc:   # noqa: BLE001
            out["problems"].append(f"/metrics did not answer: {type(exc).__name__}: {exc}")
            return out
        if load["num_requests_running"] or load["num_requests_waiting"]:
            out["problems"].append(f"not idle: {load['num_requests_running']:.0f} requests running and "
                                   f"{load['num_requests_waiting']:.0f} waiting")
            return out
    for req in reqs:
        body = request_body(knobs, req)
        chars = sum(len(str(m.get("content") or "")) for m in body["messages"])
        call = {"role": req["role"], "prompt_chars": chars, "max_tokens": body["max_tokens"]}
        out["calls"].append(call)
        if isinstance(context, int) and chars + body["max_tokens"] > context:
            out["problems"].append(f"{req['role']}: {chars} prompt characters + max_tokens {body['max_tokens']} > "
                                   f"max_model_len {context}")
            continue
        t0 = time.perf_counter()
        try:
            reply = json.loads(_open(base + "/v1/chat/completions", key, timeout, json.dumps(body).encode()))
        except urllib.error.HTTPError as exc:
            out["problems"].append(f"{req['role']}: HTTP {exc.code} {exc.reason}")
            continue
        except TimeoutError:
            out["problems"].append(f"{req['role']}: no reply within {timeout:g} s")
            continue
        except Exception as exc:   # noqa: BLE001
            out["problems"].append(f"{req['role']}: {type(exc).__name__}: {exc}")
            continue
        choice = (reply.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usable = req["parse"](content)
        call.update(latency_s=round(time.perf_counter() - t0, 3), finish_reason=choice.get("finish_reason"),
                    usage=reply.get("usage"), usable=usable is not None)
        if req["role"] == "generator" and usable is not None:
            call["parameters_filled"] = [len(p) for p in usable]
        if not content.strip():
            out["problems"].append(f"{req['role']}: an empty reply")
    return out


def log_problems(folder: str | None) -> list[str]:
    """The first Traceback or EngineDeadError line of each vllm-<i>.log under `folder`, outside what a server prints
    once it is stopping (from a LOG_STOPPING line until the next line that starts with LOG_START)."""
    out = []
    for f in sorted(Path(folder).rglob("vllm-*.log")) if folder else []:
        stopping = False
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if line.startswith(LOG_START):
                stopping = False
            elif LOG_STOPPING in line:
                stopping = True
            elif not stopping and any(p in line for p in LOG_PROBLEMS):
                out.append(f"{f}: line {i}: {line.strip()[:200]}")
                break
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--logs", metavar="DIR", help="the model jobs' folders, searched for vllm-<i>.log")
    p.add_argument("--timeout", type=float, default=300.0, help="seconds per request (this check's own)")
    p.add_argument("--idle", action="store_true", help="also require every server to have no request running "
                                                       "or waiting (a model job reused from an earlier run)")
    mine, rest = p.parse_known_args(argv)
    a = layer_autonmi.parse(rest)
    try:
        knobs = layer_autonmi.load_knobs(a)   # as the run loads them
    except knob_files.KnobError as exc:
        print(f"probe_model: {exc}", file=sys.stderr)
        return 1
    from autonmi.models import read_endpoints

    urls = read_endpoints(a.endpoints_file) if a.endpoints_file else []
    if knobs["llm"]["mode"] != "openai" or not urls:
        print("probe_model: needs llm.mode openai (or --llm openai) and an --endpoints-file that lists the servers",
              file=sys.stderr)
        return 1
    key = Path(a.api_key_file).read_text().strip() if a.api_key_file else None
    reqs = role_requests(knobs)
    with ThreadPoolExecutor(max_workers=min(64, len(urls))) as pool:
        results = list(pool.map(lambda u: check_server(u, key, knobs, reqs, mine.timeout, mine.idle), urls))
    for r in results:
        print(json.dumps(r), flush=True)
    problems = [f"{r['server']}: {x}" for r in results for x in r["problems"]] + log_problems(mine.logs)
    if problems:
        print("probe_model: the run would not start:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"probe_model: {len(urls)} servers answered each of the {len(reqs)} roles' requests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
