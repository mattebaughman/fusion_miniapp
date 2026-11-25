"""Knob files: load, validate, presets.

A knob file is YAML with nine sections (see CONTRACT.md). `knobs/smoke.yaml` at the project
root is the complete base: a preset, then any knob files in order, are merged over it, so a file
only needs the keys it changes (a machine file under knobs/machines/ sets only the machine's keys; the
hundredk and ignos_shape presets take the knobs newer than them from smoke). Unknown sections or keys, wrong types and out-of-range values raise KnobError
with the dotted name of the offending knob. This module imports no torch.
"""
from __future__ import annotations

import copy
import functools
import math
import re
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
# knobs/ at the project root (source tree, editable install), or packaged inside the wheel (pyproject)
PRESETS_DIR = next((d for d in (_HERE / "knobs", _HERE.parents[1] / "knobs") if d.is_dir()), _HERE / "knobs")
PRESETS = ("smoke", "hundredk", "ignos_shape")

# (kind, *args): ("int", lo, hi) ("num", lo, hi) ("nums", lo, hi) ("bool",) ("choice", options) ("str",) ("env",)
# ("optstr",) ("optint", lo) (null or an integer) ("optnum", lo, hi[, "open"]) (null or a number; "open": above lo)
# ("autoint", lo) ("auto" or an integer) ("onoff",) (null, off or on). YAML reads a bare off or on as false or
# true: in an onoff knob, or a choice that has "off", they are kept as "off" and "on".
SCHEMA = {
    "sim": {
        "nodes_per_plane": ("int", 16, None),
        "planes": ("int", 1, None),
        "rollout_steps": ("int", 1, None),
        "puffs": ("int", 1, None),
        "fixed_point_cap": ("int", 1, None),
        "model_width": ("int", 2, None),
        "checkpoint_mb": ("num", 0, None),
        "device": ("choice", ("auto", "cpu", "cuda", "mps")),
        "cold_start_fraction": ("num", 0, 1),
        "keep_artifacts": ("bool",),
        "work_root": ("optstr",),
        "threads_per_worker": ("autoint", 1),   # CPU threads per simulation worker; auto = CPUs / workers
    },
    "agents": {
        "count": ("int", 1, None),
        "first": ("int", 0, None),            # campaigns first .. first+count-1, each seeded with its index
        "per_process": ("int", 1, None),
        "max_live": ("optint", 1),            # campaigns live at once per agent process; null = all of them
        "start_together": ("bool",),          # the AUTONMI layer: generation 0 waits until every agent has started
        "cost_scale": ("nums", 0, None),      # a number, or a list: agent i uses entry i % len (each > 0)
        "thinking_budget": ("int", 0, None),
        "budget_scales_with_cost": ("bool",),
    },
    "tools": {
        "constraints_lookup": ("bool",),
        "abstract_lookup": ("bool",),
        "latency_ms": ("num", 0, None),
    },
    "llm": {
        # off: no model calls, every agent decides on numbers alone (the asynchronous mode only)
        "mode": ("choice", ("stub", "openai", "off")),
        "base_url_env": ("env",),
        "api_key_env": ("env",),
        "model": ("str",),
        "send_thinking_budget": ("bool",),    # false for servers that reject vLLM's thinking_token_budget
        "timeout_s": ("num", 0.001, None),
        "max_retries": ("int", 0, None),
        "min_model_share": ("num", 0, 1),     # openai mode: fail the run if fewer refill slots came from the model
        "max_empty_share": ("num", 0, 1),     # openai mode: fail the run if more replies were empty
        "stub_latency_s": ("num", 0, None),   # the stub's seconds per call (emulated: never a model's decoding)
        "max_in_flight": ("int", 1, None),    # model requests admitted at once per model server (the AUTONMI layer)
        "admission_tokens": ("int", 0, None),  # tokens admitted at once per model server (the AUTONMI layer; 0: none)
        # the AUTONMI layer's request settings; null leaves each call as before
        "thinking": ("onoff",),               # off: chat_template_kwargs enable_thinking false, and no budget
        "reply_cap": ("optint", 1),           # max_tokens: this (plus agents.thinking_budget unless thinking is off)
        "temperature": ("optnum", 0, None),   # sent with every call; null: the server's default
        "top_p": ("optnum", 0, 1, "open"),    # sent with every call when set (vLLM takes (0, 1])
        "outage_s": ("num", 0, None),         # how long a call waits out a server that went away (AUTONMI's Recovery)
    },
    "workers": {
        "sim_workers": ("int", 1, None),
        "gpu_per_worker": ("bool",),          # the AUTONMI layer, --sim real: inner-loop worker i sees GPU i mod n
    },
    "outer": {                                # each campaign's IGNOS outer loop (parent run_evolutionary_loop)
        "population": ("int", 1, None),       # P
        "generations": ("int", 1, None),      # G
        "generator": ("choice", ("llm", "mutation")),   # llm: the model replaces mutants; mutation: the parent's loop
        "rng": ("choice", ("shared", "per_agent")),     # shared: run_evolutionary_loop; per_agent: evo_agents.py
                                                        # (the asynchronous mode always has per_agent's two streams)
        "logs": ("choice", ("per_agent", "none")),      # the parent's evolution_log.csv/.jsonl, config.json, best_result.json
        "wall_weight": ("num", 0, 1),
        "noise": ("num", 0, None),
        "parent_stdout": ("choice", ("keep", "drop")),  # the parent's per-generation prints
    },
    "monitor": {                              # the inner-loop trace (trace.py): added instrumentation, not in IGNOS
        "capture": ("bool",),                 # each evaluation's record carries `trace` (the AUTONMI layer)
        "fixture_trace": ("choice", ("smooth", "hash")),   # --sim fixture: its synthetic trace
        "weight": ("num", 0, 1),              # the Monitor's m in a result's score (the asynchronous mode)
    },
    "campaign": {                             # the AUTONMI layer: each campaign's Monitor slots and candidate queue
        "slots": ("int", 1, None),            # Monitor agents per campaign, so 3 + slots hosted agents
        # generations: the parent's generation barrier, as IGNOS; async (the AUTONMI layer only, validate's
        # `modes`): each result applied as it arrives, each agent making its own model call unless llm.mode is off
        "mode": ("choice", ("generations", "async")),
        "warmup": ("int", 1, None),           # traces the good and the bad side each hold before m is not 0; in
                                              # the asynchronous mode also the first results, kept whatever their score
        "notes": ("bool",),                   # the generations mode: also a Monitor call per candidate (its note) and
                                              # an Evaluator call per generation (its assessment); they change no result
    },
    "record": {
        "extras": ("bool",),                  # per-candidate detail (the Monitor's numbers and note, the Evaluator's
                                              # scores, the Coordinator's guidance) and the model calls' (each call's
                                              # server, the request settings, each role's time in a call, the model
                                              # servers' setup files); false: counts only
    },
}


class KnobError(ValueError):
    """A knob file or dict is malformed; the message names the knob."""


def _check(name: str, value, rule: tuple) -> None:
    kind = rule[0]
    if kind == "nums":
        for v in (value if isinstance(value, list) and value else [value]):
            _check(name, v, ("num", *rule[1:]))
    elif kind == "optint":
        if value is not None:
            _check(name, value, ("int", rule[1], None))
    elif kind == "optnum":
        if value is not None:
            _check(name, value, ("num", *rule[1:]))
            if len(rule) > 3 and value == rule[1]:   # an open lower end
                raise KnobError(f"{name} must be in ({rule[1]}, {rule[2]}], got {value!r}")
    elif kind == "autoint":
        if value != "auto":
            if isinstance(value, str):
                raise KnobError(f"{name} must be auto or an integer, got {value!r}")
            _check(name, value, ("int", rule[1], None))
    elif kind in ("int", "num"):
        types = (int,) if kind == "int" else (int, float)
        if isinstance(value, bool) or not isinstance(value, types):
            raise KnobError(f"{name} must be {'an integer' if kind == 'int' else 'a number'}, got {value!r}")
        if not math.isfinite(value):
            raise KnobError(f"{name} must be finite, got {value!r}")
        lo, hi = rule[1], rule[2]
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise KnobError(f"{name} must be in [{lo}, {'inf' if hi is None else hi}], got {value!r}")
    elif kind == "onoff" and value not in (None, "off", "on"):
        raise KnobError(f"{name} must be off, on or null, got {value!r}")
    elif kind == "bool" and not isinstance(value, bool):
        raise KnobError(f"{name} must be true or false, got {value!r}")
    elif kind == "choice" and value not in rule[1]:
        raise KnobError(f"{name} must be one of {list(rule[1])}, got {value!r}")
    elif kind == "str" and not (isinstance(value, str) and value):
        raise KnobError(f"{name} must be a non-empty string, got {value!r}")
    elif kind == "optstr" and not (value is None or (isinstance(value, str) and value)):
        raise KnobError(f"{name} must be a path string or null, got {value!r}")
    elif kind == "env" and not (isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)):
        raise KnobError(f"{name} must be the NAME of an environment variable (not its value), got {value!r}")
    if name == "agents.cost_scale" and kind == "num" and value <= 0:
        raise KnobError(f"agents.cost_scale must be > 0, got {value!r}")
    if name == "sim.model_width" and value % 2:
        raise KnobError(f"sim.model_width must be even (the grid stand-in uses 2 attention heads), got {value}")


def _merge_checked(base: dict, overlay: dict, where: str) -> dict:
    """Return base updated with overlay, checking every overlay section, key and value."""
    if not isinstance(overlay, dict):
        raise KnobError(f"{where}: expected a mapping of sections, got {type(overlay).__name__}")
    out = copy.deepcopy(base)
    for section, values in overlay.items():
        if section not in SCHEMA:
            raise KnobError(f"{where}: unknown section {section!r}; allowed: {list(SCHEMA)}")
        if values is None:
            continue
        if not isinstance(values, dict):
            raise KnobError(f"{where}: section {section!r} must be a mapping, got {values!r}")
        for key, value in values.items():
            if key not in SCHEMA[section]:
                raise KnobError(f"{where}: unknown knob {section}.{key}; allowed: {list(SCHEMA[section])}")
            rule = SCHEMA[section][key]
            if isinstance(value, bool) and (rule[0] == "onoff" or (rule[0] == "choice" and "off" in rule[1])):
                value = "on" if value else "off"   # YAML's bare off and on
            _check(f"{section}.{key}", value, rule)
            out[section][key] = value
    return out


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise KnobError(f"knob file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise KnobError(f"{path}: not valid YAML: {exc}") from None
    return data or {}


@functools.lru_cache(maxsize=1)
def _base_cached() -> dict:
    path = PRESETS_DIR / "smoke.yaml"
    empty = {section: {} for section in SCHEMA}
    base = _merge_checked(empty, _read_yaml(path), str(path))
    missing = [f"{s}.{k}" for s in SCHEMA for k in SCHEMA[s] if k not in base[s]]
    if missing:
        raise KnobError(f"{path} must set every knob; missing: {missing}")
    return base


def base() -> dict:
    """The smoke preset, complete; a fresh copy each call."""
    return copy.deepcopy(_base_cached())


AUTONMI_ONLY = ("thinking", "reply_cap", "temperature", "top_p")   # the llm knobs run.py's layers do not read


def validate(knobs: dict | None, where: str = "knobs", modes: tuple = ("generations",)) -> dict:
    """Check a (possibly partial) knob dict and return it merged over the smoke preset. `modes`: the
    campaign.mode values the caller runs (run.py's layers the generations mode alone, and they refuse the knobs
    only the AUTONMI layer reads, set: AUTONMI_ONLY, campaign.notes and workers.gpu_per_worker; the AUTONMI layer
    also async)."""
    out = _merge_checked(base(), knobs or {}, where)
    if out["campaign"]["mode"] not in modes:
        raise KnobError(f"campaign.mode {out['campaign']['mode']} is not run by this layer (it runs "
                        f"{', '.join(modes)}); the asynchronous mode runs on the AUTONMI layer (layer_autonmi)")
    if out["llm"]["mode"] == "off" and out["campaign"]["mode"] != "async":
        raise KnobError("llm.mode off (no model calls) is the asynchronous mode's only: the generations mode's "
                        "refill and summary calls need llm.mode stub or openai")
    if out["campaign"]["mode"] == "async" and out["outer"]["generator"] == "mutation" and out["llm"]["mode"] != "off":
        raise KnobError("outer.generator mutation is the generations mode's: in the asynchronous mode the Generator's "
                        "model call reconciles every candidate; for no model calls set llm.mode off")
    if "async" not in modes:   # run.py's layers send each call as before and make no Monitor or Evaluator call
        unread = [f"llm.{k}" for k in AUTONMI_ONLY if out["llm"][k] is not None] + \
            (["campaign.notes"] if out["campaign"]["notes"] else []) + \
            (["workers.gpu_per_worker"] if out["workers"]["gpu_per_worker"] else [])
        if unread:
            raise KnobError(f"{', '.join(unread)} set: only the AUTONMI layer (layer_autonmi) reads these knobs, "
                            "and this layer would ignore them")
    return out


def load(path: str | Path | None = None, preset: str | None = None, paths=()) -> dict:
    """The smoke preset, then the named preset, then each knob file in order (`path`, then `paths`),
    each merged over the result so far."""
    files = ([path] if path is not None else []) + list(paths)
    out = base()
    if preset is not None:
        if preset not in PRESETS:
            raise KnobError(f"unknown preset {preset!r}; available: {list(PRESETS)}")
        p = PRESETS_DIR / f"{preset}.yaml"
        out = _merge_checked(out, _read_yaml(p), str(p))
    for f in files:
        f = Path(f)
        out = _merge_checked(out, _read_yaml(f), str(f))
    return out


def monitor_section(monitor: dict | None) -> dict | None:
    """The `monitor` section that evaluate() takes, merged over the smoke preset's and checked; None stays
    None (no trace)."""
    if monitor is None:
        return None
    if not isinstance(monitor, dict):
        raise KnobError(f"evaluate's monitor must be the 'monitor' section as a dict, got {type(monitor).__name__}")
    return _merge_checked({"monitor": _base_cached()["monitor"]}, {"monitor": monitor}, "monitor knobs")["monitor"]


def sim_section(sim: dict | None) -> dict:
    """The `sim` section that evaluate() takes, merged over the smoke preset's and checked (that section alone, as
    monitor_section: once per evaluation, a copy of the whole preset cost the fixture about 0.3 ms)."""
    if sim is None:
        return base()["sim"]
    if not isinstance(sim, dict):
        raise KnobError(f"evaluate knobs must be the 'sim' section as a dict, got {type(sim).__name__}")
    return _merge_checked({"sim": _base_cached()["sim"]}, {"sim": sim}, "sim knobs")["sim"]
