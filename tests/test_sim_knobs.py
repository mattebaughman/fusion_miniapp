"""knobs.py: presets, merging over smoke, and clear validation errors."""
import pytest
import yaml

from fusion_miniapp import knobs
from fusion_miniapp.knobs import KnobError


# knobs added after the hundredk and ignos_shape presets, which take them from smoke (hundredk.yaml is kept as it is)
NEWER = {"monitor.weight", "campaign.slots", "campaign.mode", "campaign.warmup", "record.extras", "campaign.notes",
         "llm.thinking", "llm.reply_cap", "llm.temperature", "llm.top_p", "llm.outage_s", "workers.gpu_per_worker"}


@pytest.mark.parametrize("name", knobs.PRESETS)
def test_presets_are_complete_and_valid(name):
    """smoke sets every knob; the other presets every knob but the newer ones, which they take from smoke."""
    raw = yaml.safe_load((knobs.PRESETS_DIR / f"{name}.yaml").read_text())
    have = {f"{s}.{k}" for s, v in raw.items() for k in v}
    every = {f"{s}.{k}" for s, v in knobs.SCHEMA.items() for k in v}
    assert have <= every and every - have == (set() if name == "smoke" else NEWER)
    assert knobs.load(preset=name) == knobs.validate(raw)


def test_partial_file_merges_over_smoke(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text("sim: {rollout_steps: 3}\nagents: {count: 7}\n")
    k = knobs.load(path)
    assert k["sim"]["rollout_steps"] == 3 and k["agents"]["count"] == 7
    assert k["sim"]["puffs"] == knobs.base()["sim"]["puffs"]
    assert knobs.sim_section(None) == knobs.base()["sim"]
    assert knobs.sim_section({"puffs": 4})["puffs"] == 4
    assert knobs.validate({"agents": {"cost_scale": [0.5, 1, 4]}})["agents"]["cost_scale"] == [0.5, 1, 4]


@pytest.mark.parametrize("bad, message", [
    ({"sim": {"rollout_step": 2}}, "unknown knob sim.rollout_step"),
    ({"simulation": {}}, "unknown section 'simulation'"),
    ({"sim": {"puffs": 0}}, "sim.puffs must be in [1, inf]"),
    ({"sim": {"puffs": 2.5}}, "sim.puffs must be an integer"),
    ({"sim": {"keep_artifacts": "yes"}}, "sim.keep_artifacts must be true or false"),
    ({"sim": {"device": "cuda:1"}}, "sim.device must be one of"),
    ({"sim": {"cold_start_fraction": 1.5}}, "sim.cold_start_fraction must be in [0, 1]"),
    ({"sim": {"model_width": 9}}, "sim.model_width must be even"),
    ({"llm": {"api_key_env": "sk-not a name"}}, "llm.api_key_env must be the NAME"),
    ({"llm": {"mode": "vllm"}}, "llm.mode must be one of"),
    ({"agents": {"cost_scale": [1, "x"]}}, "agents.cost_scale must be a number"),
    ({"agents": {"cost_scale": []}}, "agents.cost_scale must be a number"),
    ({"workers": {"gpus_for_sim": 1}}, "unknown knob workers.gpus_for_sim"),
    ({"monitor": {"capture": "yes"}}, "monitor.capture must be true or false"),
    ({"monitor": {"fixture_trace": "random"}}, "monitor.fixture_trace must be one of"),
    ({"monitor": {"weight": 1.5}}, "monitor.weight must be in [0, 1]"),
    ({"campaign": {"slots": 0}}, "campaign.slots must be in [1, inf]"),
    ({"campaign": {"mode": "steady"}}, "campaign.mode must be one of ['generations', 'async']"),
    ({"campaign": {"mode": "async"}}, "campaign.mode async is not run by this layer"),   # (run.py's layers)
    ({"campaign": {"warmup": 0}}, "campaign.warmup must be in [1, inf]"),
    ({"record": {"extras": "yes"}}, "record.extras must be true or false"),
    ({"campaign": {"notes": 1}}, "campaign.notes must be true or false"),
    ({"llm": {"thinking": "maybe"}}, "llm.thinking must be off, on or null"),
    ({"llm": {"reply_cap": 0}}, "llm.reply_cap must be in [1, inf]"),
    ({"llm": {"temperature": -0.5}}, "llm.temperature must be in [0, inf]"),
    ({"llm": {"top_p": 1.5}}, "llm.top_p must be in [0, 1]"),
    ({"llm": {"top_p": 0}}, "llm.top_p must be in (0, 1]"),   # vLLM's range
    ({"llm": {"thinking": "off", "top_p": 0.95}}, "llm.thinking, llm.top_p set: only the AUTONMI layer"),   # run.py's
    ({"campaign": {"notes": True}}, "campaign.notes set: only the AUTONMI layer"),
    ({"llm": {"top_p": "high"}}, "llm.top_p must be a number"),
    ({"llm": {"outage_s": -1}}, "llm.outage_s must be in [0, inf]"),
    ({"llm": {"mode": "off"}}, "llm.mode off (no model calls) is the asynchronous mode's only"),
])
def test_validation_errors_name_the_knob(bad, message):
    with pytest.raises(KnobError) as info:
        knobs.validate(bad)
    assert message in str(info.value)


def test_load_errors(tmp_path):
    with pytest.raises(KnobError, match="unknown preset"):
        knobs.load(preset="huge")
    (tmp_path / "m.yaml").write_text("sim: {device: cuda}\n")
    layered = knobs.load(preset="hundredk", paths=[tmp_path / "m.yaml"])   # preset, then files in order
    assert layered["agents"]["count"] == 100000 and layered["sim"]["device"] == "cuda"
    with pytest.raises(KnobError, match="not found"):
        knobs.load(tmp_path / "missing.yaml")
    with pytest.raises(KnobError, match="'sim' section as a dict"):
        knobs.sim_section([1, 2])
    assert knobs.monitor_section(None) is None                   # evaluate records no trace
    assert knobs.monitor_section({}) == knobs.base()["monitor"] == {"capture": True, "fixture_trace": "smooth",
                                                                      "weight": 0.25}
    assert knobs.monitor_section({"capture": False}) == {"capture": False, "fixture_trace": "smooth", "weight": 0.25}
    base = knobs.base()   # the new knobs' defaults: today's results and requests
    assert (base["campaign"], base["record"]) == ({"slots": 2, "mode": "generations", "warmup": 4, "notes": False},
                                                  {"extras": False})
    assert {k: base["llm"][k] for k in ("thinking", "reply_cap", "temperature", "top_p", "outage_s")} == {
        "thinking": None, "reply_cap": None, "temperature": None, "top_p": None, "outage_s": 1800}
    with pytest.raises(KnobError, match="'monitor' section as a dict"):
        knobs.monitor_section([1, 2])


def test_the_asynchronous_mode_is_the_autonmi_layers(tmp_path):
    """campaign.mode async loads, and validate takes it only from a caller that runs it (the AUTONMI layer's
    modes); run.py's layers, which call validate as it is, refuse it."""
    (tmp_path / "a.yaml").write_text("campaign: {mode: async}\n")
    k = knobs.load(tmp_path / "a.yaml")
    assert k["campaign"]["mode"] == "async"
    assert knobs.validate(k, modes=("generations", "async"))["campaign"]["mode"] == "async"
    with pytest.raises(KnobError, match="runs on the AUTONMI layer"):
        knobs.validate(k)
    (tmp_path / "t.yaml").write_text("llm: {thinking: off, mode: off}\n")   # YAML's bare off is false: kept as "off"
    assert {k: v for k, v in knobs.load(tmp_path / "t.yaml")["llm"].items() if k in ("thinking", "mode")} == {
        "thinking": "off", "mode": "off"}
    assert knobs.validate({"llm": {"thinking": True}}, modes=("generations", "async"))["llm"]["thinking"] == "on"
    off = {**k, "llm": {**k["llm"], "mode": "off"}}   # no model calls: the asynchronous mode's only
    assert knobs.validate(off, modes=("generations", "async"))["llm"]["mode"] == "off"
