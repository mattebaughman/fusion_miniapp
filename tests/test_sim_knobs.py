"""knobs.py: presets, merging over smoke, and clear validation errors."""
import pytest
import yaml

from fusion_miniapp import knobs
from fusion_miniapp.knobs import KnobError


@pytest.mark.parametrize("name", knobs.PRESETS)
def test_presets_are_complete_and_valid(name):
    raw = yaml.safe_load((knobs.PRESETS_DIR / f"{name}.yaml").read_text())
    assert {s: set(v) for s, v in raw.items()} == {s: set(v) for s, v in knobs.SCHEMA.items()}
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
])
def test_validation_errors_name_the_knob(bad, message):
    with pytest.raises(KnobError) as info:
        knobs.validate(bad)
    assert message in str(info.value)


def test_load_errors(tmp_path):
    with pytest.raises(KnobError, match="unknown preset"):
        knobs.load(preset="huge")
    with pytest.raises(KnobError, match="not both"):
        knobs.load(tmp_path / "x.yaml", preset="smoke")
    with pytest.raises(KnobError, match="not found"):
        knobs.load(tmp_path / "missing.yaml")
    with pytest.raises(KnobError, match="'sim' section as a dict"):
        knobs.sim_section([1, 2])
