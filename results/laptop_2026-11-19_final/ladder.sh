#!/usr/bin/env bash
# The laptop ladder of 2026-11-19, re-run on the final build, flag off then on for each rung. Run from the fusion_miniapp folder.
# Stub LLM throughout (the knob default). Laptop functional checks, not performance claims.
set -euo pipefail
D=results/laptop_2026-11-19_final
PY=${PY:-.venv/bin/python}
# (a) 1,000 agents, 1 simulation each, the real evaluate on the smoke preset
$PY $D/ladder.py $D/a_real_1k_langgraph --agents 1000 --knobs $D/real_1k.yaml --sim real
$PY $D/ladder.py $D/a_real_1k_academy   --agents 1000 --knobs $D/real_1k.yaml --sim real --use-academy
# (b) 10,000 and (c) 100,000 agents, fixture simulation, hundredk shape at 10,000 agents per process
$PY $D/ladder.py $D/b_fixture_10k_langgraph  --agents 10000  --knobs $D/fixture_hundredk.yaml --sim fixture
$PY $D/ladder.py $D/b_fixture_10k_academy    --agents 10000  --knobs $D/fixture_hundredk.yaml --sim fixture --use-academy
$PY $D/ladder.py $D/c_fixture_100k_langgraph --agents 100000 --knobs $D/fixture_hundredk.yaml --sim fixture
$PY $D/ladder.py $D/c_fixture_100k_academy   --agents 100000 --knobs $D/fixture_hundredk.yaml --sim fixture --use-academy
