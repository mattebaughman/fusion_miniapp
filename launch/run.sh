#!/usr/bin/env bash
# The shared run script: activate the environment named by FUSION_ENV, then run the mini-app with
# this script's arguments (or the device probe, if the first argument is "probe").
#
#   FUSION_ENV   a virtualenv directory (one with bin/activate) or a conda environment name or path; no default
#
#   launch/run.sh --agents 100 --preset smoke [--use-academy] [--sim real|fixture] [--out DIR]
#   launch/run.sh probe
set -euo pipefail
: "${FUSION_ENV:?set FUSION_ENV to the environment to activate (a virtualenv directory or a conda environment)}"
if [[ -f "$FUSION_ENV/bin/activate" ]]; then
  source "$FUSION_ENV/bin/activate"
else
  eval "$(conda shell.bash hook)"   # no conda init: nothing is written to the user's shell files
  conda activate "$FUSION_ENV"
fi
if [[ "${1:-}" == probe ]]; then
  exec python "$(dirname "$0")/probe.py"
fi
exec python -m fusion_miniapp.run "$@"
