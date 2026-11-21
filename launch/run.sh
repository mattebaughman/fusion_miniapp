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
set +u   # conda and environment activation hooks may read unset shell variables
if [[ -f "$FUSION_ENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$FUSION_ENV/bin/activate" >&2
else
  conda_hook=$(conda shell.bash hook)
  eval "$conda_hook" >&2   # no conda init: nothing is written to the user's shell files
  conda activate "$FUSION_ENV" >&2
fi
set -u
if [[ "${1:-}" == probe ]]; then
  shift
  exec python "$(dirname "$0")/probe.py" "$@"
fi
exec python -m fusion_miniapp.run "$@"
