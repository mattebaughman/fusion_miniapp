#!/usr/bin/env bash
# Shared body of the single-node Slurm jobs. Run from the submission directory so
# the user's relative --knobs paths still resolve. Slurm opens its own log at submission.
set -euo pipefail
machine=$1
scratch=$2
gpu_request=$3
shift 3
: "${FUSION_ENV:?set FUSION_ENV to the Python environment}" "${FUSION_MINIAPP_DIR:?set FUSION_MINIAPP_DIR to the package checkout}"
root=$(cd "$FUSION_MINIAPP_DIR" && pwd -P)
out="$scratch/fusion-miniapp/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
if [[ ${SLURM_RESTART_COUNT:-0} != 0 ]]; then
  out="$out-restart-$SLURM_RESTART_COUNT"
fi
mkdir -p "$(dirname "$out")"
mkdir "$out"   # an existing campaign directory may already hold parent logs
step=(srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" "$gpu_request")
# The probe emits one JSON document even for CPU-only workers. It exits nonzero
# when the required GPU or a finite forward is missing; keep its evidence on disk.
"${step[@]}" "$root/launch/run.sh" probe --require-gpu --out "$out/probe.json"
exec "${step[@]}" "$root/launch/run.sh" "$@" --knobs "$root/knobs/machines/$machine.yaml" --out "$out/run"
