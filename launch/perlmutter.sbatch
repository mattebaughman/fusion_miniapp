#!/bin/bash
# Perlmutter (NERSC), one A100 in shared QOS. The simulations require a GPU and fail without one.
# Export SBATCH_ACCOUNT, FUSION_ENV and absolute FUSION_MINIAPP_DIR. Submit with:
#   sbatch --output="$PSCRATCH/%x-%j.out" "$FUSION_MINIAPP_DIR/launch/perlmutter.sbatch" --preset smoke
# Site resource example: https://docs.nersc.gov/systems/perlmutter/running-jobs/
# Planned torch build (untested here): torch 2.15.0, https://download.pytorch.org/whl/cu132
#SBATCH --job-name=fusion-miniapp
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --signal=TERM@120

set -euo pipefail
module load conda
exec bash "${FUSION_MINIAPP_DIR:?}/launch/batch.sh" perlmutter "${PSCRATCH:?}" --gpus-per-task=1 "$@"
