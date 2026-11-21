#!/bin/bash
# Lux (OLCF), one MI355X on an exclusive node. The simulations require a GPU and fail without one.
# Export SBATCH_ACCOUNT (lowercase project short name), SBATCH_PARTITION, FUSION_ENV and absolute FUSION_MINIAPP_DIR.
#   sbatch --output="$MEMBERWORK/$SBATCH_ACCOUNT/%x-%j.out" "$FUSION_MINIAPP_DIR/launch/lux.sbatch" --preset smoke
# ROCm PyTorch uses the torch.cuda interface. Its wheel supplies ROCm runtime libraries.
# Planned torch build (untested here): torch 2.15.0, https://download.pytorch.org/whl/rocm7.14
#SBATCH --job-name=fusion-miniapp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --signal=TERM@120

set -euo pipefail
module load miniforge3/26.3.2-3
module load rocm/7.14.0
# Lux allocates a per-job NVMe slice when a GPU is requested.
export TMPDIR="${SLURM_TMPDIR:-/mnt/bb/${SLURM_JOB_ID:?}}"
cache="$TMPDIR/miopen"
export MIOPEN_USER_DB_PATH="$cache/db"
export MIOPEN_CUSTOM_CACHE_DIR="$cache/kernels"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
project=$(tr '[:upper:]' '[:lower:]' <<< "${SLURM_JOB_ACCOUNT:?}")
exec bash "${FUSION_MINIAPP_DIR:?}/launch/batch.sh" lux "${MEMBERWORK:?}/$project" --gpus-per-task=1 "$@"
