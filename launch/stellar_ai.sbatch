#!/bin/bash
# Stellar-AI (Princeton/PPPL), one B200. The simulations require a GPU and fail without one.
# Export SBATCH_PARTITION, FUSION_ENV, absolute FUSION_MINIAPP_DIR and FUSION_SCRATCH
# (your existing GPFS scratch directory). Export SBATCH_ACCOUNT if needed. Submit with:
#   sbatch --output="$FUSION_SCRATCH/%x-%j.out" "$FUSION_MINIAPP_DIR/launch/stellar_ai.sbatch" --preset smoke
# Planned torch build (untested here): torch 2.15.0, https://download.pytorch.org/whl/cu132
#SBATCH --job-name=fusion-miniapp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --signal=TERM@120

set -euo pipefail
module purge
module load anaconda3/2026.7
exec bash "${FUSION_MINIAPP_DIR:?}/launch/batch.sh" stellar_ai "${FUSION_SCRATCH:?set to your GPFS scratch directory}" --gres=gpu:1 "$@"
