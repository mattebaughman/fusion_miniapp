#!/bin/bash
# NOT RUN ON THIS MACHINE YET.
# Perlmutter (NERSC): one GPU node, 4 x NVIDIA A100, 64 cores (128 hardware threads), 256 GB.
# Submit (no defaults; sbatch reads SBATCH_ACCOUNT and SBATCH_QOS itself):
#   export SBATCH_ACCOUNT=... SBATCH_QOS=... FUSION_ENV=... FUSION_MINIAPP_DIR=...
#   sbatch launch/perlmutter.sbatch --agents 100 --preset smoke [--use-academy]
# torch for this machine (untested here): torch==2.15.0 from https://download.pytorch.org/whl/cu132
#SBATCH --job-name=fusion-miniapp
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One GPU: every simulation worker uses it (how to split GPUs with model servers is open; README).
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load conda

# ---- identical in all three batch files from here ----
set -euo pipefail
: "${FUSION_ENV:?}" "${FUSION_MINIAPP_DIR:?set FUSION_MINIAPP_DIR to the fusion_miniapp folder}"
cd "$FUSION_MINIAPP_DIR"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"   # srun does not inherit --cpus-per-task
OUT="results/$SLURM_JOB_NAME-$SLURM_JOB_ID"
mkdir -p "$OUT"
# One step on one node with the job's GPU and cores; every simulation worker uses that GPU.
srun --nodes=1 --ntasks=1 launch/run.sh probe > "$OUT/probe.json"
grep -q '"cuda_available": true' "$OUT/probe.json" || { echo "torch sees no GPU in this step; see $OUT/probe.json" >&2; exit 1; }
srun --nodes=1 --ntasks=1 launch/run.sh --out "$OUT/run" "$@"
