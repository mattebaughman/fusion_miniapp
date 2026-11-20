#!/bin/bash
# NOT RUN ON THIS MACHINE YET.
# Stellar-AI (Princeton/PPPL): one GPU node, 8 x NVIDIA B200, 96 cores, 980 GB. QOS follows the time limit.
# Submit (no defaults; sbatch reads SBATCH_PARTITION and, if you have several accounts, SBATCH_ACCOUNT):
#   export SBATCH_PARTITION=... FUSION_ENV=... FUSION_MINIAPP_DIR=...
#   sbatch launch/stellar_ai.sbatch --agents 100 --preset smoke [--use-academy]
# torch for this machine (untested here): torch==2.15.0 from https://download.pytorch.org/whl/cu132,
# installed from a visualisation node (compute nodes have no internet).
#SBATCH --job-name=fusion-miniapp
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One GPU: every simulation worker uses it (how to split GPUs with model servers is open; README).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module purge
module load anaconda3/2026.7

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
