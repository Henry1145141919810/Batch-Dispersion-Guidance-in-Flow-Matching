#!/bin/bash
#SBATCH --job-name=cgm-final
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/final-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/final-%j.err
#
# FULL-SCALE FINAL DATA -- only for the arms that survived screening.
#
# The survivor list is NOT hard-coded here. It is recomputed from the screening
# cells by select_arms.py every time this job starts, so the final run can
# never silently disagree with the screen it is supposed to follow. If the
# screen has not run, this job stops instead of guessing.
#
# ERROR BARS COME FROM SEEDS, NOT FROM A FORMULA. Each seed writes into its own
# directory, so a seed can be added later without recomputing anything, and the
# spread across seeds is a measured quantity rather than an assumed one. Submit
# one job per seed:
#
#   for S in 20260921 20260922 20260923; do
#     sbatch --export=ALL,SEED=$S,SWEEP_PROPS=mu,alpha,gap proj1/cluster/final_benchmark.slurm
#   done
#
# Resumable on exactly the same contract as the screening sweep: one json per
# cell, cells with a json are skipped, so chain with --dependency=afterany.
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
SEED="${SEED:-20260921}"
OUTDIR="$PROJ/results/final/seed${SEED}"
mkdir -p "$PROJ/logs" "$OUTDIR"
cd "$PROJ"
source "$PROJ/.venv/bin/activate"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "seed      : $SEED"
echo "out       : $OUTDIR"
echo "started   : $(date -Is)"
echo

ARMS="$(python proj1/scripts/select_arms.py --stage v2 \
        --props "${SWEEP_PROPS:-mu,alpha,gap}" --emit-arms)"
if [ -z "$ARMS" ]; then
    echo "select_arms.py returned nothing -- the screening sweep has not produced"
    echo "cells yet. Run guidance_sweep_v2.slurm first. Refusing to guess."
    exit 1
fi
echo "survivors : $ARMS"
echo

srun python -u proj1/scripts/guidance_sweep.py \
    --stage v2 \
    --arms "$ARMS" \
    --fm proj1/checkpoints/fm_last.pt \
    --props "${SWEEP_PROPS:-mu,alpha,gap}" \
    --out-dir "$OUTDIR" \
    --seed "$SEED" \
    --n "${FINAL_N:-4096}" \
    --batch 128 \
    --steps 100 \
    --n-probe 1 \
    --max-minutes 225

echo "finished  : $(date -Is)"
