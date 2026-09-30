#!/bin/bash
#SBATCH --job-name=cgm-sweep-v2
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/sweepv2-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/sweepv2-%j.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# Stage 2 screening: the three new arms (SPBC, BTVG, SHG) plus the three v1
# arms that were implemented and gated but never queued (smg_mean, rch = Haimo
# v1, band), against re-run references in the same cells.
#
# SMALL SCALE ON PURPOSE. This is the screen that decides what proceeds, so it
# runs at n=512 -- the same sample size as the v1 sweep, which is what makes the
# two directly comparable. The full-scale numbers come from
# final_benchmark.slurm and only for the survivors.
#
# Resumable by exactly the same contract as the v1 sweep: one json per cell in
# results/sweep/, cells with a json are skipped, so this is submitted as an
# afterany chain and each link continues where the last stopped. Stage-v2 cell
# names carry a variant suffix, so they can never collide with the v1 cells
# already on disk.
#
# WINDOW. Screening runs at t_min_guide = 0.5, not the v1 default of 0.05:
# the window effect is U-shaped and large (alpha MAE 10.02 -> 5.32 -> 6.59 as
# t_min_guide goes 0.05 -> 0.5 -> 0.75) and 0.5 is its measured optimum.
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
mkdir -p "$PROJ/logs" "$PROJ/results/sweep"
cd "$PROJ"
source "$PROJ/.venv/bin/activate"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "props     : ${SWEEP_PROPS:-mu,alpha,gap}"
echo "stage     : v2"
echo "started   : $(date -Is)"
echo

# Fail fast. One throwaway cell per arm, ~1 minute, writes nothing. A wiring
# error or a missing checkpoint otherwise burns the whole four-hour window
# writing .failed files nobody reads until the job is over -- which is exactly
# what a parameter-shadowing bug in the rch branch would have done.
srun python -u proj1/scripts/guidance_sweep.py \
    --stage "${STAGE:-v2}" --preflight \
    --fm proj1/checkpoints/fm_last.pt \
    --props "${SWEEP_PROPS:-mu,alpha,gap}"
echo "preflight passed"
echo

srun python -u proj1/scripts/guidance_sweep.py \
    --stage "${STAGE:-v2}" \
    --fm proj1/checkpoints/fm_last.pt \
    --props "${SWEEP_PROPS:-mu,alpha,gap}" \
    --n 512 \
    --batch 128 \
    --steps 100 \
    --n-probe 1 \
    --max-minutes 225

echo "finished  : $(date -Is)"
