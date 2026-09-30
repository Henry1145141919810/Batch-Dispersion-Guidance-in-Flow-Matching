#!/bin/bash
#SBATCH --job-name=cgm-sweep
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/sweep-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/sweep-%j.err
#
# Stage 1 guidance screening on the FROZEN fm_v1 base model.
#
# Resumable by construction: guidance_sweep.py writes one json per cell into
# results/sweep/ and skips cells that already have one. So this script is
# submitted as a chain exactly like the trainers -- each link works through as
# many cells as fit in the QOS window and the next link continues. There is no
# shared mutable state between links, only the completed-cell files.
#
# Cells are ordered so a partial run is still a complete experiment: the full
# arm comparison lands first, then strength, then the guidance window, then the
# remaining targets.
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
mkdir -p "$PROJ/logs" "$PROJ/results/sweep"
cd "$PROJ"
source "$PROJ/.venv/bin/activate"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "props     : ${SWEEP_PROPS:-mu,alpha,gap}"
echo "started   : $(date -Is)"
echo

srun python -u proj1/scripts/guidance_sweep.py \
    --fm proj1/checkpoints/fm_last.pt \
    --props "${SWEEP_PROPS:-mu,alpha,gap}" \
    --n 512 \
    --batch 128 \
    --steps 100 \
    --n-probe 1 \
    --max-minutes 225

echo "finished  : $(date -Is)"
