#!/bin/bash
#SBATCH --job-name=cgm-bdg-vf
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdgvf-%A_%a.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdgvf-%A_%a.err
#
# THE V_F LEG of the V_b-vs-V_F contrast, re-run after the first attempt was
# OOM-killed at batch=512.
#
# btvg_var needs an HVP (guide_hvp) that batch=512 could not fit in 4 cores x
# 5632 MB. Two changes: 8 cores (45 GB) and batch=128.
#
# DROPPING TO batch=128 IS SOUND HERE AND ONLY HERE. btvg_var servos the
# PER-SAMPLE posterior variance V_F = g^T Sigma g, which is computed per
# trajectory and does not reference the batch at all, so batch size cannot
# change its field. The BDG cells must stay at batch=512 because for BDG the
# batch IS the estimator. The asymmetry is deliberate and must be stated in
# the write-up rather than hidden.
set -uo pipefail
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

CELLS=(
 "mu btvg_var 1 tau1"    "mu btvg_var 4 tau1"
 "alpha btvg_var 1 tau1" "alpha btvg_var 4 tau1"
 "gap btvg_var 1 tau1"   "gap btvg_var 4 tau1"
)
T=${SLURM_ARRAY_TASK_ID:-0}
read -r P A W V <<< "${CELLS[$T]}"
echo "task $T -> prop=$P arm=$A w=$W variant=$V  host=$(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage bdg \
    --fm weights/fm_ema.pt \
    --props "$P" --only-arm "$A" --only-w "$W" --only-variant "$V" \
    --n 512 --batch 128 --steps 100 --n-probe 1 \
    --seed 20260921 --device cpu --per-mol \
    --out-dir "$PROJ/results/bdg" \
    --max-minutes 220
echo "finished $(date -Is)"
