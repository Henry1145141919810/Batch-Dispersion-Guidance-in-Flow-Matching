#!/bin/bash
#SBATCH --job-name=cgm-bdg
#SBATCH --partition=b200-mig45
#SBATCH --qos=mig-max
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdg-%j.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdg-%j.err
#
# BDG: two-sided setpoint control of the ACROSS-BATCH variance of the predicted
# property, plus the ablation ladder and its MATCHED controls.
#
# WHY THE CONTROLS ARE IN THIS JOB AND NOT TAKEN FROM results/sweep.
# Every cell in results/sweep* was generated with fm_last.pt. That checkpoint is
# NOT present in this tree; the only generator here is weights/fm_ema.pt. A BDG
# number produced on fm_ema compared against a plug number produced on fm_last
# would be a cross-checkpoint comparison wearing the costume of an ablation.
# So unguided and plug are re-run beside BDG on the same checkpoint, the same
# seed and the same window, and ONLY within-job comparisons are reported.
#
# THE LADDER (w fixed at 1.0 so only the dispersion knobs move):
#   e0t1        eta = 0            -> must reproduce plug BIT-IDENTICALLY
#   e{1,4}t0.5  contract below the natural spread
#   e{1,4}t1.0  hold  (e = 0 at the setpoint: the fixed point)
#   e{1,4}t1.5  WIDEN -- the branch no arm in this repo can currently reach
#   e4t0.5o / e4t1.5o   the one-sided (BTVG contract-only) restriction
#
# tau is expressed as a multiple of s = f_A.y_std, which sits within ~7% of the
# unguided population sd, so tau_mult reads as "fraction of the natural spread".
#
# Resumable by the same contract as every other sweep: one json per cell, cells
# with a json are skipped, so this can be resubmitted to continue.
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf

PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
VENV=/vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv
mkdir -p "$PROJ/logs" "$PROJ/results/sweep"
cd "$PROJ"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "props     : ${SWEEP_PROPS:-mu,alpha,gap}"
echo "generator : weights/fm_ema.pt  (fm_last.pt is absent from this tree)"
echo "started   : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no gpu visible"
echo

# Gate on the exact-math tests before spending the window. The bit-identity
# gate is the base-model control for the whole ablation table; if it regressed,
# the table is meaningless and the run should not happen.
srun "$VENV/bin/python" -u proj1/tests/test_bdg.py
echo "bdg gates passed"
echo

# Fail fast on wiring, one throwaway cell per arm, writes nothing.
srun "$VENV/bin/python" -u proj1/scripts/guidance_sweep.py \
    --stage bdg --preflight \
    --fm weights/fm_ema.pt \
    --props "${SWEEP_PROPS:-mu,alpha,gap}"
echo "preflight passed"
echo

srun "$VENV/bin/python" -u proj1/scripts/guidance_sweep.py \
    --stage bdg \
    --fm weights/fm_ema.pt \
    --props "${SWEEP_PROPS:-mu,alpha,gap}" \
    --n 512 \
    --batch 512 \
    --steps 100 \
    --n-probe 1 \
    --max-minutes 205

echo "finished  : $(date -Is)"
