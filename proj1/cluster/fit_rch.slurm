#!/bin/bash
#SBATCH --job-name=cgm-fit-rch
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/fitrch-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/fitrch-%j.err
#
# Fit the Residual Calibration Head (Haimo v1) for every property.
#
# WHY THIS EXISTS AS ITS OWN JOB. rch is the one arm that trains, and the
# stage-v2 sweep REFUSES to run it without a fitted head for that property
# rather than falling back to an unfitted one and reporting a number that looks
# like a result. Only rch_mu.pt exists, so alpha and gap must be fitted before
# the v2 sweep can measure Haimo v1 on all three -- this job is a hard
# dependency of guidance_sweep_v2.slurm.
#
# Each fit is a one-off ridge solve on cached (phi, target) pairs, so n-probe
# is raised to 8: the target must be accurate even though inference uses 1.
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
mkdir -p "$PROJ/logs"
cd "$PROJ"
source "$PROJ/.venv/bin/activate"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "started   : $(date -Is)"
echo

for PROP in mu alpha gap; do
    echo "=== $PROP ==="
    srun python -u proj1/scripts/fit_rch.py \
        --fm proj1/checkpoints/fm_last.pt \
        --guide "proj1/checkpoints/f_A_${PROP}.pt" \
        --prop "$PROP" \
        --n 2000 \
        --n-probe 8
    echo
done

echo "Read the held-out R2 of the 'residual' fit as the deflation test for SMG."
echo "finished  : $(date -Is)"
