#!/bin/bash
#SBATCH --job-name=cgm-bdg-hold
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdgh-%A_%a.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdgh-%A_%a.err
#
# THE TRUE HOLD CONTROL for BDG, one rung per property at its MEASURED fixed
# point. A setpoint servo should do nothing when the batch already sits at the
# setpoint, and this is the cell that tests it.
#
# The fixed point is tau_mult = sqrt(V_b)/y_std using the MID-TRAJECTORY V_b,
# read out of bdg_e in array 8706324. It is NOT 1.0: the pre-run estimate used
# the terminal population spread and was about 2x off on alpha and gap.
#   mu 1.21   alpha 0.73   gap 0.65
# It cannot be a shared tau_mult because the fixed point depends on each
# guide's own scale.
set -uo pipefail
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

CELLS=(
 "mu bdg 1 e1t1.21"    "mu bdg 1 e4t1.21"
 "alpha bdg 1 e1t0.73" "alpha bdg 1 e4t0.73"
 "gap bdg 1 e1t0.65"   "gap bdg 1 e4t0.65"
)
T=${SLURM_ARRAY_TASK_ID:-0}
read -r P A W V <<< "${CELLS[$T]}"
echo "task $T -> prop=$P arm=$A w=$W variant=$V  host=$(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage bdg \
    --fm weights/fm_ema.pt \
    --props "$P" --only-arm "$A" --only-w "$W" --only-variant "$V" \
    --n 512 --batch 512 --steps 100 --n-probe 1 \
    --seed 20260921 --device cpu --per-mol \
    --out-dir "$PROJ/results/bdg" \
    --max-minutes 220
echo "finished $(date -Is)"
