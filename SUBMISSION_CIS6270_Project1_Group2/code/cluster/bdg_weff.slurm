#!/bin/bash
#SBATCH --job-name=cgm-bdg-weff
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/bdgw-%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/bdgw-%A_%a.err
#
# THE REDUCTION CONTROL. BDG's numerator is affine in F_i, so
#   (y - F_i) - eta*e*(F_i - F_bar) = (1 + eta*e) * (y_eff - F_i)
# and BDG is exactly plug at a batch-adaptive scalar weight w_eff = 1 + eta*e.
# Every floor-clearing BDG cell that WIDENS has w_eff < 0, so the widening is
# plug at a NEGATIVE guidance weight -- repulsion from an extrapolated target,
# which is the only way to exceed the unguided spread.
# These cells run plug at those measured w_eff values. If they reproduce the
# matching BDG cells, the reduction is demonstrated rather than merely derived,
# and the honest claim becomes a structural one about the whole affine family.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
source <path redacted> || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CELLS=("mu plug -0.414 weff" "mu plug -0.288 weff" "alpha plug -0.465 weff" "gap plug -0.8 weff")
T=${SLURM_ARRAY_TASK_ID:-0}
read -r P A W V <<< "${CELLS[$T]}"
echo "task $T -> prop=$P arm=$A w=$W variant=$V host=$(hostname)"; date -Is
python -u proj1/scripts/guidance_sweep.py --stage bdg --fm weights/fm_ema.pt \
    --props "$P" --only-arm "$A" --only-w "$W" --only-variant "$V" \
    --n 512 --batch 512 --steps 100 --n-probe 1 \
    --seed 20260921 --device cpu --per-mol \
    --out-dir "$PROJ/results/bdg" --max-minutes 220
echo "finished $(date -Is)"
