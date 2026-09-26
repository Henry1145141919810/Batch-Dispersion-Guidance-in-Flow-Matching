#!/bin/bash
#SBATCH --job-name=cgm-bdg-s2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdgs2-%A_%a.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdgs2-%A_%a.err
#
# SEED 2 (20260922) for the BDG claims that depend on seed noise at the margin.
#
# WHY ONLY THESE CELLS. The structural claims -- eta=0 == plug at cell level,
# monotone setpoint tracking, the V_b-vs-V_F contrast, and the in_band null --
# do not sit near a noise threshold. The WIDENING MAGNITUDES do: on seed 1 the
# bootstrap CI for alpha e4t1.5 (1.034x) and mu e4t1.21 (1.016x) both contain 1,
# and those CIs are optimistic because BDG couples all 512 samples through F_bar
# and V_b, so the draws are not independent.
#
# The eta=4 ladder plus its controls is enough to replicate both the
# monotonicity and the widening endpoints. Same checkpoint, window, device and
# batch as seed 1; ONLY the seed changes. Written to its own directory so a
# seed-1 cell can never be mistaken for a seed-2 one.
set -uo pipefail
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

CELLS=(
 "mu unguided 1 bdgref"    "mu plug 1 bdgref"    "mu bdg 1 e0t1"
 "alpha unguided 1 bdgref" "alpha plug 1 bdgref" "alpha bdg 1 e0t1"
 "gap unguided 1 bdgref"   "gap plug 1 bdgref"   "gap bdg 1 e0t1"
 "mu bdg 1 e4t0.5"    "mu bdg 1 e4t1"    "mu bdg 1 e4t1.5"    "mu bdg 1 e4t1.21"
 "alpha bdg 1 e4t0.5" "alpha bdg 1 e4t1" "alpha bdg 1 e4t1.5"
 "gap bdg 1 e4t0.5"   "gap bdg 1 e4t1"   "gap bdg 1 e4t1.5"
)
T=${SLURM_ARRAY_TASK_ID:-0}
read -r P A W V <<< "${CELLS[$T]}"
echo "task $T -> prop=$P arm=$A w=$W variant=$V seed=20260922 host=$(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage bdg --fm weights/fm_ema.pt \
    --props "$P" --only-arm "$A" --only-w "$W" --only-variant "$V" \
    --n 512 --batch 512 --steps 100 --n-probe 1 \
    --seed 20260922 --device cpu --per-mol \
    --out-dir "$PROJ/results/bdg_seed2" --max-minutes 220
echo "finished $(date -Is)"
