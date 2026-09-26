#!/bin/bash
#SBATCH --job-name=unguided-floor
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/ufloor-%A_%a.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/ufloor-%A_%a.err
#
# FIX A DENOMINATOR ERROR BEFORE IT REACHES THE PAPER.
# DIST_PROTOCOL.md:181 registers the chemistry floor as mol_stability >= 0.9 x UNGUIDED
# -- ONE number for all three properties. The chem-budget sweep (job 8681476) had no
# unguided cell and substituted 0.9 x each arm's OWN w=1 stability, which is stricter AND
# biased toward whichever arm degrades least, i.e. the guard. Under the registered floor
# alpha's headline +24% becomes +2%; under the adjudicator's 0.35496 it becomes -20%.
#
# These six cells (3 properties x 2 seeds) supply the missing denominator at the same n,
# batch, steps, window and target protocol as the sweep, so the floor is measured rather
# than assumed. Unguided is the cheapest arm (~2.9 s/mol), so this is ~25 min per cell.
set -uo pipefail
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PROPS=(mu alpha gap); SEEDS=(20261001 20261002)
T=${SLURM_ARRAY_TASK_ID:-0}
P=${PROPS[$(( T % 3 ))]}; S=${SEEDS[$(( T / 3 ))]}
echo "task $T -> prop $P  seed $S  arm unguided  on $(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage targets \
    --arms unguided --strengths 1 --only-w 1 --targets dist \
    --props "$P" --n 512 --seed "$S" --batch 64 --steps 100 --device cpu \
    --out-dir "$PROJ/results/chem_budget/unguided_floor" --per-mol --max-minutes 110
echo "finished $(date -Is)"
