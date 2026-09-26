#!/bin/bash
#SBATCH --job-name=chem-budget
#SBATCH --partition=b200-mig45
#SBATCH --qos=mig-max
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/chembudget-%A_%a.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/chembudget-%A_%a.err
#
# HYPOTHESIS. The chemistry guard is not a targeting method - it is a STRENGTH
# BUDGET. FR3a caps every arm at the strength where molecule stability falls
# below 0.9 x unguided; lgd_mc is pinned at w=4, the grid edge, still improving.
# The pilot (results/pilot_chem, n=2048 w=8) shows the guard raises mol_stab
# +4.8 pts on mu and +3.3 on gap, and on gap at w=8 it is the difference
# between FAILING the floor (0.3389) and passing it (0.3716) - while reaching
# in-band 0.177 against the registered best of 0.152.
#
# So: sweep BOTH arms past the registered grid edge and find where each one's
# chemistry floor binds. If the guard's floor binds later, it converts stability
# headroom into usable guidance strength, and that is the contribution.
#
# One array task per property so cells land incrementally and a bad arm can be
# cancelled without losing the others.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1

PROPS=(mu alpha gap)
P=${PROPS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "host $(hostname)  job ${SLURM_JOB_ID}  task ${SLURM_ARRAY_TASK_ID:-0}  prop $P"
nvidia-smi -L
date -Is

python -u proj1/scripts/guidance_sweep.py --stage targets \
    --arms lgd_mc,lgd_mc_chem --strengths 4,8,16,32 --targets dist \
    --props "$P" --n 2048 --seed 20261001 --batch 128 --steps 100 \
    --out-dir "$PROJ/results/chem_budget/seed20261001_n2048" --per-mol \
    --max-minutes 170
echo "finished $(date -Is)"
