#!/bin/bash
#SBATCH --job-name=chembudget-cpu
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=$PROJECT_ROOT/logs/chembudgetcpu-%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/chembudgetcpu-%A_%a.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# WHY CPU. Every GPU on the cluster is allocated (32/32 mig45, 16/16 mig90,
# 216/216 dgx-b200); the shortest running job has ~14 h left and SLURM estimated
# our start as 28 Sep. Our own account is at its GPU cap - <user>'s jobs are
# pending on MaxGRESPerAccount - so queuing more GPU work competes with a
# teammate for the same allowance. genoa-std-mem has 20 idle nodes.
#
# THE QUESTION. Does the chemistry guard buy STRENGTH BUDGET? FR3a caps each arm
# at the strength where mol_stability drops below 0.9 x unguided = 0.3618.
# lgd_mc is frozen at w=4, the grid edge, still improving. The n=2048 pilot at
# w=8 showed, on gap: lgd_mc mol_stab 0.3389 (FAILS the floor) vs lgd_mc_chem
# 0.3716 (passes), with in-band 0.177 against the registered best of 0.152.
# So sweep both arms PAST the grid edge and find where each one's floor binds.
#
# Task order puts gap first: it is the property where the pilot effect was
# clearest, so a partial run still answers the question.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
source $PROJECT_ROOT/redacted || exit 1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

PROPS=(gap mu alpha)
P=${PROPS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "host $(hostname)  task ${SLURM_ARRAY_TASK_ID:-0}  prop $P  threads $OMP_NUM_THREADS"
date -Is
python -c "import torch;print('torch',torch.__version__,'threads',torch.get_num_threads())"

# n=512 keeps each cell tractable on CPU; se on mol_stability ~0.022, and the
# pilot effect on gap was +0.033, so the decisive comparison is resolvable.
python -u proj1/scripts/guidance_sweep.py --stage targets \
    --arms lgd_mc,lgd_mc_chem --strengths 4,16,32 --targets dist \
    --props "$P" --n 512 --seed 20261001 --batch 64 --steps 100 \
    --device cpu \
    --out-dir "$PROJ/results/chem_budget/cpu_seed20261001_n512" --per-mol \
    --max-minutes 560
echo "finished $(date -Is)"
