#!/bin/bash
#SBATCH --job-name=chembudget2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/cb2-%A_%a.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/cb2-%A_%a.err
#
# ONE CELL PER TASK. The first attempt gave each task 32 cores and got ~1 core of
# real use: this model's tensors (batch 64 x 29 atoms) are too small for torch's
# intra-op parallelism to engage, so threads buy nothing. Parallelising across
# CELLS instead turns a ~4-8 h serial walk into ~30 min of wall clock.
# Cells already written by job 8680190 are skipped automatically.
#
# 24 tasks = 3 properties x 2 arms x 4 strengths.
set -uo pipefail
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PROPS=(gap mu alpha); ARMS=(lgd_mc lgd_mc_chem); WS=(4 16 32 1)
T=${SLURM_ARRAY_TASK_ID:-0}
P=${PROPS[$(( T / 8 ))]}; A=${ARMS[$(( (T % 8) / 4 ))]}; W=${WS[$(( T % 4 ))]}
echo "task $T -> prop $P  arm $A  w $W  on $(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage targets \
    --arms "$A" --strengths "$W" --only-w "$W" --targets dist \
    --props "$P" --n 512 --seed 20261001 --batch 64 --steps 100 --device cpu \
    --out-dir "$PROJ/results/chem_budget/cpu_seed20261001_n512" --per-mol \
    --max-minutes 220
echo "finished $(date -Is)"
