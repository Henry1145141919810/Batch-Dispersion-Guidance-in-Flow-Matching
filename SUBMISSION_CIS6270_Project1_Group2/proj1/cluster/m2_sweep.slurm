#!/bin/bash
#SBATCH --job-name=m2-sweep
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=$PROJECT_ROOT/logs/m2sw-%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/m2sw-%A_%a.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# Modality 2 transfer run. Specs mirrored from Modality 1: n=512, batch=512 (one
# batch -- for BDG the batch IS the estimator), 100 Euler steps, t_min=0.5,
# clip=1.0, w=1.0 for the ladder, eta in {1,4}, tau_mult in {0.5,1,1.5},
# seeds 20260921/20260922. ONE CELL PER TASK: parallelism across cells, not
# threads.
#
# 13 cells x 2 seeds:
#   unguided | plug w=0.25,1,4 | bdg e0t1 (bit-identity control)
#   bdg e{1,4}t{0.5,1,1.5} | bdg e4t0.5o, e4t1.5o (one-sided ablation)
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ/proj1/m2" || exit 1
source $PROJECT_ROOT/redacted || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CELLS=(
 "unguided - 1" "plug - 1" "plug - 0.25" "plug - 4" "bdg e0t1 1"
 "bdg e1t0.5 1" "bdg e4t0.5 1" "bdg e1t1 1" "bdg e4t1 1"
 "bdg e1t1.5 1" "bdg e4t1.5 1" "bdg e4t0.5o 1" "bdg e4t1.5o 1"
)
SEEDS=(20260921 20260922)
T=${SLURM_ARRAY_TASK_ID:-0}
S=${SEEDS[$(( T / 13 ))]}
read -r ARM VAR W <<< "${CELLS[$(( T % 13 ))]}"
echo "task $T -> arm=$ARM variant=$VAR w=$W seed=$S host=$(hostname)"; date -Is
python -u m2_sweep.py --ckpt "$PROJ/proj1/m2/ckpt/fm_m2.pt" \
    --arm "$ARM" --variant "$VAR" --w "$W" --seed "$S" \
    --n 512 --steps 100 --t-min 0.5 --clip 1.0 --delta-ratio 0.16 \
    --out-dir "$PROJ/results/m2"
echo "done $(date -Is)"
