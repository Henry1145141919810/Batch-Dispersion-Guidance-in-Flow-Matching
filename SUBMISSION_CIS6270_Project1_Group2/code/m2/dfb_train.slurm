#!/bin/bash
#SBATCH --job-name=m2-dfb-base
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=$PROJECT_ROOT/proj1/m2/logs/dfb_%j.out
set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28
cd $PROJECT_ROOT
PY=<path redacted>
echo "host $(hostname)  start $(date -Is)"
# 489 epochs x 328 steps x batch 256 = 41.1M sample-views, the published Linear
# FM budget. Far beyond what 36h of CPU reaches; early stopping or the wall ends
# it first, and every improvement is checkpointed, so a kill still yields a model.
# Early stopping on held-out validation may end it sooner; the checkpoint is
# written on every improvement, so a wall-kill still leaves the best model.
srun $PY proj1/m2/simplex_fm.py \
  --dfb --crop 500 --hidden 128 --layers 10 \
  --batch 256 --lr 2e-3 \
  --epochs 489 --val-every-epochs 5 --patience 10 \
  --out proj1/m2/ckpt/fm_m2_dfb500.pt
echo "done $(date -Is)"
