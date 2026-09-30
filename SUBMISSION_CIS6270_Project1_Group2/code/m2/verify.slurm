#!/bin/bash
#SBATCH --job-name=m2-verify
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --array=0-1
#SBATCH --output=$PROJECT_ROOT/proj1/m2/logs/verify_%A_%a.out
set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28
PY=<path redacted>
S=$PROJECT_ROOT/proj1/m2
echo "host $(hostname) task $SLURM_ARRAY_TASK_ID start $(date -Is)"
case $SLURM_ARRAY_TASK_ID in
  0) srun $PY $S/gate_arms.py ;;
  1) srun $PY $S/nfe_check.py ;;
esac
echo "done $(date -Is)"
