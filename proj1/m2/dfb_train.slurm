#!/bin/bash
#SBATCH --job-name=m2-dfb-base
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/proj1/m2/logs/dfb_%j.out
set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28
cd /vast/projects/pranam/lab/boboli/cis6270-project1-group2
PY=/vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/python
echo "host $(hostname)  start $(date -Is)"
# 80,000 steps x batch 256 = 20.5M sample-views ~= 245 epochs over 83,722 seqs,
# ~33h at the measured 170 views/s, inside the 36h wall so cosine LR fully anneals.
# Early stopping on held-out validation may end it sooner; the checkpoint is
# written on every improvement, so a wall-kill still leaves the best model.
srun $PY proj1/m2/simplex_fm.py \
  --dfb --crop 500 --hidden 128 --layers 10 \
  --batch 256 --steps 80000 --lr 2e-3 \
  --val-every 2000 --patience 8 \
  --out proj1/m2/ckpt/fm_m2_dfb500.pt
echo "done $(date -Is)"
