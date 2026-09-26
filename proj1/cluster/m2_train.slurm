#!/bin/bash
#SBATCH --job-name=m2-train
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/m2train-%j.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/m2train-%j.err
# Modality 2 base model: naive linear flow matching on the DNA simplex.
# The gate (proj1/m2/gate.py, job 8715948) cleared all three bars at 3000 steps;
# this is the real run at 20k with a cosine schedule so the base is not
# embarrassing beside the quoted simplex baselines.
set -uo pipefail
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ/proj1/m2" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "host $(hostname)  $(date -Is)"
python -u simplex_fm.py --steps 20000 --batch 64 --crop 200 \
    --out "$PROJ/proj1/m2/ckpt/fm_m2.pt"
echo "done $(date -Is)"
