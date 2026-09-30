#!/bin/bash
#SBATCH --job-name=m2-gate
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=$PROJECT_ROOT/logs/m2gate-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/m2gate-%j.err
#
# GO/NO-GO for Modality 2. Trains a NAIVE linear-simplex flow-matching model on
# Drosophila Kenyon-cell enhancers (200 bp) and tests only the three properties
# BDG actually needs from a base model: that samples decode confidently, that the
# GC distribution has a real spread to steer, and that the model learned anything
# at all (3-mer JS beats uniform-random sequences).
#
# The point is to spend one hour deciding instead of one day guessing. Dirichlet
# Flow Matching exists because linear simplex paths underperform -- but BDG is a
# guidance rule layered on top of whatever base exists, so "underperforms" is
# survivable and "collapses" is not. This tells us which.
#
# PASS -> build Modality 2 here. FAIL -> low-res images, where mean brightness is
# the exact analogue of GC content, at the cost of one hour.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
source <path redacted> || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host $(hostname)  $(date -Is)"
python -u proj1/m2/gate.py --crop 200 --steps 3000 --batch 64 --n-sample 512
echo "exit=$?  $(date -Is)"
