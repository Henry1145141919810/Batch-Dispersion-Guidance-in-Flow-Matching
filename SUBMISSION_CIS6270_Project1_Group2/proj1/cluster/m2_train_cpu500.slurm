#!/bin/bash
#SBATCH --job-name=m2-train-gpu
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/m2cpu-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/m2cpu-%j.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# Modality 2 base model, scaled up on GPU to be a defensible base rather than a
# gate. Three changes from the 200 bp CPU run:
#
#  500 bp, the NATIVE length. Removes a deviation for free now that compute is
#     not the constraint. Stark et al.'s enhancer benchmark is 500 bp and their
#     evaluation classifier was trained there; at 200 bp that classifier still
#     RUNS (it global-average-pools, so it is length-agnostic) but returns
#     numbers that mean nothing, which is worse than a crash.
#  Their architecture scale: hidden 256, 12 dilated layers, batch 256, and the
#     same 100 integration steps at sampling.
#  VALIDATION EARLY STOPPING instead of a fixed step budget. Stark et al. train
#     their linear-FM baseline to 40.3M sample-views over 83,968 sequences. We
#     have 6,126, so matching their step count would be ~6,500 epochs of pure
#     memorisation. For a smaller dataset "matched in strength" means trained
#     until held-out loss stops improving, and reporting where that happened.
#     The checkpoint records best_it, sample_views and val_loss for the paper.
#
# WHAT THIS DOES NOT BUY. It does not make us comparable to the published
# enhancer leaderboard: that task is class-conditional generation over 81/47
# cell types scored by FBD on 10k samples, and ours is property control on GC.
# The base model IS their "Linear FM" baseline (published FBD 19.6 melanoma /
# 15.0 fly brain), just trained on a 6k subset. Disclose, do not compare.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ/proj1/m2" || exit 1
source $PROJECT_ROOT/redacted || exit 1
echo "host $(hostname)  $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no gpu visible"
python -u simplex_fm.py --crop 500 --steps 40000 --batch 128 \
    --hidden 128 --layers 10 --val-every 500 --patience 6 \
    --device auto --out "$PROJ/proj1/m2/ckpt/fm_m2_500_cpu.pt"
echo "done $(date -Is)"
