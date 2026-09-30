#!/bin/bash
# Pin to an IDLE GPU (2 or 3 were free). Adjust if that changed.
export CUDA_VISIBLE_DEVICES=${1:-2}
set -euo pipefail
python bench.py cuda 2>&1 | tee bench.log
echo "--- if throughput looks right, training starts now ---"
nohup python simplex_fm.py --npz dfb500.npz --crop 500 \
  --hidden 128 --layers 10 --batch 256 --lr 2e-3 \
  --epochs 1500 --val-every-epochs 5 --patience 300 --device cuda \
  --out ./fm_m2_dfb500.pt > train.log 2>&1 &
echo "pid $! -> train.log ; checkpoint fm_m2_dfb500.pt"
