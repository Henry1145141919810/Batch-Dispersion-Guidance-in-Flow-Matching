#!/bin/bash
#SBATCH --job-name=diff-bench
#SBATCH --partition=b200-mig45
#SBATCH --qos=mig-max
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/diffbench-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/diffbench-%j.err
set -euo pipefail
REPO=$PROJECT_ROOT
cd "$REPO"
source <path redacted>
echo "host: $(hostname)"; nvidia-smi -L
# NOTE filenames deliberately avoid "_euler_" -- .gitignore:88 is
# results/bench/*_euler_*.json and would silently untrack these artifacts.
for CK in diff_ep1475 diff_last_ep1500; do
  for S in 0 1 2; do
    OUT=results/bench/${CK}_nfe100_s${S}.json
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    echo "=== $CK seed $S ==="
    python -u proj1/scripts/benchmark_base.py \
        --ckpt betty_pull/${CK}.pt --n 10000 --steps 100 --solver euler \
        --seed $S --out "$OUT"
  done
done
echo "ALL DONE"
