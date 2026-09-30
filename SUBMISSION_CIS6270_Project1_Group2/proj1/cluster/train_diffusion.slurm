#!/bin/bash
#SBATCH --job-name=cgm-diff
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/diff-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/diff-%j.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# VP diffusion generator -- the second required model family.
#
# Resource line is fixed. The submit filter requires --mem to be an exact
# multiple of the partition's memory-per-CPU (8 GB on the B200 partitions), and
# b200-mig45 has additionally rejected anything but 1 GPU / 6 CPU / 48G since
# 2026-08-03. 6 x 8 GB = 48 GB satisfies both. Do not "optimise" these values.
#
# Wall time: the partition allows 2 days but the wharton QOS caps every job at
# 04:00:00 (sacctmgr show qos, 18 Sep). Anything longer sits in PD forever with
# QOSMaxWallDurationPerJobLimit. Long runs are chained with --resume.
#
# Everything lives on /vast: /tmp is node-local and compute nodes cannot see
# what a login node wrote there.
set -euo pipefail

# SLURM's bin dir sometimes drops out of PATH mid-session on Betty.
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
mkdir -p "$PROJ/logs" "$PROJ/proj1/checkpoints"
cd "$PROJ"

# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "started   : $(date -Is)"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("gpu       :", torch.cuda.get_device_name(0) if ok else "NONE")
print("capability:", torch.cuda.get_device_capability(0) if ok else "-")
PY
echo

# --resume is always on: the first job finds no diff_last.pt and starts fresh;
# every later submission of this same script continues where the last stopped.
# --max-minutes leaves 15 min under the 4 h QOS cap so no partial epoch is lost.
srun python -u proj1/scripts/train_diffusion.py \
    --epochs 1500 \
    --batch 256 \
    --hidden 256 \
    --layers 8 \
    --split train_a \
    --tag diff \
    --resume \
    --max-minutes 225

echo "finished  : $(date -Is)"
