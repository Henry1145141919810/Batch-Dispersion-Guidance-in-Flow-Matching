#!/bin/bash
# One-time remote setup. Run ON a Betty login node -- setup only, no compute.
#
#   bash setup_betty.sh
#
# Creates the working tree under the group allocation, builds a venv, and checks
# the data landed. Training itself always goes through sbatch.
set -euo pipefail

export PATH="/cm/local/apps/slurm/current/bin:$PATH"
PROJ=$PROJECT_ROOT

echo "== creating $PROJ"
mkdir -p "$PROJ"/logs "$PROJ"/proj1/checkpoints

echo "== python (Lmod)"
# Take whatever python module exists rather than guessing a version.
if command -v module >/dev/null 2>&1; then
    PYMOD=$(module -t avail python 2>&1 | grep -iE "^python/3\.(1[0-9]|[9])" | sort -V | tail -1 || true)
    if [ -n "${PYMOD:-}" ]; then
        echo "   module load $PYMOD"
        module load "$PYMOD"
    else
        echo "   no python module found; using system python3 ($(python3 --version))"
    fi
fi

if [ ! -d "$PROJ/.venv" ]; then
    python3 -m venv "$PROJ/.venv"
fi
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"
python -m pip install --quiet --upgrade pip

# Torch for the B200s (Blackwell, sm_100). cu128 wheels cover that generation;
# an older CUDA build would silently fall back to CPU.
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install --quiet numpy rdkit

echo "== versions"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

echo "== partitions visible to this account"
sinfo -o "%P %a %l %D %G" 2>/dev/null | head -12 || echo "   (sinfo unavailable)"

echo "== contents"
ls -la "$PROJ"
echo "== data present?"
if [ -f "$PROJ/data/qm9.pt" ]; then
    du -h "$PROJ/data/qm9.pt"
else
    echo "   qm9.pt MISSING -- copy it from the laptop before submitting jobs"
fi

# Derived, not hardcoded: this script lives beside the SLURM files, and an
# earlier hardcoded "$PROJ/cluster/..." pointed at a directory that does not
# exist (they are under $PROJ/proj1/cluster/).
HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo
echo "setup done. Submit work with:"
echo "  sbatch $HERE_DIR/train_fm.slurm"
echo
echo "Chain further 4-hour links onto a running job with:"
echo "  sbatch --dependency=afterany:<jobid> $HERE_DIR/train_fm.slurm"
