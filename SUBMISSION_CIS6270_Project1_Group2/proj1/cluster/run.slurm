#!/bin/bash
#SBATCH --job-name=cgm-run
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=$PROJECT_ROOT/logs/run-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/run-%j.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# Generic one-off job: tests, M-0, M-1, sampling sweeps. Pass the command in.
#
#   sbatch --export=ALL,CMD="python proj1/tests/test_guidance.py" proj1/cluster/run.slurm
#   sbatch --export=ALL,CMD="python proj1/scripts/m1_signed_bias.py --fm proj1/checkpoints/fm.pt --guide proj1/checkpoints/f_A_mu.pt" proj1/cluster/run.slurm
#
# Same fixed resource line as the trainers (the submit filter insists). One hour
# is plenty for anything here; raise --time on the sbatch line if not, up to the
# 4 h QOS cap.
set -euo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
cd "$PROJ"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

: "${CMD:?set CMD, e.g. sbatch --export=ALL,CMD=\"python ...\" proj1/cluster/run.slurm}"
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:?not inside a SLURM job}"
echo "started : $(date -Is)"
echo "cmd     : $CMD"
echo
srun bash -c "$CMD"
echo
echo "finished: $(date -Is)"
