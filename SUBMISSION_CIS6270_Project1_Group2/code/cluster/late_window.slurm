#!/bin/bash
#SBATCH --job-name=latewindow
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/lw-%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/lw-%A_%a.err
#
# RETIRE A CONFOUND. FINDINGS 4.4 states as STRUCTURAL that "the region where the
# closure becomes valid is the region where guidance has no effect" - at
# t_min=0.75 all seven arms land within 0.3% of each other and of unguided.
# But every t_min=0.75 cell in results/** is w=1: I enumerated them, 24 files,
# 24 at w=1, zero above. And the applied guidance share is
#     r_t = w * ((1-t)/t) * rho(t),  with rho(t) MEASURED FLAT (0.29 at t=0.50,
#     0.26 at 0.71, 0.06 at 0.90)
# so the 415x collapse of r_t is the (1-t)/t conversion carrying a CONSTANT w,
# not the field vanishing. Only 12.5% of the impulse lands at t >= 0.75, and
# matching the t_min=0.5 budget inside that window needs w ~ 8.
#
# So the late window has never been tested with authority. These 6 cells fill
# the empty region (t_min >= 0.75, w > 1) and retire the confound either way:
# either the late window converts strength into band yield, or it does not and
# FINDINGS 4.4 becomes a properly-evidenced claim instead of a one-strength one.
#
# One cell per task: this model's tensors are too small for torch intra-op
# threading (measured: 1 core of 32 in use), so parallelism must be across cells.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
source <path redacted> || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PROPS=(mu gap); WS=(1 16 64)
T=${SLURM_ARRAY_TASK_ID:-0}
P=${PROPS[$(( T / 3 ))]}; W=${WS[$(( T % 3 ))]}
echo "task $T -> prop $P  arm plug  w $W  t_min 0.85  on $(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage targets \
    --arms plug --strengths "$W" --only-w "$W" --targets dist --win 0.85 \
    --props "$P" --n 512 --seed 20261001 --batch 64 --steps 100 --device cpu \
    --out-dir "$PROJ/results/late_window/n512" --per-mol --max-minutes 220
echo "finished $(date -Is)"
