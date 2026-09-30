#!/bin/bash
#SBATCH --job-name=s-sweep
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/ssweep-%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/ssweep-%A_%a.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# FINDINGS Gap 0, never run: the guidance tolerance s is hardcoded to f_A.y_std
# (guidance_sweep.py:918) and is 9.2x / 17.0x / 6.2x wider than the delta the
# samples are SCORED against. "We told guidance 'anywhere within 1.5 D is
# acceptable' and scored it on 'within 0.168 D'."
#
# WHY smg_var AND NOT plug. For the plug family s enters only as 1/s^2, a scalar
# on the step, so it is degenerate with w (kappa = w/s^2) and the w grid already
# IS the s grid. For the smg/tmpd family the denominator is s^2 + v_f, so s
# competes with a STATE-DEPENDENT quantity and changing it changes the SHAPE of
# the guidance, not just its scale.
#
# AND IT TESTS ILG FOR FREE. guidance.py:1572-3 is num = y - fval - c_term,
# den = s^2 + v_f + v2. At s=0 the smg_var path returns exactly e/V, which is the
# interval-likelihood (band-probability) objective to within a measured
# 0.87-1.00x amplitude, direction bit-identical. So the `zero` rung is ILG,
# with no new mode and no novelty claim.
#
# One cell per task: this model's tensors are too small for torch intra-op
# threading (measured 1 core of 32), so parallelism must be across cells.
# Separate out-dir per s-mode because cell_name does not encode s.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
source $PROJECT_ROOT/redacted || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

SMODES=(y_std y_std_4 2delta delta delta_4 zero)
PROPS=(mu gap); WS=(1 4)
T=${SLURM_ARRAY_TASK_ID:-0}
S=${SMODES[$(( T / 4 ))]}; P=${PROPS[$(( (T % 4) / 2 ))]}; W=${WS[$(( T % 2 ))]}
echo "task $T -> s_mode $S  prop $P  arm smg_var  w $W  on $(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage targets \
    --arms smg_var --strengths "$W" --only-w "$W" --targets dist \
    --props "$P" --s-mode "$S" --n 512 --seed 20261001 \
    --batch 64 --steps 100 --device cpu \
    --out-dir "$PROJ/results/s_sweep/$S" --per-mol --max-minutes 220
echo "finished $(date -Is)"
