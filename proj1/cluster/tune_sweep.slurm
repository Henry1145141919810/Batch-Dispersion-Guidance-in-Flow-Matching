#!/bin/bash
#SBATCH --job-name=cgm-tune
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/tune-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/tune-%j.err
#
# STAGE 1 of 2: the JOINT strength x start-time screen.
#
# WHY IT EXISTS. Every earlier screen swept strength at one window and window
# at one strength, so it could not see an interaction between them -- and this
# project's best-supported claim is that WHEN you guide matters more than HOW.
# This stage sweeps both together, at one fixed target (q90), on our own FM
# base, for all seven arms. freeze_tune.py then picks two operating points per
# (property, arm): the best in-band that clears the chemistry floor, and the
# best in-band with no chemistry constraint at all.
#
# THE DELTA IS OVERRIDDEN, and this changes sampling, not only scoring.
# --delta-json takes the in-band tolerance from results/local_fb_mae.json
# (2 x f_B's MAE on val molecules whose TRUE property sits in train_a's
# q85-q95, i.e. the target's own neighbourhood). BTVG's tau is delta/1.96, so
# btvg and btvg_var cells sampled under a different delta are a DIFFERENT
# experiment. freeze_tune.py refuses a tree that mixes two deltas.
#
# GRID: 9 strengths x 3 windows x 3 properties x 7 arms = 489 cells at
# n = 1000, ~26 GPU-h.
#
# THE ARRAY IS SPLIT BY COST, NOT EVENLY. Guidance runs only for
# t >= t_min_guide, so a cell's cost scales with the guided fraction: the
# t_min = 0.05 slice guides 95 of 100 steps and costs ~3.7x the t_min = 0.75
# slice. A whole (property, window) task at 0.05 needs ~4.6 h, over the 4-hour
# wall. So each slice is halved by ARM GROUP, into two cost-balanced halves
# (18.5 and 17.0 cost units):
#     group A   lgd_mc, tfg, plug   + unguided (one cheap cell per property)
#     group B   btvg, btvg_var, tmpd
# The longest task is then ~2.4 h.
#
# SUBMIT: the whole chain in one paste is in docs/protocol/TUNE_SWEEP_PLAN.md.
#   sbatch --array=0-17 --export=ALL proj1/cluster/tune_sweep.slurm
#       task = 9 * arm_group + 3 * window_index + property_index
#   sbatch --dependency=afterany:<ARRAY_ID> --export=ALL proj1/cluster/tune_sweep.slurm
#       no --array: the insurance sweeper. It walks EVERY slice and finishes
#       whatever the array left. It CANNOT do the whole sweep alone inside one
#       4-hour window -- it is a finisher, not a substitute -- which is why the
#       paste chains two of them. The sweep is resumable: a finished cell is a
#       file, and a file on disk is never recomputed.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

PROPS=(mu alpha gap)
WINDOWS=(0.05 0.5 0.75)
# cost-balanced halves; unguided rides with group A and is planned first
GROUP_A="${GROUP_A:-unguided,lgd_mc,tfg,plug}"
GROUP_B="${GROUP_B:-btvg,btvg_var,tmpd}"
N="${TUNE_N:-1000}"
ARMS="${ARMS:-unguided,plug,tmpd,lgd_mc,tfg,btvg,btvg_var}"
FM=proj1/checkpoints/fm_last.pt
FM_MD5=a190ac8394902027d4a951f8d30e8c5c      # every cell of this project
DELTA_JSON="$PROJ/results/local_fb_mae.json"
OUT="$PROJ/results/tune/n${N}"
SEED="${TUNE_SEED:-20260925}"
BUDGET_MIN="${BUDGET_MIN:-215}"
T_START=$(date +%s)

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "out       : $OUT"
echo "n, seed   : $N, $SEED"
echo "arms      : $ARMS"

# --- the generator must be the one every other cell used ---------------------
if [ ! -f "$FM" ]; then
  echo "FATAL: $FM not found" >&2; exit 1
fi
GOT=$(md5sum "$FM" | cut -d' ' -f1)
if [ "$GOT" != "$FM_MD5" ]; then
  echo "FATAL: $FM md5 $GOT, expected $FM_MD5." >&2
  echo "Every cell in this project used a190ac83. A different generator here" >&2
  echo "would silently produce a table that cannot be compared with any other." >&2
  exit 1
fi

# --- the delta file must exist and match this f_B ----------------------------
if [ ! -f "$DELTA_JSON" ]; then
  echo "FATAL: $DELTA_JSON not found." >&2
  echo "Generate it once (CPU, ~1 min on a GPU node, or on the login node):" >&2
  echo "  python proj1/scripts/local_fb_mae.py" >&2
  exit 1
fi
mkdir -p "$OUT"

run_slice () {          # $1 = property, $2 = t_min_guide, $3 = arm list
  local prop="$1" win="$2" armset="$3"
  local elapsed=$(( ($(date +%s) - T_START) / 60 ))
  local left=$(( BUDGET_MIN - elapsed ))
  if [ "$left" -le 5 ]; then
    echo "[budget] $elapsed min used, stopping before $prop/tmin=$win/$armset"
    return 0
  fi
  echo "=== $prop  t_min=$win  arms=$armset  (${left} min left) ==="
  srun python -u proj1/scripts/guidance_sweep.py \
    --stage tune \
    --props "$prop" \
    --arms "$armset" \
    --windows "$win" \
    --n "$N" \
    --seed "$SEED" \
    --fm "$FM" \
    --delta-json "$DELTA_JSON" \
    --out-dir "$OUT" \
    --max-minutes "$left"
  return $?
}

# PREFLIGHT: one throwaway cell per arm at n = 8, writing nothing. A mis-wired
# arm dies here in ~2 minutes instead of after 24 GPU-h of `.failed` cells that
# the freeze then refuses on.
echo "=== preflight ==="
srun python -u proj1/scripts/guidance_sweep.py \
  --stage tune --props "${PROPS[0]}" --arms "$ARMS" --windows 0.5 \
  --fm "$FM" --delta-json "$DELTA_JSON" --out-dir "$OUT" --preflight || {
    echo "FATAL: preflight failed - an arm is mis-wired. Nothing was written." >&2
    exit 1
  }

RC=0
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  T=$SLURM_ARRAY_TASK_ID
  G=$(( T / 9 ))
  R=$(( T % 9 ))
  PROP=${PROPS[$(( R % 3 ))]}
  WIN=${WINDOWS[$(( R / 3 ))]}
  if [ "$G" -eq 0 ]; then ARMSET="$GROUP_A"; else ARMSET="$GROUP_B"; fi
  run_slice "$PROP" "$WIN" "$ARMSET" || RC=$?
else
  # insurance sweeper: every slice, expensive windows first
  for WIN in "${WINDOWS[@]}"; do
    for PROP in "${PROPS[@]}"; do
      for ARMSET in "$GROUP_A" "$GROUP_B"; do
        run_slice "$PROP" "$WIN" "$ARMSET" || RC=$?
      done
    done
  done
fi

echo "=== cells on disk: $(find "$OUT" -maxdepth 1 -name '*__tune.json' | wc -l) / 489 ==="
echo "elapsed: $(( ($(date +%s) - T_START) / 60 )) min, rc=$RC"
exit $RC
