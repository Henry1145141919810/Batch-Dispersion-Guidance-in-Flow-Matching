#!/bin/bash
#SBATCH --job-name=cgm-tuned-full
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/tunedfull-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/tunedfull-%j.err
#
# STAGE 2 of 2: the v2 full-run protocol at BOTH operating points the tune
# sweep chose.
#
# Protocol: docs/protocol/FULL_RUN_V2_PROTOCOL.md (rules V1-V8), with two
# deliberate departures, both recorded in TUNE_SWEEP_PLAN.md:
#   (a) the strength AND the guidance window come from freeze_tune.py, which
#       screened them jointly, instead of the strength alone at a fixed
#       window. Each arm therefore runs at its own best start time.
#   (b) TWO operating points per arm -- floor-clearing and unconstrained --
#       so the chemistry budget's cost is measured rather than asserted.
# Everything else is v2: same q90 targets, same three seeds, sizes from val,
# 100-step Euler, velocity clip 1.0, per-molecule sidecars on.
#
# n IS 4000, NOT 5000. Deliberate, to fit the 15-hour cluster budget: pooled
# over three seeds that is 12,000 molecules against v2's 15,000, so the se of
# an in-band proportion near 0.05 goes from 0.0018 to 0.0020. Set FULL_N=5000
# to restore the protocol number if there is time.
#
# CELLS ARE DE-DUPLICATED. When an arm's best in-band already clears the
# chemistry floor, its two picks coincide and the cell is sampled ONCE;
# frozen_tune.json records which file each set points at. Worst case (the two
# picks differ for all six guided arms) is 13 cells per (property, seed).
#
# SUBMIT
#   sbatch --dependency=afterok:<FREEZE_ID> --array=0-8 --export=ALL \
#          proj1/cluster/tuned_full_run.slurm
#       task = 3 * seed_index + property_index, so tasks 0-2 are a complete
#       seed-1 table.
#   sbatch --dependency=afterany:<ARRAY_ID> --export=ALL \
#          proj1/cluster/tuned_full_run.slurm
#       no --array: the insurance sweeper, walks every (seed, property).
#
# `afterok` on the freeze is load-bearing: if freeze_tune.py refused, this job
# must never start.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

PROPS=(mu alpha gap)
SEEDS=(20261001 20261002 20261003)
# Same cost-balanced halves as the sweep, and for the same reason: cost scales
# with the guided fraction, so if the freeze puts several picks at
# t_min = 0.05 a whole 7-arm group needs ~4.1 h, over the 4-hour wall. Split in
# two it is ~2.1 h worst case. unguided rides with group A and is planned
# first, because it is the chemistry floor's reference and every z-test's
# denominator -- the one cell that must never be the one the budget drops.
GROUP_A="${GROUP_A:-unguided,lgd_mc,tfg,plug}"
GROUP_B="${GROUP_B:-btvg,btvg_var,tmpd}"
N="${FULL_N:-4000}"
TUNE_N="${TUNE_N:-1000}"
ARMS="${ARMS:-unguided,plug,tmpd,lgd_mc,tfg,btvg,btvg_var}"
FM=proj1/checkpoints/fm_last.pt
FM_MD5=a190ac8394902027d4a951f8d30e8c5c
DELTA_JSON="$PROJ/results/local_fb_mae.json"
FROZEN="$PROJ/results/tune/n${TUNE_N}/frozen_tune.json"
RUNDIR="$PROJ/results/full/v2_tuned/n${N}"
BUDGET_MIN="${BUDGET_MIN:-215}"
T_START=$(date +%s)

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "n         : $N"
echo "frozen    : $FROZEN"
echo "out       : $RUNDIR"

if [ ! -f "$FROZEN" ]; then
  echo "FATAL: $FROZEN not found. The freeze job must run first." >&2
  exit 1
fi
if [ ! -f "$DELTA_JSON" ]; then
  echo "FATAL: $DELTA_JSON not found." >&2; exit 1
fi
if [ ! -f "$FM" ]; then echo "FATAL: $FM not found" >&2; exit 1; fi
GOT=$(md5sum "$FM" | cut -d' ' -f1)
if [ "$GOT" != "$FM_MD5" ]; then
  echo "FATAL: $FM md5 $GOT, expected $FM_MD5." >&2; exit 1
fi
mkdir -p "$RUNDIR"

run_cell_group () {      # $1 = property, $2 = seed, $3 = arm list
  local prop="$1" seed="$2" armset="$3"
  local elapsed=$(( ($(date +%s) - T_START) / 60 ))
  local left=$(( BUDGET_MIN - elapsed ))
  if [ "$left" -le 5 ]; then
    echo "[budget] $elapsed min used, stopping before $prop/seed $seed/$armset"
    return 0
  fi
  echo "=== $prop  seed $seed  arms=$armset  (${left} min left) ==="
  srun python -u proj1/scripts/guidance_sweep.py \
    --stage tuned_full \
    --props "$prop" \
    --arms "$armset" \
    --frozen "$FROZEN" \
    --n "$N" \
    --seed "$seed" \
    --fm "$FM" \
    --delta-json "$DELTA_JSON" \
    --per-mol --save-coords \
    --out-dir "$RUNDIR/seed${seed}" \
    --max-minutes "$left"
  return $?
}

RC=0
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  T=$SLURM_ARRAY_TASK_ID
  G=$(( T / 9 ))
  R=$(( T % 9 ))
  PROP=${PROPS[$(( R % 3 ))]}
  SEED=${SEEDS[$(( R / 3 ))]}
  if [ "$G" -eq 0 ]; then ARMSET="$GROUP_A"; else ARMSET="$GROUP_B"; fi
  run_cell_group "$PROP" "$SEED" "$ARMSET" || RC=$?
else
  for SEED in "${SEEDS[@]}"; do
    for PROP in "${PROPS[@]}"; do
      for ARMSET in "$GROUP_A" "$GROUP_B"; do
        run_cell_group "$PROP" "$SEED" "$ARMSET" || RC=$?
      done
    done
  done
fi

TOTAL=$(find "$RUNDIR" -name '*__tuned.json' | wc -l)
echo "=== cells on disk: $TOTAL ==="
echo "elapsed: $(( ($(date +%s) - T_START) / 60 )) min, rc=$RC"
exit $RC
