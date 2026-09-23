#!/bin/bash
#SBATCH --job-name=cgm-transfer
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/transfer-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/transfer-%j.err
#
# THE TRANSFER: the main sweep's protocol, stage for stage, on TFG's EDMsecond
# with TFG's guide (tf_predict_<p>) and TFG's oracle (evaluate_<p>). Nothing in
# it is ours except the guidance field. Arms: unguided, plug, tmpd, lgd_mc,
# tfg, btvg, btvg_var. See docs/protocol/TRANSFER_EXPERIMENT_PLAN.md section 9.
# (tfg replaced dflow in the main compare set on 23 Sep; it was added here
# before any transfer cell ran.)
#
# Every task runs ONE ARM GROUP, because a whole (property, target) or
# (property, seed) does not fit the 225-min guard on EDM. Measured on the
# local RTX 5080 at n=512, seconds per cell: unguided 35, plug 106, tmpd 164,
# lgd_mc 204, btvg_var 332, btvg ~400 (638 under contention); tfg not yet
# measured on EDM (it costs ~plug + 4 guide passes per guided step).
#   group A = unguided, plug, tmpd, lgd_mc, tfg     group B = btvg, btvg_var
#
# STAGE=compare  (n=512, seed 20260922, 7 strengths x q50/q90, 258 cells)
#   sbatch --array=0-11 --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm
#       task = 2 x (3 x target_index + prop_index) + group; q50 first.
#       ~1 h (A) / ~1.5 h (B) per task at 5080 speed. Per-molecule sidecars on.
#
# STAGE=full     (dist target, n=5000, seeds 20261001-3, frozen strengths)
#   sbatch --array=0-23 --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm
#       0-17  PRIMARY (FR3a), task = 2 x (3 x seed_index + prop_index) + group,
#             so tasks 0-5 are a complete seed-1 table
#       18-23 SECONDARY (FR3 as registered), seed 1, only cells not in primary
#       ~1.4 h (A) / ~2 h (B) per task at 5080 speed.
#   The first task to start freezes the strengths ONCE from the finished
#   compare cells (transfer_sweep.py --stage freeze) into
#   results/transfer/full/n5000/frozen_q90.json; every task reuses that file.
#   An incomplete compare stage stops the job before any sampling.
#
# INSURANCE, either stage: the same command WITHOUT --array walks every task
# in order inside one 225-min budget and finishes whatever the array left.
#   sbatch --dependency=afterany:<ARRAY_ID> --export=ALL,STAGE=<stage> proj1/cluster/transfer_run.slurm
# Chain a second one afterany on the first.
#
# Every task first checks the EDMsecond checkpoint against the md5 pinned in
# fetch_tfg_assets.py, so a cluster cell can never come from a different file
# than a local one; then runs a preflight (one tiny cell per arm, writes
# nothing); then the real cells. Cells are resumable and written atomically.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

STAGE="${STAGE:?set STAGE=compare or STAGE=full via --export=ALL,STAGE=...}"
PROPS=(mu alpha gap)
TARGETS=(q50 q90)
ARM_GROUPS=(unguided,plug,tmpd,lgd_mc,tfg btvg,btvg_var)
SEEDS=(20261001 20261002 20261003)
FULL_N="${FULL_N:-5000}"
T_START=$(date +%s)
BUDGET_MIN=225
TS=proj1/scripts/transfer_sweep.py
COMPARE_DIR="$PROJ/results/transfer/compare"
RUNDIR="$PROJ/results/transfer/full/n${FULL_N}"
FROZEN="$RUNDIR/frozen_q90.json"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "stage     : $STAGE"
echo "started   : $(date -Is)"
echo

python -u proj1/scripts/fetch_tfg_assets.py --verify || {
    echo "STOP: the EDMsecond checkpoint is missing or does not match its pinned md5"
    exit 1
}

case "$STAGE" in
    compare) N_TASKS=12 ;;
    full)    N_TASKS=24 ;;
    *) echo "STAGE must be compare or full, got '$STAGE'"; exit 1 ;;
esac

freeze() {
    if [ -s "$FROZEN" ]; then
        echo "strengths already frozen: $FROZEN"
        return 0
    fi
    mkdir -p "$RUNDIR" || return 1
    local tag="${SLURM_JOB_ID:-local}"
    python -u "$TS" --stage freeze --out-dir "$COMPARE_DIR" \
        --json-out "$FROZEN.pending.$tag"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "STOP: the transfer compare stage is incomplete or inconsistent"
        echo "(freeze rc=$rc). Nothing was run. Resubmit once it has finished."
        rm -f "$FROZEN.pending.$tag"
        return 3
    fi
    mv -f "$FROZEN.pending.$tag" "$FROZEN" || return 1
    echo "strengths frozen: $FROZEN"
    return 0
}

# task -> the transfer_sweep.py arguments that select its cells
task_args() {
    local t=$1
    local arms=${ARM_GROUPS[$((t % 2))]}
    local u=$((t / 2))
    if [ "$STAGE" = "compare" ]; then
        echo "--stage compare --props ${PROPS[$((u % 3))]} --targets ${TARGETS[$((u / 3))]} --arms $arms --out-dir $COMPARE_DIR --per-mol"
    elif [ "$u" -lt 9 ]; then
        local seed=${SEEDS[$((u / 3))]}
        echo "--stage full --frozen $FROZEN --props ${PROPS[$((u % 3))]} --arms $arms --seed $seed --n $FULL_N --sets primary --out-dir $RUNDIR/seed$seed --per-mol"
    else
        local seed=${SEEDS[0]}
        echo "--stage full --frozen $FROZEN --props ${PROPS[$((u - 9))]} --arms $arms --seed $seed --n $FULL_N --sets mae --exclude-sets primary --out-dir $RUNDIR/seed$seed --per-mol"
    fi
}

run_task() {
    local task=$1 maxmin=$2
    if [ "$STAGE" = "full" ]; then
        freeze || return $?
    fi
    local a
    a=$(task_args "$task")
    echo "=== task $task : $a"
    local left
    # shellcheck disable=SC2086
    left=$(python "$TS" $a --dry-run | sed -n 's/.*to run \([0-9][0-9]*\).*/\1/p')
    if [ -z "$left" ]; then
        echo "could not read the plan for task $task"
        return 1
    fi
    if [ "$left" = "0" ]; then
        echo "task $task already complete"
        return 0
    fi
    echo "$left cells to run for task $task"
    # shellcheck disable=SC2086
    srun python -u "$TS" $a --preflight || return 1
    echo "preflight passed"
    # shellcheck disable=SC2086
    srun python -u "$TS" $a --max-minutes "$maxmin" || return 1
    return 0
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK=$SLURM_ARRAY_TASK_ID
    if [ "$TASK" -ge "$N_TASKS" ]; then
        echo "task $TASK out of range: submit with --array=0-$((N_TASKS - 1))"
        exit 1
    fi
    run_task "$TASK" "$BUDGET_MIN"
    RC=$?
else
    RC=0
    for TASK in $(seq 0 $((N_TASKS - 1))); do
        USED=$(( ($(date +%s) - T_START) / 60 ))
        LEFT=$(( BUDGET_MIN - USED ))
        if [ "$LEFT" -lt 20 ]; then
            echo "time budget spent ($USED min) -- the next sweeper continues"
            RC=1
            break
        fi
        run_task "$TASK" "$LEFT" || RC=1
    done
fi

echo
echo "finished  : $(date -Is)  rc=$RC"
exit $RC
