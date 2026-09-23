#!/bin/bash
#SBATCH --job-name=cgm-tfg
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/tfg-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/tfg-%j.err
#
# THE TFG ARM, added after the full run (Henry, 23 Sep: tfg replaces dflow in
# the compare set). The main protocol, stage for stage, for ONE arm -- so its
# cells pair with every cell already on disk: same generator (md5 pinned
# below), same compare seed, n, batch, steps, window and clip; the same FR3a
# freeze rule; the same full-run seeds and the same 5000 test molecules.
# Every tfg number is POST HOC (chosen after the full run was read) and is
# labelled so everywhere it appears.
#
# STAGE=compare   42 cells (3 props x q50/q90 x 7 strengths), n=512, seed
#                 20260921, into results/sweep/ beside the other 216.
#   sbatch --array=0-2 --export=ALL,STAGE=compare proj1/cluster/tfg_run.slurm
#       task = property index (mu, alpha, gap), 14 cells each.
#
# STAGE=full      dist target, n=5000, seeds 20261001-3, frozen strength.
#   sbatch --array=0-3 --export=ALL,STAGE=full proj1/cluster/tfg_run.slurm
#       0-2  PRIMARY (FR3a), one task per seed, all three properties
#       3    SECONDARY (FR3 as registered), seed 1, only cells whose strength
#            differs from the primary one
#   The first task to start freezes ONCE into results/full/n5000/
#   frozen_q90_tfg.json -- never the full run's frozen_q90.json, which is the
#   record of the pre-registered freeze and is only READ here (--must-match:
#   every arm the two files share must get the same strength and floor, or
#   nothing is written and the job stops). FR1 with tfg among the competitors
#   is POST HOC and report-only: a FAIL (rc 10) is printed, never a stop.
#   An incomplete compare stage stops the job before any sampling.
#
# INSURANCE, either stage: the same command WITHOUT --array walks every task
# in order inside one 225-min budget and finishes whatever the array left.
#   sbatch --dependency=afterany:<ARRAY_ID> --export=ALL,STAGE=<stage> proj1/cluster/tfg_run.slurm
#
# COST (estimated from the measured compare-stage counters, not yet measured
# on Betty): ~14 min per n=5000 cell on a 1g slice, so a full task (3 cells)
# ~45 min; a compare task (14 cells at n=512) ~25 min.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

STAGE="${STAGE:?set STAGE=compare or STAGE=full via --export=ALL,STAGE=...}"
PROPS=(mu alpha gap)
SEEDS=(20261001 20261002 20261003)
N="${FULL_N:-5000}"
FM=proj1/checkpoints/fm_last.pt
FM_MD5=a190ac8394902027d4a951f8d30e8c5c     # every compare and full-run cell
T_START=$(date +%s)
BUDGET_MIN=225
RUNDIR="$PROJ/results/full/n${N}"
BASE_FROZEN="$RUNDIR/frozen_q90.json"
FROZEN="$RUNDIR/frozen_q90_tfg.json"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "stage     : $STAGE"
echo "started   : $(date -Is)"
echo

# Pair or nothing: a cell from another generator would silently not pair.
GOT_MD5=$(md5sum "$FM" | cut -d' ' -f1)
if [ "$GOT_MD5" != "$FM_MD5" ]; then
    echo "STOP: $FM has md5 $GOT_MD5, not the compare/full-run generator $FM_MD5"
    exit 1
fi

case "$STAGE" in
    compare) N_TASKS=3 ;;
    full)    N_TASKS=4 ;;
    *) echo "STAGE must be compare or full, got '$STAGE'"; exit 1 ;;
esac

freeze() {
    if [ -s "$FROZEN" ]; then
        echo "strengths already frozen: $FROZEN"
        return 0
    fi
    if [ ! -s "$BASE_FROZEN" ]; then
        echo "STOP: $BASE_FROZEN (the full run's freeze) is missing"
        return 3
    fi
    local tag="${SLURM_JOB_ID:-local}"
    python -u proj1/scripts/check_fullrun_go.py --stage compare --target q50 \
        --json-out "$RUNDIR/fr1_q50_tfg.json.$tag"
    local rc50=$?
    python -u proj1/scripts/check_fullrun_go.py --stage compare --target q90 \
        --must-match "$BASE_FROZEN" --json-out "$FROZEN.pending.$tag"
    local rc90=$?
    # 0 = FR1 pass, 10 = FR1 fail: both are verdicts, and with tfg in the set
    # both are post hoc and report-only. Anything else = incomplete (2), a
    # mismatch with the full run's freeze (2) or a crash (3): stop.
    if { [ "$rc50" -ne 0 ] && [ "$rc50" -ne 10 ]; } || \
       { [ "$rc90" -ne 0 ] && [ "$rc90" -ne 10 ]; }; then
        echo "STOP: the compare stage is incomplete, inconsistent with"
        echo "$BASE_FROZEN, or unreadable (q50 rc=$rc50, q90 rc=$rc90)."
        echo "Nothing was run. Resubmit once the tfg compare cells are done."
        rm -f "$FROZEN.pending.$tag" "$RUNDIR/fr1_q50_tfg.json.$tag"
        return 3
    fi
    if [ "$rc50" -eq 10 ]; then
        echo "NOTE (post hoc, report only): with tfg among the competitors,"
        echo "FR1 on q50 FAILS. The registered FR1 (with dflow) passed."
    fi
    mv -f "$RUNDIR/fr1_q50_tfg.json.$tag" "$RUNDIR/fr1_q50_tfg.json" || return 1
    mv -f "$FROZEN.pending.$tag" "$FROZEN" || return 1
    echo "strengths frozen: $FROZEN"
    return 0
}

# task -> the guidance_sweep.py arguments that select its cells
task_args() {
    local t=$1
    if [ "$STAGE" = "compare" ]; then
        echo "--stage compare --arms tfg --props ${PROPS[$t]} --n 512"
    elif [ "$t" -lt 3 ]; then
        local seed=${SEEDS[$t]}
        echo "--stage full --frozen $FROZEN --arms tfg --props mu,alpha,gap --sets primary --n $N --seed $seed --out-dir $RUNDIR/seed$seed --per-mol"
    else
        local seed=${SEEDS[0]}
        echo "--stage full --frozen $FROZEN --arms tfg --props mu,alpha,gap --sets mae --exclude-sets primary --n $N --seed $seed --out-dir $RUNDIR/seed$seed --per-mol"
    fi
}

# Every command's status is checked explicitly: `set -e` is suspended inside
# a function called from `if`/`||`.
run_task() {
    local task=$1 maxmin=$2
    if [ "$STAGE" = "full" ]; then
        freeze || return $?
        mkdir -p "$RUNDIR/seed${SEEDS[0]}" "$RUNDIR/seed${SEEDS[1]}" \
                 "$RUNDIR/seed${SEEDS[2]}" || return 1
    fi
    local a
    a=$(task_args "$task")
    echo "=== task $task : $a"
    local left
    # shellcheck disable=SC2086
    left=$(python proj1/scripts/guidance_sweep.py $a --fm "$FM" --dry-run \
               | sed -n 's/.*to run \([0-9][0-9]*\).*/\1/p')
    if [ -z "$left" ]; then
        echo "could not read the plan for task $task"
        return 1
    fi
    if [ "$left" = "0" ]; then
        echo "task $task already complete"
        return 0
    fi
    echo "$left cells to run for task $task"

    # Fail fast: one throwaway tfg cell at n=8, writes nothing.
    # shellcheck disable=SC2086
    srun python -u proj1/scripts/guidance_sweep.py $a --fm "$FM" --preflight || return 1
    echo "preflight passed"

    # shellcheck disable=SC2086
    srun python -u proj1/scripts/guidance_sweep.py $a --fm "$FM" \
        --batch 128 --steps 100 --n-probe 1 --max-minutes "$maxmin" || return 1
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
