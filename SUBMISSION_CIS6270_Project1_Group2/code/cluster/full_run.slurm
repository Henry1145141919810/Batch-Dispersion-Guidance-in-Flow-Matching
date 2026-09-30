#!/bin/bash
#SBATCH --job-name=cgm-full
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/full-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/full-%j.err
#
# THE FULL-SCALE RUN. The compare-set arms EXCEPT dflow -- unguided, plug,
# tmpd, lgd_mc, btvg, btvg_var -- once per property, at its FROZEN strength, on the
# `dist` target, with new seeds. This is the headline table (FR2-FR6 in
# docs/status/SCOPE_FM_GUIDANCE_STATUS.md, "PRE-REGISTRATION").
#
# TWO WAYS TO SUBMIT, SAME FILE:
#
#   sbatch --array=0-11 --export=ALL proj1/cluster/full_run.slurm
#       0-8:  the PRIMARY (headline) set, FR3a strengths, 3 seeds -- one task
#             per (property, seed), lowest indices so they start first;
#             tasks 0-2 are seed 1, so a complete one-seed table lands first
#             if GPUs are scarce.
#       9-11: the SECONDARY set, FR3 as registered (unconstrained best MAE),
#             seed 1 only, only the 10 cells whose strength differs.
#       Longest task ~115 min on the slowest slice (1g.45gb), far under the
#       225-min guard. All 12: ~20 GPU-h.
#
#   sbatch --dependency=afterany:<ARRAY_ID> --export=ALL proj1/cluster/full_run.slurm
#       NO --array: the insurance sweeper. Walks all 12 tasks in order and
#       finishes whatever the array left (a timeout, a node failure). Exits
#       in minutes if there is nothing left. One job, not 12, so it cannot
#       trip a per-user submit limit. Chain a SECOND sweeper afterany on the
#       first: the time guard never refuses an invocation's first cell, so a
#       sweeper that starts a 31-min btvg cell with 25 min left is killed,
#       and the second one finishes it.
#
# MEASURED COST, from the compare stage's own Betty logs (n=512, scaled to
# 5000), minutes per cell on 1g.45gb / 2g.45gb: unguided 4.9/4.1, plug
# 11.6/9.9, tmpd 18.2/15.8, lgd_mc 20.5/17.7, dflow 104.5/88.9, btvg
# 31.1/27.0, btvg_var 27.0/23.6.
#
# THE GATE, run at job start and never by hand-copied numbers:
#   1. FR1 on q50 (the pre-registered go/no-go). FAIL => stop, because FR1
#      says btvg does not proceed. Override only with FORCE_FULL=1, and only
#      after deciding to report it as a negative result anyway.
#   2. FR1 on q90 -- reported, never a stop (FR1 amended).
#   3. Strengths frozen from the q90 compare cells (FR3-corrected). PRIMARY =
#      FR3a: best MAE among strengths with mol_stability >= 0.9 x unguided,
#      because FR3's unconstrained choice (w=4) fails that floor for plug,
#      tmpd and btvg. SECONDARY = FR3 as registered. check_fullrun_go.py
#      refuses a partial compare grid, so this job cannot freeze a
#      best-of-half-a-sweep.
#   ONE frozen file for the whole run (results/full/n<N>/frozen_q90.json),
#   written by whichever task gets there first and reused by every other task
#   and every later link, so no two seeds or links can run different
#   strengths. check_fullrun_go.py exits 0 pass / 10 FR1 FAIL / 2 incomplete
#   / 3 crash; anything but 0 or 10 is an error here, so a Python traceback
#   can never be read as a pre-registered negative result.
#
# SCALE: n = FULL_N (default 5000), 3 seeds. FR4 said 10,000; halved on 22 Sep
# BEFORE any full-run cell existed, because 10,000 x 3 seeds is ~48 GPU-h
# (~16 h wall on three slices). Recorded as the FR4 amendment in the status
# doc. Seeds vary the initial noise only: all seeds and all arms see the SAME
# 5000 test molecules (sizes and targets), so every comparison is paired.
#
# Each cell also writes <cell>.permol.pt (per-molecule f_A, f_B, target,
# stability) so arms can be compared paired and re-scored without re-sampling.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

PROPS=(mu alpha gap)
SEEDS=(20261001 20261002 20261003)
N="${FULL_N:-5000}"
FM=proj1/checkpoints/fm_last.pt          # the compare stage's generator
T_START=$(date +%s)
BUDGET_MIN=225                            # under the 4 h QOS cap

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "n         : $N"
echo "started   : $(date -Is)"
echo

RUNDIR="$PROJ/results/full/n${N}"
FROZEN="$RUNDIR/frozen_q90.json"
mkdir -p "$RUNDIR"

# Freeze (or reuse) the strengths for the whole run. Returns 0 when the full
# run may proceed, nonzero otherwise, having said why.
freeze() {
    if [ -s "$FROZEN" ]; then
        echo "strengths already frozen: $FROZEN"
        return 0
    fi
    # per-job scratch names: nine tasks may reach this line at the same moment
    local tag="${SLURM_JOB_ID:-local}"
    python -u proj1/scripts/check_fullrun_go.py --stage compare --target q50 \
        --json-out "$RUNDIR/fr1_q50.json.$tag"
    local rc50=$?
    python -u proj1/scripts/check_fullrun_go.py --stage compare --target q90 \
        --json-out "$FROZEN.pending.$tag"
    local rc90=$?
    if { [ "$rc50" -ne 0 ] && [ "$rc50" -ne 10 ]; } || \
       { [ "$rc90" -ne 0 ] && [ "$rc90" -ne 10 ]; }; then
        echo "STOP: the compare stage is incomplete, inconsistent or unreadable"
        echo "(q50 rc=$rc50, q90 rc=$rc90; 2 = incomplete, 3 = crash). Nothing"
        echo "was run. Resubmit once it has finished."
        rm -f "$FROZEN.pending.$tag" "$RUNDIR/fr1_q50.json.$tag"
        return 3
    fi
    if [ "$rc50" -eq 10 ] && [ "${FORCE_FULL:-0}" != "1" ]; then
        echo "STOP: FR1 FAILS on q50 -- btvg does not proceed (pre-registered)."
        echo "It is reported as a negative result. To run the table anyway,"
        echo "resubmit with --export=ALL,FORCE_FULL=1."
        rm -f "$FROZEN.pending.$tag"
        mv -f "$RUNDIR/fr1_q50.json.$tag" "$RUNDIR/fr1_q50.json"   # the record
        return 4
    fi
    if [ "$rc90" -eq 10 ]; then
        echo "NOTE: FR1 fails on q90. Reported, not a stop (FR1 amended)."
    fi
    # atomic rename; concurrent tasks computed the same file from the same
    # finished grid (ties break on w, never on listing order)
    mv -f "$RUNDIR/fr1_q50.json.$tag" "$RUNDIR/fr1_q50.json" || return 1
    mv -f "$FROZEN.pending.$tag" "$FROZEN" || return 1
    echo "strengths frozen: $FROZEN"
    return 0
}


# dflow is NOT in tonight's run (decision 23 Sep: it will be replaced by another
# trajectory-optimisation competitor later). Its code (proj1/src/dflow.py),
# gates and 42 compare-stage cells are kept; it is only left out of this list.
# Measured Betty time per cell at n=5000 on B200 MIG 1g.45gb: unguided 4.9,
# plug 11.6, tmpd 18.2, lgd_mc 20.5, btvg 31.1, btvg_var 27.0 => ~113 min per
# (property, seed), well under the 225-min guard.
FULL_ARMS="unguided,plug,tmpd,lgd_mc,btvg,btvg_var"
N_TASKS=12            # 0-8 primary (the headline), 9-11 secondary

# Map a task index to: prop seed arms sets exclude-sets. Primary tasks come
# FIRST so Slurm starts the must-run work first.
#   0-8   primary set (FR3a), task = 3 x seed_index + prop_index, so tasks
#         0-2 are a complete seed-1 table
#   9-11  secondary set (FR3 as registered), seed 1 only, one per property,
#         ONLY the cells whose strength differs from the primary set
task_spec() {
    local t=$1
    if [ "$t" -lt 9 ]; then
        echo "${PROPS[$((t % 3))]} ${SEEDS[$((t / 3))]} $FULL_ARMS primary -"
    else
        echo "${PROPS[$((t - 9))]} ${SEEDS[0]} $FULL_ARMS mae primary"
    fi
}

# Run one task. Every command's status is checked explicitly: `set -e` is
# suspended inside a function called from `if`/`||`, so relying on it would
# let a failed preflight fall through to the real run.
run_task() {
    local task=$1 maxmin=$2
    local prop seed arms sets excl
    read -r prop seed arms sets excl <<< "$(task_spec "$task")"
    [ "$excl" = "-" ] && excl=""
    local outdir="$RUNDIR/seed${seed}"
    mkdir -p "$outdir" || return 1
    echo "=== task $task : prop $prop  seed $seed  arms $arms  sets $sets${excl:+ minus $excl}  n $N"

    freeze || return $?

    local left
    left=$(python proj1/scripts/guidance_sweep.py --stage full --frozen "$FROZEN" \
               --props "$prop" --arms "$arms" --sets "$sets" --exclude-sets "$excl" \
               --n "$N" --seed "$seed" --out-dir "$outdir" \
               --fm "$FM" --dry-run | sed -n 's/.*to run \([0-9][0-9]*\).*/\1/p')
    if [ -z "$left" ]; then
        echo "could not read the plan for task $task"
        return 1
    fi
    if [ "$left" = "0" ]; then
        echo "task $task already complete"
        return 0
    fi
    echo "$left cells to run for task $task"

    # Fail fast: one throwaway cell per arm at n=8, writes nothing.
    srun python -u proj1/scripts/guidance_sweep.py --stage full --frozen "$FROZEN" \
        --props "$prop" --arms "$arms" --sets "$sets" --exclude-sets "$excl" \
        --fm "$FM" --per-mol --preflight || return 1
    echo "preflight passed"

    srun python -u proj1/scripts/guidance_sweep.py --stage full --frozen "$FROZEN" \
        --props "$prop" --arms "$arms" --sets "$sets" --exclude-sets "$excl" \
        --fm "$FM" \
        --n "$N" --batch 128 --steps 100 --n-probe 1 \
        --seed "$seed" --out-dir "$outdir" --per-mol \
        --max-minutes "$maxmin" || return 1
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
    # insurance sweeper: every task in order, inside one shared time budget
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
