#!/bin/bash
#SBATCH --job-name=cgm-v2
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/v2-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/v2-%j.err
#
# THE v2 FULL RUN: fixed target (q90 of QM9), n = 5000, 3 seeds.
# Protocol: docs/protocol/FULL_RUN_V2_PROTOCOL.md (rules V1-V8).
#
# WHAT IS DIFFERENT FROM v1 (full_run.slurm), and why nothing collides:
#   target      q90, a FIXED value per property, not v1's per-molecule `dist`
#   strengths   rule V2 (best in_band clearing the 0.9x chemistry floor),
#               frozen by freeze_v2.py -- NOT check_fullrun_go.py, which
#               implements v1's FR3a and selects on MAE
#   output      results/full/v2/n5000/seed<S>/ -- a separate tree, because
#               full_run_table.py asserts target_name == "dist" and would
#               refuse a directory holding both
#   cell names  carry the target, so `..__q90__..__full.json` can never be
#               confused with v1's `..__dist__..__full.json`
#
# ARMS. This job runs the COMPARISON SET ONLY:
#   unguided plug tmpd lgd_mc tfg
# `btvg` and `btvg_var` are deliberately NOT here (Henry, 24 Sep: queue the
# must-do comparison methods first). They are added later with
# --export=ALL,ARMS=btvg,btvg_var and land in the same tree as extra cells;
# nothing already run has to be repeated, because every other arm's strength,
# seed and target are already fixed.
#
# SUBMIT
#   sbatch --array=0-8 --export=ALL proj1/cluster/v2_run.slurm
#       task = 3 x seed_index + prop_index, so tasks 0-2 are a complete
#       seed-1 table. ~58 min per task on a 2g slice; ~9 GPU-h for all nine.
#   sbatch --dependency=afterany:<ARRAY_ID> --export=ALL proj1/cluster/v2_run.slurm
#       no --array: the insurance sweeper, walks every task and finishes what
#       the array left. Chain a second one after it.
#
# THE FREEZE happens once, in whichever task starts first, and every later
# task reuses the file. freeze_v2.py REFUSES (exit 2) while any arm still
# clears the chemistry floor at the top of its measured strength grid, because
# the rule would then be capping that arm at wherever the grid stopped rather
# than at its chemistry limit. If it refuses, run the strengths it names at
# n = 512 first; nothing is sampled until it succeeds.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

PROPS=(mu alpha gap)
SEEDS=(20261001 20261002 20261003)
N="${FULL_N:-5000}"
ARMS="${ARMS:-unguided,plug,tmpd,lgd_mc,tfg}"
FM=proj1/checkpoints/fm_last.pt
FM_MD5=a190ac8394902027d4a951f8d30e8c5c      # every screen and full-run cell
T_START=$(date +%s)
BUDGET_MIN=225
RUNDIR="$PROJ/results/full/v2/n${N}"
FROZEN="$RUNDIR/frozen_v2.json"
N_TASKS=9

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "arms      : $ARMS"
echo "n         : $N"
echo "started   : $(date -Is)"
echo

# Pair or nothing: a cell from another generator would silently not pair with
# the screen that chose its strength.
GOT_MD5=$(md5sum "$FM" | cut -d' ' -f1)
if [ "$GOT_MD5" != "$FM_MD5" ]; then
    echo "STOP: $FM has md5 $GOT_MD5, not the screen's generator $FM_MD5"
    exit 1
fi

freeze() {
    if [ -s "$FROZEN" ]; then
        echo "strengths already frozen: $FROZEN"
        return 0
    fi
    mkdir -p "$RUNDIR" || return 1
    local tag="${SLURM_JOB_ID:-local}"
    # --arms $ARMS: gate only on the arms THIS job runs. Freezing all seven
    # would let an arm that is not being sampled (btvg_var) veto the run.
    python -u proj1/scripts/freeze_v2.py --arms "$ARMS"         --json-out "$FROZEN.pending.$tag"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "STOP: freeze_v2 refused (rc=$rc). Either the q90 screen is"
        echo "incomplete, or an arm is still grid-limited -- the lines above"
        echo "name the strengths to run at n=512 first. Nothing was sampled."
        rm -f "$FROZEN.pending.$tag"
        return 3
    fi
    # -n, not -f: concurrent array tasks each compute a freeze, and the first
    # one to land must win. A later -f could swap the file out from under a
    # task that has already read it, leaving two strengths in one run.
    mv -n "$FROZEN.pending.$tag" "$FROZEN"
    rm -f "$FROZEN.pending.$tag"
    [ -s "$FROZEN" ] || return 1
    echo "strengths frozen: $FROZEN"
    return 0
}

run_task() {
    local task=$1 maxmin=$2
    local prop=${PROPS[$((task % 3))]}
    local seed=${SEEDS[$((task / 3))]}
    local outdir="$RUNDIR/seed${seed}"

    freeze || return $?
    mkdir -p "$outdir" || return 1
    echo "=== task $task : prop $prop  seed $seed  arms $ARMS  n $N"

    local common="--stage full --frozen $FROZEN --full-target q90 \
--props $prop --arms $ARMS --sets primary --fm $FM"

    local left
    # shellcheck disable=SC2086
    left=$(python proj1/scripts/guidance_sweep.py $common --n "$N" --seed "$seed" \
               --out-dir "$outdir" --dry-run \
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

    # Fail fast: one throwaway cell per arm at n=8, writes nothing.
    # shellcheck disable=SC2086
    srun python -u proj1/scripts/guidance_sweep.py $common --preflight         --per-mol --save-coords || return 1
    echo "preflight passed"

    # shellcheck disable=SC2086
    srun python -u proj1/scripts/guidance_sweep.py $common \
        --n "$N" --batch 128 --steps 100 --n-probe 1 \
        --seed "$seed" --out-dir "$outdir" --per-mol --save-coords \
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
