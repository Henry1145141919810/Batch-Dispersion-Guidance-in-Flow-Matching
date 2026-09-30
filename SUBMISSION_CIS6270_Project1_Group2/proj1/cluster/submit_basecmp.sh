#!/bin/bash
# Queue the WHOLE base-model comparison in one go. Nothing to resubmit.
#
#   bash proj1/cluster/submit_basecmp.sh              # the default budget
#   BUDGET_GPUH=30 bash proj1/cluster/submit_basecmp.sh
#   DRY=1 bash proj1/cluster/submit_basecmp.sh        # print, submit nothing
#
# Run it from a LOGIN NODE. It submits and exits in seconds; it computes nothing.
#
# THE CHAIN. Each stage waits on the one before with --dependency, so the queue
# holds the entire experiment from the start and no stage can begin on inputs
# that are not there:
#
#   probe(2) -> size(1) -> screen(96) -> sweep -> plan(2) -> refine(24)
#            -> sweep -> freeze(2) -> full(36) -> sweep -> table(1)
#
# WHY `afterany` AND NOT `afterok`, between stages. A task that hits the 225-min
# guard exits non-zero by design, having written every cell it finished; with
# `afterok` one such task would strand the rest of the queue forever. So stages
# are chained `afterany` and every stage that CONSUMES another's output refuses
# loudly instead: the freeze will not pick from a screen with a hole in it (the
# picks are an argmax, and a hole in an argmax is invisible in its result), and
# the full run will not start without a frozen file for its own backend. The
# failure mode is a stalled stage with a clear reason in its log, never a
# plausible-looking number computed from half a grid.
#
# THE SWEEPERS are the same script submitted WITHOUT --array: one job that walks
# every task of the stage in order inside a 225-min budget and finishes whatever
# the array left behind. Cells are resumable, so it only pays for what is missing.
#
# WHAT DECIDES n. Nothing here. STAGE=size measures the real per-cell cost from
# the probe's cells and picks n for the screen and the full run off fixed ladders
# to fit BUDGET_GPUH; the later jobs read it from results/basecmp/plan.json at
# run time. That is the whole reason the probe exists: EquiFM had never had a
# single cell run on it, so every cost estimate before it is an extrapolation
# from a different network.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
JOB=proj1/cluster/basecmp_run.slurm
BUDGET_GPUH="${BUDGET_GPUH:-96}"
FULL_BACKEND="${FULL_BACKEND:-equifm}"
DRY="${DRY:-0}"

cd "$PROJ" || exit 1
if [ ! -f "$JOB" ]; then
    echo "STOP: $JOB not found. Ship the code first (see docs/protocol/BETTY_RUNBOOK.md)."
    exit 1
fi
mkdir -p "$PROJ/logs" "$PROJ/results/basecmp"

EXPORTS="ALL,BUDGET_GPUH=$BUDGET_GPUH,FULL_BACKEND=$FULL_BACKEND"

# sub NAME ARRAY_SPEC DEPENDENCY STAGE -> echoes the new job id
sub() {
    local name=$1 arr=$2 dep=$3 stage=$4
    local cmd=(sbatch --parsable --job-name="cgm-bc-$name")
    [ -n "$arr" ] && cmd+=(--array="$arr")
    [ -n "$dep" ] && cmd+=(--dependency="$dep")
    cmd+=(--export="$EXPORTS,STAGE=$stage" "$JOB")
    if [ "$DRY" = "1" ]; then
        echo "DRY: ${cmd[*]}" >&2
        echo "dry-$name"
        return 0
    fi
    local id
    id=$("${cmd[@]}") || {
        echo "STOP: sbatch failed for $name" >&2
        exit 1
    }
    printf '%-10s %-9s dep=%-28s job %s\n' "$name" "${arr:-single}" "${dep:-none}" "$id" >&2
    echo "$id"
}

echo "submitting the base-model comparison"
echo "  budget      : $BUDGET_GPUH GPU-hours (screen n and full-run n are chosen to fit it)"
echo "  full run on : $FULL_BACKEND"
echo "  dry run     : $DRY"
echo

PROBE=$(sub probe   "0-1"  ""                     probe)
SIZE=$(sub  size    ""     "afterany:$PROBE"      size)
SCREEN=$(sub screen "0-95" "afterany:$SIZE"       screen)
SWEEP1=$(sub sweep1 ""     "afterany:$SCREEN"     screen)
SWEEP2=$(sub sweep2 ""     "afterany:$SWEEP1"     screen)
PLAN=$(sub  plan    "0-1"  "afterany:$SWEEP2"     plan)
REFINE=$(sub refine "0-23"  "afterany:$PLAN"       refine)
SWEEP3=$(sub sweep3 ""     "afterany:$REFINE"     refine)
FREEZE=$(sub freeze "0-1"  "afterany:$SWEEP3"     freeze)
FULL=$(sub  full    "0-35" "afterany:$FREEZE"     full)
SWEEP4=$(sub sweep4 ""     "afterany:$FULL"       full)
TABLE=$(sub  table  ""     "afterany:$SWEEP4"     table)

echo
echo "queued. watch it with:"
echo "  squeue -u \$USER -o '%.10i %.14j %.9T %.10M %R'"
echo
echo "what to read when it lands:"
echo "  results/basecmp/plan.json                  the n it chose, and why"
echo "  docs/results/BASECMP_SCREEN_fm.md          our base: every cell + the picks"
echo "  docs/results/BASECMP_SCREEN_equifm.md      EquiFM: every cell + the picks"
echo "  results/basecmp/<b>/screen_table.csv       the same, machine-readable"
echo "  results/basecmp/<b>/frozen_basecmp.json    the frozen (w, t) per arm, both sets"
echo "  results/basecmp/$FULL_BACKEND/full/n*/seed*/  the v2 full run's cells"
echo "  docs/results/BASECMP_FULL_$FULL_BACKEND.md   the full run, read under v2's V4-V8"
echo
echo "if a stage stalls, its log says why; the sweeper for that stage can be"
echo "resubmitted on its own:"
echo "  sbatch --export=$EXPORTS,STAGE=screen $JOB"
