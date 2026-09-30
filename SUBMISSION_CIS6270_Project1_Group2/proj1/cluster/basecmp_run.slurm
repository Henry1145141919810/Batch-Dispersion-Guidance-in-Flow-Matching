#!/bin/bash
#SBATCH --job-name=cgm-basecmp
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/logs/basecmp-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/basecmp-%j.err
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
#
# THE BASE-MODEL COMPARISON. One question: for each guidance method, does its
# behaviour survive a change of base model when NOTHING ELSE changes?
#
# Two generators -- ours (fm_last.pt) and the borrowed EquiFM -- are run through
# ONE external property pair: TFG's tf_predict_<p> as the guide and EDM's
# classifier (TFG evaluate_<p>) as the oracle, with OC-Flow's clean EGNN as a
# second opinion on both. Neither the guide nor the oracle is ours, and both are
# better than ours on all three properties. delta is the local rule (k=2 x the
# oracle's error NEAR the target) and depends on f_B and QM9 only, so it is the
# same number for both bases by construction.
#
# Two axes are screened JOINTLY, per arm, per property:
#   w        strength, log-spaced 0.05 .. 16 -- it reaches past w=4 on purpose,
#            because the v1 grid stopped at 4 and capped lgd_mc there
#   t_start  the flow-time instant guidance switches on; guidance runs on
#            [t_start, 1). On EquiFM the same instant is tau_max_guide =
#            1 - t_start. The project has measured this to be its LARGEST single
#            effect (alpha MAE 10.02 -> 5.32 from t=0.05 to 0.5) and has never
#            measured anything between 0.05 and 0.5, which this grid does.
#
# Each (property, arm) gets TWO picks: `floor` (best decoded in-band among cells
# clearing 0.9 x unguided molecule stability -- v2's rule V2, the only pick a
# headline may rest on) and `free` (the same with the floor removed, reported for
# reach and never headlined). Then the v2 full run on EquiFM uses BOTH.
#
# Arms, all seven: unguided, plug, tmpd, lgd_mc, tfg, btvg, btvg_var.
# Protocol: docs/protocol/BASECMP_PROTOCOL.md. Decision logic gated by
# proj1/tests/test_basecmp.py (30 gates, CPU, seconds).
#
# EIGHT STAGES. Submit them all at once with proj1/cluster/submit_basecmp.sh,
# which chains them on --dependency=afterany so nothing needs resubmitting --
# afterany, not afterok, because a task stopped by the 225-min guard exits
# non-zero by design and one of those would strand the rest of the queue.
# Every stage that CONSUMES another's output refuses loudly instead.
#
# 1. STAGE=probe    19 real cells per backend at n=256 (7 arms x 3 start-times,
#                   unguided once), to MEASURE what a cell costs AND how that
#                   cost scales with the guided fraction. EquiFM has never had
#                   a cell run on it, so every estimate before this is
#                   extrapolation from a different network.
#     sbatch --array=0-1 --export=ALL,STAGE=probe proj1/cluster/basecmp_run.slurm
# 2. STAGE=size     reads the probe, picks n for the screen and the full run off
#                   fixed ladders to fit BUDGET_GPUH, writes results/basecmp/plan.json.
#     sbatch --export=ALL,STAGE=size proj1/cluster/basecmp_run.slurm
# 3. STAGE=screen   the joint grid: 363 cells per backend (6 guided arms x 5
#                   strengths x 4 start-times x 3 properties, + 1 unguided each).
#     sbatch --array=0-95 --export=ALL,STAGE=screen proj1/cluster/basecmp_run.slurm
#     task = 2*(4*(3*t_index + prop) + group) + backend
#     96 tasks of 5, 10 or 11 cells (the six carrying an unguided cell are 11)
# 4. STAGE=plan     basecmp_freeze.py --emit-refine, per backend: which cells to
#                   add around each pick (geometric midpoints; an edge pick is
#                   extended outward instead, capped at w=64).
#     sbatch --array=0-1 --export=ALL,STAGE=plan proj1/cluster/basecmp_run.slurm
# 5. STAGE=refine   runs them.
#     sbatch --array=0-23 --export=ALL,STAGE=refine proj1/cluster/basecmp_run.slurm
#     task = 2*(N_GROUPS*prop + group) + backend
# 6. STAGE=freeze   the final picks + the whole screen as a csv and a markdown
#                   table, per backend.
#     sbatch --array=0-1 --export=ALL,STAGE=freeze proj1/cluster/basecmp_run.slurm
# 7. STAGE=full     the v2 full run on EquiFM at both frozen (w, t) sets, three
#                   seeds, per-molecule sidecars.
#     sbatch --array=0-35 --export=ALL,STAGE=full proj1/cluster/basecmp_run.slurm
#     task = 4*(3*seed_index + prop) + group
# 8. STAGE=table    reads the full run under v2's reporting rules (V4-V8) and
#                   writes docs/results/BASECMP_FULL_<backend>.md. Refuses a
#                   partial run: pooling a different number of seeds per arm
#                   makes the standard errors incomparable.
#     sbatch --export=ALL,STAGE=table proj1/cluster/basecmp_run.slurm
#
# INSURANCE, any stage: the same command WITHOUT --array walks every task of that
# stage in order inside one 225-min budget and finishes whatever the array left.
# Cells are resumable and written atomically (the json is the completion marker),
# so a killed task loses at most the cell it was on.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

STAGE="${STAGE:?set STAGE=probe|size|screen|plan|refine|freeze|full|table via --export=ALL,STAGE=...}"
PROPS=(mu alpha gap)
BACKS=(fm equifm)
T_STARTS=(0.05 0.25 0.5 0.75)
T_REF_INDEX=2                       # T_STARTS[2] = 0.5, where unguided is planned
# FOUR arm groups, not two, and the reason is the start-time axis. Guidance
# cost scales with the GUIDED FRACTION of the trajectory: at t_start = 0.05
# about 95 of the 100 Euler steps are guided, against 50 at t_start = 0.5 and 25
# at 0.75. A group of three expensive arms at the earliest window is roughly 1.9x
# its cost at the reference window, which puts it past the 225-min guard -- the
# failure project 1 main 2 measured on its own sweep (a 4.6 h task against a 4 h
# wall). The two costly arms therefore get a task each.
ARM_GROUPS=(plug,tmpd lgd_mc,tfg btvg btvg_var)
N_GROUPS=${#ARM_GROUPS[@]}
SEEDS=(20261001 20261002 20261003)
FULL_BACKEND="${FULL_BACKEND:-equifm}"
BUDGET_GPUH="${BUDGET_GPUH:-96}"
PROBE_N="${PROBE_N:-256}"
T_START_EPOCH=$(date +%s)
BUDGET_MIN=225
TS=proj1/scripts/transfer_sweep.py
BF=proj1/scripts/basecmp_freeze.py
BS=proj1/scripts/basecmp_size.py
BT=proj1/scripts/basecmp_table.py
ROOT="$PROJ/results/basecmp"
PLAN="$ROOT/plan.json"
FM_MD5=a190ac8394902027d4a951f8d30e8c5c

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "stage     : $STAGE"
echo "budget    : $BUDGET_GPUH GPU-h"
echo "started   : $(date -Is)"
echo

# ---- every asset checked against its pinned hash, before anything samples ----
python -u proj1/scripts/fetch_tfg_assets.py --verify --models "" || {
    echo "STOP: a vendored TFG property network is missing or fails its pinned md5"
    exit 1
}
python -u proj1/scripts/fetch_equifm_assets.py --verify || {
    echo "STOP: an EquiFM / OC-Flow file is missing or fails its pinned sha256"
    exit 1
}
# our generator is pinned the way v2_run.slurm pins it: the comparison is
# meaningless if the two columns were produced by different checkpoints
GOT=$(md5sum "$PROJ/proj1/checkpoints/fm_last.pt" | cut -d' ' -f1)
if [ "$GOT" != "$FM_MD5" ]; then
    echo "STOP: proj1/checkpoints/fm_last.pt md5 is $GOT, expected $FM_MD5"
    exit 1
fi
echo "assets    : TFG pair + EquiFM + OC-Flow verified; fm_last.pt $FM_MD5"
echo

case "$STAGE" in
    probe)  N_TASKS=2 ;;
    size)   N_TASKS=1 ;;
    screen) N_TASKS=$((2 * N_GROUPS * 3 * 4)) ;;   # backend x group x prop x t_start = 96
    plan)   N_TASKS=2 ;;
    refine) N_TASKS=$((2 * N_GROUPS * 3)) ;;   # backend x group x prop = 24
    freeze) N_TASKS=2 ;;
    full)   N_TASKS=$((N_GROUPS * 3 * 3)) ;;       # group x prop x seed = 36
    table)  N_TASKS=1 ;;
    *) echo "STAGE must be probe|size|screen|plan|refine|freeze|full|table, got '$STAGE'"; exit 1 ;;
esac

# read_n KEY: the sample size chosen by STAGE=size. Refuses to guess.
read_n() {
    local key=$1
    if [ ! -s "$PLAN" ]; then
        echo "MISSING_PLAN"
        return 1
    fi
    python - "$PLAN" "$key" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))[sys.argv[2]]))
PY
}

screen_dir()  { echo "$ROOT/$1/screen"; }
probe_dir()   { echo "$ROOT/$1/probe"; }
refine_json() { echo "$ROOT/$1/refine.json"; }
frozen_json() { echo "$ROOT/$1/frozen_basecmp.json"; }

# task -> the transfer_sweep.py arguments that select its cells
task_args() {
    local t=$1 n=$2
    case "$STAGE" in
    probe)
        local b=${BACKS[$t]}
        # One property, one strength, every arm, at THREE start-times. Three,
        # not one, because guidance cost scales with the guided fraction of the
        # trajectory and a single window cannot separate the fixed sampling cost
        # from the per-guided-step cost -- so it would price the t=0.05 tasks as
        # if they cost what the t=0.5 ones do, which is how a task ends up over
        # the wall. It writes to its OWN directory so the screen's consistency
        # gate never sees these small-n cells.
        echo "--backend $b --stage basecmp --props mu --arms unguided,plug,tmpd,lgd_mc,tfg,btvg,btvg_var --strengths 1 --t-starts 0.05,0.5,0.75 --n $PROBE_N --out-dir $(probe_dir "$b")" ;;
    screen)
        local b=${BACKS[$((t % 2))]}
        local u=$((t / 2))
        local g=$((u % N_GROUPS))
        local v=$((u / N_GROUPS))
        local p=${PROPS[$((v % 3))]}
        local k=$((v / 3))
        local arms=${ARM_GROUPS[$g]}
        # exactly one task writes the single unguided cell: the reference window,
        # group A. Any other task planning it would give the freeze several
        # unguided cells and an ambiguous chemistry floor.
        if [ "$k" -eq "$T_REF_INDEX" ] && [ "$g" -eq 0 ]; then
            arms="unguided,$arms"
        fi
        echo "--backend $b --stage basecmp --props $p --arms $arms --t-starts ${T_STARTS[$k]} --n $n --out-dir $(screen_dir "$b") --per-mol" ;;
    refine)
        # Split by arm group as well as property. With one task per
        # (backend, property) the refine carried every arm and both sets, and the
        # cells it adds by extending an edge are disproportionately at the early,
        # expensive windows -- the sizer priced that task at 182 of the 225
        # available minutes, the largest in the whole chain.
        local b=${BACKS[$((t % 2))]}
        local u=$((t / 2))
        local g=$((u % N_GROUPS))
        local p=${PROPS[$((u / N_GROUPS))]}
        echo "--backend $b --stage basecmprefine --refine $(refine_json "$b") --props $p --arms ${ARM_GROUPS[$g]} --n $n --out-dir $(screen_dir "$b") --per-mol" ;;
    full)
        local g=$((t % N_GROUPS))
        local u=$((t / N_GROUPS))
        local seed=${SEEDS[$((u / 3))]}
        local p=${PROPS[$((u % 3))]}
        local arms=${ARM_GROUPS[$g]}
        if [ "$g" -eq 0 ]; then
            arms="unguided,$arms"
        fi
        echo "--backend $FULL_BACKEND --stage basecmpfull --frozen $(frozen_json "$FULL_BACKEND") --props $p --arms $arms --seed $seed --n $n --out-dir $ROOT/$FULL_BACKEND/full/n$n/seed$seed --per-mol" ;;
    esac
}

run_sweep_task() {
    local task=$1 maxmin=$2 n=$3
    local a
    a="$(task_args "$task" "$n")"
    echo "=== task $task : $a"
    local left
    # shellcheck disable=SC2086
    left=$(python "$TS" $a --dry-run | sed -n 's/.*to run \([0-9][0-9]*\).*/\1/p')
    if [ -z "$left" ]; then
        echo "could not read the plan for task $task (is the stage before it finished?)"
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

run_task() {
    local task=$1 maxmin=$2 n
    case "$STAGE" in
    probe)
        run_sweep_task "$task" "$maxmin" "$PROBE_N" ;;
    size)
        srun python -u "$BS" --root "$ROOT" --budget-gpuh "$BUDGET_GPUH" \
            --full-backend "$FULL_BACKEND" --out "$PLAN" ;;
    screen|refine)
        n=$(read_n n_screen) || { echo "STOP: $PLAN missing -- STAGE=size has not run"; return 3; }
        run_sweep_task "$task" "$maxmin" "$n" ;;
    full)
        n=$(read_n n_full) || { echo "STOP: $PLAN missing -- STAGE=size has not run"; return 3; }
        run_sweep_task "$task" "$maxmin" "$n" ;;
    plan)
        local b=${BACKS[$task]}
        srun python -u "$BF" --backend "$b" --screen-dir "$(screen_dir "$b")" \
            --allow-floor-limited --emit-refine "$(refine_json "$b")"
        local rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "STOP: the refine plan refused (rc=$rc) -- the screen for $b is"
            echo "incomplete or inconsistent. Nothing was run."
            return 3
        fi
        ;;
    freeze)
        local b=${BACKS[$task]}
        # --allow-floor-limited: an arm that never clears the floor on this grid
        # is a RESULT (it cannot steer this base model within the chemistry
        # budget), not a reason to stop the run. It is labelled floor_limited in
        # the frozen file, in the csv, in the markdown and in every cell the full
        # run writes from it, and it must not be reported as a win.
        srun python -u "$BF" --backend "$b" --screen-dir "$(screen_dir "$b")" \
            --allow-floor-limited \
            --json-out "$(frozen_json "$b")" \
            --table-out "$ROOT/$b/screen_table.csv" \
            --md-out "$PROJ/docs/results/BASECMP_SCREEN_$b.md"
        local rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "STOP: the freeze refused (rc=$rc) for $b. Nothing was frozen."
            return 3
        fi
        ;;
    table)
        # The reader for this stage's own cells. The v2 table scripts cannot read
        # them (they filter on stage == "full", key by (prop, arm) so the two
        # strength sets collide, and refuse a directory with mixed t_min_guide,
        # which every basecmp full directory has because each arm carries its own
        # frozen window). That same filter is why a basecmp cell can never
        # contaminate the real v2 table.
        srun python -u "$BT" --backend "$FULL_BACKEND" --root "$ROOT"             --frozen "$(frozen_json "$FULL_BACKEND")"             --md-out "$PROJ/docs/results/BASECMP_FULL_$FULL_BACKEND.md"
        local rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "STOP: the table refused (rc=$rc) -- the full run is incomplete"
            echo "or inconsistent. Nothing was written."
            return 3
        fi
        ;;
    esac
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
        USED=$(( ($(date +%s) - T_START_EPOCH) / 60 ))
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
