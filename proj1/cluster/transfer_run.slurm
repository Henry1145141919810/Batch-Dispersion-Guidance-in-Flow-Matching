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
# THE TRANSFER: every guidance arm on a BORROWED flow-matching base model,
# EquiFM (Song et al. 2023), steered by TFG's guide tf_predict_<p> and scored
# by EDM's classifier (TFG evaluate_<p>) plus OC-Flow's second oracle. Arms:
# unguided, plug, tmpd, lgd_mc, tfg, btvg, btvg_var -- all seven fixed before
# any EquiFM cell ran. Protocol: docs/protocol/TRANSFER_EXPERIMENT_PLAN.md
# sections 10-11. BACKEND=edm runs the dropped EDMsecond variant instead.
#
# SIX STAGES, in two independent chains. Each task runs ONE ARM GROUP (or one arm and one
# strength chunk), so no task comes near the 225-min guard. 5080 minutes per
# cell at n=512: unguided 0.6, plug 1.8, tmpd 2.7, lgd_mc 3.4, tfg ~2.5,
# btvg_var 5.5, btvg ~6.7; n=2000 is ~3.9x, n=5000 ~9.8x.
#   group A = unguided, plug, tmpd, lgd_mc, tfg     group B = btvg, btvg_var
#
# 1. STAGE=compare   n=512, seed 20260922, 9 strengths (the main 7 + 0.1, 0.15
#                    for every arm) x q50/q90: 330 cells.
#      sbatch --array=0-11 --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm
#      task = 2 x (3 x target_index + prop_index) + group
# 2. STAGE=extend    the edge rule: an arm whose FR3a q90 pick is at a grid edge
#                    gets two strengths beyond it (x4, x16 or /4, /16). 0-12
#                    cells per property; freeze refuses until they exist.
#      sbatch --array=0-2 --export=ALL,STAGE=extend proj1/cluster/transfer_run.slurm
# 3. STAGE=full      the headline: `dist` on test[0:5000], seeds 20261001-3,
#                    strengths frozen ONCE (FR3a, decoded MAE) into
#                    results/transfer_equifm/full/n5000/frozen_q90.json.
#      sbatch --array=0-23 --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm
#      0-17 primary: task = 2 x (3 x seed_index + prop_index) + group
#      18-23 secondary (FR3 as registered): seed 1, only cells not in primary
#
# THE EQUAL-CHEMISTRY-COST COMPARISON (independent of 1-3, can run beside them)
# 4. STAGE=eqtune    `dist` on the fresh block test[10000:12000], n=2000, seed
#                    20261001, every arm over 0.05 ... 16 (10 strengths): 183 cells.
#      sbatch --array=0-56 --export=ALL,STAGE=eqtune proj1/cluster/transfer_run.slurm
#      0-2 unguided per property; then (property, arm, strength chunk)
# 5. STAGE=eqextend  the edge rule on the tune grid (as stage 2 for the screen)
#      sbatch --array=0-2 --export=ALL,STAGE=eqextend proj1/cluster/transfer_run.slurm
# 6. STAGE=eqconfirm frozen on the tune block (FR3a, decoded MAE) ONCE into
#                    results/transfer_equifm/eqchem/frozen_eqtune.json with the
#                    per-arm frontier; confirmed on test[5000:10000], n=5000,
#                    seeds 20261004-6.
#      sbatch --array=0-17 --export=ALL,STAGE=eqconfirm proj1/cluster/transfer_run.slurm
#      task = 2 x (3 x seed_index + prop_index) + group
#
# INSURANCE, any stage: the same command WITHOUT --array walks every task of
# that stage in order inside one 225-min budget and finishes whatever the
# array left. Chain it afterany on the array, and a second afterany on it.
#
# Every task first checks the checkpoints against their pinned hashes, then
# runs a preflight (one tiny cell per arm, writes nothing), then the real
# cells. Cells are resumable and written atomically; a stage whose inputs are
# incomplete stops before any sampling.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

STAGE="${STAGE:?set STAGE=compare|extend|full|eqtune|eqextend|eqconfirm via --export=ALL,STAGE=...}"
PROPS=(mu alpha gap)
TARGETS=(q50 q90)
BACKEND="${BACKEND:-equifm}"
if [ "$BACKEND" = "equifm" ]; then
    RESULTS="$PROJ/results/transfer_equifm"
else
    RESULTS="$PROJ/results/transfer"
fi
ARM_GROUPS=(unguided,plug,tmpd,lgd_mc,tfg btvg,btvg_var)
GUIDED=(plug tmpd lgd_mc tfg btvg btvg_var)
EQ_CHUNKS=(0.05,0.1,0.15,0.25 0.5,1,2,4 8,16)
SEEDS=(20261001 20261002 20261003)
EQ_SEEDS=(20261004 20261005 20261006)
FULL_N="${FULL_N:-5000}"
T_START=$(date +%s)
BUDGET_MIN=225
TS=proj1/scripts/transfer_sweep.py
COMPARE_DIR="$RESULTS/compare"
RUNDIR="$RESULTS/full/n${FULL_N}"
FROZEN="$RUNDIR/frozen_q90.json"
EQ_TUNE_DIR="$RESULTS/eqchem/tune"
EQ_CONFIRM_DIR="$RESULTS/eqchem/confirm/n5000"
EQ_FROZEN="$RESULTS/eqchem/frozen_eqtune.json"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array     : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "stage     : $STAGE"
echo "backend   : $BACKEND"
echo "started   : $(date -Is)"
echo

if [ "$BACKEND" = "equifm" ]; then
    python -u proj1/scripts/fetch_tfg_assets.py --verify --models "" || {
        echo "STOP: a vendored TFG property network is missing or does not match its pinned md5"
        exit 1
    }
    python -u proj1/scripts/fetch_equifm_assets.py --verify || {
        echo "STOP: an EquiFM / OC-Flow file is missing or does not match its pinned sha256"
        exit 1
    }
else
    python -u proj1/scripts/fetch_tfg_assets.py --verify || {
        echo "STOP: the EDMsecond checkpoint is missing or does not match its pinned md5"
        exit 1
    }
fi

case "$STAGE" in
    compare)   N_TASKS=12 ;;
    extend)    N_TASKS=3 ;;
    full)      N_TASKS=24 ;;
    eqtune)    N_TASKS=57 ;;
    eqextend)  N_TASKS=3 ;;
    eqconfirm) N_TASKS=18 ;;
    *) echo "STAGE must be compare|extend|full|eqtune|eqextend|eqconfirm, got '$STAGE'"; exit 1 ;;
esac

# freeze_once STAGE_NAME SOURCE_DIR TARGET_FILE: write the frozen strengths
# ONCE; concurrent tasks each compute the same file from the same finished
# cells and the atomic rename keeps one.
freeze_once() {
    local fstage=$1 src=$2 target=$3
    if [ -s "$target" ]; then
        echo "strengths already frozen: $target"
        return 0
    fi
    mkdir -p "$(dirname "$target")" || return 1
    local tag="${SLURM_JOB_ID:-local}"
    python -u "$TS" --stage "$fstage" --backend "$BACKEND" --out-dir "$src" \
        --json-out "$target.pending.$tag"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "STOP: $fstage refused (rc=$rc) -- its input stage is incomplete or"
        echo "inconsistent (see above). Nothing was run. Resubmit once it has finished."
        rm -f "$target.pending.$tag"
        return 3
    fi
    mv -f "$target.pending.$tag" "$target" || return 1
    echo "strengths frozen: $target"
    return 0
}

# task -> the transfer_sweep.py arguments that select its cells
task_args() {
    local t=$1
    local arms=${ARM_GROUPS[$((t % 2))]}
    local u=$((t / 2))
    case "$STAGE" in
    compare)
        echo "--stage compare --props ${PROPS[$((u % 3))]} --targets ${TARGETS[$((u / 3))]} --arms $arms --out-dir $COMPARE_DIR --per-mol" ;;
    extend)
        echo "--stage extend --props ${PROPS[$t]} --out-dir $COMPARE_DIR --per-mol" ;;
    full)
        if [ "$u" -lt 9 ]; then
            local seed=${SEEDS[$((u / 3))]}
            echo "--stage full --frozen $FROZEN --props ${PROPS[$((u % 3))]} --arms $arms --seed $seed --n $FULL_N --sets primary --out-dir $RUNDIR/seed$seed --per-mol"
        else
            local seed=${SEEDS[0]}
            echo "--stage full --frozen $FROZEN --props ${PROPS[$((u - 9))]} --arms $arms --seed $seed --n $FULL_N --sets mae --exclude-sets primary --out-dir $RUNDIR/seed$seed --per-mol"
        fi ;;
    eqtune)
        if [ "$t" -lt 3 ]; then
            echo "--stage eqtune --props ${PROPS[$t]} --arms unguided --out-dir $EQ_TUNE_DIR --per-mol"
        else
            local v=$((t - 3))
            local r=$((v % 18))
            echo "--stage eqtune --props ${PROPS[$((v / 18))]} --arms ${GUIDED[$((r / 3))]} --strengths ${EQ_CHUNKS[$((r % 3))]} --out-dir $EQ_TUNE_DIR --per-mol"
        fi ;;
    eqextend)
        echo "--stage eqextend --props ${PROPS[$t]} --out-dir $EQ_TUNE_DIR --per-mol" ;;
    eqconfirm)
        local seed=${EQ_SEEDS[$((u / 3))]}
        echo "--stage eqconfirm --frozen $EQ_FROZEN --props ${PROPS[$((u % 3))]} --arms $arms --seed $seed --out-dir $EQ_CONFIRM_DIR/seed$seed --per-mol" ;;
    esac
}

run_task() {
    local task=$1 maxmin=$2
    if [ "$STAGE" = "full" ]; then
        freeze_once freeze "$COMPARE_DIR" "$FROZEN" || return $?
    elif [ "$STAGE" = "eqconfirm" ]; then
        freeze_once eqfreeze "$EQ_TUNE_DIR" "$EQ_FROZEN" || return $?
    fi
    local a
    a="--backend $BACKEND $(task_args "$task")"
    echo "=== task $task : $a"
    local left
    # shellcheck disable=SC2086
    left=$(python "$TS" $a --dry-run | sed -n 's/.*to run \([0-9][0-9]*\).*/\1/p')
    if [ -z "$left" ]; then
        echo "could not read the plan for task $task (for extend/eqextend: is the stage before it finished?)"
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
