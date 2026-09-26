#!/bin/bash
#SBATCH --job-name=cgm-v3
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/v3-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/v3-%j.err
#
# PROTOCOL v3 (docs/protocol/FULL_RUN_V3_PROTOCOL.md) and its BDG ablation.
# One job script, three stages, two base models.
#
#   target    q50 for EVERY property (not v2's q90)
#   strength  w = 1 for EVERY arm
#   window    guidance on for t >= 0.5
#   seeds     20261001 / 20261002 / 20261003
#   n         2000 per cell, in batches of 500 (see THE BATCH below)
#   floor     NONE. No arm is excluded for poor chemistry; chemistry is REPORTED
#   arms      unguided, plug, tmpd, lgd_mc, tfg, and BDG. btvg is dropped
#   BDG       headline eta = 4, tau_mult in {0.5, 1}  (SETPOINT, not window)
#             ablation eta in {0,1,2,4,8} x tau_mult in {0.5,0.75,1,1.5}
#   bases     ours (fm) and EquiFM -- identical code, one flag
#
# THE BATCH IS BDG'S ESTIMATOR, so --batch is passed explicitly, and it is 500.
# guidance.py asks for more: "V_b is computed over whatever tensor the sampler
# hands in, so running n in several batches runs several independent
# controllers. The sweep must use one batch per cell." ONE BATCH OF 2000 DOES
# NOT FIT AND THIS SCRIPT ONCE ASKED FOR 5000. Measured peak allocation for one
# bdg_e4t0.5 cell, batches 32/64/128, on an RTX 5080 (docs/results/V3_BATCH_MEMORY.md):
#
#     our base   0.0214 GiB allocated, 0.0401 GiB reserved per molecule
#     EquiFM     0.0395 GiB allocated, 0.0560 GiB reserved per molecule  (1.85x)
#
# Linear in the batch and flat in the step count, both reproduced exactly. One
# batch of 2000 therefore needs ~43 GiB (ours) and ~79 GiB (EquiFM) allocated
# -- 80 and 112 GiB reserved -- against this partition's 45 GB. EquiFM would
# have died on its first cell, and the n = 5000 this run used to plan needed
# 200 and 280 GiB reserved, so BOTH backends would have.
#
# 500 still fits with headroom -- measured ON BETTY at 17.57 GiB reserved on our
# base and 26.65 on EquiFM, inside an 80 % budget of 33.5 GiB -- and it is still
# the largest divisor that does: 1000 needs ~56 GiB, i.e. OOM. So it did NOT
# change when n did. A cell therefore runs FOUR controllers of 500, not ten and
# not one. Each estimates V_b from 500 samples -- a 6.3 % standard error on the
# variance, against 53 % for the 8-molecule remainder the old batch-128 default
# would have produced -- and the cell's reported controller state is the mean
# over those four. CONTROLLER COUNT is what the n cut economised; controller
# QUALITY is untouched, which is the only sane way to make a BDG run cheaper. That is a real deviation from "one batch per cell" and it is
# recorded in FULL_RUN_V3_PROTOCOL.md section 2.4 and in section 6's "cannot"
# list, not just here. Every arm uses the same batch, so arms stay paired on the
# same initial noise. V3_BATCH overrides it; it must divide N.
#
# WHY GPU AND NOT CPU. Measured on one identical cell (mu, plug, n = 128, batch
# 128, 100 steps) on the LAPTOP, not on Betty: 0.1 min on an RTX 5080 against
# 2.6 min on its CPU, a 26x gap. A v3 guided cell is 3500 batched passes (10
# batches x a recorded 350, below), each carrying 500 molecules; the BDG author's
# CPU choice was right for their n = 512 single-batch cells (350 passes on one
# batch) and does not transfer.
#
# THE TWO BASES RUN THE SAME NUMBER OF PASSES. Read out of the cells' own cost
# counters at n = 64, 100 steps, bdg_e4t0.5: BOTH backends record nfe = 350
# (gen_fwd 200, gen_vjp 50, guide_fwd 50, guide_bwd 50). So the backends are
# matched in function evaluations and EquiFM's extra cost is per pass only -- it
# is the bigger net (nf = 256, 9 layers). That per-pass ratio is NOT measured:
# the laptop was contended by another job and the same fm cell timed 17.9 s and
# then 12.2 s, a 46 % swing, so no wall-clock number from it is usable. The
# repo's one timed external backend (TFG/EDMsecond) runs 1.83x ours and EquiFM's
# memory slope is 1.85x ours, so the split below is sized for 2.5x as a margin,
# and --max-minutes plus the [timing] print are what actually protect the wall.
#
# TASKS ARE SPLIT BY PROPERTY, which is what keeps them inside the 4-hour QOS
# window. Rate: 2.468 min/unit on our base (v2_run.slurm's measured 58 min for
# 23.5 units at n=5000). Per (property, seed) the units are 23.5 for the 5
# comparison arms plus 3.5 per BDG arm, so:
#
# Units are DEFINED AT n = 5000 and scale linearly in n, so multiply by n/5000
# -- 0.4 for the headline's 2000, 0.2 for the ablation's 1000:
#
#   headline task, 7 arms = 30.5 units x 0.4 = 12.2 units
#                         = 30 min ours, 55 min at 1.83x, 75 min at 2.5x
#   ablation task, 17 arms = 59.5 units x 0.2 = 11.9 units
#                         = 29 min ours, 54 min at 1.83x, 73 min at 2.5x
#
# The longest task is therefore 75 min against BUDGET_MIN 225 and a 4-hour wall
# -- a 3x margin, where at n = 5000 it was 210 against 225. 2.5x is a MARGIN,
# not a measurement (see above).
#
# CAVEAT ON LINEARITY. The only MEASURED datum is v2's 58 min for 23.5 units at
# n = 5000; the scaling is assumed. Per-task FIXED cost does not scale with n --
# checkpoint load, calibration_indices + build_pair over 3000 molecules, and the
# second oracle's calibration are paid once per task. At an 84-min task that was
# noise; at 30 min it is a visible fraction, so treat 12.2 units as a FLOOR and
# read the real number off the [timing] line.
#
#   STAGE=v3      18 tasks = 2 backends x 3 properties x 3 seeds
#                 task = 9*backend + 3*property + seed; all 7 headline arms
#   STAGE=v3abl   6 tasks = 2 backends x 3 properties, ONE seed, all 17 arms
#                 task = 3*backend + property
#   STAGE=table   one job, no array
#
# proj1/tests/test_v3.py PARSES this arithmetic out of this file and asserts the
# tasks tile the plan with no holes, no duplicates and no task resolving to an
# empty arm list, and that every flag below exists in transfer_sweep.py. Both
# gates exist because both bugs shipped once: a wrong stride left 27 cells
# unproducible, and --save-coords (guidance_sweep.py's flag, not this script's)
# would have killed every task before it wrote a single cell.
#
# SUBMIT: proj1/cluster/submit_v3.sh chains everything in one paste.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"
# Without this the login node's SLURM_CONF is not inherited and srun dies on a
# DNS SRV lookup. Recorded in BETTY_RUNBOOK and hit twice before.
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

TS=proj1/scripts/transfer_sweep.py
VT=proj1/scripts/v3_table.py
STAGE="${STAGE:-v3}"
# NO DEFAULT. v3 pre-registers no cell size: the operator picks it for the
# cluster they are on (Henry, 26 Sep). A default here would let a job run --
# and write a tree that looks real -- at a size nobody chose. Pick it against
# the power table in docs/protocol/FULL_RUN_V3_PROTOCOL.md section 6.1, and
# make it divisible by BATCH, because the batch is BDG's ESTIMATOR and a
# remainder batch is a second, far noisier controller pooled in as an equal.
N="${V3_N:-}"
if [ -z "$N" ]; then
  echo "FATAL: V3_N is not set, and v3 has no default cell size." >&2
  echo "The protocol leaves n to the operator. Choose it against the power" >&2
  echo "table in docs/protocol/FULL_RUN_V3_PROTOCOL.md section 6.1, then:" >&2
  echo "  V3_N=<n> bash proj1/cluster/submit_v3.sh" >&2
  exit 1
fi
BATCH="${V3_BATCH:-500}"
# THE ABLATION IS SMALLER THAN THE HEADLINE (Henry, 26 Sep): n = 1000 and one
# seed, 4.2 GPU-h against the headline's 12.8. (The 51.3 this line used to cite
# was the ABLATION's own first plan -- 14 grid arms in a shared n5000 tree --
# never the headline. Do not read it as a headline figure.) See
# docs/protocol/ABLATION_V3_PROTOCOL.md. The batch does NOT shrink with it --
# 1000 / 500 = 2, so each controller still estimates V_b from 500 molecules (a
# 6.3 % standard error); the cell runs two of them against the headline's four.
# Controller COUNT is what was economised, not controller QUALITY, which is the
# only sane way to make a BDG run cheaper.
#
# 1000 AND NOT 2000, because 2000 is now the HEADLINE's n. At equal n both
# stages write results/v3/<be>/n2000/ under one stage label, and v3_table.py --
# which requires every (prop, arm) at all three seeds -- would glob the
# ablation's 15 single-seed arms into the headline's table and refuse. It must
# stay divisible by BATCH.
# The ablation runs at the SAME n and the SAME seeds as the headline. It used
# to run smaller on both, purely so that n could tell the two trees apart;
# transfer_sweep now writes results/v3/<be>/<stage>/n<N>/seed<S>/, so the
# STAGE directory does that job and n is free to match.
if [ "$BATCH" -le 0 ] || [ $(( N % BATCH )) -ne 0 ]; then
  echo "FATAL: batch $BATCH does not divide n $N." >&2
  echo "BDG's controller is the batch: a remainder batch is a second, smaller" >&2
  echo "controller whose variance estimate is far noisier, and it would be" >&2
  echo "pooled into this cell's diagnostics as though it were an equal." >&2
  exit 1
fi
PROPS=(mu alpha gap)
SEEDS=(20261001 20261002 20261003)
# The ablation uses ALL THREE, the same three (Henry, 26 Sep, final). It ran a
# single seed for two days; every v3 row now carries an across-seed spread, so
# an ablation row and a headline row at the same n ARE comparable.
BACKENDS=(fm equifm)
FM_CKPT=proj1/checkpoints/fm_last.pt
FM_MD5=a190ac8394902027d4a951f8d30e8c5c
V3_ARMS="unguided,plug,tmpd,lgd_mc,tfg,bdg_e4t0.5,bdg_e4t1"
# ALL 17 grid arms, in one list. They are not split into halves any more and the
# headline's are no longer excluded: at n = 1000 the ablation writes to
# results/v3/<be>/n1000/, a different tree from the headline's n2000/, so there
# is nothing to reuse and the grid must be self-contained to be readable at all.
# The grid keeps ALL FOUR tau_mults even though the headline now runs two: the
# 0.75 rung lives here, and a monotonicity or curvature claim about the tau
# ladder must be read off this grid, never off the headline's two endpoints.
# 17 arms = eta 0 once at tau_mult 1 (the dispersion term is multiplied by zero,
# so tau_mult cannot change those samples) + eta {1,2,4,8} x tau_mult
# {0.5,0.75,1,1.5}. One task carries all 17: at n = 1000 that is 17 x 3.5 x 0.2
# = 11.9 cost units, 29 min on our base and about 54 min on EquiFM (73 at the
# 2.5x margin), well inside BUDGET_MIN.
# ONE line on purpose: test_v3.py PARSES this literal out of this file rather
# than re-deriving it, which is how a stride bug was caught once before. A
# generated list would force the test to execute shell instead.
ABL_ARMS="bdg_e0t1,bdg_e1t0.5,bdg_e1t0.75,bdg_e1t1,bdg_e1t1.5,bdg_e2t0.5,bdg_e2t0.75,bdg_e2t1,bdg_e2t1.5,bdg_e4t0.5,bdg_e4t0.75,bdg_e4t1,bdg_e4t1.5,bdg_e8t0.5,bdg_e8t0.75,bdg_e8t1,bdg_e8t1.5"
BUDGET_MIN="${BUDGET_MIN:-225}"
T_START=$(date +%s)

echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "array   : ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "stage   : $STAGE"
echo "n,batch : $N, $BATCH"

if [ "$STAGE" != "table" ]; then
  if [ ! -f "$FM_CKPT" ]; then
    echo "FATAL: $FM_CKPT not found." >&2; exit 1
  fi
  GOT=$(md5sum "$FM_CKPT" | cut -d' ' -f1)
  if [ "$GOT" != "$FM_MD5" ]; then
    echo "FATAL: $FM_CKPT md5 $GOT, expected $FM_MD5." >&2
    echo "Every cell in this project used a190ac83. A different generator would" >&2
    echo "make v3's two halves incomparable to each other and to v2." >&2
    exit 1
  fi
  python -u proj1/scripts/fetch_tfg_assets.py --verify --models "" || {
      echo "FATAL: TFG assets missing or corrupt (the guide and the oracle)." >&2
      exit 1; }
  python -u proj1/scripts/fetch_equifm_assets.py --verify || {
      echo "FATAL: EquiFM assets missing or corrupt." >&2
      echo "v3 runs on BOTH bases, so this is fatal even for --backend fm:" >&2
      echo "a half-run comparison is not a comparison." >&2
      exit 1; }
fi

run_one () {            # $1 backend, $2 property, $3 seed, $4 arms, $5 min left
  local be="$1" prop="$2" seed="$3" armset="$4" maxmin="$5"
  echo "=== $STAGE  $be  $prop  seed=$seed  arms=$armset  (${maxmin} min left) ==="
  local t0; t0=$(date +%s)
  srun python -u "$TS" \
      --stage "$STAGE" \
      --backend "$be" \
      --props "$prop" \
      --arms "$armset" \
      --n "$N" \
      --batch "$BATCH" \
      --seed "$seed" \
      --per-mol \
      --max-minutes "$maxmin"
  local rc=$?
  echo "    [timing] $be $prop seed=$seed took $(( ($(date +%s) - t0) / 60 )) min"
  return $rc
}

# ONE GRID FOR BOTH STAGES. The ablation used to collapse to a single seed and
# needed its own strides; it now runs the SAME three seeds as the headline
# (Henry, 26 Sep, final), so the two differ ONLY in which arm list a task
# carries. Deriving the count from ${#BACKENDS[@]} rather than hardcoding it is
# what lets a backend be added without touching the arithmetic -- the stride bug
# this file's gates exist for came from a hand-maintained constant.
NP=${#PROPS[@]}
NS=${#SEEDS[@]}
NB=${#BACKENDS[@]}

n_tasks () { echo $(( NB * NP * NS )); }

task_spec () {          # $1 task id -> echoes "backend property seed armset"
  local t="$1" arms
  if [ "$STAGE" = "v3abl" ]; then arms="$ABL_ARMS"; else arms="$V3_ARMS"; fi
  echo "${BACKENDS[$(( t / (NP * NS) ))]}"        "${PROPS[$(( (t / NS) % NP ))]}"        "${SEEDS[$(( t % NS ))]}"        "$arms"
}

RC=0
if [ "$STAGE" = "preflight" ]; then
  # Two checks the array cannot do for itself, ~10 min, and the chain depends on
  # this with afterok so a failure stops the run instead of decorating it.
  #
  #   1. EVERY ARM RUNS ON EVERY BACKEND. One throwaway cell per arm at n=8,
  #      nothing written, nonzero exit if any arm raises. This is the check that
  #      catches an arm that is broken only on EquiFM -- a missing kwarg, a shape
  #      that only the two-clock sampler hits. Without it, run_cell's per-cell
  #      `except Exception: continue` would log the failure and carry on, and a
  #      BASELINE arm could be missing from all 18 tasks before anyone noticed.
  #   2. THE BATCH FITS AT THE REAL SIZE on this machine. The slope that chose
  #      500 was fitted on a 16 GB laptop card at batches <= 128; this measures
  #      the actual slice with the actual checkpoint.
  # V3_ARMS and ABL_ARMS overlap on the three eta=4 rungs; --preflight dedupes
  # arms, so the repeat costs nothing. (This line named $ABL_0/$ABL_1 until the
  # ablation was reshaped, which under `set -u` is a fatal unbound variable --
  # the arm-list gates all passed and the job would still have died here, which
  # is why shell_references_only_defined_vars exists below.)
  ALL_ARMS="$V3_ARMS,$ABL_ARMS"
  for BE in "${BACKENDS[@]}"; do
    echo "=== preflight: every arm on $BE (n=8, nothing written) ==="
    # --seed is REQUIRED by --stage v3 even under --preflight, which writes
    # nothing: the stage validates its arguments before it looks at the mode.
    # Omitting it cost a submission -- both arm checks died in argparse in 7
    # seconds, and test_v3.py now asserts every "$TS" call here passes it.
    srun python -u "$TS" --stage v3 --backend "$BE" --props mu \
        --arms "$ALL_ARMS" --seed "${SEEDS[0]}" --preflight || RC=$?
  done
  echo "=== preflight: does batch $BATCH fit on this card? ==="
  srun python -u proj1/scripts/batch_memprobe.py --confirm "$BATCH" \
      --backends "$(IFS=,; echo "${BACKENDS[*]}")" \
      --n "$N" --fm-ckpt "$FM_CKPT" \
      --json-out results/v3_batch_memory_betty.json \
      --md-out docs/results/V3_BATCH_MEMORY_BETTY.md || RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "PREFLIGHT FAILED (rc=$RC). The chain depends on this with afterok, so" >&2
    echo "nothing else will run. Fix the arm or lower V3_BATCH and resubmit." >&2
  fi
elif [ "$STAGE" = "table" ]; then
  # TWO TREES, told apart by the STAGE directory, not by n. Both stages run at
  # the same n and the same three seeds now, so --stage selects the directory
  # AND the arm set v3_table expects. Passing the wrong one makes it refuse:
  # the ablation carries no comparison arm by design, so reading it with the
  # headline's plan reports all five as missing.
  ALL_SEEDS="$(IFS=,; echo "${SEEDS[*]}")"
  # a tree with no cells is SKIPPED, not failed: submitting only the ablation
  # must not make its table job red over a headline that was never run
  for SPEC in "v3|" "v3abl|_ABL"; do
    TST="${SPEC%%|*}"; SUF="${SPEC#*|}"
    FOUND=0
    for BE in "${BACKENDS[@]}"; do
      [ -d "$PROJ/results/v3/$BE/$TST/n$N" ] && FOUND=1
    done
    if [ "$FOUND" -eq 0 ]; then
      echo "=== table: no $TST cells under n$N on any backend, skipping ==="
      continue
    fi
    echo "=== table: stage=$TST n=$N seeds=$ALL_SEEDS -> V3_RESULTS$SUF*.md ==="
    srun python -u "$VT" --both --stage "$TST" --n "$N" --seeds "$ALL_SEEDS" \
        --md-out "$PROJ/docs/results/V3_RESULTS$SUF.md" || RC=$?
    for BE in "${BACKENDS[@]}"; do
      srun python -u "$VT" --backend "$BE" --stage "$TST" --n "$N" --seeds "$ALL_SEEDS" \
          --md-out "$PROJ/docs/results/V3_RESULTS${SUF}_$BE.md" || RC=$?
    done
  done
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  MAX=$(( $(n_tasks) - 1 ))
  if [ "$SLURM_ARRAY_TASK_ID" -gt "$MAX" ]; then
    echo "FATAL: task $SLURM_ARRAY_TASK_ID, but STAGE=$STAGE has $(n_tasks) tasks" >&2
    echo "(0-$MAX). Submit --array=0-$MAX for this stage." >&2
    exit 1
  fi
  read -r BE PROP SEED ARMSET <<<"$(task_spec "$SLURM_ARRAY_TASK_ID")"
  run_one "$BE" "$PROP" "$SEED" "$ARMSET" "$BUDGET_MIN" || RC=$?
else
  # the insurance sweeper: walk every task of THIS stage and finish what the
  # array left. It cannot do a whole stage inside one 4-hour window -- it is a
  # finisher, not a substitute -- which is why the paste chains two of them.
  for T in $(seq 0 $(( $(n_tasks) - 1 ))); do
    USED=$(( ($(date +%s) - T_START) / 60 ))
    LEFT=$(( BUDGET_MIN - USED ))
    [ "$LEFT" -le 5 ] && { echo "[budget] $USED min used, stopping"; break; }
    read -r BE PROP SEED ARMSET <<<"$(task_spec "$T")"
    run_one "$BE" "$PROP" "$SEED" "$ARMSET" "$LEFT" || RC=$?
  done
fi

for BE in "${BACKENDS[@]}"; do
  # Both stages run 3 properties x 3 seeds; they differ only in the arm count,
  # so the target is derived rather than hardcoded -- the headline's used to be
  # a literal 198 that counted results/v3/$BE entire, swept in the ablation's
  # cells and could never be reached. Each count is scoped to its OWN stage
  # subtree, which is what now separates them.
  if [ "$STAGE" = "v3abl" ]; then N_ARMS=$(( $(echo "$ABL_ARMS" | tr ',' '
' | wc -l) ))
  else N_ARMS=$(( $(echo "$V3_ARMS" | tr ',' '
' | wc -l) )); fi
  WANT=$(( N_ARMS * NP * NS ))
  TREE="$PROJ/results/v3/$BE/$STAGE/n$N"
  echo "cells $BE: $(find "$TREE" -name 'tr__*.json' 2>/dev/null | wc -l) / $WANT"
done
echo "elapsed: $(( ($(date +%s) - T_START) / 60 )) min, rc=$RC"
exit $RC
