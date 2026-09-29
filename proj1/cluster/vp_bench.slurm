#!/bin/bash
#SBATCH --job-name=cgm-vp
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vp-%j.out
#SBATCH --error=logs/vp-%j.err
#
# THE VP BENCH SUITE -- our own QM9 diffusion model, and ONLY the gaps.
#
# The log paths above are RELATIVE on purpose. sbatch parses #SBATCH lines
# BEFORE the body runs, so CGM_PROJ cannot reach them; an absolute path built
# from a variable fails at launch with the log -- the thing that would have
# said why -- being what broke. Same reasoning as v3_run.slurm. The SUBMITTER
# must therefore mkdir logs/ before sbatch; submit_vp_bench.sh does.
#
# WHAT THIS PRODUCES
#   The 18 v3 cells for backend `vp`, and docs/results/V3_RESULTS_vp.md.
#   Those cells are what paper/main.tex:141 (tab:fmvd, row "VP, trained here")
#   needs for its four \pend values -- Mol. stable, Valid, Unique, s/sample.
#   NFE is 101 by design, not measured.
#
#   >> THE PAPER DOES NOT UPDATE ITSELF. paper/tools/build_results.py:67-69
#   >> HARDCODES the \pend row, and its `heads` loader at :34 reads only
#   >> ('fm','equifm','edm'). After this run, both need editing, plus a
#   >> "vp": ("unguided","plug") entry in v3_final_summary.BACKEND_ARMS:81.
#   >> (The upside of that hardcoding: dropping vp cells into results/v3/
#   >> cannot break the paper build.)
#
#   These come from the v3 cells, NOT from benchmark_base.py. results/bench/
#   is a different surface with a different protocol (10,000 samples, EDM
#   evaluator) and fills no pending paper cell.
#
# WHAT IT MUST NOT TOUCH
#   results/v3/{fm,equifm,edm}/ -- 441 + 441 + 18 cells, all committed.
#   This job writes ONLY under results/v3/vp/ (transfer_sweep.py:1939 builds
#   the path from the backend name), plus logs/ and the one results page.
#   Existing cells are skipped, never overwritten.
#   DO NOT use submit_v3.sh: its chain re-plans fm/equifm/edm, and its
#   preflight is the one afterok link, so a vp failure would stop all three.
#
#   STAGE=gate    one job, no array -- identity, unit gates, GPU preflight
#   STAGE=run     array 0-8, one (property, seed) per task   <- the 18 cells
#   STAGE=table   one job, no array -- sanity + the vp results page
#
# Submit the whole thing with proj1/cluster/submit_vp_bench.sh.
set -uo pipefail

PROJ="${CGM_PROJ:-/vast/projects/ajw/wharton/hyhuang/cgm}"
if [ ! -d "$PROJ" ]; then echo "FATAL: PROJ=$PROJ does not exist." >&2; exit 1; fi
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs" || { echo "FATAL: cannot create $PROJ/logs" >&2; exit 1; }
VENV="${CGM_VENV:-$PROJ/.venv}"
source "$VENV/bin/activate" || { echo "FATAL: no virtualenv at $VENV" >&2; exit 1; }

# Insurance, carried over from v3_run.slurm: without these the login node's
# SLURM_CONF is not inherited and `srun` dies on a DNS SRV lookup. That has
# been hit twice on this cluster and is recorded in BETTY_RUNBOOK.md.
export PATH="/cm/local/apps/slurm/current/bin:$PATH"
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}"

TS=proj1/scripts/transfer_sweep.py
STAGE="${STAGE:-run}"

# The grid. v3 pre-registers every one of these; none is a free parameter.
# FULL_RUN_V3_PROTOCOL.md section 1.3 (the 28 Sep amendment).
PROPS=(mu alpha gap)
SEEDS=(20261001 20261002 20261003)
ARMS="unguided,plug"          # V3_BACKENDS["vp"]["arms"], exactly. Not 7.
N="${VP_N:-2000}"
BATCH="${VP_BATCH:-500}"
NP=${#PROPS[@]}
NS=${#SEEDS[@]}

# DELTA is the pre-registered acceptance band for our property pair. If the
# wrong pair loads, every in_band number downstream is measured against a
# different band and nothing is comparable -- and NOTHING ELSE WOULD SHOW IT.
# v3_sanity.DELTA[("ours","mu")]; docs/results/V3_PAIR_DELTA.md measures
# 0.175407 and transfer_sweep prints it with %.5f.
DELTA_MU=0.17541

# THE CHECKPOINT MD5, PINNED -- the way FM_MD5 pins ours in v3_run.slurm.
#
# THIS EXISTS BECAUSE ITS ABSENCE ALREADY COST A RUN. On 28 Sep the first vp
# sweep silently loaded proj1/checkpoints/diff.pt, the epoch-75 remains of the
# crashed job 8590174, instead of the published epoch-1475 model. It has the
# SAME family, split, hidden and layers, so every field check passed; only the
# md5 differed (93ab4f72 against 8a3390a6). All 18 cells came out at atom_stab
# 0.763 -- which is job 8590174's own epoch-75 number, 0.7667 -- against the
# real model's 0.905, and nothing raised.
#
# The resolution order prefers proj1/checkpoints/diff.pt so that a genuine
# local training run wins. That is right, and it is exactly why the identity
# of what gets loaded must be pinned rather than inferred.
VP_MD5="${VP_MD5:-8a3390a693ed39ebad154b0640145acc}"

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none}  array ${SLURM_ARRAY_TASK_ID:-none}"
echo "stage  : $STAGE   n=$N batch=$BATCH arms=$ARMS"

# n AND batch ARE PINNED, not defaults. v3_sanity.PINNED = {"n": 2000,
# "batch": 500, ...} and the table stage runs v3_sanity BEFORE writing the
# page -- so a run at any other n passes the gate, produces all 18 cells, and
# is then refused after the GPU time is spent. build_results.py:60 also
# divides by a hardcoded 2000 for s/sample. Fail here instead.
if [ "$N" -ne 2000 ] || [ "$BATCH" -ne 500 ]; then
  echo "FATAL: protocol 1.3 fixes n=2000 and batch=500 for vp. v3_sanity.PINNED" >&2
  echo "refuses anything else, so STAGE=table would reject all 18 cells AFTER" >&2
  echo "they are paid for. Got n=$N batch=$BATCH." >&2
  exit 1
fi
if [ $(( N % BATCH )) -ne 0 ]; then
  echo "FATAL: batch $BATCH does not divide n $N." >&2
  exit 1
fi

# A GPU assertion, not a GPU report. transfer_sweep runs --device auto and
# falls back to CPU SILENTLY; this project measured CPU at 26x slower
# (v3_run.slurm:70-75), which turns a ~10 min task into ~260 min -- past
# --max-minutes 225 and past the 4 h wall. Nine of those is the whole
# allocation burned for partial cells, with the gate having said PASS.
if [ "$STAGE" != "table" ]; then
  # Defensive on purpose: is_available() can be True while device 0 is not
  # usable (an empty CUDA_VISIBLE_DEVICES does exactly this), and a traceback
  # here would report a device problem as a crash.
  python - <<'PY'
import sys
try:
    import torch
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e:
    name = None
    print("gpu    : unusable (%s: %s)" % (type(e).__name__, e))
else:
    print("gpu    :", name or "CPU")
sys.exit(0 if name else 1)
PY
  if [ $? -ne 0 ]; then
    echo "FATAL: no usable CUDA device on this node. A vp cell is ~26x slower" >&2
    echo "on CPU and would burn the whole wall for partial cells." >&2
    exit 1
  fi
fi

# --------------------------------------------------------------------------
# STAGE gate -- everything that can fail cheaply, before 9 tasks queue
# --------------------------------------------------------------------------
if [ "$STAGE" = "gate" ]; then
  echo "=== 1. checkpoint identity ==="
  CKPT="$(python -c 'import sys;sys.path.insert(0,"proj1/src");from checkpoint_paths import default_vp_generator as g;print(g())')"
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$CKPT" ]; then
    echo "FATAL: could not resolve the VP checkpoint path (python exited $rc)." >&2
    echo "That is an import failure, not a missing file -- check proj1/src." >&2
    exit 1
  fi
  echo "resolved: $CKPT"
  if [ ! -f "$CKPT" ]; then
    echo "FATAL: no VP checkpoint at $CKPT" >&2
    echo "Resolution order: betty_pull/diff.pt, proj1/checkpoints/diff.pt," >&2
    echo "betty_pull/diff_last.pt, proj1/checkpoints/diff_last.pt," >&2
    echo "weights/diff_ema.pt, weights/vp_ema.pt." >&2
    echo >&2
    echo "weights/ is NOT in the code tarball, and this checkout is NOT a git" >&2
    echo "clone -- Betty is fed by scp (code_vNN.tgz + a separate assets" >&2
    echo "tarball), never by git pull. The model was also trained on the blade" >&2
    echo "box, not here, so nothing would have put it on this filesystem." >&2
    echo >&2
    echo "From the laptop, after pulling origin/main (commit 34da5a0):" >&2
    echo "    ssh betty 'mkdir -p $PROJ/weights'" >&2
    echo "    scp weights/diff_ema.pt betty:$PROJ/weights/" >&2
    echo "Then re-run the gate. Expect md5 8a3390a693ed39ebad154b0640145acc." >&2
    exit 1
  fi
  GOT=$(md5sum "$CKPT" | cut -d' ' -f1)
  echo "md5      : $GOT"
  if [ "$GOT" != "$VP_MD5" ]; then
    echo "FATAL: this is not the pinned VP checkpoint." >&2
    echo "  want $VP_MD5" >&2
    echo "  got  $GOT   ($CKPT)" >&2
    echo >&2
    echo "The field checks below would PASS on the wrong file: the epoch-75" >&2
    echo "remains of the crashed job 8590174 carry the same family, split," >&2
    echo "hidden and layers as the finished model. That is what produced a" >&2
    echo "full set of 18 cells at atom_stab 0.763 instead of 0.905 on 28 Sep." >&2
    echo >&2
    echo "Either move the stale file aside:" >&2
    echo "    mkdir -p proj1/checkpoints/old_partial" >&2
    echo "    mv proj1/checkpoints/diff*.pt proj1/checkpoints/old_partial/" >&2
    echo "or point at the right one: --vp-ckpt weights/diff_ema.pt" >&2
    echo "(override the pin only deliberately, with VP_MD5=...)" >&2
    exit 1
  fi
  # The red fields (family/split/hidden/layers) are refused by the loader in
  # transfer_sweep too. Printing them here means the ONE number a reader needs
  # to tell a vp cell from an edm one -- gen_md5 -- is in the log before any
  # GPU time is spent, and the amber fields (epochs/batch/ema/seed, which
  # nothing checks) are on the record.
  #
  # Deliberately NO epoch assertion: the published diff.pt is the SELECTED
  # checkpoint at epoch 1475 of a 1500-epoch run, and asserting 1500 would
  # reject the real artifact.
  python - "$CKPT" <<'PY'
import sys, torch
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = st.get("args", {})
family = st.get("family") or a.get("family")      # same fallback as the loader
print("family :", family)
print("epoch  :", st.get("epoch"), "(SELECTED; may be < epochs, that is normal)")
for k in ("split", "hidden", "layers", "epochs", "batch", "seed", "ema"):
    print("  %-7s: %r" % (k, a.get(k)))
for k in ("atom_stability", "mol_stability", "val_loss"):
    if k in st:
        print("  %-7s: %r" % (k, st[k]))
bad = [(k, a.get(k), w) for k, w in
       (("split", "train_a"), ("hidden", 256), ("layers", 8))
       if a.get(k) != w]
if family != "vp_diffusion":
    bad.append(("family", family, "vp_diffusion"))
if bad:
    print("FATAL: protocol mismatch (field, got, want):", bad)
    sys.exit(1)
print("checkpoint OK: matches fm on family/split/hidden/layers")
PY
  [ $? -ne 0 ] && exit 1

  echo "=== 2. unit gates ==="
  python proj1/tests/test_v3.py   || exit 1
  python proj1/tests/test_bdg.py  || exit 1

  echo "=== 3. GPU preflight, ALL THREE properties (writes nothing) ==="
  # All three, not just mu: --preflight forces n=8/batch=8/steps=10, so the
  # extra two properties are free, and a wrong pair on alpha or gap would
  # otherwise surface only in the table stage -- after 18 cells are spent.
  python -u "$TS" --stage v3 --backend vp --props mu,alpha,gap --arms "$ARMS" \
      --n "$N" --batch "$BATCH" --seed "${SEEDS[0]}" --preflight \
      2>&1 | tee logs/vp_preflight.txt
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then echo "FATAL: preflight exited $rc" >&2; exit $rc; fi

  echo "=== 4. the delta assertion ==="
  if [ ! -s logs/vp_preflight.txt ]; then
    echo "FATAL: logs/vp_preflight.txt is empty or missing -- the tee failed." >&2
    echo "This is a filesystem problem, not a property-pair problem." >&2
    exit 1
  fi
  if grep -q "delta $DELTA_MU" logs/vp_preflight.txt; then
      echo "delta $DELTA_MU confirmed -- our property pair loaded"
  else
      echo "FATAL: mu delta is not $DELTA_MU. The wrong property pair loaded," >&2
      echo "and every in_band number would be measured against a different band." >&2
      grep -i "delta" logs/vp_preflight.txt >&2
      exit 1
  fi
  echo "GATE PASS -- safe to submit the array"
  exit 0
fi

# --------------------------------------------------------------------------
# STAGE run -- the 18 cells, one (property, seed) per array task
# --------------------------------------------------------------------------
if [ "$STAGE" = "run" ]; then
  t="${SLURM_ARRAY_TASK_ID:-}"
  if [ -z "$t" ]; then
    echo "FATAL: STAGE=run needs an array. Submit --array=0-$(( NP * NS - 1 ))" >&2
    exit 1
  fi
  if [ "$t" -ge $(( NP * NS )) ]; then
    echo "FATAL: array task $t is outside the grid (0-$(( NP * NS - 1 )))." >&2
    echo "The array was submitted wider than PROPS x SEEDS." >&2
    exit 1
  fi
  PROP="${PROPS[$(( t / NS ))]}"
  SEED="${SEEDS[$(( t % NS ))]}"
  echo "=== task $t: vp $PROP seed=$SEED arms=$ARMS ==="
  t0=$(date +%s)
  srun python -u "$TS" \
      --stage v3 \
      --backend vp \
      --props "$PROP" \
      --arms "$ARMS" \
      --n "$N" \
      --batch "$BATCH" \
      --seed "$SEED" \
      --per-mol \
      --max-minutes 225
  rc=$?
  echo "    [timing] vp $PROP seed=$SEED took $(( ($(date +%s) - t0) / 60 )) min"
  exit $rc
fi

# --------------------------------------------------------------------------
# STAGE table -- sanity, then the vp results page
# --------------------------------------------------------------------------
if [ "$STAGE" = "table" ]; then
  echo "=== cell count (expect 6 JSON per seed: 2 arms x 3 props) ==="
  # *.json only: the failure marker is <cell>.json.failed and the sidecar is
  # <cell>.permol.pt, so neither can pad this count. Cells are written tmp +
  # os.replace, so a wall-clock kill cannot leave a partial JSON behind.
  missing=0
  for S in "${SEEDS[@]}"; do
    d="results/v3/vp/v3/n$N/seed$S"
    c=$(ls "$d"/*.json 2>/dev/null | wc -l)
    echo "  $d : $c"
    if [ "$c" -ne 6 ]; then missing=1; fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "REFUSING: the run is incomplete. Re-run the missing array tasks." >&2
    echo "transfer_sweep skips cells that already exist, so resubmitting the" >&2
    echo "whole array is safe and only fills the gaps." >&2
    exit 1
  fi

  echo "=== v3_sanity, scoped to vp ==="
  # Scoped on purpose. Unfiltered it walks all 756 committed cells, which
  # would gate this page on the entire tree staying clean -- a warning in an
  # fm cell from last week is not a reason to withhold the vp page.
  python proj1/scripts/v3_sanity.py --root results/v3 --backend vp --stage v3 || exit 1

  echo "=== the vp results page ==="
  # --backend vp, NOT --both: --both walks only the chain's three backends,
  # and v3_run.slurm's STAGE=table will never produce a vp page because its
  # BACKENDS excludes vp by design. --n is REQUIRED and has no default, and
  # without --md-out the table goes to stdout and no file is written.
  python proj1/scripts/v3_table.py --backend vp --stage v3 --n "$N" \
      --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
      --md-out docs/results/V3_RESULTS_vp.md || exit 1
  echo "wrote docs/results/V3_RESULTS_vp.md"
  echo
  echo "NEXT, and NOT done by this job: paper/tools/build_results.py:67-69"
  echo "hardcodes the \\pend vp row and :34 loads only fm/equifm/edm."
  echo "Edit those plus v3_final_summary.BACKEND_ARMS:81 to fill tab:fmvd."
  exit 0
fi

echo "FATAL: unknown STAGE=$STAGE (want gate, run or table)" >&2
exit 1
