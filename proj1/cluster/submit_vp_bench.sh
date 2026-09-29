#!/bin/bash
# The VP bench suite, in one paste.
#
#   bash proj1/cluster/submit_vp_bench.sh
#
# Prints the job ids and exits.
#
# THE CHAIN
#   A  gate    identity + unit gates + GPU preflight + the delta assertion
#   B  run     array 0-8, one (property, seed) each -> the 18 cells
#   C  table   v3_sanity + docs/results/V3_RESULTS_vp.md
#
# A -> B is `afterok`: if the checkpoint is wrong or the wrong property pair
# loads, nothing downstream should burn a slot.
#
# C depends on BOTH: `afterok:$A,afterany:$B` (a comma is AND in SLURM). The
# afterany alone is not enough -- a job cancelled for a never-satisfiable
# dependency has TERMINATED, which satisfies afterany, so a failed gate would
# still launch the table job and it would go red with "run is incomplete",
# hiding the real cause. Chaining afterok:$A as well means a gate failure
# kills C outright. The afterany on B is still right: a task that runs out of
# its 4-hour window exits nonzero with its finished cells on disk, and the
# table must still run -- it REFUSES on an incomplete run and names the seed,
# which is the report you want.
#
# THIS SUITE IS vp-ONLY BY CONSTRUCTION. transfer_sweep builds its output path
# from the backend name, so --backend vp can only write results/v3/vp/. It is
# deliberately NOT submit_v3.sh: that chain re-plans fm, equifm and edm, whose
# 900 cells are already on disk.
set -uo pipefail
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}"

PROJ="${CGM_PROJ:-/vast/projects/ajw/wharton/hyhuang/cgm}"
JOB=proj1/cluster/vp_bench.slurm
N="${VP_N:-2000}"
BATCH="${VP_BATCH:-500}"

cd "$PROJ" || { echo "FATAL: no $PROJ" >&2; exit 1; }
[ -f "$JOB" ] || { echo "FATAL: no $JOB -- did the code tarball land?" >&2; exit 1; }

# BEFORE sbatch, not in the job. #SBATCH --output=logs/vp-%j.out is opened by
# slurmd before the body runs, so the job's own mkdir is too late: on a
# checkout without logs/ every task fails at launch with the log being what
# broke. The code tarball does not carry logs/.
mkdir -p "$PROJ/logs" || { echo "FATAL: cannot create $PROJ/logs" >&2; exit 1; }

# Derive the array range from the job file rather than hardcoding it -- the
# stride bug this project's gates exist for came from a hand-maintained
# constant. `sed` without a backreference and `wc -w` on purpose: submit_v3.sh
# notes that an earlier version used \1 "and the tool that wrote this file ate
# the backslash", and wc -w is whitespace-agnostic where `tr ' ' '\n'` would
# read a tab-separated array as a single element and submit a SHORT array.
NPROPS=$(sed -n 's/^PROPS=(//p' "$JOB" | tr -d ')' | wc -w)
NSEEDS=$(sed -n 's/^SEEDS=(//p' "$JOB" | tr -d ')' | wc -w)
MAX=$(( NPROPS * NSEEDS - 1 ))
if [ "$NPROPS" -ne 3 ] || [ "$NSEEDS" -ne 3 ]; then
  echo "FATAL: expected 3 properties x 3 seeds in $JOB, read $NPROPS x $NSEEDS." >&2
  echo "Protocol 1.3 fixes both. A short array would produce a partial run" >&2
  echo "that only fails at the table stage." >&2
  exit 1
fi
echo "grid: $NPROPS properties x $NSEEDS seeds -> array 0-$MAX ($(( MAX + 1 )) tasks, $(( (MAX + 1) * 2 )) cells)"
echo "n=$N batch=$BATCH"

EXP="ALL,VP_N=$N,VP_BATCH=$BATCH"

# Every sbatch is checked. Unchecked, a failed submission leaves the id empty,
# the next --dependency=afterok: is rejected as invalid, and the operator gets
# three sbatch errors interleaved with success-looking output.
sub () {   # $1 job-name, $2.. extra flags
  local name="$1"; shift
  local id
  id=$(sbatch --parsable "$@" "$JOB") \
    || { echo "FATAL: sbatch failed for $name" >&2; exit 1; }
  [ -n "$id" ] || { echo "FATAL: sbatch returned no job id for $name" >&2; exit 1; }
  echo "$id"
}

# Short walls for the two one-shot stages: the array needs 4 h, the gate runs
# ~15 min and the table ~1 min, and a 4 h reservation on each hurts backfill.
A=$(sub gate  --export="$EXP,STAGE=gate"  --job-name=vp-gate  --time=00:30:00)
echo "A gate  : $A"
B=$(sub run   --export="$EXP,STAGE=run"   --job-name=vp-run   --array=0-$MAX \
              --dependency=afterok:$A --kill-on-invalid-dep=yes)
echo "B run   : $B  (array 0-$MAX)"
C=$(sub table --export="$EXP,STAGE=table" --job-name=vp-table --time=00:20:00 \
              --dependency=afterok:$A,afterany:$B --kill-on-invalid-dep=yes)
echo "C table : $C"
echo
echo "watch:   squeue -u \$USER -o \"%.18i %.14j %.9T %.10M %.28E\""
echo "timing:  grep -h '\[timing\]' logs/vp-*.out"
echo "cells:   ls results/v3/vp/v3/n$N/seed*/*.json | wc -l    # want 18"
echo "page:    docs/results/V3_RESULTS_vp.md"
