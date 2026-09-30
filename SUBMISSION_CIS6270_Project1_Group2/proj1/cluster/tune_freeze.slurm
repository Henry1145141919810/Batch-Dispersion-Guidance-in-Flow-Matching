#!/bin/bash
#SBATCH --job-name=cgm-tune-freeze
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=$PROJECT_ROOT/logs/tunefreeze-%j.out
#SBATCH --error=$PROJECT_ROOT/logs/tunefreeze-%j.err
#
# THE GATE between the two stages. Reads the finished tune sweep, picks the two
# operating points per (property, arm), and writes frozen_tune.json plus the
# sweep table.
#
# IT IS A SEPARATE JOB ON PURPOSE. The full run is chained to it with
# `afterok`, so if the freeze REFUSES -- missing cells, a missing unguided
# reference, or cells sampled under two different deltas -- the full run never
# starts and no GPU time is spent on a table that could not be interpreted.
# Putting the freeze inside the full-run job (as v2_run.slurm does) would have
# each of nine array tasks discover the same failure independently.
#
# It asks for a GPU only because the cluster's CPU-only partition is not
# guaranteed here; the script itself is seconds of CPU and loads no model.
#
# SUBMIT
#   sbatch --dependency=afterany:<SWEEP_INSURANCE_ID> --export=ALL \
#          proj1/cluster/tune_freeze.slurm
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

N="${TUNE_N:-1000}"
OUT="$PROJ/results/tune/n${N}"
FROZEN="$OUT/frozen_tune.json"

echo "host   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:?not inside a SLURM job}"
echo "root   : $OUT"
echo "cells  : $(find "$OUT" -maxdepth 1 -name '*__tune.json' 2>/dev/null | wc -l) / 489"

python proj1/scripts/freeze_tune.py \
  --root "$OUT" \
  --json-out "$FROZEN" \
  --md-out docs/results/TUNE_SWEEP_TABLE.md
RC=$?

if [ $RC -ne 0 ]; then
  echo "" >&2
  echo "FREEZE REFUSED (rc=$RC). The chained full run will NOT start." >&2
  echo "Fix what it named, then re-run the sweep links and this job:" >&2
  echo "  sbatch --export=ALL proj1/cluster/tune_sweep.slurm     # insurance, no --array" >&2
  echo "  sbatch --dependency=afterany:<that id> --export=ALL proj1/cluster/tune_freeze.slurm" >&2
  echo "" >&2
  echo "To freeze from a deliberately partial sweep instead, add --allow-partial" >&2
  echo "to the command above and re-run it by hand. The result is labelled" >&2
  echo "partial and records what was missing." >&2
  exit $RC
fi

echo ""
echo "frozen: $FROZEN"
python - "$FROZEN" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
n = d["n_full_run_cells_per_seed"]          # all properties, one seed
props = sorted({c["prop"] for c in d["full_run_cells"]})
print("full run will sample %d cells per seed across %d properties "
      "(after de-duplication)" % (n, len(props)))
for pr in props:
    k = sum(1 for c in d["full_run_cells"] if c["prop"] == pr)
    print("   %-6s %d cells per seed  -> %d over 3 seeds" % (pr, k, k * 3))
print("total %d cells over 3 seeds" % (n * 3))
for k, v in d["flags"].items():
    if v:
        print("FLAG %-24s %d: %s" % (k, len(v), "; ".join(v[:4])))
PY
exit 0
