#!/bin/bash
#SBATCH --job-name=cgm-xproj
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/ajw/wharton/hyhuang/cgm/logs/xproj-%j.out
#SBATCH --error=/vast/projects/ajw/wharton/hyhuang/cgm/logs/xproj-%j.err
#
# THE GEOMETRY-REPAIR RE-TEST. 12 cells: lgd_mc and btvg2_xproj at w = 8 and
# w = 16, on mu/alpha/gap, on the `dist` target.
#
# WHAT IT DECIDES, AND WHAT IT DOES NOT.
# The status doc claims "the variance term adds nothing on top of lgd_mc",
# supported by `btvg2 - lgd_mc`. That difference isolates the IMPLEMENTED
# correction, not an ideal mean-preserving variance intervention: btvg2
# imposes orthogonality and non-widening in m-space and then
# applies J^T, and neither property survives (docs/methods/
# BTVG_FAILURE_AUDIT_AND_REDESIGN.md section 3; measured leak 3.59 against the
# repaired arm's 3.9e-7 in proj1/tests/test_btvg2_xproj.py). So the claim is
# currently confounded by an implementation defect. This run re-tests it with
# the invariants imposed in x_t, the space the sampler updates.
#
# It is HYGIENE, not a bid for a result. Per docs/methods/
# BTVG_EVOLUTION_AFTER_CHEM_GUARD.md section 7, any retained variance claim
# must beat lgd_mc PLUS ordinary endpoint repair at matched cost, which this
# run does not test. A null here closes the confound; a positive here would
# only license that harder comparison.
#
# PRE-REGISTERED READING, fixed before submission: the repaired variance term
# is worth pursuing only if it gains >= +1 pp decoded in_band over lgd_mc on
# at least 2 of the 3 properties at a matched strength, with the two strengths
# agreeing in sign.
#
# THE BASELINE IT HAS TO BEAT, recomputed from the pilot cells themselves
# (results/pilot_btvg2/seed20261001_n2048, soft-feature in_band, btvg2 minus
# lgd_mc, mu / alpha / gap):
#     w = 4   -0.0010 / -0.0107 / +0.0015
#     w = 8   -0.0117 / -0.0137 / -0.0112
# So the UNREPAIRED arm did not merely add nothing at w = 8, it LOST about
# 1.1-1.4 pp of coverage on all three properties. There are no btvg2 cells at
# w = 16 anywhere in the repo, so w = 16 has no unrepaired baseline and the
# lgd_mc cells in this job supply its only comparison.
# (An earlier draft of this header quoted +0.0005 / +0.0049 / -0.0054 and
# -0.001 / -0.003 / +0.001 here. Those were `lgd_mc_chemn` minus lgd_mc USEFUL
# YIELD figures from the chemistry-guard experiment, a different arm and a
# different metric, copied in by mistake and caught in review.)
#
# BOTH ARMS RUN HERE, on purpose. Local lgd_mc cells exist at these settings,
# but the 5080 reproduces Betty only to a median |df_B| of 5e-6 with about 1 %
# of molecules diverging chaotically. Re-running lgd_mc on the same machine
# keeps every comparison exactly paired; the local cells then serve as an
# independent reproducibility check.
#
# SETTINGS ARE THE PILOT'S, so the new cells are comparable to it: n = 2048,
# dist targets from test[:2048] (--dist-offset 0), seed 20261001, t_min 0.5,
# 100 Euler steps, batch 128, n_probe 1. EXPLORATORY: those targets are
# already consumed by the full run and the earlier pilots, so nothing here is
# confirmation.
#
# COST. lgd_mc is MEASURED on Betty (20.5 min per n = 5000 cell on a 1g.45gb
# slice) and rescaled linearly to ~8.4 min at n = 2048. btvg2_xproj has never
# run on Betty: ~12 min/cell is an ESTIMATE from its two extra VJPs per step,
# not a measurement. Twelve cells therefore ~2 h expected, inside the 200-min
# budget coded below, but treat the total as provisional until the first log
# reports its own per-cell seconds. The job is resumable, so an underestimate
# costs a resubmission, not the run.
set -uo pipefail
export PATH="/cm/local/apps/slurm/current/bin:$PATH"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
cd "$PROJ" || exit 1
mkdir -p "$PROJ/logs"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate" || exit 1

N=2048
SEED=20261001
FM=proj1/checkpoints/fm_last.pt          # the full run's generator
ARMS="lgd_mc,btvg2_xproj"
OUT="$PROJ/results/xproj/seed${SEED}_n${N}"
BUDGET_MIN=200
T_START=$(date +%s)

mkdir -p "$OUT"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:?not inside a SLURM job - never run this on a login node}"
echo "n         : $N   seed: $SEED   arms: $ARMS"
echo "out       : $OUT"
echo "started   : $(date -Is)"
echo

RC=0
for W in 16 8; do
    USED=$(( ($(date +%s) - T_START) / 60 ))
    LEFT=$(( BUDGET_MIN - USED ))
    if [ "$LEFT" -lt 25 ]; then
        echo "time budget spent ($USED min) -- resubmit to continue; cells already"
        echo "written are skipped automatically"
        RC=1
        break
    fi
    echo "=== strength w=$W  (${LEFT} min left)"
    srun python -u proj1/scripts/guidance_sweep.py \
        --stage targets --targets dist --arms "$ARMS" --props mu,alpha,gap \
        --strengths "$W" --only-w "$W" \
        --n "$N" --batch 128 --steps 100 --n-probe 1 --dist-offset 0 \
        --seed "$SEED" --fm "$FM" --out-dir "$OUT" \
        --per-mol --save-coords --max-minutes "$LEFT" || RC=1
done

# guidance_sweep.py exits 0 whenever no cell RAISED, including when its own
# --max-minutes guard stopped it with cells still unwritten. So completeness is
# checked here explicitly; without this the job could exit 0 with 8 of 12 cells
# and nothing would say so.
DONE=$(ls "$OUT"/*.json 2>/dev/null | wc -l)
FAILED=$(ls "$OUT"/*.failed 2>/dev/null | wc -l)
echo
echo "cells present: $DONE of 12"
echo "failed cells : $FAILED"
if [ "$DONE" -lt 12 ]; then
    echo "INCOMPLETE -- resubmit this same script; finished cells are skipped."
    RC=1
fi
if [ "$FAILED" -gt 0 ]; then
    echo "SOME CELLS RAISED -- read the .failed sidecars before resubmitting."
    RC=1
fi
if [ "$RC" -eq 0 ]; then
    echo "COMPLETE -- all 12 cells written."
fi
echo "finished  : $(date -Is)  rc=$RC"
exit $RC
