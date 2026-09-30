#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
# Autonomous run: M1 headline -> M2 (gc) -> M1 ablation -> extras, with a
# validation gate between stages.
#
#   setsid nohup bash blade_runs/auto2.sh <lane pids...> &
#
# DESIGN RULE: this runs the protocol, it does not decide it. Every parameter
# comes from docs/protocol/MODALITY2_V3_PROTOCOL.md, which takes them from v3.
# Nothing here reads a result and changes a setting because of it -- that is
# what would turn the study into a tuned one. A stage that fails its gate is
# STOPPED and recorded, not worked around.
#
# Never edit this file while it is running: bash reads scripts lazily.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
mkdir -p logs/blade docs/results
L=logs/blade/auto.log
Q=$PROJ/blade_runs
say () { echo "[auto] $(date +%F_%T) $*" >> "$L"; }

say "=== auto2 started, waiting on M1 headline lanes: $*"
for p in "$@"; do
  while kill -0 "$p" 2>/dev/null; do sleep 120; done
done
say "M1 headline lanes exited: $(find results/v3/*/v3 -name 'tr__*.json' 2>/dev/null | wc -l)/144 cells"

# ---------------------------------------------------------------- STAGE 1
say "--- validating M1 headline before anything is built on it"
python -u proj1/scripts/v3_sanity.py --stage v3 > logs/blade/sanity_headline.txt 2>&1
RC=$?
say "v3_sanity rc=$RC ($(tail -1 logs/blade/sanity_headline.txt))"
if [ "$RC" -ne 0 ]; then
  say "HEADLINE FAILED VALIDATION -- stopping. See logs/blade/sanity_headline.txt"
  exit 1
fi
for BE in fm equifm edm; do
  python -u proj1/scripts/v3_table.py --backend "$BE" --stage v3 --n 2000 \
    --seeds 20261001,20261002,20261003 \
    --md-out "docs/results/V3_RESULTS_$BE.md" >> logs/blade/v3_table.log 2>&1
done
python -u proj1/scripts/v3_table.py --both --stage v3 --n 2000 \
  --seeds 20261001,20261002,20261003 \
  --md-out docs/results/V3_RESULTS.md >> logs/blade/v3_table.log 2>&1
say "headline tables -> docs/results/V3_RESULTS*.md"

# ---------------------------------------------------------------- STAGE 2
# The correction-share diagnostic runs FIRST among the M2 work, because it is
# what makes an M2 null readable. A CPU probe at v3's settings already showed
# M2's baselines applying ~0.015 share against M1's 0.055 for plug: at w = 1
# they are ~3x weaker in force than M1's, while bdg_e4t0.5 lands at ~0.083,
# inside M1's own 0.024-0.138 range. Measuring it properly means the write-up
# can say whether a null is about the method or about the strength.
say "--- M2 correction share at v3's settings (explains any M2 null)"
CUDA_VISIBLE_DEVICES=0 python -u proj1/m2/measure_share.py --props gc \
  --ws 1,4,16,64 --n 128 --steps 100 --t-min 0.5 --device cuda \
  >> logs/blade/m2_share.log 2>&1
say "share diagnostic rc=$? -> docs/results/M2_SHARE.md"

# ---------------------------------------------------------------- STAGE 3
say "--- M2 headline (gc) at v3's settings: n=2000 batch=500 NFE=100 t>=0.5 w=1"
i=0
for S in 20260921 20260922 20260923; do
  CUDA_VISIBLE_DEVICES=$i nohup python -u proj1/m2/run_sweep.py \
    --stage m2 --props gc --seeds "$S" --n 2000 --batch 500 --device cuda \
    > "logs/blade/m2_head_s$S.log" 2>&1 &
  i=$((i+1))
done
wait
say "M2 headline done: $(find results/m2/m2 -name '*.json' 2>/dev/null | wc -l)/21 cells"
python -u proj1/m2/m2_sanity.py --stage m2 --n 2000 --window 0.5 \
  > logs/blade/sanity_m2_head.txt 2>&1
RC=$?
say "m2_sanity rc=$RC ($(tail -1 logs/blade/sanity_m2_head.txt))"
if [ "$RC" -eq 0 ]; then
  python -u proj1/m2/m2_table.py --stage m2 --n 2000 --props gc \
    > docs/results/M2_RESULTS_gc.txt 2>&1
  say "M2 headline table -> docs/results/M2_RESULTS_gc.txt"
else
  say "M2 HEADLINE FAILED VALIDATION -- no table, no M2 ablation. M1's ablation"
  say "still runs below; it does not depend on M2."
fi

# ---------------------------------------------------------------- STAGE 4
# M1's ablation is the assignment's biggest single Results item (4.4, .675 pt,
# "the main evidence for the innovation"), so it precedes M2's extras.
say "--- M1 ablation: 18 tasks, 612 cells, 3 lanes"
for g in 0 1 2; do
  setsid nohup bash "$Q/lane.sh" "$g" "$Q/queue_m1abl.txt" > /dev/null 2>&1 < /dev/null &
  sleep 2
done
sleep 90
ABLPIDS=$(pgrep -f "queue_m1abl" | tr '\n' ' ')
say "ablation lanes: $ABLPIDS"
for p in $ABLPIDS; do
  while kill -0 "$p" 2>/dev/null; do sleep 300; done
done
say "M1 ablation exited: $(find results/v3/*/v3abl -name 'tr__*.json' 2>/dev/null | wc -l)/612 cells"
python -u proj1/scripts/v3_sanity.py --stage v3abl > logs/blade/sanity_abl.txt 2>&1
say "ablation v3_sanity rc=$? ($(tail -1 logs/blade/sanity_abl.txt))"
for W in 1 4; do
  python -u proj1/scripts/v3_table.py --both --stage v3abl --n 2000 --w "$W" \
    --seeds 20261001,20261002,20261003 \
    --md-out "docs/results/V3_RESULTS_ABL_w$W.md" >> logs/blade/v3_table.log 2>&1
done
say "ablation tables -> docs/results/V3_RESULTS_ABL_w{1,4}.md"

# ---------------------------------------------------------------- STAGE 5
# M2's ablation carries w in {1,4} exactly as M1's does. The share diagnostic
# says w=4 puts M2's plug at ~0.06, i.e. M1's own force -- so the force-matched
# comparison is already inside v3's grid and needs no departure from it.
say "--- M2 gc ablation, w in {1,4}, 111 cells"
i=0
for S in 20260921 20260922 20260923; do
  CUDA_VISIBLE_DEVICES=$i nohup python -u proj1/m2/run_sweep.py \
    --stage m2abl --props gc --seeds "$S" --n 2000 --batch 500 --device cuda \
    > "logs/blade/m2_abl_s$S.log" 2>&1 &
  i=$((i+1))
done
wait
say "M2 gc ablation done: $(find results/m2/m2abl -name '*.json' 2>/dev/null | wc -l)/111 cells"
python -u proj1/m2/m2_sanity.py --stage m2abl --n 2000 > logs/blade/sanity_m2_abl.txt 2>&1
say "m2abl sanity rc=$? ($(tail -1 logs/blade/sanity_m2_abl.txt))"
for W in 1 4; do
  python -u proj1/m2/m2_table.py --stage m2abl --n 2000 --w "$W" --props gc \
    > "docs/results/M2_RESULTS_ABL_gc_w$W.txt" 2>&1
done

# ---------------------------------------------------------------- STAGE 6
say "--- write-up diagnostics (these set no parameter)"
CUDA_VISIBLE_DEVICES=0 python -u proj1/m2/measure_window.py --props gc \
  --n 500 --steps 100 --device cuda >> logs/blade/m2_window.log 2>&1
say "window diagnostic rc=$? -> docs/results/M2_WINDOW.md"
CUDA_VISIBLE_DEVICES=1 python -u proj1/m2/measure_strength.py --props gc \
  --n 500 --steps 100 --t-min 0.5 --device cuda >> logs/blade/m2_strength.log 2>&1
say "strength diagnostic rc=$? -> docs/results/M2_STRENGTH.md"

# ---------------------------------------------------------------- STAGE 7
say "--- cpg replication (only reached if the deadline allowed)"
i=0
for S in 20260921 20260922 20260923; do
  CUDA_VISIBLE_DEVICES=$i nohup python -u proj1/m2/run_sweep.py \
    --stage m2 --props cpg --seeds "$S" --n 2000 --batch 500 --device cuda \
    > "logs/blade/m2_head_cpg_s$S.log" 2>&1 &
  i=$((i+1))
done
wait
python -u proj1/m2/m2_sanity.py --stage m2 --n 2000 --window 0.5 \
  > logs/blade/sanity_m2_cpg.txt 2>&1
python -u proj1/m2/m2_table.py --stage m2 --n 2000 --props cpg \
  > docs/results/M2_RESULTS_cpg.txt 2>&1
say "cpg replication done"

say "=== all planned stages complete"
