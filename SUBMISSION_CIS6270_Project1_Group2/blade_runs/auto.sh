#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
# Autonomous run: M1 headline -> M2 (gc) -> M1 ablation -> extras, with a
# validation gate between every stage. Launched once and left alone.
#
#   setsid nohup bash blade_runs/auto.sh <lane pids...> &
#
# DESIGN RULE: this script runs the protocol, it does not decide it. Every
# parameter comes from docs/protocol/MODALITY2_V3_PROTOCOL.md, which takes them
# from v3. Nothing here reads a result and changes a setting because of it --
# that is what would turn the study into a tuned one. Where a stage fails its
# validation gate, the script STOPS that stage and records why, rather than
# continuing and producing a complete but meaningless table.
#
# Never edit this file while it is running: bash reads scripts lazily.
set -uo pipefail
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
mkdir -p logs/blade docs/results
L=logs/blade/auto.log
Q=$PROJ/blade_runs
say () { echo "[auto] $(date +%F_%T) $*" >> "$L"; }
gpu_free () {  # all of GPUs 0,1,2 idle of OUR jobs
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$n" -le 3 ]
}

say "=== started, waiting on M1 headline lanes: $*"
for p in "$@"; do
  while kill -0 "$p" 2>/dev/null; do sleep 120; done
done
say "M1 headline lanes exited: $(find results/v3/*/v3 -name 'tr__*.json' 2>/dev/null | wc -l)/144 cells"

# ---------------------------------------------------------------- STAGE 1
# Validate the headline before anything is built on it.
say "--- validating M1 headline"
python -u proj1/scripts/v3_sanity.py --stage v3 > logs/blade/sanity_headline.txt 2>&1
RC=$?
say "v3_sanity rc=$RC ($(tail -1 logs/blade/sanity_headline.txt))"
if [ "$RC" -ne 0 ]; then
  say "HEADLINE FAILED VALIDATION -- stopping. See logs/blade/sanity_headline.txt"
  exit 1
fi

say "--- building M1 headline tables"
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
# M2, gc only, on v3's settings. One seed per GPU; cells are independent and
# skipped if already present, so the three workers never collide.
say "--- M2 headline (gc), n=2000 batch=500 NFE=100 t_min=0.5 w=1"
i=0
for S in 20260921 20260922 20260923; do
  CUDA_VISIBLE_DEVICES=$i nohup python -u proj1/m2/run_sweep.py \
    --stage m2 --props gc --seeds "$S" --n 2000 --batch 500 --device cuda \
    > "logs/blade/m2_head_s$S.log" 2>&1 &
  echo $! >> logs/blade/m2_pids.txt
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
  say "M2 HEADLINE FAILED VALIDATION -- not building its table, not starting its"
  say "ablation. M1's ablation still runs below; it does not depend on M2."
fi

# ---------------------------------------------------------------- STAGE 3
# M1's ablation is the assignment's single biggest Results item (4.4, .675 pt,
# 'the main evidence for the innovation'), so it runs before M2's extras.
say "--- M1 ablation: 18 tasks, 612 cells, 3 lanes"
for g in 0 1 2; do
  setsid nohup bash "$Q/lane.sh" "$g" "$Q/queue_m1abl.txt" > /dev/null 2>&1 < /dev/null &
  sleep 2
done
sleep 60
ABLPIDS=$(pgrep -f "lane.sh . $Q/queue_m1abl.txt" | tr '\n' ' ')
say "ablation lanes: $ABLPIDS"
for p in $ABLPIDS; do
  while kill -0 "$p" 2>/dev/null; do sleep 300; done
done
say "M1 ablation lanes exited: $(find results/v3/*/v3abl -name 'tr__*.json' 2>/dev/null | wc -l)/612 cells"

python -u proj1/scripts/v3_sanity.py --stage v3abl > logs/blade/sanity_abl.txt 2>&1
say "ablation v3_sanity rc=$? ($(tail -1 logs/blade/sanity_abl.txt))"
for W in 1 4; do
  python -u proj1/scripts/v3_table.py --both --stage v3abl --n 2000 --w "$W" \
    --seeds 20261001,20261002,20261003 \
    --md-out "docs/results/V3_RESULTS_ABL_w$W.md" >> logs/blade/v3_table.log 2>&1
done
say "ablation tables -> docs/results/V3_RESULTS_ABL_w{1,4}.md"

# ---------------------------------------------------------------- STAGE 4
# Extras, in the order the rubric values them. Each is independently useful, so
# whichever the deadline reaches is what gets reported.
say "--- extras: M2 gc ablation, then the write-up diagnostics, then cpg"
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

say "--- diagnostics for the write-up (these set nothing; protocol 2.1 and 3)"
CUDA_VISIBLE_DEVICES=0 python -u proj1/m2/measure_window.py --props gc \
  --device cuda >> logs/blade/m2_window.log 2>&1
say "window diagnostic rc=$? -> docs/results/M2_WINDOW.md"
CUDA_VISIBLE_DEVICES=1 python -u proj1/m2/measure_strength.py --props gc \
  --t-min 0.5 --device cuda >> logs/blade/m2_strength.log 2>&1
say "strength diagnostic rc=$? -> docs/results/M2_STRENGTH.md"

say "=== all planned stages complete"
