#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
# Fires when the M1 headline lanes exit. Runs the two M2 measurements that the
# protocol requires BEFORE any sweep cell -- it does NOT start the sweep, because
# choosing the window and the strength from those tables is a judgement call and
# the protocol says so. Never edit this file while it is running.
PROJ=$PROJECT_ROOT
cd "$PROJ" || exit 1
mkdir -p logs/blade
L=logs/blade/m2_chain.log
say () { echo "[m2] $(date +%F_%T) $*" >> "$L"; }

for p in "$@"; do
  while kill -0 "$p" 2>/dev/null; do sleep 120; done
done
say "M1 headline lanes exited; headline cells = $(find results/v3/*/v3 -name 'tr__*.json' 2>/dev/null | wc -l)/144"

# 1. What does ONE M2 cell cost? The protocol refuses to carry a budget it has
#    not measured, and nothing here has ever timed an M2 guidance cell.
say "timing one gc/plug cell at the protocol's n and NFE"
/usr/bin/time -f '%e s' -o logs/blade/m2_cell_time.txt \
  env CUDA_VISIBLE_DEVICES=0 python -u proj1/m2/m2_sweep.py \
  --prop gc --arm plug --w 16 --n 2000 --batch 500 --steps 400 --t-min 0.0 \
  --device cuda --stage m2timing --out-dir results/_m2timing \
  >> logs/blade/m2_timing.log 2>&1
say "one cell took $(cat logs/blade/m2_cell_time.txt 2>/dev/null | tr -d '\n')"

# 2. The window (protocol 2.1 + 1.2c): gap closure AND clip saturation together.
#    This is the measurement that decides t_min, so it runs before anything else.
say "window sweep on gc, t_min in {0,0.05,0.1,0.2,0.3,0.5}"
CUDA_VISIBLE_DEVICES=0 python -u proj1/m2/measure_window.py \
  --props gc --n 500 --batch 500 --steps 400 --device cuda \
  >> logs/blade/m2_window.log 2>&1
say "window sweep rc=$? -> docs/results/M2_WINDOW.md"

say "STOPPING HERE. The strength measurement runs at the CHOSEN window, so it"
say "waits for the window to be read off M2_WINDOW.md. Nothing is swept yet."
