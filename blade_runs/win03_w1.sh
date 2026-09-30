#!/bin/bash
# The t >= 0.3 cells Henry needs for tab:m2 and tab:full-m2.
#
# Batches of 4 ON PURPOSE: each guided cell holds ~5.7 GiB and the card is
# 48 GiB, so 14 at once OOMs -- which is exactly how ten cells were lost on
# 28 Sep. Do not raise this without re-measuring.
#
# Everything else matches the existing win0.3 probe cells and stage m2:
#   n 2000, batch 500, steps 100 (NFE), target q50, clip 1.0, delta_ratio 0.16
set -u
PROJ=/scratch2/boboli/projects/cis6270-project1-group2
cd "$PROJ" || exit 1
GPU=${GPU:-3}
L=logs/blade/win03_g3.log
cell () {   # arm variant w prop seed
  CUDA_VISIBLE_DEVICES=$GPU python -u proj1/m2/m2_sweep.py \
    --prop "$4" --arm "$1" --variant "$2" --w "$3" \
    --n 2000 --batch 500 --steps 100 --t-min 0.3 --seed "$5" \
    --target q50 --clip 1.0 --delta-ratio 0.16 \
    --device cuda --stage m2win --out-dir results/m2 >> "$L" 2>&1
}
ARMS_A=("plug -" "tmpd -" "lgd_mc -" "tfg_mc -")
ARMS_B=("bdg e4t0.5" "bdg e4t1" "bdg e0t1" "unguided -")
run_set () {  # w prop
  local w=$1 prop=$2 n=0
  for spec in "${ARMS_A[@]}" "${ARMS_B[@]}"; do
    set -- $spec
    for s in 20260921 20260922 20260923; do
      cell "$1" "$2" "$w" "$prop" "$s" &
      n=$((n+1))
      if [ $((n % 4)) -eq 0 ]; then wait; fi
    done
  done
  wait
  echo "[win03] $(date +%T) done w=$w prop=$prop -> $(find results/m2/m2win -name "*win0.3*.json" | wc -l) cells total" >> "$L"
}
echo "[win03] $(date +%F_%T) start" >> "$L"


run_set 1 gc
run_set 1 cpg
echo "[win03] $(date +%F_%T) ALL DONE: $(find results/m2/m2win -name '*win0.3*.json' | wc -l) cells" >> "$L"
