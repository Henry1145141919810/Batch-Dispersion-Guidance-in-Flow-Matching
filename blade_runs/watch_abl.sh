#!/bin/bash
# Validate the ablation as it lands. Silent while healthy; exits on the first
# real problem or on completion, so being woken means something happened.
PROJ=/scratch2/boboli/projects/cis6270-project1-group2
cd "$PROJ" || exit 1
while true; do
  N=$(find results/v3/*/v3abl -name 'tr__*.json' 2>/dev/null | wc -l)
  # "$N" -gt 0 first: v3_sanity.py now exits nonzero on a tree with no cells,
  # which is right for a gate but wrong for a poll loop -- started before the
  # first cell lands, this would report SANITY FAILED at 0/612 and give up
  # instead of waiting.
  if [ "$N" -gt 0 ] && ! python proj1/scripts/v3_sanity.py --quiet > logs/blade/sanity_abl_live.txt 2>&1; then
    echo "SANITY FAILED at $N/612 cells, $(date +%T)"
    grep FAIL logs/blade/sanity_abl_live.txt | head -8
    exit 1
  fi
  if [ "$N" -ge 612 ]; then
    echo "ABLATION COMPLETE: $N/612 at $(date +%T), sanity clean"
    tail -1 logs/blade/sanity_abl_live.txt
    exit 0
  fi
  sleep 1800
done
