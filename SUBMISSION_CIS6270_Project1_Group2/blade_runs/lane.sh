#!/bin/bash
# blade stand-in for proj1/cluster/v3_run.slurm's run_one: blade has no scheduler.
#
#   setsid nohup bash blade_runs/lane.sh <gpu> <ABSOLUTE queue file> &
#
# Queue lines: <stage> <backend> <prop> <seed> <arms>. Each lane claims the next
# unclaimed line under flock (claims recorded in <queue>.claimed), runs it on its
# own GPU, and repeats until the queue is exhausted. transfer_sweep resumes by
# cell file, so a re-claimed task redoes at most the cell it was killed in.
# Never edit this file while a lane runs it -- bash reads scripts lazily.
GPU="$1"; Q="$2"
PROJ=$PROJECT_ROOT
N="${V3_N:-2000}"; BATCH="${V3_BATCH:-500}"
cd "$PROJ" || exit 1
mkdir -p logs/blade
LANES=logs/blade/lanes.log

claim () {
  exec 9>"$Q.lock"; flock 9
  touch "$Q.claimed"
  local ln
  ln=$(awk -v cl="$Q.claimed" 'BEGIN { while ((getline l < cl) > 0) { split(l, a, " "); c[a[1]] = 1 } }
       !/^#/ && NF && !(FNR in c) { print FNR; exit }' "$Q")
  if [ -n "$ln" ]; then
    echo "$ln g$GPU $(date +%F_%T)" >> "$Q.claimed"
    sed -n "${ln}p" "$Q"
  fi
  flock -u 9; exec 9>&-
}

echo "[g$GPU] $(date +%F_%T) lane start pid $$ queue $Q n=$N batch=$BATCH" >> "$LANES"
while true; do
  LINE="$(claim)"
  [ -z "$LINE" ] && break
  read -r STAGE BE PROP SEED ARMS <<<"$LINE"
  LOG="logs/blade/${STAGE}_${BE}_${PROP}_s${SEED}.log"
  echo "[g$GPU] $(date +%F_%T) start $STAGE $BE $PROP $SEED" >> "$LANES"
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" python -u proj1/scripts/transfer_sweep.py \
      --stage "$STAGE" --backend "$BE" --props "$PROP" --arms "$ARMS" \
      --n "$N" --batch "$BATCH" --seed "$SEED" --per-mol >> "$LOG" 2>&1
  RC=$?
  echo "[g$GPU] $(date +%F_%T) done $STAGE $BE $PROP $SEED rc=$RC min=$(( ($(date +%s) - T0) / 60 ))" >> "$LANES"
done
echo "[g$GPU] $(date +%F_%T) queue exhausted" >> "$LANES"
