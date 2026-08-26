#!/usr/bin/env bash
# One card's share of a queue stage, on a shared machine.
#
#   exp/run_task.sh <stage> <gpu> <shard> <total> <results-dir> <keep:0|1>
#
# Waits for free VRAM before each run instead of assuming it. This box has
# other people's jobs on it; the queue lines up behind them and never touches
# their processes. Gates on artifacts on disk, never on a PID.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
STAGE="$1"; GPU="$2"; SHARD="$3"; TOTAL="$4"; OUT="$5"; KEEP="$6"
LOG="exp/results/queue_${STAGE}_gpu${GPU}.log"
NEED_MB=3500
[ "$OUT" = "exp/results" ] && ARGS=() || ARGS=(--results "$OUT")
[ "$KEEP" = "1" ] && ARGS+=(--keep-epochs)

mapfile -t RUNS < <("$PY" exp/gpu_queue.py "$STAGE" --shard "$SHARD/$TOTAL")
echo "=== $(date '+%F %T') $STAGE gpu=$GPU ${#RUNS[@]} runs -> $OUT ===" \
    | tee -a "$LOG"

for RUN in "${RUNS[@]}"; do
    [ -f "$OUT/$RUN/final.pt" ] && { echo "--- $RUN done" >> "$LOG"; continue; }
    while true; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
        [ "$FREE" -ge "$NEED_MB" ] && break
        echo "$(date '+%T') gpu$GPU ${FREE}MB free, need $NEED_MB; waiting" >> "$LOG"
        sleep 300
    done
    echo "--- $(date '+%F %T') $RUN ---" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" exp/train.py "${ARGS[@]}" "$RUN" 2>&1 \
        | tee -a "$LOG" || echo "!!! $RUN FAILED, continuing" | tee -a "$LOG"
done
echo "=== $(date '+%F %T') $STAGE gpu=$GPU done ===" | tee -a "$LOG"
