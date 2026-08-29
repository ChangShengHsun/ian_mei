#!/usr/bin/env bash
# One card's share of a queue stage under the held-out protocol.
#
#   exp/run_task_heldout.sh <stage> <gpu> <shard> <total> <results-dir>
#
# A separate file from run_task.sh on purpose. run_task.sh was executing when
# this protocol was added, and bash reads a script by byte offset as it goes:
# editing a running script makes it resume at the wrong place. New behaviour
# for a queue that has not started yet goes in a new file.
#
# Always --keep-epochs: the selection rules are recomputed from log.csv's dev
# column afterwards, and a rule that picks epoch 60 needs epoch060.pt to exist.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
STAGE="$1"; GPU="$2"; SHARD="$3"; TOTAL="$4"; OUT="$5"
LOG="exp/results/queue_${STAGE}_gpu${GPU}.log"
NEED_MB=3500

mkdir -p "$OUT"
mapfile -t RUNS < <("$PY" exp/gpu_queue.py "$STAGE" --shard "$SHARD/$TOTAL")
echo "=== $(date '+%F %T') $STAGE gpu=$GPU ${#RUNS[@]} runs -> $OUT (heldout) ===" \
    | tee -a "$LOG"

for RUN in "${RUNS[@]}"; do
    [ -f "$OUT/$RUN/final.pt" ] && { echo "--- $RUN done" >> "$LOG"; continue; }
    # Waits for VRAM rather than assuming it. Other people's jobs share this
    # box; the queue lines up behind them and never touches their processes.
    while true; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
        [ "$FREE" -ge "$NEED_MB" ] && break
        echo "$(date '+%T') gpu$GPU ${FREE}MB free, need $NEED_MB; waiting" >> "$LOG"
        sleep 300
    done
    echo "--- $(date '+%F %T') $RUN ---" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" exp/train.py \
        --results "$OUT" --keep-epochs --protocol heldout "$RUN" 2>&1 \
        | tee -a "$LOG" || echo "!!! $RUN FAILED, continuing" | tee -a "$LOG"
done
echo "=== $(date '+%F %T') $STAGE gpu=$GPU done ===" | tee -a "$LOG"
