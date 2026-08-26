#!/usr/bin/env bash
# Task A1: retrain the four 117k arms keeping every validated epoch.
#
#   exp/run_sweep.sh <gpu> <shard> <total>
#
# Writes to exp/results/selection_sweep/, NOT to exp/results/. Retraining into
# the published directories would replace the final.pt and best.pt that
# stratify.csv and erl.csv were computed from an hour ago, and those files
# would stop being reproducible from what is on disk.
#
# Waits for free VRAM instead of assuming it: this is a shared lab machine and
# another user's job sits on the cards without warning. It never touches their
# processes -- it queues behind them.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
GPU="$1"; SHARD="$2"; TOTAL="$3"
OUT=exp/results/selection_sweep
LOG="exp/results/sweep_gpu${GPU}.log"
NEED_MB=2000

ARMS=(A_dice H_aug G_focal K_focal_aug)
RUNS=()
for SEED in 0 1 2 3 4 5; do
    for ARM in "${ARMS[@]}"; do RUNS+=("${ARM}_s${SEED}"); done
done

MINE=()
for ((i = SHARD; i < ${#RUNS[@]}; i += TOTAL)); do MINE+=("${RUNS[$i]}"); done
echo "=== $(date '+%F %T') gpu=$GPU shard=$SHARD/$TOTAL, ${#MINE[@]} runs ===" \
    | tee -a "$LOG"

for RUN in "${MINE[@]}"; do
    if [ -f "$OUT/$RUN/final.pt" ]; then
        echo "--- $RUN already done" | tee -a "$LOG"; continue
    fi
    # Queue behind other users rather than crashing on OOM or evicting them.
    while true; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
               -i "$GPU")
        [ "$FREE" -ge "$NEED_MB" ] && break
        echo "$(date '+%T') gpu$GPU has ${FREE}MB free, need ${NEED_MB}; waiting" \
            | tee -a "$LOG"
        sleep 300
    done
    echo "--- $(date '+%F %T') $RUN ---" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" exp/train.py --results "$OUT" \
        --keep-epochs "$RUN" 2>&1 | tee -a "$LOG" \
        || echo "!!! $RUN FAILED, continuing" | tee -a "$LOG"
done
echo "=== $(date '+%F %T') gpu=$GPU shard=$SHARD done ===" | tee -a "$LOG"
