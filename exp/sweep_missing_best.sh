#!/usr/bin/env bash
# One run finished before best.pt existed and the queue had already walked
# past it. Retrain it once every card is idle, so the best-epoch protocol has
# no hole at seed 0 of the base=16 baseline -- seed 0 is in every paired
# comparison, and the seed gate needs every seed to agree in sign.
#
# The gate is artifact-based, not PID-based: recover_rest is the LAST stage on
# both cards, so "recover_rest has nothing pending" means both queues are
# done. Polling for "no train.py running" would fire in the one-second gap
# between two runs of a live queue.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/sweep_missing_best.log
RUN=A_dice_s0

while true; do
    LEFT=$("$PY" exp/gpu_queue.py recover_rest --pending | wc -l)
    E13=$("$PY" exp/gpu_queue.py e13 --pending | wc -l)
    [ "$LEFT" -eq 0 ] && [ "$E13" -eq 0 ] && break
    echo "$(date '+%T') waiting: recover_rest $LEFT, e13 $E13" >> "$LOG"
    sleep 300
done

echo "$(date '+%F %T') cards idle, retraining $RUN for its best.pt" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 "$PY" exp/train.py "$RUN" 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T') sweep done" | tee -a "$LOG"
