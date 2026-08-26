#!/usr/bin/env bash
# A1 -> A2 -> A3, gated on artifacts on disk rather than on a PID.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=exp/results/selection_sweep
LOG=exp/results/post_sweep.log
EXPECTED=24

while true; do
    DONE=$(find "$OUT" -name final.pt 2>/dev/null | wc -l)
    [ "$DONE" -ge "$EXPECTED" ] && break
    echo "$(date '+%T') A1: $DONE/$EXPECTED runs finished" >> "$LOG"
    sleep 300
done
echo "$(date '+%F %T') A1 complete, $EXPECTED runs" | tee -a "$LOG"
echo "kept checkpoints: $(find "$OUT" -name 'epoch*.pt' | wc -l)" | tee -a "$LOG"

# A2. One card is enough: this is inference, and the other may be someone
# else's. Queue behind them the same way the training did.
while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "$FREE" -ge 2000 ] && break
    echo "$(date '+%T') A2 waiting, gpu0 has ${FREE}MB free" >> "$LOG"
    sleep 300
done
CUDA_VISIBLE_DEVICES=0 "$PY" exp/sweep_score.py 2>&1 | tail -5 | tee -a "$LOG"

# A3.
"$PY" exp/summarize_selection.py 2>&1 \
    | tee exp/results/selection_summary.txt | tee -a "$LOG"
echo "$(date '+%F %T') A3 done" | tee -a "$LOG"
