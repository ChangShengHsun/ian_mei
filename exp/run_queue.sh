#!/usr/bin/env bash
# One card's worth of a stage. Serial within the card: CLAUDE.md's measured
# finding is that two training jobs on one device finish later than one after
# the other, and a 12 GB card at 31M has room but no reason.
#
#   exp/run_queue.sh <stage> <shard> <total> <gpu>
#
# train.py skips a run whose final.pt exists and resumes one whose ckpt.pt
# does, so re-running this script after any interruption is safe and is the
# intended way to recover. It gates on artifacts on disk, never on a PID.
set -u
cd "$(dirname "$0")/.." || exit 1

STAGE="$1"; SHARD="$2"; TOTAL="$3"; GPU="$4"
PY=.venv/bin/python
LOG="exp/results/queue_${STAGE}_gpu${GPU}.log"

echo "=== $(date '+%F %T') stage=$STAGE shard=$SHARD/$TOTAL gpu=$GPU ===" \
    | tee -a "$LOG"

mapfile -t RUNS < <("$PY" exp/gpu_queue.py "$STAGE" --shard "$SHARD/$TOTAL")
echo "${#RUNS[@]} runs on this card" | tee -a "$LOG"

FAILED=0
for RUN in "${RUNS[@]}"; do
    echo "--- $(date '+%F %T') $RUN ---" | tee -a "$LOG"
    if ! CUDA_VISIBLE_DEVICES="$GPU" "$PY" exp/train.py "$RUN" 2>&1 \
            | tee -a "$LOG"; then
        # Keep going: one arm crashing must not cost the night's other runs,
        # and the failure is on disk to read in the morning.
        echo "!!! $RUN FAILED, continuing" | tee -a "$LOG"
        FAILED=$((FAILED + 1))
    fi
done

echo "=== $(date '+%F %T') stage=$STAGE gpu=$GPU done, $FAILED failed ===" \
    | tee -a "$LOG"
exit "$FAILED"
