#!/usr/bin/env bash
# After the queue: score the new runs and print E13's three-point verdict.
#
# Runs unattended at the end of the night, so it gates on artifacts on disk
# rather than on a PID -- CLAUDE.md's rule, and the reason a queue that waits
# on an already-dead process starts its next job against a half-finished one.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/post_e13.log

wait_for () {                       # $1 = stage name
    local missing
    while true; do
        missing=$("$PY" exp/gpu_queue.py "$1" --pending | wc -l)
        [ "$missing" -eq 0 ] && break
        echo "$(date '+%T') waiting on $missing run(s) of $1" >> "$LOG"
        sleep 300
    done
    echo "$(date '+%T') $1 complete" | tee -a "$LOG"
}

wait_for e13
wait_for curve

# ONLY the new runs. stratify.py appends and does not skip what is already
# there, and the 27 recovered runs' rows are already in stratify.csv from the
# laptop. Re-scoring them would double every one of their rows: the means
# would not move, but the paired t-test's n would double and every t would
# inflate -- lesson four's pseudo-replication, arriving through the back door.
NEW=$("$PY" exp/gpu_queue.py e13)
ALREADY=$(cut -d, -f1 exp/results/stratify.csv | tail -n +2 | sort -u)
TODO=""
for RUN in $NEW; do
    if echo "$ALREADY" | grep -qx "$RUN"; then
        echo "$RUN already in stratify.csv, skipping" | tee -a "$LOG"
    else
        TODO="$TODO $RUN"
    fi
done

if [ -n "$TODO" ]; then
    echo "$(date '+%T') scoring:$TODO" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 "$PY" exp/stratify.py $TODO 2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "===== E13, three capacity points =====" | tee -a "$LOG"
"$PY" exp/summarize_capacity.py 2>&1 \
    | tee exp/results/capacity3_summary.txt | tee -a "$LOG"
echo "$(date '+%F %T') post-processing done" | tee -a "$LOG"
