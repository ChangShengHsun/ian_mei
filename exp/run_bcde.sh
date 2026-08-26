#!/usr/bin/env bash
# Tasks B, C, D, E after task A, in the work order's priority order.
#
# B is CONDITIONAL. Ivan's rule: if no selection rule beats the status quo,
# B's fifteen GPU hours should not be spent. exp/gate_task_b.py decides, and
# it was written before A3 produced a number.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/bcde.log
B_OUT=exp/results/selection_sweep_31m

both () {   # both() <stage> <results-dir> <keep>
    bash exp/run_task.sh "$1" 0 0 2 "$2" "$3" &
    local a=$!
    bash exp/run_task.sh "$1" 1 1 2 "$2" "$3" &
    local b=$!
    wait "$a"; wait "$b"
}

echo "=== $(date '+%F %T') waiting for task A3 ===" | tee -a "$LOG"
until [ -f exp/results/selection_summary.txt ]; do sleep 300; done
echo "$(date '+%F %T') A3 done" | tee -a "$LOG"

echo "=== task B gate ===" | tee -a "$LOG"
# NOT `if cmd | tee`: bash tests the LAST command of a pipeline, which is
# tee, and tee always exits 0. That swallowed the gate's exit 1 on 2026-08-26
# and task B ran its fifteen GPU hours against a CLOSED verdict, with the word
# CLOSED printed in the log directly above it. Capture first, test after.
"$PY" exp/gate_task_b.py > /tmp/gate_task_b.out 2>&1
GATE=$?
cat /tmp/gate_task_b.out | tee -a "$LOG"
if [ "$GATE" -eq 0 ]; then
    df -h /home/ivanchang | tail -1 | tee -a "$LOG"
    FREE_GB=$(df --output=avail -BG /home/ivanchang | tail -1 | tr -dc '0-9')
    if [ "$FREE_GB" -lt 40 ]; then
        # 15 runs x 10 epochs x 118 MB = 17.4 GB, and a margin for everything
        # else the night writes. Below this, keeping every epoch is not safe.
        echo "SKIPPING B: only ${FREE_GB}GB free, need 40" | tee -a "$LOG"
    else
        mkdir -p "$B_OUT"
        both taskb "$B_OUT" 1
        echo "$(date '+%F %T') task B done" | tee -a "$LOG"
    fi
else
    echo "$(date '+%F %T') task B gate CLOSED, skipping to C" | tee -a "$LOG"
fi

echo "=== task C: seeds 6-11 at 117k ===" | tee -a "$LOG"
both taskc exp/results 0
echo "$(date '+%F %T') task C done" | tee -a "$LOG"

echo "=== task C2/E1: rescore everything, both protocols ===" | tee -a "$LOG"
for CKPT in final.pt best.pt; do
    CUDA_VISIBLE_DEVICES=0 "$PY" exp/stratify.py --checkpoint "$CKPT" 2>&1 \
        | tail -2 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 "$PY" exp/erl.py --checkpoint "$CKPT" 2>&1 \
        | grep -E "wrote|WARNING" | tee -a "$LOG"
done
"$PY" exp/summarize_capacity.py > exp/results/capacity3_summary.txt 2>&1
"$PY" exp/summarize_erl.py > exp/results/erl_summary.txt 2>&1
"$PY" exp/summarize_combo.py > exp/results/combo_summary_12seed.txt 2>&1
echo "$(date '+%F %T') rescoring done" | tee -a "$LOG"

echo "=== all queued work finished $(date '+%F %T') ===" | tee -a "$LOG"
