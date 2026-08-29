#!/usr/bin/env bash
# The extended centreline-weight sweep: 16, 32, 64.
#
# WHY. The first sweep ran 1/2/4/8 and the response was MONOTONE INCREASING to
# its endpoint on all three bases, at rule (iv) and at matched Dice both. A
# sweep whose best value is its largest value has not found a peak, it has
# found the edge of the range. The pre-registered prediction -- single-peaked,
# with 8 costing Dice and buying nothing -- was simply wrong.
#
# 64 is here to BRACKET rather than to win: at 64 the centreline outweighs the
# vessel body 65:1, so the model should collapse toward drawing the skeleton
# alone. A sweep that ends before the collapse cannot locate the peak; one
# that contains it can.
#
# SEPARATE RUNNER, not an edit to run_heldout.sh. That script was executing
# when these arms were added and had already passed its last pending check, so
# it will never see them; and bash reads a running script by byte offset, so
# editing it mid-run makes it resume at the wrong place.
#
# NO WAIT ON THE OTHER QUEUES. run_task_heldout.sh waits for free VRAM before
# every run, which is what actually keeps concurrent jobs polite. Gating on
# another script's "all done" marker cost thirteen idle GPU hours on 2026-08-28
# because that marker comes after its CPU scoring, not after its training.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/weights.log
OUT=exp/results/heldout
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in "exp/gpu_queue.py --selftest" "exp/test_protocol.py" \
             "exp/select_heldout.py --selftest" "exp/matched_cost.py --selftest"; do
    if ! $PY $CHECK >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to queue"; exit 1
    fi
done
say "selftests passed"

both () {
    bash exp/run_task_heldout.sh "$1" 0 0 2 "$OUT" & local a=$!
    bash exp/run_task_heldout.sh "$1" 1 1 2 "$OUT" & local b=$!
    wait "$a"; wait "$b"
}

# heldout first: six seeds of the new weights is enough to locate the peak.
# heldout_seeds brings them to the twelve the other arms already have.
for PASS in 1 2; do
    for STAGE in heldout heldout_seeds; do
        LEFT=$("$PY" exp/gpu_queue.py "$STAGE" --pending --results "$OUT" | wc -l)
        [ "$LEFT" -eq 0 ] && { say "$STAGE nothing pending"; continue; }
        say "pass $PASS: $STAGE, $LEFT run(s) pending"
        both "$STAGE"
    done
    if [ "$PASS" = 1 ]; then
        say "scoring after pass 1"
        CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py --results "$OUT" \
            >> "$LOG" 2>&1 && {
            CUDA_VISIBLE_DEVICES="" "$PY" exp/select_heldout.py \
                > exp/results/heldout_summary.txt 2>&1
            CUDA_VISIBLE_DEVICES="" "$PY" exp/matched_cost.py \
                > exp/results/matched_cost.txt 2>&1
            cat exp/results/heldout_summary.txt | tee -a "$LOG"; }
    fi
done

say "final scoring"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py --results "$OUT" \
   >> "$LOG" 2>&1; then
    CUDA_VISIBLE_DEVICES="" "$PY" exp/select_heldout.py \
        > exp/results/heldout_summary.txt 2>&1
    CUDA_VISIBLE_DEVICES="" "$PY" exp/matched_cost.py \
        > exp/results/matched_cost.txt 2>&1
    cat exp/results/heldout_summary.txt | tee -a "$LOG"
    cat exp/results/matched_cost.txt | tee -a "$LOG"
else
    say "scoring FAILED"
fi
say "all done"
