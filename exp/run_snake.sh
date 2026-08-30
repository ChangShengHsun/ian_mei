#!/usr/bin/env bash
# D-C: the redesign of D-B, and the frontier reading that judges it.
#
# WHY D-B FAILED, measured (exp/snake.py carries the evidence):
#   reach 2 vessel widths against gaps whose p90 is 7.4 widths;
#   a straight kernel long enough for those gaps leaves the vessel 25-73%
#   of the time, so no sweep over `along` escapes it;
#   and it could only ADD foreground, which the loss punishes and which
#   calibration.md showed a lower threshold gives away for free.
#
# 96 runs: 2 bases x (2 lengths x {snake, snkstr, snkshf} + 2 iterated arms)
# x 6 seeds, into exp/results/heldout under --protocol heldout.
#
# Then the two frontier sweeps, because D-C must be judged at each arm's own
# operating point and not at 0.5. That reading is the one that reversed
# K_focal_aug from +13.6% to -4.2%, and a propagation layer changes how much
# foreground is predicted, so it is exactly the kind of thing 0.5 flatters.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/snake.log
OUT=exp/results/heldout
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in "exp/snake.py" "exp/gpu_queue.py --selftest" \
             "exp/summarize_snake.py --selftest" "exp/test_protocol.py" \
             "exp/frontier.py --selftest" "exp/calibration.py --selftest"; do
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

for PASS in 1 2; do
    LEFT=$("$PY" exp/gpu_queue.py snake --pending --results "$OUT" | wc -l)
    [ "$LEFT" -eq 0 ] && { say "snake nothing pending"; break; }
    say "pass $PASS: snake, $LEFT run(s) pending"
    both snake
done

say "scoring: checkpoints on test"
CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py --results "$OUT" >> "$LOG" 2>&1 \
    || say "sweep_score FAILED"
say "frontier: dev then test (threshold is chosen on dev, read on test)"
CUDA_VISIBLE_DEVICES="" "$PY" exp/frontier.py --dev >> "$LOG" 2>&1 \
    || say "frontier --dev FAILED"
CUDA_VISIBLE_DEVICES="" "$PY" exp/frontier.py >> "$LOG" 2>&1 \
    || say "frontier FAILED"

say "verdicts"
CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_snake.py > exp/results/snake_verdict.txt 2>&1
CUDA_VISIBLE_DEVICES="" "$PY" exp/calibration.py > exp/results/calibration.txt 2>&1
CUDA_VISIBLE_DEVICES="" "$PY" exp/select_heldout.py > exp/results/heldout_summary.txt 2>&1
cat exp/results/snake_verdict.txt | tee -a "$LOG"
say "all done"
