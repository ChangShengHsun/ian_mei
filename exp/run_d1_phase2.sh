#!/usr/bin/env bash
# D1 phase 2: build what phase 1's ceiling licenses, and the competitor that
# could make the whole line unnecessary.
#
#   D-E  centreline-weighted loss, NO direction. Always runs. If one weight
#        map captures the budget, D1 is an expensive route to it.
#   D-B  the oriented propagation layer driven by the model's own direction
#        head, plus the same layer driven by NOISE. Runs only if phase 1 says
#        knowing the axis is worth at least 3 points over isotropic dilation.
#
# Gates on artifacts, never on a PID. Waits for other people's jobs, never
# touches them.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/d1_phase2.log
SWEEP=exp/results/selection_sweep
export OMP_NUM_THREADS=4

say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }
both () {   # both <stage>
    bash exp/run_task.sh "$1" 0 0 2 "$SWEEP" 1 & local a=$!
    bash exp/run_task.sh "$1" 1 1 2 "$SWEEP" 1 & local b=$!
    wait "$a"; wait "$b"
}

say "phase 2 waiting for phase 1's ceiling"
until [ -f "$SWEEP/direction_ceiling.csv" ]; do sleep 120; done
say "phase 1 done"

# NOT `if cmd | tee`: bash tests the last command of a pipeline, which is tee,
# and tee always exits 0. That swallowed a CLOSED gate on 2026-08-26 and cost
# fifteen GPU hours with the word CLOSED printed directly above them.
"$PY" exp/gate_d1.py > /tmp/gate_d1.out 2>&1
GATE=$?
cat /tmp/gate_d1.out | tee -a "$LOG"

# D-E first either way: it is the competitor, it is cheap, and its answer is
# worth having whichever way the gate fell.
say "D-E: centreline-weighted loss, 12 runs"
both d1e

if [ "$GATE" -eq 0 ]; then
    say "D-B: the propagation layer and its shuffled control, 24 runs"
    both d1b
else
    say "D-B skipped by the gate -- the mechanism did not clear 3 points"
fi

# Rescore the whole sweep so every arm is judged by the same rule from the
# same CSV. No arguments: sweep_score.py OVERWRITES, and an explicit subset
# would silently shrink the file to that subset.
say "rescoring the sweep"
cp "$SWEEP/checkpoint_scores.csv" "$SWEEP/checkpoint_scores.pre_phase2.csv"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py >> "$LOG" 2>&1; then
    say "rescore OK"
    CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_direction.py \
        > exp/results/direction_summary.txt 2>&1
    CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_selection.py \
        > exp/results/selection_summary.txt 2>&1
    cat exp/results/direction_summary.txt | tee -a "$LOG"
else
    say "rescore FAILED -- restoring the pre-phase-2 scores"
    cp "$SWEEP/checkpoint_scores.pre_phase2.csv" "$SWEEP/checkpoint_scores.csv"
fi
say "phase 2 done"
