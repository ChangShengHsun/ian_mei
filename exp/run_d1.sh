#!/usr/bin/env bash
# The D1 direction line: measure the ceiling before building the method.
#
# Phase 0 is done and is in the record: the intact budget is 36.6 points of
# traced tree, of which 19.9 is erl.py's splitting rule and 16.7 is real;
# severing breaks are worth 5.1 more. So a direction-driven method is playing
# for about 21.8 points, against C1's 5.1.
#
# Phase 1, here, asks whether knowing the axis can reach any of it. No
# training: every checkpoint it reads is already on disk. If the answer is
# under 3 points the line stops and nothing in phase 2 is built.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/d1.log
SWEEP=exp/results/selection_sweep
export OMP_NUM_THREADS=4

say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }
cpu () { CUDA_VISIBLE_DEVICES="" "$PY" "$@"; }

say "D1 phase 1: the direction ceiling"

# The mechanism and both verdict scripts self-test before anything is scored.
# A sweep that runs for two hours and then cannot be trusted is worse than one
# that refuses to start.
for CHECK in exp/anisotropic.py exp/direction_ceiling.py \
             exp/summarize_direction_ceiling.py exp/erl_convention.py; do
    if ! cpu "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to run the sweep"
        exit 1
    fi
done
say "all selftests passed"

if [ -f "$SWEEP/direction_ceiling.csv" ]; then
    say "direction_ceiling.csv already exists, skipping the sweep"
else
    say "sweeping (4 arms x 6 seeds x 20 images, ~2h on CPU)"
    if cpu exp/direction_ceiling.py >> "$LOG" 2>&1; then
        say "sweep OK"
    else
        say "sweep FAILED"
        exit 1
    fi
fi

cpu exp/summarize_direction_ceiling.py > exp/results/d1_ceiling.txt 2>&1
cat exp/results/d1_ceiling.txt | tee -a "$LOG"
say "D1 phase 1 done"
