#!/usr/bin/env bash
# Task 1 of prompt_postproc.md: one direction predictor under the clean
# protocol, so the post-processing layer has a field to run on.
#
# WHY IT DOES NOT EXIST YET. exp/results/heldout holds 455 runs and not one
# plain _dir head; the twelve that exist are under exp/results/selection_sweep,
# trained before --protocol heldout, with their checkpoints selected on the
# test set. The ceiling table's `predicted` column is therefore `--` for every
# arm without a head of its own, which is exactly the column the paper's claim
# rests on.
#
# TWO ARMS, NOT ONE. A_dice_dir and H_aug_dir, so "does the field depend on
# which arm trained it" is a measurement rather than an assumption. The field
# is a property of the image; if the two disagree, that framing is wrong and
# it is better to know before the sweep than after.
#
# Cost, from the measured step time on this machine (9.9 ms/step for _dir at
# batch 32, 312 steps an epoch, 100 epochs): about 5 minutes a run, so 12 runs
# on two cards is around half an hour. Budgeted from a measurement, as
# CLAUDE.md requires -- a guess has cost this project a night before.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/dirhead.log
OUT=exp/results/heldout
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in "exp/gpu_queue.py --selftest" "exp/test_protocol.py" \
             "exp/summarize_direction_ceiling.py --selftest"; do
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
    LEFT=$("$PY" exp/gpu_queue.py dirhead --pending --results "$OUT" | wc -l)
    [ "$LEFT" -eq 0 ] && { say "dirhead nothing pending"; break; }
    say "pass $PASS: dirhead, $LEFT run(s) pending"
    both dirhead
done
say "all done"
