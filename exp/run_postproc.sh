#!/usr/bin/env bash
# Task 2 of prompt_postproc.md: the post-processing sweep, and its verdict.
#
# GATED ON THE FIELD, NOT ON A JOB MARKER. It waits until the dirhead stage
# has no pending runs, which is what "the field exists" actually means.
# Waiting on a runner's "all done" line cost thirteen idle GPU hours on
# 2026-08-28, because that line comes after its CPU scoring.
#
# CPU ONLY, and sharded. The cost was MEASURED before queueing, as CLAUDE.md
# requires: oriented_dilation is 85 ms on a DRIVE image, 19 geometries x 3
# sources plus 5 isotropic settings is about 5 s per (run, image), and
# 12 arms x 6 seeds x 20 images is roughly two hours in one process. Six
# shards bring it under half an hour and leave both GPUs free for training.
#
# Each shard writes its OWN csv. Two processes appending to one file
# interleave partial lines under load, and a half-written row is worse than a
# missing one because it parses.
#
# TWO FIELDS, not one. H_aug_dir supplies the headline; A_dice_dir is run
# afterwards to answer "does the field depend on which arm trained it" as a
# measurement rather than an assumption. If the two disagree, the claim that
# a tangent field is a property of the IMAGE is wrong, and that is worth
# knowing before the paper is written around it.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/postproc.log
OUT=exp/results/heldout
SHARDS=6
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in "exp/postproc_ceiling.py --selftest" \
             "exp/summarize_postproc.py --selftest" \
             "exp/summarize_direction_ceiling.py --selftest"; do
    if ! $PY $CHECK >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed"

WAITED=0
while [ "$("$PY" exp/gpu_queue.py dirhead --pending --results "$OUT" | wc -l)" -gt 0 ]; do
    [ "$WAITED" -ge 144 ] && { say "dirhead unfinished after 12h; starting anyway"; break; }
    [ $(( WAITED % 12 )) -eq 0 ] && say "waiting for dirhead ($(( WAITED / 12 ))h)"
    sleep 300
    WAITED=$(( WAITED + 1 ))
done
say "direction heads ready"

sweep_with () {   # sweep_with <field arm>
    local FIELD="$1" PIDS=()
    say "sweeping with field $FIELD, $SHARDS shards"
    for INDEX in $(seq 0 $(( SHARDS - 1 ))); do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" exp/postproc_ceiling.py \
            --field "$FIELD" --shard "$INDEX/$SHARDS" \
            >> "exp/results/postproc_${FIELD}_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
    say "field $FIELD done: $(cat "$OUT"/postproc_ceiling*.csv 2>/dev/null | wc -l) rows total"
}

sweep_with H_aug_dir
CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_postproc.py \
    > exp/results/postproc_verdict.txt 2>&1
cat exp/results/postproc_verdict.txt | tee -a "$LOG"

sweep_with A_dice_dir
CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_postproc.py \
    > exp/results/postproc_verdict.txt 2>&1
cat exp/results/postproc_verdict.txt | tee -a "$LOG"
say "all done"
