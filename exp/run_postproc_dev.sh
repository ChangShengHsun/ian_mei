#!/usr/bin/env bash
# The dev half of the post-processing sweep, and the corrected verdict.
#
# WHY THIS EXISTS. summarize_postproc.py's first version chose the dilation
# geometry on the same twenty test images it then reported on. That is the
# same leak as selecting a checkpoint on the test set and selecting a
# threshold on the test curve -- one level further down, and the third time
# this project has made it. Its selftest now demonstrates the size: on a
# synthetic case where dev and test disagree about the best geometry, picking
# on test reads +0.20 where picking on dev reads +0.06.
#
# The fix needs the same sweep over the 5 DEV images, which is a quarter of
# the cost. The geometry is then chosen there and only READ on test.
#
# The first table (postproc_verdict.txt, 2026-09-01 00:33) is superseded and
# was optimistic by an unknown amount.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/postproc_dev.log
OUT=exp/results/heldout
SHARDS=6
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in "exp/postproc_ceiling.py --selftest" \
             "exp/summarize_postproc.py --selftest"; do
    if ! $PY $CHECK >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed"

# Behind the test sweep: both are CPU-bound and this box is shared.
while pgrep -f "postproc_ceiling.py --field" | grep -qv "$$"; do
    sleep 120
done
say "test sweep finished; starting dev"

for FIELD in H_aug_dir A_dice_dir; do
    say "dev sweep with field $FIELD"
    PIDS=()
    for INDEX in $(seq 0 $(( SHARDS - 1 ))); do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" exp/postproc_ceiling.py \
            --dev --field "$FIELD" --shard "$INDEX/$SHARDS" \
            >> "exp/results/postproc_dev_${FIELD}_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
    say "field $FIELD dev done"
done

say "corrected verdict: geometry chosen on dev, read on test"
CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_postproc.py \
    > exp/results/postproc_verdict.txt 2>&1
cat exp/results/postproc_verdict.txt | tee -a "$LOG"
say "all done"
