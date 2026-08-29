#!/usr/bin/env bash
# The held-out protocol batch. Queues BEHIND the running d1all job.
#
# WHY. exp/drive.py's "val" split is DRIVE's official TEST set. Two things
# chose a checkpoint on it:
#
#   best.pt        highest Dice over all 20 test images -- a real leak. The
#                  number reported from it is the maximum of ten draws on the
#                  set it is reported on. erl_best.csv and stratify_best.csv
#                  are read off it; erl.csv and stratify.csv are read off
#                  final.pt at a fixed epoch 100 and are NOT affected.
#   rules (i)-(iv) select on the odd test images, report on the even ones. No
#                  leak into the reported half, but it still touches the test
#                  set before reporting and halves the reporting set to 10.
#
# Under --protocol heldout the model is fitted on 15 of DRIVE's TRAINING
# images, every rule reads the 5 held out from the same directory, and the
# test set is read once, whole -- 20 report images instead of 10.
#
# STAGES, in the order a queue stopped early stays readable:
#
#   heldout         3 bases x (baseline + 4 centreline weights) x 6 seeds = 90
#                   D-E is the only intervention in this series that passed
#                   the seed gate, and its weight was never swept. Seed-major
#                   with each seed's baselines first, so one seed of the whole
#                   grid lands before any arm reaches seed 1.
#   heldout_series  the 60+ runs whose published numbers came off best.pt,
#                   enumerated from erl_best.csv rather than typed. 117k arms
#                   go to six seeds; a paired t on three has two df.
#
# GATING. On the artifact, never on a PID: run_d1_all.sh writes "all done" to
# its log when it finishes. The wait is capped, because run_task_heldout.sh
# waits for free VRAM anyway -- if d1all is stuck, this queues politely behind
# whatever is actually running rather than blocking for ever.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/heldout.log
OUT=exp/results/heldout
D1_LOG=exp/results/d1_all.log
MAX_WAIT_HOURS=30
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

mkdir -p "$OUT"

for CHECK in "exp/drive.py" "exp/test_protocol.py" \
             "exp/select_heldout.py --selftest" "exp/gpu_queue.py --selftest"; do
    if ! $PY $CHECK >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to queue"; exit 1
    fi
done
say "selftests passed (splits, protocol, verdict script, queue)"

# SKIP_WAIT=1 starts immediately. Used when d1all has finished TRAINING and
# is only post-processing: on 2026-08-28 it sat in transfer_ceiling.py for
# hours with card 0 completely idle, and this queue waited for a marker that
# says "the whole job is done" rather than "the cards are free". The per-run
# VRAM gate in run_task_heldout.sh is what actually keeps the two polite.
WAITED=0
while [ "${SKIP_WAIT:-0}" != "1" ] && ! grep -q "all done" "$D1_LOG" 2>/dev/null; do
    if [ "$WAITED" -ge $(( MAX_WAIT_HOURS * 12 )) ]; then
        say "d1all has not finished in ${MAX_WAIT_HOURS}h; starting anyway "\
"(run_task_heldout.sh waits for VRAM, so this queues behind whatever runs)"
        break
    fi
    [ $(( WAITED % 12 )) -eq 0 ] && \
        say "waiting for d1all ($(( WAITED / 12 ))h so far)"
    sleep 300
    WAITED=$(( WAITED + 1 ))
done
if [ "${SKIP_WAIT:-0}" = "1" ]; then
    say "SKIP_WAIT set; starting alongside whatever d1all still has running"
else
    grep -q "all done" "$D1_LOG" 2>/dev/null && say "d1all finished; starting"
fi

both () {
    bash exp/run_task_heldout.sh "$1" 0 0 2 "$OUT" & local a=$!
    bash exp/run_task_heldout.sh "$1" 1 1 2 "$OUT" & local b=$!
    wait "$a"; wait "$b"
}

# Two passes: a run that died on a transient (a full card, someone else's OOM)
# is retried once. pending() gates on final.pt, so a finished run is skipped.
for PASS in 1 2; do
    for STAGE in heldout heldout_series; do
        LEFT=$("$PY" exp/gpu_queue.py "$STAGE" --pending --results "$OUT" | wc -l)
        [ "$LEFT" -eq 0 ] && { say "$STAGE nothing pending"; continue; }
        say "pass $PASS: $STAGE, $LEFT run(s) pending"
        both "$STAGE"
    done
    # The D-E verdict is readable as soon as `heldout` lands; printing it
    # after pass 1 means the answer arrives hours before the cleanup half
    # finishes, rather than only at the very end.
    if [ "$PASS" = 1 ]; then
        say "scoring the sweep on the test set (pass 1)"
        CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py --results "$OUT" \
            >> "$LOG" 2>&1 && \
        CUDA_VISIBLE_DEVICES="" "$PY" exp/select_heldout.py \
            > exp/results/heldout_summary.txt 2>&1
        cat exp/results/heldout_summary.txt 2>/dev/null | tee -a "$LOG"
    fi
done

say "final scoring"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py --results "$OUT" \
   >> "$LOG" 2>&1; then
    CUDA_VISIBLE_DEVICES="" "$PY" exp/select_heldout.py \
        > exp/results/heldout_summary.txt 2>&1
    cat exp/results/heldout_summary.txt | tee -a "$LOG"
else
    say "scoring FAILED"
fi
say "all done"
