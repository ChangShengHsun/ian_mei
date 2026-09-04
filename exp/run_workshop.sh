#!/usr/bin/env bash
# The two measurements the workshop story is missing, queued behind axisfix.
#
# 1. terminal_anatomy -- the objection that ERL is blind to DRIVE's failures
#    because DRIVE's failures are at the tips. ERL is length-weighted, so a
#    terminal break costs almost nothing: the selftest reproduces it at 11.5%
#    against 54.8% for the same break mid-vessel. This measures where the loss
#    ACTUALLY is on DRIVE, and whether the arm ranking survives deleting the
#    tips. It can lose, and losing is a result.
#
# 2. transfer_calibration -- whether calibration.md's headline is a property
#    of topology losses or a property of one 20-image dataset. The 54 transfer
#    checkpoints already exist, so this is scoring, not training.
#
# BOTH CPU ONLY. Gated on axisfix finishing rather than on a PID or a marker:
# waiting on a runner's "all done" line cost thirteen idle GPU hours on
# 2026-08-28, and the artifact is what "finished" actually means.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/workshop.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in "exp/terminal_anatomy.py --selftest" \
             "exp/transfer_calibration.py --selftest"; do
    if ! $PY $CHECK >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed"

# axisfix writes composition_verdict.txt last. Gate on that file, not on the
# tmux session or a pid.
WAITED=0
while [ ! -f exp/results/composition_verdict.txt ]; do
    [ "$WAITED" -ge 96 ] && { say "axisfix unfinished after 8h; starting anyway"; break; }
    [ $(( WAITED % 12 )) -eq 0 ] && say "waiting for axisfix ($(( WAITED / 12 ))h)"
    sleep 300
    WAITED=$(( WAITED + 1 ))
done
say "axisfix done"

run_shards () {   # run_shards <script> <count> <args...>
    local SCRIPT="$1" COUNT="$2"; shift 2
    local PIDS=()
    for INDEX in $(seq 0 $(( COUNT - 1 ))); do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" "$SCRIPT" "$@" \
            --shard "$INDEX/$COUNT" \
            >> "exp/results/$(basename "$SCRIPT" .py)_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
}

say "terminal anatomy"
run_shards exp/terminal_anatomy.py 4
$PY exp/terminal_anatomy.py --report > exp/results/terminal_verdict.txt 2>&1
say "terminal verdict written"

for DATASET in stare vessmap hrf; do
    say "transfer calibration: $DATASET"
    run_shards exp/transfer_calibration.py 3 "$DATASET"
done
$PY exp/transfer_calibration.py --report > exp/results/transfer_calibration_verdict.txt 2>&1
say "transfer calibration verdict written"

say "all done"
