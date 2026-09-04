#!/usr/bin/env bash
# The control postproc_verdict.txt is missing: does simply lowering the
# threshold buy more ERL than the oriented-dilation layer, at the same Dice?
#
# CPU ONLY and cheap -- 7 s per run on the 5 dev images, 30 s on the 20 test
# images, measured 2026-09-01 before queueing. Four shards finish both splits
# in about fifteen minutes, so this does not wait for anything: it runs
# beside the postproc sweep rather than behind it.
#
# DEV FIRST. The report picks the threshold on dev and reads it on test, so
# a test file without its dev counterpart cannot be reported anyway.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/threshold_control.log
SHARDS=4
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

if ! $PY exp/threshold_control.py --selftest >> "$LOG" 2>&1; then
    say "SELFTEST FAILED -- refusing to start"; exit 1
fi
say "selftest passed"

for SPLIT in "--dev" ""; do
    say "sweeping ${SPLIT:-test}"
    PIDS=()
    for INDEX in $(seq 0 $(( SHARDS - 1 ))); do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" \
            exp/threshold_control.py $SPLIT --shard "$INDEX/$SHARDS" \
            >> "exp/results/threshold_control_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
    say "${SPLIT:-test} done"
done

say "report"
$PY exp/threshold_control.py --report | tee exp/results/threshold_control_verdict.txt
say "all done"
