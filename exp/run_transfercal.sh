#!/usr/bin/env bash
# transfer_calibration, relaunched after the root was corrected.
#
# The first attempt on 2026-09-01 pointed at results/cross/, which holds the
# PRE-heldout transfer runs, found no final.pt, and wrote three empty CSVs in
# eighteen seconds. The selftest now asserts the root holds finished runs for
# every arm and that every protocol.txt in it reads "heldout", so the same
# failure cannot report as "nothing scored yet" again.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/transfercal.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

if ! $PY exp/transfer_calibration.py --selftest >> "$LOG" 2>&1; then
    say "SELFTEST FAILED -- refusing to start"; exit 1
fi
say "selftest passed"

for DATASET in stare vessmap hrf; do
    say "$DATASET"
    PIDS=()
    for INDEX in 0 1 2; do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" \
            exp/transfer_calibration.py "$DATASET" --shard "$INDEX/3" \
            >> "exp/results/transfer_calibration_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
done

$PY exp/transfer_calibration.py --report \
    > exp/results/transfer_calibration_verdict.txt 2>&1
say "verdict written"
say "all done"
