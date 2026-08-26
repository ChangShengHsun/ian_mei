#!/usr/bin/env bash
# D1 phase 3: does the correction's geometry transfer, or is "1.0 widths" a
# DRIVE pixel count wearing a unit?
#
# CPU only, and independent of phases 1 and 2 -- it reads DRIVE-trained
# checkpoints that are already on disk and applies them to STARE, HRF and
# VessMAP. It waits for nothing, because it needs nothing they produce.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/d1_phase3.log
SWEEP=exp/results/selection_sweep
export OMP_NUM_THREADS=4
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in exp/transfer_ceiling.py exp/summarize_transfer.py; do
    if ! CUDA_VISIBLE_DEVICES="" "$PY" "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to run"
        exit 1
    fi
done
say "selftests passed"

if [ -f "$SWEEP/transfer_ceiling.csv" ]; then
    say "transfer_ceiling.csv exists, skipping the sweep"
else
    say "sweeping DRIVE / STARE / HRF / VessMAP"
    if ! CUDA_VISIBLE_DEVICES="" "$PY" exp/transfer_ceiling.py >> "$LOG" 2>&1
    then
        say "sweep FAILED"
        exit 1
    fi
fi
CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_transfer.py \
    > exp/results/transfer_summary.txt 2>&1
cat exp/results/transfer_summary.txt | tee -a "$LOG"
say "phase 3 done"
