#!/usr/bin/env bash
# The paper's Table 1, rebuilt on the protocol the paper is about.
#
# WHY. erl_reference.txt (08-28), erl_length.txt (08-29) and
# erl_convention.txt (08-27) are the three tables that price ERL's three
# under-specifications -- and all three predate the held-out protocol, which
# landed 09-01. Their scripts hardwire `selection.SWEEP`, `drive.load_split
# ("val")` and the report-half rule. CLAUDE.md's standing rule is that
# pre-heldout numbers are NOT comparable to held-out ones, so a paper whose
# thesis is "selection leaks change conclusions" would open on a table built
# under the leaking protocol. A reviewer finds that by comparing two dates.
#
# exp/erl_spec.py measures all three axes together, at 12 seeds, on the clean
# protocol -- and crossed rather than separate, because whether the axes
# interact is exactly what three separate tables cannot answer.
#
# THE THREE ORIGINALS ARE NOT TOUCHED. They are the record of the pre-heldout
# measurement; superseded, not overwritten.
#
# COST, measured before queueing as the repo requires: 13.6 s per run on one
# core-set, 120 runs, four shards -- about seven minutes. No GPU.
#
# PRE-REGISTERED 2026-09-03, before any row of it exists:
#   1. The identity `full = covered x coverage` holds in every cell. It is
#      asserted per image in the selftest; if the aggregated table breaks it,
#      the aggregation is wrong, not the identity.
#   2. The splitting rule stays the largest of the three axes, worth more
#      than 15 points. It was +19.9 to +27.4 pre-heldout, and the held-out
#      protocol changes which epoch and which threshold, not what a bridged
#      gap is.
#   3. pixels vs edges stays under 1 point. Numerator and denominator take
#      the same correction; a larger gap here would mean the two are no
#      longer being measured the same way.
#   4. The per-arm SPREAD -- highest cell minus lowest -- exceeds 20 points
#      on every arm. That number is the paper's headline for this table: the
#      range of ERLs one set of predictions can honestly be reported as.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/spec.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

if ! $PY exp/erl_spec.py --selftest >> "$LOG" 2>&1; then
    say "SELFTEST FAILED: exp/erl_spec.py -- refusing to start"; exit 1
fi
say "selftest passed (both anchors, the identity, and the units)"

PIDS=()
for INDEX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 $PY exp/erl_spec.py \
        --shard "$INDEX/4" >> "exp/results/erl_spec_${INDEX}.log" 2>&1 &
    PIDS+=($!)
done
say "four shards launched"
for PID in "${PIDS[@]}"; do wait "$PID"; done
say "scoring done"

$PY exp/erl_spec.py --report > exp/results/erl_spec.txt 2>&1
say "wrote exp/results/erl_spec.txt"
say "all done"
