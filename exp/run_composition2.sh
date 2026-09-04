#!/usr/bin/env bash
# composition, re-run with the two controls the first table was missing.
#
# THE FIRST TABLE'S RESULT: at a 0.02 Dice budget on top of each arm's own
# dev-picked threshold -- the configuration that beat the whole-mask layer
# 10/10 -- whole-mask oriented dilation could not afford ANY setting, while
# endpoint-restricted growth passed the gate on all ten arms under both ERL
# conventions (+0.8 to +5.6 points, convention B).
#
# THAT RESULT HAS TWO READINGS and the first table cannot separate them:
#   the ENDPOINTS are the right place to spend the budget, or
#   the FIELD is the right direction to spend it in.
# `endpoint_iso` keeps the place and drops the direction; `endpoint_shuf`
# keeps the place and replaces the field with noise. If either scores like
# `endpoint`, the contribution is the restriction and not the field -- the
# same shape as the closing baseline that beat the C1 oracle until its cost
# was matched.
#
# The first table is kept as composition_verdict.pre_controls.txt and its
# CSVs under results/heldout/pre_controls/. Superseded, not deleted: it is
# internally consistent, it just cannot answer the question it raised.
#
# CPU ONLY. Measured 83 min for 87 measures per (run, image) at 4 shards;
# the two controls take it to 119, so budget about two hours.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/composition2.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

if ! $PY exp/composition.py --selftest >> "$LOG" 2>&1; then
    say "SELFTEST FAILED -- refusing to start"; exit 1
fi
say "selftest passed"

# Behind transfer_calibration: both are CPU, and 3+4 shards plus another
# user's work is more contention than the box should carry.
WAITED=0
while pgrep -u "$USER" -f "transfer_calibration.py .*--shard" >/dev/null; do
    [ "$WAITED" -ge 72 ] && { say "transfercal unfinished after 6h; starting anyway"; break; }
    sleep 300
    WAITED=$(( WAITED + 1 ))
done
say "transfercal done"

for SPLIT in "--dev" ""; do
    say "composition ${SPLIT:-test}"
    PIDS=()
    for INDEX in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" exp/composition.py \
            $SPLIT --shard "$INDEX/4" \
            >> "exp/results/composition_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
done

$PY exp/composition.py --report > exp/results/composition_verdict.txt 2>&1
say "verdict written"
say "all done"
