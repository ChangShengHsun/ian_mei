#!/usr/bin/env bash
# Price the local decision rule before anyone builds a tracer.
#
# WHERE THIS CAME FROM. Two of this repo's own measurements, neither acted on:
#
#   link_ceiling.py (2026-08-27)  a PERFECT fragment linker buys +2.9 to +4.7
#                                 points; filling every missed centreline
#                                 pixel buys +58 to +65 at 1.05x foreground.
#                                 "The tree is not fragmented so much as
#                                 unseen."
#   composition.py (24 seeds)     every operator loses to `lower` on all ten
#                                 arms at both budgets, and `endpoint_shuf`
#                                 matches `endpoint`.
#
# So the prize is in recovering unseen vessel, and the best tool so far is a
# GLOBAL threshold drop -- which accepts a faint pixel in the middle of
# nowhere on the same terms as a faint pixel continuing a confident vessel.
# Hysteresis is the smallest rule that separates those, and grep says this
# repo has never tried it. Predictions are pre-registered in the header of
# exp/hysteresis.py, not here.
#
# WHY NOT JUST BUILD THE TRACER. The literature's answer to this problem is
# iterative tracing (NETracer, ICCV 2025; the RoadTracer line). That is a
# large build. C1.0's lesson is that the cheap ceiling measurement comes
# first: if the crudest form of "use context from where you came from" buys
# nothing here, a tracer is a big build on a small prize.
#
# CPU ONLY. The two 4070 Ti are shared and were held by another user at
# 2.7 GB free when this was written; nothing here needs a GPU.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/hyst.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for OTHER in run_deep.sh run_survival.sh run_gaps.sh run_anchor.sh; do
    while pgrep -u "$USER" -f "exp/$OTHER" > /dev/null 2>&1; do
        say "$OTHER still up, waiting"; sleep 300
    done
done
say "no other queue running"

if ! $PY exp/hysteresis.py --selftest >> "$LOG" 2>&1; then
    say "SELFTEST FAILED -- refusing to start"; exit 1
fi
say "selftest passed (operator takes the attached continuation and drops the \
detached blob; control matched and reproducible; foreground monotone)"

run_shards () {   # run_shards <count> <args...>
    local COUNT="$1"; shift
    local PIDS=()
    for INDEX in $(seq 0 $(( COUNT - 1 ))); do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" exp/hysteresis.py \
            "$@" --shard "$INDEX/$COUNT" \
            >> "exp/results/hysteresis_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
}

# Dev first: the operating point for every source is chosen on these 5
# images, and scoring test before dev would leave a table whose settings
# cannot be picked without a second pass.
say "phase 1: the 5 dev images"
run_shards 6 --dev
say "phase 1 done"

say "phase 2: the 20 test images"
run_shards 6
say "phase 2 done"

say "phase 3: report"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/hysteresis.py --report \
    > exp/results/hysteresis_verdict.txt 2>&1
say "phase 3 done: exp/results/hysteresis_verdict.txt"
say "all done"
