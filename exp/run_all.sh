#!/usr/bin/env bash
# The whole night, in prompt.md's priority order, across every card.
#
#   exp/run_all.sh [gpus]        # default 2
#
# Three stages with a real gate between the first two. The gate is task 1's
# own instruction: two runs first, and if the 31M baseline is not credible the
# other 23 are wasted. 31M parameters on DRIVE's 20 training images is badly
# overparameterised, so this is not ceremony.
set -u
cd "$(dirname "$0")/.." || exit 1

GPUS="${1:-2}"
PY=.venv/bin/python

run_stage () {
    local stage="$1" pids=()
    echo ""
    echo "########## stage $stage on $GPUS card(s), $(date '+%F %T') ##########"
    for ((gpu = 0; gpu < GPUS; gpu++)); do
        bash exp/run_queue.sh "$stage" "$gpu" "$GPUS" "$gpu" &
        pids+=($!)
    done
    # Waiting on our OWN children is fine; what CLAUDE.md forbids is gating on
    # a PID that may already be dead, which is the detached-queue case.
    for pid in "${pids[@]}"; do wait "$pid"; done
}

# The gate stage is two runs, so one per card whatever GPUS is; the shard
# arithmetic already deals them out and any extra card simply gets nothing.
run_stage gate

echo ""
echo "########## backbone gate ##########"
if ! "$PY" exp/gate_backbone.py; then
    echo ""
    echo "STOPPING. The 31M baseline is outside the pre-registered band, so"
    echo "the remaining task-1 runs would measure a broken backbone. Read"
    echo "exp/results/A_dice_w64_d5_s0/log.csv before restarting anything."
    exit 1
fi

run_stage task1
run_stage recover

echo ""
echo "########## all stages finished $(date '+%F %T') ##########"
"$PY" exp/gpu_queue.py task1 --pending
"$PY" exp/gpu_queue.py recover --pending
echo "(anything listed above never produced a final.pt -- check the logs)"
