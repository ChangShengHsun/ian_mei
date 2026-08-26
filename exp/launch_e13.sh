#!/usr/bin/env bash
# Start the night: E13's third capacity point on one card, the checkpoints its
# first two points need on the other.
#
# Waits for anything already training to finish first -- two jobs on one card
# finish later than one after the other (CLAUDE.md, measured), and a run
# started against a busy card just makes both slower.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python

echo "waiting for in-flight runs to finish..."
while pgrep -f 'exp/train\.py' > /dev/null; do sleep 20; done
echo "card is free, $(date '+%F %T')"

# The three 31M runs that finished BEFORE best.pt existed have no best
# checkpoint, and it cannot be reconstructed: ckpt.pt rolls forward and
# final.pt is epoch 100. Seed 0 appears in every comparison, so a hole there
# is the worst possible one. Move the old weights aside -- not deleted, and
# the run retrains into a complete set. rerun_path() sends the repeat's log to
# log_rerun.csv, so the finished log.csv stays as it is.
for RUN in A_dice_w64_d5_s0 B_cldice_w64_d5_s0 H_aug_w64_d5_s0; do
    DIR="exp/results/$RUN"
    if [ -f "$DIR/final.pt" ] && [ ! -f "$DIR/best.pt" ]; then
        mv "$DIR/final.pt" "$DIR/final_nobest.pt"
        rm -f "$DIR/ckpt.pt"     # a rolling checkpoint from the old code
        echo "$RUN: weights moved aside, will retrain with best.pt"
    fi
done

tmux kill-session -t e13 2>/dev/null
tmux new-session -d -s e13 -n curve-31M -c "$PWD"
tmux send-keys -t e13:curve-31M \
    'bash exp/run_queue.sh e13 0 1 0; bash exp/run_queue.sh recover_rest 0 2 0' C-m
tmux new-window -t e13 -n curve-narrow -c "$PWD"
tmux send-keys -t e13:curve-narrow \
    'bash exp/run_queue.sh curve 0 1 1; bash exp/run_queue.sh recover_rest 1 2 1' C-m
tmux new-window -t e13 -n gpus -c "$PWD"
tmux send-keys -t e13:gpus 'watch -n 10 nvidia-smi' C-m
echo "tmux session 'e13' started: curve-31M on GPU 0, curve-narrow on GPU 1"
