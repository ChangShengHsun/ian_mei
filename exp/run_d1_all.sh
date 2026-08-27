#!/usr/bin/env bash
# The D1 line, rebuilt: sweep the layer's reach instead of choosing it, then
# transfer.
#
# WHY A SWEEP. The first D-B attempt handed the layer one geometry, taken from
# the post-hoc ceiling at its tightest Dice budget. That built a 5x5 kernel
# holding three pixels per orientation. The gate opened for a real direction
# field (0.15) and shut for a random one (0.03), so the network could tell
# them apart -- and a three-pixel operator still had nothing to give. The Dice
# constraint had been applied twice: once when choosing the reach, and again
# by the training loss, which is what the gate exists to handle.
#
#   d1b  three reaches x two bases x {real field, random field} x 6 seeds = 72
#   d1f  the same three reaches crossed with the centreline-weighted loss = 36
#   then transfer: train STARE, VessMAP and HRF on themselves and ask whether
#   the winning reach is the same multiple of vessel width on each.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/d1_all.log
SWEEP=exp/results/selection_sweep
export OMP_NUM_THREADS=4
NEED_MB=3500
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }
both () {
    bash exp/run_task.sh "$1" 0 0 2 "$SWEEP" 1 & local a=$!
    bash exp/run_task.sh "$1" 1 1 2 "$SWEEP" 1 & local b=$!
    wait "$a"; wait "$b"
}

if ! "$PY" exp/gpu_queue.py --selftest >> "$LOG" 2>&1; then
    say "gpu_queue selftest FAILED -- refusing to queue"; exit 1
fi
say "queue selftest passed"

for PASS in 1 2; do
    for STAGE in d1b d1f; do
        LEFT=$("$PY" exp/gpu_queue.py "$STAGE" --pending --results "$SWEEP" \
               | wc -l)
        [ "$LEFT" -eq 0 ] && { say "$STAGE nothing pending"; continue; }
        say "pass $PASS: $STAGE, $LEFT run(s) pending"
        both "$STAGE"
    done
done

say "rescoring the sweep (no arguments: sweep_score OVERWRITES)"
cp "$SWEEP/checkpoint_scores.csv" "$SWEEP/checkpoint_scores.pre_reach.csv"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py >> "$LOG" 2>&1; then
    say "rescore OK"
    CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_reach.py \
        > exp/results/reach_summary.txt 2>&1
    cat exp/results/reach_summary.txt | tee -a "$LOG"
else
    say "rescore FAILED -- restoring"
    cp "$SWEEP/checkpoint_scores.pre_reach.csv" "$SWEEP/checkpoint_scores.csv"
fi

# ---------------------------------------------------------------- transfer
ARMS=(K_focal_aug A_dice)
SEEDS=(0 1 2)
train_on () {   # train_on <dataset> <gpu> <offset>
    local DS="$1" GPU="$2" OFFSET="$3" OUT="exp/results/transfer/$1"
    mkdir -p "$OUT"
    for INDEX in "${!SEEDS[@]}"; do
        [ $(( INDEX % 2 )) -ne "$OFFSET" ] && continue
        for ARM in "${ARMS[@]}"; do
            local RUN="${ARM}_s${SEEDS[$INDEX]}"
            [ -f "$OUT/$RUN/final.pt" ] && continue
            while true; do
                FREE=$(nvidia-smi --query-gpu=memory.free \
                       --format=csv,noheader,nounits -i "$GPU")
                [ "$FREE" -ge "$NEED_MB" ] && break
                echo "$(date '+%T') gpu$GPU ${FREE}MB, waiting" >> "$LOG"
                sleep 300
            done
            echo "--- $(date '+%F %T') $DS/$RUN gpu$GPU ---" | tee -a "$LOG"
            CUDA_VISIBLE_DEVICES="$GPU" "$PY" exp/train.py \
                --dataset "$DS" --results "$OUT" "$RUN" 2>&1 \
                | tail -3 | tee -a "$LOG" \
                || echo "!!! $DS/$RUN FAILED, continuing" | tee -a "$LOG"
        done
    done
}
for DS in stare vessmap hrf; do
    say "transfer training on $DS"
    train_on "$DS" 0 0 & A=$!
    train_on "$DS" 1 1 & B=$!
    wait "$A"; wait "$B"
done

say "transfer geometry sweep"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/transfer_ceiling.py >> "$LOG" 2>&1; then
    CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_transfer.py \
        > exp/results/transfer_summary.txt 2>&1
    cat exp/results/transfer_summary.txt | tee -a "$LOG"
else
    say "transfer sweep FAILED"
fi
say "all done"
