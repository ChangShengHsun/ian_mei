#!/usr/bin/env bash
# Phase 3, redone: train each dataset's own models, then ask whether the
# correction's geometry is the same multiple of vessel width everywhere.
#
# The first attempt used DRIVE-trained models on every dataset and measured
# nothing: raw traced run length was 3.4% on HRF and 0.0% on VessMAP, and a
# correction cannot be measured on a prediction with nothing to correct. That
# run measured the transfer of the SEGMENTATION, which is E4's question.
#
# Waits for the D-B queue to release the cards. Never kills anything.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/d1_phase3b.log
export OMP_NUM_THREADS=4
NEED_MB=3500
ARMS=(K_focal_aug A_dice)
SEEDS=(0 1 2)
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

say "waiting for the D-B queue to finish with the cards"
until [ ! -f exp/results/d1_phase2b.log ] || \
      grep -q "^=== .* done ===$" exp/results/d1_phase2b.log; do
    sleep 300
done
say "cards released"

train_on () {   # train_on <dataset> <gpu> <seed-offset>
    local DS="$1" GPU="$2" OFFSET="$3"
    local OUT="exp/results/transfer/$DS"
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
                | tail -4 | tee -a "$LOG" \
                || echo "!!! $DS/$RUN FAILED, continuing" | tee -a "$LOG"
        done
    done
}

for DS in stare vessmap hrf; do
    say "training on $DS"
    train_on "$DS" 0 0 & A=$!
    train_on "$DS" 1 1 & B=$!
    wait "$A"; wait "$B"
done
say "transfer training done"

say "sweeping the geometry on every dataset"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/transfer_ceiling.py >> "$LOG" 2>&1; then
    CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_transfer.py \
        > exp/results/transfer_summary.txt 2>&1
    cat exp/results/transfer_summary.txt | tee -a "$LOG"
else
    say "sweep FAILED"
fi
say "phase 3 done"
