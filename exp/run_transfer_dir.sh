#!/usr/bin/env bash
# Task 3's prerequisite: a direction head on each transfer dataset.
#
# exp/results/heldout_transfer already holds 36 SEGMENTATION runs (A_dice,
# H_aug, H_aug_clw, K_focal_aug x 3 seeds x STARE/VessMAP/HRF), so task 3's
# "apply the layer, no training" is true for the segmenters. It is not true
# for the FIELD: a tangent predictor trained on DRIVE has never seen HRF's
# resolution, and using the DRIVE one would test transfer of the predictor
# rather than transfer of the mechanism. Both are interesting; they are not
# the same question, and the paper's claim is about the mechanism.
#
# 18 runs: 2 direction arms x 3 datasets x 3 seeds. Seeds match the
# segmentation runs already there, so field and segmentation can be paired by
# seed exactly as they are on DRIVE.
#
# Radii stay in multiples of median vessel width. HRF is about six times
# DRIVE's resolution, and that unit is the whole reason task 3 is a test of
# the claim rather than a restatement of it.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/transfer_dir.log
NEED_MB=3500
ARMS=(A_dice_dir H_aug_dir)
SEEDS=(0 1 2)
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

if ! "$PY" exp/test_protocol.py >> "$LOG" 2>&1; then
    say "protocol selftest FAILED -- refusing to queue"; exit 1
fi
say "selftest passed"

# Queue behind the DRIVE direction heads: they are the headline and this is
# an ablation, and four training jobs on two cards finish later than three.
WAITED=0
while [ "$("$PY" exp/gpu_queue.py dirhead --pending --results exp/results/heldout | wc -l)" -gt 0 ]; do
    [ "$WAITED" -ge 144 ] && { say "dirhead unfinished after 12h; starting anyway"; break; }
    sleep 300
    WAITED=$(( WAITED + 1 ))
done
say "DRIVE direction heads done; starting transfer"

train_on () {   # train_on <dataset> <gpu> <offset>
    local DS="$1" GPU="$2" OFFSET="$3" ROOT="exp/results/heldout_transfer/$1"
    mkdir -p "$ROOT"
    for INDEX in "${!SEEDS[@]}"; do
        [ $(( INDEX % 2 )) -ne "$OFFSET" ] && continue
        for ARM in "${ARMS[@]}"; do
            local RUN="${ARM}_s${SEEDS[$INDEX]}"
            [ -f "$ROOT/$RUN/final.pt" ] && continue
            while true; do
                FREE=$(nvidia-smi --query-gpu=memory.free \
                       --format=csv,noheader,nounits -i "$GPU")
                [ "$FREE" -ge "$NEED_MB" ] && break
                echo "$(date '+%T') gpu$GPU ${FREE}MB, waiting" >> "$LOG"
                sleep 300
            done
            echo "--- $(date '+%F %T') $DS/$RUN gpu$GPU ---" | tee -a "$LOG"
            CUDA_VISIBLE_DEVICES="$GPU" "$PY" exp/train.py \
                --dataset "$DS" --results "$ROOT" --keep-epochs \
                --protocol heldout "$RUN" 2>&1 \
                | tail -3 | tee -a "$LOG" \
                || echo "!!! $DS/$RUN FAILED, continuing" | tee -a "$LOG"
        done
    done
}

for DS in stare vessmap hrf; do
    say "direction head on $DS"
    train_on "$DS" 0 0 & A=$!
    train_on "$DS" 1 1 & B=$!
    wait "$A"; wait "$B"
done
say "all done"
