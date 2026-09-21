#!/usr/bin/env bash
# Train the flux arms. The first positive signal on the method line -- but a
# signal about the FIELD, not yet about the segmentation.
#
# WHERE THIS COMES FROM. hysteresis.py closed the post-hoc map, so the only
# square left is changing what the model predicts. Geometry has been tried in
# this repo three ways: auxiliary head (D1 -- learned its target, did not
# help), post-hoc operator (D-B -- dead, beaten by its shuffled control), and
# loss weight (D-E = clw -- works, and is prior art). Geometry as the OUTPUT
# is untried, and D1's diagnosis (an auxiliary head can be ignored by the mask
# head) is the thing this line has to escape.
#
# WHAT THE SMOKE RUN SHOWED, A_dice_flux_s0 at its dev-chosen epoch 20, on the
# 5 dev images. Mean |displacement error| inside the band:
#
#                              all band   mask FINDS   mask MISSES
#   flux head                    1.097       0.828        2.133
#   zero field (degenerate)      2.514
#   from its own mask (free)     1.931       0.853        6.081
#
# The head beats both references, and the interesting column is the last one:
# on centreline the mask does NOT predict, the head is at 2.1 px against the
# mask-derived reference's 6.1. It knows roughly where vessels are that its
# own segmentation misses. That is the hypothesis, and it is the first time
# anything on this line has produced a number pointing the right way.
#
# WHAT IT DOES NOT SHOW, and why this queue is training and not concluding:
# knowing the displacement is not the same as the DECODE recovering the
# centreline. The decode is a vote, and a field that is roughly right but
# diffuse produces no centreline at all. That is measured by the readout
# script, not assumed here. One run, one arm, one epoch, five images.
#
# PRE-REGISTERED 2026-09-07, before any of these 48 runs exists:
#   1. Every arm's flux head beats its own mask-derived reference on the
#      centreline the mask misses, on at least 10 of 12 seeds. The smoke run
#      showed 2.1 against 6.1 on the weakest arm; if that does not survive
#      other arms and other seeds it was a fluke of A_dice_s0.
#   2. `_flux` does NOT beat its namesake on segmentation through the gate.
#      This is D1's result and D1's diagnosis says it should repeat: the mask
#      head can still ignore the flux head. Predicting the null so that a
#      surprise is a real surprise, and so the readout -- not the auxiliary
#      loss -- is what gets the credit if anything works.
#   3. The gap in prediction 1 is larger for the arms whose masks miss more.
#      A_dice misses 9.2% of centreline below p=0.01 and H_aug_clw16 far less,
#      so if the head is genuinely supplying what the mask lacks, the weaker
#      arm should gain more.
#
# Four arms x twelve seeds, all at base width 16, about six minutes each.
# The two 4070 Ti are SHARED: wait for VRAM, never touch another user's work.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/flux.log
NEED_MB=3500
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for OTHER in run_deep.sh run_survival.sh run_gaps.sh run_hyst.sh run_anchor.sh; do
    while pgrep -u "$USER" -f "exp/$OTHER" > /dev/null 2>&1; do
        say "$OTHER still up, waiting"; sleep 300
    done
done
say "no other queue running"

if ! $PY exp/test_flux.py >> "$LOG" 2>&1; then
    say "test_flux.py FAILED -- refusing to start"; exit 1
fi
say "test_flux passed (the dihedral check bites: 1.0000 / 1.0000 / 0.4307)"

WORK=()
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
    for ARM in A_dice_flux H_aug_flux H_aug_clw16_flux K_focal_aug_flux; do
        WORK+=("${ARM}_s${SEED}")
    done
done

work_gpu () {   # work_gpu <gpu> <offset>
    local GPU="$1" OFFSET="$2" ROOT="exp/results/heldout"
    for INDEX in "${!WORK[@]}"; do
        [ $(( INDEX % 2 )) -ne "$OFFSET" ] && continue
        local RUN="${WORK[$INDEX]}"
        [ -f "$ROOT/$RUN/final.pt" ] && continue
        while true; do
            FREE=$(nvidia-smi --query-gpu=memory.free \
                   --format=csv,noheader,nounits -i "$GPU")
            [ "$FREE" -ge "$NEED_MB" ] && break
            echo "$(date '+%T') gpu$GPU ${FREE}MB free, waiting" >> "$LOG"
            sleep 300
        done
        echo "--- $(date '+%F %T') $RUN gpu$GPU ---" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 "$PY" exp/train.py \
            --results "$ROOT" --keep-epochs --protocol heldout "$RUN" 2>&1 \
            | tail -3 | tee -a "$LOG" \
            || echo "!!! $RUN FAILED, continuing" | tee -a "$LOG"
    done
}
say "phase 1: ${#WORK[@]} flux runs"
work_gpu 0 0 >> "$LOG" 2>&1 &
GPU0=$!
work_gpu 1 1 >> "$LOG" 2>&1 &
GPU1=$!
wait "$GPU0" "$GPU1"
say "phase 1 done"

# Scored by the ordinary machinery, so a _flux arm sits in the same tables as
# every other arm and its segmentation can be read against its namesake with
# no new code. The readout that decodes the field is a separate script and a
# separate question.
say "phase 2: the mask readout, through the usual scoring"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY exp/frontier.py --dev \
    >> "$LOG" 2>&1 || say "!!! frontier --dev FAILED"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY exp/frontier.py \
    >> "$LOG" 2>&1 || say "!!! frontier FAILED"
say "phase 2 done"
say "all done"
