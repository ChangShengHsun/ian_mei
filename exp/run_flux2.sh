#!/usr/bin/env bash
# Retrain the flux arms WITH the band channel, so the readout can exist.
#
# WHY A SECOND FAMILY. exp/run_flux.sh trained 48 two-channel arms and they
# answered the question they could: the head learns where the mask is blind.
# On centreline the mask MISSES, mean |displacement error| is 2.0-2.3 px
# against 5.9-7.8 px for the flux you get free by skeletonising the mask --
# 12 of 12 seeds on all four arms, 48 of 48. Pre-registered prediction 1 held
# unanimously.
#
# WHAT THEY COULD NOT ANSWER, and why. The decode is a vote: banded pixels
# point somewhere and a centreline pixel is one enough pixels point at. The
# band was defined by distance to the TRUE skeleton, so at inference there is
# nothing to say which pixels may vote. Measured on a trained two-channel
# head, the magnitude carries none of that information:
#
#   inside the band    507255 px   mean |flux| 2.24   99.6% under the radius
#   outside, in FOV    687283 px   mean |flux| 2.09   99.3% under the radius
#
# A magnitude threshold would admit 1.3x as many false voters as true ones.
# Using the predicted MASK as the band instead would leave the vessels the
# mask misses with no voters at all -- the one thing this line exists to fix.
# So the band has to be predicted, and that is a third channel and a retrain.
# The older name is not reused: those 48 checkpoints are on disk and
# build_model must keep producing the net they were saved from.
#
# PRE-REGISTERED 2026-09-13, before any _flux2 run exists:
#   1. The band channel reaches Dice > 0.85 against the true band on dev.
#      The band is a dilated centreline, a far easier target than the vessel
#      itself; if a 117k net cannot fit it, the representation is not
#      learnable at this capacity and the readout is dead before it is built.
#   2. Prediction 1 of run_flux.sh survives the extra term: on centreline the
#      mask misses, the head still beats the free reference on at least 10 of
#      12 seeds per arm. Adding a second task can cost the first one; if it
#      does, the band channel is not free and the trade has to be reported.
#   3. `_flux2` does NOT beat its namesake on segmentation through the gate.
#      Registered as a null for the same reason `_flux` was: the mask head
#      can still ignore the geometry. `_flux` came in at +0.1% to +2.9%, all
#      eight comparisons positive, none through the gate; if `_flux2` does
#      pass, the band term did it and that is a finding about the auxiliary
#      loss rather than about the readout.
#   4. The decode from a TRAINED field recovers at least half the centreline
#      the mask misses, at under 1.5x the mask's foreground. This is the
#      first test of the readout itself, and the first that can kill the
#      line: flux_ceiling showed the decode is robust to 2 px of isotropic
#      jitter, but a real head's error is structured and concentrated exactly
#      where the mask is blind.
#
# Four arms x twelve seeds at base width 16, about six minutes each.
# The two 4070 Ti are SHARED: wait for VRAM, never touch another user's work.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/flux2.log
NEED_MB=3500
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for OTHER in run_flux.sh run_deep.sh run_survival.sh run_gaps.sh run_hyst.sh; do
    while pgrep -u "$USER" -f "exp/$OTHER" > /dev/null 2>&1; do
        say "$OTHER still up, waiting"; sleep 300
    done
done
say "no other queue running"

if ! $PY exp/test_flux.py >> "$LOG" 2>&1; then
    say "test_flux.py FAILED -- refusing to start"; exit 1
fi
say "test_flux passed (both families build their own head; the band channel \
is trained; the dihedral check still bites)"

WORK=()
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
    for ARM in A_dice_flux2 H_aug_flux2 H_aug_clw16_flux2 K_focal_aug_flux2; do
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
say "phase 1: ${#WORK[@]} flux2 runs"
work_gpu 0 0 >> "$LOG" 2>&1 &
GPU0=$!
work_gpu 1 1 >> "$LOG" 2>&1 &
GPU1=$!
wait "$GPU0" "$GPU1"
say "phase 1 done"

say "phase 2: the mask readout, through the usual scoring"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY exp/frontier.py --dev \
    >> "$LOG" 2>&1 || say "!!! frontier --dev FAILED"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY exp/frontier.py \
    >> "$LOG" 2>&1 || say "!!! frontier FAILED"
say "phase 2 done"
say "all done"
