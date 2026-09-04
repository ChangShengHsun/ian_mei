#!/usr/bin/env bash
# Does the gate's seed-count decay generalise past STARE?
#
# WHAT run_gate.sh TURNED UP LAST NIGHT. STARE went 12 -> 24 seeds and exactly
# two cells moved, both HOLDS -> fails, both under erl_bridged vs A_dice at the
# shared 0.5 threshold:
#
#   H_aug_clw      +10.3% t 5.51 HOLDS  ->  +10.6% t 6.93 fails
#   K_focal_aug    +10.1% t 5.79 HOLDS  ->  +10.3% t 7.51 fails
#
# In both the effect got BIGGER and t got BIGGER, and the verdict went the
# other way. Nothing moved fails -> HOLDS. That is calibration.decide's third
# condition -- every per-seed difference positive -- tightening with n while
# the mean and t conditions loosen, which is the mechanism run_gate.sh
# pre-registered. It is now measured on one dataset with one real anchor.
#
# WHY THIS QUEUE. One dataset is an anecdote. HRF and VessMAP sit at twelve
# seeds; taking both to twenty-four costs 96 runs and buys two more measured
# k=24 points. It also buys the harder test below (prediction 2): whether
# seed_stability's resampling curve, which is computed from the seeds already
# on disk, actually PREDICTS what happens when real new seeds arrive. If it
# does not, artefact #7 is a description of the twelve seeds we have rather
# than a tool anyone else can use, and the paper has to say so.
#
# PRE-REGISTERED 2026-09-04, before a single run of this queue exists:
#   1. Nothing moves fails -> HOLDS on either dataset. This is the same
#      prediction run_gate.sh made for STARE and it held there. A single
#      fails -> HOLDS would mean the all-positive rule is not monotone in the
#      direction claimed, and the whole artefact needs restating.
#   2. The two cells still reading HOLDS at twelve seeds SURVIVE to 24:
#        hrf      split    H_aug_clw vs H_aug at 0.5   +3.5% t 8.46
#        vessmap  bridged  H_aug     vs A_dice at own  +1.9% t 3.62
#      Both read 100% at k=12 in seed_stability.txt (the HRF cell reads 100%
#      at every k from 3 up; the VessMAP one climbs 37% -> 100% between k=3
#      and k=8), while STARE's two casualties read 24% and 50% at k=12. So
#      this is the resampling curve making a falsifiable out-of-sample call.
#      If either dies at 24 despite reading 100% at 12, the curve does NOT
#      predict the anchor -- report that as the finding and stop selling the
#      curve as a planning tool.
#   3. Every `lower` cell in transfer_postproc stays HOLDS on both datasets.
#      They read 100% at every k on every dataset and their effects are an
#      order of magnitude clear of the gate. If one of these wobbles, no cell
#      in this repo is safe.
#   4. STARE's signature repeats: going 12 -> 24 leaves the effect sizes alone
#      and grows |t|. Concretely, median |effect| ratio in [0.90, 1.10] and
#      median |t| ratio in [1.15, 1.65] (sqrt(2) = 1.414 is the null). If the
#      effects SHRINK instead, the twelve-seed numbers were inflated and that
#      is a bigger problem than the gate.
#
# NOT IN THIS QUEUE, on purpose: DRIVE to 24 seeds. Ten arms, mostly 31M nets
# at ~0.27 h each, is about 32 GPU-hours -- a separate decision, not a rider.
#
# The lab's two 4070 Ti are SHARED. wait-for-VRAM only; never touch another
# user's processes.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/anchor.log
NEED_MB=3500
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for OTHER in run_gate.sh run_rescore.sh run_ten.sh run_paper.sh; do
    while pgrep -u "$USER" -f "exp/$OTHER" > /dev/null 2>&1; do
        say "$OTHER still up, waiting"; sleep 300
    done
done

for CHECK in exp/transfer_calibration.py exp/transfer_postproc.py \
             exp/seed_stability.py; do
    if ! $PY "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed"

# Keep last night's STARE-24 verdicts readable beside the new ones. The
# .12seeds.txt copies run_gate.sh made are already on disk and are not touched.
for NAME in transfer_calibration_verdict transfer_postproc_verdict; do
    SRC="exp/results/${NAME}.txt"
    [ -f "$SRC" ] && [ ! -f "exp/results/${NAME}.stare24.txt" ] \
        && cp "$SRC" "exp/results/${NAME}.stare24.txt"
done
[ -f exp/results/seed_stability.txt ] \
    && [ ! -f exp/results/seed_stability.stare24.txt ] \
    && cp exp/results/seed_stability.txt exp/results/seed_stability.stare24.txt
say "stare-24 verdicts preserved as *.stare24.txt"

# ---------------------------------------------------------------- phase 1
WORK=()
for DATASET in hrf vessmap; do
    for SEED in 12 13 14 15 16 17 18 19 20 21 22 23; do
        for ARM in A_dice H_aug H_aug_clw K_focal_aug; do
            WORK+=("$DATASET/${ARM}_s${SEED}")
        done
    done
done

work_gpu () {   # work_gpu <gpu> <offset>
    local GPU="$1" OFFSET="$2"
    for INDEX in "${!WORK[@]}"; do
        [ $(( INDEX % 2 )) -ne "$OFFSET" ] && continue
        local ITEM="${WORK[$INDEX]}"
        local DATASET="${ITEM%%/*}" RUN="${ITEM##*/}"
        local ROOT="exp/results/heldout_transfer/$DATASET"
        [ -f "$ROOT/$RUN/final.pt" ] && continue
        while true; do
            FREE=$(nvidia-smi --query-gpu=memory.free \
                   --format=csv,noheader,nounits -i "$GPU")
            [ "$FREE" -ge "$NEED_MB" ] && break
            echo "$(date '+%T') gpu$GPU ${FREE}MB free, waiting" >> "$LOG"
            sleep 300
        done
        echo "--- $(date '+%F %T') $ITEM gpu$GPU ---" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 "$PY" exp/train.py \
            --dataset "$DATASET" --results "$ROOT" --keep-epochs \
            --protocol heldout "$RUN" 2>&1 \
            | tail -3 | tee -a "$LOG" \
            || echo "!!! $ITEM FAILED, continuing" | tee -a "$LOG"
    done
}
say "phase 1: ${#WORK[@]} runs -- HRF and VessMAP to twenty-four seeds"
work_gpu 0 0 >> "$LOG" 2>&1 &
GPU0=$!
work_gpu 1 1 >> "$LOG" 2>&1 &
GPU1=$!
wait "$GPU0" "$GPU1"
say "phase 1 done"

# ---------------------------------------------------------------- phase 2
run_shards () {   # run_shards <script> <count> <args...>
    local SCRIPT="$1" COUNT="$2"; shift 2
    local PIDS=()
    for INDEX in $(seq 0 $(( COUNT - 1 ))); do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" "$SCRIPT" "$@" \
            --shard "$INDEX/$COUNT" \
            >> "exp/results/$(basename "$SCRIPT" .py)_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
}
say "phase 2: rescoring HRF and VessMAP"
run_shards exp/transfer_calibration.py 4 hrf vessmap
run_shards exp/transfer_postproc.py 4 hrf vessmap
$PY exp/transfer_calibration.py --report \
    > exp/results/transfer_calibration_verdict.txt 2>&1
$PY exp/transfer_postproc.py --report \
    > exp/results/transfer_postproc_verdict.txt 2>&1
$PY exp/seed_stability.py --report > exp/results/seed_stability.txt 2>&1
say "phase 2 done: all three transfer datasets now anchored at k=24"
say "all done"
