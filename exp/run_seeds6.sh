#!/usr/bin/env bash
# Lift the cross-dataset calibration table from 3 seeds to 6.
#
# WHY THIS AND NOT SOMETHING ELSE. transfer_calibration_verdict.txt states
# its own weakness in its header: "THREE SEEDS is the gate's minimum -- every
# pass here is narrow." That table is now load-bearing. It is what turns "no
# topology loss fixes connectivity on DRIVE" into a claim about four
# datasets, and the K_focal_aug row -- 0 of 6 cells, three datasets, two
# conventions -- is the paper's central negative result. A negative result
# resting on the gate's bare minimum is the first thing a reviewer pushes on,
# and the answer to "underpowered" is seeds, not argument.
#
# 36 runs: A_dice, H_aug, H_aug_clw, K_focal_aug x seeds 3,4,5 x STARE,
# VessMAP, HRF. Seeds extend the existing 0,1,2 and bring the transfer
# datasets level with DRIVE's 6.
#
# Then 18 more, LAST and separately: the two _dir arms at the same new seeds.
# These are NOT needed for the calibration table. They are the field for a
# cross-dataset reading of the composition result, and that result only
# exists if endpoint_iso and endpoint_shuf come back dead. Ordering them last
# means killing this queue after phase 1 costs nothing if the controls say
# the endpoint win was the restriction and not the field.
#
# PRE-REGISTERED 2026-09-02, before any of these runs exists, so the wider
# table cannot be read after the fact:
#   - K_focal_aug at 6 seeds stays 0 of 6 cells. If any cell flips to HOLDS,
#     the 3-seed table was underpowered and the paper must report the 6-seed
#     version as primary -- the narrow one cannot be kept because it is
#     friendlier.
#   - H_aug_clw keeps HRF (+3.2 / +4.8) and stays short of the gate on STARE
#     and VessMAP. A clw that passes everywhere at 6 seeds is a real finding
#     and changes the story from "nothing survives" to "one family does".
#   - Nothing here can rescue the shared-0.5 column. That gap is calibration
#     -- STARE peaks at 0.74-0.81 -- and seeds do not move a peak.
#   - Direction of the effect is fixed too: no arm is expected to move from
#     HOLDS to fails purely by adding seeds; if one does, the 3-seed pass was
#     noise and every 3-seed pass in the table is suspect.
#
# RUNS BESIDE THE CPU QUEUE, as run_night.sh did: these nets are dominated by
# the numpy patch pipeline, not the card, so OMP is capped at 3 per job to
# leave run_shardfix.sh's four composition shards their cores. Phase 3 waits
# for that queue rather than competing with it.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/seeds6.log
NEED_MB=3500
SEEDS=(3 4 5)
CAL_ARMS=(A_dice H_aug H_aug_clw K_focal_aug)
DIR_ARMS=(A_dice_dir H_aug_dir)
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

if ! "$PY" exp/test_protocol.py >> "$LOG" 2>&1; then
    say "protocol selftest FAILED -- refusing to queue"; exit 1
fi
say "protocol selftest passed"

train_on () {   # train_on <dataset> <gpu> <offset> <arm...>
    local DS="$1" GPU="$2" OFFSET="$3"; shift 3
    local ARMS=("$@") ROOT="exp/results/heldout_transfer/$DS"
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
            CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=3 "$PY" exp/train.py \
                --dataset "$DS" --results "$ROOT" --keep-epochs \
                --protocol heldout "$RUN" 2>&1 \
                | tail -3 | tee -a "$LOG" \
                || echo "!!! $DS/$RUN FAILED, continuing" | tee -a "$LOG"
        done
    done
}

# Phase 1 -- the four arms the calibration table is made of.
for DS in stare vessmap hrf; do
    say "phase 1: calibration arms on $DS, seeds ${SEEDS[*]}"
    train_on "$DS" 0 0 "${CAL_ARMS[@]}" & A=$!
    train_on "$DS" 1 1 "${CAL_ARMS[@]}" & B=$!
    wait "$A"; wait "$B"
done
say "phase 1 done: 36 segmentation runs"

# Phase 2 -- the field, only useful if the endpoint controls survive.
for DS in stare vessmap hrf; do
    say "phase 2: direction heads on $DS, seeds ${SEEDS[*]}"
    train_on "$DS" 0 0 "${DIR_ARMS[@]}" & A=$!
    train_on "$DS" 1 1 "${DIR_ARMS[@]}" & B=$!
    wait "$A"; wait "$B"
done
say "phase 2 done: 18 direction runs"

# Phase 3 -- re-read the table at 6 seeds. Behind the CPU queue, not beside
# it: composition holds the cell that decides whether we have a method.
WAITED=0
while pgrep -u "$USER" -f "run_shardfix.sh" >/dev/null; do
    [ "$WAITED" -ge 96 ] && { say "shardfix unfinished after 8h; starting anyway"; break; }
    sleep 300
    WAITED=$(( WAITED + 1 ))
done
say "CPU queue clear"

# Superseded, not deleted -- the 3-seed table is the pre-registration's
# control and the paper may have to show both.
cp exp/results/transfer_calibration_verdict.txt \
   exp/results/transfer_calibration_verdict.3seeds.txt 2>/dev/null || true

for DS in stare vessmap hrf; do
    say "transfer calibration at 6 seeds: $DS"
    PIDS=()
    for INDEX in 0 1 2; do
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$PY" \
            exp/transfer_calibration.py "$DS" --shard "$INDEX/3" \
            >> "exp/results/transfer_calibration_${INDEX}.log" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
done
"$PY" exp/transfer_calibration.py --report \
    > exp/results/transfer_calibration_verdict.txt 2>&1
say "6-seed verdict written"
say "all done"
