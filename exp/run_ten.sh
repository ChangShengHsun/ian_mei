#!/usr/bin/env bash
# Ten hours, aimed at the one thing the 6-seed table cannot answer about itself.
#
# WHY THIS AND NOT MORE OF THE SAME. run_seeds6.sh pre-registered, on
# 2026-09-02, that "no arm is expected to move from HOLDS to fails purely by
# adding seeds; if one does, the 3-seed pass was noise and every 3-seed pass
# in the table is suspect." EIGHT CELLS MOVED. HRF/H_aug_clw went +3.2 HOLDS
# to +2.3 fails and every STARE and VessMAP cell against A_dice collapsed with
# effect sizes down about 40%. The written consequence was that the 6-seed
# table becomes primary. The consequence NOT written, and the one a reviewer
# reaches first, is that nothing in that argument stops at six: if three was
# underpowered by a factor that flipped eight cells, six is a claim about
# power that six seeds cannot check. Twelve can.
#
# Phase 0 doubles every transfer arm to twelve seeds. 72 runs, both cards,
# about six and a half hours at the 8-12 min per run measured in seeds6.log.
#
# Phase 1 opens the cross-dataset reading of the newest artefact while those
# train. exp/transfer_postproc.py, written and selftested today BEFORE it
# scored anything, asks off DRIVE what composition.py's `lower` column asks on
# it: priced against simply moving the threshold down, does the
# post-processing layer -- whole-mask or endpoint-restricted -- buy anything?
# On DRIVE the answer was no for the layer and, after the 2026-09-02 controls,
# no for the field. If it is also no on STARE, HRF and VessMAP, that is a
# property of the measurement rather than of one 20-image dataset.
#
# Phase 3 trains the DRIVE direction heads and H_aug_w64_d5 to twelve seeds.
# composition.py caps every arm at the seeds its FIELD arm has, so the DRIVE
# table is held at six by H_aug_dir alone even though nine of its ten arms
# have twelve checkpoints sitting on disk. Training is the expensive half and
# it is what the idle card is for; the scoring is CPU and phase 4 starts it.
#
# PRE-REGISTERED 2026-09-03, before any of these rows exists:
#   1. At twelve seeds the cross-dataset table stays 0 of 12. A cell that
#      re-appears means the 6-seed collapse was itself noise and the honest
#      report is that this comparison does not converge at any seed count we
#      can afford -- which is a finding about the measurement, not a rescue.
#   2. Effect sizes shrink again from 6 to 12, but by less than the ~40% seen
#      from 3 to 6. Shrinking by as much would say the sequence has not
#      settled and no seed count in reach is enough.
#   3. transfer_postproc: `lower` beats `isotropic` on all three datasets
#      under both conventions. This is the DRIVE 10-of-10 reproducing; if it
#      fails, the DRIVE result was dataset-specific and artefact six is dead.
#   4. transfer_postproc: `endpoint_iso` also loses to `lower`. If instead it
#      survives off DRIVE while dying on DRIVE, the endpoint restriction is
#      worth a second look and this queue found the one live lead.
#   5. STARE picks its settings on THREE dev images. Any STARE cell that
#      disagrees with both other datasets is to be read as a split-size
#      artefact first and a dataset difference second.
#
# Phase 4 is deliberately LAST and will run past the ten hours. Every stage
# resumes by key, so killing it costs nothing but the current run.
#
# Does not touch run_lower.sh's tables until that queue has exited: two
# writers on one csv is how duplicate rows happen.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/ten.log
NEED_MB=3500
XFER_SEEDS=(6 7 8 9 10 11)
XFER_ARMS=(A_dice H_aug H_aug_clw K_focal_aug)
XFER_SETS=(stare vessmap hrf)
DRIVE_RUNS=()
for SEED in 6 7 8 9 10 11; do
    for ARM in H_aug_dir A_dice_dir H_aug_w64_d5; do
        DRIVE_RUNS+=("${ARM}_s${SEED}")
    done
done
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in exp/transfer_postproc.py exp/transfer_calibration.py \
             exp/composition.py; do
    if ! $PY "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed (partition, split disjointness, operator promises)"

wait_vram () {   # wait_vram <gpu>
    while true; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1")
        [ "$FREE" -ge "$NEED_MB" ] && break
        echo "$(date '+%T') gpu$1 ${FREE}MB free, waiting" >> "$LOG"
        sleep 300
    done
}

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

# One card's share of a flat work list, taken by index parity. seeds6.sh split
# by SEED index instead, which left a card idle for a third of every dataset
# whenever the seed count was odd; a flat list keeps both busy to the end.
work_gpu () {   # work_gpu <gpu> <offset> <"dataset:run" ...>
    local GPU="$1" OFFSET="$2"; shift 2
    local ITEMS=("$@")
    for INDEX in "${!ITEMS[@]}"; do
        [ $(( INDEX % 2 )) -ne "$OFFSET" ] && continue
        local DS="${ITEMS[$INDEX]%%:*}" RUN="${ITEMS[$INDEX]#*:}"
        local ROOT="exp/results/heldout" ARGS=()
        if [ "$DS" != "drive" ]; then
            ROOT="exp/results/heldout_transfer/$DS"
            ARGS=(--dataset "$DS")
        fi
        mkdir -p "$ROOT"
        [ -f "$ROOT/$RUN/final.pt" ] && continue
        wait_vram "$GPU"
        echo "--- $(date '+%F %T') $DS/$RUN gpu$GPU ---" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 "$PY" exp/train.py \
            "${ARGS[@]+"${ARGS[@]}"}" --results "$ROOT" --keep-epochs \
            --protocol heldout "$RUN" 2>&1 \
            | tail -3 | tee -a "$LOG" \
            || echo "!!! $DS/$RUN FAILED, continuing" | tee -a "$LOG"
    done
}

# ---------------------------------------------------------------- phase 0
# Ordered dataset-major so a kill leaves whole datasets finished rather than
# every dataset two thirds done: an 8-seed table on one dataset and a 12-seed
# table on another cannot be printed side by side.
XFER_WORK=()
for DS in "${XFER_SETS[@]}"; do
    for SEED in "${XFER_SEEDS[@]}"; do
        for ARM in "${XFER_ARMS[@]}"; do
            XFER_WORK+=("$DS:${ARM}_s${SEED}")
        done
    done
done
say "phase 0: ${#XFER_WORK[@]} transfer runs to twelve seeds, two cards"
work_gpu 0 0 "${XFER_WORK[@]}" >> "$LOG" 2>&1 &
GPU0=$!
work_gpu 1 1 "${XFER_WORK[@]}" >> "$LOG" 2>&1 &
GPU1=$!

# ---------------------------------------------------------------- phase 1
# run_lower.sh owns composition/postproc/threshold_control until it exits.
# transfer_postproc writes different csvs, but its four shards would fight
# run_lower's four for the same cores, so it lines up behind rather than
# beside.
WAITED=0
while pgrep -u "$USER" -f "exp/run_lower.sh" > /dev/null 2>&1; do
    sleep 120; WAITED=$(( WAITED + 120 ))
    [ "$WAITED" -ge 21600 ] && { say "run_lower.sh still up after 6h -- \
going ahead on transfer csvs only"; break; }
done
say "phase 1: transfer_postproc at six seeds (${WAITED}s waited for run_lower)"
run_shards exp/transfer_postproc.py 4
$PY exp/transfer_postproc.py --report \
    > exp/results/transfer_postproc_verdict.6seeds.txt 2>&1
say "phase 1 done: six-seed cross-dataset verdict written"

# ---------------------------------------------------------------- phase 2
wait "$GPU0" "$GPU1"
say "phase 0 done: transfer training finished"
for NAME in transfer_calibration transfer_postproc; do
    SRC="exp/results/${NAME}_verdict.txt"
    [ -f "$SRC" ] && [ ! -f "exp/results/${NAME}_verdict.6seeds.txt" ] \
        && cp "$SRC" "exp/results/${NAME}_verdict.6seeds.txt"
done
for DS in "${XFER_SETS[@]}"; do
    say "phase 2: transfer calibration at twelve seeds: $DS"
    run_shards exp/transfer_calibration.py 4 "$DS"
done
say "phase 2: transfer_postproc at twelve seeds"
run_shards exp/transfer_postproc.py 4
$PY exp/transfer_calibration.py --report \
    > exp/results/transfer_calibration_verdict.txt 2>&1
$PY exp/transfer_postproc.py --report \
    > exp/results/transfer_postproc_verdict.txt 2>&1
say "phase 2 done: twelve-seed cross-dataset verdicts written"

# ---------------------------------------------------------------- phase 3
say "phase 3: ${#DRIVE_RUNS[@]} DRIVE runs to twelve seeds, two cards"
DRIVE_WORK=()
for RUN in "${DRIVE_RUNS[@]}"; do DRIVE_WORK+=("drive:$RUN"); done
work_gpu 0 0 "${DRIVE_WORK[@]}" >> "$LOG" 2>&1 &
DGPU0=$!
work_gpu 1 1 "${DRIVE_WORK[@]}" >> "$LOG" 2>&1 &
DGPU1=$!
wait "$DGPU0" "$DGPU1"
say "phase 3 done: DRIVE direction heads and H_aug_w64_d5 at twelve seeds"

# ---------------------------------------------------------------- phase 4
# PAST THE TEN HOURS BY DESIGN, and resumable to the (config, seed) if killed.
# threshold_control first: composition reads its base threshold from that
# table, and a table built on six seeds under a comparison run at twelve is
# the kind of mismatch that shows up as nothing at all.
# postproc_ceiling is NOT here. It is five hours at six seeds, composition
# supersedes it as the headline, and running it would push this queue past
# fifteen. It stays at six seeds and the verdict says so.
for NAME in composition threshold_control; do
    SRC="exp/results/${NAME}_verdict.txt"
    [ -f "$SRC" ] && [ ! -f "exp/results/${NAME}_verdict.6seeds.txt" ] \
        && cp "$SRC" "exp/results/${NAME}_verdict.6seeds.txt"
done
say "phase 4: threshold_control at twelve seeds"
run_shards exp/threshold_control.py 4 --dev
run_shards exp/threshold_control.py 4
say "phase 4: composition at twelve seeds"
run_shards exp/composition.py 4 --dev
run_shards exp/composition.py 4
say "phase 4: lower comparator at twelve seeds"
run_shards exp/composition.py 4 --lower --dev
run_shards exp/composition.py 4 --lower
$PY exp/threshold_control.py --report \
    > exp/results/threshold_control_verdict.txt 2>&1
$PY exp/composition.py --report > exp/results/composition_verdict.txt 2>&1
say "phase 4 done: DRIVE verdicts at twelve seeds"
say "all done"
