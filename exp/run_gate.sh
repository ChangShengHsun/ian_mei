#!/usr/bin/env bash
# The gate itself is now the thing under test.
#
# WHAT THE LAST TWO QUEUES TURNED UP. Going 3 -> 6 seeds moved eight cells of
# the cross-dataset table; going 6 -> 12 moved five more. But at 6 -> 12 the
# effect sizes did NOT move (median |effect| ratio 0.968) while |t| grew about
# as fast as sqrt(n) predicts (median 1.596 against 1.414). Three of the four
# HOLDS -> fails flips carried a LARGER t than the pass they replaced. That is
# not an effect settling down. It is calibration.decide's third condition --
# every per-seed difference positive -- getting strictly harder as seeds are
# added, while the mean and t conditions get easier.
#
# So "this cell passed the gate" is, in part, a statement about how few seeds
# were run. That is a property of the instrument, and it is the seventh
# artefact on the list. Phase 1 measures it without a single new training run
# by resampling the twelve seeds already on disk.
#
# Phase 2 buys the one thing resampling cannot: a real anchor past twelve.
# STARE is the cheapest of the three (7 fit images, ~8 min per run), so it
# gets seeds 12-23 and the seed curve gets a measured point at 24 instead of
# an extrapolated one.
#
# ANOTHER USER IS ON gpu1 (9.1 GB, 2026-09-03 13:03). wait_vram lines up
# behind them and never touches their processes; if only gpu0 is free this
# phase takes about six and a half hours instead of three and a quarter.
#
# PRE-REGISTERED 2026-09-03, before phase 1 has printed anything:
#   1. At least one cell reading HOLDS at twelve seeds passes fewer than 90%
#      of its 12-seed resamples. If every HOLDS cell reads 100%, the flips
#      were about WHICH seeds and not HOW MANY, and the non-monotonicity is a
#      theoretical worry rather than a live one -- which is also a result, and
#      the honest one to report.
#   2. Every `lower` cell in transfer_postproc reads 100% at every k from 3
#      up. Its effect sizes are an order of magnitude clear of the gate; if
#      even that comparison wobbles under resampling, no cell in this repo is
#      safe and the paper's claim has to be about the instrument alone.
#   3. Extending STARE to 24 moves at least one more cell. Under the stated
#      mechanism the all-positive rule keeps tightening, so HOLDS cells should
#      keep decaying. If nothing moves between 12 and 24, the decay has a
#      floor and we can name where it is.
#   4. Nothing moves fails -> HOLDS at 24. One that does would mean the effect
#      was real all along and every table here is underpowered at twelve.
#
# Waits for run_ten.sh: two writers on one csv is how duplicate rows happen.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/gate.log
NEED_MB=3500
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

WAITED=0
while pgrep -u "$USER" -f "exp/run_ten.sh" > /dev/null 2>&1; do
    sleep 120; WAITED=$(( WAITED + 120 ))
    [ "$WAITED" -ge 28800 ] && { say "run_ten.sh still up after 8h -- \
starting anyway"; break; }
done
say "run_ten.sh clear after ${WAITED}s"

for CHECK in exp/seed_stability.py exp/transfer_postproc.py; do
    if ! $PY "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed (draw reproducibility, sign-rule bite, 60-cell \
reconstruction against the shipped verdict)"

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

# ---------------------------------------------------------------- phase 1
say "phase 1: seed stability at twelve seeds"
$PY exp/seed_stability.py --report \
    > exp/results/seed_stability.12seeds.txt 2>&1
cp exp/results/seed_stability.12seeds.txt exp/results/seed_stability.txt
say "phase 1 done: exp/results/seed_stability.txt"

# ---------------------------------------------------------------- phase 2
STARE_WORK=()
for SEED in 12 13 14 15 16 17 18 19 20 21 22 23; do
    for ARM in A_dice H_aug H_aug_clw K_focal_aug; do
        STARE_WORK+=("${ARM}_s${SEED}")
    done
done
work_gpu () {   # work_gpu <gpu> <offset>
    local GPU="$1" OFFSET="$2" ROOT="exp/results/heldout_transfer/stare"
    for INDEX in "${!STARE_WORK[@]}"; do
        [ $(( INDEX % 2 )) -ne "$OFFSET" ] && continue
        local RUN="${STARE_WORK[$INDEX]}"
        [ -f "$ROOT/$RUN/final.pt" ] && continue
        while true; do
            FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
            [ "$FREE" -ge "$NEED_MB" ] && break
            echo "$(date '+%T') gpu$GPU ${FREE}MB free, waiting" >> "$LOG"
            sleep 300
        done
        echo "--- $(date '+%F %T') stare/$RUN gpu$GPU ---" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 "$PY" exp/train.py \
            --dataset stare --results "$ROOT" --keep-epochs \
            --protocol heldout "$RUN" 2>&1 \
            | tail -3 | tee -a "$LOG" \
            || echo "!!! stare/$RUN FAILED, continuing" | tee -a "$LOG"
    done
}
say "phase 2: ${#STARE_WORK[@]} STARE runs to twenty-four seeds"
work_gpu 0 0 >> "$LOG" 2>&1 &
GPU0=$!
work_gpu 1 1 >> "$LOG" 2>&1 &
GPU1=$!
wait "$GPU0" "$GPU1"
say "phase 2 done: STARE at twenty-four seeds"

# ---------------------------------------------------------------- phase 3
for NAME in transfer_calibration transfer_postproc seed_stability; do
    SRC="exp/results/${NAME}_verdict.txt"
    [ "$NAME" = seed_stability ] && SRC="exp/results/seed_stability.txt"
    [ -f "$SRC" ] && [ ! -f "${SRC%.txt}.12seeds.txt" ] \
        && cp "$SRC" "${SRC%.txt}.12seeds.txt"
done
say "phase 3: rescoring STARE at twenty-four seeds"
run_shards exp/transfer_calibration.py 4 stare
run_shards exp/transfer_postproc.py 4 stare
$PY exp/transfer_calibration.py --report \
    > exp/results/transfer_calibration_verdict.txt 2>&1
$PY exp/transfer_postproc.py --report \
    > exp/results/transfer_postproc_verdict.txt 2>&1
$PY exp/seed_stability.py --report > exp/results/seed_stability.txt 2>&1
say "phase 3 done: verdicts and the seed curve now reach k=24 on STARE"
say "all done"
