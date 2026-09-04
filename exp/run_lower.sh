#!/usr/bin/env bash
# The comparator composition.py was missing, plus the seed that table is short.
#
# WHY NOW. The endpoint controls came back on 2026-09-02 and killed the
# direction claim: `endpoint_shuf` (endpoints kept, field replaced by noise)
# matched or beat `endpoint` in 7 of 10 arms under convention B. So what the
# endpoint arm bought was the RESTRICTION, not the field.
#
# That leaves the restriction itself untested against the thing that already
# beat the whole post-processing layer 10 of 10 in threshold_control: moving
# the threshold down. Every source in composition.py spends its Dice budget on
# morphology. None of them spends it on the threshold. Until that column
# exists, "growing from endpoints helps" is a claim measured only against
# operators that were already known to lose.
#
# WHAT THE NEW COLUMN COSTS. Nothing to train: the same forward pass already
# being computed is thresholded again at every value below the arm's base.
#
# THE GRID HAD TO BE EXTENDED, and this is a finding on its own. frontier's
# grid stops at 0.10, and the base threshold picked at the 0.02 budget ALREADY
# SITS on that floor for A_dice, H_aug and H_aug_w64_d5. Offering those arms
# only the standard grid would have handed them an empty comparator, and the
# table would have read "thresholding has nothing left to give" when the true
# cause was that the grid ran out. composition.EXTENDED reaches 0.01.
#
# THE SEED. H_aug_w64_d5 rests on 5 seeds while the other nine arms have 6,
# and no verdict file says so. Phase 0 trains s5 on an idle card beside the
# CPU work; phases 2-3 top up every table that reports that arm.
#
# PRE-REGISTERED 2026-09-03, before any of these rows exists:
#   1. `lower` matches or beats `endpoint` at the -0.02 budget in at least 6
#      of 10 arms under convention B. If it does, the endpoint restriction is
#      not a method either and the post-processing layer is a budget illusion
#      end to end -- the sixth artefact.
#   2. For the three floor-limited arms the winning `lower` threshold comes
#      from the EXTENDED grid, below 0.10. If instead they can afford nothing
#      below the floor, their curve is already at its knee and those three are
#      the only place left where an operator could still earn its Dice.
#   3. `lower` beats `shuffled` and `isotropic` in every arm. This one is an
#      instrument check, not a finding: if it fails, the new pass is wrong and
#      the rest of the table must not be read.
#   4. Adding H_aug_w64_d5_s5 flips no HOLDS/fails cell for that arm. If it
#      flips one, the 5-seed row was noise -- and the 6-seed rows beside it
#      are only one seed better.
#
# CPU ONLY except phase 0. Old verdicts kept as *.prelower.txt -- superseded,
# not deleted.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/lower.log
SEED_RUN=H_aug_w64_d5_s5
NEED_MB=3500
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

$PY exp/composition.py --selftest >> "$LOG" 2>&1 || {
    say "SELFTEST FAILED: composition.py -- refusing to start"; exit 1; }
say "composition selftest passed (lower grid and monotonicity asserted)"

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

# ---------------------------------------------------------------- phase 0
# The card is idle and the CPU queue below does not touch it. Backgrounded so
# the lower pass -- the question that actually decides something -- starts now
# rather than 13 minutes from now.
train_seed () {
    if [ -f "exp/results/heldout/$SEED_RUN/final.pt" ]; then
        say "phase 0: $SEED_RUN already trained"; return
    fi
    while true; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
        [ "$FREE" -ge "$NEED_MB" ] && break
        echo "$(date '+%T') gpu0 ${FREE}MB free, waiting" >> "$LOG"
        sleep 300
    done
    say "phase 0: training $SEED_RUN on gpu0"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 "$PY" exp/train.py \
        --results exp/results/heldout --keep-epochs --protocol heldout \
        "$SEED_RUN" >> "$LOG" 2>&1 \
        || say "phase 0: $SEED_RUN FAILED -- later phases will run at 5 seeds"
    say "phase 0: $SEED_RUN done"
}
train_seed &
TRAIN_PID=$!

# ---------------------------------------------------------------- phase 1
say "phase 1: lower comparator, dev"
run_shards exp/composition.py 4 --lower --dev
say "phase 1: lower comparator, test"
run_shards exp/composition.py 4 --lower
say "phase 1 done"

# ---------------------------------------------------------------- phase 2
wait "$TRAIN_PID"
if [ ! -f "exp/results/heldout/$SEED_RUN/final.pt" ]; then
    say "phase 2 SKIPPED: no $SEED_RUN/final.pt; tables stay at 5 seeds"
else
    # threshold_control first: composition's base threshold is read from it,
    # and a sixth seed can move that base. If it moves, the arm's existing
    # rows sit at the old threshold and the report will print BOTH -- which is
    # the point of printing the raw threshold set rather than one value.
    say "phase 2: threshold_control dev (picks up seed 5)"
    run_shards exp/threshold_control.py 4 --dev
    say "phase 2: threshold_control test"
    run_shards exp/threshold_control.py 4
    $PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys
sys.path.insert(0, "exp")
import composition, threshold_control as control
rows = control.load("dev")
for arm in composition.ARMS:
    print(f"  base threshold {arm}: {composition.base_threshold(rows, arm, 'erl_bridged')}")
PYEOF
    say "phase 2: composition dev + test at 6 seeds"
    run_shards exp/composition.py 4 --dev
    run_shards exp/composition.py 4
    say "phase 2: lower comparator at 6 seeds"
    run_shards exp/composition.py 4 --lower --dev
    run_shards exp/composition.py 4 --lower
    for FIELD in H_aug_dir A_dice_dir; do
        say "phase 2: postproc dev+test, field $FIELD"
        run_shards exp/postproc_ceiling.py 6 --dev --field "$FIELD"
        run_shards exp/postproc_ceiling.py 6 --field "$FIELD"
    done
    say "phase 2 done"
fi

# ---------------------------------------------------------------- phase 3
for NAME in composition postproc threshold_control; do
    SRC="exp/results/${NAME}_verdict.txt"
    [ -f "$SRC" ] && [ ! -f "exp/results/${NAME}_verdict.prelower.txt" ] \
        && cp "$SRC" "exp/results/${NAME}_verdict.prelower.txt"
done
$PY exp/composition.py --report > exp/results/composition_verdict.txt 2>&1
$PY exp/summarize_postproc.py > exp/results/postproc_verdict.txt 2>&1
$PY exp/threshold_control.py --report \
    > exp/results/threshold_control_verdict.txt 2>&1
say "verdicts written"
say "all done"
