#!/usr/bin/env bash
# The paper's own tables, made immune to the paper's own critique.
#
# THE ARGUMENT FOR THE BIGGEST ITEM HERE. run_gate.sh measured, on STARE, that
# a verdict can flip from HOLDS to fails when seeds go 12 -> 24 while the
# effect size and t BOTH GROW. That is the paper's fourth artefact. Every
# DRIVE table in this repo is at twelve seeds. A paper whose finding is "your
# verdict depends on how many seeds you ran" cannot report its own primary
# dataset at the seed count it just called insufficient. Twelve arms x twelve
# new seeds is about 35 GPU-hours, roughly 18 on two cards, and it is the
# price of not handing a reviewer that sentence.
#
# THE SECOND ITEM is CPU-only and runs alongside the training: erl_spec on
# STARE, HRF and VessMAP. The DRIVE specification table found a per-arm spread
# of 38.6 to 42.0 percentage points; whether that is a property of ERL or of
# DRIVE is the first question it invites. Predictions for it are pre-registered
# in exp/erl_spec_transfer.py's header, not here.
#
# THE THIRD ITEM is forced by a defect found 2026-09-04. sweep_score.py was the
# only file in the held-out pipeline still normalising on stack_split("train")
# -- all 20 training images, including the 5 held back for selection. Every
# other file (frontier, composition, threshold_control, postproc_ceiling) uses
# "fit". Measured cost at threshold 0.5 on three runs: +0.40, +0.06, +0.14 ERL
# points, always in favour of the leaked stack, against the repo's +1.4-point
# bar for the largest honest effect. The GAPS the leak ledger reports are
# nearly unaffected -- both sides shared the constant -- but the absolute
# columns were optimistic, in a paper about protocol impurity. Fixed; the whole
# table must be rebuilt because two normalisations cannot coexist in one file.
#
# chosen_epochs() reads each run's own log.csv, NOT checkpoint_scores.csv, so
# the fix does not move any selected epoch and nothing downstream of the
# ledger needs rebuilding. Verified 2026-09-04 before this queue was written.
#
# PRE-REGISTERED 2026-09-04, before any run of this queue exists:
#   1. The ledger's checkpoint headline moves by less than 0.5 points from
#      +2.3. Evidence for the prior: the 360-run and 487-run ledgers agreed to
#      0.4 with the worst run identical at +24.9. A bigger move means twelve
#      seeds was not enough for the LEDGER either, which would be a finding.
#   2. No DRIVE composition cell flips HOLDS -> fails at 24 seeds. Every one
#      reads 100% at every k in seed_stability.txt and their effects are an
#      order of magnitude clear of the gate. This is the same out-of-sample
#      test run_anchor.sh puts to HRF and VessMAP, on the dataset where the
#      resampling curve says the cells are safest. A flip here would mean the
#      curve does not predict the anchor even in the easy case.
#   3. erl_spec's per-arm spread stays between 35 and 45 points for every arm.
#      Doubling the seeds averages the cells, it does not narrow the gap
#      between two DEFINITIONS.
#   4. `endpoint_shuf` still passes 100% and `predicted` still 0% at 24 seeds.
#      The control structure is a property of the operators, not of twelve
#      seeds; if it moves, composition.md's conclusion was a seed artefact.
#
# The lab's two 4070 Ti are SHARED. wait-for-VRAM only; never touch another
# user's processes.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/deep.log
NEED_MB=3500
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for OTHER in run_anchor.sh run_gate.sh run_rescore.sh run_ten.sh run_paper.sh; do
    while pgrep -u "$USER" -f "exp/$OTHER" > /dev/null 2>&1; do
        say "$OTHER still up, waiting"; sleep 300
    done
done
say "no other queue running"

for CHECK in exp/erl_spec_transfer.py exp/erl_spec.py exp/sweep_score.py \
             exp/leak_ledger.py exp/composition.py exp/seed_stability.py; do
    if ! $PY "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed"

# The 12-seed DRIVE tables, kept readable beside the 24-seed ones.
for NAME in leak_ledger convention_flip erl_spec composition_verdict \
            seed_stability; do
    SRC="exp/results/${NAME}.txt"
    [ -f "$SRC" ] && [ ! -f "exp/results/${NAME}.12seeds.txt" ] \
        && cp "$SRC" "exp/results/${NAME}.12seeds.txt"
done
say "12-seed DRIVE tables preserved as *.12seeds.txt"

# ------------------------------------------------- phase 1a: GPU, background
ARMS=(A_dice H_aug H_aug_clw2 H_aug_clw8 H_aug_clw16 H_aug_clw64
      K_focal_aug_clw32 K_focal_aug_clw64 A_dice_clw64 H_aug_w64_d5
      A_dice_dir H_aug_dir)
WORK=()
for SEED in 12 13 14 15 16 17 18 19 20 21 22 23; do
    for ARM in "${ARMS[@]}"; do
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
say "phase 1a: ${#WORK[@]} DRIVE runs to twenty-four seeds (background)"
work_gpu 0 0 >> "$LOG" 2>&1 &
GPU0=$!
work_gpu 1 1 >> "$LOG" 2>&1 &
GPU1=$!

# ---------------------------------- phase 1b: CPU, alongside the training
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
say "phase 1b: erl_spec on stare, hrf and vessmap (CPU, alongside phase 1a)"
run_shards exp/erl_spec_transfer.py 4 stare
run_shards exp/erl_spec_transfer.py 4 vessmap
run_shards exp/erl_spec_transfer.py 4 hrf
$PY exp/erl_spec_transfer.py --report \
    > exp/results/erl_spec_transfer.txt 2>&1
say "phase 1b done: exp/results/erl_spec_transfer.txt"

wait "$GPU0" "$GPU1"
say "phase 1a done: DRIVE at twenty-four seeds"

# ------------------------------------------------------------------ phase 2
# ONE invocation over EVERY run, never a subset in a loop: sweep_score.py
# rebuilds the table with "w". A loop here is what destroyed it on 09-03, and
# a subset call is now refused outright by the guard added 09-04. The rebuild
# is also mandatory rather than incremental this time: the normalisation fix
# means rows written before today are not comparable with rows written after.
say "phase 2: full rebuild of checkpoint_scores on the fit-split normalisation"
KEEP=exp/results/heldout/checkpoint_scores.trainstack.csv
[ ! -f "$KEEP" ] && cp exp/results/heldout/checkpoint_scores.csv "$KEEP" \
    && say "old train-stack table preserved as $(basename "$KEEP")"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 $PY exp/sweep_score.py \
    --results exp/results/heldout >> "$LOG" 2>&1 \
    || say "!!! rebuild FAILED"
$PY - <<'PYEOF' | tee -a "$LOG"
import csv, sys
sys.path.insert(0, "exp")
import select_heldout as heldout
rows = list(csv.DictReader(heldout.SCORES.open()))
runs = {r["run"] for r in rows}
assert len(runs) * 200 == len(rows), (len(runs), len(rows))
print(f"rebuilt: {len(runs)} runs, {len(rows)} rows")
PYEOF
say "phase 2 done"

# ------------------------------------------------------------------ phase 3
say "phase 3: the DRIVE curves at twenty-four seeds"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY exp/frontier.py --dev \
    >> "$LOG" 2>&1 || say "!!! frontier --dev FAILED"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY exp/frontier.py \
    >> "$LOG" 2>&1 || say "!!! frontier FAILED"
say "phase 3a done: frontier"
run_shards exp/erl_spec.py 4
say "phase 3b done: erl_spec"
run_shards exp/composition.py 4 --dev
run_shards exp/composition.py 4
run_shards exp/composition.py 4 --lower --dev
run_shards exp/composition.py 4 --lower
say "phase 3c done: composition"

# ------------------------------------------------------------------ phase 4
say "phase 4: every DRIVE report at twenty-four seeds"
CUDA_VISIBLE_DEVICES="" $PY exp/leak_ledger.py --report \
    > exp/results/leak_ledger.txt 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/composition.py --report \
    > exp/results/composition_verdict.txt 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/convention_flip.py --report \
    > exp/results/convention_flip.txt 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/erl_spec.py --report \
    > exp/results/erl_spec.txt 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/seed_stability.py --report \
    > exp/results/seed_stability.txt 2>&1
say "phase 4 done"
say "all done"
