#!/usr/bin/env bash
# The second held-out batch: more seeds, then transfer under the same protocol.
#
# GATED ON TRAINING, NOT ON THE JOB. run_heldout.sh writes "all done" only
# after its final CPU scoring pass, and on 2026-08-28 an earlier queue waited
# on exactly that kind of marker for thirteen hours with a card idle -- d1all
# had finished training at 09:07 and spent the rest of the day in
# transfer_ceiling.py. So this waits until the two training stages have no
# pending runs, which is what "the cards are free" actually means.
#
# STAGES:
#   heldout_seeds  seeds 6-11 of the D-E sweep, 90 runs. The gate that decides
#                  D-E is the SIGN rule, and six seeds is where it is weakest:
#                  A_dice_clw beat A_dice by +240.0 ERL at t 3.12 -- the same
#                  size as H_aug_clw's +249.5, which HELD -- and failed on one
#                  seed of six coming back -201.
#   transfer       STARE, VessMAP and HRF, each trained on itself under
#                  --protocol heldout. The old transfer path handed the TEST
#                  list in as the validation set, so best.pt was chosen on the
#                  images the transfer numbers are reported from. H_aug_clw is
#                  in the arm list because "does the centreline weight survive
#                  a change of camera" is the question D-E has to answer next.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/heldout_more.log
OUT=exp/results/heldout
NEED_MB=3500
MAX_WAIT_HOURS=36
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

if ! "$PY" exp/gpu_queue.py --selftest >> "$LOG" 2>&1; then
    say "queue selftest FAILED -- refusing to queue"; exit 1
fi
if ! "$PY" exp/test_protocol.py >> "$LOG" 2>&1; then
    say "protocol selftest FAILED -- refusing to queue"; exit 1
fi
say "selftests passed"

left () {
    local total=0
    for STAGE in heldout heldout_series; do
        total=$(( total + $("$PY" exp/gpu_queue.py "$STAGE" --pending \
                            --results "$OUT" | wc -l) ))
    done
    echo "$total"
}

WAITED=0
while [ "$(left)" -gt 0 ]; do
    if [ "$WAITED" -ge $(( MAX_WAIT_HOURS * 12 )) ]; then
        say "first batch still has $(left) pending after ${MAX_WAIT_HOURS}h; "\
"starting anyway (the per-run VRAM gate keeps this polite)"
        break
    fi
    [ $(( WAITED % 12 )) -eq 0 ] && \
        say "waiting: $(left) run(s) left in the first batch ($(( WAITED / 12 ))h)"
    sleep 300
    WAITED=$(( WAITED + 1 ))
done
say "first batch training done; starting"

both () {
    bash exp/run_task_heldout.sh "$1" 0 0 2 "$OUT" & local a=$!
    bash exp/run_task_heldout.sh "$1" 1 1 2 "$OUT" & local b=$!
    wait "$a"; wait "$b"
}

for PASS in 1 2; do
    LEFT=$("$PY" exp/gpu_queue.py heldout_seeds --pending --results "$OUT" | wc -l)
    [ "$LEFT" -eq 0 ] && { say "heldout_seeds nothing pending"; break; }
    say "pass $PASS: heldout_seeds, $LEFT run(s) pending"
    both heldout_seeds
done

say "rescoring the sweep at twelve seeds"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py --results "$OUT" \
   >> "$LOG" 2>&1; then
    CUDA_VISIBLE_DEVICES="" "$PY" exp/select_heldout.py \
        > exp/results/heldout_summary.txt 2>&1
    cat exp/results/heldout_summary.txt | tee -a "$LOG"
else
    say "rescore FAILED"
fi

# ------------------------------------------------------------------ transfer
# H_aug_clw carries D-E; the other three are the arms the legacy transfer used,
# so the two tables line up arm for arm.
ARMS=(A_dice H_aug H_aug_clw K_focal_aug)
SEEDS=(0 1 2)
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
    say "transfer training on $DS (heldout)"
    train_on "$DS" 0 0 & A=$!
    train_on "$DS" 1 1 & B=$!
    wait "$A"; wait "$B"
done
say "all done"
