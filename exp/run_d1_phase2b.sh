#!/usr/bin/env bash
# D-B and the combination arm, after the gate was repaired.
#
# The gate that skipped D-B on 2026-08-27 05:02 was not saying no; it could
# not evaluate. Its Dice floor was absolute and derived from an oracle that
# places pixels exactly, which no dilation can match, so every source came
# back empty and an empty table was read as a verdict. Under the matched-cost
# comparison the mechanism clears the pre-registered BUILD IT threshold on
# three of four arms at the tightest budget and all four at the next one.
#
#   d1b  the propagation layer driven by the model's own direction head, and
#        the same layer driven by NOISE. 24 runs.
#   d1f  both interventions in one arm: centreline weighting AND propagation.
#        The only question worth asking once both work separately. 12 runs.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/d1_phase2b.log
SWEEP=exp/results/selection_sweep
export OMP_NUM_THREADS=4
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }
both () {
    bash exp/run_task.sh "$1" 0 0 2 "$SWEEP" 1 & local a=$!
    bash exp/run_task.sh "$1" 1 1 2 "$SWEEP" 1 & local b=$!
    wait "$a"; wait "$b"
}

# The layer's geometry must exist and must not be the do-nothing setting: a
# zero radius builds a one-pixel kernel, the layer is the identity, and the
# arm trains for hours and reports that propagation does nothing.
GEO=exp/results/d1_geometry.txt
if [ ! -f "$GEO" ]; then say "no $GEO -- run exp/gate_d1.py first"; exit 1; fi
read -r ALONG ACROSS < "$GEO"
if [ "$ALONG" = "0.0" ] && [ "$ACROSS" = "0.0" ]; then
    say "geometry is (0, 0): the layer would be the identity. Refusing."
    exit 1
fi
say "geometry along=$ALONG across=$ACROSS widths"

for PASS in 1 2; do
    for STAGE in d1b d1f; do
        LEFT=$("$PY" exp/gpu_queue.py "$STAGE" --pending --results "$SWEEP" \
               | wc -l)
        [ "$LEFT" -eq 0 ] && { say "$STAGE nothing pending"; continue; }
        say "pass $PASS: $STAGE, $LEFT run(s) pending"
        both "$STAGE"
    done
done

say "rescoring the sweep"
cp "$SWEEP/checkpoint_scores.csv" "$SWEEP/checkpoint_scores.pre_prop.csv"
if CUDA_VISIBLE_DEVICES="" "$PY" exp/sweep_score.py >> "$LOG" 2>&1; then
    say "rescore OK"
    CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_direction.py \
        > exp/results/direction_summary.txt 2>&1
    CUDA_VISIBLE_DEVICES="" "$PY" exp/summarize_selection.py \
        > exp/results/selection_summary.txt 2>&1
else
    say "rescore FAILED -- restoring"
    cp "$SWEEP/checkpoint_scores.pre_prop.csv" "$SWEEP/checkpoint_scores.csv"
fi
say "done"
