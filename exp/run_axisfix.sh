#!/usr/bin/env bash
# Re-run every dilation-based table with anisotropic.axis_element.
#
# WHY: the structuring element used by every oriented_dilation result written
# before 2026-09-01 tested lattice points against an ellipse, and so delivered
# LESS reach than it was asked for -- by 22% at the long end and by 100% on
# diagonal vessels at ALONG=0.5 ACROSS=0.25, which is the geometry the sweep
# actually picked. So the post-processing question was never asked at the
# reach it was written to test. The old CSVs are kept under
# results/heldout/pre_axisfix/ and the old verdict as
# results/postproc_verdict.pre_axisfix.txt; they are superseded, not deleted.
#
# ALSO: ALONG now runs to 4.0 widths. The measured break stretches reach 21 px
# (about 7 widths) and a dilation closes half a gap from each side, so a grid
# stopping at 2.0 could not have closed a typical break even with a correct
# element.
#
# CPU ONLY, sequential by stage so the box stays usable. Both GPUs stay free.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/axisfix.log
OUT=exp/results/heldout
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in "exp/anisotropic.py" "exp/postproc_ceiling.py --selftest" \
             "exp/composition.py --selftest" \
             "exp/summarize_postproc.py --selftest"; do
    if ! $PY $CHECK >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed"

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

# Dev before test in both stages: the geometry is chosen on dev, so a test
# file without its dev counterpart cannot be reported anyway.
for FIELD in H_aug_dir A_dice_dir; do
    say "postproc dev, field $FIELD"
    run_shards exp/postproc_ceiling.py 6 --dev --field "$FIELD"
    say "postproc test, field $FIELD"
    run_shards exp/postproc_ceiling.py 6 --field "$FIELD"
done
say "postproc done: $(cat "$OUT"/postproc_ceiling*.csv 2>/dev/null | wc -l) test rows"
$PY exp/summarize_postproc.py > exp/results/postproc_verdict.txt 2>&1
say "postproc verdict written"

say "composition dev"
run_shards exp/composition.py 4 --dev
say "composition test"
run_shards exp/composition.py 4
$PY exp/composition.py --report > exp/results/composition_verdict.txt 2>&1
say "composition verdict written"

say "all done"
