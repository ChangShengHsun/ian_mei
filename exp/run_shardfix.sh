#!/usr/bin/env bash
# Re-run every sharded table after the shard-coverage bug.
#
# THE BUG: `abs(hash((config, seed))) % total`. Python randomises str/tuple
# hashes per process, so each shard computed a different partition. Duplicates
# were caught by the resume set; OMISSIONS were caught by nothing. Measured
# damage before the fix:
#   postproc_ceiling      A_dice 10 of 12 seeds, H_aug_clw16 9 of 12
#   composition           A_dice / H_aug_clw2 / H_aug_clw16 3 of 6
#   terminal_anatomy      H_aug 7 of 12
#   transfer_calibration  stare/A_dice 2 of 3 -- under the gate's minimum, so
#                         every cell printed "--" and the stage read as merely
#                         unfinished rather than wrong
# threshold_control is NOT affected: it already sharded by list index.
#
# Fixed by sweep.shard_filter (stride over a sorted list), asserted to be an
# exact partition in four selftests. The shuffled control's RNG seed moved
# from hash() to zlib.crc32 for the same reason.
#
# ORDER IS BY VALUE PER HOUR, not by dependency: the two cheap tables that
# answer live questions run first, the five-hour postproc sweep last, because
# `composition` has superseded it as the headline.
#
# CPU ONLY. Old CSVs under results/heldout/pre_shardfix/, old verdicts as
# *_verdict.pre_shardfix.txt. Superseded, not deleted.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/shardfix.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in exp/postproc_ceiling.py exp/composition.py \
             exp/terminal_anatomy.py exp/transfer_calibration.py; do
    if ! $PY "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed (partition asserted in all four)"

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

for DATASET in stare vessmap hrf; do
    say "transfer calibration: $DATASET"
    run_shards exp/transfer_calibration.py 3 "$DATASET"
done
$PY exp/transfer_calibration.py --report \
    > exp/results/transfer_calibration_verdict.txt 2>&1
say "transfer calibration verdict written"

say "terminal anatomy"
run_shards exp/terminal_anatomy.py 4
$PY exp/terminal_anatomy.py --report > exp/results/terminal_verdict.txt 2>&1
say "terminal verdict written"

say "composition dev"
run_shards exp/composition.py 4 --dev
say "composition test"
run_shards exp/composition.py 4
$PY exp/composition.py --report > exp/results/composition_verdict.txt 2>&1
say "composition verdict written"

for FIELD in H_aug_dir A_dice_dir; do
    say "postproc dev, field $FIELD"
    run_shards exp/postproc_ceiling.py 6 --dev --field "$FIELD"
    say "postproc test, field $FIELD"
    run_shards exp/postproc_ceiling.py 6 --field "$FIELD"
done
$PY exp/summarize_postproc.py > exp/results/postproc_verdict.txt 2>&1
say "postproc verdict written"

say "all done"
