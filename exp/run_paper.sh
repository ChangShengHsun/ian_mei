#!/usr/bin/env bash
# Everything the workshop paper still needs that a machine can produce.
#
# THE FRAMING IS SETTLED. 2026-09-03: the supervisor accepts a workshop
# submission on the diagnosis, with the method left for a main-conference
# paper later. That removes the one blocker no amount of compute could clear,
# and it changes what is missing from "a method" to "the tables a diagnosis
# paper is made of". Three of those tables did not exist this morning.
#
# WHAT THIS QUEUE PRODUCES, in the order a reader meets it:
#
#   leak_ledger      Table 1. What each protocol leak is WORTH, in ERL
#                    points, against the largest honest effect any published
#                    topology loss produces here (+1.4). Reads csvs that
#                    already exist -- no GPU, no scoring pass.
#   convention_flip  Whether the ERL convention changes the NUMBERS or the
#                    WINNER. erl_reference settled the first years ago; the
#                    second is new and is the sharper claim.
#   seed_stability   Extended today to the DRIVE composition cells, so the
#                    headline table is seed-audited like the transfer ones.
#
# AND ONE GAP THAT NEEDS SCORING. checkpoint_scores.csv covers 460 of the 487
# runs on disk; the 27 missing are the direction heads and H_aug_w64_d5 seeds
# trained in the last two days. leak_ledger's checkpoint level is the one
# whose leak is largest (+2.3 points mean, +24.9 worst) so it is the row that
# must not rest on a partial table. Phase 3 scores them and rebuilds it.
#
# RUNS BESIDE run_gate.sh, not behind it. That queue is training STARE on the
# cards; everything here is CPU and writes files nothing else touches
# (leak_ledger.txt, convention_flip.txt, checkpoint_scores.csv). The one
# shared reader is seed_stability.py, which run_gate calls in its phase 3 --
# it was patched and selftested BEFORE this queue was launched, so that call
# gets the extended version rather than a half-written one.
#
# PRE-REGISTERED 2026-09-03, before phases 2-4 have printed anything:
#   1. The winner flips between conventions in at least a quarter of the
#      resolved cells. Below that, "report both conventions" stays a
#      precision argument and not a correctness one, and the paper's fourth
#      artefact keeps its old and weaker form.
#   2. `lower` is the convention-A winner in every DRIVE cell at -0.02.
#      It already is at 6 seeds; twelve should not move it.
#   3. Scoring the missing 27 runs does not move the checkpoint leak by more
#      than 0.3 points. They are direction heads and one capacity arm, not a
#      different population; a larger move would mean the 460-run table was
#      not representative and every row of it needs re-reading.
#   4. On the DRIVE composition cells, `lower` reads 100% at every k while at
#      least one endpoint cell falls below 90% at k=12. That is the seed
#      audit of the headline, and the shape that would make the endpoint
#      result unreportable at any seed count we ran.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/paper.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for CHECK in exp/leak_ledger.py exp/convention_flip.py exp/seed_stability.py; do
    if ! $PY "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed (leak sign rule, control exclusion, 60-cell rebuild)"

# ---------------------------------------------------------------- phase 1
say "phase 1: protocol leak ledger"
CUDA_VISIBLE_DEVICES="" $PY exp/leak_ledger.py --report \
    > exp/results/leak_ledger.460runs.txt 2>&1
cp exp/results/leak_ledger.460runs.txt exp/results/leak_ledger.txt
say "phase 1 done: exp/results/leak_ledger.txt"

# ---------------------------------------------------------------- phase 2
say "phase 2: convention flip table"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/convention_flip.py --report \
    > exp/results/convention_flip.txt 2>&1
say "phase 2 done: exp/results/convention_flip.txt"

# ---------------------------------------------------------------- phase 3
# sweep_score takes run names on the command line, which is the repo's way of
# letting a partial set be analysed while the rest trains.
#
# FIXED 2026-09-04, after this phase destroyed the table on 2026-09-03. The
# loop below used to call sweep_score ONCE PER RUN, on the reasoning that two
# writers on one csv interleave partial lines. The serialisation was right;
# the assumption under it was not. sweep_score.py:152 opens the file with
# "w", not "a", so each of the 27 calls TRUNCATED checkpoint_scores.csv and
# wrote only the run it was given. The table went from 460 runs to 1. It was
# restored from git and rescored by exp/run_rescore.sh.
#
# One invocation with every missing name is the only safe shape: it truncates
# once and writes everything it was asked for.
MISSING=$($PY - <<'PYEOF'
import csv, sys
sys.path.insert(0, "exp")
import select_heldout as heldout
scored = {r["run"] for r in csv.DictReader(heldout.SCORES.open())}
disk = {p.parent.name for p in heldout.ROOT.glob("*_s*/final.pt")}
print(" ".join(sorted(disk - scored)))
PYEOF
)
if [ -n "$MISSING" ]; then
    COUNT=$(echo "$MISSING" | wc -w)
    say "phase 3: scoring $COUNT checkpoints missing from checkpoint_scores"
    KEEP=exp/results/heldout/checkpoint_scores.before_phase3.csv
    cp exp/results/heldout/checkpoint_scores.csv "$KEEP"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 $PY exp/sweep_score.py \
        --results exp/results/heldout $MISSING >> "$LOG" 2>&1 \
        || say "!!! scoring FAILED, continuing"
    $PY - "$KEEP" exp/results/heldout/checkpoint_scores.csv <<'PYEOF'
import csv, sys
from pathlib import Path
kept, fresh = Path(sys.argv[1]), Path(sys.argv[2])
with kept.open() as handle:
    header = handle.readline().rstrip("\n")
    old = handle.read()
with fresh.open() as handle:
    assert handle.readline().rstrip("\n") == header, "header drift, refusing"
    new = handle.read()
merged = fresh.with_suffix(".merged")
merged.write_text(header + "\n" + old + new)
rows = list(csv.DictReader(merged.open()))
runs = {r["run"] for r in rows}
assert len(runs) * 200 == len(rows), (len(runs), len(rows))
merged.replace(fresh)
print(f"merged: {len(runs)} runs, {len(rows)} rows")
PYEOF
    say "phase 3: merge verified"
    say "phase 3: rebuilding the ledger on the full table"
    CUDA_VISIBLE_DEVICES="" $PY exp/leak_ledger.py --report \
        > exp/results/leak_ledger.txt 2>&1
    say "phase 3 done"
else
    say "phase 3 skipped: checkpoint_scores already covers every run on disk"
fi

# ---------------------------------------------------------------- phase 4
say "phase 4: seed stability including the DRIVE headline"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/seed_stability.py --report \
    > exp/results/seed_stability.txt 2>&1
say "phase 4 done"
say "all done"
