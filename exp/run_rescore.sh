#!/usr/bin/env bash
# Rebuild checkpoint_scores.csv after run_paper.sh phase 3 truncated it.
#
# WHAT HAPPENED, 2026-09-03. run_paper.sh phase 3 called
#   sweep_score.py --results exp/results/heldout "$RUN"
# once per missing run, in a loop, on the stated reasoning that "two writers
# on one file interleave partial lines". The loop was serial and that reasoning
# was right. The assumption underneath it was not: sweep_score.py:152 opens
# the file with "w", NOT "a". So every call TRUNCATED the table and wrote only
# the run it was given. Twenty-seven calls later the file held one run instead
# of 460.
#
# The rule that was broken is in CLAUDE.md and in the global rules: a fact you
# can check -- a path, a flag, a file mode -- is checked, not assumed. One
# grep for `OUT.open` would have caught it before the first call.
#
# WHAT WAS AND WAS NOT LOST. Not lost: leak_ledger.460runs.txt, written in
# phase 1 BEFORE phase 3 ran, holds the ledger on the full table. Not lost:
# the threshold and geometry levels, which read frontier*.csv and postproc*,
# untouched. Recovered: 360 of the 460 runs, from the last commit that carried
# the csv. Genuinely lost: the scoring of ~127 runs, which is CPU time, not
# information -- this script buys it back.
#
# WHY NOT JUST RE-RUN EVERYTHING. sweep_score takes many run names in one
# invocation and writes them all in a single "w" pass, which is safe. But that
# is 487 runs at about 3 minutes each. Scoring only what is missing and
# concatenating is six hours instead of twenty-four, and the header is
# identical so the concat is exact.
#
# THE ORDER BELOW IS DELIBERATE. The good table is copied aside FIRST, because
# step 2 truncates the live file on purpose. Nothing here is destructive to
# anything that has not already been saved.
set -eu
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
SCORES=exp/results/heldout/checkpoint_scores.csv
KEEP=exp/results/heldout/checkpoint_scores.recovered360.csv
LOG=exp/results/rescore.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

MISSING=$($PY - <<'PYEOF'
import csv, sys
sys.path.insert(0, "exp")
import select_heldout as heldout
scored = {r["run"] for r in csv.DictReader(heldout.SCORES.open())}
disk = {p.parent.name for p in heldout.ROOT.glob("*_s*/final.pt")}
print(" ".join(sorted(disk - scored)))
PYEOF
)
COUNT=$(echo "$MISSING" | wc -w)
BEFORE=$(tail -n +2 "$SCORES" | cut -d, -f1 | sort -u | wc -l)
say "table holds $BEFORE runs; $COUNT missing"
[ "$COUNT" -eq 0 ] && { say "nothing to do"; exit 0; }

cp "$SCORES" "$KEEP"
say "good table copied to $(basename "$KEEP") -- the next step truncates the live one"

# ONE invocation, every missing run. sweep_score writes with "w", so a single
# call is the only safe shape: it truncates once and writes everything it was
# asked for. A loop here is the bug this script exists to repair.
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 $PY exp/sweep_score.py \
    --results exp/results/heldout $MISSING >> "$LOG" 2>&1
say "scored $COUNT runs"

$PY - "$KEEP" "$SCORES" <<'PYEOF'
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
say "merge verified"

CUDA_VISIBLE_DEVICES="" $PY exp/leak_ledger.py --report \
    > exp/results/leak_ledger.txt 2>&1
say "leak ledger rebuilt on the full table"
say "all done"
