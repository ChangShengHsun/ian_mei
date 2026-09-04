#!/usr/bin/env bash
# Re-audit every seed claim once DRIVE reaches twenty-four.
#
# WHY THIS EXISTS AS A SEPARATE QUEUE. exp/seed_survival.py was written on
# 2026-09-04 AFTER run_deep.sh had already launched, in response to
# run_anchor.sh's pre-registered prediction 2 failing. run_deep.sh's phase 4
# therefore does not know about it, and editing a running script is forbidden
# here. So this queue waits for run_deep.sh and re-runs the audit on the
# 24-seed DRIVE tables it produces.
#
# WHAT run_anchor.sh SETTLED, and what is still open. Across the three
# transfer datasets HOLDS went 6 cells -> 2 when seeds doubled, 0 the other
# way, and 3 of the 4 deaths had an effect that held or grew with a larger t.
# The count-based bound (rule of three) cannot distinguish a settled cell from
# an under-sampled one -- every d = 0 cell gets the same 3/n -- so
# seed_survival adds a normal model of the per-seed differences. At twelve
# seeds that model splits the two populations cleanly:
#
#   transfer calibration   60 cells,   2 survivors,  0/2 decisive
#   DRIVE composition     100 cells,  80 survivors, 55/80 decisive
#
# The open question is whether the model's verdict on DRIVE survives real new
# seeds -- the same out-of-sample test that killed prediction 2, now put to
# the estimator that replaced it.
#
# PRE-REGISTERED 2026-09-04, before DRIVE has a single seed past eleven:
#   1. Of the 55 DRIVE cells the model calls decisive at twelve seeds
#      (half-life > 1000), at least 53 still have zero dissenters at
#      twenty-four. The model says they die after thousands of seeds; if more
#      than two die after twelve more, the normal approximation is wrong in
#      the tail and the two-population claim collapses to "we do not know".
#   2. Of the 25 DRIVE cells with d = 0 that the model does NOT call decisive,
#      at least 5 pick up a dissenter by twenty-four. These are the cells the
#      count-based bound cannot separate from the 55; if none of them moves,
#      the model is not buying discrimination, it is just ranking.
#   3. `predicted` still reads d = n at twenty-four on every arm, and `lower`
#      still reads d = 0 on every arm. The controls are a property of the
#      operators, not of twelve seeds.
#   4. The transfer table's 2 survivors are NOT re-examined here -- no new
#      transfer seeds are queued -- so nothing in this queue can rescue them.
#      Stated so that a later reader does not mistake their absence for a
#      result.
#
# CPU only. No GPU is requested and none should be taken: run_deep.sh's own
# training is what this waits for.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/survival.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for OTHER in run_deep.sh run_anchor.sh run_gate.sh run_paper.sh; do
    while pgrep -u "$USER" -f "exp/$OTHER" > /dev/null 2>&1; do
        sleep 600
    done
done
say "no other queue running"

for CHECK in exp/seed_survival.py exp/seed_stability.py; do
    if ! $PY "$CHECK" --selftest >> "$LOG" 2>&1; then
        say "SELFTEST FAILED: $CHECK -- refusing to start"; exit 1
    fi
done
say "selftests passed"

# The twelve-seed audit, kept beside the twenty-four-seed one. Predictions 1
# and 2 above are scored by diffing these two files, so losing the old one
# would make the queue unfalsifiable.
[ -f exp/results/seed_survival.txt ] \
    && [ ! -f exp/results/seed_survival.12seeds.txt ] \
    && cp exp/results/seed_survival.txt exp/results/seed_survival.12seeds.txt \
    && say "twelve-seed audit preserved"

say "re-auditing on the 24-seed tables"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 $PY exp/seed_survival.py --report \
    > exp/results/seed_survival.txt 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 $PY exp/seed_stability.py --report \
    > exp/results/seed_stability.txt 2>&1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY exp/erl_spec_transfer.py \
    --report > exp/results/erl_spec_transfer.txt 2>&1
say "reports rebuilt"

# Score prediction 3 mechanically, so it cannot be quietly skipped.
$PY - <<'PYEOF' | tee -a "$LOG"
import re
from pathlib import Path
text = Path("exp/results/seed_survival.txt").read_text()
rows = re.findall(r"erl_\w+/(\w+)/\S+\s+(\d+)/(\d+)", text)
for source in ("predicted", "lower"):
    got = [(int(d), int(n)) for name, d, n in rows if name == source]
    if not got:
        print(f"prediction 3: {source} NOT FOUND -- cannot score")
        continue
    if source == "predicted":
        ok = all(d == n for d, n in got)
    else:
        ok = all(d == 0 for d, n in got)
    print(f"prediction 3 ({source}): {len(got)} cells, "
          f"{'HOLDS' if ok else 'FALSIFIED'}")
PYEOF
say "all done"
