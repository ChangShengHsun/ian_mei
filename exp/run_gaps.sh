#!/usr/bin/env bash
# Re-run the workshop paper's two newest tables once DRIVE reaches 24 seeds.
#
# WHY A THIRD QUEUE. exp/metric_redundancy.py was written 2026-09-04, after
# run_deep.sh and run_survival.sh had launched, so neither knows about it and
# neither may be edited while running. This queue runs last.
#
# WHAT metric_redundancy SETTLED at twelve seeds, and what is exposed to more.
# The paper's existential check -- does ERL tell a reader anything Dice,
# clDice and Betti-0 do not -- came back clear, with the pre-registered kill
# condition not triggered:
#
#   rho(ERL, Dice)   -0.304   60.2% of arm pairs ordered differently
#   rho(ERL, clDice) +0.307   36.6%
#   rho(ERL, Betti0) +0.177   42.5%
#   at clDice matched to +-0.002, ERL spans 51.6 points
#
# Those are arm-level means over seeds, so doubling the seeds changes the
# means and could in principle move them. Nothing else in this repo is as
# load-bearing: if the kill condition triggers at 24 seeds the paper's
# motivation changes, so it gets re-run rather than assumed stable.
#
# PRE-REGISTERED 2026-09-04, before DRIVE has a seed past eleven:
#   1. The kill condition still does not trigger: rho(ERL, clDice) over the
#      56-arm set stays below 0.95, or the discordance stays above 10%.
#      Doubling the seeds tightens each arm's mean; it does not make two
#      metrics measure the same thing. If it DOES trigger, the paper must
#      rest on interpretability and say so.
#   2. Every sign in the 56-arm panel is unchanged: Dice negative, clDice
#      positive, Betti-0 positive. A sign flip on twelve more seeds would
#      mean the arm-level means were never resolved.
#   3. The matched-clDice spread stays above 30 points. It is 51.6 now, and
#      the median window is 33.3; more seeds add runs to each window, which
#      can only widen the observed range or leave it alone.
#   4. The convention panel (1b) keeps its null: rho(erl_split, erl_bridged)
#      over the arms stays above 0.9 and the sign against Dice is the same
#      under both. This one is a NULL being defended, not a finding.
#
# CPU only.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/gaps.log
say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

for OTHER in run_deep.sh run_survival.sh run_anchor.sh; do
    while pgrep -u "$USER" -f "exp/$OTHER" > /dev/null 2>&1; do
        sleep 600
    done
done
say "no other queue running"

if ! $PY exp/metric_redundancy.py --selftest >> "$LOG" 2>&1; then
    say "SELFTEST FAILED -- refusing to start"; exit 1
fi
say "selftest passed"

[ -f exp/results/metric_redundancy.txt ] \
    && [ ! -f exp/results/metric_redundancy.12seeds.txt ] \
    && cp exp/results/metric_redundancy.txt \
          exp/results/metric_redundancy.12seeds.txt \
    && say "twelve-seed table preserved -- predictions are scored by diff"

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 $PY exp/metric_redundancy.py \
    --report > exp/results/metric_redundancy.txt 2>&1
say "rebuilt at the seed count now on disk"

# Score prediction 1 mechanically. A kill condition that only a human
# remembers to check is not a kill condition.
grep -E "KILL CONDITION" exp/results/metric_redundancy.txt | tee -a "$LOG"
say "all done"
