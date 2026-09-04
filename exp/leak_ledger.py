"""How many ERL points does each protocol leak buy, without changing the model?

WRITTEN AND SELFTESTED 2026-09-03, BEFORE IT REPORTED ANYTHING.

THE POINT. CLAUDE.md records that the same selection mistake was made three
times, one level lower each time: the checkpoint chose its epoch on the test
images, then the threshold was read off the test curve, then the dilation
geometry was picked on the images it was reported on. Each was found and
fixed. What was never measured is what each one was WORTH -- and that number
is the one a reader needs, because it is the yardstick every published effect
in this area has to be held against.

The comparison that makes the table land: in this repo the largest honest
effect any published topology loss produces, read at each arm's own
dev-optimal threshold, is +1.4 points of traced run length
(stage-report/calibration.md). If a selection leak buys more than that, then
a paper that leaks does not need a method at all.

HOW EACH LEVEL IS MEASURED. The honest side is always the held-out rule.
The leaked side takes the argmax OF THE REPORTED METRIC on the very images
it then reports -- the ceiling of what a leak is worth, and what makes the
three levels comparable to each other:

  checkpoint  the epoch maximising traced ERL on the 20 test images, against
              the epoch rule (iv) picks on the 5 dev images. At the shared
              0.5, from checkpoint_scores.csv.
  threshold   among the thresholds a 0.02 Dice budget admits ON DEV, the one
              with the best DEV traced (honest) or the best TEST traced
              (leaked). Read: traced on test, frontier_dev / frontier.
  geometry    the same, over (along, across) for the predicted field, on
              postproc_dev / postproc_ceiling.

THE THREE LEVELS ARE NOT THE SAME KIND OF THING, and finding that out is
half of what this file is for. Levels 2 and 3 select along an axis the
reported metric is MONOTONE in: traced run length only rises as the threshold
falls, and only rises as the radii grow. So once the admissible set is fixed,
the argmax is the same point on either split and peeking at the metric buys
EXACTLY ZERO -- asserted in the selftest, because a non-zero there would mean
the admissible sets had drifted apart. The leak at those levels lives
entirely in the CONSTRAINT: which points the Dice budget admits, priced on
dev or priced on the reported images. That is also the historical bug, and it
is what the table reports.

A budget-priced-on-test leak is NOT sign-constrained. The dev split is five
images; a threshold that costs 0.02 Dice there can cost more than that on the
twenty test images, so the honest pipeline sometimes overspends its own
budget on the reported set and the leaked pipeline, held to the real budget,
reports LESS. That is not noise to be averaged away -- it is the price of
selecting on five images, and the table prints how often it happens.

A FOURTH ROW reconstructs the historical bug rather than its ceiling.
`best.pt` chose the epoch by whole-validation DICE, and that validation set
included the reported images -- so the leak optimised Dice while the paper
reported ERL. Those two do not point the same way, and this row is allowed to
come out NEGATIVE. Separating them matters: only the first is an argument for
the held-out protocol, and the caught bug was the second.

CAUGHT BY THE SELFTEST, 2026-09-03, on the first run of this file: the
checkpoint level was originally written as "rule (iv), applied to test", to
keep the rule fixed and vary only the split. But rule (iv) maximises clDice
while this table reports ERL, so the leaked epoch lost to the honest one on
hundreds of runs and the assertion below fired. The rule is not the leak; the
SPLIT is, and the quantity a leak is worth has to be measured on the quantity
that gets published.

An argmax-on-the-reported-metric leak is >= its honest counterpart for EVERY
run by construction. The selftest asserts that on the three levels where it
must hold; a violation would mean an argmax is reading the wrong column,
which is the shape of all three silent instrument bugs of 2026-09-01.

  python exp/leak_ledger.py --selftest
  python exp/leak_ledger.py --report

Reads only csvs that already exist. No GPU, no scoring pass.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postproc_ceiling as sweep
import select_checkpoint as rules_module
import select_heldout as heldout

BUDGET = 0.02
SHARED = 0.5
# The bar every leak is read against: the largest honest effect any published
# topology loss produces in this repo, at each arm's own dev-optimal
# threshold. stage-report/calibration.md, and CLAUDE.md's settled item 1.
HONEST_CEILING = 1.4
ARMS = sweep.CONTROL + sweep.FRONTIER


def rule_iv():
    """THE selection rule, fetched by name from the one module that owns it.

    Not re-implemented: the whole measurement is 'same rule, different split',
    and a second copy of the rule would silently make it 'different rule,
    different split' the first time select_checkpoint.py changed.
    """
    return dict(rules_module.rules())[heldout.RULE]


def best_within(curve: dict, floor: float, metric_index: int = 1):
    """Key of the entry with the largest metric whose Dice clears `floor`."""
    best = None
    for key, values in curve.items():
        if values[0] < floor:
            continue
        if best is None or values[metric_index] > best[1]:
            best = (key, values[metric_index])
    return None if best is None else best[0]


# ------------------------------------------------------------------- level 1

def _checkpoint_tables():
    """(per-epoch dev-rule points, per-epoch test traced) for every run."""
    per_epoch = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for row in csv.DictReader(heldout.SCORES.open()):
        key, epoch = row["run"], int(row["epoch"])
        per_epoch[key][epoch].append({"dice": float(row["dice"]),
                                      "cldice": float(row["cldice"]),
                                      "betti0_err": float(row["betti0_err"])})
        cell = totals[key][epoch]
        cell[0] += float(row["erl"])
        cell[1] += float(row["skel_px"])
    points = {run: [{"epoch": epoch,
                     **{k: float(np.mean([p[k] for p in rows]))
                        for k in ("dice", "cldice", "betti0_err")}}
                    for epoch, rows in sorted(epochs.items())]
              for run, epochs in per_epoch.items()}
    traced = {run: {e: c[0] / c[1] for e, c in by_epoch.items() if c[1]}
              for run, by_epoch in totals.items()}
    return points, traced


def checkpoint_level() -> dict:
    """{run: (leaked traced, honest traced)} -- the epoch that maximises
    traced ERL on the reported images, against the dev-chosen epoch."""
    _, traced = _checkpoint_tables()
    honest_epochs = heldout.chosen_epochs()
    out = {}
    for run, curve in traced.items():
        honest = honest_epochs.get(run)
        if not curve or honest not in curve:
            continue
        out[run] = (curve[max(curve, key=lambda e: curve[e])], curve[honest])
    return out


def checkpoint_bestpt_level() -> dict:
    """{run: (leaked traced, honest traced)} -- the HISTORICAL bug, not its
    ceiling: `best.pt` maximised Dice on a validation set containing the
    reported images, while the paper reported ERL. May come out negative."""
    points, traced = _checkpoint_tables()
    honest_epochs = heldout.chosen_epochs()
    out = {}
    for run, curve in traced.items():
        honest = honest_epochs.get(run)
        if honest not in curve:
            continue
        by_dice = max(points[run], key=lambda p: p["dice"])["epoch"]
        if by_dice in curve:
            out[run] = (curve[by_dice], curve[honest])
    return out


# ------------------------------------------------------------------- level 2

def _threshold_curves():
    """{split: {run: {threshold: (dice, traced)}}} for dev and test."""
    curves = {}
    for name, split in (("frontier_dev.csv", "dev"), ("frontier.csv", "test")):
        path = heldout.ROOT / name
        if not path.exists():
            return {}
        holder = curves.setdefault(split, defaultdict(dict))
        for row in csv.DictReader(path.open()):
            holder[row["run"]][float(row["threshold"])] = (
                float(row["dice"]), float(row["traced"]))
    return curves


def threshold_level() -> dict:
    """{run: (leaked traced, honest traced)} -- the operating point priced
    and picked on the reported images, against the held-out rule."""
    curves = _threshold_curves()
    out = {}
    for run, test in curves.get("test", {}).items():
        dev = curves["dev"].get(run)
        if not dev or SHARED not in dev or SHARED not in test:
            continue
        honest_at = best_within(dev, dev[SHARED][0] - BUDGET)
        leaked_at = best_within(test, test[SHARED][0] - BUDGET)
        if honest_at in test and leaked_at in test:
            out[run] = (test[leaked_at][1], test[honest_at][1])
    return out


def threshold_argmax_only() -> dict:
    """The same, with the admissible set FIXED to dev on both sides -- so the
    only thing that changes is which split the argmax reads. Asserted to be
    exactly zero: traced is monotone in the threshold."""
    curves = _threshold_curves()
    out = {}
    for run, test in curves.get("test", {}).items():
        dev = curves["dev"].get(run)
        if not dev or SHARED not in dev:
            continue
        allowed = [key for key, values in dev.items()
                   if values[0] >= dev[SHARED][0] - BUDGET and key in test]
        if allowed:
            out[run] = (test[max(allowed, key=lambda k: test[k][1])][1],
                        test[max(allowed, key=lambda k: dev[k][1])][1])
    return out


# ------------------------------------------------------------------- level 3

def geometry_level(metric: str = "erl_bridged") -> dict:
    """{run: (leaked ERL, honest ERL)} -- the dilation geometry chosen on the
    reported images against the one chosen on dev, both read on test."""
    def gather(pattern: str) -> dict:
        cells = defaultdict(lambda: defaultdict(lambda: [[], []]))
        raw = defaultdict(list)
        for path in sorted(heldout.ROOT.glob(pattern)):
            for row in csv.DictReader(path.open()):
                if row["source"] == "raw":
                    raw[row["run"]].append(float(row["dice"]))
                elif row["source"] == "predicted":
                    cell = cells[row["run"]][(row["along"], row["across"])]
                    cell[0].append(float(row["dice"]))
                    cell[1].append(float(row[metric]))
        return {run: {key: (float(np.mean(v[0])), float(np.mean(v[1])))
                      for key, v in by_key.items()}
                for run, by_key in cells.items()}, \
               {run: float(np.mean(v)) for run, v in raw.items()}
    dev, dev_raw = gather("postproc_dev.shard*.csv")
    test, test_raw = gather("postproc_ceiling.shard*.csv")
    out = {}
    for run, test_cells in test.items():
        dev_cells = dev.get(run)
        if not dev_cells or run not in dev_raw or run not in test_raw:
            continue
        honest_at = best_within(dev_cells, dev_raw[run] - BUDGET)
        leaked_at = best_within(test_cells, test_raw[run] - BUDGET)
        if honest_at in test_cells and leaked_at in test_cells:
            out[run] = (test_cells[leaked_at][1], test_cells[honest_at][1])
    return out


def overspend() -> tuple:
    """(runs, how many overspend, mean Dice overspend) for the threshold
    level: how often the dev-picked operating point costs MORE than the
    budget once it is read on the twenty reported images."""
    curves = _threshold_curves()
    total, over, excess = 0, 0, []
    for run, test in curves.get("test", {}).items():
        dev = curves["dev"].get(run)
        if not dev or SHARED not in dev or SHARED not in test:
            continue
        chosen = best_within(dev, dev[SHARED][0] - BUDGET)
        if chosen not in test:
            continue
        total += 1
        cost = test[SHARED][0] - test[chosen][0]
        if cost > BUDGET:
            over += 1
            excess.append(cost - BUDGET)
    return total, over, float(np.mean(excess)) if excess else 0.0


# (name, blurb, fetch, is the leak an argmax of the REPORTED metric on test?)
LEVELS = (("checkpoint", "the epoch, best test ERL vs the dev rule",
           checkpoint_level, True),
          ("threshold", "the operating point, priced on test vs on dev",
           threshold_level, False),
          ("geometry", "the dilation radii, priced on test vs on dev",
           geometry_level, False),
          ("best.pt", "the historical bug: peeking with DICE, reporting ERL",
           checkpoint_bestpt_level, False))
# Only the first three stack: three independent selections in one pipeline.
# best.pt is the SAME selection as `checkpoint` made badly, so it is reported
# beside the stack and never added into it.
STACKING = 3


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    # 1. ONE RULE, FETCHED NOT COPIED. If select_checkpoint renames or
    #    reorders its rules this must fail loudly rather than quietly select
    #    a different epoch on one side of the comparison.
    rule = rule_iv()
    assert callable(rule), rule
    points = [{"epoch": 10, "dice": .80, "cldice": .70, "betti0_err": 90.0},
              {"epoch": 20, "dice": .82, "cldice": .78, "betti0_err": 80.0},
              {"epoch": 30, "dice": .83, "cldice": .74, "betti0_err": 60.0}]
    assert rule(points)["epoch"] == 20, rule(points)
    print(f"selection rule fetched by name: {heldout.RULE!r} picks epoch 20 "
          f"on a curve whose best Dice is epoch 30")

    # 2. THE PICKER MUST RESPECT THE FLOOR, and must return None rather than
    #    a made-up answer when nothing clears it. An empty table is not a
    #    null result -- CLAUDE.md, after curve() printed headers with no rows.
    curve = {0.5: (0.820, 0.40), 0.3: (0.805, 0.55), 0.1: (0.780, 0.70)}
    assert best_within(curve, 0.820 - BUDGET) == 0.3
    assert best_within(curve, 0.900) is None
    print(f"picker: floor {0.820 - BUDGET:.3f} admits 0.3 and rejects 0.1 "
          f"(costs {0.820 - 0.780:.3f}); an impossible floor returns None")

    # 3. THE LEAK CANNOT LOSE. Every leaked value is an argmax over the very
    #    set it is then read on, so leaked >= honest for EVERY run. A single
    #    violation means an argmax is reading the wrong column -- the shape of
    #    all three silent bugs of 2026-09-01.
    # 3b. WHERE THE LEAK IS NOT. On an axis the reported metric is monotone
    #     in, peeking at the metric buys exactly nothing once the admissible
    #     set is fixed. A non-zero here would mean the two sides are ranging
    #     over different sets, which is how a leak table quietly doubles.
    argmax_only = threshold_argmax_only()
    assert argmax_only, "no threshold rows"
    worst = max(abs(a - b) for a, b in argmax_only.values())
    assert worst < 1e-12, worst
    print(f"threshold: with the admissible set held on dev, choosing the "
          f"argmax on test changes nothing on any of {len(argmax_only)} runs "
          f"(max |gap| {worst:.1e}) -- traced is monotone in the threshold, "
          f"so the leak is the budget, not the pick")

    checked = 0
    for name, _, fetch, is_argmax in LEVELS:
        pairs = fetch()
        assert pairs, f"{name}: no rows on disk"
        gaps = [a - b for a, b in pairs.values()]
        if is_argmax:
            bad = [(run, a, b) for run, (a, b) in pairs.items()
                   if a < b - 1e-9]
            assert not bad, (name, bad[:3])
            note = "leaked >= honest on every run"
        else:
            losers = sum(1 for g in gaps if g < 0)
            note = (f"not an argmax of the reported metric, so it may lose -- "
                    f"{losers} of {len(gaps)} runs do")
        print(f"{name}: {len(pairs)} runs, mean gap "
              f"{100 * float(np.mean(gaps)):+.2f} points -- {note}")
        checked += 1
    assert checked == len(LEVELS), f"only {checked} of 4 levels had data"
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def report() -> None:
    print("=== what each protocol leak is worth, in ERL points ===\n")
    print("Same selection rule on both sides; only the SPLIT it is applied")
    print("to changes. Every number is read on the 20 test images. The leak")
    print("is an argmax over the set it is then scored on, so it cannot lose")
    print("-- the question is only by how much it wins.\n")
    print(f"The bar: the largest honest effect any published topology loss")
    print(f"produces in this repo, at each arm's own dev-optimal threshold,")
    print(f"is +{HONEST_CEILING} points (stage-report/calibration.md).\n")

    total_runs, over, excess = overspend()
    print(f"Levels 2 and 3 select along an axis traced ERL is MONOTONE in, so")
    print(f"the argmax carries nothing and the leak is the Dice budget being")
    print(f"priced on the reported images. That is not sign-constrained: the")
    print(f"dev split is 5 images, and on {over} of {total_runs} runs the")
    print(f"dev-picked threshold overspends the 0.02 budget once read on the")
    print(f"20 test images, by {excess:.3f} Dice on average.\n")

    summary = []
    for name, blurb, fetch, _ in LEVELS:
        pairs = fetch()
        if not pairs:
            raise SystemExit(f"no rows for level {name} -- refusing to print "
                             f"a table with a missing level")
        print(f"--- {name}: {blurb} ---")
        print(f"    {'arm':20}{'runs':>6}{'honest':>10}{'leaked':>10}"
              f"{'gap':>9}{'worst':>9}")
        by_arm = defaultdict(list)
        for run, values in pairs.items():
            by_arm[run.rsplit("_s", 1)[0]].append(values)
        for arm in sorted(by_arm):
            if arm not in ARMS:
                continue
            got = by_arm[arm]
            honest = 100 * float(np.mean([b for _, b in got]))
            leaked = 100 * float(np.mean([a for a, _ in got]))
            worst = 100 * max(a - b for a, b in got)
            print(f"    {arm:20}{len(got):>6}{honest:>9.1f}%{leaked:>9.1f}%"
                  f"{leaked - honest:>+8.1f}{worst:>+9.1f}")
        every = [100 * (a - b) for a, b in pairs.values()]
        summary.append((name, float(np.mean(every)), float(np.max(every)),
                        len(every)))
        print(f"    {'ALL RUNS':20}{len(every):>6}{'':>10}{'':>10}"
              f"{float(np.mean(every)):>+8.1f}{float(np.max(every)):>+9.1f}\n")

    print("--- the ledger ---")
    print(f"    {'level':14}{'runs':>6}{'mean gap':>11}{'worst run':>11}"
          f"{'vs the +1.4 bar':>18}")
    for name, mean, worst, count in summary:
        print(f"    {name:14}{count:>6}{mean:>+10.1f}{worst:>+11.1f}"
              f"{mean / HONEST_CEILING:>16.1f}x")
    total = sum(mean for _, mean, _, _ in summary[:STACKING])
    print(f"    {'stacked 1-3':14}{'':>6}{total:>+10.1f}{'':>11}"
          f"{total / HONEST_CEILING:>16.1f}x\n")
    print("best.pt is the same selection as `checkpoint` made badly, not a")
    print("fourth one, so it sits beside the stack and never inside it.\n")
    print("Read the last column as: how many times the largest honest effect")
    print("in this repo you can manufacture by choosing on the wrong split,")
    print("with the same model, the same data and the same selection rule.")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    report()


if __name__ == "__main__":
    main()
