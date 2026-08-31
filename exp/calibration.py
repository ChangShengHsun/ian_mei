"""Is the connectivity advantage a method effect, or where 0.5 happens to cut?

POST-HOC. This analysis was written on 2026-08-30 AFTER seeing the frontier
sweep, not before it. It is exploratory and is labelled as such wherever it is
reported. The confirmatory version is frontier.py's dev-threshold-matched
table, whose rule was fixed before any run.

THE OBSERVATION THAT PROMPTED IT. Every arm in this series is thresholded at
`prob >= 0.5`. Nothing justifies 0.5 beyond convention. Sweeping the threshold
showed the arms do not PEAK there:

    A_dice        Dice peaks at threshold 0.490
    K_focal_aug   Dice peaks at threshold 0.692

K_focal_aug is systematically under-confident -- its focal gate up-weights the
topology term exactly where the model hesitates, which flattens the
probability distribution. So at 0.5 it predicts more foreground than its own
optimum. More foreground buys connectivity. The question is whether anything
is left once that is removed.

THE TEST. Read every arm at ITS OWN Dice-maximising threshold, chosen on the
5 dev images. Every arm is then at its own best operating point, and none is
being penalised for a convention it never agreed to.

  If the advantage is a method effect, it survives.
  If it is calibration, the arms collapse onto one number.

WHY THIS IS NOT THE SAME AS matched_cost.py. That script matches by EPOCH at
threshold 0.5. This one matches by OPERATING POINT at a fixed epoch. An arm
can pass one and fail the other, and which it fails says what kind of thing it
is: an arm that only wins at a shared threshold won a calibration argument,
not a topology one.

  python exp/calibration.py --selftest
  python exp/calibration.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontier


def peak_threshold(dev_rows: list[dict]) -> float:
    """The threshold maximising DEV Dice. Chosen on dev, never on test."""
    return max(dev_rows, key=lambda row: row["dice"])["threshold"]


def read_at(test_rows: list[dict], threshold: float) -> dict | None:
    for row in test_rows:
        if abs(row["threshold"] - threshold) < 1e-9:
            return row
    return None


def collect() -> tuple[dict, dict]:
    """({arm: [traced at 0.5]}, {arm: [traced at its own dev peak]})."""
    test, dev = frontier._by_run(frontier.OUT), frontier._by_run(frontier.DEV)
    fixed, tuned = defaultdict(dict), defaultdict(dict)
    for run, dev_rows in dev.items():
        test_rows = test.get(run)
        if not test_rows:
            continue
        arm, seed = run.rsplit("_s", 1)
        half = read_at(test_rows, 0.5)
        best = read_at(test_rows, peak_threshold(dev_rows))
        if half is not None:
            fixed[arm][seed] = half
        if best is not None:
            tuned[arm][seed] = best
    return fixed, tuned


def decide(paired: list, per_seed: list) -> dict:
    """THE gate. One definition, so two scripts cannot drift apart.

    `paired` is the list of (mine, theirs) values the t-test runs on -- pairs
    of seeds in this file, pairs of (image, seed) in
    summarize_direction_ceiling.py, because that table has a value per image.
    `per_seed` is the mean difference within each seed, and it carries the
    sign rule: a mean and a t can both be large while one seed of six points
    the other way, which is the shape E5 was caught by.
    """
    diffs = np.array([a - b for a, b in paired], dtype=float)
    result = stats.ttest_rel([a for a, _ in paired], [b for _, b in paired])
    statistic = float(result.statistic)
    return {"mean": float(np.mean(per_seed)), "t": statistic,
            "seeds": len(per_seed), "per_seed": [float(d) for d in per_seed],
            "pairs": len(paired),
            "holds": bool(diffs.mean() > 0 and statistic > 2
                          and all(d > 0 for d in per_seed)
                          and len(per_seed) >= 3)}


def gate(mine: dict, theirs: dict, key: str = "traced") -> dict | None:
    """The repo's gate, paired on seed."""
    seeds = sorted(set(mine) & set(theirs))
    if len(seeds) < 3:
        return None
    paired = [(mine[s][key], theirs[s][key]) for s in seeds]
    return decide(paired, [a - b for a, b in paired])


def selftest() -> None:
    # 1. peak_threshold must read DICE, not traced. Reading traced would pick
    #    the most-foreground end every time and the whole test would be
    #    circular.
    rows = [{"threshold": 0.3, "dice": 0.70, "traced": 0.60},
            {"threshold": 0.5, "dice": 0.82, "traced": 0.40},
            {"threshold": 0.7, "dice": 0.78, "traced": 0.20}]
    assert peak_threshold(rows) == 0.5, peak_threshold(rows)
    print("  the peak is chosen by Dice, not by the metric under comparison")

    # 2. THE CASE THIS FILE EXISTS FOR. Two arms with identical frontiers that
    #    differ ONLY in calibration: one peaks at 0.5, the other at 0.7. At a
    #    shared 0.5 the second looks far better; at its own peak it must not.
    fixed, tuned = defaultdict(dict), defaultdict(dict)
    for seed in range(3):
        fixed["calibrated"][str(seed)] = {"traced": 0.30, "dice": 0.82}
        tuned["calibrated"][str(seed)] = {"traced": 0.30, "dice": 0.82}
        # Shifted arm at 0.5 is off its own peak, over-predicting: more
        # traced, less Dice. At its own peak it is the same model as above.
        fixed["shifted"][str(seed)] = {"traced": 0.44, "dice": 0.80}
        tuned["shifted"][str(seed)] = {"traced": 0.30, "dice": 0.82}
    at_half = gate(fixed["shifted"], fixed["calibrated"])
    at_peak = gate(tuned["shifted"], tuned["calibrated"])
    assert at_half["holds"] and abs(at_half["mean"] - 0.14) < 1e-9, at_half
    assert not at_peak["holds"] and abs(at_peak["mean"]) < 1e-9, at_peak
    print(f"  a pure calibration shift passes at 0.5 ({at_half['mean']:+.1%}) "
          f"and vanishes at its own peak ({at_peak['mean']:+.1%})")

    # 3. And a real frontier gain must survive both.
    for seed in range(3):
        fixed["better"][str(seed)] = {"traced": 0.38, "dice": 0.82}
        tuned["better"][str(seed)] = {"traced": 0.38, "dice": 0.82}
    assert gate(fixed["better"], fixed["calibrated"])["holds"]
    assert gate(tuned["better"], tuned["calibrated"])["holds"]
    print("  a genuine frontier gain survives both readings")

    assert gate({"0": {"traced": 1.0}}, {"0": {"traced": 0.0}}) is None
    print("  fewer than three seeds is not a verdict")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    fixed, tuned = collect()
    if not fixed:
        raise SystemExit("no frontier data; run exp/frontier.py and "
                         "exp/frontier.py --dev first")
    order = [a for a in ("A_dice", "H_aug", "G_focal", "F_gated",
                         "K_focal_aug") if a in fixed]
    order += sorted(a for a in fixed if a not in order)

    print("POST-HOC (written after seeing the frontier sweep, not before).\n")
    print("Every arm read twice: at the conventional threshold 0.5, and at "
          "the threshold that")
    print("maximises its DEV Dice. Traced fraction on the 20 test images "
          "both times.\n")
    header = (f"  {'arm':<20}{'thr*':>7}{'dice@.5':>9}{'traced@.5':>11}"
              f"{'dice@thr*':>11}{'traced@thr*':>13}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arm in order:
        seeds = sorted(set(fixed[arm]) & set(tuned[arm]))
        if len(seeds) < 3:
            continue
        peak = np.mean([tuned[arm][s]["threshold"] for s in seeds])
        print(f"  {arm:<20}{peak:7.3f}"
              f"{np.mean([fixed[arm][s]['dice'] for s in seeds]):9.4f}"
              f"{np.mean([fixed[arm][s]['traced'] for s in seeds]):11.1%}"
              f"{np.mean([tuned[arm][s]['dice'] for s in seeds]):11.4f}"
              f"{np.mean([tuned[arm][s]['traced'] for s in seeds]):13.1%}")

    print(f"\n  Against A_dice, paired on seed, the repo's gate:\n")
    print(f"  {'arm':<20}{'at 0.5':>10}{'':>3}{'at own peak':>13}{'':>3}"
          f"  what that means")
    print("  " + "-" * 76)
    for arm in order:
        if arm == "A_dice":
            continue
        half = gate(fixed[arm], fixed["A_dice"])
        peak = gate(tuned[arm], tuned["A_dice"])
        if half is None or peak is None:
            continue
        verdict = ("frontier" if peak["holds"] else
                   "calibration" if half["holds"] else "nothing")
        print(f"  {arm:<20}{half['mean']:>+10.1%}"
              f"{'*' if half['holds'] else ' ':>3}"
              f"{peak['mean']:>+13.1%}{'*' if peak['holds'] else ' ':>3}"
              f"  {verdict}")
    print("\n  * passes the gate (t > 2, every seed agreeing in sign, >= 3 "
          "seeds).")
    print("  frontier    = ahead at its own best operating point. A method "
          "effect.")
    print("  calibration = ahead only at the shared threshold 0.5. The arm "
          "is under-")
    print("                confident, and 0.5 therefore predicts more "
          "foreground for it.")
    print("  nothing     = ahead under neither reading.")


if __name__ == "__main__":
    main()
