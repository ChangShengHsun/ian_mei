"""How many seeds does a claim survive? The gate's third condition, priced.

WRITTEN AND SELFTESTED 2026-09-04, AFTER a pre-registered prediction failed.

WHAT FAILED. exp/run_anchor.sh predicted, in writing before the runs existed,
that the two cells reading HOLDS at twelve seeds would survive to
twenty-four, because both read 100% in seed_stability's resampling curve:

    hrf      split    H_aug_clw vs H_aug at 0.5   +3.5% t 8.46   100% at every k
    vessmap  bridged  H_aug     vs A_dice at own  +1.9% t 3.62   100% at k>=8

BOTH DIED. And across all three transfer datasets the count of HOLDS cells
went 6 -> 2 when the seeds doubled, with 0 going the other way. So the
resampling curve does not predict the anchor, and run_anchor.sh's own
instruction was to report that and stop selling the curve as a planning tool.

WHY IT FAILED, WHICH IS THE POINT. The curve resamples the seeds ALREADY ON
DISK. A cell reading 100% at every k has ZERO dissenting seeds among the n it
has -- and a rate cannot be estimated from a sample containing none of the
event. The rule of three puts the 95% upper bound on a rate after 0 events in
n trials at 3/n: with n=12 that is p < 0.25, and at p = 0.25 the chance of
surviving 24 seeds is 0.75^24 = 0.001. The curve reading 100% at twelve was
therefore entirely consistent with certain death at twenty-four. The
prediction was statistically naive, not the measurement.

WHAT THIS FILE MEASURES INSTEAD. The gate's third condition -- every per-seed
difference positive -- survives n seeds with probability (1-p)^n, where p is
the rate at which a fresh seed dissents. That is estimable, from the seeds on
disk, as d/n. So every cell gets:

    d/n     how many seeds dissent, out of those run
    p       the point estimate, d/n
    hi      the 95% upper bound on p (Wilson, or 3/n by the rule of three
            when d = 0), because for d = 0 the point estimate is useless
    n50     the seed count at which survival drops below 50%, at p
    n50_hi  the same at the upper bound -- the honest planning number

The headline this produces is not "the gate is not monotone". It is: for any
p > 0 the survival probability goes to zero, so EVERY cell in this repo dies
at some seed count, and passing the gate is a statement about how few seeds
were run. n50 says how few.

  python exp/seed_survival.py --selftest
  python exp/seed_survival.py --report

Writes nothing; the report is the artefact.
"""
import sys
from math import comb, log, log1p, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_stability as stability
import transfer_calibration as calib

DATASETS = calib.DATASETS
# Wilson at 95%. Not Wald: at d = 0 Wald gives the interval [0, 0], which is
# the exact false confidence this file exists to remove.
Z = 1.959964


def wilson_upper(dissent: int, total: int) -> float:
    """Upper end of the 95% Wilson interval for a rate.

    At dissent = 0 this is close to, and slightly below, the rule of three's
    3/n; the selftest asserts they agree to within a factor of 1.3 so that a
    reader can use either mental model.
    """
    if total == 0:
        return 1.0
    phat = dissent / total
    denom = 1.0 + Z * Z / total
    centre = phat + Z * Z / (2 * total)
    spread = Z * sqrt(phat * (1 - phat) / total + Z * Z / (4 * total * total))
    return min(1.0, (centre + spread) / denom)


def half_life(rate: float) -> float:
    """Seeds at which (1 - rate)^n first drops below 0.5. inf when rate = 0.

    log1p(-rate), never log(1 - rate): the parametric estimate reaches 1e-20
    for a decisive cell, and there `1.0 - rate` rounds to exactly 1.0, whose
    log is 0.0 and divides by zero. Caught by the selftest 2026-09-04.
    """
    if rate <= 0.0:
        return float("inf")
    if rate >= 1.0:
        return 0.0
    return log(0.5) / log1p(-rate)


def per_seed_diffs(per_seed: dict) -> list[float]:
    """One number per seed: that seed's mean paired difference."""
    return [float(np.mean([mine - theirs for mine, theirs in pairs]))
            for pairs in per_seed.values()]


def dissent(per_seed: dict) -> tuple[int, int]:
    """(dissenting seeds, seeds) for one cell, by the gate's own sign rule."""
    diffs = per_seed_diffs(per_seed)
    return sum(1 for value in diffs if value <= 0.0), len(diffs)


def parametric_rate(diffs: list[float]) -> float:
    """P(a fresh seed dissents), from the spread of the per-seed differences.

    WHY A SECOND ESTIMATE. The count-based bound is nearly useless at d = 0:
    the rule of three says only p < 3/n, which at n = 12 admits p = 0.25 and
    therefore near-certain death by 24 seeds. But d = 0 arising from an effect
    ten standard deviations clear of zero is not the same situation as d = 0
    arising from four seeds that happened to agree, and a count cannot tell
    them apart. Modelling the per-seed differences as normal gives
    P(next < 0) = Phi(-mean/sd), which distinguishes them.

    This is a MODEL and the count is not, so the report prints both. Where
    they disagree by orders of magnitude, the cell is not robustly safe --
    it is merely under-sampled, and which number to believe depends on
    whether the per-seed differences look normal. Where both say the same
    thing, the cell is settled.
    """
    if len(diffs) < 3:
        return float("nan")
    mean = float(np.mean(diffs))
    # ddof=1: the sample sd, because n is 12 or 24, not large.
    spread = float(np.std(diffs, ddof=1))
    if spread <= 0.0:
        return 0.0 if mean > 0.0 else 1.0
    from math import erfc
    # Phi(-mean/sd) written with erfc to stay accurate far in the tail, where
    # 1 - Phi(x) loses every significant digit to cancellation.
    return 0.5 * erfc(mean / (spread * sqrt(2.0)))


def selftest() -> None:
    # 1. THE CLOSED FORM seed_stability's ladders turned out to be. With d
    #    dissenters among n, an all-positive k-subset must avoid all of them,
    #    so P(k) = C(n-d, k)/C(n, k). The observed rows 75/67/58/50/33/17/0
    #    and 55/42/32/23/9/2/0 are d=1 and d=2 at n=12; assert both, because
    #    this identity is what licenses reading a pass rate as a count.
    for dissenters, want in ((1, [75, 67, 58, 50, 33, 17, 0]),
                             (2, [55, 42, 32, 23, 9, 2, 0])):
        got = [round(100 * comb(12 - dissenters, k) / comb(12, k))
               for k in (3, 4, 5, 6, 8, 10, 12)]
        assert got == want, (dissenters, got, want)
    print("pass-rate ladders are C(n-d,k)/C(n,k): d=1 and d=2 reproduced")

    # 2. THE RULE OF THREE, which is the whole reason d = 0 is not safety.
    #    Wilson's upper bound after zero events must be near 3/n, and must
    #    NOT be zero.
    for total in (12, 24, 48):
        upper = wilson_upper(0, total)
        assert upper > 0.0, total
        assert 0.77 < upper / (3.0 / total) < 1.3, (total, upper, 3.0 / total)
    print(f"zero dissenters in 12 seeds still allows p up to "
          f"{wilson_upper(0, 12):.3f} (rule of three: {3/12:.3f})")

    # 3. AND WHAT THAT UPPER BOUND IMPLIES. The prediction that failed said a
    #    cell reading 100% at twelve would survive twenty-four. At the upper
    #    bound the survival chance is a fraction of a percent, so the
    #    prediction was never supported by the curve it cited.
    survival = (1.0 - wilson_upper(0, 12)) ** 24
    assert survival < 0.01, survival
    print(f"a cell with 0 dissenters in 12 survives 24 seeds with "
          f"probability >= {survival:.4f} at the 95% upper bound -- "
          f"which is why run_anchor.sh's prediction 2 failed")

    # 4. HALF-LIFE IS MONOTONE AND HAS THE RIGHT ANCHORS.
    assert half_life(0.0) == float("inf")
    assert abs(half_life(0.5) - 1.0) < 1e-9, half_life(0.5)
    rates = [0.04, 0.08, 0.17, 0.25, 0.5]
    lives = [half_life(r) for r in rates]
    assert lives == sorted(lives, reverse=True), lives
    print("half-life anchors: p=0.5 -> 1.0 seeds, p=0.04 -> "
          f"{half_life(0.04):.1f}, and it decreases with p")

    # 5. THE PARAMETRIC ESTIMATE MUST SEPARATE THE TWO KINDS OF d = 0.
    #    Four seeds that barely agree, and twelve seeds ten sigma clear of
    #    zero, both give d = 0 and both give the same rule-of-three bound.
    #    The whole reason for a second estimate is that it must not.
    barely = [0.001, 0.002, 0.0005, 0.0015]
    decisive = [0.05 + 0.001 * step for step in range(12)]
    weak, strong = parametric_rate(barely), parametric_rate(decisive)
    assert 0.01 < weak < 0.5, weak
    assert strong < 1e-6, strong
    assert half_life(strong) > 1e5, half_life(strong)
    # The separation is the claim, not the absolute values: both cells have
    # d = 0 and therefore the identical rule-of-three bound.
    assert wilson_upper(0, len(barely)) > wilson_upper(0, len(decisive))
    assert weak > 1000 * strong, (weak, strong)
    print(f"parametric rate separates the two kinds of d=0: barely-positive "
          f"{weak:.3f} (half-life {half_life(weak):.0f} seeds) vs decisive "
          f"{strong:.2e} (half-life {half_life(strong):.0e})")
    #    And it must agree with the count when dissenters are plentiful.
    mixed = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert 0.3 < parametric_rate(mixed) < 0.7, parametric_rate(mixed)
    #    A degenerate cell (no spread at all) must not divide by zero.
    assert parametric_rate([0.5] * 5) == 0.0
    assert parametric_rate([-0.5] * 5) == 1.0
    print("parametric rate handles half-negative and zero-variance cells")

    # 6. THE CELL READER MUST AGREE WITH seed_stability's OWN, on real data.
    #    A second way of counting the same thing is how two tables drift.
    for dataset in DATASETS:
        cells = stability.calibration_cells(dataset)
        assert cells, dataset
        for per_seed in cells.values():
            dissenters, total = dissent(per_seed)
            assert 0 <= dissenters <= total and total > 0
            rate = stability.pass_rate(per_seed, total)
            # All-positive over every seed on disk is exactly "d == 0".
            assert (rate == 1.0) == (dissenters == 0), (dissenters, rate)
        print(f"  {dataset}: {len(cells)} cells, d==0 matches pass_rate==1 "
              f"in every one")
    print("all checks passed")


def report() -> None:
    print("=== how many seeds does a claim survive? ===\n")
    print("The gate's third condition -- every per-seed difference positive")
    print("-- survives n seeds with probability (1-p)^n, where p is the rate")
    print("at which a fresh seed dissents. For any p > 0 that goes to zero.")
    print("So every cell here dies at some seed count, and `HOLDS` is a")
    print("statement about how few seeds were run. n50 says how few.\n")
    print("d=0 is NOT p=0. A rate cannot be estimated from a sample with no")
    print("events; the rule of three caps it at 3/n. That is why")
    print("run_anchor.sh's pre-registered prediction 2 -- that two cells")
    print("reading 100% in the resampling curve would survive to 24 seeds --")
    print("FAILED on both. The curve could not have supported it.\n")
    print("n50 = seeds at which survival drops below 50%, at the point")
    print("estimate. n50_hi = the same at the 95% upper bound: the honest")
    print("planning number, and the only one defined when d = 0.\n")

    def measure(cells: dict) -> list:
        rows = []
        for key, per_seed in sorted(cells.items()):
            diffs = per_seed_diffs(per_seed)
            dissenters, total = dissent(per_seed)
            rows.append((key, dissenters, total, dissenters / total,
                         wilson_upper(dissenters, total),
                         parametric_rate(diffs)))
        return rows

    def show(title: str, rows: list) -> None:
        if not rows:
            print(f"--- {title}: no cells ---\n")
            return
        print(f"--- {title}: {len(rows)} cells, {rows[0][2]} seeds on disk ---")
        print(f"    {'cell':44}{'d/n':>8}{'p 95% hi':>10}{'p model':>10}"
              f"{'n50_hi':>9}{'n50_model':>11}")
        for key, dissenters, total, _, upper, model in rows:
            name = "/".join(str(part) for part in key)
            life = half_life(model)
            # A decisive cell's half-life reaches 1e26 seeds, which is not a
            # number anyone reads -- and unformatted it overruns the column
            # and glues itself to the previous one. Anything past a million
            # says the same thing: this comparison is not seed-limited.
            if life == float("inf"):
                shown = "inf"
            elif life > 1e6:
                shown = ">1e6"
            else:
                shown = f"{life:.0f}"
            print(f"    {name[:43]:44}{f'{dissenters}/{total}':>8}"
                  f"{upper:>10.2f}{model:>10.2e}{half_life(upper):>9.1f}"
                  f"{shown:>11}")
        print()

    everything = []
    for dataset in DATASETS:
        rows = measure(stability.calibration_cells(dataset))
        show(f"{dataset} calibration", rows)
        everything.extend(rows)

    # The DRIVE headline gets the same audit. composition.py holds the claim
    # the paper actually makes -- that lowering the threshold matches the
    # post-processing layer -- and a headline that has not been seed-audited
    # is exactly the kind of cell this file exists to find.
    drive = measure(stability.drive_cells(0.02))
    show("drive composition at -0.02 Dice", drive)

    def summarise(title: str, rows: list) -> None:
        alive = [r for r in rows if r[1] == 0]
        rates = sorted(r[3] for r in rows)
        median = rates[len(rates) // 2]
        print(f"--- {title} ---")
        print(f"    cells                                      {len(rows)}")
        print(f"    still zero dissenters at the seeds run     {len(alive)}")
        print(f"    median observed dissent rate               {median:.2f}")
        print(f"    half-life at that rate                     "
              f"{half_life(median):.1f} seeds")
        if alive:
            # The model is what separates "safe" from "under-sampled": the
            # count-based bound is the same 3/n for every one of these.
            models = sorted(r[5] for r in alive)
            settled = sum(1 for m in models if half_life(m) > 1000)
            print(f"    of those, decisive under the normal model")
            print(f"    (half-life > 1000 seeds)                   "
                  f"{settled} of {len(alive)}")
            print(f"    median model rate among them               "
                  f"{models[len(models) // 2]:.2e}")
        print()

    summarise("transfer calibration, all three datasets", everything)
    summarise("DRIVE composition at -0.02 Dice", drive)

    rates = sorted(r[3] for r in everything)
    median = rates[len(rates) // 2]
    print("--- how to read this ---")
    print("Two estimates of the same rate, and the gap between them is the")
    print("finding. The count-based bound is identical (3/n) for every cell")
    print("with d = 0, so on its own it cannot tell a settled comparison from")
    print("an under-sampled one. The normal model can, and it splits the")
    print("d = 0 cells cleanly: the transfer table's survivors sit near the")
    print("gate and die within a few more seeds; the DRIVE `lower` cells sit")
    print("many standard deviations clear of zero and would survive")
    print("thousands. Same verdict word, different objects.")
    print()
    print("So: at the transfer table's median dissent rate a comparison has")
    print(f"an even chance of failing the all-positive rule after "
          f"{half_life(median):.0f} seed(s).")
    print("`HOLDS` without a seed count is not a comparable claim, and the")
    print("third condition is a count of dissenters rather than a test of an")
    print("effect. Where the model says the half-life is in the thousands,")
    print("the claim is safe -- but that is the model saying so, not the gate.")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    raise SystemExit("pass --selftest or --report")


if __name__ == "__main__":
    main()
