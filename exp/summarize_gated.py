"""E7: did gating the topology loss by image contrast actually help?

Written and committed BEFORE the six runs finished, with the decision rule in
VERDICT below fixed in advance. This series has overturned itself three times
already (stage 0's free clDice win, E3's vk ranking, E6's structure credit),
and every one of those was caught by a control rather than by a better model.
The cheapest remaining protection is to write down what counts as success
before seeing the numbers, so main() prints a verdict it cannot argue with.

The paired unit is one (image, seed) pair, 20 x 3 = 60, which is the unit E2
used for its +0.0213 / t=9.74. Comparisons are always same-image, same-seed:
the seeds differ by initialisation only, so pairing removes the image
difficulty variance that otherwise swamps a 0.02 effect.

  python exp/summarize_gated.py            # the report
  python exp/summarize_gated.py --selftest # check paired() against scipy

Reads results/stratify.csv, which stratify.py must have been re-run to produce
after F_gated and G_focal exist.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stratify

RESULTS = Path(__file__).resolve().parent / "results"
SEEDS = (0, 1, 2)
# Primary reading is the FILTERED one. E4 measured that dropping small
# components buys 2.4-2.6x what the topology loss buys on betti-0, and the
# contrast gate hands its highest weight to empty background, so an unfiltered
# gain is exactly what a speckle suppressor would produce. If F only wins at
# min_size 0, that is stage 0's mistake a second time.
PRIMARY_MIN_SIZE = 20
DIM, CLEAR = "Q1_dimmest", "Q4_clearest"


def load() -> list[dict]:
    rows = list(csv.DictReader((RESULTS / "stratify.csv").open()))
    for row in rows:
        for key in ("dice", "cldice", "tprec", "tsens"):
            row[key] = float(row[key])
        for key in ("breaks", "gt_px", "skel_px", "min_size"):
            row[key] = int(row[key])
    return rows


def series(rows: list[dict], config: str, band: str, min_size: int,
           key: str) -> dict:
    """One value per (image, seed), keyed so two configs can be lined up."""
    return {(r["image"], r["run"].rsplit("_s", 1)[1]): r[key] for r in rows
            if r["config"] == config and r["band"] == band
            and r["min_size"] == min_size}


def paired(rows, better, worse, band, min_size, key) -> dict:
    """better minus worse over the pairs both configs share.

    Returns the mean difference, its t statistic, and how many pairs moved in
    the positive direction. Direction is left to the caller: this function
    reports the arithmetic difference and nothing about which sign is good.
    """
    left, right = (series(rows, better, band, min_size, key),
                   series(rows, worse, band, min_size, key))
    keys = sorted(set(left) & set(right))
    diff = np.array([left[k] - right[k] for k in keys], dtype=float)
    if len(diff) < 2 or diff.std() == 0:
        # A zero-variance difference is a real answer (identical predictions),
        # not a failure, so report t as 0 rather than dividing by zero.
        return {"mean": float(diff.mean()) if len(diff) else float("nan"),
                "t": 0.0, "wins": int((diff > 0).sum()), "n": len(diff)}
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    return {"mean": float(diff.mean()), "t": float(t),
            "wins": int((diff > 0).sum()), "n": len(diff)}


def per_seed(rows, better, worse, band, min_size, key) -> list[float]:
    """The same difference, computed inside each seed separately.

    Added 2026-08-20, BEFORE any F_gated or G_focal run finished, because E5
    had just been caught by exactly the failure this guards against. There,
    a paired test over 700 (image, seed) pairs returned p = 3e-4 for a
    difference whose SIGN flipped between the only two seeds: -0.0271 at seed 0
    and +0.0102 at seed 1. The images are not independent replicates of a
    training run -- 700 pairs from two models is two models -- so the t
    statistic was measuring image agreement, which is high, and reporting it
    as evidence about the loss, which it is not.

    Pairing on the image is still worth doing: it removes the image difficulty
    variance, which is what makes a 0.02 effect visible at all. It just cannot
    stand alone. So the verdict now needs BOTH the paired t and every seed
    agreeing in sign, and with three seeds that second condition is the one
    that is hard to satisfy by luck.
    """
    out = []
    for seed in SEEDS:
        left = {k: v for k, v in
                series(rows, better, band, min_size, key).items()
                if k[1] == str(seed)}
        right = {k: v for k, v in
                 series(rows, worse, band, min_size, key).items()
                 if k[1] == str(seed)}
        keys = sorted(set(left) & set(right))
        if keys:
            out.append(float(np.mean([left[k] - right[k] for k in keys])))
    return out


def line(label: str, result: dict, decimals: int = 4) -> str:
    return (f"{label:26}{result['mean']:+10.{decimals}f}"
            f"{result['t']:9.2f}{result['wins']:6d}/{result['n']:<4d}")


def report(rows: list[dict], better: str, worse: str) -> None:
    print(f"\n=== {better} - {worse}（配對，每格 60 對）===")
    print(f"{'':26}{'差值':>10}{'t':>9}{'正向':>11}")
    for min_size in stratify.MIN_SIZES:
        state = "unfiltered" if min_size == 0 else f"filtered<{min_size}"
        for key, decimals in (("dice", 4), ("cldice", 4), ("breaks", 1)):
            for band in stratify.BANDS:
                result = paired(rows, better, worse, band, min_size, key)
                print(line(f"{state} {key} {band}", result, decimals))
            print()


def verdict(rows: list[dict]) -> None:
    """The rule fixed in advance. Every criterion is a sign plus |t| > 2.

    n = 60, so |t| > 2.00 is roughly p < 0.05 and |t| > 2.66 is p < 0.01.
    Criterion 2 is non-inferiority: E2 measured clDice LOSING 0.0070 Dice in
    the clear bands, and the whole point of gating is to stop paying that, so
    the bar there is "not significantly worse", not "better".
    """
    checks = []

    def both(better, worse, band, want):
        """The paired test AND seed agreement. See per_seed for why."""
        result = paired(rows, better, worse, band, PRIMARY_MIN_SIZE, "dice")
        seeds = per_seed(rows, better, worse, band, PRIMARY_MIN_SIZE, "dice")
        result["seeds"] = seeds
        if want == "better":
            passed = (result["mean"] > 0 and result["t"] > 2.0
                      and all(s > 0 for s in seeds))
        else:  # non-inferiority: it may not be significantly worse
            passed = not (result["mean"] < 0 and result["t"] < -2.0
                          and all(s < 0 for s in seeds))
        return passed, result

    passed, gain = both("F_gated", "B_cldice", DIM, "better")
    checks.append(("1. F beats B in the dimmest band, filtered "
                   "(all 3 seeds must agree)", passed, gain))

    passed, cost = both("F_gated", "B_cldice", CLEAR, "not_worse")
    checks.append(("2. F does not lose to B in the clearest band",
                   passed, cost))

    passed, signal = both("F_gated", "G_focal", DIM, "better")
    checks.append(("3. F beats G in the dimmest band (signal source matters, "
                   "all 3 seeds must agree)", passed, signal))

    print("\n=== 事前判準（結果出來前寫死）===")
    for label, passed, result in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        seeds = " ".join(f"{s:+.4f}" for s in result.get("seeds", []))
        print(f"         {result['mean']:+.4f}  t={result['t']:.2f}  "
              f"{result['wins']}/{result['n']}   per seed: {seeds}")

    passed = [c[1] for c in checks]
    if all(passed):
        print("\n  -> 主張成立：影像端的對比權重改善了暗處且沒有賠掉亮處，"
              "而且模型端的權重做不到同一件事。")
    elif passed[0] and passed[1] and not passed[2]:
        print("\n  -> 較弱的結論：加權有用，但訊號來源不重要。"
              "E1' 的預測在訓練層級沒有實現，這件事本身要解釋。")
    elif not passed[0]:
        print("\n  -> 主張不成立。")
        unfiltered = paired(rows, "F_gated", "B_cldice", DIM, 0, "dice")
        if unfiltered["mean"] > 0 and unfiltered["t"] > 2.0:
            print("     注意：未過濾時 F 是贏的"
                  f"（{unfiltered['mean']:+.4f}, t={unfiltered['t']:.2f}），"
                  "過濾後消失 = 它買到的是雜點抑制，不是拓樸。"
                  "這正是 stage 0 犯過的那個錯。")
    else:
        print("\n  -> 混合結果，逐條看上面。")


def selftest() -> None:
    """paired() must match scipy on the same numbers, and must line the two
    configs up by (image, seed) rather than by row order -- a mismatch there
    would silently compare image 1 against image 7 and still look plausible."""
    rows = []
    rng = np.random.default_rng(0)
    truth = {}
    for image in [f"img{i}" for i in range(20)]:
        for seed in SEEDS:
            base = rng.normal()
            for config, offset in (("F_gated", 0.03), ("B_cldice", 0.0)):
                truth[(config, image, seed)] = base + offset + rng.normal(0, 0.01)
                rows.append({"config": config, "run": f"{config}_s{seed}",
                             "image": image, "band": "Q1_dimmest",
                             "min_size": 20,
                             "dice": truth[(config, image, seed)]})
    # Shuffle so anything relying on row order breaks here.
    rng.shuffle(rows)
    result = paired(rows, "F_gated", "B_cldice", "Q1_dimmest", 20, "dice")
    keys = [(i, s) for i in [f"img{j}" for j in range(20)] for s in SEEDS]
    left = [truth[("F_gated", i, s)] for i, s in keys]
    right = [truth[("B_cldice", i, s)] for i, s in keys]
    expected = stats.ttest_rel(left, right)
    assert result["n"] == 60, result
    assert abs(result["t"] - expected.statistic) < 1e-9, (result, expected)
    assert abs(result["mean"] - 0.03) < 0.005, result
    print(f"paired() matches scipy: t={result['t']:.6f} against "
          f"{expected.statistic:.6f}, n={result['n']}")

    # An unpaired comparison of the same data would still be significant, so
    # agreement above is not enough on its own; check that scrambling the
    # pairing changes the answer, i.e. the pairing is load-bearing.
    scrambled = [{**r, "image": f"img{rng.integers(20)}"} for r in rows]
    other = paired(scrambled, "F_gated", "B_cldice", "Q1_dimmest", 20, "dice")
    assert abs(other["t"] - result["t"]) > 1.0, (other, result)
    print(f"pairing is load-bearing: scrambled t={other['t']:.2f} "
          f"against paired t={result['t']:.2f}")

    # The E5 failure in the shape E7 can actually suffer it: two seeds favour F
    # and one favours B, and the paired t is large anyway. This is the case the
    # seed gate exists to reject, so it is asserted rather than trusted.
    #
    # E5's own offsets (-0.027, +0.010) give |t| = 0.99 here rather than the
    # 11.9 they gave there, because the seed disagreement enters the variance
    # and only sqrt(n) fights it: at n = 700 that is enough to reach p = 3e-4,
    # at n = 60 it is not. So E7's smaller design is less exposed than E5 was,
    # not immune -- an effect large enough to matter still clears |t| > 2 on
    # two seeds out of three.
    flipped = []
    offsets = {0: -0.005, 1: +0.030, 2: +0.030}
    for image in [f"img{i}" for i in range(20)]:
        for seed in SEEDS:
            base = rng.normal()
            for config, offset in (("F_gated", offsets[seed]),
                                   ("B_cldice", 0.0)):
                flipped.append({"config": config, "run": f"{config}_s{seed}",
                                "image": image, "band": "Q1_dimmest",
                                "min_size": 20,
                                "dice": base + offset + rng.normal(0, 0.001)})
    seeds = per_seed(flipped, "F_gated", "B_cldice", "Q1_dimmest", 20, "dice")
    naive = paired(flipped, "F_gated", "B_cldice", "Q1_dimmest", 20, "dice")
    print(f"seed-flip case: per seed {[round(s, 4) for s in seeds]}, "
          f"paired t={naive['t']:.2f}")
    assert not all(s > 0 for s in seeds), seeds
    assert abs(naive["t"]) > 2.0, naive
    print("  -> a large |t| with seeds disagreeing is rejected by the gate")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    rows = load()
    present = {r["config"] for r in rows}
    missing = {"F_gated", "G_focal"} - present
    if missing:
        print(f"still untrained: {sorted(missing)} -- rerun stratify.py "
              "after those runs exist")
        return
    for better, worse in (("F_gated", "B_cldice"), ("G_focal", "B_cldice"),
                          ("F_gated", "G_focal"), ("F_gated", "A_dice")):
        report(rows, better, worse)
    verdict(rows)


if __name__ == "__main__":
    main()
