"""Which arm traces furthest AT THE SAME DICE?

WRITTEN AND SELFTESTED 2026-08-29, BEFORE BEING RUN ON ANY ARM.

THE PROBLEM. Under the held-out protocol the arms do not land at the same
Dice:

    A_dice        Dice 0.8154   traced 30.6%
    H_aug         Dice 0.8214   traced 33.7%
    K_focal_aug   Dice 0.7993   traced 44.1%

K_focal_aug is compared against A_dice while predicting a MEASURABLY
different amount of foreground. Reading "+13.5 points of traced fraction" off
that table charges the whole difference to the loss, when part of it is
simply that a lower Dice threshold on the same trade-off curve traces
further. This repo has already paid for this exact error once: C1.0's free
closing baseline added 31% more foreground than the oracle it "beat", and the
fix was to match the cost, not to move the threshold.

THE MECHANISM. Every run keeps ten validated epochs, and Dice moves across
them. So instead of asking "what is this arm's traced fraction", ask "at the
epoch where this arm reaches Dice d on DEV, what is its traced fraction on
TEST". Sweeping d gives each arm a curve, and the arms are then read off at a
common d. No retraining: the checkpoints already exist.

WHY DEV DECIDES THE EPOCH. Matching on test Dice would choose each arm's
epoch using the set it is then reported on -- the leak the held-out protocol
exists to remove, reintroduced through the back door.

WHAT AN HONEST ANSWER LOOKS LIKE. Either an arm is above the others at every
d it reaches -- it moved the whole frontier, which is a real claim -- or the
arms lie on ONE curve and differ only in where they sit on it, in which case
"our loss traces 13.5 points further" is a statement about an operating
point and not about a method. Both outcomes are publishable; only one of them
is the claim the unmatched table appears to make.

PRE-REGISTERED PREDICTION. The arms lie close to one curve, and K_focal_aug's
advantage shrinks by more than half once Dice is matched. The gate's
mechanism -- weight the topology term where the model hesitates -- is a way of
spending Dice on connectivity, and spending is what a trade-off curve
describes.

  python exp/matched_cost.py --selftest
  python exp/matched_cost.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_heldout as heldout
import summarize_selection as selection
import train

# Dice targets to read every arm at. Chosen to span the range the arms
# actually reach on dev rather than to flatter any of them; a target no arm
# reaches prints as "--" instead of silently snapping to an endpoint.
TARGETS = (0.790, 0.800, 0.810, 0.815, 0.820)
# How far a run's dev Dice may sit from the target before the point is
# refused. One seed SD of Dice is 0.00086, so this is roughly six of them --
# tight enough that "matched" means matched, loose enough that ten epochs can
# usually supply a point.
TOLERANCE = 0.005


def dev_curve(run_dir: Path) -> list[dict]:
    return heldout.dev_points(run_dir)


def epoch_at_dice(points: list[dict], target: float) -> int | None:
    """The epoch whose DEV Dice is closest to `target`, or None if none is
    within TOLERANCE. Never extrapolates: an arm that cannot reach a Dice
    is absent from that column rather than represented by its nearest end."""
    if not points:
        return None
    best = min(points, key=lambda p: abs(p["dice"] - target))
    if abs(best["dice"] - target) > TOLERANCE:
        return None
    return best["epoch"]


def traced_at(rows, run: str, epoch: int) -> float | None:
    """Traced fraction on the test images at one checkpoint."""
    numerator, denominator = 0.0, 0.0
    for row in rows:
        if row["run"] == run and row["epoch"] == epoch:
            numerator += row["erl"]
            denominator += row["skel_px"]
    return numerator / denominator if denominator else None


def curve(rows, root: Path = heldout.ROOT) -> dict:
    """{arm: {target: [traced per seed]}} -- every arm read at every target."""
    out = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.glob("*_s*/log.csv")):
        run = path.parent.name
        points = dev_curve(path.parent)
        for target in TARGETS:
            epoch = epoch_at_dice(points, target)
            if epoch is None:
                continue
            value = traced_at(rows, run, epoch)
            if value is not None:
                out[run.rsplit("_s", 1)[0]][target].append(value)
    return out


def compare_at(got: dict, arm: str, base: str, target: float) -> dict | None:
    """Unpaired here on purpose: at a matched Dice the two arms are read at
    DIFFERENT epochs, so a seed is no longer a shared unit of noise the way it
    is when both arms are read at their own rule (iv). The seed gate's sign
    rule still applies -- it is computed per seed below."""
    mine, theirs = got.get(arm, {}).get(target), got.get(base, {}).get(target)
    if not mine or not theirs or min(len(mine), len(theirs)) < 3:
        return None
    count = min(len(mine), len(theirs))
    per_seed = [mine[i] - theirs[i] for i in range(count)]
    result = stats.ttest_rel(mine[:count], theirs[:count])
    return {"mean": float(np.mean(per_seed)), "t": float(result.statistic),
            "seeds": count, "per_seed": per_seed,
            "holds": bool(np.mean(per_seed) > 0 and result.statistic > 2
                          and all(d > 0 for d in per_seed) and count >= 3)}


def selftest() -> None:
    # 1. The matcher must refuse rather than extrapolate. An arm whose Dice
    #    never comes near the target has no point at that target -- snapping
    #    to its nearest endpoint is exactly how an unmatched comparison
    #    disguises itself as a matched one.
    points = [{"epoch": 10, "dice": 0.700, "cldice": 0.7, "betti0_err": 90},
              {"epoch": 20, "dice": 0.750, "cldice": 0.7, "betti0_err": 90}]
    assert epoch_at_dice(points, 0.820) is None
    assert epoch_at_dice(points, 0.7505) == 20
    assert epoch_at_dice(points, 0.7490) == 20
    assert epoch_at_dice(points, 0.7000) == 10
    print(f"  the matcher refuses a target more than {TOLERANCE} away "
          f"instead of extrapolating")

    # 2. THE CASE THIS FILE EXISTS FOR. Two arms on ONE trade-off curve,
    #    differing only in where they sit. Read at their own best epochs one
    #    looks far better; read at a matched Dice they are identical, and the
    #    verdict must say so.
    rows, logs = [], {}
    for arm, offset in (("cheap", 0), ("expensive", 1)):
        for seed in range(3):
            run = f"{arm}_s{seed}"
            # Dice falls and traced rises along the curve, the same curve for
            # both arms; `expensive` simply stops further along it.
            entries = []
            for step in range(5):
                position = step + offset
                dice = 0.820 - 0.010 * position
                traced = 0.30 + 0.05 * position
                entries.append({"epoch": 10 * (step + 1), "dice": dice,
                                "cldice": traced, "betti0_err": 90})
                rows.append({"run": run, "epoch": 10 * (step + 1),
                             "image": "01", "erl": traced * 1000.0,
                             "skel_px": 1000.0, "dice": dice})
            logs[run] = entries
    got = defaultdict(lambda: defaultdict(list))
    for run, entries in logs.items():
        for target in TARGETS:
            epoch = epoch_at_dice(entries, target)
            if epoch is not None:
                got[run.rsplit("_s", 1)[0]][target].append(
                    traced_at(rows, run, epoch))
    for target in (0.810, 0.800, 0.790):
        verdict = compare_at(got, "expensive", "cheap", target)
        assert verdict is not None, target
        assert abs(verdict["mean"]) < 1e-9, (target, verdict["mean"])
        assert not verdict["holds"], target
    print("  two arms on one curve come out EQUAL at every matched Dice, "
          "though their own-best epochs differ by 5 points of traced fraction")

    # 3. And an arm that genuinely moves the frontier must still pass.
    for seed in range(3):
        run = f"better_s{seed}"
        for step in range(5):
            dice = 0.820 - 0.010 * step
            traced = 0.30 + 0.05 * step + 0.04      # lifted at every Dice
            got["better"][0.0]  # touch, keeps defaultdict shape
            rows.append({"run": run, "epoch": 10 * (step + 1), "image": "01",
                         "erl": traced * 1000.0, "skel_px": 1000.0,
                         "dice": dice})
            logs[run] = logs.get(run, []) + [
                {"epoch": 10 * (step + 1), "dice": dice, "cldice": traced,
                 "betti0_err": 90}]
    for run, entries in logs.items():
        if not run.startswith("better"):
            continue
        for target in TARGETS:
            epoch = epoch_at_dice(entries, target)
            if epoch is not None:
                got["better"][target].append(traced_at(rows, run, epoch))
    verdict = compare_at(got, "better", "cheap", 0.810)
    assert verdict is not None and verdict["holds"], verdict
    assert abs(verdict["mean"] - 0.04) < 1e-9, verdict["mean"]
    print(f"  an arm lifted at EVERY Dice still passes "
          f"(+{verdict['mean']:.3f}, t {verdict['t']:.1f})")

    # 4. Fewer than three seeds is not a verdict.
    assert compare_at({"a": {0.81: [0.3, 0.3]}, "b": {0.81: [0.2, 0.2]}},
                      "a", "b", 0.81) is None
    print("  fewer than three seeds returns None, not a passing verdict")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not heldout.SCORES.exists():
        raise SystemExit(f"{heldout.SCORES} not built yet")
    rows = selection.load(heldout.SCORES)
    got = curve(rows)
    arms = [a for a in ("A_dice", "H_aug", "K_focal_aug") if a in got]
    arms += sorted(a for a in got if a not in arms)

    print("Traced fraction read at a MATCHED dev Dice, on all 20 test "
          "images.\n")
    header = "  " + f"{'arm':<24}" + "".join(f"{t:>9.3f}" for t in TARGETS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arm in arms:
        cells = []
        for target in TARGETS:
            values = got[arm].get(target, [])
            cells.append(f"{np.mean(values):8.1%}" if len(values) >= 3
                         else f"{'--':>8} ")
        print(f"  {arm:<24}" + " ".join(cells))
    print("\n  '--' means fewer than three seeds reached that Dice on dev; "
          "the cell is\n  refused rather than extrapolated.\n")

    print("  Against A_dice at each matched Dice:\n")
    for arm in arms:
        if arm == "A_dice":
            continue
        line = f"  {arm:<24}"
        for target in TARGETS:
            verdict = compare_at(got, arm, "A_dice", target)
            line += (f"{'--':>9}" if verdict is None else
                     f"{verdict['mean']:>+8.1%}" +
                     ("*" if verdict["holds"] else " "))
        print(line)
    print("\n  * passes the seed gate at that Dice. An arm that is starred at")
    print("  every Dice moved the frontier. An arm starred nowhere differs")
    print("  from the baseline only in where it sits on the same curve.")


if __name__ == "__main__":
    main()
