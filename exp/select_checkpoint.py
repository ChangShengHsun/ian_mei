"""Task A3: which validation rule should pick a run's weights?

WRITTEN AND SELFTESTED 2026-08-26, BEFORE ANY TEST-SET ERL WAS COMPUTED.
Nothing in this file may be tuned after seeing a result; that is the whole
point of it existing as a file rather than as a paragraph.

The question. best.pt is chosen by highest whole-image validation Dice, and
E13b R.1 measured that whole-image Dice peaks at wildly different epochs for
different arms -- median 10 for K_focal_aug and G_focal, 65 for H_aug, with
K and H_aug carrying IDENTICAL augmentation, so it is the confidence-gated
clDice loss doing it. E13b R.3/R.7 then measured that early stopping makes the
baseline BETTER on ERL (27.5% -> 31.5% of the tree traced) and K_focal_aug
WORSE (46.1% -> 38.1%). Read together: K keeps improving topologically through
the epochs where its Dice is falling, and a Dice-selected checkpoint throws
exactly that away. log.csv has recorded betti0_err at every validation point
since the first experiment and no selection has ever used it.

THE TWO TRAPS, and what guards each.

  1. Bare betti-0 is gameable. Predict less and the component count falls on
     its own; a model that predicts almost nothing scores a near-perfect
     betti-0 error. Rule (iii) is rule (ii) with a Dice floor for exactly this
     reason, and the floor is derived below rather than chosen.

  2. Selecting and reporting on the same images inflates the result. This repo
     has no third split -- erl.py scores DRIVE's `val`, images 01-20, which is
     the SAME 20 images train.validate() writes into log.csv. So the 20 are
     split in half here, by parity of the image id, fixed before any number
     was looked at. Selection reads the odd images, the reported ERL reads the
     even ones. The all-20 number is also printed, labelled as the optimistic
     bound it is.

THE DICE FLOOR. Not a round number and not chosen: the median, across the six
configs with six seeds, of the per-config standard deviation of best
validation Dice between seeds. Measured 2026-08-26 from the committed logs,
which have nothing to do with ERL:

    A_dice 0.00077  B_cldice 0.00090  H_aug 0.00082
    G_focal 0.00329  K_focal_aug 0.00198  I_coletra 0.00062   median 0.00086

Giving up less Dice than one seed differs from another is giving up nothing
measurable. Rule (iii) is reported at ONE, TWO and FOUR of those, rather than
at a single value, so that no one number carries the argument and the
sensitivity is visible. A rule that only wins at one rung is not a finding.

  python exp/select_checkpoint.py --selftest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pre-registered, 2026-08-26. Derived above; see the docstring.
SEED_SD_DICE = 0.00086
TOLERANCES = tuple(round(multiple * SEED_SD_DICE, 5)
                   for multiple in (1, 2, 4))

# DRIVE val is images 01-20. Parity, fixed before any result was seen; the ids
# carry no ordering this could exploit, and any deterministic half would do --
# what matters is that it was not chosen from the numbers.
def is_selection_image(name: str) -> bool:
    """Odd image ids select; even ones are reported on."""
    return int(name) % 2 == 1


def rule_best_dice(points: list[dict]) -> dict:
    """(i) Highest validation Dice. What best.pt already does."""
    return max(points, key=lambda p: p["dice"])


def rule_best_betti0(points: list[dict]) -> dict:
    """(ii) Lowest validation betti-0 error. Gameable on purpose -- it is the
    arm of the comparison that shows how much the Dice floor is doing."""
    return min(points, key=lambda p: p["betti0_err"])


def rule_betti0_with_floor(points: list[dict], tolerance: float) -> dict:
    """(iii) Lowest validation betti-0 among epochs whose Dice is within
    `tolerance` of this run's own best validation Dice.

    The floor is per-run, not global: arms differ in absolute Dice, and a
    global floor would silently forbid the whole search for the weaker arm
    while leaving the stronger one free.
    """
    ceiling = max(p["dice"] for p in points)
    allowed = [p for p in points if p["dice"] >= ceiling - tolerance]
    return min(allowed, key=lambda p: p["betti0_err"])


def rule_best_cldice(points: list[dict]) -> dict:
    """(iv) Highest validation clDice -- the centreline overlap, which is the
    metric closest to what ERL measures without being ERL."""
    return max(points, key=lambda p: p["cldice"])


def rules() -> list:
    """(name, function) for every pre-registered rule, in report order."""
    out = [("(i) best Dice [current]", rule_best_dice),
           ("(ii) best betti0 [gameable]", rule_best_betti0)]
    for tolerance in TOLERANCES:
        multiple = round(tolerance / SEED_SD_DICE)
        out.append((f"(iii) best betti0, Dice within {multiple}sd "
                    f"({tolerance:.5f})",
                    lambda points, t=tolerance: rule_betti0_with_floor(
                        points, t)))
    out.append(("(iv) best clDice", rule_best_cldice))
    return out


def selftest() -> None:
    # Epoch 10 has the best Dice; epoch 50 has the best betti-0 at a small
    # Dice cost; epoch 90 is the degenerate "predict almost nothing" point --
    # near-perfect betti-0 and a collapsed Dice.
    points = [
        {"epoch": 10, "dice": 0.8200, "betti0_err": 95.0, "cldice": 0.8300},
        {"epoch": 50, "dice": 0.8194, "betti0_err": 68.0, "cldice": 0.8480},
        {"epoch": 90, "dice": 0.4000, "betti0_err": 3.0, "cldice": 0.5000},
    ]
    assert rule_best_dice(points)["epoch"] == 10
    assert rule_best_cldice(points)["epoch"] == 50

    # Trap 1, demonstrated rather than asserted about: bare betti-0 picks the
    # degenerate epoch, and every rung of the floor refuses it.
    assert rule_best_betti0(points)["epoch"] == 90, "rule (ii) should be gamed"
    print("rule (ii) picks the degenerate epoch (dice 0.40, betti0 3.0) -- "
          "it is gameable, as designed")
    for tolerance in TOLERANCES:
        picked = rule_betti0_with_floor(points, tolerance)
        assert picked["epoch"] != 90, (tolerance, picked)
        assert picked["dice"] >= 0.8200 - tolerance - 1e-12, picked
    print(f"rule (iii) refuses it at all {len(TOLERANCES)} tolerances "
          f"{TOLERANCES}")

    # And the floor must actually bind: at 1sd the 0.0006 give-up of epoch 50
    # is allowed, at a tolerance smaller than that it is not.
    assert rule_betti0_with_floor(points, TOLERANCES[0])["epoch"] == 50
    assert rule_betti0_with_floor(points, 0.0001)["epoch"] == 10
    print("  the floor binds: a tolerance below the give-up falls back to (i)")

    # The tolerances must be ordered and none of them round -- a round number
    # is the shape a tuned number takes.
    assert list(TOLERANCES) == sorted(TOLERANCES), TOLERANCES
    assert TOLERANCES[0] < 0.001 < TOLERANCES[-1], TOLERANCES

    # The image split must be a genuine half, and disjoint.
    names = [f"{index:02d}" for index in range(1, 21)]
    select = [n for n in names if is_selection_image(n)]
    report = [n for n in names if not is_selection_image(n)]
    assert len(select) == len(report) == 10, (len(select), len(report))
    assert not set(select) & set(report)
    print(f"selection images {select[0]}..{select[-1]} ({len(select)}), "
          f"reported on {report[0]}..{report[-1]} ({len(report)}), disjoint")

    assert len(rules()) == 6, len(rules())
    print(f"{len(rules())} pre-registered rules: "
          f"{', '.join(n.split(' [')[0] for n, _ in rules())}")
    print("all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
