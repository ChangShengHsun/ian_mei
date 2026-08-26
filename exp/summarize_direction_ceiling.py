"""Does knowing the vessel's axis reach D1's budget? The pre-registered call.

WRITTEN AND SELFTESTED 2026-08-27, BEFORE direction_ceiling.py PRODUCED A
NUMBER. Thresholds below are fixed here and are not revisited once the table
is on screen -- the same discipline that let C1 be retired in two hours
instead of two weeks.

THE PRIZE, measured this morning on K_focal_aug at rule (iv):

    raw                                              47.4% traced
    + not splitting runs the prediction bridges     +19.9  (a convention,
                                                            not a gain)
    + every intact break filled from ground truth   +16.7  } the 21.8 points
    + every severing break filled                   + 5.1  } a method can win

THE DICE FLOOR, and why it is zero. Filling the intact breaks from ground
truth RAISED Dice, 0.8015 -> 0.8213: putting foreground on the centreline is
not a trade, it is free. So a correction is only allowed to count if it costs
no Dice at all. Isotropic dilation cannot meet that bar -- one pixel in every
direction costs 0.14 Dice -- and that is the point of the bar, not an
accident of it. Settings are chosen on the SELECTION half under this floor
and reported on the other half.

WHAT IS COMPARED. `oracle` (ground-truth axis) minus `isotropic` (no axis),
both at their own best setting under the floor. That difference is what
knowing the direction is worth, with the foreground it costs already paid for
on both sides.

  under 3 points     the mechanism is wrong. Direction does not reach this
                     budget, and the D1 line stops here.
  3 to 8 points      worth letting a network learn to use the field (D-A, a
                     refinement stage), not worth a hand-built layer.
  over 8 points      build it: D-B's propagation layer, or D-C's steered
                     convolutions.

TWO CONTROLS THAT CAN INVALIDATE THE WHOLE TABLE, checked before the verdict:

  shuffled   a random per-pixel axis field. Oriented dilation adds
             foreground, and foreground raises ERL by itself -- that is how
             the closing baseline beat the C1 oracle until its Dice cost was
             matched. If oracle does not clearly beat shuffled, this measures
             dilation and the verdict above is void.
  predicted  a trained _dir head. oracle - predicted is the part of the prize
             lost to the PREDICTOR rather than to the mechanism, and it says
             whether the next effort goes into the head or the layer.

  python exp/summarize_direction_ceiling.py --selftest
  python exp/summarize_direction_ceiling.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as rules_module
import summarize_selection as selection

SCORES = selection.SWEEP / "direction_ceiling.csv"
# Pre-registered, 2026-08-27. Points of traced tree, against a 21.8-point prize.
DEAD, CLEAR = 0.03, 0.08
# The correction must cost no Dice. Derived, not chosen: the ground-truth
# intact fill RAISED Dice by 0.0198.
DICE_FLOOR = 0.0
SOURCES = ("isotropic", "shuffled", "predicted", "oracle")


def load(path: Path = SCORES) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ("along", "across", "erl_split", "erl_bridged", "dice"):
            row[key] = float(row[key])
        row["fg"] = int(row["fg"])
    return rows


def half(rows, selection_half: bool) -> list[dict]:
    return [r for r in rows
            if rules_module.is_selection_image(r["image"]) == selection_half]


def by_setting(rows, config: str, source: str, metric: str) -> dict:
    """{(along, across): (mean metric, mean dice)} for one arm and source."""
    grouped = defaultdict(list)
    for row in rows:
        if row["config"] == config and row["source"] == source:
            grouped[(row["along"], row["across"])].append(row)
    return {key: (float(np.mean([r[metric] for r in these])),
                  float(np.mean([r["dice"] for r in these])))
            for key, these in grouped.items()}


def pick(selection_rows, config: str, source: str, raw_dice: float,
         metric: str):
    """The setting with the best metric whose Dice clears the floor.

    None when nothing clears it: falling back to the unconstrained best would
    restore exactly the comparison the floor exists to prevent, and a source
    that cannot pay its own way should say so rather than be given a pass.
    """
    table = by_setting(selection_rows, config, source, metric)
    allowed = {key: value for key, value in table.items()
               if value[1] >= raw_dice - DICE_FLOOR}
    if not allowed:
        return None
    return max(allowed, key=lambda key: allowed[key][0])


def selftest() -> None:
    def make(config, source, along, across, image, erl, dice):
        return {"config": config, "run": f"{config}_s0", "seed": "0",
                "source": source, "along": along, "across": across,
                "image": image, "erl_split": erl, "erl_bridged": erl,
                "dice": dice, "fg": 1000}

    rows = []
    for index in range(1, 21):
        image = f"{index:02d}"
        rows.append(make("A", "raw", 0.0, 0.0, image, 0.50, 0.820))
        # A setting that traces further but pays Dice: the floor must refuse
        # it, however good it looks.
        rows.append(make("A", "oracle", 2.0, 1.0, image, 0.90, 0.700))
        rows.append(make("A", "oracle", 1.0, 0.25, image, 0.60, 0.825))
        rows.append(make("A", "isotropic", 0.5, 0.5, image, 0.70, 0.690))

    chosen = pick(rows, "A", "oracle", 0.820, "erl_split")
    assert chosen == (1.0, 0.25), chosen
    print(f"the Dice floor refuses the 0.90 setting that costs 0.12 Dice and "
          f"takes {chosen} at 0.60 -- a gain bought with overlap is a trade")

    # A source that cannot clear the floor at any setting returns None rather
    # than its own best.
    assert pick(rows, "A", "isotropic", 0.820, "erl_split") is None
    print("  isotropic clears the floor at no setting, so it scores nothing "
          "-- which is the finding, not a gap in the table")

    assert verdict(0.02).startswith("MECHANISM"), verdict(0.02)
    assert verdict(0.05).startswith("learn"), verdict(0.05)
    assert verdict(0.12) == "BUILD IT", verdict(0.12)
    print(f"thresholds: <{DEAD:.0%} stop, {DEAD:.0%}-{CLEAR:.0%} let the net "
          f"learn it, >{CLEAR:.0%} build the layer -- fixed before the run")
    print("all checks passed")


def verdict(gain: float) -> str:
    if gain < DEAD:
        return "MECHANISM WRONG - stop the D1 line"
    if gain < CLEAR:
        return "learn it (D-A refinement stage)"
    return "BUILD IT"


def report(rows, metric: str, label: str) -> None:
    print(f"--- {label} ---")
    selection_rows, report_rows = half(rows, True), half(rows, False)
    configs = sorted({r["config"] for r in rows})
    header = (f"  {'arm':<14}{'raw':>8}" +
              "".join(f"{s:>12}" for s in SOURCES) + f"{'oracle-iso':>12}"
              "   verdict")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config in configs:
        raw = by_setting(report_rows, config, "raw", metric).get((0.0, 0.0))
        if raw is None:
            continue
        raw_erl, raw_dice = raw
        line = f"  {config:<14}{raw_erl:7.1%}"
        got = {}
        for source in SOURCES:
            chosen = pick(selection_rows, config, source, raw_dice, metric)
            if chosen is None:
                line += f"{'--':>12}"
                got[source] = None
                continue
            table = by_setting(report_rows, config, source, metric)
            value = table[chosen][0]
            got[source] = value
            line += f"{value:11.1%}*"
        if got.get("oracle") is not None:
            gain = got["oracle"] - (got.get("isotropic") or raw_erl)
            line += f"{gain:+11.1%}   {verdict(gain)}"
        print(line)
        for source in ("oracle", "predicted", "shuffled"):
            chosen = pick(selection_rows, config, source, raw_dice, metric)
            if chosen is not None:
                print(f"  {'':<14}{source} chose along={chosen[0]} "
                      f"across={chosen[1]} widths")
    print("  * best setting under the Dice floor, chosen on the selection "
          "half, reported on the other. '--' = no setting cleared it.")
    print()


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not SCORES.exists():
        raise SystemExit(f"{SCORES} not built -- run exp/direction_ceiling.py")
    rows = load()
    print("=== D1 ceiling: what is knowing the vessel's axis worth? ===\n")
    report(rows, "erl_split", "erl.py as written (a bridged gap splits a run)")
    report(rows, "erl_bridged", "not splitting runs the prediction bridges")

    print("READ THE CONTROLS FIRST.")
    print("  oracle vs shuffled: if these are close, oriented dilation is")
    print("    just dilation and every verdict above is void.")
    print("  oracle vs predicted: the part of the prize lost to the direction")
    print("    HEAD rather than to the mechanism. A large gap says the next")
    print("    effort is a better head; a small one says it is the layer.")
    print()
    print("The prize is 21.8 points: 16.7 for covering the centreline the")
    print("prediction runs beside, 5.1 for the severing breaks. The other")
    print("19.9 points between the two tables above are erl.py's splitting")
    print("rule and are not available to any method.")


if __name__ == "__main__":
    main()
