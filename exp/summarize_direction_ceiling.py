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

CORRECTED 2026-08-27, AFTER THE FIRST RUN. THE THRESHOLDS BELOW ARE
UNCHANGED; what changed is the device that selects a setting to apply them to.

The first version required a correction to cost NO Dice, reasoning that
filling the intact breaks from ground truth RAISED Dice (0.8015 -> 0.8213).
But that oracle PLACES PIXELS EXACTLY, and a dilation adds a whole
neighbourhood; no dilation can ever be Dice-free. Every candidate of every
source failed the floor, the table came back entirely empty, and gate_d1.py
read "no arm produced a usable oracle setting" as a verdict and skipped four
GPU hours of D-B. That is a script failing to RUN being reported as a
mechanism failing to WORK, and the two are not the same answer.

This is C1.0's first-run error in mirror image. There the closing baseline was
free to spend unlimited foreground and beat an oracle that spent almost none;
here nobody was allowed to spend any. Both are the same mistake: comparing
operations at unmatched cost.

THE DEVICE IS NOW A MATCHED BUDGET. Every source is given the same Dice
allowance and asked how far it can trace within it. The allowance is
DERIVED: 0.0187 is what K_focal_aug already surrenders against A_dice
(0.8010 vs 0.8197) for its topology gain -- the trade this series has already
made and published. Three budgets are reported, at that value and at 2.5x and
5x it, because the verdict is sensitive to the choice and a single number
would hide that. The verdict is taken at the TIGHTEST, which is the least
favourable to the mechanism.

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
# Derived, not chosen: K_focal_aug gives up 0.0187 Dice against A_dice for
# its topology gain, which is the trade this repo already publishes. The
# verdict is read at the tightest.
BUDGETS = (0.02, 0.05, 0.10)
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
         metric: str, budget: float):
    """Best setting for this source WITHIN a Dice budget, on the selection half.

    Every source gets the same budget, which is what makes the comparison a
    comparison. None only when the source has no rows at all -- the
    do-nothing setting always costs zero, so any source with data has at
    least one admissible answer, and an empty column now means missing data
    rather than a failed floor.
    """
    table = by_setting(selection_rows, config, source, metric)
    allowed = {key: value for key, value in table.items()
               if value[1] >= raw_dice - budget}
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

    # The budget is what makes it a comparison: at 0.05 the 0.90 setting is
    # affordable, at 0.02 it is not and the source has to fall back.
    assert pick(rows, "A", "oracle", 0.820, "erl_split", 0.02) == (1.0, 0.25)
    assert pick(rows, "A", "oracle", 0.820, "erl_split", 0.20) == (2.0, 1.0)
    print("the same source picks (1.0, 0.25) on a 0.02 budget and (2.0, 1.0) "
          "on a 0.20 one -- the budget is the comparison")

    # A source whose grid contains a free do-nothing setting must return it
    # rather than None on any budget, however tight. In the real sweep every
    # source has one -- the smallest radius rounds to a single pixel and costs
    # nothing -- so after this fix an empty column means missing data and not
    # a failed floor, which is what gate_d1 misread as a verdict.
    for index in range(1, 21):
        rows.append(make("A", "isotropic", 0.25, 0.25, f"{index:02d}",
                         0.50, 0.820))
    free = pick(rows, "A", "isotropic", 0.820, "erl_split", 0.0)
    assert free == (0.25, 0.25), free
    print(f"  at a budget of zero a source still returns its free setting "
          f"{free}, never None")

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
    for config in configs:
        raw = by_setting(report_rows, config, "raw", metric).get((0.0, 0.0))
        if raw is None:
            continue
        raw_erl, raw_dice = raw
        print(f"  {config}  raw {raw_erl:.1%} traced at Dice {raw_dice:.4f}")
        header = (f"    {'budget':<10}" +
                  "".join(f"{s:>13}" for s in SOURCES) +
                  f"{'oracle-iso':>12}   verdict")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for budget in BUDGETS:
            line = f"    -{budget:.2f}     "
            got = {}
            for source in SOURCES:
                chosen = pick(selection_rows, config, source, raw_dice,
                              metric, budget)
                if chosen is None:
                    line += f"{'--':>13}"
                    got[source] = None
                    continue
                value = by_setting(report_rows, config, source,
                                   metric)[chosen][0]
                got[source] = (value, chosen)
                line += f"{value:12.1%} "
            if got.get("oracle") and got.get("isotropic"):
                gain = got["oracle"][0] - got["isotropic"][0]
                line += f"{gain:+11.1%}   {verdict(gain)}"
            print(line)
        # The geometry the oracle and the predicted field chose at the
        # tightest budget: this is the handover to D-B, and it is also the
        # answer to what the sweep was built to ask -- along, or across.
        for source in ("oracle", "predicted"):
            chosen = pick(selection_rows, config, source, raw_dice, metric,
                          BUDGETS[0])
            if chosen is not None:
                print(f"    {source} chose along={chosen[0]} "
                      f"across={chosen[1]} widths at the tightest budget")
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
    print("The verdict is read at the TIGHTEST budget, which is the least")
    print("favourable to the mechanism. If it changes across the three, say")
    print("so rather than quoting the one that flatters.")
    print()
    print("The prize is 21.8 points: 16.7 for covering the centreline the")
    print("prediction runs beside, 5.1 for the severing breaks. The other")
    print("19.9 points between the two tables above are erl.py's splitting")
    print("rule and are not available to any method.")


if __name__ == "__main__":
    main()
