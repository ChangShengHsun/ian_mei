"""What C1.0 says: is severs-guided linking worth building?

No gate here and no claim: an oracle is an upper bound, not a method. The
table reports, per arm, the fraction of the ground-truth tree an error-free
trace covers under each condition, and the two numbers the decision turns on:

  headroom   oracle_sever - raw. Everything a perfect linker could add.
  unclaimed  oracle_sever - closing_fair. What is left AFTER blind closing
             has taken what it can AT THE ORACLE'S OWN DICE COST. This is
             C1's actual budget, and it decides whether the line is worth
             weeks.

WHY closing_fair AND NOT closing. Corrected 2026-08-27, after the first run.
ERL is gameable by predicting more: it counts ground-truth skeleton covered
by predicted components, so a fatter mask scores higher, and filling in every
missed centreline pixel (`oracle_all`) reaches 97.8%. Left free, the closing
search took the largest radius on offer -- 6 px, 2.1x the median vessel width
-- and added 31% MORE FOREGROUND, against the severs oracle's 0.7%. On that
comparison `unclaimed` came out NEGATIVE on all four arms and the verdict
read NOT WORTH BUILDING, which was an artefact of comparing a fat mask with
a thin one.

The pre-registered thresholds below are UNCHANGED. What was wrong was the
quantity fed to them. Both closings are printed, with their Dice, so the size
of the confound stays visible instead of being quietly corrected away.

The threshold is stated before the numbers are read, because "is 4 points a
lot" is exactly the question that gets answered differently once the answer
is on screen. E18's headline -- the result this series is currently built on
-- moved the traced fraction by 18.6 points (27.5% -> 46.1%). A method that
costs one to two weeks should be worth a fair share of that. So:

  unclaimed < 3 points          C1 is not worth building. Say so and move
                                the budget to D1 or B1.
  3 to 8 points                 worth building only if it composes with D1;
                                not worth it standing alone.
  > 8 points                    build it.

  python exp/summarize_ceiling.py --selftest
  python exp/summarize_ceiling.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as rules_module
import summarize_selection as selection

SCORES = selection.SWEEP / "link_ceiling.csv"
CONDITIONS = ("raw", "closing", "closing_fair", "oracle_sever", "oracle_all")
# PRE-REGISTERED 2026-08-27, before link_ceiling.py was run. Points of the
# traced fraction, measured against E18's 18.6-point headline.
DEAD = 0.03
CLEAR = 0.08


def load(path: Path = SCORES) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ("erl", "skel_px", "foreground", "dice"):
            row[key] = float(row[key])
    return rows


def shares(rows) -> dict:
    """{(config, condition): mean traced fraction} on the report half."""
    grouped = defaultdict(list)
    for row in rows:
        if rules_module.is_selection_image(row["image"]):
            continue
        grouped[(row["config"], row["condition"])].append(
            row["erl"] / row["skel_px"])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def column(rows, key: str) -> dict:
    """{(config, condition): mean of `key`} on the report half."""
    grouped = defaultdict(list)
    for row in rows:
        if rules_module.is_selection_image(row["image"]):
            continue
        grouped[(row["config"], row["condition"])].append(float(row[key]))
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def verdict(unclaimed: float) -> str:
    if unclaimed < DEAD:
        return "NOT WORTH BUILDING"
    if unclaimed < CLEAR:
        return "only worth it composed with D1"
    return "BUILD IT"


def selftest() -> None:
    rows = []
    for seed in range(2):
        for index in range(1, 21):
            for condition, value in (("raw", 4000.0), ("closing", 4200.0),
                                     ("oracle_sever", 6000.0),
                                     ("oracle_all", 8000.0)):
                rows.append({"config": "A_dice", "run": f"A_dice_s{seed}",
                             "seed": str(seed), "condition": condition,
                             "image": f"{index:02d}", "erl": value,
                             "skel_px": 10000.0, "foreground": 1.0})
    got = shares(rows)
    assert abs(got[("A_dice", "raw")] - 0.40) < 1e-9, got
    assert abs(got[("A_dice", "oracle_sever")] - 0.60) < 1e-9, got
    print("shares(): raw 40.0%, closing 42.0%, oracle_sever 60.0%")

    # The report half only. Plant a huge oracle on the selection half.
    for row in rows:
        if rules_module.is_selection_image(row["image"]) \
                and row["condition"] == "oracle_sever":
            row["erl"] = 99999.0
    again = shares(rows)
    assert abs(again[("A_dice", "oracle_sever")] - 0.60) < 1e-9, again
    print("  and it reads the report half only")

    assert verdict(0.02) == "NOT WORTH BUILDING", verdict(0.02)
    assert verdict(0.05).startswith("only worth"), verdict(0.05)
    assert verdict(0.18) == "BUILD IT", verdict(0.18)
    print(f"thresholds: <{DEAD:.0%} dead, {DEAD:.0%}-{CLEAR:.0%} only with "
          f"D1, >{CLEAR:.0%} build it -- fixed before the numbers")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not SCORES.exists():
        raise SystemExit(f"{SCORES} not built -- run exp/link_ceiling.py")
    rows = load()
    got = shares(rows)
    configs = [c for c in selection.ARMS
               if (c, "raw") in got] or sorted({r["config"] for r in rows})
    radius = sorted({int(float(r["radius_px"])) for r in rows})
    print("=== C1.0: the ceiling on severs-guided linking ===")
    print(f"traced fraction of the ground-truth tree, report half, weights "
          f"chosen by {'; '.join(sorted({r['epoch'] for r in rows})[:1])}"
          f" (rule iv); closing radii used {radius} px\n")
    header = (f"  {'condition':<14}{'traced':>9}{'Dice':>9}{'fg':>9}"
              f"{'vs raw':>9}")
    dices = column(rows, "dice")
    fg = column(rows, "foreground")
    for config in configs:
        raw = got[(config, "raw")]
        print(f"{config}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for condition in CONDITIONS:
            if (config, condition) not in got:
                continue
            share = got[(config, condition)]
            print(f"  {condition:<14}{share:8.1%}{dices[(config, condition)]:9.4f}"
                  f"{fg[(config, condition)] / fg[(config, 'raw')]:8.2f}x"
                  f"{share - raw:+9.1%}")
        oracle = got[(config, "oracle_sever")]
        fair = got.get((config, "closing_fair"))
        free = got.get((config, "closing"))
        if fair is None:
            print()
            continue
        unclaimed = oracle - fair
        print(f"  -> headroom {oracle - raw:+.1%}, unclaimed over the fair "
              f"closing {unclaimed:+.1%}: {verdict(unclaimed)}")
        if free is not None and free > oracle:
            print(f"     (the UNCONSTRAINED closing reaches {free:.1%} at "
                  f"{fg[(config, 'closing')] / fg[(config, 'raw')]:.2f}x the "
                  f"foreground -- not a comparable condition)")
        print()
    print()
    print("headroom  = oracle_sever - raw: everything a perfect linker adds.")
    print("unclaimed = oracle_sever - closing_fair: what is left after blind")
    print("            closing has taken what it can AT THE ORACLE'S OWN DICE")
    print("            COST. C1's real budget.")
    print("fg        = foreground relative to raw. This column is why the")
    print("            unconstrained closing is not a valid baseline: it buys")
    print("            run length by painting more, which ERL rewards and")
    print("            Dice charges for.")
    print("oracle_all is the bound for ANY post-processing: the distance from")
    print("oracle_sever to it is tree that no linker can reach, because")
    print("nothing was predicted near it. That part is D1's and B1's problem,")
    print("not C1's.")
    print()
    print("Both oracle columns use ground truth by construction. They are")
    print("upper bounds, not methods, and nothing here is a claim about a")
    print("model. The one decision they support is where the next weeks go.")


if __name__ == "__main__":
    main()
