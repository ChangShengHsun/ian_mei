"""Does task A's answer justify task B's fifteen GPU hours?

WRITTEN 2026-08-26, BEFORE A3 PRODUCED A SINGLE NUMBER. The condition is
Ivan's, from the work order: "if no rule beats the status quo, task B's 15
hours should not be spent." Written as a script rather than as a judgement
call so that the decision cannot drift once the table is on screen.

THE CONDITION. Task B proceeds if and only if at least one rule other than
(i) beats (i) on reported ERL, under the full repo gate -- paired t over
(image, seed) AND every seed agreeing in sign -- on AT LEAST TWO of the four
arms.

Two arms, not one. A rule that helps exactly one arm is an arm-specific
accident, and carrying it to 31M would be fitting the selection rule to the
arm it happened to suit; A4 says the cheap capacity is where the pattern gets
found, which means the pattern has to be a pattern. Two of four is the
smallest majority-of-evidence this design can express with four arms.

The rule carried to 31M, if the gate opens, is the one holding on the most
arms; ties break toward the more conservative rule, meaning the one that gives
up less validation Dice.

  python exp/gate_task_b.py --selftest
  python exp/gate_task_b.py            # exit 0 = run task B, 1 = do not
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as rules_module
import summarize_selection as selection

MIN_ARMS = 2


def evaluate(rows) -> dict:
    """{rule name: {arm: (mean_diff, holds, dice_given_up)}} against rule (i)."""
    points = selection.selection_points(rows)
    erl = selection.report_erl(rows)
    by_arm = defaultdict(list)
    for run in sorted(points):
        by_arm[run.rsplit("_s", 1)[0]].append(run)

    named = rules_module.rules()
    reference_name, reference_rule = named[0]
    out = {}
    for name, rule in named[1:]:
        out[name] = {}
        for arm, runs in by_arm.items():
            paired, per_seed, given_up = [], [], []
            for run in runs:
                mine = erl[(run, rule(points[run])["epoch"])]
                base = erl[(run, reference_rule(points[run])["epoch"])]
                inside = [(mine[i], base[i]) for i in sorted(mine)]
                paired.extend(inside)
                per_seed.append(float(np.mean([a - b for a, b in inside])))
                pick = rule(points[run])
                given_up.append(max(p["dice"] for p in points[run])
                                - pick["dice"])
            mean, _, holds = selection.gate(paired, per_seed)
            out[name][arm] = (mean, holds, float(np.mean(given_up)))
    return out


def decide(table) -> tuple[bool, str]:
    winners = {name: sum(1 for v in arms.values() if v[1])
               for name, arms in table.items()}
    best = max(winners.values()) if winners else 0
    if best < MIN_ARMS:
        return False, ""
    tied = [name for name, count in winners.items() if count == best]
    # Conservative tie-break: least validation Dice surrendered.
    chosen = min(tied, key=lambda name: np.mean(
        [v[2] for v in table[name].values()]))
    return True, chosen


def selftest() -> None:
    # One rule holding on one arm must NOT open the gate; the same rule
    # holding on two must.
    one = {"(ii)": {"A": (10.0, True, 0.001), "B": (5.0, False, 0.001),
                    "C": (1.0, False, 0.001), "D": (0.0, False, 0.001)}}
    assert decide(one) == (False, ""), decide(one)
    print(f"a rule holding on one arm does not open the gate "
          f"(needs {MIN_ARMS})")

    two = {"(ii)": {"A": (10.0, True, 0.004), "B": (5.0, True, 0.004),
                    "C": (1.0, False, 0.004), "D": (0.0, False, 0.004)},
           "(iii)": {"A": (8.0, True, 0.001), "B": (4.0, True, 0.001),
                     "C": (1.0, False, 0.001), "D": (0.0, False, 0.001)}}
    opened, chosen = decide(two)
    assert opened, two
    # Both hold on two arms; the tie must break to the one giving up less
    # Dice, not to the one with the bigger ERL gain -- the bigger gain here
    # belongs to the gameable rule, which is exactly the trap.
    assert chosen == "(iii)", chosen
    print(f"a tie breaks toward the rule that surrenders less Dice: {chosen}")

    none = {"(ii)": {"A": (10.0, False, 0.001), "B": (5.0, False, 0.001)}}
    assert decide(none) == (False, ""), decide(none)
    print("no rule holding anywhere keeps the gate shut")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not selection.SCORES.exists():
        print(f"CLOSED: {selection.SCORES} does not exist; task A did not "
              f"finish, so task B has nothing to stand on.")
        sys.exit(1)
    table = evaluate(selection.load())
    print("rule vs (i) best Dice, per arm (ERL on the report half):")
    for name, arms in table.items():
        marks = "  ".join(
            f"{arm}:{'HOLDS' if holds else 'fails'}{mean:+.0f}"
            for arm, (mean, holds, _) in sorted(arms.items()))
        print(f"  {name:<44}{marks}")
    opened, chosen = decide(table)
    print()
    if opened:
        print(f"OPEN: '{chosen}' holds on >= {MIN_ARMS} arms. Task B runs, "
              f"carrying only this rule to 31M.")
        Path(selection.SWEEP / "task_b_rule.txt").write_text(chosen + "\n")
        sys.exit(0)
    print(f"CLOSED: no rule holds on {MIN_ARMS} or more arms. Task B's 15 GPU "
          f"hours are not justified; re-plan the priority order instead.")
    sys.exit(1)


if __name__ == "__main__":
    main()
