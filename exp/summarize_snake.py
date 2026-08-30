"""D-C: does a kernel that FOLLOWS the vessel beat one that crosses it?

WRITTEN AND SELFTESTED 2026-08-31, BEFORE THE FIRST _snake RUN.

D-B FAILED FOR THREE MEASURED REASONS (exp/snake.py has the evidence):
its reach was 2 vessel widths against gaps whose p90 is 7.4 widths; a straight
kernel long enough to bridge those gaps leaves the vessel 25-73% of the time;
and it could only ADD foreground, which the loss punishes and which
calibration.md then showed a lower threshold gives away for free.

THREE ARMS AT EACH LENGTH, so each fault is isolated by a control trained the
same way:

  snake - snkstr   curvature, and nothing else. Same operator, same length,
                   same gate; the control's line is straight.
  snake - snkshf   direction, and nothing else. Same walk on a RANDOM axis
                   field. Without this, any gain could be smoothing.
  snake - base     the headline, and the least informative of the three.

READ AT EACH ARM'S OWN BEST OPERATING POINT, not at 0.5. This is the whole
lesson of calibration.md: at a shared threshold, K_focal_aug beat A_dice by
+13.6% and passed the gate, and at each arm's own dev-Dice-maximising
threshold it LOST by 4.2%. Any propagation layer changes how much foreground
is predicted, so reading it at 0.5 measures its calibration shift. A method
effect has to survive the arms being put on equal footing.

Both readings are printed. If they disagree, the operating-point reading wins,
and the disagreement is itself the finding.

PRE-REGISTERED PREDICTIONS.
  1. `snake - snkstr` is POSITIVE at k16 and near zero at k08. At 8 taps
     (17 px) a straight line strays 1.5 px, about one vessel width; at 16
     (33 px) it strays 6 px and is simply off the vessel. Curvature should
     only pay where straightness fails.
  2. `snake - snkshf` is positive at both lengths. D-B already showed the
     network can tell a real axis field from a random one.
  3. The headline `snake - base` is SMALLER at the own-peak reading than at
     0.5, for every arm, because part of it is calibration.
  4. t2 beats t1 at k16. Following a curve is sequential.

THE GATE is the repo's: paired t over seeds with t > 2, every seed agreeing in
sign, at least three seeds.

  python exp/summarize_snake.py --selftest
  python exp/summarize_snake.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration
import train

BASES = ("A_dice", "H_aug")


def arms(base: str, taps: int, steps: int) -> dict:
    tag = f"k{taps:02d}_t{steps}"
    return {"snake": f"{base}_dir_snake_{tag}",
            "snkstr": f"{base}_dir_snkstr_{tag}",
            "snkshf": f"{base}_dir_snkshf_{tag}",
            "base": base}


def line(fixed: dict, tuned: dict, arm: str, other: str, label: str) -> None:
    if arm not in fixed or other not in fixed:
        print(f"  {label:<38}{'not trained yet':>32}")
        return
    at_half = calibration.gate(fixed[arm], fixed[other])
    at_peak = calibration.gate(tuned[arm], tuned[other])
    if at_half is None or at_peak is None:
        print(f"  {label:<38}{'too few seeds':>32}")
        return
    print(f"  {label:<38}{at_half['mean']:>+9.1%}"
          f"{'*' if at_half['holds'] else ' '}"
          f"{at_peak['mean']:>+12.1%}{'*' if at_peak['holds'] else ' '}"
          f"   {at_peak['seeds']} seeds")


def selftest() -> None:
    for base in BASES:
        for taps in train.SNAKE_TAPS:
            for key, name in arms(base, taps, 1).items():
                assert name in train.CONFIGS, (key, name)
        for key in ("snake", "snkshf"):
            assert arms(base, 16, 2)[key] in train.CONFIGS
    print(f"every arm this script names is in CONFIGS")

    # The lengths must actually differ, and must bracket the gaps that matter:
    # the missed-centreline runs have a p90 of 21 px.
    spans = [2 * taps + 1 for taps in train.SNAKE_TAPS]
    assert spans == sorted(spans) and len(set(spans)) == len(spans), spans
    assert min(spans) < 21 < max(spans), spans
    print(f"  kernel spans {spans} px of arc, bracketing the 21 px p90 gap")

    # The controls must differ from the arm in EXACTLY one thing each.
    for base in BASES:
        for taps in train.SNAKE_TAPS:
            got = arms(base, taps, 1)
            main = train.snake_geometry(got["snake"])
            straight = train.snake_geometry(got["snkstr"])
            shuffled = train.snake_geometry(got["snkshf"])
            assert straight["straight"] and not main["straight"]
            assert straight["taps"] == main["taps"]
            assert straight["shuffle"] == main["shuffle"] is False
            assert shuffled["shuffle"] and not main["shuffle"]
            assert shuffled["taps"] == main["taps"]
            assert shuffled["straight"] == main["straight"] is False
            assert train.AUGMENTS.get(got["snake"], ()) == \
                train.AUGMENTS.get(base, ()), got["snake"]
    print("  each control differs from its arm in exactly one property")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    fixed, tuned = calibration.collect()
    if not fixed:
        raise SystemExit("no frontier data; run exp/frontier.py and "
                         "exp/frontier.py --dev first")
    print("=== D-C: a kernel that follows the vessel ===\n")
    print("Traced fraction, twelve seeds. Two readings of the SAME "
          "checkpoints:")
    print("  at 0.5      the conventional threshold every earlier report used")
    print("  at own peak the threshold maximising each arm's DEV Dice\n")
    header = (f"  {'comparison':<38}{'at 0.5':>10}{'at own peak':>13}")
    for base in BASES:
        print(f"\n{base}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for taps in train.SNAKE_TAPS:
            got = arms(base, taps, 1)
            tag = f"k{taps:02d}"
            line(fixed, tuned, got["snake"], got["snkstr"],
                 f"{tag}: snake - snkstr  (curvature)")
            line(fixed, tuned, got["snake"], got["snkshf"],
                 f"{tag}: snake - snkshf  (direction)")
            line(fixed, tuned, got["snake"], got["base"],
                 f"{tag}: snake - {base}  (headline)")
        two = arms(base, 16, 2)
        line(fixed, tuned, two["snake"], arms(base, 16, 1)["snake"],
             "k16: t2 - t1        (iteration)")
        line(fixed, tuned, two["snake"], two["snkshf"],
             "k16 t2: snake - snkshf")
    print("\n  * passes the gate. Where the two columns disagree, the")
    print("  own-peak column wins: a gain that exists only at a shared")
    print("  threshold is a calibration shift, which calibration.md showed")
    print("  reverses K_focal_aug from +13.6% to -4.2%.")


if __name__ == "__main__":
    main()
