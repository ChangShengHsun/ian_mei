"""Does the propagation layer help, and at what reach?

WRITTEN AND SELFTESTED 2026-08-27, BEFORE A SINGLE SWEPT REACH FINISHED.

WHAT WENT WRONG THE FIRST TIME. D-B was trained at one reach, handed over from
the post-hoc ceiling at its tightest Dice budget. That built a 5x5 kernel with
three pixels per orientation. The result was flat, and the diagnosis was in
the layer's own gate:

    A_dice_dir_prop       real field    gate 0.119 -> 0.146
    A_dice_dir_prop_shuf  random field  gate 0.119 -> 0.028
    H_aug_dir_prop        real field    gate 0.119 -> 0.155
    H_aug_dir_prop_shuf   random field  gate 0.119 -> 0.034

The network could tell a real direction field from noise and opened the layer
five times wider for the real one, on every seed. It just had a three-pixel
operator to work with. The Dice cost had been paid for twice -- once by
choosing the reach at the tightest budget, and again by the training loss,
which is what the gate is for.

So the reach is swept, and it is in the config name.

THREE QUESTIONS, in the order that makes each one readable.

  1. THE GATE, per reach. It is the cheapest signal and it does not depend on
     the segmentation: does the network still open the layer more for a real
     field than for a random one when the operator is big enough to matter?
     If the gap closes as the reach grows, a bigger operator is being driven
     by something other than direction.

  2. _prop vs _shuf, per reach. The load-bearing comparison. Oriented
     propagation adds evidence, and adding evidence moves ERL by itself; only
     the random-field control separates direction from dilation.

  3. _clw_dir_prop vs _clw and vs _dir_prop. Do the two interventions
     compose? D-E, one weight map with no direction in it, was the only thing
     that passed the seed gate on 2026-08-27 (H_aug_clw +249.5 ERL at t 5.23,
     six seeds of six). If the layer adds nothing on top of it, the cheap one
     wins outright.

PRE-REGISTERED PREDICTION. The gate gap holds at every reach. _prop beats
_shuf at a100 and a200 and not at a050. The two interventions are partly
additive: the combination beats _clw, by less than _prop beats _dir alone,
because both are putting foreground on the same missing centreline.

THE GATE is the repo's: paired t over (image, seed), every seed agreeing in
sign, at least three seeds, at rule (iv) on the report half.

  python exp/summarize_reach.py --selftest
  python exp/summarize_reach.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_direction as sd
import summarize_selection as selection
import train

BASES = ("A_dice", "H_aug")


def reach_arms(base: str, reach: str) -> dict:
    return {"prop": f"{base}_dir_prop_{reach}_c025",
            "shuf": f"{base}_dir_prop_shuf_{reach}_c025",
            "combo": f"{base}_clw_dir_prop_{reach}_c025",
            "dir": f"{base}_dir", "clw": f"{base}_clw", "plain": base}


def gates() -> dict:
    """{config: [learned gate per seed]} read off the trained checkpoints.

    Independent of every metric and every selection rule -- it is a parameter
    the network chose. That is what makes it the first thing to look at.
    """
    out = defaultdict(list)
    for path in sorted(selection.SWEEP.glob("*_prop*_s*/final.pt")):
        config = path.parent.name.rsplit("_s", 1)[0]
        state = train.load_checkpoint(path)["model"]
        if "propagation.strength" in state:
            out[config].append(
                float(torch.sigmoid(state["propagation.strength"])))
    return out


def report_gates() -> None:
    found = gates()
    if not found:
        print("no trained propagation arms yet\n")
        return
    print("1. THE GATE the network chose (initialised at 0.119)")
    header = (f"  {'base':<8}{'reach':<7}{'real field':>12}{'random':>10}"
              f"{'ratio':>8}   seeds")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for base in BASES:
        for reach in train.PROPAGATION_REACHES:
            arms = reach_arms(base, reach)
            real = found.get(arms["prop"], [])
            noise = found.get(arms["shuf"], [])
            if not real or not noise:
                continue
            ratio = float(np.mean(real)) / max(float(np.mean(noise)), 1e-9)
            print(f"  {base:<8}{reach:<7}{np.mean(real):12.3f}"
                  f"{np.mean(noise):10.3f}{ratio:7.1f}x   "
                  f"{len(real)}/{len(noise)}")
    print("  A ratio near 1 means the network cannot tell a real axis field")
    print("  from a random one at that reach, and nothing below is about")
    print("  direction.\n")


def line(erl, arm, base, label) -> None:
    got = sd.compare(erl, arm, base)
    if got is None:
        print(f"  {label:<34}{'not trained yet':>16}")
        return
    print(f"  {label:<34}{got['seeds']:>4} seeds{got['mean']:+10.1f}"
          f"{got['t']:7.2f}  {'HOLDS' if got['holds'] else 'fails'}")
    if not got["holds"]:
        print(f"  {'':<34}per seed "
              f"[{' '.join(f'{d:+.0f}' for d in got['per_seed'])}]")


def selftest() -> None:
    # The arm names must be exactly the configs the queue trains, or this
    # reads an empty table and reports "not trained yet" for work that ran.
    for base in BASES:
        for reach in train.PROPAGATION_REACHES:
            for key, name in reach_arms(base, reach).items():
                assert name in train.CONFIGS, (key, name)
    print(f"every arm this script names is in CONFIGS: "
          f"{len(BASES) * len(train.PROPAGATION_REACHES)} reaches x 3 swept "
          f"arms plus 3 baselines")

    # The reaches must be ordered and must actually differ in kernel size,
    # or "swept" is a word rather than an experiment.
    sizes = []
    for reach in train.PROPAGATION_REACHES:
        along, _ = train.propagation_geometry(f"A_dice_dir_prop_{reach}_c025")
        sizes.append(round(along, 2))
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), sizes
    print(f"reaches {train.PROPAGATION_REACHES} are {sizes} px along the "
          f"vessel -- distinct and ordered")

    # And the gate reader must survive a checkpoint with no such parameter.
    assert isinstance(gates(), dict)
    print("the gate reader returns a dict even with nothing trained")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not selection.SCORES.exists():
        raise SystemExit(f"{selection.SCORES} not built yet")
    print("=== D-B: the propagation layer, swept over its reach ===\n")
    report_gates()

    rows = selection.load()
    erl = sd.erl_by_run(rows)
    print("2. Does the layer beat the SAME layer on a random field?")
    print("   ERL at rule (iv), report half, paired on (image, seed)\n")
    for base in BASES:
        for reach in train.PROPAGATION_REACHES:
            arms = reach_arms(base, reach)
            line(erl, arms["prop"], arms["shuf"],
                 f"{base} {reach}: prop - shuf")
        for reach in train.PROPAGATION_REACHES:
            arms = reach_arms(base, reach)
            line(erl, arms["prop"], arms["dir"],
                 f"{base} {reach}: prop - dir (no layer)")
        print()

    print("3. Do the propagation layer and the centreline loss compose?\n")
    for base in BASES:
        for reach in train.PROPAGATION_REACHES:
            arms = reach_arms(base, reach)
            line(erl, arms["combo"], arms["clw"],
                 f"{base} {reach}: combo - clw")
            line(erl, arms["combo"], arms["prop"],
                 f"{base} {reach}: combo - prop")
        print()
    print("Pre-registered: the gate gap holds at every reach; prop beats shuf")
    print("at a100 and a200 and not at a050; the combination beats clw by")
    print("less than prop beats dir, because both put foreground on the same")
    print("missing centreline.")


if __name__ == "__main__":
    main()
