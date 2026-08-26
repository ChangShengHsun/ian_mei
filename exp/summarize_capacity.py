"""E13's verdict. Written and selftested before the first w32 training step.

Every conclusion in this series was measured on a 117k-parameter U-Net against
a field standard nearer 30M. So "the loss barely matters" (E5, stage 0) has
never been distinguishable from "the model is too small for the loss to
matter", and E14's "what you show the network beats what you ask of it" has
never been distinguishable from "augmentation compensates for a model that
would otherwise overfit twenty images".

Three arms at 4x the base width, 467k parameters:

  A_dice_w32     plain BCE+Dice, the reference
  B_cldice_w32   the topology loss, E13's original question
  H_aug_w32      augmentation, the question E14 made more interesting

The comparison that matters is not w32 against w32. It is whether the GAP
between arms changes with width, so every criterion below is a difference of
differences, and each is checked at both widths.

  python exp/summarize_capacity.py --selftest
  python exp/summarize_capacity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_confidence as confidence
import summarize_gated as gated

DIM, CLEAR = "Q1_dimmest", "Q4_clearest"
MIN_SIZE = gated.PRIMARY_MIN_SIZE
NARROW = ("A_dice", "B_cldice", "H_aug")
WIDE = tuple(f"{name}_w32" for name in NARROW)
# E13's third point, pre-registered 2026-08-26 before its first training step.
# 117k and 467k are both far below the field standard, so "the loss advantage
# is a small-model artifact" is currently a claim about two small models. The
# third point is 31M in the shape a reviewer expects -- 5 levels at base 64,
# not this net widened to base 256, which would spend every parameter at full
# patch resolution. Suffix order matches train.CONFIGS.
POINTS = (("base=16", ""), ("base=32", "_w32"), ("31M/d5", "_w64_d5"))
DEEP = tuple(f"{name}_w64_d5" for name in NARROW)


def paired_seeds(*arms: str) -> tuple[str, ...]:
    import train
    runs = set(train.trained_runs())
    seeds = {name.rsplit("_s", 1)[1] for name in runs}
    return tuple(sorted(seed for seed in seeds
                        if all(f"{arm}_s{seed}" in runs for arm in arms)))


def gap(rows, better: str, worse: str, band: str) -> dict:
    """One arm's advantage over another, with the per-seed means for the gate."""
    result = gated.paired(rows, better, worse, band, MIN_SIZE, "dice")
    result["per_seed"] = gated.per_seed(rows, better, worse, band, MIN_SIZE,
                                        "dice")
    return result


def compare_widths(rows, better: str, worse: str, band: str,
                   label: str) -> None:
    """Does the gap survive 4x the width?

    Reported, not gated. A difference of differences over three seeds a side is
    not something this design can put a threshold on honestly, and inventing
    one after the fact is exactly what the rest of this file exists to prevent.
    What IS gated is each width's own gap, so the readable claim is "the gap
    holds at both widths" or "it holds at one and not the other".
    """
    print(f"  {label}")
    verdicts = {}
    for width, suffix in POINTS:
        seeds = paired_seeds(better + suffix, worse + suffix)
        if not seeds:
            print(f"    {width:9} not trained yet")
            verdicts[width] = None
            continue
        gated.SEEDS = seeds
        result = gap(rows, better + suffix, worse + suffix, band)
        per = result["per_seed"]
        holds = (result["mean"] > 0 and result["t"] > 2
                 and all(d > 0 for d in per))
        verdicts[width] = (result["mean"], holds)
        # A sign gate over one seed is not a gate: one seed cannot disagree
        # with itself, so it would print HOLDS for anything positive. Say so
        # rather than letting a partial queue read as evidence.
        if len(per) < 3:
            label = f"only {len(per)} seed, NOT gated"
        else:
            label = "HOLDS" if holds else "fails gate"
        print(f"    {width:9} {result['mean']:+.4f}  t={result['t']:6.2f}  "
              f"{len(per)} seeds  {label}  "
              f"[{' '.join(f'{d:+.4f}' for d in per)}]")
    # Report every consecutive step of the curve rather than only its ends:
    # "shrinks then shrinks again" and "shrinks then rebounds" are different
    # findings and a single first-to-last delta cannot tell them apart.
    for (left_name, _), (right_name, _) in zip(POINTS, POINTS[1:]):
        left, right = verdicts[left_name], verdicts[right_name]
        if left and right:
            change = right[0] - left[0]
            print(f"    -> gap changes by {change:+.4f} from {left_name} to "
                  f"{right_name} ({'grows' if change > 0 else 'shrinks'})")
    print()


def selftest() -> None:
    import train
    for name in NARROW + WIDE + DEEP:
        assert name in train.CONFIGS, name
    for narrow, wide in zip(NARROW, WIDE):
        assert train.CONFIGS[narrow] == train.CONFIGS[wide], narrow
        assert train.AUGMENTS.get(narrow, ()) == train.AUGMENTS.get(wide, ()), \
            f"{wide} is not {narrow} at another width"
        assert train.base_width(wide) == 4 * train.base_width(narrow) // 2, wide
    # The third point must be the same three arms again and nothing else: an
    # arm that quietly loses its augmentation tuple at a new suffix is E13's
    # H_aug_w32 bug, which is why AUGMENTS is checked and not just CONFIGS.
    for narrow, deep in zip(NARROW, DEEP):
        assert train.CONFIGS[narrow] == train.CONFIGS[deep], narrow
        assert train.AUGMENTS.get(narrow, ()) == train.AUGMENTS.get(deep, ()), \
            f"{deep} is not {narrow} at another capacity"
        assert train.base_width(deep) == 64 and train.net_depth(deep) == 5, deep
    # Every suffix in POINTS has to name real configs, or a point silently
    # reads "not trained yet" forever.
    for _, suffix in POINTS:
        for arm in NARROW:
            assert arm + suffix in train.CONFIGS, arm + suffix
    counts = {n: sum(p.numel() for p in train.build_model(n).parameters())
              for n in (NARROW[0], WIDE[0], DEEP[0])}
    ratio = counts[WIDE[0]] / counts[NARROW[0]]
    assert 3.5 < ratio < 4.0, counts
    assert 25e6 < counts[DEEP[0]] < 40e6, counts
    print(f"three arms, each present at all {len(POINTS)} capacities; "
          f"{counts[NARROW[0]]:,} -> {counts[WIDE[0]]:,} -> "
          f"{counts[DEEP[0]]:,} params")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    rows = gated.load()
    present = {r["config"] for r in rows}
    missing = [n for n in WIDE if n not in present]
    if missing:
        print(f"not trained yet: {missing}\n")

    print("=== E13: does the gap between arms survive 4x the width? ===\n")
    print(f"-- {DIM} --")
    compare_widths(rows, "H_aug", "A_dice", DIM,
                   "1. augmentation over the plain baseline")
    compare_widths(rows, "B_cldice", "A_dice", DIM,
                   "2. the topology loss over the plain baseline")
    compare_widths(rows, "H_aug", "B_cldice", DIM,
                   "3. augmentation over the topology loss")
    print(f"-- {CLEAR} --")
    compare_widths(rows, "H_aug", "A_dice", CLEAR,
                   "4. augmentation in the band with no headroom left")

    print("Reading it: if gap 1 holds at both widths, the input side matters")
    print("independently of capacity. If it shrinks toward zero, augmentation")
    print("was partly standing in for a model too small, and E14's ordering is")
    print("a small-model result. Gap 2 is E13's original question, which E14")
    print("demoted: whether the loss starts to matter once there is capacity")
    print("for it to matter.")


if __name__ == "__main__":
    main()
