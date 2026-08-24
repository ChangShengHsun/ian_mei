"""E18's verdict. Written and selftested before the first K_focal_aug step.

Two arms in this series work, and they have never been run together:

  G_focal  confidence-gated clDice. Best dim-band Dice anywhere (+0.0184 over
           clDice, six seeds, E12) and it PAYS for it: 1.85 MORE severing
           breaks per image, five of six seeds positive.
  H_aug    geometry + photometric augmentation, loss untouched. REMOVES 3.2
           severing breaks per image, six of six seeds (E14), and is the only
           thing in this series to pass that criterion.

E15 puts them level on expected run length (-20.6, gate fails, so they are
indistinguishable there) by opposite routes: one covers more centreline while
severing more, the other severs less. Two mechanisms reaching the same score
from opposite directions is the case for combining them.

What this is NOT: a novelty claim. CoLeTra (arXiv:2503.05541) already crossed
augmentation with six loss functions on DRIVE among others, so "augmentation
plus a topology loss" is well-trodden. Nor is weighted clDice new -- Smooth
clDice (Morand 2025) weights it by a spatial uncertainty zone at vessel
boundaries. What the search did not turn up is a clDice weighted by the MODEL's
own hesitancy, and nobody has combined that with augmentation. The value here
is internal: it decides which model we recommend.

K_focal_aug differs from H_aug in the loss alone and from G_focal in the
augmentation alone, so either comparison isolates one change.

  python exp/summarize_combo.py --selftest
  python exp/summarize_combo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_confidence as confidence
import summarize_gated as gated

DIM, CLEAR = "Q1_dimmest", "Q4_clearest"
MIN_SIZE = gated.PRIMARY_MIN_SIZE
COMBO, AUG, GATE, BASE = "K_focal_aug", "H_aug", "G_focal", "A_dice"


def paired_seeds(*arms: str) -> tuple[str, ...]:
    import train
    runs = set(train.trained_runs())
    seeds = {name.rsplit("_s", 1)[1] for name in runs}
    return tuple(sorted(seed for seed in seeds
                        if all(f"{arm}_s{seed}" in runs for arm in arms)))


def dice_check(rows, better, worse, band, want, label) -> bool:
    result = gated.paired(rows, better, worse, band, MIN_SIZE, "dice")
    per = gated.per_seed(rows, better, worse, band, MIN_SIZE, "dice")
    return confidence.check(label, result["mean"], result["t"], per, want)


def severing_check(better, worse, want, label) -> bool:
    left = confidence.severing_per_image(better, DIM)
    right = confidence.severing_per_image(worse, DIM)
    mean, t, per = confidence.paired_counts(left, right)
    return confidence.check(label, mean, t, per, want)


def selftest() -> None:
    import train
    for name in (COMBO, AUG, GATE, BASE):
        assert name in train.CONFIGS, name
    # K is H_aug's augmentation with G_focal's loss, and nothing else.
    assert train.AUGMENTS[COMBO] == train.AUGMENTS[AUG], "augments must match H"
    assert train.CONFIGS[COMBO][1] == train.CONFIGS[GATE][1], "loss must match G"
    assert train.CONFIGS[COMBO][0] == train.CONFIGS[AUG][0], "same blurpool"
    assert train.CONFIGS[COMBO][1] != train.CONFIGS[AUG][1], \
        "K and H_aug would be the same arm"
    assert train.AUGMENTS.get(GATE, ()) != train.AUGMENTS[COMBO], \
        "K and G_focal would be the same arm"
    assert train.base_width(COMBO) == train.base_width(AUG)
    print(f"{COMBO}: loss {train.CONFIGS[COMBO][1]!r} from {GATE}, "
          f"augments {train.AUGMENTS[COMBO]} from {AUG}")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    seeds = paired_seeds(COMBO, AUG, GATE)
    if not seeds:
        raise SystemExit(f"no seed has all of {COMBO}, {AUG}, {GATE} yet")
    gated.SEEDS = seeds
    confidence.SEEDS = seeds
    rows = gated.load()

    print("=== E18 pre-registered verdict: gate + augmentation together ===")
    print(f"    (paired seeds: {', '.join(seeds)})\n")

    checks = [
        dice_check(rows, COMBO, AUG, DIM, "better",
                   "1. the gate still adds dim-band Dice on top of augmentation"),
        severing_check(COMBO, AUG, "not_worse",
                       "2. and its connectivity cost does NOT come back"),
        severing_check(COMBO, GATE, "fewer",
                       "3. augmentation removes the gate's severing cost"),
        dice_check(rows, COMBO, AUG, CLEAR, "not_worse",
                   "4. the clearest band is not paid out"),
    ]

    print()
    if checks[0] and checks[1]:
        print("  -> COMPLEMENTARY. The gate's dim-band gain survives "
              "augmentation and its connectivity cost does not. K_focal_aug "
              "becomes the recommended model.")
    elif checks[1] and not checks[0]:
        print("  -> REDUNDANT. Augmentation had already bought whatever the "
              "gate was buying; the gate adds nothing on top. Recommend "
              "H_aug and stop weighting clDice.")
    elif checks[0] and not checks[1]:
        print("  -> A TRADE, same shape as E12. The gate buys dim-band "
              "coverage and charges connectivity for it, and augmentation "
              "does not absorb the charge. Report both halves; which arm to "
              "recommend then depends on whether the downstream task needs "
              "recall or connectivity.")
    else:
        print("  -> the combination is worse than its parts. Check first that "
              "the augmentation reached the batches (exp/augment.py) and that "
              "the gate is active (exp/test_gated.py) before interpreting.")
    if checks[2]:
        print("     Criterion 3 holds: whatever G_focal's severing cost was, "
              "augmentation removes it.")


if __name__ == "__main__":
    main()
