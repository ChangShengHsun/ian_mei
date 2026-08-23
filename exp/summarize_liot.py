"""E16's verdict. Written and selftested BEFORE any J_liot run was launched.

The arms are H_aug and J_liot. They share a loss (BCE + soft Dice), an
augmentation set, a width and a seed list; they differ in one thing, which is
whether the network is shown grey pixels or LIOT's four contrast-invariant
channels. So a difference here is attributable to the representation.

Where the criteria come from:

  1  E15 put a ceiling on each contrast band and found the dimmest at 17-24%
     of achievable while the two brightest sit at 98-99%. Q1 is the entire
     remaining budget, and contrast is what defines Q1, so a contrast-
     invariant input has to move it or it has done nothing.
  2  E10 measured that 93% of "breaks" are not breaks in connectivity, so a
     topological claim has to be made with the severing count. E7, E12 and
     E14's criterion 1 were all decided by this rule; it is not negotiable
     here either.
  3  LIOT throws away absolute intensity. The clearest band is where absolute
     intensity is most informative and where E15 says there is nothing to win,
     so that band is where the cost, if there is one, will show. Stated as
     non-inferiority: this is the criterion expected to be at risk.
  4  The point of the experiment is comparison against the CURRENT best input,
     not against the original baseline. Beating A_dice would prove only that
     augmentation works, which E14 already established.

  python exp/summarize_liot.py --selftest
  python exp/summarize_liot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_confidence as confidence
import summarize_gated as gated

DIM, CLEAR = "Q1_dimmest", "Q4_clearest"
MIN_SIZE = gated.PRIMARY_MIN_SIZE
GREY, LIOT, PLAIN = "H_aug", "J_liot", "A_dice"


def paired_seeds(*arms: str) -> tuple[str, ...]:
    """Seeds every named arm has finished. See summarize_augment.paired_seeds:
    E12 shipped a verdict computed on seeds it had not trained because this
    was a hardcoded constant."""
    import train
    runs = set(train.trained_runs())
    seeds = {name.rsplit("_s", 1)[1] for name in runs}
    return tuple(sorted(seed for seed in seeds
                        if all(f"{arm}_s{seed}" in runs for arm in arms)))


def dice_check(rows, better, worse, band, want, label) -> bool:
    result = gated.paired(rows, better, worse, band, MIN_SIZE, "dice")
    per = gated.per_seed(rows, better, worse, band, MIN_SIZE, "dice")
    return confidence.check(label, result["mean"], result["t"], per, want)


def severing_check(better, worse, label) -> bool:
    left = confidence.severing_per_image(better, DIM)
    right = confidence.severing_per_image(worse, DIM)
    mean, t, per = confidence.paired_counts(left, right)
    return confidence.check(label, mean, t, per, "fewer")


def selftest() -> None:
    import train
    for name in (GREY, LIOT, PLAIN):
        assert name in train.CONFIGS, name
    assert train.uses_liot(LIOT) and not train.uses_liot(GREY)
    assert train.AUGMENTS[LIOT] == train.AUGMENTS[GREY], \
        "the arms must differ only in the input representation"
    assert train.CONFIGS[LIOT] == train.CONFIGS[GREY], \
        "same blurpool setting and same extra loss term"
    assert train.base_width(LIOT) == train.base_width(GREY)
    grey = train.build_model(GREY)
    coded = train.build_model(LIOT)
    assert grey.enc1[0].in_channels == 1
    assert coded.enc1[0].in_channels == 4
    print(f"{LIOT} vs {GREY}: identical except in_channels "
          f"{grey.enc1[0].in_channels} -> {coded.enc1[0].in_channels}")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    seeds = paired_seeds(GREY, LIOT)
    if not seeds:
        raise SystemExit(f"no seed has both {GREY} and {LIOT} -- nothing to "
                         f"compare yet")
    gated.SEEDS = seeds
    confidence.SEEDS = seeds
    rows = gated.load()

    print("=== E16 pre-registered verdict: LIOT input vs grey input ===")
    print(f"    (paired seeds: {', '.join(seeds)})\n")

    checks = [
        dice_check(rows, LIOT, GREY, DIM, "better",
                   "1. LIOT beats grey in the dimmest band"),
        severing_check(LIOT, GREY,
                       "2. LIOT cuts SEVERING breaks in the dimmest band"),
        dice_check(rows, LIOT, GREY, CLEAR, "not_worse",
                   "3. LIOT does not lose the clearest band"),
        dice_check(rows, LIOT, PLAIN, DIM, "better",
                   "4. LIOT beats the un-augmented baseline in the dim band"),
    ]

    print()
    if checks[0] and checks[1]:
        print("  -> deleting contrast at the input beats teaching the network "
              "to see through it. The dim band moves AND its connectivity "
              "does, on top of augmentation.")
        if not checks[2]:
            print("     It is a trade: the clearest band pays for it. Report "
                  "both halves, the way E2 requires.")
    elif checks[0]:
        print("  -> coverage again, not connectivity -- the same shape as E7 "
              "and E12. Report it as centreline recall.")
    elif checks[1]:
        print("  -> connectivity without coverage. Unusual in this series; "
              "check the severing count against ERL before believing it.")
    elif checks[3]:
        print("  -> LIOT beats no augmentation but not augmentation. The "
              "representation is subsumed by what H_aug already teaches.")
    else:
        print("  -> LIOT does not help here. Before concluding that about "
              "LIOT, note what is NOT tested: the paper's claim is "
              "cross-dataset generalisation, and this is one dataset "
              "train-and-test. See the report's limits section.")


if __name__ == "__main__":
    main()
