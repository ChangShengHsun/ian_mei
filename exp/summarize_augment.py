"""E14: does what we SHOW the network beat what we ask of it?

Twenty experiments compared loss functions on a pipeline with no augmentation
beyond random crops. The CoLeTra paper (arXiv:2503.05541, Table 1) measures, on
this same dataset, Betti error 3.687 -> 1.354 from plain data augmentation and
1.354 -> 1.282 from their method on top of it. The first of those is a 63% cut,
an order of magnitude larger than anything this series has measured from
changing the loss -- E5 found that on clean labels the losses sit closer
together than two seeds of one loss.

So the two arms both keep BCE+Dice and change only the input:

  H_aug        random symmetry of the square + gamma/gain/bias jitter
  I_coletra    the same, plus CoLeTra: paste inpainted (vessel-free) content
               over random foreground pixels and leave the LABEL alone, so the
               network is taught that structure which looks broken is connected

Criteria are fixed here BEFORE the six runs finish. Two rules from earlier
experiments are built in and neither is optional:

  - the seed gate (E5): a pooled t is not evidence on its own, because the
    images are not independent replicates of a training run.
  - the severing count (E10): only 7% of "breaks" sever the prediction, so a
    drop in raw break count is a claim about centreline COVERAGE. Any
    topological claim has to be made with the validated metric. E7 was caught
    by exactly this: its winning arm cut 25 breaks per image and severed just
    as many.

  python exp/summarize_augment.py            # the verdict
  python exp/summarize_augment.py --selftest # check the plumbing

Reads results/stratify.csv and results/break_lengths.csv, both regenerated
after the runs exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_confidence as confidence
import summarize_gated as gated

DIM, CLEAR = "Q1_dimmest", "Q4_clearest"
MIN_SIZE = gated.PRIMARY_MIN_SIZE
BASELINE, AUG, COLETRA = "A_dice", "H_aug", "I_coletra"


def paired_seeds() -> tuple[str, ...]:
    """Seeds for which ALL THREE arms have a finished checkpoint.

    Not a constant. E12 wrote its seed range out by hand in two analysis
    scripts, trained seeds 3-5, scored the old three, and printed a
    complete-looking verdict from them. The seed list has to come from disk.

    It also has to be an INTERSECTION: an extra seed of one arm pairs with
    nothing, and averaging it in would compare two different populations.
    """
    import train
    runs = set(train.trained_runs())
    seeds = {name.rsplit("_s", 1)[1] for name in runs}
    return tuple(sorted(
        seed for seed in seeds
        if all(f"{arm}_s{seed}" in runs for arm in (BASELINE, AUG, COLETRA))))


def dice_check(rows, better: str, worse: str, band: str, want: str,
               label: str) -> bool:
    result = gated.paired(rows, better, worse, band, MIN_SIZE, "dice")
    per = gated.per_seed(rows, better, worse, band, MIN_SIZE, "dice")
    return confidence.check(label, result["mean"], result["t"], per, want)


def severing_check(better: str, worse: str, label: str) -> bool:
    left = confidence.severing_per_image(better, DIM)
    right = confidence.severing_per_image(worse, DIM)
    mean, t, per = confidence.paired_counts(left, right)
    return confidence.check(label, mean, t, per, "fewer")


def selftest() -> None:
    """The reused pieces already have their own assertions; what is worth
    checking here is that this file asks for configs that exist."""
    import train
    for name in (BASELINE, AUG, COLETRA):
        assert name in train.CONFIGS, name
    for name in (AUG, COLETRA):
        assert train.AUGMENTS.get(name), f"{name} has no augmentation"
    assert train.AUGMENTS[COLETRA] != train.AUGMENTS[AUG], \
        "the two arms would be identical"
    print(f"configs present: {BASELINE}, {AUG} {train.AUGMENTS[AUG]}, "
          f"{COLETRA} {train.AUGMENTS[COLETRA]}")
    extra = set(train.AUGMENTS[COLETRA]) - set(train.AUGMENTS[AUG])
    assert extra == {"coletra"}, extra
    print(f"  the arms differ by exactly {sorted(extra)}")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    rows = gated.load()
    present = {r["config"] for r in rows}
    missing = {AUG, COLETRA} - present
    if missing:
        print(f"still untrained: {sorted(missing)} -- rerun stratify.py and "
              f"break_lengths.py once they exist\n")

    seeds = paired_seeds()
    gated.SEEDS = seeds       # per_seed() already skips a seed with no pairs;
    confidence.SEEDS = seeds  # this is so the report says which ones were used
    print("=== E14 pre-registered verdict ===")
    print(f"    (paired seeds: {', '.join(seeds)})\n")
    checks = [
        dice_check(rows, AUG, BASELINE, DIM, "better",
                   "1. augmentation beats the baseline in the dimmest band"),
        severing_check(AUG, BASELINE,
                       "2. augmentation cuts SEVERING breaks, not just breaks"),
        severing_check(COLETRA, AUG,
                       "3. CoLeTra cuts severing breaks on top of augmentation"),
        dice_check(rows, AUG, BASELINE, CLEAR, "not_worse",
                   "4. augmentation does not lose the clearest band"),
    ]

    print()
    if checks[0] and checks[1]:
        print("  -> what we SHOW the network beats what we ask of it: "
              "augmentation moves the dim band AND its connectivity.")
        if checks[2]:
            print("     CoLeTra adds a further topological gain on top.")
        else:
            print("     CoLeTra adds nothing measurable on top of it.")
    elif checks[0]:
        print("  -> augmentation improves the dim band, but the gain is "
              "COVERAGE, not connectivity. Report it as centreline recall.")
    else:
        print("  -> augmentation did not help the dim band. That contradicts "
              "the published DRIVE result, so suspect our setup before "
              "believing it -- check the augmentation actually reached the "
              "batches (exp/augment.py runs that check).")


if __name__ == "__main__":
    main()
