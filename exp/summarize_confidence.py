"""E12: does the confidence gate survive a replication it was not chosen by?

E7 pre-registered three criteria about F_gated, the CONTRAST gate, and all the
interesting ones failed. G_focal was the control arm -- same weighted clDice,
weight read from the model's own uncertainty instead of from the image -- and
it beat both F_gated (+0.0177 in the dimmest band) and B_cldice (+0.0136), on
all three seeds.

That is a post-hoc finding. It was not the hypothesis, the criteria were not
written for it, and it is exactly the shape of result that does not replicate:
three seeds, chosen after looking, with a spread (+0.0043 / +0.0070 / +0.0297)
where one seed does seven times what another does. E2's headline had the same
shape and survived; E5's had the same shape and did not.

So E12 is the confirmation run. Three fresh seeds of B_cldice and G_focal, and
the criteria below are fixed BEFORE those six runs finish. The seed gate is the
one E5 forced into existence: a pooled t is not evidence on its own, because
the images are not independent replicates of a training run.

Criterion 3 is the one E10 demanded and E7 could not supply. E7's break column
counts missed centreline, of which only 7% severs the prediction, so a drop
there is a coverage claim. The topological claim needs the severing count, and
break_lengths.py now produces it for all runs.

  python exp/summarize_confidence.py            # the verdict
  python exp/summarize_confidence.py --selftest # check the severs aggregation

Reads results/stratify.csv and results/break_lengths.csv, both of which must
have been regenerated after the s3-s5 runs exist.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_gated as gated

RESULTS = Path(__file__).resolve().parent / "results"
SEEDS = ("0", "1", "2", "3", "4", "5")
DIM, CLEAR = "Q1_dimmest", "Q4_clearest"
MIN_SIZE = gated.PRIMARY_MIN_SIZE
TREATMENT, BASELINE = "G_focal", "B_cldice"


def severing_per_image(config: str, band: str | None = DIM,
                       path: Path | None = None) -> dict:
    """Severing breaks per (image, seed) for one config.

    E10's classification: a break severs when dilating it by one pixel touches
    two or more distinct predicted components. Counting rows rather than
    reading a rate, so an image with no severing break contributes a zero
    rather than being absent -- otherwise the easy images silently leave the
    comparison.
    """
    path = path or RESULTS / "break_lengths.csv"
    counts: dict = {}
    for row in csv.DictReader(path.open()):
        if row["config"] != config:
            continue
        if band is not None and row["band"] != band:
            continue
        key = (row["image"], row["run"].rsplit("_s", 1)[1])
        counts[key] = counts.get(key, 0) + (row["kind"] == "severs")
    return counts


def paired_counts(left: dict, right: dict) -> tuple[float, float, list]:
    """Mean difference, pooled t, and the per-seed means."""
    keys = sorted(set(left) & set(right))
    diff = np.array([left[k] - right[k] for k in keys], dtype=float)
    t = (diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
         if len(diff) > 1 and diff.std() > 0 else 0.0)
    per = []
    for seed in SEEDS:
        picked = [left[k] - right[k] for k in keys if k[1] == seed]
        if picked:
            per.append(float(np.mean(picked)))
    return float(diff.mean()), float(t), per


def check(label: str, mean: float, t: float, per: list, want: str) -> bool:
    """A criterion passes only on the pooled test AND seed agreement."""
    if want == "better":
        passed = mean > 0 and t > 2.0 and all(p > 0 for p in per)
    elif want == "fewer":
        passed = mean < 0 and t < -2.0 and all(p < 0 for p in per)
    else:  # non-inferiority
        passed = not (mean < 0 and t < -2.0 and all(p < 0 for p in per))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"         {mean:+.4f}  t={t:.2f}  n_seeds={len(per)}   "
          f"per seed: {' '.join(f'{p:+.4f}' for p in per)}")
    return passed


def selftest() -> None:
    """severing_per_image must count zeros, not drop them, and must split by
    seed. Both failures would look like a working comparison."""
    rows = [{"config": "X", "run": "X_s0", "image": "a", "band": DIM,
             "kind": "severs"},
            {"config": "X", "run": "X_s0", "image": "a", "band": DIM,
             "kind": "intact"},
            {"config": "X", "run": "X_s1", "image": "a", "band": DIM,
             "kind": "intact"},
            {"config": "X", "run": "X_s0", "image": "b", "band": CLEAR,
             "kind": "severs"}]
    path = RESULTS / "_selftest_break_lengths.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        counts = severing_per_image("X", DIM, path)
        print(f"counts: {counts}")
        # ("a","0") saw one severs and one intact -> 1.
        # ("a","1") saw only an intact -> 0, and must be PRESENT as a zero.
        assert counts == {("a", "0"): 1, ("a", "1"): 0}, counts
        assert ("b", "0") not in counts, "the CLEAR-band row leaked into DIM"
        print("  zero-severance images are kept, and the band filter holds")

        mean, t, per = paired_counts({("a", "0"): 1.0, ("a", "1"): 3.0},
                                     {("a", "0"): 2.0, ("a", "1"): 1.0})
        print(f"  seed-flip case: mean {mean:+.1f}, per seed {per}")
        assert per == [-1.0, 2.0], per
        assert not check("synthetic seed flip", mean, t, per, "fewer")
        print("all checks passed")
    finally:
        path.unlink()


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    rows = gated.load()
    present = {r["run"].rsplit("_s", 1)[1] for r in rows
               if r["config"] == TREATMENT}
    missing = set(SEEDS) - present
    if missing:
        print(f"seeds still missing for {TREATMENT}: {sorted(missing)} -- "
              f"rerun stratify.py and break_lengths.py once they exist")

    # summarize_gated was written for E7's three seeds and caps per_seed() at
    # (0, 1, 2). The paired t already reads every row in the CSV, so leaving
    # this alone gives a mean over six seeds beside a seed gate over three --
    # the gate silently protecting less than it appears to.
    gated.SEEDS = tuple(sorted(present))

    print(f"=== E12 pre-registered verdict: {TREATMENT} vs {BASELINE} ===")
    print(f"    ({len(present)} seeds present of {len(SEEDS)})\n")

    gain = gated.paired(rows, TREATMENT, BASELINE, DIM, MIN_SIZE, "dice")
    per = gated.per_seed(rows, TREATMENT, BASELINE, DIM, MIN_SIZE, "dice")
    one = check("1. G beats B on Dice in the dimmest band, filtered",
                gain["mean"], gain["t"], per, "better")

    cost = gated.paired(rows, TREATMENT, BASELINE, CLEAR, MIN_SIZE, "dice")
    per_cost = gated.per_seed(rows, TREATMENT, BASELINE, CLEAR, MIN_SIZE,
                              "dice")
    two = check("2. G does not lose to B in the clearest band",
                cost["mean"], cost["t"], per_cost, "not_worse")

    left, right = (severing_per_image(TREATMENT), severing_per_image(BASELINE))
    mean, t, per_sev = paired_counts(left, right)
    three = check("3. G has FEWER severing breaks in the dimmest band "
                  "(the topological claim, E10's metric)",
                  mean, t, per_sev, "fewer")

    if one and two and three:
        print("\n  -> confirmed: the confidence gate improves the dim band, "
              "costs nothing in the clear band, and the gain is topological.")
    elif one and two:
        print("\n  -> partial: the Dice gain replicates but it is COVERAGE, "
              "not connectivity. Report it as centreline recall, not topology.")
    elif not one:
        print("\n  -> not confirmed. The E7 result was post-hoc and did not "
              "survive fresh seeds, which is what post-hoc results do.")


if __name__ == "__main__":
    main()
