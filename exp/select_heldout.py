"""D-E's weight sweep under the held-out protocol.

WRITTEN AND SELFTESTED 2026-08-28, BEFORE THE FIRST HELD-OUT RUN STARTED.

WHAT THIS REPAIRS. exp/drive.py's "val" split is DRIVE's official TEST set.
Two things chose a checkpoint on it:

  best.pt              highest Dice over all 20 test images. This leaks: the
                       number reported from it is the maximum of ten draws on
                       the set it is reported on.
  rules (i)-(iv)       select on the odd test images, report on the even ones.
                       This does NOT leak into the reported half -- but it
                       still touches the test set before reporting, it halves
                       the reporting set to 10 images, and a reviewer is right
                       to object to both.

Under --protocol heldout the model is fitted on 15 of DRIVE's TRAINING images
and every selection rule reads the 5 held out from the same directory. The
test set is then read once, whole. Two consequences, one of them a gain:

  - No selection of any kind touches a reported image.
  - The paired test runs over 20 images instead of 10, so the same effect is
    measured with twice the pairs.

The cost is real and is not hidden: 15 training images instead of 20. Every
arm pays it equally, so the comparisons are unaffected, but the absolute Dice
is NOT comparable to the legacy runs or to published DRIVE numbers, and any
table mixing the two protocols is wrong. That is what the protocol column in
the output is for.

THE QUESTION. D-E -- extra loss weight on ground-truth centreline pixels -- is
the only intervention in this series that passed the seed gate (H_aug_clw
+249.5 ERL at t 5.23, six seeds of six, 2026-08-27). Its weight was fixed at 2
by an argument from vessel geometry and never swept. Three bases x four
weights x six seeds asks whether 2 is right, and whether the effect survives
on K_focal_aug, the strongest arm measured, which has never been crossed with
D-E at all.

PRE-REGISTERED PREDICTIONS.
  1. The effect survives the protocol change on H_aug: H_aug_clw2 beats H_aug.
  2. The response to weight is single-peaked, not monotone. Weight 8 makes
     the centreline outvote the vessel body and costs Dice without buying
     run length.
  3. The effect is SMALLER on K_focal_aug than on H_aug. Focal loss already
     up-weights hard pixels, and the centreline is where the model hesitates,
     so the two overlap.
  4. A_dice, unaugmented, stays the weakest of the three bases at every
     weight.

THE GATE is the repo's, unchanged: paired t over (image, seed) with t > 2,
every seed agreeing in sign, and at least three seeds.

  python exp/select_heldout.py --selftest
  python exp/select_heldout.py                       # after the runs land
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as rules_module
import summarize_selection as selection
import train

ROOT = train.RESULTS / "heldout"
SCORES = ROOT / "checkpoint_scores.csv"
RULE = "(iv) best clDice"
BASES = ("A_dice", "H_aug", "K_focal_aug")


def dev_points(run_dir: Path) -> list[dict]:
    """[{epoch, dice, cldice, betti0_err}] from log.csv -- the DEV scores.

    log.csv is what train.py writes at every validated epoch, and under the
    held-out protocol its validation set is the 5 dev images. So the honest
    selection table already exists on disk and needs no extra scoring pass;
    the checkpoints only ever have to be scored on the test set.
    """
    rows = list(csv.DictReader((run_dir / "log.csv").open()))
    return [{"epoch": int(row["epoch"]), "dice": float(row["dice"]),
             "cldice": float(row["cldice"]),
             "betti0_err": float(row["betti0_err"])} for row in rows]


def chosen_epochs(root: Path = ROOT, rule_name: str = RULE) -> dict:
    """{run: epoch} under `rule_name`, decided entirely on dev."""
    rule = dict(rules_module.rules())[rule_name]
    out = {}
    for path in sorted(root.glob("*_s*/log.csv")):
        points = dev_points(path.parent)
        if points:
            out[path.parent.name] = rule(points)["epoch"]
    return out


def erl_by_run(rows, epochs: dict) -> dict:
    """{run: {image: erl}} at each run's dev-chosen epoch, ALL test images.

    No is_selection_image filter, and that is the point: under this protocol
    no test image was used to select, so holding half of them back would
    throw away half the measurement for nothing.
    """
    out = defaultdict(dict)
    for row in rows:
        if epochs.get(row["run"]) == row["epoch"]:
            out[row["run"]][row["image"]] = row["erl"]
    return out


def compare(erl: dict, arm: str, base: str) -> dict | None:
    """Paired over (image, seed), gated the repo's way."""
    def seeds_of(config):
        return {run.rsplit("_s", 1)[1] for run in erl
                if run.rsplit("_s", 1)[0] == config}
    seeds = sorted(seeds_of(arm) & seeds_of(base))
    if not seeds:
        return None
    paired, per_seed = [], []
    for seed in seeds:
        mine, theirs = erl[f"{arm}_s{seed}"], erl[f"{base}_s{seed}"]
        images = sorted(set(mine) & set(theirs))
        inside = [(mine[i], theirs[i]) for i in images]
        paired.extend(inside)
        per_seed.append(float(np.mean([a - b for a, b in inside])))
    diffs = np.array([a - b for a, b in paired])
    result = stats.ttest_rel([a for a, _ in paired], [b for _, b in paired])
    return {"mean": float(diffs.mean()), "t": float(result.statistic),
            "seeds": len(per_seed), "per_seed": per_seed,
            "images": len(paired) // len(per_seed),
            "holds": bool(diffs.mean() > 0 and result.statistic > 2
                          and all(d > 0 for d in per_seed)
                          and len(per_seed) >= 3)}


def dice_by_run(rows, epochs: dict) -> dict:
    out = defaultdict(list)
    for row in rows:
        if epochs.get(row["run"]) == row["epoch"]:
            out[row["run"]].append(row["dice"])
    return {run: float(np.mean(values)) for run, values in out.items()}


def report(rows) -> None:
    epochs = chosen_epochs()
    erl = erl_by_run(rows, epochs)
    dice = dice_by_run(rows, epochs)
    print(f"protocol heldout: fit on 15 DRIVE training images, checkpoint "
          f"chosen on the 5 held out from the same directory,")
    print(f"reported on all 20 test images. NOT comparable in absolute value "
          f"to the legacy runs.\n")
    print(f"D-E's weight, at {RULE}, ERL paired on (image, seed)\n")
    header = (f"  {'arm':<26}{'seeds':>6}{'dERL':>10}{'t':>7}{'dDice':>9}"
              f"  verdict")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for base in BASES:
        for weight in train.CENTRELINE_WEIGHTS:
            arm = f"{base}_clw{weight}"
            got = compare(erl, arm, base)
            if got is None:
                print(f"  {arm + ' - ' + base:<26}{'not trained yet':>32}")
                continue
            delta = float(np.mean(
                [dice[f"{arm}_s{s}"] - dice[f"{base}_s{s}"]
                 for s in sorted({r.rsplit('_s', 1)[1] for r in erl
                                  if r.rsplit('_s', 1)[0] == arm})
                 if f"{arm}_s{s}" in dice and f"{base}_s{s}" in dice]))
            print(f"  {arm + ' - ' + base:<26}{got['seeds']:>6}"
                  f"{got['mean']:>+10.1f}{got['t']:>7.2f}{delta:>+9.4f}"
                  f"  {'HOLDS' if got['holds'] else 'fails'}")
            if not got["holds"]:
                print(f"  {'':<26}per seed "
                      f"[{' '.join(f'{d:+.0f}' for d in got['per_seed'])}]")
        print()
    print(f"  A Dice cost of -0.0009 is one seed standard deviation "
          f"({rules_module.SEED_SD_DICE}). Run length bought at a Dice cost")
    print("  is a trade to state, not a free win.")


def selftest() -> None:
    for base in BASES:
        assert base in train.CONFIGS, base
        for weight in train.CENTRELINE_WEIGHTS:
            name = f"{base}_clw{weight}"
            assert name in train.CONFIGS, name
            assert train.centreline_weight(name) == float(weight)
    print(f"every arm this script names is in CONFIGS: "
          f"{len(BASES)} bases x {len(train.CENTRELINE_WEIGHTS)} weights")

    # The selection must come off log.csv -- the dev scores -- and must NOT
    # be the epoch that happens to look best on the test set. A synthetic run
    # where the two disagree is the only way to assert that.
    import tempfile
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "H_aug_clw2_s0"
        root.mkdir(parents=True)
        with (root / "log.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "loss", "dice", "cldice", "betti0_err",
                             "betti1_err", "hd95", "minutes"])
            # Epoch 50 wins on dev clDice; epoch 10 will win on test.
            writer.writerow([10, 0.5, 0.82, 0.800, 90, 30, 4.0, 1.0])
            writer.writerow([50, 0.4, 0.81, 0.860, 70, 25, 3.9, 5.0])
        picked = chosen_epochs(Path(raw))
        assert picked == {"H_aug_clw2_s0": 50}, picked
    print("the rule reads dev clDice off log.csv, not anything test-side")

    # And the reported half must be ALL of the test images, not the even ones
    # the legacy path uses. If this ever silently reverts, every effect size
    # below is measured on half the data it claims.
    rows = [{"run": "H_aug_clw2_s0", "epoch": 50, "image": f"{i:02d}",
             "erl": 100.0, "dice": 0.8} for i in range(1, 21)]
    picked = erl_by_run(rows, {"H_aug_clw2_s0": 50})
    assert len(picked["H_aug_clw2_s0"]) == 20, len(picked["H_aug_clw2_s0"])
    assert any(rules_module.is_selection_image(i)
               for i in picked["H_aug_clw2_s0"]), "odd images were dropped"
    print("the verdict reports on all 20 test images, not the even half")

    # The gate must be the repo's, and must refuse a split-sign effect however
    # large its t is -- that is the whole reason the sign rule exists.
    erl = {}
    for seed in range(3):
        # Two seeds up by 300, one down by 30: mean positive, t large, and
        # it must still fail.
        offsets = (300.0, 300.0, -30.0)
        erl[f"H_aug_clw2_s{seed}"] = {f"{i:02d}": 2000.0 + offsets[seed]
                                      for i in range(1, 21)}
        erl[f"H_aug_s{seed}"] = {f"{i:02d}": 2000.0 for i in range(1, 21)}
    got = compare(erl, "H_aug_clw2", "H_aug")
    assert got["seeds"] == 3 and got["images"] == 20, got
    assert got["mean"] > 0 and got["t"] > 2, got
    assert not got["holds"], "a split-sign effect passed the gate"
    print(f"the gate refuses a split-sign effect (mean {got['mean']:+.0f}, "
          f"t {got['t']:.1f}) -- sign agreement, not just significance")

    # A missing arm returns None rather than an empty comparison that reads
    # as a negative result.
    assert compare(erl, "K_focal_aug_clw4", "K_focal_aug") is None
    print("an untrained arm reports None, not a false negative")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not SCORES.exists():
        raise SystemExit(f"{SCORES} not built yet -- run\n"
                         f"  python exp/sweep_score.py --results {ROOT}")
    report(selection.load(SCORES))


if __name__ == "__main__":
    main()
