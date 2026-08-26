"""D1's verdict: did the tangent head learn, and did it help?

WRITTEN AND SELFTESTED 2026-08-27, BEFORE THE FIRST _dir RUN FINISHED.

Two questions, and the first gates the second.

  D1.a  DID THE HEAD LEARN THE AXIS AT ALL? Agreement between the predicted
        field and the ground truth's, on vessel pixels, in double-angle chord
        units (0 = same axis, 2 = a quarter turn off). Reported against two
        references that need no learning: a CONSTANT field, and the field a
        classical Hessian filter computes from the raw image with no training
        whatsoever. A head that does not beat the constant did not learn; a
        head that does not beat the classical filter learned nothing the
        image did not already say, and D1's premise -- that the network can
        infer direction where the vessel is invisible -- is dead.

        This is the question that must be asked FIRST, because if the answer
        is no, then "the auxiliary task did not improve segmentation" means
        the head failed, not that direction is useless, and the two would be
        reported as the same result.

  D1.b  DID THE AUXILIARY TASK IMPROVE SEGMENTATION? A_dice_dir against
        A_dice and H_aug_dir against H_aug, paired on (image, seed), under
        the repo's gate. The _dir arms carry 34 extra parameters out of
        117,393, so a difference is the auxiliary TASK and not capacity.

  PRE-REGISTERED PREDICTION. D1.a holds and D1.b does not. An auxiliary head
  that shares a decoder usually buys a small representation gain and rarely
  a measurable one on 20 training images, and E13b already measured that this
  dataset punishes added capacity. D1's value is expected to be the FIELD --
  as C1's cost map -- not the segmentation. If D1.b holds anyway, that is a
  better result than predicted and gets reported as a surprise.

  python exp/summarize_direction.py --selftest
  python exp/summarize_direction.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as rules_module
import summarize_selection as selection

# (the _dir arm, the arm it must be compared against)
PAIRS = (("A_dice_dir", "A_dice"), ("H_aug_dir", "H_aug"))
RULE = "(iv) best clDice"
FIELD = selection.SWEEP / "direction_quality.csv"


def erl_by_run(rows, rule_name: str = RULE) -> dict:
    """{run: {image: erl}} at the epoch `rule_name` picks, report half only."""
    points = selection.selection_points(rows)
    rule = dict(rules_module.rules())[rule_name]
    wanted = {run: rule(these)["epoch"] for run, these in points.items()}
    out = defaultdict(dict)
    for row in rows:
        if rules_module.is_selection_image(row["image"]):
            continue
        if wanted.get(row["run"]) == row["epoch"]:
            out[row["run"]][row["image"]] = row["erl"]
    return out


def compare(erl: dict, arm: str, base: str) -> dict | None:
    """Paired over (image, seed), gated the repo's way."""
    seeds = sorted({run.rsplit("_s", 1)[1] for run in erl
                    if run.rsplit("_s", 1)[0] == arm}
                   & {run.rsplit("_s", 1)[1] for run in erl
                      if run.rsplit("_s", 1)[0] == base})
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
            "holds": bool(diffs.mean() > 0 and result.statistic > 2
                          and all(d > 0 for d in per_seed)
                          and len(per_seed) >= 3)}


def selftest() -> None:
    rows = []
    for arm, value in (("A_dice_dir", 2400.0), ("A_dice", 2000.0)):
        for seed in range(3):
            for image in range(1, 21):
                for epoch, cldice in ((10, 0.80), (50, 0.85)):
                    rows.append({
                        "run": f"{arm}_s{seed}", "config": arm,
                        "seed": str(seed), "epoch": epoch,
                        "image": f"{image:02d}", "dice": 0.82,
                        "cldice": cldice, "betti0_err": 50.0,
                        # Only epoch 50 is the rule's pick; epoch 10 carries
                        # a value that must never reach the verdict.
                        "erl": value if epoch == 50 else 99999.0,
                        "skel_px": 8000.0})
    erl = erl_by_run(rows)
    assert set(erl["A_dice_s0"]) == {f"{i:02d}" for i in range(2, 21, 2)}
    assert all(v == 2000.0 for v in erl["A_dice_s0"].values()), "wrong epoch"
    print("the verdict reads rule (iv)'s epoch on the report half; the "
          "99999 planted on the other epoch never reaches it")

    got = compare(erl, "A_dice_dir", "A_dice")
    assert got["holds"] and abs(got["mean"] - 400.0) < 1e-6, got
    print(f"three seeds all improving: mean {got['mean']:+.0f}, gate HOLDS")

    for row in rows:
        if row["run"] == "A_dice_dir_s2" and row["epoch"] == 50:
            row["erl"] = 1995.0
    flipped = compare(erl_by_run(rows), "A_dice_dir", "A_dice")
    assert flipped["mean"] > 0 and not flipped["holds"], flipped
    print(f"  one seed disagreeing fails at mean {flipped['mean']:+.0f}: "
          f"{[round(d) for d in flipped['per_seed']]}")
    print("all checks passed")


def report_field() -> None:
    print("=== D1.a: did the tangent head learn the axis? ===")
    if not FIELD.exists():
        print(f"{FIELD.name} not built yet -- run exp/score_direction.py\n")
        return
    import csv
    rows = list(csv.DictReader(FIELD.open()))
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if rules_module.is_selection_image(row["image"]):
            continue
        for key in ("head", "constant", "classical"):
            grouped[row["config"]][key].append(float(row[key]))
    header = (f"  {'arm':<14}{'head':>8}{'constant':>10}{'classical':>11}"
              f"   verdict")
    print("mean axis gap on vessel pixels, 0 = right, 2 = a quarter turn off")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config, values in sorted(grouped.items()):
        head = float(np.mean(values["head"]))
        constant = float(np.mean(values["constant"]))
        classical = float(np.mean(values["classical"]))
        if head >= constant:
            note = "DID NOT LEARN"
        elif head >= classical:
            note = "learned nothing the image did not already say"
        else:
            note = "beats both references"
        print(f"  {config:<14}{head:8.3f}{constant:10.3f}{classical:11.3f}"
              f"   {note}")
    print("\nIf the head does not beat the constant reference, D1.b below is")
    print("not a result about direction -- it is a result about a head that")
    print("failed, and must not be read as the former.\n")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    report_field()
    print("=== D1.b: did the auxiliary task improve segmentation? ===")
    if not selection.SCORES.exists():
        raise SystemExit(f"{selection.SCORES} not built yet")
    erl = erl_by_run(selection.load())
    header = f"  {'comparison':<28}{'seeds':>7}{'vs base':>10}{'t':>7}  gate"
    print(f"ERL at rule (iv), report half, paired on (image, seed)\n")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arm, base in PAIRS:
        got = compare(erl, arm, base)
        if got is None:
            print(f"  {arm + ' - ' + base:<28}{'not trained yet':>7}")
            continue
        print(f"  {arm + ' - ' + base:<28}{got['seeds']:7d}"
              f"{got['mean']:+10.1f}{got['t']:7.2f}  "
              f"{'HOLDS' if got['holds'] else 'fails'}")
        if not got["holds"]:
            print(f"  {'':<28}per seed "
                  f"[{' '.join(f'{d:+.0f}' for d in got['per_seed'])}]")
    print("\nPre-registered prediction: D1.a holds, D1.b does not. D1's value")
    print("is expected to be the FIELD, as C1's cost map, not the")
    print("segmentation. A D1.b that holds is a better result than predicted.")


if __name__ == "__main__":
    main()
