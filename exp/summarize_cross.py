"""E4: is "filtering beats the topology loss" a DRIVE fact or a general one?

stage 0 found that on DRIVE, dropping predicted components below 20 px closed
most of the betti-0 gap between plain BCE+Dice and the clDice loss. The claim
only means something if the two gains are put on the same scale, so this
reports them as a ratio per dataset:

  loss gain    = best topology loss - BCE+Dice, both unfiltered
  filter gain  = BCE+Dice filtered - BCE+Dice unfiltered

A ratio above 1 means the post-processing one-liner bought more than the loss
function did. The filter sizes are multiples of each dataset's own median
structure width squared, so 20 px on DRIVE and 133 px on TopoMortar are the
same point on the sweep.
"""
import csv
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "cross"
LOSSES = ("A_dice", "B_cldice", "E_cbdice")
KEYS = ("dice", "cldice", "betti0_err", "betti1_err")
OPERATING = 2.08          # the multiple DRIVE's 20 px corresponds to


def load(dataset: str) -> dict:
    """run -> width_multiple -> metric -> mean over the test images."""
    out = {}
    for score_path in sorted((RESULTS / dataset).glob("*/scores.csv")):
        rows = list(csv.DictReader(score_path.open()))
        per_multiple = {}
        for row in rows:
            per_multiple.setdefault(float(row["width_multiple"]), []).append(row)
        out[score_path.parent.name] = {
            multiple: {k: float(np.mean([float(r[k]) for r in group]))
                       for k in KEYS}
            for multiple, group in per_multiple.items()
        }
    return out


def across_seeds(data: dict, loss: str, multiple: float, key: str) -> np.ndarray:
    return np.array([scores[multiple][key]
                     for name, scores in data.items()
                     if name.rsplit("_s", 1)[0] == loss])


def main() -> None:
    for dataset in sys.argv[1:]:
        data = load(dataset)
        if not data:
            print(f"{dataset}: no runs yet\n")
            continue
        multiples = sorted(next(iter(data.values())))
        print(f"===== {dataset} ({len(data)} runs) =====")

        for key in KEYS:
            print(f"\n--- {key} ---")
            print(f"{'loss':12}" + "".join(f"{m:>10.2f}" for m in multiples))
            for loss in LOSSES:
                values = [across_seeds(data, loss, m, key) for m in multiples]
                if not len(values[0]):
                    continue
                print(f"{loss:12}" + "".join(f"{v.mean():10.4f}" for v in values))

        print("\n--- 過濾的增益 vs 換 loss 的增益 ---")
        print(f"{'metric':12}{'loss 增益':>12}{'過濾增益':>12}{'比值':>10}")
        for key in KEYS:
            better = -1 if "err" in key else 1
            base = across_seeds(data, "A_dice", 0.0, key)
            if not len(base):
                continue
            rivals = [across_seeds(data, loss, 0.0, key).mean()
                      for loss in LOSSES[1:]
                      if len(across_seeds(data, loss, 0.0, key))]
            # better = -1 for error metrics, so flip into "higher is better"
            # space to pick the winner, then flip the winner back before
            # subtracting. Doing the flip only once double-counts the sign.
            best_rival = better * max(better * np.array(rivals))
            loss_gain = better * (best_rival - base.mean())
            filtered = across_seeds(data, "A_dice", OPERATING, key)
            filter_gain = better * (filtered.mean() - base.mean())
            ratio = filter_gain / loss_gain if loss_gain != 0 else np.nan
            print(f"{key:12}{loss_gain:12.4f}{filter_gain:12.4f}{ratio:10.2f}")
        print()


if __name__ == "__main__":
    main()
