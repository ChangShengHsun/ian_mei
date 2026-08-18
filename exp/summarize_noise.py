"""E5: does label noise hurt the topology losses more than it hurts BCE+Dice?

TopoMortar ships three labellings of the same training images -- accurate,
pseudo (94.2% pixel agreement, over-segmented) and noisy (81.4%, and it REMOVES
structure: 13.2% foreground against accurate's 30.4%). Every arm is scored
against the accurate test labels, so the only variable is what the model was
taught.

E4 established that betti-0 and betti-1 give opposite answers about post-
processing, so this reports them separately. The bad case for the field is
specifically noise hurting betti-1 more for the topology losses, because loops
are the one thing post-processing cannot repair.
"""
import csv
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "cross"
KINDS = ("topomortar", "topomortar_pseudo", "topomortar_noisy")
LABELS = {"topomortar": "accurate", "topomortar_pseudo": "pseudo",
          "topomortar_noisy": "noisy"}
LOSSES = ("A_dice", "B_cldice", "E_cbdice")
KEYS = ("dice", "cldice", "betti0_err", "betti1_err")
OPERATING = 2.08


def mean_of(kind: str, loss: str, key: str, multiple: float) -> float:
    values = []
    for score_path in (RESULTS / kind).glob(f"{loss}_s*/scores.csv"):
        rows = [r for r in csv.DictReader(score_path.open())
                if float(r["width_multiple"]) == multiple]
        if rows:
            values.append(np.mean([float(r[key]) for r in rows]))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    multiple = float(sys.argv[1]) if len(sys.argv) > 1 else OPERATING
    print(f"過濾門檻 = {multiple} x w^2（0 代表不過濾）\n")

    for key in KEYS:
        print(f"--- {key} ---")
        print(f"{'loss':12}" + "".join(f"{LABELS[k]:>12}" for k in KINDS)
              + f"{'accurate-noisy':>16}")
        for loss in LOSSES:
            scores = [mean_of(k, loss, key, multiple) for k in KINDS]
            # Degradation in the direction that means "worse", so every row is
            # comparable no matter whether the metric is an error or a score.
            drop = (scores[2] - scores[0]) if "err" in key else (scores[0] - scores[2])
            print(f"{loss:12}" + "".join(f"{s:12.4f}" for s in scores)
                  + f"{drop:16.4f}")
        print()

    print("=== 主要對照：噪聲對誰傷害比較大 ===")
    print("（正值 = 該 loss 被噪聲弄壞得比 BCE+Dice 更嚴重）")
    print(f"{'metric':14}" + "".join(f"{l:>14}" for l in LOSSES[1:]))
    for key in KEYS:
        baseline = [mean_of(k, "A_dice", key, multiple) for k in KINDS]
        base_drop = ((baseline[2] - baseline[0]) if "err" in key
                     else (baseline[0] - baseline[2]))
        cells = ""
        for loss in LOSSES[1:]:
            scores = [mean_of(k, loss, key, multiple) for k in KINDS]
            drop = (scores[2] - scores[0]) if "err" in key else (scores[0] - scores[2])
            cells += f"{drop - base_drop:14.4f}"
        print(f"{key:14}{cells}")


if __name__ == "__main__":
    main()
