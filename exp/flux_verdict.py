"""The flux arms, scored against the three predictions registered before them.

WRITTEN 2026-09-11. exp/run_flux.sh registered these before any of its 48
runs existed:

  1. Every arm's flux head beats its own mask-derived reference on the
     centreline the mask MISSES, on at least 10 of 12 seeds.
  2. `_flux` does NOT beat its namesake on segmentation through the gate.
     Predicted as a NULL, because D1's diagnosis says it should repeat: an
     auxiliary head can be ignored by the mask head. Registered that way so
     a surprise is a real surprise and so the readout, not the auxiliary
     loss, gets the credit if anything works.
  3. The gap in prediction 1 is larger for the arms whose masks miss more.

WHAT IS AND IS NOT BEING ASKED. This file scores the MASK readout, which is
the only readout that exists. flux.py's docstring is explicit that a flux
head bolted beside a mask head is D1 and deserves D1's result, so a null on
prediction 2 is the expected outcome and not a verdict on the
representation. The readout that decodes the field is a separate build and a
separate question.

Prediction 2 reads frontier.csv, which run_flux.sh already rebuilt, so no
forward pass is needed for it. Predictions 1 and 3 need the field, so they
run on the 5 DEV images -- a diagnostic, chosen on the split diagnostics are
allowed to use.

  python exp/flux_verdict.py --report
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration
import drive
import flux
import select_heldout as heldout
import train

PAIRS = (("A_dice_flux", "A_dice"), ("H_aug_flux", "H_aug"),
         ("H_aug_clw16_flux", "H_aug_clw16"),
         ("K_focal_aug_flux", "K_focal_aug"))
# From the headroom scan, 2026-09-06: the fraction of DRIVE centreline each
# namesake puts below p=0.01. Prediction 3 says the flux gap tracks this.
MISS_RATE = {"A_dice": 9.2, "H_aug": 8.5, "H_aug_clw16": None,
             "K_focal_aug": None}


def frontier(path: Path) -> dict:
    """{(config, seed): {threshold: (dice, traced)}}"""
    out = defaultdict(dict)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out[(row["config"], row["seed"])][float(row["threshold"])] = (
                float(row["dice"]), float(row["traced"]))
    return out


def report() -> None:
    test = frontier(heldout.ROOT / "frontier.csv")
    dev = frontier(heldout.ROOT / "frontier_dev.csv")

    print("=== the flux arms, against three predictions made before them ===\n")
    print("--- prediction 2: does the auxiliary head help SEGMENTATION? ---")
    print("    Registered as a NULL. `_flux` against its namesake, traced ERL,")
    print("    paired on seed, through calibration.decide. Read at the shared")
    print("    0.5 and at each arm's own dev-Dice-peak threshold.\n")
    print(f"    {'comparison':34}{'at 0.5':>22}{'at own':>22}")
    for flux_arm, base in PAIRS:
        cells = []
        for mode in ("0.5", "own"):
            pairs_, per_seed = [], []
            seeds = sorted({s for (c, s) in test if c == flux_arm}
                           & {s for (c, s) in test if c == base})
            for seed in seeds:
                got = []
                for config in (flux_arm, base):
                    curve_dev = dev.get((config, seed), {})
                    if mode == "0.5":
                        point = 0.5
                    elif curve_dev:
                        point = max(curve_dev, key=lambda t: curve_dev[t][0])
                    else:
                        continue
                    cell = test.get((config, seed), {}).get(point)
                    if cell is None:
                        break
                    got.append(cell[1])
                if len(got) == 2:
                    pairs_.append((got[0], got[1]))
                    per_seed.append(got[0] - got[1])
            if len(per_seed) < 3:
                cells.append(f"{'--':>22}")
                continue
            decided = calibration.decide(pairs_, per_seed)
            cells.append(f"{decided['mean']:+.1%} t{decided['t']:5.2f} "
                         f"{'HOLDS' if decided['holds'] else 'fails':<5}"
                         f" n={len(per_seed)}")
        print(f"    {flux_arm + ' - ' + base:34}" + "".join(
            f"{c:>22}" for c in cells))

    print("\n--- predictions 1 and 3: did the head learn where the mask is "
          "blind? ---")
    print("    Mean |displacement error| inside the band, 5 DEV images, split")
    print("    by whether the arm's own mask predicts that centreline pixel.")
    print("    The reference is the flux you get FREE by skeletonising the")
    print("    mask -- beating it is the claim.\n")
    items = drive.load_split("dev")
    data = train.stack_split("fit")
    epochs = heldout.chosen_epochs()
    print(f"    {'arm':20}{'seeds':>7}{'finds: head/free':>20}"
          f"{'MISSES: head/free':>21}{'wins':>7}")
    for flux_arm, _ in PAIRS:
        runs = sorted(r for r in epochs if r.rsplit("_s", 1)[0] == flux_arm
                      and (heldout.ROOT / r / "final.pt").exists())
        rows, wins = [], 0
        for run in runs:
            weights = heldout.ROOT / run / f"epoch{epochs[run]:03d}.pt"
            if not weights.exists():
                continue
            model = train.build_model(flux_arm)
            model.load_state_dict(train.load_checkpoint(weights)["model"])
            model.eval()
            mean, std = train.normalisation(run, data)
            acc = {"find": [0.0, 0.0, 0.0], "miss": [0.0, 0.0, 0.0]}
            for item in items:
                prob, pred_y, pred_x = train.predict_flux_full(
                    model, item["image"], mean, std)
                skel = skeletonize(item["label"] & item["fov"])
                (true_y, true_x), band = flux.target(skel, train.FLUX_RADIUS)
                own = skeletonize((prob >= 0.5) & item["fov"])
                (free_y, free_x), _ = flux.target(own, train.FLUX_RADIUS)
                head = np.hypot(pred_y - true_y, pred_x - true_x)
                free = np.hypot(free_y - true_y, free_x - true_x)
                rows_, cols_ = np.indices(skel.shape)
                land_y = np.clip(np.rint(rows_ + true_y), 0,
                                 skel.shape[0] - 1).astype(int)
                land_x = np.clip(np.rint(cols_ + true_x), 0,
                                 skel.shape[1] - 1).astype(int)
                seen = (prob >= 0.5)[land_y, land_x]
                for tag, mask in (("find", band & seen),
                                  ("miss", band & ~seen)):
                    acc[tag][0] += float(mask.sum())
                    acc[tag][1] += float(head[mask].sum())
                    acc[tag][2] += float(free[mask].sum())
            rows.append({tag: (acc[tag][1] / acc[tag][0],
                               acc[tag][2] / acc[tag][0])
                         for tag in ("find", "miss") if acc[tag][0] > 0})
            if "miss" in rows[-1] and rows[-1]["miss"][0] < rows[-1]["miss"][1]:
                wins += 1
        if not rows:
            continue
        find = np.mean([[r["find"][0], r["find"][1]] for r in rows
                        if "find" in r], axis=0)
        miss = np.mean([[r["miss"][0], r["miss"][1]] for r in rows
                        if "miss" in r], axis=0)
        print(f"    {flux_arm:20}{len(rows):>7}"
              f"{f'{find[0]:.2f} / {find[1]:.2f}':>20}"
              f"{f'{miss[0]:.2f} / {miss[1]:.2f}':>21}"
              f"{f'{wins}/{len(rows)}':>7}")
    print("\n`wins` counts the seeds where the head beats the free reference")
    print("on the centreline the mask misses -- prediction 1 asked for 10/12.")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        raise SystemExit("pass --report")
