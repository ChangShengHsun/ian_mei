"""Zero-training baselines: Hessian ridge filters (Frangi / Sato / Meijering).

These date from 1998 and need no GPU, no data and no training -- they read the
local second derivatives and ask "is this pixel shaped like a tube?". Meijering
was designed for neurites specifically. They are the honest floor every learned
method should be reported against, and they are the cheapest way to show the
Dice-versus-topology gap: a filter can trail U-Net on Dice while breaking the
vessel tree less often.

Writes results/classical_{train,val}.csv. ~3 min on 6 cores.
"""
import csv
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from skimage.filters import frangi, meijering, sato, threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive
import metrics

RESULTS = Path(__file__).resolve().parent / "results"
FILTERS = {"frangi": frangi, "sato": sato, "meijering": meijering}
SIGMA_SETS = {"s1-3": (1, 2, 3), "s1-5": (1, 2, 3, 4, 5), "s1-7": (1, 3, 5, 7)}
# Threshold by target vessel fraction rather than by response value: the three
# filters have wildly different output scales, so a shared numeric cutoff would
# compare nothing. Ground truth sits at ~12% of the FOV.
FRACTIONS = (0.08, 0.10, 0.12, 0.14)


def _threshold_at_fraction(response: np.ndarray, fov: np.ndarray,
                           fraction: float) -> np.ndarray:
    cutoff = np.percentile(response[fov], 100 * (1 - fraction))
    return (response >= cutoff) & fov


def score_image(item: dict) -> list[dict]:
    """Every (filter, sigmas, fraction) config on one image, plus Otsu."""
    image, gt, fov = item["image"], item["label"], item["fov"]
    rows = []

    for (filter_name, filter_fn), (sigma_name, sigmas) in product(
            FILTERS.items(), SIGMA_SETS.items()):
        # black_ridges=True: after CLAHE on the green channel vessels are the
        # dark structures, not the bright ones.
        response = filter_fn(image, sigmas=sigmas, black_ridges=True)
        for fraction in FRACTIONS:
            pred = _threshold_at_fraction(response, fov, fraction)
            rows.append({
                "method": f"{filter_name}_{sigma_name}_f{fraction}",
                "image": item["name"],
                **metrics.evaluate(pred.astype(float), gt, fov, 0.5),
            })

    otsu = (image < threshold_otsu(image[fov])) & fov
    rows.append({"method": "otsu", "image": item["name"],
                 **metrics.evaluate(otsu.astype(float), gt, fov, 0.5)})
    return rows


def run(split: str) -> dict[str, dict]:
    items = drive.load_split(split)
    with Pool(6) as pool:
        rows = [row for batch in pool.map(score_image, items) for row in batch]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"classical_{split}.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        picked = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.nanmean([row[key] for row in picked]))
            for key in ("dice", "cldice", "betti0_err", "betti1_err", "hd95")
        }
    print(f"wrote {out} ({len(rows)} rows)")
    return summary


if __name__ == "__main__":
    train = run("train")
    best_dice = max(train, key=lambda m: train[m]["dice"])
    best_cldice = max(train, key=lambda m: train[m]["cldice"])
    print(f"\nbest on train by dice:   {best_dice}  {train[best_dice]}")
    print(f"best on train by cldice: {best_cldice}  {train[best_cldice]}")

    val = run("val")
    print("\n--- val (20 images) ---")
    print(f"{'method':28} {'dice':>6} {'clDice':>7} {'b0err':>8} "
          f"{'b1err':>8} {'95HD':>6}")
    for method in sorted(val, key=lambda m: -val[m]["dice"]):
        scores = val[method]
        print(f"{method:28} {scores['dice']:6.3f} {scores['cldice']:7.3f} "
              f"{scores['betti0_err']:8.1f} {scores['betti1_err']:8.1f} "
              f"{scores['hd95']:6.2f}")
