"""E2: does the ranking of the four losses survive contrast stratification?

stage 0 reported one Dice, one clDice and one betti-0 per model, averaged over
whole images. Section 2.5 then showed that local contrast, not vessel width,
decides which centreline points a model finds. If that is true, a single
whole-image number is an average over four very different regimes, and two
methods that differ only in the dimmest quarter would look nearly identical.

This re-scores every existing DRIVE model inside four contrast territories.
A territory is defined by nearest vessel: every pixel inherits the contrast
quartile of the closest ground-truth vessel pixel, so the four regions tile the
whole field of view and a false positive is charged to the vessel it sits next
to. That keeps Dice a real Dice (it still sees false positives) instead of
collapsing into recall, which is what happens if you mask to vessel pixels only.

betti-0 cannot be split this way -- a component count is a property of the whole
image. The topology quantity that does localise is the break: a maximal run of
consecutive ground-truth centreline pixels the model missed. Every break has a
location, so it can be charged to a contrast band, and the number of breaks is
what betti-0 was trying to measure in the first place.

Writes results/stratify.csv (per run x image x band x filter state).
~6 min on 6 cores.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breaks
import speckle
import train

RESULTS = Path(__file__).resolve().parent / "results"
MIN_SIZES = (0, 20)
BANDS = ("Q1_dimmest", "Q2", "Q3", "Q4_clearest")
CONN8 = np.ones((3, 3), dtype=bool)


def band_map(gt: np.ndarray, contrast: np.ndarray,
             edges: np.ndarray) -> np.ndarray:
    """Label every pixel with the contrast band of its nearest vessel pixel."""
    _, indices = ndimage.distance_transform_edt(~gt, return_indices=True)
    return np.digitize(contrast[tuple(indices)], edges)


def band_scores(pred: np.ndarray, gt: np.ndarray, skel_gt: np.ndarray,
                inside: np.ndarray) -> dict:
    """Dice and clDice restricted to one territory.

    Correctness is judged against the full ground truth; only the locality of
    the pixels being counted is restricted, so a vessel that leaves the band is
    not penalised for leaving.
    """
    predicted, truth = pred & inside, gt & inside
    denominator = predicted.sum() + truth.sum()
    dice = 2.0 * (predicted & truth).sum() / denominator if denominator else np.nan

    skel_pred = skeletonize(pred) & inside
    skel_truth = skel_gt & inside
    precision = ((skel_pred & gt).sum() / skel_pred.sum()
                 if skel_pred.any() else np.nan)
    sensitivity = ((skel_truth & pred).sum() / skel_truth.sum()
                   if skel_truth.any() else np.nan)
    cldice = (2 * precision * sensitivity / (precision + sensitivity)
              if precision + sensitivity > 0 else np.nan)
    return {"dice": dice, "cldice": cldice,
            "tprec": precision, "tsens": sensitivity}


def break_counts(pred: np.ndarray, skel_gt: np.ndarray,
                 band: np.ndarray) -> np.ndarray:
    """Breaks per band: maximal runs of missed centreline, charged by majority.

    A run that straddles two bands is charged once, to whichever band holds
    most of it, so the totals stay comparable to a plain break count.
    """
    missed = skel_gt & ~pred
    labels, count = ndimage.label(missed, structure=CONN8)
    tally = np.zeros(len(BANDS), dtype=int)
    if count == 0:
        return tally
    for run_band in ndimage.labeled_comprehension(
            band, labels, np.arange(1, count + 1),
            lambda values: np.bincount(values, minlength=len(BANDS)).argmax(),
            int, 0):
        tally[run_band] += 1
    return tally


def main() -> None:
    train_data, val = train.stack_split("train"), train.stack_split("val")
    normalise = train_data["images"][train_data["fovs"]]
    mean, std = float(normalise.mean()), float(normalise.std())

    # Per-image geometry: computed once, reused by all twelve models.
    geometry = []
    vessel_contrast = []
    for index in range(len(val["names"])):
        fov = val["fovs"][index]
        gt = (val["labels"][index] > 0.5) & fov
        contrast = breaks.local_contrast(val["images"][index])
        geometry.append({"fov": fov, "gt": gt, "contrast": contrast,
                         "skel_gt": skeletonize(gt)})
        vessel_contrast.append(contrast[gt])

    # Quartile edges come from the ground-truth vessel pixels of the whole
    # validation set, so every model and every image is binned identically.
    edges = np.percentile(np.concatenate(vessel_contrast), [25, 50, 75])
    print(f"contrast quartile edges: {[round(float(e), 4) for e in edges]}",
          flush=True)
    for item in geometry:
        item["band"] = band_map(item["gt"], item["contrast"], edges)

    # Whatever has a finished checkpoint, at whatever seed. A config added to
    # CONFIGS ahead of its runs costs nothing, and a seed trained after this
    # script was written is picked up rather than silently dropped.
    runs = sys.argv[1:] or train.trained_runs()
    print(f"scoring {len(runs)} runs", flush=True)
    rows = []
    for run_name in runs:
        model = speckle.load_model(run_name)
        # Per run: a LIOT run needs its own constants, and predict_full
        # raises rather than silently scoring if these disagree.
        mean, std = train.normalisation(run_name, train_data)
        for index, image_name in enumerate(val["names"]):
            item = geometry[index]
            prob = train.predict_full(model, val["images"][index], mean, std)
            raw = (prob >= 0.5) & item["fov"]
            for min_size in MIN_SIZES:
                pred = speckle.drop_small(raw, min_size)
                tally = break_counts(pred, item["skel_gt"], item["band"])
                for code, name in enumerate(BANDS):
                    inside = (item["band"] == code) & item["fov"]
                    scores = band_scores(pred, item["gt"], item["skel_gt"],
                                         inside)
                    rows.append({
                        "run": run_name,
                        "config": run_name.rsplit("_s", 1)[0],
                        "image": image_name, "min_size": min_size,
                        "band": name, "breaks": int(tally[code]),
                        "gt_px": int((item["gt"] & inside).sum()),
                        "skel_px": int((item["skel_gt"] & inside).sum()),
                        **{k: round(float(v), 5) for k, v in scores.items()},
                    })
        print(f"{run_name} done", flush=True)

    with (RESULTS / "stratify.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {RESULTS / 'stratify.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
