"""E6: does structure-level uncertainty survive where pixel-level failed?

E1' showed that per-pixel hesitation (closeness of the sigmoid to 0.5) predicts
human disagreement with AUROC 0.885 in the brightest contrast quartile, which
holds 4% of all disagreement, and 0.378 in the dimmest, which holds 46%. That
is a failure of ONE definition of uncertainty. The NeurIPS 2023 topology-aware
work uses a different one: discrete Morse theory extracts one-dimensional
structures from the likelihood map and uncertainty is estimated per structure,
not per pixel. If aggregating over a whole structure fixes the dim band, the
E1' result is about a weak baseline. If it does not, the failure belongs to the
route rather than to the definition.

This is a cheap stand-in for discrete Morse theory, not a reimplementation of
it. The structures a Morse complex extracts from a likelihood map are the
one-cells of its skeleton, so the approximation is: threshold permissively so
that doubtful structures still exist as candidates, skeletonise, cut at
junctions, and treat each surviving branch as one structure with one
uncertainty. What is genuinely shared with the paper is the move being tested
-- decide about a whole connected piece of curve at once instead of pixel by
pixel -- and that is the move E6 is asking about.

Writes results/structure_uncertainty.csv. ~5 min on the six STARE models.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breaks
import stare_agreement
import stare_stratify
import stratify
import train
import train_stare

RESULTS = Path(__file__).resolve().parent / "results"
CANDIDATE_THRESHOLD = 0.2   # permissive: a doubtful branch must still exist
CONN8 = np.ones((3, 3), dtype=bool)
SUPPORTED = 0.5             # a branch counts as supported if half of it is inside


def branches(skeleton: np.ndarray) -> tuple[np.ndarray, int]:
    """Cut the skeleton at junctions; each remaining piece is one structure."""
    neighbours = ndimage.convolve(skeleton.astype(np.uint8), CONN8.astype(np.uint8),
                                  mode="constant") - skeleton
    return ndimage.label(skeleton & (neighbours < 3), structure=CONN8)


def structure_rows(prob: np.ndarray, item: dict, geo: dict,
                   run_name: str, target: str) -> list[dict]:
    candidate = (prob >= CANDIDATE_THRESHOLD) & item["fov"]
    skeleton = skeletonize(candidate)
    labels, count = branches(skeleton)
    if count == 0:
        return []
    index = np.arange(1, count + 1)

    sizes = np.array(ndimage.sum(skeleton, labels, index))
    mean_prob = np.array(ndimage.mean(prob, labels, index))
    inside_ah = np.array(ndimage.mean(item["ah"].astype(float), labels, index))
    inside_vk = np.array(ndimage.mean(item["vk"].astype(float), labels, index))
    band = np.array(ndimage.labeled_comprehension(
        geo["band"], labels, index,
        lambda values: np.bincount(values, minlength=len(stratify.BANDS)).argmax(),
        int, 0))

    supported_ah, supported_vk = inside_ah >= SUPPORTED, inside_vk >= SUPPORTED
    population = supported_ah | supported_vk
    contested = supported_ah ^ supported_vk
    # Same definition as E1', one level up: a structure is contested when
    # exactly one annotator backs it, and only backed structures are judged.
    hesitation = 1.0 - 2.0 * np.abs(mean_prob - 0.5)

    rows = []
    for code, name in enumerate(stratify.BANDS):
        keep = population & (band == code)
        if keep.sum() < 5:
            continue
        disputed = contested[keep]
        rows.append({
            "run": run_name, "trained_on": target, "image": item["name"],
            "band": name, "n_structures": int(keep.sum()),
            "median_length": round(float(np.median(sizes[keep])), 1),
            "contested_frac": round(float(disputed.mean()), 5),
            "auroc": round(stare_stratify.auroc(hesitation[keep], disputed), 5),
            "hesitation_contested": round(
                float(hesitation[keep][disputed].mean()), 5)
            if disputed.any() else float("nan"),
            "hesitation_agreed": round(
                float(hesitation[keep][~disputed].mean()), 5)
            if (~disputed).any() else float("nan"),
            "prob_contested": round(float(mean_prob[keep][disputed].mean()), 5)
            if disputed.any() else float("nan"),
            "prob_agreed": round(float(mean_prob[keep][~disputed].mean()), 5)
            if (~disputed).any() else float("nan"),
        })
    return rows


def main() -> None:
    items = stare_agreement.load_stare()
    runs = sys.argv[1:] or [f"{t}_f{f}_s{s}" for s in train_stare.SEEDS
                            for f in train_stare.FOLDS
                            for t in train_stare.TRAIN_TARGETS]
    runs = [r for r in runs
            if (stare_stratify.MODELS / r / "final.pt").exists()]
    print(f"scoring {len(runs)} runs: {', '.join(runs)}", flush=True)

    geometry = []
    for item in items:
        union = (item["ah"] | item["vk"]) & item["fov"]
        contrast = breaks.local_contrast(item["image"])
        geometry.append({"union": union, "contrast": contrast})
    edges = np.percentile(
        np.concatenate([g["contrast"][g["union"]] for g in geometry]),
        [25, 50, 75])
    for geo in geometry:
        geo["band"] = stratify.band_map(geo["union"], geo["contrast"], edges)

    rows = []
    for run_name in runs:
        target, fold_tag, _ = run_name.split("_")
        _, test_slice = train_stare.FOLDS[int(fold_tag[1:])]
        model, mean, std = stare_stratify.load(run_name)
        for index in range(len(items))[test_slice]:
            prob = train.predict_full(model, items[index]["image"], mean, std)
            rows += structure_rows(prob, items[index], geometry[index],
                                   run_name, target)
        print(f"{run_name} done", flush=True)

    with (RESULTS / "structure_uncertainty.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULTS / 'structure_uncertainty.csv'} ({len(rows)} rows)")

    print("\n=== E6 結構層級 AUROC，對照 E1' 的逐像素結果 ===")
    print(f"{'band':>14}{'結構數':>9}{'中位長度':>10}{'爭議比例':>10}"
          f"{'AUROC 中位':>12}{'<0.5':>10}")
    for band in stratify.BANDS:
        picked = [r for r in rows if r["band"] == band]
        scores = np.array([r["auroc"] for r in picked], dtype=float)
        scores = scores[~np.isnan(scores)]
        print(f"{band:>14}{np.mean([r['n_structures'] for r in picked]):9.1f}"
              f"{np.median([r['median_length'] for r in picked]):10.1f}"
              f"{100 * np.mean([r['contested_frac'] for r in picked]):9.1f}%"
              f"{np.median(scores):12.3f}"
              f"{int((scores < 0.5).sum()):6d}/{len(scores):<4d}")


if __name__ == "__main__":
    main()
