"""E3' + E1': what the annotator costs in topology, and where the model hesitates.

Two questions that need the same expensive thing (a probability map from every
STARE model), so they share one pass.

E3'  When the training target changes from one annotator to the other, what
     happens to TOPOLOGY, stratified by contrast? E0 established that the
     multi-annotator literature reports Dice, RMSE and GED on blob-shaped
     targets and never a topology metric, so a break count as a function of
     whose labels you trained on has not been reported.

E1'  Does the model hesitate where the two humans disagree? The topology-aware
     uncertainty paper (NeurIPS 2023) validates its uncertainty only against a
     single ground truth: its simulated proofreader answers yes/no from the GT
     mask. STARE has two annotators, so the honest version of that check is
     available here and was not available to them.

Writes results/stare_stratify.csv (E3') and results/stare_hesitation.csv (E1').
~10 min on 6 cores once the 12 models exist.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breaks
import speckle
import stare_agreement
import stratify
import train
import train_stare

RESULTS = Path(__file__).resolve().parent / "results"
MODELS = RESULTS / "stare_cross"
ANNOTATORS = ("ah", "vk")
MIN_SIZE = 20


def load(run_name: str) -> tuple[torch.nn.Module, float, float]:
    state = torch.load(MODELS / run_name / "final.pt", weights_only=False)
    model = train.TinyUNet()
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["mean"], state["std"]


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC: P(a contested pixel scores above an agreed one).

    0.5 means the score carries no information about disagreement; 1.0 means
    every contested pixel is more hesitant than every agreed one.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties, otherwise a probability map with many exact
    # duplicates (there are many) reports a biased AUROC.
    unique, inverse, counts = np.unique(scores, return_inverse=True,
                                        return_counts=True)
    summed = np.zeros(len(unique))
    np.add.at(summed, inverse, ranks)
    ranks = (summed / counts)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def main() -> None:
    items = stare_agreement.load_stare()
    runs = [f"{t}_f{f}_s{s}" for s in train_stare.SEEDS
            for f in train_stare.FOLDS for t in train_stare.TRAIN_TARGETS]
    missing = [r for r in runs if not (MODELS / r / "final.pt").exists()]
    if missing:
        raise SystemExit(f"missing weights for {len(missing)} runs: {missing}")

    # Contrast bands come from the union of both annotators, so the binning is
    # identical no matter which annotator a model is being scored against.
    geometry = []
    for item in items:
        union = (item["ah"] | item["vk"]) & item["fov"]
        contrast = breaks.local_contrast(item["image"])
        geometry.append({"union": union, "contrast": contrast,
                         "disagree": (item["ah"] ^ item["vk"]) & item["fov"]})
    edges = np.percentile(
        np.concatenate([g["contrast"][g["union"]] for g in geometry]),
        [25, 50, 75])
    print(f"contrast quartile edges: {[round(float(e), 4) for e in edges]}",
          flush=True)
    for item, geo in zip(items, geometry):
        geo["band"] = stratify.band_map(geo["union"], geo["contrast"], edges)

    topology_rows, hesitation_rows = [], []
    for run_name in runs:
        target, fold_tag, seed_tag = run_name.split("_")
        _, test_slice = train_stare.FOLDS[int(fold_tag[1:])]
        model, mean, std = load(run_name)

        for index in range(len(items))[test_slice]:
            item, geo = items[index], geometry[index]
            prob = train.predict_full(model, item["image"], mean, std)
            pred = speckle.drop_small((prob >= 0.5) & item["fov"], MIN_SIZE)

            # ---- E3': topology and Dice per band, against each annotator
            for annotator in ANNOTATORS:
                truth = item[annotator] & item["fov"]
                skel_truth = skeletonize(truth)
                tally = stratify.break_counts(pred, skel_truth, geo["band"])
                for code, band in enumerate(stratify.BANDS):
                    inside = (geo["band"] == code) & item["fov"]
                    scores = stratify.band_scores(pred, truth, skel_truth,
                                                  inside)
                    topology_rows.append({
                        "run": run_name, "trained_on": target,
                        "scored_against": annotator, "image": item["name"],
                        "band": band, "breaks": int(tally[code]),
                        "gt_px": int((truth & inside).sum()),
                        **{k: round(float(v), 5) for k, v in scores.items()},
                    })

            # ---- E1': is the model's hesitation where the humans argue?
            # Hesitation peaks at p = 0.5 and falls to 0 at either certainty.
            hesitation = 1.0 - 2.0 * np.abs(prob - 0.5)
            for code, band in enumerate(stratify.BANDS):
                population = geo["union"] & (geo["band"] == code)
                if not population.any():
                    continue
                contested = geo["disagree"][population]
                hesitation_rows.append({
                    "run": run_name, "trained_on": target,
                    "image": item["name"], "band": band,
                    "n_px": int(population.sum()),
                    "contested_frac": round(float(contested.mean()), 5),
                    "hesitation_contested": round(
                        float(hesitation[population][contested].mean()), 5)
                    if contested.any() else float("nan"),
                    "hesitation_agreed": round(
                        float(hesitation[population][~contested].mean()), 5)
                    if (~contested).any() else float("nan"),
                    "auroc": round(auroc(hesitation[population], contested), 5),
                })
        print(f"{run_name} done", flush=True)

    for rows, name in ((topology_rows, "stare_stratify.csv"),
                       (hesitation_rows, "stare_hesitation.csv")):
        with (RESULTS / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {RESULTS / name}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
