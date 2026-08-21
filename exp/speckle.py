"""Is the betti-0 win real topology, or just speckle suppression?

betti-0 error counts every surplus component, and two very different things
land in that count:

  on-vessel   a component that overlaps the ground-truth vessel tree -- a real
              fragment, i.e. a vessel the model genuinely broke apart
  off-vessel  a component with zero ground-truth overlap -- a false-positive
              blob in the background, nothing to do with connectivity

If the surplus is mostly off-vessel specks, then one call to
remove_small_objects should erase the gap between the losses, and the headline
claim of the report is about denoising rather than topology.

Writes results/speckle.csv. ~4 min on 6 cores.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive
import metrics
import train

RESULTS = Path(__file__).resolve().parent / "results"
MIN_SIZES = (0, 2, 5, 10, 20, 50, 100)
CONN8 = np.ones((3, 3), dtype=bool)


def load_model(run_name: str) -> torch.nn.Module:
    config_name = run_name.rsplit("_s", 1)[0]
    model = train.build_model(config_name)
    state = torch.load(RESULTS / run_name / "final.pt", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def decompose(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Split the predicted components into on-vessel and off-vessel."""
    labels, count = ndimage.label(pred, structure=CONN8)
    if count == 0:
        return {"n": 0, "on": 0, "off": 0, "off_px": 0, "median_off": 0.0}
    index = np.arange(1, count + 1)
    sizes = np.array(ndimage.sum(pred, labels, index))
    hits = np.array(ndimage.sum(gt, labels, index))
    off = hits == 0
    return {
        "n": count,
        "on": int((~off).sum()),
        "off": int(off.sum()),
        "off_px": float(sizes[off].sum()),
        "median_off": float(np.median(sizes[off])) if off.any() else 0.0,
    }


def drop_small(pred: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return pred
    labels, count = ndimage.label(pred, structure=CONN8)
    if count == 0:
        return pred
    sizes = np.array(ndimage.sum(pred, labels, np.arange(1, count + 1)))
    keep = np.concatenate([[False], sizes >= min_size])
    return keep[labels]


def main() -> None:
    train_data, val = train.stack_split("train"), train.stack_split("val")
    inside = train_data["images"][train_data["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())

    runs = [f"{name}_s{seed}" for seed in (0, 1, 2) for name in train.CONFIGS
            if (RESULTS / f"{name}_s{seed}" / "final.pt").exists()]
    rows = []
    for run_name in runs:
        model = load_model(run_name)
        for index, image_name in enumerate(val["names"]):
            gt = val["labels"][index] > 0.5
            fov = val["fovs"][index]
            prob = train.predict_full(model, val["images"][index], mean, std)
            raw = (prob >= 0.5) & fov
            parts = decompose(raw, gt)
            for min_size in MIN_SIZES:
                pred = drop_small(raw, min_size)
                b0_pred = ndimage.label(pred, structure=CONN8)[1]
                b0_gt = ndimage.label(gt & fov, structure=CONN8)[1]
                rows.append({
                    "run": run_name, "config": run_name.rsplit("_s", 1)[0],
                    "image": image_name, "min_size": min_size,
                    "dice": metrics.dice(pred, gt & fov),
                    "cldice": metrics.cl_dice(pred, gt & fov),
                    "b0_pred": b0_pred, "b0_gt": b0_gt,
                    "betti0_err": abs(b0_pred - b0_gt),
                    **parts,
                })
        print(f"{run_name} done", flush=True)

    with (RESULTS / "speckle.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    configs = list(train.CONFIGS)

    def mean_of(config: str, min_size: int, key: str) -> float:
        picked = [r[key] for r in rows
                  if r["config"] == config and r["min_size"] == min_size]
        return float(np.mean(picked))

    print("\n=== what the surplus components actually are (unfiltered) ===")
    print(f"{'config':14} {'total':>7} {'on-vessel':>10} {'off-vessel':>11} "
          f"{'median off-size':>16}")
    for config in configs:
        print(f"{config:14} {mean_of(config, 0, 'n'):7.1f} "
              f"{mean_of(config, 0, 'on'):10.1f} {mean_of(config, 0, 'off'):11.1f} "
              f"{mean_of(config, 0, 'median_off'):16.1f}")
    print(f"{'ground truth':14} {mean_of(configs[0], 0, 'b0_gt'):7.1f}")

    print("\n=== betti-0 error after dropping components below min_size ===")
    header = "".join(f"{f'>={s}px':>9}" for s in MIN_SIZES)
    print(f"{'config':14}{header}")
    for config in configs:
        cells = "".join(f"{mean_of(config, s, 'betti0_err'):9.1f}" for s in MIN_SIZES)
        print(f"{config:14}{cells}")

    print("\n=== the cost: Dice after the same filtering ===")
    print(f"{'config':14}{header}")
    for config in configs:
        cells = "".join(f"{mean_of(config, s, 'dice'):9.4f}" for s in MIN_SIZES)
        print(f"{config:14}{cells}")

    print("\n=== does B still beat A once specks are gone? ===")
    for min_size in MIN_SIZES:
        gap = mean_of("A_dice", min_size, "betti0_err") - mean_of(
            "B_cldice", min_size, "betti0_err")
        base = mean_of("A_dice", min_size, "betti0_err")
        print(f"  min_size >={min_size:3d}px   A-B = {gap:6.1f} "
              f"({100 * gap / base:5.1f}% of A)")


if __name__ == "__main__":
    main()
