"""Where do the genuine breaks happen -- and is the information even there?

speckle.py showed that after dropping components under 20 px, every loss still
leaves ~22 components against a ground truth of 3. Those survivors are real
fragments of real vessels. This asks what distinguishes the centreline the
model found from the centreline it lost, along two axes:

  radius    half-width of the vessel at that point, from the ground-truth
            distance transform. Thin vessels are the ones everyone blames.
  contrast  how much darker the vessel is than its immediate surroundings,
            via a black top-hat. This is the "is the evidence present at all"
            axis -- a vessel the camera barely recorded cannot be segmented by
            any model, however good.

Crossing the two separates the two failure modes the survey's P3 says nobody
distinguishes: low contrast = information-limited (no loss can fix it), high
contrast + high radius but still missed = model-limited (that one is on us).

Writes results/breaks.csv and results/breaks_bins.csv. ~3 min on 6 cores.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive
import speckle
import train

RESULTS = Path(__file__).resolve().parent / "results"
MIN_SIZE = 20          # the operating point speckle.py settled on
BG_WINDOW = 15         # px; must exceed the widest vessel to close over it
RADIUS_EDGES = (0.0, 1.5, 2.5, 3.5, np.inf)
RADIUS_NAMES = ("<=1.5", "1.5-2.5", "2.5-3.5", ">3.5")
CONTRAST_NAMES = ("Q1 最暗淡", "Q2", "Q3", "Q4 最清楚")
CONN8 = np.ones((3, 3), dtype=bool)


def local_contrast(image: np.ndarray) -> np.ndarray:
    """Black top-hat: how much darker each pixel is than its background.

    Grey closing dilates then erodes, which fills in dark thin structures, so
    the closed image is the retina without its vessels. The difference is the
    vessel's own contrast, free of the slow illumination gradient.
    """
    background = ndimage.grey_closing(image, size=BG_WINDOW)
    return background - image


def analyse(run_name: str, val: dict, mean: float, std: float) -> list[dict]:
    model = speckle.load_model(run_name)
    rows = []
    for index, image_name in enumerate(val["names"]):
        gt = (val["labels"][index] > 0.5) & val["fovs"][index]
        prob = train.predict_full(model, val["images"][index], mean, std)
        pred = speckle.drop_small((prob >= 0.5) & val["fovs"][index], MIN_SIZE)

        skeleton = skeletonize(gt)
        radius = ndimage.distance_transform_edt(gt)
        contrast = local_contrast(val["images"][index])

        points = np.argwhere(skeleton)
        found = pred[skeleton]
        rows.append({
            "run": run_name, "config": run_name.rsplit("_s", 1)[0],
            "image": image_name,
            "radius": radius[skeleton], "contrast": contrast[skeleton],
            "found": found, "n_points": len(points),
        })
    return rows


def main() -> None:
    train_data, val = train.stack_split("train"), train.stack_split("val")
    inside = train_data["images"][train_data["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())

    runs = train.trained_runs()
    collected = []
    for run_name in runs:
        # Per run, because a LIOT run needs its own constants; see
        # train.normalisation. predict_full raises if they disagree.
        mean, std = train.normalisation(run_name, train_data)
        collected += analyse(run_name, val, mean, std)
        print(f"{run_name} done", flush=True)

    # Contrast quartiles come from the ground truth alone, so every config is
    # binned on the same edges and the numbers stay comparable.
    all_contrast = np.concatenate(
        [r["contrast"] for r in collected if r["run"] == runs[0]])
    contrast_edges = np.concatenate(
        [[-np.inf], np.percentile(all_contrast, [25, 50, 75]), [np.inf]])
    print(f"\ncontrast quartile edges: "
          f"{[round(float(e), 4) for e in contrast_edges[1:-1]]}")

    per_config = {}
    for config in train.CONFIGS:
        picked = [r for r in collected if r["config"] == config]
        radius = np.concatenate([r["radius"] for r in picked])
        contrast = np.concatenate([r["contrast"] for r in picked])
        found = np.concatenate([r["found"] for r in picked])
        radius_bin = np.digitize(radius, RADIUS_EDGES[1:-1])
        contrast_bin = np.digitize(contrast, contrast_edges[1:-1])
        per_config[config] = (radius_bin, contrast_bin, found)

    def rate(config, rmask=None, cmask=None) -> tuple[float, int]:
        radius_bin, contrast_bin, found = per_config[config]
        keep = np.ones(len(found), dtype=bool)
        if rmask is not None:
            keep &= radius_bin == rmask
        if cmask is not None:
            keep &= contrast_bin == cmask
        return (float(found[keep].mean()) if keep.any() else float("nan"),
                int(keep.sum()))

    print("\n=== 整體中心線偵測率（過濾 <20px 之後）===")
    for config in train.CONFIGS:
        print(f"  {config:14} {rate(config)[0]:.4f}")

    print("\n=== 依血管半徑（A_dice）===")
    print(f"{'半徑 px':>10} {'偵測率':>9} {'佔全部':>9}")
    total = rate("A_dice")[1]
    for i, name in enumerate(RADIUS_NAMES):
        value, count = rate("A_dice", rmask=i)
        print(f"{name:>10} {value:9.4f} {100 * count / total:8.1f}%")

    print("\n=== 依局部對比（A_dice）===")
    print(f"{'對比分位':>12} {'偵測率':>9} {'佔全部':>9}")
    for i, name in enumerate(CONTRAST_NAMES):
        value, count = rate("A_dice", cmask=i)
        print(f"{name:>12} {value:9.4f} {100 * count / total:8.1f}%")

    print("\n=== 交叉表：半徑 x 對比 的偵測率（A_dice）===")
    print(f"{'':>10}" + "".join(f"{n:>12}" for n in CONTRAST_NAMES))
    for i, rname in enumerate(RADIUS_NAMES):
        cells = ""
        for j in range(4):
            value, count = rate("A_dice", rmask=i, cmask=j)
            cells += f"{value:9.3f}({count // 1000:2d}k)" if count >= 1000 \
                else f"{value:9.3f}( —)"
        print(f"{rname:>10}{cells}")

    print("\n=== B 相對 A 的改善，依格子 ===")
    print(f"{'':>10}" + "".join(f"{n:>12}" for n in CONTRAST_NAMES))
    for i, rname in enumerate(RADIUS_NAMES):
        cells = ""
        for j in range(4):
            a = rate("A_dice", rmask=i, cmask=j)[0]
            b = rate("B_cldice", rmask=i, cmask=j)[0]
            cells += f"{100 * (b - a):+11.2f}%"
        print(f"{rname:>10}{cells}")

    with (RESULTS / "breaks_bins.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "radius_bin", "contrast_bin",
                         "detection_rate", "n_points"])
        for config in train.CONFIGS:
            for i, rname in enumerate(RADIUS_NAMES):
                for j, cname in enumerate(CONTRAST_NAMES):
                    value, count = rate(config, rmask=i, cmask=j)
                    writer.writerow([config, rname, cname,
                                     round(value, 5), count])
    print(f"\nwrote {RESULTS / 'breaks_bins.csv'}")


if __name__ == "__main__":
    main()
