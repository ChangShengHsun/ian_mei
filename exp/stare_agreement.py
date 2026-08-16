"""Do two human experts disagree in the same places the model fails?

breaks.py showed the model's misses concentrate almost entirely in the dimmest
quartile of local contrast, and argued that is an evidence problem rather than
a loss problem. The strong form of that claim is testable: if the evidence
really is absent there, then two human experts should also stop agreeing there.

STARE ships two independent labellings of the same 20 images. Treating one
annotator as "truth" and the other as the "prediction" runs them through the
exact same measurement as a model, so the two curves are directly comparable
in shape.

Writes results/stare_agreement.csv + results/stare_bins.csv. ~1 min.
"""
import csv
import gzip
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breaks
import metrics

ROOT = Path(__file__).resolve().parent.parent / "data" / "STARE"
RESULTS = Path(__file__).resolve().parent / "results"
CONN8 = np.ones((3, 3), dtype=bool)


def load_ppm(path: Path) -> np.ndarray:
    with gzip.open(path) as handle:
        return np.asarray(Image.open(io.BytesIO(handle.read())))


def load_stare() -> list[dict]:
    items = []
    for image_path in sorted((ROOT / "images").glob("*.ppm.gz")):
        stem = image_path.name.split(".")[0]
        rgb = load_ppm(image_path)
        green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            rgb[..., 1])
        fov = ndimage.binary_dilation(
            ndimage.binary_fill_holes(rgb.sum(-1) > 30), CONN8, iterations=2)
        items.append({
            "name": stem,
            "image": green.astype(np.float32) / 255.0,
            "fov": fov,
            "ah": (load_ppm(ROOT / "labels-ah" / f"{stem}.ah.ppm.gz") > 127) & fov,
            "vk": (load_ppm(ROOT / "labels-vk" / f"{stem}.vk.ppm.gz") > 127) & fov,
        })
    return items


def main() -> None:
    items = load_stare()
    print(f"loaded {len(items)} STARE images, shape {items[0]['image'].shape}")
    print(f"fov fraction {np.mean([i['fov'].mean() for i in items]):.3f}, "
          f"ah vessels {np.mean([i['ah'][i['fov']].mean() for i in items]):.3f}, "
          f"vk vessels {np.mean([i['vk'][i['fov']].mean() for i in items]):.3f}")

    rows = []
    for item in items:
        scores = metrics.evaluate(item["vk"].astype(float), item["ah"],
                                  item["fov"])
        rows.append({"image": item["name"], **scores})
    with (RESULTS / "stare_agreement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== 兩位標註者互相比對（ah 當真值，vk 當「預測」）===")
    for key, label in [("dice", "Dice"), ("cldice", "clDice"),
                       ("betti0_err", "β₀ 誤差"), ("hd95", "95HD")]:
        values = [r[key] for r in rows]
        print(f"  {label:10} {np.nanmean(values):8.4f}   "
              f"(範圍 {np.nanmin(values):.3f} – {np.nanmax(values):.3f})")
    print(f"  {'ah 連通塊':10} {np.mean([r['b0_gt'] for r in rows]):8.1f}")
    print(f"  {'vk 連通塊':10} {np.mean([r['b0_pred'] for r in rows]):8.1f}")

    # Same stratification as breaks.py, with the second annotator standing in
    # for the model.
    radius, contrast, found = [], [], []
    for item in items:
        skeleton = skeletonize(item["ah"])
        radius.append(ndimage.distance_transform_edt(item["ah"])[skeleton])
        contrast.append(breaks.local_contrast(item["image"])[skeleton])
        found.append(item["vk"][skeleton])
    radius = np.concatenate(radius)
    contrast = np.concatenate(contrast)
    found = np.concatenate(found)

    edges = np.concatenate([[-np.inf], np.percentile(contrast, [25, 50, 75]),
                            [np.inf]])
    radius_bin = np.digitize(radius, breaks.RADIUS_EDGES[1:-1])
    contrast_bin = np.digitize(contrast, edges[1:-1])
    print(f"\n共 {len(found):,} 個 ah 中心線點，vk 涵蓋 {found.mean():.4f}")

    print("\n=== 標註者一致率：半徑 x 對比 ===")
    print(f"{'':>10}" + "".join(f"{n:>13}" for n in breaks.CONTRAST_NAMES))
    grid = {}
    for i, rname in enumerate(breaks.RADIUS_NAMES):
        cells = ""
        for j in range(4):
            keep = (radius_bin == i) & (contrast_bin == j)
            value = float(found[keep].mean()) if keep.any() else float("nan")
            grid[(i, j)] = (value, int(keep.sum()))
            cells += f"{value:9.3f}({keep.sum() // 1000:2d}k)"
        print(f"{rname:>10}{cells}")

    print("\n=== 依對比分位（邊際）===")
    print(f"{'對比':>12} {'一致率':>9} {'佔中心線':>10}")
    for j, name in enumerate(breaks.CONTRAST_NAMES):
        keep = contrast_bin == j
        print(f"{name:>12} {found[keep].mean():9.4f} "
              f"{100 * keep.mean():9.1f}%")

    with (RESULTS / "stare_bins.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius_bin", "contrast_bin", "agreement", "n_points"])
        for i, rname in enumerate(breaks.RADIUS_NAMES):
            for j, cname in enumerate(breaks.CONTRAST_NAMES):
                value, count = grid[(i, j)]
                writer.writerow([rname, cname, round(value, 5), count])
    print(f"\nwrote {RESULTS / 'stare_bins.csv'}")


if __name__ == "__main__":
    main()
