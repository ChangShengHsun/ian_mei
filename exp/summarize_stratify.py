"""Read stratify.csv and answer one question: does the ranking survive?

The four losses were separated by 0.004 Dice in the whole-image report. If that
gap is really the average of "identical in bright vessels, different in dim
ones", the per-band numbers will show it and the whole-image number was hiding
the only result worth having.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stratify
import train

RESULTS = Path(__file__).resolve().parent / "results"
SEEDS = (0, 1, 2)
CONFIGS: list[str] = []      # filled from the CSV, so an untrained config in
                             # train.CONFIGS does not turn every cell into nan


def load() -> list[dict]:
    rows = list(csv.DictReader((RESULTS / "stratify.csv").open()))
    for row in rows:
        for key in ("dice", "cldice", "tprec", "tsens"):
            row[key] = float(row[key])
        for key in ("breaks", "gt_px", "skel_px", "min_size"):
            row[key] = int(row[key])
    return rows


def per_seed(rows: list[dict], config: str, band: str, min_size: int,
             key: str) -> np.ndarray:
    """One number per seed: the mean over the 20 validation images."""
    out = []
    for seed in SEEDS:
        picked = [r[key] for r in rows
                  if r["run"] == f"{config}_s{seed}" and r["band"] == band
                  and r["min_size"] == min_size]
        out.append(np.nanmean(picked))
    return np.array(out, dtype=float)


def cell(rows, config, band, min_size, key) -> tuple[float, float]:
    values = per_seed(rows, config, band, min_size, key)
    return float(values.mean()), float(values.std())


def main() -> None:
    rows = load()
    bands = list(stratify.BANDS)
    CONFIGS[:] = [c for c in train.CONFIGS
                  if any(r["config"] == c for r in rows)]

    print("=== 每個對比層有多少血管（A_dice、未過濾、平均每張圖）===")
    print(f"{'band':>14}{'血管像素':>12}{'中心線像素':>12}{'佔全部':>10}")
    total_gt = sum(cell(rows, "A_dice", b, 0, "gt_px")[0] for b in bands)
    for band in bands:
        gt_px = cell(rows, "A_dice", band, 0, "gt_px")[0]
        skel_px = cell(rows, "A_dice", band, 0, "skel_px")[0]
        print(f"{band:>14}{gt_px:12.0f}{skel_px:12.0f}{100 * gt_px / total_gt:9.1f}%")

    for min_size in stratify.MIN_SIZES:
        state = "未過濾" if min_size == 0 else f"過濾 <{min_size}px"
        for key, label in (("dice", "Dice"), ("cldice", "clDice")):
            print(f"\n=== {label}，依對比層（{state}；3 seeds mean ± std）===")
            print(f"{'config':14}" + "".join(f"{b:>20}" for b in bands))
            for config in CONFIGS:
                cells = ""
                for band in bands:
                    mean, std = cell(rows, config, band, min_size, key)
                    cells += f"{mean:14.4f}±{std:.4f}"
                print(f"{config:14}{cells}")
            spread = ""
            for band in bands:
                values = [cell(rows, c, band, min_size, key)[0] for c in CONFIGS]
                spread += f"{max(values) - min(values):20.4f}"
            print(f"{'最好-最差':14}{spread}")

    print("\n=== 斷點數（每張圖，未過濾 / 過濾後）===")
    print(f"{'config':14}" + "".join(f"{b:>16}" for b in bands) + f"{'合計':>10}")
    for min_size in stratify.MIN_SIZES:
        for config in CONFIGS:
            cells, total = "", 0.0
            for band in bands:
                mean, std = cell(rows, config, band, min_size, "breaks")
                cells += f"{mean:11.1f}±{std:4.1f}"
                total += mean
            tag = f"{config}" if min_size == 0 else f"{config}*"
            print(f"{tag:14}{cells}{total:10.1f}")
        print()

    print("=== 斷點的分佈：每個對比層佔全部斷點的幾成（未過濾）===")
    for config in CONFIGS:
        totals = [cell(rows, config, b, 0, "breaks")[0] for b in bands]
        share = "".join(f"{100 * t / sum(totals):15.1f}%" for t in totals)
        print(f"{config:14}{share}")

    print("\n=== 關鍵對照：clDice loss 相對 Dice loss 的增益，依對比層 ===")
    print(f"{'指標':>10}{'過濾':>8}" + "".join(f"{b:>16}" for b in bands))
    for min_size in stratify.MIN_SIZES:
        for key, label in (("dice", "Dice"), ("cldice", "clDice"),
                           ("breaks", "斷點")):
            cells = ""
            for band in bands:
                a = cell(rows, "A_dice", band, min_size, key)[0]
                b = cell(rows, "B_cldice", band, min_size, key)[0]
                # Breaks: fewer is better, so report B - A as a raw count.
                cells += (f"{b - a:16.1f}" if key == "breaks"
                          else f"{b - a:+16.4f}")
            state = "否" if min_size == 0 else "是"
            print(f"{label:>10}{state:>8}{cells}")


if __name__ == "__main__":
    main()
