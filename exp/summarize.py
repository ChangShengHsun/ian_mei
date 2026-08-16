"""Aggregate the run matrix into results/summary.csv + a paired per-image check.

Three seeds is too few for a significance test, so this reports the seed range
instead: if two arms' ranges do not overlap, the ordering survived every seed.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
KEYS = ("dice", "cldice", "betti0_err", "betti1_err", "hd95")
SEEDS = (0, 1, 2)
CONFIGS = ("A_dice", "B_cldice", "C_boundary", "D_blurpool")


def final_scores(run_name: str) -> dict[str, float]:
    rows = list(csv.DictReader((RESULTS / run_name / "log.csv").open()))
    return {key: float(rows[-1][key]) for key in KEYS}


def per_image(run_name: str) -> dict[str, dict]:
    rows = csv.DictReader((RESULTS / run_name / "val_final.csv").open())
    return {row["image"]: row for row in rows}


def main() -> None:
    table = {}
    for config in CONFIGS:
        by_seed = [final_scores(f"{config}_s{seed}") for seed in SEEDS]
        table[config] = {
            key: (np.mean([s[key] for s in by_seed]),
                  np.std([s[key] for s in by_seed]),
                  min(s[key] for s in by_seed),
                  max(s[key] for s in by_seed)) for key in KEYS}

    with (RESULTS / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "metric", "mean", "std", "min", "max"])
        for config, scores in table.items():
            for key, values in scores.items():
                writer.writerow([config, key, *(round(v, 4) for v in values)])

    print(f"{'config':12} {'Dice':>16} {'clDice':>16} {'b0 err':>16} {'95HD':>14}")
    for config, scores in table.items():
        cells = []
        for key in ("dice", "cldice", "betti0_err", "hd95"):
            mean, std, low, high = scores[key]
            digits = 1 if key == "betti0_err" else 4
            cells.append(f"{mean:.{digits}f}±{std:.{digits}f}")
        print(f"{config:12} {cells[0]:>16} {cells[1]:>16} "
              f"{cells[2]:>16} {cells[3]:>14}")

    print("\nseed range of betti0_err (non-overlapping ranges = robust ordering)")
    for config, scores in table.items():
        _, _, low, high = scores["betti0_err"]
        print(f"  {config:12} {low:6.1f} .. {high:6.1f}")

    # Paired per-image: same 20 val images, seed-averaged, A versus each arm.
    averaged = defaultdict(dict)
    for config in CONFIGS:
        per_seed = [per_image(f"{config}_s{seed}") for seed in SEEDS]
        for name in per_seed[0]:
            averaged[config][name] = {
                key: float(np.mean([float(s[name][key]) for s in per_seed]))
                for key in ("dice", "betti0_err")}

    print("\npaired per-image versus A_dice (20 val images, seed-averaged)")
    for config in CONFIGS[1:]:
        better_topology = sum(
            averaged[config][n]["betti0_err"] < averaged["A_dice"][n]["betti0_err"]
            for n in averaged[config])
        better_dice = sum(
            averaged[config][n]["dice"] > averaged["A_dice"][n]["dice"]
            for n in averaged[config])
        print(f"  {config:12} fewer breaks on {better_topology:2d}/20 images, "
              f"higher Dice on {better_dice:2d}/20")


if __name__ == "__main__":
    main()
