"""Aggregate the cross-annotator runs: how much of a score is the annotator?

Each run trained on one annotator and was then scored against both, so the
drop from "own" to "other" is the part of the score that belongs to that
person's labelling habit rather than to vessel anatomy. The reference is the
human-human agreement measured in stare_agreement.py: a model that reaches
0.740 against the other annotator is exactly as close to them as its own
teacher is.
"""
import csv
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "stare_cross"
TARGETS = ("ah", "vk")
KEYS = ("dice", "cldice", "betti0_err", "hd95")
HUMAN_DICE, HUMAN_BETTI = 0.7401, 6.6
# Spread across the four DRIVE losses, from summary.csv -- the yardstick the
# annotator effect gets compared against.
LOSS_DICE_SPREAD = 0.8114 - 0.8071


def run_rows(run_dir: Path) -> list[dict]:
    return list(csv.DictReader((run_dir / "scores.csv").open()))


def main() -> None:
    runs = sorted(d for d in RESULTS.iterdir()
                  if (d / "scores.csv").exists())
    print(f"{len(runs)} runs: {', '.join(d.name for d in runs)}\n")

    per_run = {}
    for run_dir in runs:
        target = run_dir.name.split("_")[0]
        rows = run_rows(run_dir)
        scores = {}
        for annotator in TARGETS:
            picked = [r for r in rows if r["scored_against"] == annotator]
            scores[annotator] = {k: float(np.nanmean([float(r[k]) for r in picked]))
                                 for k in KEYS}
        per_run[run_dir.name] = (target, scores)

    print("=== 每次訓練：對自己的老師 vs 對另一位 ===")
    print(f"{'run':14}{'訓練目標':>9}{'對自己 Dice':>13}{'對另一位 Dice':>15}{'落差':>9}")
    for name, (target, scores) in per_run.items():
        other = "vk" if target == "ah" else "ah"
        own_dice, other_dice = scores[target]["dice"], scores[other]["dice"]
        print(f"{name:14}{target:>9}{own_dice:13.4f}{other_dice:15.4f}"
              f"{own_dice - other_dice:9.4f}")

    print("\n=== 依訓練目標彙整（4 次 = 2 折 x 2 種子）===")
    summary = {}
    for target in TARGETS:
        other = "vk" if target == "ah" else "ah"
        picked = [s for t, s in per_run.values() if t == target]
        summary[target] = {
            "own": np.array([s[target]["dice"] for s in picked]),
            "other": np.array([s[other]["dice"] for s in picked]),
            "own_b0": np.array([s[target]["betti0_err"] for s in picked]),
            "other_b0": np.array([s[other]["betti0_err"] for s in picked]),
        }
        own, oth = summary[target]["own"], summary[target]["other"]
        print(f"\n  用 {target} 訓練（n={len(own)}）")
        print(f"    對 {target}（自己的老師）Dice  {own.mean():.4f} ± {own.std():.4f}")
        print(f"    對 {other}（另一位）    Dice  {oth.mean():.4f} ± {oth.std():.4f}")
        print(f"    落差                        {own.mean() - oth.mean():.4f}")
        print(f"    對 {other} 相對人類基準 {HUMAN_DICE:.4f}   "
              f"{oth.mean() - HUMAN_DICE:+.4f}")

    drop_ah = summary["ah"]["own"].mean() - summary["ah"]["other"].mean()
    drop_vk = summary["vk"]["own"].mean() - summary["vk"]["other"].mean()
    print("\n=== 主要對照 ===")
    print(f"  換標註者造成的 Dice 落差（ah 訓練）  {drop_ah:.4f}")
    print(f"  換標註者造成的 Dice 落差（vk 訓練）  {drop_vk:.4f}")
    print(f"  不對稱倍數                          {drop_ah / drop_vk:.1f}x")
    print(f"  換 loss 造成的 Dice 落差（DRIVE 四種）{LOSS_DICE_SPREAD:.4f}")
    print(f"  換標註者 / 換 loss                   "
          f"{drop_ah / LOSS_DICE_SPREAD:.0f}x")

    # ASCII "b0" rather than the subscript: the Windows console codepage
    # (cp950 here) cannot encode U+2080 and the script dies on the print.
    print("\n=== b0 (betti-0) 誤差（過濾 20px 後；人類之間是 " f"{HUMAN_BETTI}）===")
    for target in TARGETS:
        other = "vk" if target == "ah" else "ah"
        print(f"  用 {target} 訓練 → 對 {target} {summary[target]['own_b0'].mean():6.1f}"
              f"   對 {other} {summary[target]['other_b0'].mean():6.1f}")

    with (RESULTS.parent / "stare_cross_summary.csv").open("w", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(["run", "trained_on", "scored_against", *KEYS])
        for name, (target, scores) in per_run.items():
            for annotator in TARGETS:
                writer.writerow([name, target, annotator]
                                + [round(scores[annotator][k], 5) for k in KEYS])
    print(f"\nwrote {RESULTS.parent / 'stare_cross_summary.csv'}")


if __name__ == "__main__":
    main()
