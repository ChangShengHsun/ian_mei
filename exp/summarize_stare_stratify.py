"""Turn stare_stratify.csv and stare_hesitation.csv into the two answers.

E3'  Does the choice of annotator change TOPOLOGY, and does training on the
     soft consensus help? The reference is stage 0: swapping annotator cost
     0.097 Dice, 22x more than swapping the loss did on DRIVE. Nobody has
     reported the same swap in break counts.

E1'  Does the model hesitate where the two humans disagree? AUROC of 0.5 means
     the probability map carries no information about which pixels are
     contested. Anything clearly above 0.5 means a model trained on one
     annotator already knows where the other one would have argued.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stratify
import train_stare

RESULTS = Path(__file__).resolve().parent / "results"
TARGETS = ("ah", "vk", "soft")
BANDS = list(stratify.BANDS)


def load(name: str, floats: tuple) -> list[dict]:
    rows = list(csv.DictReader((RESULTS / name).open()))
    for row in rows:
        for key in floats:
            row[key] = float(row[key])
    return rows


def cell(rows, band, key, **filters) -> tuple[float, float, int]:
    """Mean over runs of the per-run image mean, plus the spread across runs."""
    picked = [r for r in rows if r["band"] == band
              and all(r[k] == v for k, v in filters.items())]
    per_run = {}
    for row in picked:
        per_run.setdefault(row["run"], []).append(row[key])
    values = np.array([np.nanmean(v) for v in per_run.values()])
    return (float(np.nanmean(values)), float(np.nanstd(values)), len(values))


def main() -> None:
    topology = load("stare_stratify.csv",
                    ("breaks", "dice", "cldice", "tprec", "tsens"))
    hesitation = load("stare_hesitation.csv",
                      ("auroc", "contested_frac", "hesitation_contested",
                       "hesitation_agreed", "prob_contested", "prob_agreed",
                       "n_px"))

    print("=== E3' 每個對比層有多少爭議（union 內，兩位標註者不一致的比例）===")
    print(f"{'band':>14}{'爭議比例':>12}{'像素數':>12}")
    for band in BANDS:
        # Both are properties of the labels, not of any model, so one run
        # carries the whole answer.
        frac, _, _ = cell(hesitation, band, "contested_frac", run="ah_f0_s0")
        size, _, _ = cell(hesitation, band, "n_px", run="ah_f0_s0")
        print(f"{band:>14}{100 * frac:11.1f}%{size:12.0f}")

    print("\n=== E3' 斷點數，依訓練目標與評分對象（每張圖；mean ± std over runs）===")
    for annotator in ("ah", "vk"):
        print(f"\n  對 {annotator} 評分")
        print(f"{'trained_on':>12}" + "".join(f"{b:>17}" for b in BANDS)
              + f"{'合計':>10}")
        for target in TARGETS:
            cells, total = "", 0.0
            for band in BANDS:
                mean, std, _ = cell(topology, band, "breaks",
                                    trained_on=target,
                                    scored_against=annotator)
                cells += f"{mean:12.1f}±{std:4.1f}"
                total += mean
            print(f"{target:>12}{cells}{total:10.1f}")

    print("\n=== E3' Dice，依訓練目標與評分對象 ===")
    for annotator in ("ah", "vk"):
        print(f"\n  對 {annotator} 評分")
        print(f"{'trained_on':>12}" + "".join(f"{b:>17}" for b in BANDS))
        for target in TARGETS:
            cells = ""
            for band in BANDS:
                mean, std, _ = cell(topology, band, "dice",
                                    trained_on=target,
                                    scored_against=annotator)
                cells += f"{mean:12.4f}±{std:.3f}"
            print(f"{target:>12}{cells}")

    print("\n=== E3' 主要對照：軟標籤有沒有買到拓樸 ===")
    print("  （對兩位標註者的平均斷點；越低越好）")
    print(f"{'trained_on':>12}{'對 ah':>12}{'對 vk':>12}{'平均':>12}{'最差':>12}")
    for target in TARGETS:
        per_annotator = []
        for annotator in ("ah", "vk"):
            total = sum(cell(topology, b, "breaks", trained_on=target,
                             scored_against=annotator)[0] for b in BANDS)
            per_annotator.append(total)
        print(f"{target:>12}{per_annotator[0]:12.1f}{per_annotator[1]:12.1f}"
              f"{np.mean(per_annotator):12.1f}{max(per_annotator):12.1f}")

    print("\n=== E1' 模型的猶豫落在人類的爭議上嗎（AUROC；0.5 = 毫無資訊）===")
    print(f"{'trained_on':>12}" + "".join(f"{b:>17}" for b in BANDS)
          + f"{'全部':>12}")
    for target in TARGETS:
        cells = ""
        for band in BANDS:
            mean, std, n = cell(hesitation, band, "auroc", trained_on=target)
            cells += f"{mean:12.3f}±{std:.3f}"
        overall = np.nanmean([r["auroc"] for r in hesitation
                              if r["trained_on"] == target])
        print(f"{target:>12}{cells}{overall:12.3f}")

    print("\n=== E1' 猶豫程度與機率：有爭議 vs 兩人同意 ===")
    print("  （猶豫 = 1-2|p-0.5|，1 代表剛好卡在 0.5；機率欄解釋方向）")
    print(f"{'trained_on':>12}{'band':>16}{'猶豫:爭議':>11}{'猶豫:同意':>11}"
          f"{'差':>9}{'p:爭議':>10}{'p:同意':>10}")
    for target in TARGETS:
        for band in BANDS:
            values = [cell(hesitation, band, key, trained_on=target)[0]
                      for key in ("hesitation_contested", "hesitation_agreed",
                                  "prob_contested", "prob_agreed")]
            hesitate_contested, hesitate_agreed, p_contested, p_agreed = values
            print(f"{target:>12}{band:>16}{hesitate_contested:11.4f}"
                  f"{hesitate_agreed:11.4f}"
                  f"{hesitate_contested - hesitate_agreed:+9.4f}"
                  f"{p_contested:10.4f}{p_agreed:10.4f}")


if __name__ == "__main__":
    main()
