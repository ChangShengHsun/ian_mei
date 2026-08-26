"""Task A3's verdict: which selection rule yields the best model?

Written 2026-08-26 alongside select_checkpoint.py, BEFORE any test ERL existed.

Selection reads the odd DRIVE val images, the reported ERL reads the even
ones, and the two never mix -- select_checkpoint.is_selection_image is the
single place that decides which is which. This repo has no third split
(erl.py's "test set" IS train.validate()'s validation set, images 01-20), so
without the halving every ERL here would be reported on the images its own
weights were chosen on.

A rule beats the current one only under the repo's gate: paired over
(image, seed) AND every seed agreeing in sign. A mean improvement with a seed
disagreeing is E5's failure and is reported as a failure.

  python exp/summarize_selection.py --selftest
  python exp/summarize_selection.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as rules_module

SWEEP = Path(__file__).resolve().parent / "results" / "selection_sweep"
SCORES = SWEEP / "checkpoint_scores.csv"
ARMS = ("A_dice", "H_aug", "G_focal", "K_focal_aug")


def load(path: Path = SCORES) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row["epoch"] = int(row["epoch"])
        for key in ("dice", "cldice", "betti0_err", "erl", "skel_px"):
            row[key] = float(row[key])
    return rows


def selection_points(rows) -> dict:
    """{run: [{epoch, dice, cldice, betti0_err}, ...]} over SELECTION images.

    Averaged per checkpoint, because a rule chooses one epoch for the run, not
    one per image.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if rules_module.is_selection_image(row["image"]):
            grouped[row["run"]][row["epoch"]].append(row)
    out = {}
    for run, epochs in grouped.items():
        out[run] = [{
            "epoch": epoch,
            "dice": float(np.mean([r["dice"] for r in items])),
            "cldice": float(np.mean([r["cldice"] for r in items])),
            "betti0_err": float(np.mean([r["betti0_err"] for r in items])),
        } for epoch, items in sorted(epochs.items())]
    return out


def report_erl(rows) -> dict:
    """{(run, epoch): {image: erl}} over REPORT images only."""
    out = defaultdict(dict)
    for row in rows:
        if not rules_module.is_selection_image(row["image"]):
            out[(row["run"], row["epoch"])][row["image"]] = row["erl"]
    return out


def gate(paired: list[tuple], per_seed: list[float]) -> tuple[float, float,
                                                              bool]:
    """Mean, t and verdict for a list of (chosen, reference) ERL pairs."""
    diffs = np.array([a - b for a, b in paired])
    t = stats.ttest_rel([a for a, _ in paired], [b for _, b in paired])
    holds = bool(diffs.mean() > 0 and t.statistic > 2
                 and all(d > 0 for d in per_seed) and len(per_seed) >= 3)
    return float(diffs.mean()), float(t.statistic), holds


def evaluate(rows) -> None:
    points = selection_points(rows)
    erl = report_erl(rows)
    skel = {row["image"]: row["skel_px"] for row in rows}
    runs_by_arm = defaultdict(list)
    for run in sorted(points):
        runs_by_arm[run.rsplit("_s", 1)[0]].append(run)

    for arm in ARMS:
        runs = runs_by_arm.get(arm, [])
        if not runs:
            print(f"{arm}: not swept yet\n")
            continue
        print(f"=== {arm} ({len(runs)} seeds) ===")
        header = (f"  {'rule':<40}{'epochs':<26}{'traced':>8}"
                  f"{'vs (i)':>10}{'t':>7}  gate")
        print(header)
        print("  " + "-" * (len(header) - 2))
        baseline = None
        for name, rule in rules_module.rules():
            chosen, shares, given_up = {}, [], []
            for run in runs:
                pick = rule(points[run])
                chosen[run] = pick["epoch"]
                best_dice = max(p["dice"] for p in points[run])
                given_up.append(best_dice - pick["dice"])
                values = erl[(run, pick["epoch"])]
                shares.append(float(np.mean(
                    [v / skel[image] for image, v in values.items()])))
            paired, per_seed = [], []
            for run in runs:
                mine = erl[(run, chosen[run])]
                if baseline is None:
                    reference = mine
                else:
                    reference = erl[(run, baseline[run])]
                inside = [(mine[i], reference[i]) for i in sorted(mine)]
                paired.extend(inside)
                per_seed.append(float(np.mean([a - b for a, b in inside])))
            share = float(np.mean(shares))
            epochs = str([chosen[r] for r in runs])
            if baseline is None:
                print(f"  {name:<40}{epochs:<26}{share:7.1%}"
                      f"{'--':>10}{'--':>7}  reference")
                baseline = dict(chosen)
                reference_share = share
            else:
                mean, t, holds = gate(paired, per_seed)
                verdict = "HOLDS" if holds else "fails"
                print(f"  {name:<40}{epochs:<26}{share:7.1%}"
                      f"{mean:+10.1f}{t:7.2f}  {verdict}")
                if not holds and mean > 0:
                    print(f"  {'':<40}per seed "
                          f"[{' '.join(f'{d:+.0f}' for d in per_seed)}]")
            print(f"  {'':<40}gave up {np.mean(given_up):.5f} Dice on the "
                  f"selection half")
        print()


def selftest() -> None:
    # Two seeds' worth of a run where epoch 50 traces further than epoch 10 on
    # the report half, and epoch 10 has the better Dice. Rule (i) must pick 10
    # and a betti-driven rule must pick 50 -- and the ERL credited to each must
    # come from the REPORT images only.
    rows = []
    for seed in range(3):
        for image in range(1, 21):
            for epoch, dice, betti, erl_odd, erl_even in (
                    (10, 0.8200, 95.0, 9999.0, 2000.0),
                    (50, 0.8194, 68.0, 9999.0, 2600.0)):
                odd = image % 2 == 1
                rows.append({
                    "run": f"A_dice_s{seed}", "config": "A_dice",
                    "seed": str(seed), "epoch": epoch, "image": f"{image:02d}",
                    "dice": dice, "cldice": 0.83, "betti0_err": betti,
                    "erl": erl_odd if odd else erl_even, "skel_px": 8000.0})

    points = selection_points(rows)
    assert sorted(p["epoch"] for p in points["A_dice_s0"]) == [10, 50]
    assert rules_module.rule_best_dice(points["A_dice_s0"])["epoch"] == 10
    assert rules_module.rule_best_betti0(points["A_dice_s0"])["epoch"] == 50
    print("selection uses the odd images and ranks the epochs as expected")

    erl = report_erl(rows)
    assert set(erl[("A_dice_s0", 10)]) == {f"{i:02d}" for i in
                                           range(2, 21, 2)}, "report half only"
    assert all(v == 2000.0 for v in erl[("A_dice_s0", 10)].values())
    print("reported ERL reads the even images only -- the 9999 planted on the "
          "selection half never reaches it")

    paired = [(2600.0, 2000.0)] * 30
    mean, t, holds = gate(paired, [600.0, 600.0, 600.0])
    assert mean == 600.0 and holds, (mean, t, holds)
    flipped = gate(paired, [600.0, -10.0, 600.0])
    assert not flipped[2], "a seed disagreeing in sign must fail the gate"
    print("the gate needs the paired t AND every seed agreeing in sign")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not SCORES.exists():
        raise SystemExit(f"{SCORES} not built yet -- run exp/sweep_score.py")
    rows = load()
    checkpoints = len({(r["run"], r["epoch"]) for r in rows})
    print(f"=== task A3: checkpoint selection rules ===")
    print(f"{checkpoints} checkpoints over {len({r['run'] for r in rows})} "
          f"runs; selection on odd images, ERL reported on even ones\n")
    evaluate(rows)
    print("'traced' is the mean fraction of the ground-truth skeleton an")
    print("error-free trace covers, on the REPORT half. 'vs (i)' is paired")
    print("against the epoch rule (i) chose for the same run, so it isolates")
    print("the selection rule and nothing else.")


if __name__ == "__main__":
    main()
