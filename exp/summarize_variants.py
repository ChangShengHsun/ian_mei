"""The verdict on A1 (SWA), A2 (seed ensemble) and A3 (TTA).

WRITTEN AND SELFTESTED 2026-08-27, BEFORE score_variants.py PRODUCED A SINGLE
NUMBER. The three predictions below are pre-registered. Two of them can hurt
this series' own conclusions, which is why they are written here rather than
recalled afterwards.

  P1 (SWA).      Averaging epochs 60-100 beats rule (i), the current
                 protocol, on reported ERL. Against rule (iv), the best rule
                 task A3 found, it ties or wins slightly.
                 IF SWA LOSES TO (iv): there is real signal in the tail worth
                 picking, and "do not pick, average" is wrong for this
                 problem. That is a result, and it gets reported as one.

  P2 (TTA).      TTA helps the arms WITHOUT dihedral training (A_dice,
                 G_focal) more than the arms with it (H_aug, K_focal_aug),
                 because the latter are already close to invariant under
                 exactly the group TTA averages over.
                 THIS PREDICTION WEAKENS E14. If it holds, part of what E14
                 measured as an augmentation advantage is test-time
                 invariance that costs eight forward passes and no training,
                 and E14's claim has to say so. It is written down first so
                 that it cannot be quietly dropped if it comes true.

  P3 (ensemble). Six seeds averaged beat the BEST single seed of the same
                 config, not merely the mean seed. Beating the mean seed
                 would only say that averaging reduces variance, which needs
                 no experiment.
                 An ensemble is six models against one and is NOT a fair
                 comparison against a single arm. The fair one is ensemble
                 against ensemble, which is the last table here, and it is
                 the only comparison in this series with seed noise removed
                 from both sides.

THE GATE is the repo's: paired t over (image, seed) AND every seed agreeing
in sign, at least 3 seeds. Ensembles have no seeds -- there is one ensemble
per config -- so their rows are paired over images only and are labelled
"unseeded", never "HOLDS".

  python exp/summarize_variants.py --selftest
  python exp/summarize_variants.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as rules_module
import summarize_selection as selection

SCORES = selection.SWEEP / "variant_scores.csv"
ARMS = selection.ARMS
# Which arms train under the dihedral group TTA averages over. P2 is about
# exactly this split, so it is named here rather than inferred from a string.
DIHEDRAL_TRAINED = ("H_aug", "K_focal_aug")
# (variant, its baseline) for every paired comparison, in report order.
PAIRS = (("swa", "rule_i"), ("swa", "rule_iv"),
         ("tta_i", "rule_i"), ("tta_iv", "rule_iv"),
         ("swa_tta", "swa"), ("swa_tta", "rule_iv"))
ENSEMBLE_PAIRS = (("ens_i", "rule_i"), ("ens_iv", "rule_iv"),
                  ("ens_swa", "swa"))


def load(path: Path = SCORES) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ("dice", "cldice", "betti0_err", "erl", "skel_px"):
            row[key] = float(row[key])
    return rows


def report_only(rows) -> list[dict]:
    """The half of the images no weights were selected on."""
    return [r for r in rows if not rules_module.is_selection_image(r["image"])]


def indexed(rows) -> dict:
    """{(config, seed, variant, image): row}"""
    return {(r["config"], r["seed"], r["variant"], r["image"]): r
            for r in rows}


def traced(rows) -> float:
    """Mean fraction of the ground-truth tree an error-free trace covers."""
    return float(np.mean([r["erl"] / r["skel_px"] for r in rows]))


def compare(rows, config: str, variant: str, base: str) -> dict | None:
    """Paired ERL difference for one arm, gated the repo's way."""
    table = indexed(rows)
    seeds = sorted({r["seed"] for r in rows
                    if r["config"] == config and r["variant"] == variant})
    images = sorted({r["image"] for r in rows if r["config"] == config})
    paired, per_seed, dice_cost = [], [], []
    for seed in seeds:
        inside = []
        for image in images:
            mine = table.get((config, seed, variant, image))
            theirs = table.get((config, seed, base, image))
            if mine is None or theirs is None:
                return None
            inside.append((mine["erl"], theirs["erl"]))
            dice_cost.append(mine["dice"] - theirs["dice"])
        paired.extend(inside)
        per_seed.append(float(np.mean([a - b for a, b in inside])))
    if not paired:
        return None
    diffs = np.array([a - b for a, b in paired])
    result = stats.ttest_rel([a for a, _ in paired], [b for _, b in paired])
    unseeded = seeds == [""]
    holds = bool(not unseeded and diffs.mean() > 0
                 and result.statistic > 2
                 and all(d > 0 for d in per_seed) and len(per_seed) >= 3)
    return {"mean": float(diffs.mean()), "t": float(result.statistic),
            "holds": holds, "unseeded": unseeded, "seeds": len(per_seed),
            "per_seed": per_seed, "dice": float(np.mean(dice_cost)),
            "share": traced([r for r in rows if r["config"] == config
                             and r["variant"] == variant])}


def table(rows, pairs) -> None:
    header = (f"  {'arm':<14}{'traced':>8}{'vs base':>10}{'t':>7}"
              f"{'dDice':>9}  gate")
    for variant, base in pairs:
        print(f"{variant} vs {base}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for config in ARMS:
            got = compare(rows, config, variant, base)
            if got is None:
                print(f"  {config:<14}{'not scored':>8}")
                continue
            if got["unseeded"]:
                verdict = "unseeded"
            else:
                verdict = "HOLDS" if got["holds"] else "fails"
            print(f"  {config:<14}{got['share']:7.1%}{got['mean']:+10.1f}"
                  f"{got['t']:7.2f}{got['dice']:+9.5f}  {verdict}")
            if not got["holds"] and not got["unseeded"] and got["mean"] > 0:
                print(f"  {'':<14}per seed "
                      f"[{' '.join(f'{d:+.0f}' for d in got['per_seed'])}]")
        print()


def check_p2(rows) -> None:
    """P2: does TTA help the un-augmented arms more than the augmented ones?"""
    print("P2, pre-registered: TTA should help the arms that do NOT train")
    print("under the dihedral group more than the arms that do.")
    gains = {}
    for config in ARMS:
        got = compare(rows, config, "tta_i", "rule_i")
        if got is not None:
            gains[config] = got["mean"]
    if len(gains) < len(ARMS):
        print("  not every arm scored yet\n")
        return
    trained = [gains[c] for c in ARMS if c in DIHEDRAL_TRAINED]
    untrained = [gains[c] for c in ARMS if c not in DIHEDRAL_TRAINED]
    for config in ARMS:
        mark = "dihedral-trained" if config in DIHEDRAL_TRAINED else "not"
        print(f"  {config:<14}{gains[config]:+9.1f} ERL   ({mark})")
    print(f"  mean gain: not-trained {np.mean(untrained):+.1f}, "
          f"dihedral-trained {np.mean(trained):+.1f}")
    if np.mean(untrained) > np.mean(trained):
        print("  P2 HELD. Part of what E14 measured as an augmentation")
        print("  advantage is test-time invariance, available for eight")
        print("  forward passes and no training. E14's claim has to say so.")
    else:
        print("  P2 did NOT hold: TTA helps the augmented arms at least as")
        print("  much, so the two are not substitutes.")
    print()


def check_p3(rows) -> None:
    """P3: does the ensemble beat the BEST single seed, not just the mean?"""
    print("P3, pre-registered: a six-seed ensemble should beat the arm's")
    print("BEST single seed, not merely its average seed.")
    print(f"  {'arm':<14}{'best seed':>11}{'ensemble':>11}{'verdict':>10}")
    for config in ARMS:
        singles = defaultdict(list)
        ensemble = []
        for row in rows:
            if row["config"] != config:
                continue
            if row["variant"] == "rule_iv":
                singles[row["seed"]].append(row)
            elif row["variant"] == "ens_iv":
                ensemble.append(row)
        if not singles or not ensemble:
            print(f"  {config:<14}{'not scored':>11}")
            continue
        best = max(traced(these) for these in singles.values())
        got = traced(ensemble)
        print(f"  {config:<14}{best:10.1%}{got:10.1%}"
              f"{'beats' if got > best else 'loses':>10}")
    print("  Six models against one is not a fair comparison and this table")
    print("  does not claim to be one; the fair one is ensemble vs ensemble.\n")


def ensemble_versus_ensemble(rows) -> None:
    """The one comparison in this series with seed noise removed both sides."""
    print("Ensemble against ensemble -- the fair comparison, and the only")
    print("one here with seed variation removed from BOTH arms:")
    baseline = "A_dice"
    table_rows = indexed(rows)
    images = sorted({r["image"] for r in rows})
    for config in ARMS:
        if config == baseline:
            continue
        paired = []
        for image in images:
            mine = table_rows.get((config, "", "ens_iv", image))
            theirs = table_rows.get((baseline, "", "ens_iv", image))
            if mine is None or theirs is None:
                break
            paired.append((mine["erl"], theirs["erl"]))
        if len(paired) != len(images) or not paired:
            print(f"  {config:<14}not scored")
            continue
        diffs = np.array([a - b for a, b in paired])
        result = stats.ttest_rel([a for a, _ in paired],
                                 [b for _, b in paired])
        print(f"  {config:<14}vs {baseline}: {diffs.mean():+8.1f} ERL  "
              f"t {result.statistic:6.2f} over {len(paired)} images")
    print("  Paired over images only: there is one ensemble per arm, so the")
    print("  seed gate does not apply and is not claimed.\n")


def selftest() -> None:
    def make(config, seed, variant, image, value, dice=0.82):
        return {"config": config, "run": f"{config}_s{seed}", "seed": seed,
                "variant": variant, "image": image, "dice": dice,
                "cldice": 0.83, "betti0_err": 50.0, "erl": value,
                "skel_px": 8000.0}

    # Three seeds where the variant wins on every one: must HOLD.
    rows = []
    for seed in ("0", "1", "2"):
        for image in [f"{i:02d}" for i in range(1, 21)]:
            rows.append(make("A_dice", seed, "rule_i", image, 2000.0))
            rows.append(make("A_dice", seed, "swa", image, 2300.0))
    got = compare(report_only(rows), "A_dice", "swa", "rule_i")
    assert got["holds"] and abs(got["mean"] - 300.0) < 1e-6, got
    assert got["seeds"] == 3, got
    print(f"three seeds all improving: mean {got['mean']:+.0f}, gate HOLDS")

    # Only the REPORT half may reach the verdict. Plant a huge value on the
    # selection half; it must not move the mean.
    for row in rows:
        if rules_module.is_selection_image(row["image"]) \
                and row["variant"] == "swa":
            row["erl"] = 99999.0
    again = compare(report_only(rows), "A_dice", "swa", "rule_i")
    assert abs(again["mean"] - 300.0) < 1e-6, again
    print("  the 99999 planted on the selection half never reaches the "
          "verdict")

    # One seed disagreeing must fail, however large the mean.
    for row in rows:
        if row["seed"] == "2" and row["variant"] == "swa" \
                and not rules_module.is_selection_image(row["image"]):
            row["erl"] = 1990.0
    flipped = compare(report_only(rows), "A_dice", "swa", "rule_i")
    assert flipped["mean"] > 0 and not flipped["holds"], flipped
    print(f"  one seed disagreeing fails the gate at mean "
          f"{flipped['mean']:+.0f}: {flipped['per_seed']}")

    # An ensemble has no seeds and must never be reported as HOLDS.
    ens = [make("A_dice", "", "ens_iv", f"{i:02d}", 2500.0)
           for i in range(1, 21)]
    ens += [make("A_dice", "", "rule_iv", f"{i:02d}", 2000.0)
            for i in range(1, 21)]
    unseeded = compare(report_only(ens), "A_dice", "ens_iv", "rule_iv")
    assert unseeded["unseeded"] and not unseeded["holds"], unseeded
    print(f"an ensemble improving by {unseeded['mean']:+.0f} is reported "
          f"'unseeded', never 'HOLDS' -- one ensemble is not three seeds")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not SCORES.exists():
        raise SystemExit(f"{SCORES} not built yet -- run exp/score_variants.py")
    rows = report_only(load())
    print("=== A1 / A2 / A3: inference-time methods ===")
    print(f"{len({(r['config'], r['seed'], r['variant']) for r in rows})} "
          f"(arm, seed, variant) combinations; ERL on the report half only\n")
    table(rows, PAIRS)
    check_p2(rows)
    table(rows, ENSEMBLE_PAIRS)
    check_p3(rows)
    ensemble_versus_ensemble(rows)
    print("'traced' is the mean fraction of the ground-truth skeleton an")
    print("error-free trace covers. 'dDice' is what the variant costs in")
    print("whole-image Dice: a topology gain bought with overlap is a trade,")
    print("not an improvement, and this column is where that shows.")


if __name__ == "__main__":
    main()
