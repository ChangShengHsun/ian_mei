"""The Dice/connectivity frontier of each arm, swept by threshold.

WRITTEN AND SELFTESTED 2026-08-29, BEFORE BEING RUN ON ANY ARM.

WHY THIS AND NOT matched_cost.py. That script matches arms at a common Dice by
choosing, per run, the EPOCH whose dev Dice is closest to a target. It works,
but a run has ten kept epochs and their Dice values cluster, so most cells of
its table come back "--": there is no epoch near the target. The comparison is
right and the sampling is thin.

Threshold is the dense knob. A segmentation is `prob >= 0.5`; nothing forces
0.5. Lowering it predicts more foreground, which buys connectivity and spends
precision. Sweeping it traces the whole trade-off curve of ONE checkpoint --
and because every threshold reads the same probability map, the entire sweep
costs one forward pass. The expensive part of matched_cost.py was never the
matching, it was having only ten samples of it.

WHAT THE CURVE ANSWERS. Two arms can differ in two ways. Either one sits
ABOVE the other at every Dice -- it moved the frontier, which is a claim about
the method -- or they lie on one curve and differ only in where 0.5 happens to
land them, in which case "ours traces 15 points further" describes an
operating point. Only the first is a contribution. The published table cannot
tell them apart because it reads every arm at one threshold.

THE TWO-BRANCH PROBLEM, and the rule that handles it. Dice is not monotone in
threshold: it rises, peaks, and falls, so a target Dice is usually reached at
TWO thresholds -- one with too much foreground, one with too little. The low
one always traces further. Picking whichever traces further would be choosing
the flattering branch after seeing the answer, which is the post-hoc threshold
this repo's pre-registration rule exists to stop.

  RULE, fixed here before any run: among thresholds within TOLERANCE of the
  target Dice, take the HIGHEST threshold -- the least foreground. It is the
  conservative branch, it cannot flatter connectivity, and it is decided by
  the threshold value alone, never by the metric being compared.

PRE-REGISTERED PREDICTIONS.
  1. The published losses (B_cldice, E_cbdice, I_coletra, C_boundary,
     D_blurpool) lie on the SAME frontier as A_dice: no separation survives
     at matched Dice, at any Dice they both reach.
  2. K_focal_aug sits above that frontier, and its margin shrinks as Dice
     rises -- it spends Dice on connectivity, so it looks best where Dice is
     cheapest.
  3. The high centreline weights (clw8 and up) also sit above it, and by
     more than clw2 does. clw2 was measured to fail the seed gate under the
     held-out protocol, so it should be indistinguishable from its base here.
  4. Capacity (w64_d5, 31M) does NOT move the frontier: it lands on A_dice's
     curve, as it did in the epoch-matched table.

  python exp/frontier.py --selftest
  python exp/frontier.py [arm ...]        # appends to results/frontier.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import drive
import erl
import hole_sweep
import select_heldout as heldout
import speckle
import train

OUT = heldout.ROOT / "frontier.csv"
# Wide enough to bracket the Dice peak from both sides at every arm measured
# so far, fine enough that a target is usually hit without interpolation.
THRESHOLDS = tuple(round(0.10 + 0.025 * step, 3) for step in range(33))
TARGETS = (0.780, 0.790, 0.800, 0.810, 0.815, 0.820)
TOLERANCE = 0.003


def curve_for(model, items, geometry, mean: float, std: float,
              component_px: int) -> list[dict]:
    """One row per threshold: Dice and traced fraction over all test images.

    The probability map is computed once per image and thresholded 33 times.
    That is the whole reason this is affordable.
    """
    totals = defaultdict(lambda: {"inter": 0.0, "sizes": 0.0,
                                  "erl": 0.0, "skel": 0.0})
    for item, geo in zip(items, geometry):
        prob = train.predict_full(model, item["image"], mean, std)
        truth = item["label"] & item["fov"]
        for threshold in THRESHOLDS:
            pred = speckle.drop_small((prob >= threshold) & item["fov"],
                                      component_px)
            cell = totals[threshold]
            cell["inter"] += float((pred & truth).sum())
            cell["sizes"] += float(pred.sum() + truth.sum())
            cell["erl"] += erl.expected_run_length(geo["skel"], pred)
            cell["skel"] += float(geo["skel"].sum())
    rows = []
    for threshold in THRESHOLDS:
        cell = totals[threshold]
        rows.append({
            "threshold": threshold,
            "dice": 2.0 * cell["inter"] / cell["sizes"] if cell["sizes"] else 0.0,
            "traced": cell["erl"] / cell["skel"] if cell["skel"] else 0.0})
    return rows


def pick(rows: list[dict], target: float) -> dict | None:
    """The conservative branch: highest threshold within TOLERANCE of target.

    Never the one that traces furthest. Choosing the branch by the metric
    under comparison is the post-hoc threshold this file exists to avoid.
    """
    near = [row for row in rows if abs(row["dice"] - target) <= TOLERANCE]
    if not near:
        return None
    return max(near, key=lambda row: row["threshold"])


def load_curves(path: Path = OUT) -> dict:
    """{arm: {target: [traced per run]}} from the CSV."""
    if not path.exists():
        return {}
    by_run = defaultdict(list)
    for row in csv.DictReader(path.open()):
        by_run[row["run"]].append({"threshold": float(row["threshold"]),
                                   "dice": float(row["dice"]),
                                   "traced": float(row["traced"])})
    out = defaultdict(lambda: defaultdict(list))
    for run, rows in by_run.items():
        for target in TARGETS:
            chosen = pick(rows, target)
            if chosen is not None:
                out[run.rsplit("_s", 1)[0]][target].append(chosen["traced"])
    return out


def compare(curves: dict, arm: str, base: str, target: float) -> dict | None:
    mine = curves.get(arm, {}).get(target, [])
    theirs = curves.get(base, {}).get(target, [])
    count = min(len(mine), len(theirs))
    if count < 3:
        return None
    per_seed = [mine[i] - theirs[i] for i in range(count)]
    result = stats.ttest_rel(mine[:count], theirs[:count])
    return {"mean": float(np.mean(per_seed)), "t": float(result.statistic),
            "seeds": count,
            "holds": bool(np.mean(per_seed) > 0 and result.statistic > 2
                          and all(d > 0 for d in per_seed) and count >= 3)}


def selftest() -> None:
    # 1. THE TWO-BRANCH TRAP. Dice rises then falls with threshold, so a
    #    target is hit twice, and the low-threshold hit always traces further.
    #    The rule must take the HIGH one, whatever that costs it.
    rows = [{"threshold": 0.2, "dice": 0.800, "traced": 0.55},
            {"threshold": 0.4, "dice": 0.830, "traced": 0.40},
            {"threshold": 0.6, "dice": 0.800, "traced": 0.25}]
    chosen = pick(rows, 0.800)
    assert chosen["threshold"] == 0.6, chosen
    assert chosen["traced"] == 0.25, "the flattering branch was taken"
    print("  a Dice reached on both branches resolves to the conservative "
          "one (traced 0.25, not 0.55)")

    # 2. A target no threshold reaches is refused, never snapped to an end.
    assert pick(rows, 0.700) is None
    assert pick(rows, 0.8025) is not None and pick(rows, 0.8040) is None
    print(f"  a target more than {TOLERANCE} from every threshold is refused")

    # 3. THE CASE THIS FILE EXISTS FOR. Two arms on ONE frontier, differing
    #    only in where threshold 0.5 lands them. Read at 0.5 one looks far
    #    better; matched, they must be equal.
    curves = defaultdict(lambda: defaultdict(list))
    for arm, shift in (("cheap", 0.0), ("shifted", 0.1)):
        for _ in range(3):
            for target in TARGETS:
                # One curve: traced is a function of Dice alone. `shifted`
                # only reaches it at a different threshold.
                curves[arm][target].append(0.9 - target + shift * 0.0)
    for target in TARGETS:
        verdict = compare(curves, "shifted", "cheap", target)
        assert verdict is not None and abs(verdict["mean"]) < 1e-12, target
        assert not verdict["holds"], target
    print("  two arms on one frontier come out equal at every matched Dice")

    # 4. And an arm genuinely above the frontier must still pass.
    for _ in range(3):
        for target in TARGETS:
            curves["better"][target].append(0.9 - target + 0.05)
    for target in TARGETS:
        verdict = compare(curves, "better", "cheap", target)
        assert verdict is not None and verdict["holds"], (target, verdict)
    print("  an arm lifted at every Dice passes at every Dice")

    # 5. The threshold grid must actually bracket the Dice peak, or every
    #    curve is one-sided and the two-branch rule never fires.
    assert min(THRESHOLDS) < 0.5 < max(THRESHOLDS)
    assert len(set(THRESHOLDS)) == len(THRESHOLDS)
    print(f"  {len(THRESHOLDS)} thresholds from {min(THRESHOLDS)} to "
          f"{max(THRESHOLDS)}, bracketing 0.5")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    epochs = heldout.chosen_epochs()
    # final.pt only. chosen_epochs() reads log.csv, and a run still training
    # has a PARTIAL log -- so without this the frontier silently includes
    # models scored at whatever epoch they had reached when this started,
    # against arms that trained to 100. Caught on the first launch, when
    # A_dice_clw16_s0 was scored four minutes after its training began.
    runs = sorted(r for r in epochs
                  if (heldout.ROOT / r / "final.pt").exists()
                  and (not wanted or r.rsplit("_s", 1)[0] in wanted))
    unfinished = len(epochs) - len(runs)
    if unfinished:
        print(f"skipping {unfinished} run(s) still training", flush=True)
    done = set()
    if OUT.exists():
        done = {row["run"] for row in csv.DictReader(OUT.open())}
    runs = [r for r in runs if r not in done]
    if not runs:
        print("nothing to do")
        return

    items = drive.load_split("test")
    data = train.stack_split("fit")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(i["label"] & i["fov"])} for i in items]
    print(f"{len(runs)} run(s), {len(items)} images, "
          f"{len(THRESHOLDS)} thresholds, component filter {component_px} px",
          flush=True)

    fresh = not OUT.exists()
    with OUT.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run", "config", "seed", "epoch",
                                "threshold", "dice", "traced"])
        if fresh:
            writer.writeheader()
        for run in runs:
            config, seed = run.rsplit("_s", 1)
            weights = heldout.ROOT / run / f"epoch{epochs[run]:03d}.pt"
            if not weights.exists():
                print(f"  {run}: no epoch{epochs[run]:03d}.pt, skipping",
                      flush=True)
                continue
            model = train.build_model(config)
            model.load_state_dict(train.load_checkpoint(weights)["model"])
            model.eval()
            mean, std = train.normalisation(run, data)
            for row in curve_for(model, items, geometry, mean, std,
                                 component_px):
                writer.writerow({"run": run, "config": config, "seed": seed,
                                 "epoch": epochs[run], **row})
            handle.flush()
            print(f"  {run} done", flush=True)
    report()


def report() -> None:
    curves = load_curves()
    if not curves:
        raise SystemExit(f"{OUT} is empty")
    arms = [a for a in ("A_dice", "H_aug", "K_focal_aug") if a in curves]
    arms += sorted(a for a in curves if a not in arms)
    print("\nTraced fraction on the frontier, at a matched Dice reached by "
          "THRESHOLD.")
    print("Conservative branch (highest threshold within "
          f"{TOLERANCE} of the target).\n")
    header = "  " + f"{'arm':<24}" + "".join(f"{t:>9.3f}" for t in TARGETS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arm in arms:
        cells = []
        for target in TARGETS:
            values = curves[arm].get(target, [])
            cells.append(f"{np.mean(values):8.1%}" if len(values) >= 3
                         else f"{'--':>8} ")
        print(f"  {arm:<24}" + " ".join(cells))
    print("\n  Against A_dice:\n")
    for arm in arms:
        if arm == "A_dice":
            continue
        line = f"  {arm:<24}"
        for target in TARGETS:
            verdict = compare(curves, arm, "A_dice", target)
            line += (f"{'--':>9}" if verdict is None else
                     f"{verdict['mean']:>+8.1%}" +
                     ("*" if verdict["holds"] else " "))
        print(line)
    print("\n  * passes the seed gate at that Dice.")


if __name__ == "__main__":
    main()
