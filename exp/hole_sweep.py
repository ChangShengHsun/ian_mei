"""E9: the dual of E4. Component filtering fixes betti-0; what fixes betti-1?

E4 swept "drop predicted components smaller than k * w^2" and found it buys
2.4-2.6x what the topology loss buys on betti-0. The 2026-08-19 revision to E4
explains why that never transferred to betti-1: removing a component changes
the component count by one and the loop count by however many loops it
contained, which for a speckle blob is zero. Component filtering is
structurally a betti-0 tool.

But that argument is about ONE post-processing operation, and there is an
obvious dual. remove_small_holes deletes spurious loops the way
remove_small_objects deletes spurious components. It has never been swept here,
so "betti-1 is a model-level problem" is currently a conjecture, not a result.

E9 sweeps it, in the same width-relative units as E4 so the two are readable
side by side, and asks the same question E4 asked:

    does post-processing close the gap between the losses on betti-1,
    the way it closed the gap on betti-0?

If yes, E4's headline generalises and the topology loss's betti-1 advantage is
purchasable too. If no, "loops are a model-level problem" becomes a real
finding, and a much stronger one than E4 could support on its own.

Runs on the existing E4 checkpoints -- inference only, no training.

  python exp/hole_sweep.py hrf
  python exp/hole_sweep.py topomortar
  python exp/hole_sweep.py --selftest

Writes results/cross/<dataset>/hole_sweep.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.morphology import remove_small_holes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import metrics
import speckle
import train

torch.set_num_threads(2)  # the trainer already holds six; see CLAUDE.md

# The filter level E4 reports its headline at, so the "on top of E4's filter"
# arm answers the question a reader actually has: given that I am already
# filtering components, does hole filling add anything?
E4_COMPONENT_MULTIPLE = 2.08
COMPONENT_ARMS = (0.0, E4_COMPONENT_MULTIPLE)
# metrics.betti counts an 8-connected foreground, so holes must be found with
# a 4-connected background or the two disagree about diagonal pinch points.
# selftest() checks this against metrics.betti rather than trusting the flag.
BACKGROUND_CONNECTIVITY = 1


def fill_small_holes(mask: np.ndarray, area: int) -> np.ndarray:
    """Mirror of speckle.drop_small: removes holes STRICTLY smaller than area.

    skimage's max_size is inclusive (<=) while drop_small keeps sizes >=
    min_size, so the two only sweep the same x axis if max_size is area - 1.
    The older area_threshold argument had the exclusive meaning but is
    deprecated in skimage 0.26 and warns once per call, which at 2100 images
    x 6 models would be the whole log.
    """
    if area <= 1:
        return mask
    return remove_small_holes(mask, max_size=area - 1,
                              connectivity=BACKGROUND_CONNECTIVITY)


def score_run(model, test_items: list[dict], mean: float, std: float,
              width: float, run_name: str) -> list[dict]:
    rows = []
    for item in test_items:
        prob = train.predict_full(model, item["image"], mean, std)
        raw = (prob >= 0.5) & item["fov"]
        truth = item["label"]
        b0_gt, b1_gt = metrics.betti(truth)
        for component_multiple in COMPONENT_ARMS:
            base = speckle.drop_small(
                raw, int(round(component_multiple * width * width)))
            for hole_multiple in cross_dataset.WIDTH_MULTIPLES:
                area = int(round(hole_multiple * width * width))
                pred = fill_small_holes(base, area)
                b0, b1 = metrics.betti(pred)
                rows.append({
                    "run": run_name,
                    "config": run_name.rsplit("_s", 1)[0],
                    "image": item["name"],
                    "component_multiple": component_multiple,
                    "hole_multiple": hole_multiple, "hole_area": area,
                    "dice": round(metrics.dice(pred, truth), 5),
                    "cldice": round(metrics.cl_dice(pred, truth), 5),
                    "betti0_err": abs(b0 - b0_gt),
                    "betti1_err": abs(b1 - b1_gt),
                })
    return rows


def selftest() -> None:
    """Filling a hole must cost exactly one loop and no components.

    Checked against metrics.betti, the same counter E4 and E9 report with, so
    a connectivity mismatch between the filler and the counter shows up here
    rather than as a quietly wrong sweep.
    """
    canvas = np.zeros((60, 60), dtype=bool)
    canvas[5:55, 28:32] = True          # the real structure
    canvas[8:15, 8:15] = True           # a square...
    canvas[10:13, 10:13] = False        # ...with a 9 px hole
    b0, b1 = metrics.betti(canvas)
    assert (b0, b1) == (2, 1), (b0, b1)

    filled = fill_small_holes(canvas, 20)
    b0_filled, b1_filled = metrics.betti(filled)
    print(f"before {b0},{b1} -> after filling holes under 20px "
          f"{b0_filled},{b1_filled}")
    assert (b0_filled, b1_filled) == (2, 0), (b0_filled, b1_filled)

    # Below threshold it must do nothing, or the sweep's x axis is a lie.
    untouched = fill_small_holes(canvas, 5)
    assert metrics.betti(untouched) == (2, 1), metrics.betti(untouched)
    print("a 9px hole survives a 5px threshold and dies at 20px")

    # And the dual has to be genuinely different from E4's tool: component
    # filtering leaves this loop alone no matter how hard it is turned up.
    survived = speckle.drop_small(canvas, 40)
    b0_dropped, b1_dropped = metrics.betti(survived)
    print(f"component filter <40px on the same mask: {b0_dropped},{b1_dropped}"
          f" (the 49px ring survives, its loop with it)")
    assert b1_dropped == 1, b1_dropped

    # Hole filling is supposed to leave betti-0 alone, and on the HRF sweep it
    # did on 172 of 180 curves. The other eight all moved by exactly -1, which
    # is this: a false positive stranded inside a loop is its own component,
    # and filling the loop's interior absorbs it. Kept as a check so nobody
    # later reads a betti-0 movement here as a defect in the sweep.
    stranded = np.zeros((40, 40), dtype=bool)
    stranded[10:21, 15:26] = True
    stranded[12:19, 17:24] = False      # a 45 px lake
    stranded[15:17, 20:22] = True       # an island in it
    assert metrics.betti(stranded) == (2, 1), metrics.betti(stranded)
    assert metrics.betti(fill_small_holes(stranded, 40)) == (2, 1)
    absorbed = metrics.betti(fill_small_holes(stranded, 60))
    print(f"speck stranded inside a loop: filling the loop absorbs it, "
          f"betti0 2 -> {absorbed[0]}")
    assert absorbed == (1, 0), absorbed
    print("all checks passed")


def setup_drive() -> tuple[list[dict], Path, list[str]]:
    """DRIVE differs from the cross-dataset runs in two ways that matter here.

    Its checkpoints live at results/<run>/ rather than results/cross/<ds>/<run>,
    and train.py saves final.pt without the normalisation constants, so mean and
    std have to be recomputed from the training split the way stratify.py does.
    """
    import drive
    test_items = drive.load_split("val")
    runs = train.trained_runs()  # every seed on disk, not a hardcoded three
    return test_items, train.RESULTS, runs


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    dataset = sys.argv[1]
    if dataset == "drive":
        test_items, out_dir, default_runs = setup_drive()
        stacked = train.stack_split("train")
        inside = stacked["images"][stacked["fovs"]]
        drive_norm = (float(inside.mean()), float(inside.std()))
    else:
        _, test_items = cross_dataset.loader_for(dataset)()
        out_dir = cross_dataset.out_root(dataset)
        drive_norm = None
        default_runs = [f"{loss}_s{seed}"
                        for seed in cross_dataset.SEEDS
                        for loss in cross_dataset.LOSSES]
    width = cross_dataset.median_width(test_items)
    runs = sys.argv[2:] or default_runs
    print(f"{dataset}: {len(test_items)} test images, width {width:.2f} px, "
          f"hole sweep "
          f"{[int(round(m * width * width)) for m in cross_dataset.WIDTH_MULTIPLES]} px",
          flush=True)

    rows = []
    for run_name in runs:
        weights = out_dir / run_name / "final.pt"
        if not weights.exists():
            print(f"[{run_name}] no checkpoint, skipping", flush=True)
            continue
        state = torch.load(weights, weights_only=False)
        model = train.build_model(run_name.rsplit("_s", 1)[0])
        model.load_state_dict(state["model"])
        model.eval()
        mean, std = drive_norm or (state["mean"], state["std"])
        rows += score_run(model, test_items, mean, std, width, run_name)
        print(f"[{run_name}] done", flush=True)

    if not rows:
        raise SystemExit("no checkpoints found")
    out = out_dir / "hole_sweep.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    for component_multiple in COMPONENT_ARMS:
        state = ("no component filter" if component_multiple == 0
                 else f"component filter {component_multiple} w^2")
        print(f"\n=== {dataset}, {state} ===")
        print(f"{'hole w^2 x':>12}{'px':>7}{'betti1':>10}{'betti0':>10}"
              f"{'dice':>9}")
        for hole_multiple in cross_dataset.WIDTH_MULTIPLES:
            picked = [r for r in rows
                      if r["component_multiple"] == component_multiple
                      and r["hole_multiple"] == hole_multiple]
            print(f"{hole_multiple:12.2f}{picked[0]['hole_area']:7d}"
                  f"{np.mean([r['betti1_err'] for r in picked]):10.1f}"
                  f"{np.mean([r['betti0_err'] for r in picked]):10.1f}"
                  f"{np.mean([r['dice'] for r in picked]):9.4f}")


if __name__ == "__main__":
    main()
