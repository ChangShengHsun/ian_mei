"""E9b: the control E9's own conclusion demands.

E9 concluded that on tree-shaped structures (DRIVE, HRF) betti-1 error is
dominated by MISSING loops and therefore cannot be post-processed away. The
argument was: filling holes only ever removes loops, and removing a component
cannot create one, so no removal-type operation can help.

That argument is airtight about REMOVAL and says nothing about the rest of
morphology. Binary closing welds a one- or two-pixel gap shut, and welding a
gap in a vessel arcade creates a loop. So there is a post-processing operation
that can move betti-1 in the direction the retinal datasets need, and E9 did
not run it. By E9's own rule -- every topological claim needs its matching
post-processing control -- the conclusion is unguarded until this is measured.

The sweep is a closing radius in the same width-relative units the rest of the
series uses, so "radius 0.5 w" means half a structure width and transfers
across datasets. Closing also thickens everything it touches, so Dice is
reported alongside: the question is not only whether loops come back but what
they cost.

  python exp/bridge_sweep.py drive
  python exp/bridge_sweep.py hrf
  python exp/bridge_sweep.py --selftest

Writes <results>/bridge_sweep.csv next to the hole sweep for that dataset.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import hole_sweep
import metrics
import speckle
import train

torch.set_num_threads(2)  # the trainer already holds six; see CLAUDE.md

# Radii as multiples of the median structure width. Beyond about one width the
# operation stops bridging gaps and starts merging neighbouring vessels, which
# is a different (and much worse) thing; the sweep goes there anyway so the
# turning point is visible rather than assumed.
WIDTH_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def bridge(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary closing: dilate then erode, so gaps narrower than the structuring
    element are welded shut while the outline is otherwise preserved."""
    if radius < 1:
        return mask
    return ndimage.binary_closing(mask, structure=disk(radius))


def bridge_only(mask: np.ndarray, radius: int) -> np.ndarray:
    """Closing with its hole-filling side effect subtracted back out.

    Closing does two things at once and they pull betti-1 in opposite
    directions: welding a gap CREATES a loop, and swallowing a small enclosed
    background region DESTROYS one. E9 measured that the second is strongly
    negative on retinal data, so a plain closing sweep cannot answer whether
    the first is positive -- the two are confounded inside one operation.

    Restoring the background regions that were enclosed BEFORE the closing
    isolates the welding. Anything still filled afterwards was a real gap.
    """
    if radius < 1:
        return mask
    enclosed = ndimage.binary_fill_holes(mask) & ~mask
    return bridge(mask, radius) & ~enclosed


OPERATIONS = {"closing": bridge, "bridge_only": bridge_only}


def score_run(model, test_items: list[dict], mean: float, std: float,
              width: float, run_name: str, component_px: int) -> list[dict]:
    rows = []
    for item in test_items:
        prob = train.predict_full(model, item["image"], mean, std)
        raw = speckle.drop_small((prob >= 0.5) & item["fov"], component_px)
        truth = item["label"]
        b0_gt, b1_gt = metrics.betti(truth)
        for name, operation in OPERATIONS.items():
          for fraction in WIDTH_FRACTIONS:
            radius = int(round(fraction * width))
            pred = operation(raw, radius) & item["fov"]
            b0, b1 = metrics.betti(pred)
            rows.append({
                "run": run_name, "config": run_name.rsplit("_s", 1)[0],
                "image": item["name"], "operation": name,
                "width_fraction": fraction, "radius_px": radius,
                "dice": round(metrics.dice(pred, truth), 5),
                "cldice": round(metrics.cl_dice(pred, truth), 5),
                "betti0_err": abs(b0 - b0_gt), "betti1_err": abs(b1 - b1_gt),
                "betti1_signed": b1 - b1_gt,
            })
    return rows


def selftest() -> None:
    """A broken ring. Closing must reconnect it and bring the loop back --
    the exact thing E9 argued no post-processing could do."""
    canvas = np.zeros((40, 40), dtype=bool)
    canvas[10:25, 10:13] = True          # left wall
    canvas[10:25, 22:25] = True          # right wall
    canvas[10:13, 10:25] = True          # top
    canvas[22:25, 10:25] = True          # bottom
    assert metrics.betti(canvas) == (1, 1), metrics.betti(canvas)

    canvas[16:19, 22:25] = False         # cut the right wall: loop destroyed
    broken = metrics.betti(canvas)
    print(f"ring with a 3px gap: {broken}")
    assert broken == (1, 0), broken

    # Removal-type tools cannot bring it back, which is E9's argument.
    assert metrics.betti(hole_sweep.fill_small_holes(canvas, 400)) == (1, 0)
    assert metrics.betti(speckle.drop_small(canvas, 50)) == (1, 0)
    print("  filling holes and dropping components both leave it at betti1 0")

    # Closing can, and that is why this experiment exists. The radius has to
    # reach the gap: a 3 px gap needs radius 3, not 2. That relation is the
    # thing worth asserting -- it is what makes the sweep's x axis meaningful,
    # and it says the closing radius must be read against the size of the
    # breaks, not against the vessel width.
    restored = [radius for radius in (1, 2, 3, 4)
                if metrics.betti(bridge(canvas, radius))[1] == 1]
    for radius in (1, 2, 3, 4):
        b0, b1 = metrics.betti(bridge(canvas, radius))
        print(f"  closing radius {radius}: betti0 {b0}, betti1 {b1}")
    assert restored and min(restored) == 3, restored
    print("  a 3px gap needs radius 3: the radius must reach the gap")

    # The isolation itself: add a second component that owns a real hole, and
    # check that closing swallows the hole while bridge_only keeps it. Without
    # this, a null result from bridge_only could just mean the subtraction is
    # removing the welds too.
    canvas[30:37, 5:12] = True
    canvas[32:35, 7:10] = False          # a genuine hole, 9 px
    assert metrics.betti(canvas) == (2, 1), metrics.betti(canvas)
    closed = metrics.betti(bridge(canvas, 3))
    isolated = metrics.betti(bridge_only(canvas, 3))
    print(f"  ring-with-gap plus blob-with-hole: closing {closed}, "
          f"bridge_only {isolated}")
    # Closing welds the gap (+1 loop) and swallows the hole (-1): net 1.
    # bridge_only welds and keeps the hole: 2. Same betti0 either way.
    assert closed == (1, 1), closed
    assert isolated == (1, 2), isolated
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    dataset = sys.argv[1]
    if dataset == "drive":
        test_items, out_dir, default_runs = hole_sweep.setup_drive()
        stacked = train.stack_split("train")
        inside = stacked["images"][stacked["fovs"]]
        norm = (float(inside.mean()), float(inside.std()))
    else:
        _, test_items = cross_dataset.loader_for(dataset)()
        out_dir = cross_dataset.out_root(dataset)
        norm = None
        default_runs = [f"{loss}_s{seed}" for seed in cross_dataset.SEEDS
                        for loss in cross_dataset.LOSSES]
    width = cross_dataset.median_width(test_items)
    # Bridge on top of E4's component filter, so this measures what closing
    # ADDS to the post-processing we already know about rather than rediscovering
    # speckle removal.
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    runs = sys.argv[2:] or default_runs
    print(f"{dataset}: {len(test_items)} test images, width {width:.2f} px, "
          f"component filter {component_px} px, closing radii "
          f"{[int(round(f * width)) for f in WIDTH_FRACTIONS]} px", flush=True)

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
        mean, std = norm or (state["mean"], state["std"])
        rows += score_run(model, test_items, mean, std, width, run_name,
                          component_px)
        print(f"[{run_name}] done", flush=True)

    if not rows:
        raise SystemExit("no checkpoints found")
    out = out_dir / "bridge_sweep.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    print(f"\n=== {dataset}: closing on top of the component filter ===")
    print(f"{'radius w x':>11}{'px':>6}{'betti1':>10}{'signed':>9}"
          f"{'betti0':>10}{'dice':>9}{'cldice':>9}")
    for fraction in WIDTH_FRACTIONS:
        picked = [r for r in rows if r["width_fraction"] == fraction]
        print(f"{fraction:11.2f}{picked[0]['radius_px']:6d}"
              f"{np.mean([r['betti1_err'] for r in picked]):10.1f}"
              f"{np.mean([r['betti1_signed'] for r in picked]):9.1f}"
              f"{np.mean([r['betti0_err'] for r in picked]):10.1f}"
              f"{np.mean([r['dice'] for r in picked]):9.4f}"
              f"{np.mean([r['cldice'] for r in picked]):9.4f}")


if __name__ == "__main__":
    main()
