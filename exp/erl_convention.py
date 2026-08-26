"""Is the +36.6-point `intact` budget real, or a convention of erl.py?

erl.py decomposes the ground-truth skeleton into maximal runs lying inside one
predicted component. It labels `skel_gt & pred`, so ANY uncovered centreline
pixel splits the run -- including a run the prediction is connected AROUND,
which E10's classifier calls `intact` and which by construction does not
disconnect anything.

That matters because those runs carry 12.0% of the ground-truth centreline
against `severs`' 2.3%, and filling them lifts K_focal_aug from 47.4% to
84.0% traced. If a bridged gap should not split the run, most of that 36.6
points is a property of this implementation rather than of the segmentation,
and a method chasing it would be chasing the metric.

Two conventions, same predictions:

  split     erl.py as written. An uncovered centreline pixel ends the run.
  bridged   fragments separated by an `intact` gap are ONE fragment, since
            the prediction connects them. The gap's own length is not added
            back -- a tracer still has no foreground to follow there -- so
            this isolates the splitting rule and nothing else.

  python exp/erl_convention.py --selftest
  python exp/erl_convention.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import cross_dataset
import drive
import erl
import hole_sweep
import select_checkpoint as rules_module
import speckle
import summarize_selection as selection
import train

CONN8 = break_lengths.CONN8


def bridged_run_length(skel_gt: np.ndarray, pred: np.ndarray) -> float:
    """ERL where a gap the prediction connects around does not split the run."""
    total = int(skel_gt.sum())
    if total == 0:
        return 0.0
    pieces, _ = ndimage.label(pred, structure=CONN8)
    covered = skel_gt & (pieces > 0)
    labels, count = ndimage.label(covered, structure=CONN8)
    if count == 0:
        return 0.0
    parent = list(range(count + 1))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    missed = skel_gt & ~pred
    gaps, gap_count = ndimage.label(missed, structure=CONN8)
    for index, box in enumerate(ndimage.find_objects(gaps), start=1):
        grown = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
        pixels = gaps[grown] == index
        if break_lengths.classify(pixels, pred[grown], pieces[grown]) != "intact":
            continue
        touching = set(np.unique(labels[grown][
            ndimage.binary_dilation(pixels, CONN8)])) - {0}
        touching = sorted(touching)
        for other in touching[1:]:
            union(touching[0], other)

    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    merged = np.zeros(count + 1, dtype=np.float64)
    for fragment in range(1, count + 1):
        merged[find(fragment)] += sizes[fragment]
    return float((merged ** 2).sum() / total)


def selftest() -> None:
    # A straight vessel; the prediction runs one row over for its whole
    # length, so every centreline pixel is missed and the run is `intact`.
    skel = np.zeros((30, 100), dtype=bool)
    skel[15, 5:95] = True
    beside = np.zeros_like(skel)
    beside[16:19, 5:95] = True
    assert erl.expected_run_length(skel, beside) == 0.0
    assert bridged_run_length(skel, beside) == 0.0
    print("a prediction entirely off the centreline traces nothing under "
          "either convention -- there is no covered fragment to bridge")

    # Now a prediction that covers the vessel except for a bridged notch:
    # foreground detours around 10 px of centreline.
    detour = np.zeros_like(skel)
    detour[14:17, 5:95] = True
    detour[:, 45:55] = False
    detour[17:20, 44:56] = True          # the way around
    kinds = [k for _, _, _, k in break_lengths.break_runs(
        detour, skel, np.zeros_like(skel, dtype=int))]
    assert kinds == ["intact"], kinds
    split = erl.expected_run_length(skel, detour)
    bridged = bridged_run_length(skel, detour)
    print(f"a 10px detour around the centreline: split {split:.0f}, "
          f"bridged {bridged:.0f} (whole vessel {skel.sum()})")
    assert bridged > split, (split, bridged)

    # A real cut must NOT be bridged under either convention.
    cut = np.zeros_like(skel)
    cut[14:17, 5:95] = True
    cut[:, 45:55] = False
    kinds = [k for _, _, _, k in break_lengths.break_runs(
        cut, skel, np.zeros_like(skel, dtype=int))]
    assert kinds == ["severs"], kinds
    assert abs(bridged_run_length(skel, cut)
               - erl.expected_run_length(skel, cut)) < 1e-9
    print("a severing cut scores the same under both -- only bridged gaps "
          "are merged")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    points = selection.selection_points(selection.load())
    rule = dict(rules_module.rules())["(iv) best clDice"]
    items = drive.load_split("val")
    data = train.stack_split("train")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))

    print(f"{'arm':<14}{'split':>9}{'bridged':>10}{'change':>9}")
    print("-" * 42)
    for config in selection.ARMS:
        runs = sorted(r for r in points if r.rsplit("_s", 1)[0] == config)
        split_all, bridged_all = [], []
        for run in runs:
            epoch = rule(points[run])["epoch"]
            model = train.build_model(config)
            model.load_state_dict(train.load_checkpoint(
                selection.SWEEP / run / f"epoch{epoch:03d}.pt")["model"])
            model.eval()
            mean, std = train.normalisation(run, data)
            for item in items:
                if rules_module.is_selection_image(item["name"]):
                    continue
                skel = skeletonize(item["label"] & item["fov"])
                prob = train.predict_full(model, item["image"], mean, std)
                pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                          component_px)
                split_all.append(
                    erl.expected_run_length(skel, pred) / skel.sum())
                bridged_all.append(bridged_run_length(skel, pred) / skel.sum())
        split, bridged = np.mean(split_all), np.mean(bridged_all)
        print(f"{config:<14}{split:8.1%}{bridged:9.1%}{bridged - split:+8.1%}",
              flush=True)
    print()
    print("If `bridged` is close to `split`, the 36.6-point intact budget is")
    print("real geometry. If it closes most of the gap to 84.0%, that budget")
    print("is this implementation's splitting rule and not the segmentation.")


if __name__ == "__main__":
    main()
