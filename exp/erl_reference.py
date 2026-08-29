"""Does exp/erl.py compute the ERL the field's reference implementation does?

WRITTEN 2026-08-28. The question is not "is our number good" but "does our
number mean what a reader will take it to mean". Every ERL in this repo is
compared against published ERLs by readers who assume one definition.

THE REFERENCE is Allen Institute's `segmentation-skeleton-metrics` (v5.9.3,
PyPI), the implementation the connectomics ERL literature uses. Its ERL is
skeleton_metrics.py:643:

    wgts, run_lengths = [], []
    for label in graph.node_labels():              # node_labels() DISCARDS "0"
        run_length = graph.run_length_from(nodes[0])
        wgts.append(run_length)
        run_lengths.append(0 if label in graph.labels_with_merge
                           else run_length)
    return np.average(run_lengths, weights=wgts)

With no merges that is sum(l_i^2) / sum(l_i). Ours (erl.py:71) is

    sum(l_i^2) / skel_gt.sum()

THE THREE DIFFERENCES, and which way each pushes the number:

  1. THE DENOMINATOR. Theirs sums the COVERED fragments; label "0" -- the
     ground truth the prediction never found -- is discarded. Ours divides by
     the whole ground-truth skeleton. So a vessel the model misses entirely
     costs us and is invisible to them. Ours is the smaller and the stricter
     number, and the gap is exactly the coverage fraction.
  2. MERGES. A predicted component spanning two different ground-truth
     objects contributes ZERO to theirs. We have no merge penalty at all,
     because a retinal image is one connected vessel tree: there are no two
     objects to merge. Reported as measured, not assumed.
  3. SPLIT vs BRIDGED. Ours ends a run at any uncovered centreline pixel,
     including one the prediction connects around. erl_convention.py already
     measures that; it is named here so the three are not confused.

WHAT THIS SCRIPT DOES. It transcribes the reference formula and runs both on
the same predictions. It does NOT run the reference package itself: that
package consumes SWC skeletons and label volumes through a graph pipeline
built for 3D connectomics, and porting DRIVE into it would put the port, not
the metric, under test. The formula is short enough to transcribe and the
transcription is asserted against hand-computed cases below.

  python exp/erl_reference.py --selftest
  python exp/erl_reference.py
"""
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import drive
import erl
import erl_convention
import hole_sweep
import select_checkpoint as rules_module
import speckle
import summarize_selection as selection
import train


def reference_erl(skel_gt: np.ndarray, pred: np.ndarray) -> float:
    """The Allen formula: run length weighted by run length, over COVERED
    ground truth only.

    Same fragment decomposition as erl.py -- only the denominator differs, so
    any gap between the two is the denominator and nothing else.
    """
    lengths = erl.fragments(skel_gt, pred).astype(np.float64)
    if lengths.sum() == 0:
        return 0.0
    return float((lengths ** 2).sum() / lengths.sum())


def coverage(skel_gt: np.ndarray, pred: np.ndarray) -> float:
    """The whole difference between the two conventions, in one number."""
    total = int(skel_gt.sum())
    if total == 0:
        return 0.0
    return float(erl.fragments(skel_gt, pred).sum()) / total


def selftest() -> None:
    skel = np.zeros((20, 100), dtype=bool)
    skel[10, 10:90] = True                      # 80 px of centreline
    total = int(skel.sum())
    assert total == 80

    # 1. Full coverage: the denominators are the same number, so the two
    #    conventions MUST agree. If they ever disagree here, the difference
    #    is not the denominator and this file's whole claim is wrong.
    full = np.zeros_like(skel)
    full[9:12, 10:90] = True
    assert abs(erl.expected_run_length(skel, full) - 80.0) < 1e-9
    assert abs(reference_erl(skel, full) - 80.0) < 1e-9
    assert abs(coverage(skel, full) - 1.0) < 1e-9
    print("  full coverage: ours 80.0, reference 80.0 -- identical")

    # 2. A two-pixel cut. 39 + 39 covered of 80. Hand-computed:
    #      ours       (39^2 + 39^2) / 80      = 38.025
    #      reference  (39^2 + 39^2) / (39+39) = 39.0
    cut = full.copy()
    cut[:, 49:51] = False
    ours, theirs = erl.expected_run_length(skel, cut), reference_erl(skel, cut)
    assert abs(ours - (39 ** 2 * 2) / 80) < 1e-9, ours
    assert abs(theirs - (39 ** 2 * 2) / 78) < 1e-9, theirs
    assert theirs > ours, (theirs, ours)
    print(f"  one 2 px cut:  ours {ours:.3f}, reference {theirs:.3f} "
          f"(coverage {coverage(skel, cut):.3f})")

    # 3. The case that separates them hardest: half the vessel never found.
    #    The reference cannot see the missing half at all.
    half = np.zeros_like(skel)
    half[9:12, 10:50] = True
    ours, theirs = erl.expected_run_length(skel, half), reference_erl(skel, half)
    assert abs(theirs - 40.0) < 1e-9, theirs
    assert abs(ours - 20.0) < 1e-9, ours
    print(f"  half not found: ours {ours:.1f}, reference {theirs:.1f} "
          f"-- the reference is blind to omitted ground truth")

    # 4. And the exact algebraic relation, which is what lets any number in
    #    this repo be converted to the other convention after the fact.
    for pred in (full, cut, half):
        exact = reference_erl(skel, pred) * coverage(skel, pred)
        assert abs(exact - erl.expected_run_length(skel, pred)) < 1e-9
    print("  ours == reference x coverage, exactly, on every case")
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

    print("ERL under three conventions, rule (iv), report half.\n")
    header = (f"  {'arm':<16}{'ours':>8}{'reference':>11}{'bridged':>9}"
              f"{'coverage':>10}  runs")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config in selection.ARMS:
        runs = sorted(r for r in points if r.rsplit("_s", 1)[0] == config)
        ours, theirs, bridged, covered = [], [], [], []
        for run in runs:
            epoch = rule(points[run])["epoch"]
            weights = selection.SWEEP / run / f"epoch{epoch:03d}.pt"
            if not weights.exists():
                continue
            model = train.build_model(config)
            model.load_state_dict(train.load_checkpoint(weights)["model"])
            model.eval()
            mean, std = train.normalisation(run, data)
            for item in items:
                if rules_module.is_selection_image(item["name"]):
                    continue
                skel = skeletonize(item["label"] & item["fov"])
                prob = train.predict_full(model, item["image"], mean, std)
                pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                          component_px)
                length = skel.sum()
                ours.append(erl.expected_run_length(skel, pred) / length)
                theirs.append(reference_erl(skel, pred) / length)
                bridged.append(
                    erl_convention.bridged_run_length(skel, pred) / length)
                covered.append(coverage(skel, pred))
        if not ours:
            print(f"  {config:<16}{'no checkpoints':>38}")
            continue
        print(f"  {config:<16}{np.mean(ours):7.1%}{np.mean(theirs):11.1%}"
              f"{np.mean(bridged):9.1%}{np.mean(covered):10.1%}  {len(runs)}",
              flush=True)
    print()
    print("  ours = reference x coverage, exactly. The reference discards the")
    print("  ground truth the prediction never found; we divide by all of it.")
    print("  Both are defensible. Only one of them is what a reader comparing")
    print("  against a published ERL will assume, so the paper has to say")
    print("  which, and report the coverage column beside it.")


if __name__ == "__main__":
    main()
