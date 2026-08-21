"""Why component filtering cannot fix betti-1, and what that means for E4.

E4 reported that dropping small predicted components buys 2.4-2.6x what the
topology loss buys on betti-0, and then that betti-1 reverses: filtering does
not move it at all on HRF while the loss buys 15x what filtering buys. That
reversal was measured and left unexplained.

It is not a property of retinas or of mortar. It follows from what a connected
component IS. Removing one changes the component count by exactly one and the
loop count by however many loops that component contained -- which for a speckle
blob is zero. So filtering can only ever touch betti-0 unless the noise it
removes is ring-shaped, and small false-positive blobs almost never are.

That is a strong claim, so it gets checked two ways here:

  1. mechanism, on shapes with known Betti numbers (main() --selftest);
  2. the prediction it makes about the E4 sweep, read off the existing CSVs:
     HRF (tree-shaped vessels, speckle is blobs) should be pinned at zero
     movement, TopoMortar (a grid, so fragments can close a loop) should move
     a little and only at the largest filters.

The consequence is the part worth carrying forward: betti-1 error cannot be
post-processed away, so the loop half of topology is a model-level problem by
construction, not by accident.

  python exp/analyse_betti1.py             # the sweep table
  python exp/analyse_betti1.py --selftest  # the mechanism, no data needed
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
import speckle

RESULTS = Path(__file__).resolve().parent / "results" / "cross"
DATASETS = ("hrf", "topomortar")


def sweep(dataset: str) -> list[dict]:
    rows = []
    for path in (RESULTS / dataset).glob("*/scores.csv"):
        config = path.parent.name.rsplit("_s", 1)[0]
        for row in csv.DictReader(path.open()):
            rows.append({"config": config,
                         "multiple": float(row["width_multiple"]),
                         "min_size": int(row["min_size"]),
                         "betti0_err": float(row["betti0_err"]),
                         "betti1_err": float(row["betti1_err"]),
                         "dice": float(row["dice"])})
    return rows


def per_curve(dataset: str) -> None:
    """The sharp version of the claim, one curve per (run, image).

    An average of zero could be two large opposite effects cancelling, which is
    the mistake this whole series keeps finding elsewhere. So count how many
    individual curves move at all, and which way. Blob-shaped noise predicts
    "mostly pinned, and the few that move split evenly"; ring-shaped noise
    predicts "moves often, and almost always for the better".
    """
    curves = {}
    for path in (RESULTS / dataset).glob("*/scores.csv"):
        for row in csv.DictReader(path.open()):
            key = (path.parent.name, row["image"])
            curves.setdefault(key, {})[float(row["width_multiple"])] = (
                float(row["betti0_err"]), float(row["betti1_err"]))

    moved = {0: 0, 1: 0}
    better = worse = 0
    for sweep_by_multiple in curves.values():
        multiples = sorted(sweep_by_multiple)
        for index in (0, 1):
            values = [sweep_by_multiple[m][index] for m in multiples]
            if len(set(values)) > 1:
                moved[index] += 1
                if index == 1:
                    better += values[-1] < values[0]
                    worse += values[-1] > values[0]
    total = len(curves)
    print(f"  per (run, image) curve, n={total}: "
          f"betti0 moves on {moved[0]}, betti1 moves on {moved[1]}")
    print(f"  of the betti1 curves that move: {better} better, {worse} worse")


def report(dataset: str) -> None:
    rows = sweep(dataset)
    if not rows:
        print(f"{dataset}: no scores yet")
        return
    print(f"\n=== {dataset}: what the filter sweep moves ===")
    print(f"{'w^2 x':>8}{'px':>7}{'betti0':>9}{'betti1':>9}{'dice':>9}"
          f"{'b0 change':>12}{'b1 change':>12}")
    base = None
    for multiple in sorted({r["multiple"] for r in rows}):
        picked = [r for r in rows if r["multiple"] == multiple]
        means = {k: float(np.mean([r[k] for r in picked]))
                 for k in ("betti0_err", "betti1_err", "dice")}
        base = base or means
        print(f"{multiple:8.2f}{picked[0]['min_size']:7d}"
              f"{means['betti0_err']:9.1f}{means['betti1_err']:9.1f}"
              f"{means['dice']:9.4f}"
              f"{100 * (means['betti0_err'] / base['betti0_err'] - 1):11.1f}%"
              f"{100 * (means['betti1_err'] / base['betti1_err'] - 1):11.1f}%")
    per_curve(dataset)


def selftest() -> None:
    """A blob and a ring, same size, filtered at the same threshold.

    metrics.betti is the same counter E4 used, so this measures the claim in
    the units the claim is made in rather than restating it in prose.
    """
    canvas = np.zeros((60, 60), dtype=bool)
    # One large structure that must survive filtering, so the counts below are
    # "the real structure plus the noise" exactly as they are in a prediction.
    canvas[5:55, 28:32] = True
    b0_clean, b1_clean = metrics.betti(canvas)
    assert (b0_clean, b1_clean) == (1, 0), (b0_clean, b1_clean)

    with_blob = canvas.copy()
    with_blob[8:12, 8:12] = True                 # 16 px solid square
    with_ring = canvas.copy()
    with_ring[8:15, 8:15] = True                 # 49 px square...
    with_ring[10:13, 10:13] = False              # ...with a 9 px hole

    b0_blob, b1_blob = metrics.betti(with_blob)
    b0_ring, b1_ring = metrics.betti(with_ring)
    print(f"clean {b0_clean},{b1_clean}   "
          f"+blob {b0_blob},{b1_blob}   +ring {b0_ring},{b1_ring}")
    # Both kinds of noise cost one component; only the ring costs a loop.
    assert (b0_blob, b1_blob) == (2, 0), (b0_blob, b1_blob)
    assert (b0_ring, b1_ring) == (2, 1), (b0_ring, b1_ring)

    for name, noisy, expect_b1 in (("blob", with_blob, 0), ("ring", with_ring, 1)):
        filtered = speckle.drop_small(noisy, 60)
        b0, b1 = metrics.betti(filtered)
        recovered_b1 = expect_b1 - b1
        print(f"  filter <60px removes the {name}: "
              f"betti0 {b0_blob if name == 'blob' else b0_ring} -> {b0}, "
              f"betti1 recovered {recovered_b1}")
        assert b0 == 1, (name, b0)
        assert b1 == 0, (name, b1)

    # The asymmetry, stated as the number the E4 sweep should show: filtering
    # recovers a loop only from ring-shaped noise, so a dataset whose false
    # positives are blobs sees exactly zero betti-1 movement.
    print("mechanism: filtering always recovers 1 component, and recovers a "
          "loop only when the removed noise was a ring")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    for dataset in DATASETS:
        report(dataset)
    print("\nPrediction under test: betti-1 is pinned wherever false positives "
          "are blobs (HRF) and moves only under the largest filters where the "
          "structure itself is a grid (TopoMortar). Run --selftest for why.")


if __name__ == "__main__":
    main()
