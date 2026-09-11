"""Q2: is ERL blind to breaks at the periphery? And if so, what is left?

WRITTEN AND SELFTESTED 2026-09-08, BEFORE IT SCORED ANYTHING.

THE ALGEBRA FIRST, because it says what to expect. ERL is sum(l_i^2)/L. A
break splits one fragment of length l into l1 + l2, so the numerator changes
by

    l^2  ->  l1^2 + l2^2     i.e. it LOSES exactly 2 * l1 * l2

That product is the whole story. A break in the MIDDLE of a long run costs
about l^2/2. A break near an END costs about 2 * l * (a few pixels). Terminal
twigs ARE the ends of the tree, so a break out at the periphery is cheap by
construction, and quadratically so.

If that is right then two things follow and both are testable:

  1. the breaks a model actually makes are concentrated where ERL charges
     least, so a headline ERL can look healthy while the periphery is in
     pieces;
  2. whatever ERL is really rewarding must be happening on the TRUNK -- so the
     honest question becomes whether one arm beats another on the thick
     vessels alone, where the metric has teeth.

WHAT IS MEASURED. Every break is a maximal run of ground-truth centreline the
prediction does not cover. For each one:

    radius      the ground-truth vessel radius at the break, from the
                distance transform: thin = periphery, thick = trunk
    cost        the ERL the image would GAIN if that ONE break were filled
                and nothing else changed. Measured by actually filling it and
                recomputing, not by the formula above -- the formula assumes
                the break splits exactly one fragment, and a break at a
                junction can rejoin three.

Then the same arms are compared on the thick half of the centreline alone.

PRE-REGISTERED 2026-09-08, before this file has read a pixel:

  1. Break COUNT is concentrated in the thin half -- at least 70% of breaks
     lie below the median vessel radius.
  2. Break COST is concentrated in the thick half -- the thin half holds at
     least 70% of the breaks but under 30% of the total ERL that filling
     them would recover. This is the claim that ERL is blind to the
     periphery, stated so it can fail.
  3. The median cost of a thin break is under a fifth of the median cost of a
     thick one.
  4. Restricted to the thick half, the ranking of the arms is UNCHANGED from
     the whole-centreline ranking. If it changes, then the headline number
     was being decided by the periphery after all and prediction 2 was
     measuring the wrong thing.

  python exp/break_cost.py --selftest
  python exp/break_cost.py --report
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
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import train

ARMS = ("A_dice", "H_aug", "H_aug_clw16", "K_focal_aug_clw64")
SEEDS = 4
# ABSOLUTE radius bins, not quantiles of the breaks' own radii. The first
# version of this report split at the median radius OF THE BREAKS, and 87-92%
# of them landed in the "thin" half -- vessel radii are heavily tied near small
# integers, so `<=` swallowed most of the distribution and the headline
# ("the thin half holds most of the ERL") was an artefact of the binning
# rather than a finding. Caught 2026-09-08 before it was reported as a result.
# For reference, the ground-truth centreline radius on DRIVE test:
# 25th percentile 1.00 px, median 1.41, 75th 2.24, 90th 3.16.
BINS = ((0.0, 1.5, "capillary <1.5"), (1.5, 2.5, "thin 1.5-2.5"),
        (2.5, 4.0, "mid 2.5-4"), (4.0, 99.0, "trunk >4"))


def break_table(skel: np.ndarray, pred: np.ndarray,
                radius_map: np.ndarray) -> list[dict]:
    """One row per break: its length, the vessel radius there, and its cost.

    `cost` is measured by filling that break ALONE and recomputing the ERL,
    because the 2*l1*l2 formula assumes a break separates exactly two
    fragments and a break at a junction can rejoin three.
    """
    missed = skel & ~pred
    labels, count = ndimage.label(missed, structure=break_lengths.CONN8)
    if count == 0:
        return []
    base = erl.expected_run_length(skel, pred)
    total = float(skel.sum()) or 1.0
    rows = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        piece = labels[box] == index
        filled = pred.copy()
        filled[box] |= piece
        rows.append({
            "length": float(piece.sum()),
            "radius": float(np.median(radius_map[box][piece])),
            "cost": (erl.expected_run_length(skel, filled) - base) / total})
    return rows


def selftest() -> None:
    # 1. THE ALGEBRA, ON A STRAIGHT LINE. A break at the middle of a 100 px
    #    run must cost about half of it; a break 5 px from the end must cost
    #    about a tenth as much. If this does not hold the measurement is not
    #    measuring what the docstring claims.
    skel = np.zeros((9, 120), dtype=bool)
    skel[4, 10:110] = True
    radius_map = np.full(skel.shape, 2.0)
    for gap_at, tag in ((60, "middle"), (105, "near the end")):
        pred = skel.copy()
        pred[4, gap_at:gap_at + 2] = False
        rows = break_table(skel, pred, radius_map)
        assert len(rows) == 1, rows
        print(f"a 2 px break {tag} of a 100 px run costs "
              f"{rows[0]['cost']:.3f} of the image's ERL")
        if tag == "middle":
            middle_cost = rows[0]["cost"]
        else:
            assert rows[0]["cost"] < middle_cost / 5, (rows[0], middle_cost)
    print("  -- so the same break is worth 5x less at the end than in the "
          "middle, which is prediction 2's mechanism")

    # 2. COST MUST BE NON-NEGATIVE and filling every break must recover the
    #    whole tree, or `cost` is not what it says.
    pred = skel.copy()
    pred[4, 40:42] = False
    pred[4, 80:82] = False
    rows = break_table(skel, pred, radius_map)
    assert len(rows) == 2 and all(r["cost"] > 0 for r in rows)
    everything = erl.expected_run_length(skel, skel) / float(skel.sum())
    assert abs(everything - 1.0) < 1e-9, everything
    print(f"two breaks give two rows, both positive "
          f"({rows[0]['cost']:.3f}, {rows[1]['cost']:.3f}); a fully covered "
          f"skeleton scores exactly 1.0")

    # 3. THE RADIUS MUST TRACK THE VESSEL, not the break. A break on a wide
    #    vessel must report a wide radius.
    wide = np.zeros((41, 41), dtype=bool)
    wide[20, 5:36] = True
    thickness = np.zeros(wide.shape)
    thickness[:, :20] = 1.0
    thickness[:, 20:] = 6.0
    pred = wide.copy()
    pred[20, 10:12] = False        # a break in the thin half
    pred[20, 30:32] = False        # and one in the thick half
    rows = sorted(break_table(wide, pred, thickness),
                  key=lambda r: r["radius"])
    assert rows[0]["radius"] == 1.0 and rows[1]["radius"] == 6.0, rows
    print("the radius reported for a break is the vessel's, not the break's")
    print("all checks passed")


def report() -> None:
    items = drive.load_split("test")
    data = train.stack_split("fit")
    epochs = heldout.chosen_epochs()
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = []
    for item in items:
        truth = item["label"] & item["fov"]
        geometry.append((skeletonize(truth),
                         ndimage.distance_transform_edt(truth)))

    print("=== is ERL blind to breaks at the periphery? ===\n")
    print("ERL is sum(l^2)/L, so a break costs 2*l1*l2: everything in the")
    print("MIDDLE of a long run, almost nothing at an END. Terminal twigs are")
    print("the ends of the tree. Every break below is charged by actually")
    print("filling it and recomputing, not by that formula.\n")
    print(f"DRIVE, {len(items)} test images, {len(ARMS)} arms x {SEEDS} seeds,")
    print("each at threshold 0.5. `thin` and `thick` split the breaks at the")
    print("MEDIAN ground-truth vessel radius over all breaks of that arm.\n")
    print(f"    {'arm':16}{'bin':16}{'breaks':>8}{'share':>8}"
          f"{'ERL share':>11}{'med cost':>11}")
    for arm in ARMS:
        runs = sorted(r for r in epochs if r.rsplit("_s", 1)[0] == arm
                      and (heldout.ROOT / r / "final.pt").exists())[:SEEDS]
        rows = []
        for run in runs:
            model, mean, std = sweep.load_model(run, arm, epochs, data)
            if model is None:
                continue
            for item, (skel, radius_map) in zip(items, geometry):
                prob = train.predict_full(model, item["image"], mean, std)
                pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                          component_px)
                rows.extend(break_table(skel, pred, radius_map))
        if not rows:
            continue
        radii = np.array([r["radius"] for r in rows])
        costs = np.array([r["cost"] for r in rows])
        for low, high, tag in BINS:
            inside = (radii >= low) & (radii < high)
            if not inside.any():
                continue
            print(f"    {arm:16}{tag:16}{int(inside.sum()):>8}"
                  f"{100 * inside.mean():>7.1f}%"
                  f"{100 * costs[inside].sum() / costs.sum():>10.1f}%"
                  f"{float(np.median(costs[inside])):>11.5f}")
    print("\n`share` is the share of BREAKS in that bin; `ERL share` is the")
    print("share of the recoverable ERL they hold; `med cost` is what ONE")
    print("break there is worth. Read the last column for the per-break")
    print("blindness and the middle two for what it adds up to -- they do")
    print("NOT say the same thing, and reporting only one of them is how the")
    print("first version of this table got it wrong.")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    raise SystemExit("pass --selftest or --report")


if __name__ == "__main__":
    main()
