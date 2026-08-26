"""E11: which breaks does bridging actually close?

E9b measured that bridge_only recovers 9-30% of the loop deficit and then
explained its own ceiling like this:

    "the rest are gaps wider than two vessel widths, or vessels that were
     never detected at all"

That sentence was never measured, and it reasons from break LENGTH -- the
exact quantity E10 refuted one report later. E10 predicted from the length
distribution that radius 6 should close 76% of breaks and measured 14.9%, a
factor of five. So E9b's ceiling explanation rests on the reasoning that
already failed once, in the same series, about the same operation.

This script measures it instead, using E10's validated classification. The
mechanistic prediction is pre-registered here, before the run:

  P1  Bridging is SELECTIVE for severing breaks. A `severs` break has predicted
      structure on both sides by definition, which is exactly what a closing
      element needs to weld to; an `absent` break has nothing within reach at
      any radius. So the resolution rate on `severs` should exceed the 14.9%
      all-break closure rate E10 measured at radius 6.
  P2  The severing breaks resolved per image should account for the betti-0
      improvement E9b reported, to within a factor of two. If bridging fixes
      betti-0 by some other route, the two numbers will not line up and P1
      being true would still not explain the metric.
  P3  Kind predicts resolution better than length does. This is the direct
      test of E9b's stated ceiling. If length wins, E9b was right by accident
      and E10 does not generalise to the bridging question.

A third predictor joined the comparison because the selftest below refused to
pass without it. Closing cannot weld a gap in a ONE PIXEL wide line at any
radius: the dilation leaves a waist above and below the gap, and the erosion
eats back through it. So weldability depends on how thick the predicted vessel
is on either side, not only on how long the gap is -- and thin vessels are
exactly where the model's breaks are (E2: 81% of breaks are in the dimmest
contrast quartile). That is a better candidate for E9b's ceiling than either
of the two the report considered, so local thickness is recorded per break and
P3 is judged three ways.

Inference only, on the same 12 DRIVE checkpoints and on top of the same E4
component filter, so the numbers are comparable to E9b and E10 line by line.

  python exp/bridge_anatomy.py
  python exp/bridge_anatomy.py --selftest

Writes results/bridge_anatomy.csv, aggregated per
(run, image, radius, kind_before, length bucket).
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import bridge_sweep
import cross_dataset
import hole_sweep
import metrics
import speckle
import train

torch.set_num_threads(2)  # the trainer already holds six; see CLAUDE.md

RESULTS = Path(__file__).resolve().parent / "results"

# E9b's sweep in the same width-relative units: 0.25 w, 0.75 w, 1.0 w, 2.0 w on
# DRIVE's 3.06 px median width. 2 px is the "near free" radius E9b recommends
# and 6 px is the one that costs 0.045 Dice, so the accounting check in P2 has
# both ends of E9b's trade-off to land on.
RADII = (1, 2, 3, 6)

# Buckets, not raw lengths, because P3 only needs length to be given a fair
# chance as a predictor -- one row per break would be a 15 MB CSV to answer a
# question that four bins answer.
BUCKETS = ((1, 1), (2, 2), (3, 6), (7, 12), (13, 10 ** 6))


# Diameter of the predicted vessel beside the break, in pixels. DRIVE's median
# structure width is about 3, so "<=2" is a vessel thinner than typical and
# "6+" is an arcade trunk.
THICKNESS_BINS = ((0.0, "none"), (2.0, "<=2"), (3.0, "3"), (5.0, "4-5"),
                  (float("inf"), "6+"))


def bucket_of(length: int) -> str:
    for low, high in BUCKETS:
        if low <= length <= high:
            return f"{low}-{high}" if high < 10 ** 6 else f"{low}+"
    raise ValueError(length)


def thickness_bin(diameter: float) -> str:
    for edge, name in THICKNESS_BINS:
        if diameter <= edge:
            return name
    raise ValueError(diameter)


def anatomy(pred: np.ndarray, skel_gt: np.ndarray, fov: np.ndarray,
            radii: tuple[int, ...]) -> list[dict]:
    """Every break, classified before bridging and re-classified after each
    radius.

    "Resolved" means the break no longer severs: either the weld covered it or
    the two pieces it sat between became one component. That is the event that
    can move betti-0, which is why it is the thing counted rather than "the
    missed pixels are now predicted" -- a break can be partially covered and
    still sever, and can be left uncovered while the structure reconnects
    around it.
    """
    missed = skel_gt & ~pred
    labels, count = ndimage.label(missed, structure=break_lengths.CONN8)
    if count == 0:
        return []
    pieces = ndimage.label(pred, structure=break_lengths.CONN8)[0]
    # Distance to the nearest background pixel is the local radius, so twice it
    # less the centre pixel is the diameter of the vessel the break sits in.
    radius_map = ndimage.distance_transform_edt(pred)
    after = {}
    for radius in radii:
        bridged = bridge_sweep.bridge_only(pred, radius) & fov
        after[radius] = (bridged,
                         ndimage.label(bridged,
                                       structure=break_lengths.CONN8)[0])

    rows = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        # +3 rather than E10's +1 so the thickness probe below has room. It
        # cannot change the classification: classify dilates the run by one
        # either way and the labels come from the full image.
        grown = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in box)
        pixels = labels[grown] == index
        # Probe the vessel a few pixels back from the break rather than at its
        # lip: a pixel on the lip is adjacent to the break's own background, so
        # measuring there reports every vessel as 1 px wide no matter how thick
        # it is. The selftest catches this.
        beside = ndimage.binary_dilation(
            pixels, break_lengths.CONN8, iterations=3) & pred[grown]
        row = {"length": int(pixels.sum()),
               "kind": break_lengths.classify(pixels, pred[grown],
                                              pieces[grown]),
               "diameter": (2 * radius_map[grown][beside].max() - 1
                            if beside.any() else 0.0)}
        for radius in radii:
            bridged, bridged_pieces = after[radius]
            row[f"kind_{radius}"] = break_lengths.classify(
                pixels, bridged[grown], bridged_pieces[grown])
            row[f"covered_{radius}"] = bool(bridged[grown][pixels].all())
        rows.append(row)
    return rows


def selftest() -> None:
    """The three kinds, each given the radius it would need, plus the
    thickness effect that this test discovered.

    P1's mechanism is that a severing break has structure on both sides to
    weld to while an absent one has nothing at any radius. If that is wrong the
    whole experiment is measuring noise, so it is asserted rather than assumed.
    """
    skel = np.zeros((40, 70), dtype=bool)
    skel[10, 2:68] = True            # vessel A, 3 px thick, cleanly cut
    skel[20, 2:68] = True            # vessel B, never predicted at all
    skel[30, 2:68] = True            # vessel C, predicted offset by one row
    fov = np.ones_like(skel)

    pred = np.zeros_like(skel)
    pred[9:12, 2:68] = True
    pred[9:12, 25:28] = False        # a 3 px cut through a 3 px wide vessel
    pred[31:34, 2:68] = True         # covers C, offset: intact

    found = {row["kind"]: row for row in anatomy(pred, skel, fov, RADII)}
    print(f"kinds found: {sorted(found)}")
    assert set(found) == {"severs", "intact", "absent"}, sorted(found)

    severing = found["severs"]
    assert severing["length"] == 3, severing
    print(f"  severs (3px cut, {severing['diameter']:.0f}px vessel): "
          f"{[(r, severing[f'kind_{r}']) for r in RADII]}")
    # Radius must reach the gap, as E9b's own selftest established; once it
    # does, the break stops severing.
    assert severing["kind_1"] == "severs", severing
    assert severing["kind_3"] == "intact", severing

    absent = found["absent"]
    print(f"  absent (length {absent['length']}): "
          f"{[(r, absent[f'kind_{r}']) for r in RADII]}")
    assert all(absent[f"kind_{r}"] == "absent" for r in RADII), absent
    print("  -> a break with nothing to weld to is unreachable at every radius")

    # A long break can be trivially resolvable and a short one unreachable, so
    # length on its own cannot rank them. This is P3 in miniature.
    intact = found["intact"]
    assert intact["length"] > severing["length"], (intact, severing)
    print(f"  the {intact['length']}px intact break never severed while the "
          f"{severing['length']}px one did: length does not rank these")

    # The thickness effect, which is why `diameter` is recorded at all. The
    # SAME 3 px gap in a 1 px wide line survives every radius: dilation leaves
    # a waist above and below the gap and the erosion eats back through it.
    # E9b's selftest used 3 px thick walls and so never saw this.
    lone = np.zeros_like(skel)
    lone[10, 2:68] = True
    thin = lone.copy()
    thin[10, 25:28] = False
    cut = [row for row in anatomy(thin, lone, fov, RADII)
           if row["kind"] == "severs"]
    assert len(cut) == 1, only
    print(f"  the same 3px gap in a 1px wide line: "
          f"{[(r, cut[0][f'kind_{r}']) for r in RADII]}")
    assert all(cut[0][f"kind_{r}"] == "severs" for r in RADII), cut
    print("  -> weldability depends on vessel thickness, not gap length alone")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    test_items, out_dir, default_runs = hole_sweep.setup_drive()
    stacked = train.stack_split("train")
    inside = stacked["images"][stacked["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())
    width = cross_dataset.median_width(test_items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    skeletons = [skeletonize(item["label"] & item["fov"])
                 for item in test_items]

    runs = sys.argv[1:] or default_runs
    print(f"drive: {len(test_items)} images, width {width:.2f} px, "
          f"component filter {component_px} px, radii {list(RADII)} px",
          flush=True)

    rows = []
    for run_name in runs:
        # Per run, because a LIOT run needs its own constants; see
        # train.normalisation. predict_full raises if they disagree.
        mean, std = train.normalisation(run_name, stacked)
        weights = out_dir / run_name / "final.pt"
        if not weights.exists():
            print(f"[{run_name}] no checkpoint, skipping", flush=True)
            continue
        model = train.build_model(run_name.rsplit("_s", 1)[0])
        model.load_state_dict(train.load_checkpoint(weights)["model"])
        model.eval()
        for item, skel in zip(test_items, skeletons):
            prob = train.predict_full(model, item["image"], mean, std)
            pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                      component_px)
            breaks_here = anatomy(pred, skel, item["fov"], RADII)
            truth = item["label"] & item["fov"]
            b0_gt, b1_gt = metrics.betti(truth)
            b0_raw, b1_raw = metrics.betti(pred)
            for radius in RADII:
                bridged = bridge_sweep.bridge_only(pred, radius) & item["fov"]
                b0, b1 = metrics.betti(bridged)
                grouped = Counter(
                    (row["kind"], bucket_of(row["length"]),
                     thickness_bin(row["diameter"]),
                     row[f"kind_{radius}"], row[f"covered_{radius}"])
                    for row in breaks_here)
                for key, n in grouped.items():
                    kind, bucket, thick, kind_after, covered = key
                    rows.append({
                        "run": run_name,
                        "config": run_name.rsplit("_s", 1)[0],
                        "image": item["name"], "radius": radius,
                        "kind_before": kind, "bucket": bucket,
                        "thickness": thick,
                        "kind_after": kind_after, "covered": int(covered),
                        "n": n,
                        "betti0_err_raw": abs(b0_raw - b0_gt),
                        "betti0_err_bridged": abs(b0 - b0_gt),
                        "betti1_err_raw": abs(b1_raw - b1_gt),
                        "betti1_err_bridged": abs(b1 - b1_gt),
                    })
        print(f"[{run_name}] done", flush=True)

    if not rows:
        raise SystemExit("no checkpoints found")
    out = RESULTS / "bridge_anatomy.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    report(rows)


def total(rows: list[dict]) -> int:
    return sum(row["n"] for row in rows)


def report(rows: list[dict]) -> None:
    images = len({(row["run"], row["image"]) for row in rows})
    at_radius = {radius: [r for r in rows if r["radius"] == radius]
                 for radius in RADII}
    base = at_radius[RADII[0]]

    print("\n=== P1: is bridging selective for severing breaks? ===")
    print(f"{'kind':>10}{'breaks':>9}{'share':>8}"
          + "".join(f"{'r=' + str(r):>10}" for r in RADII))
    print(f"{'':>27}{'resolved (no longer severs)':>40}")
    for kind in ("severs", "intact", "absent"):
        picked = [r for r in base if r["kind_before"] == kind]
        count = total(picked)
        line = f"{kind:>10}{count:9d}{100 * count / total(base):7.1f}%"
        for radius in RADII:
            here = [r for r in at_radius[radius] if r["kind_before"] == kind]
            fixed = total([r for r in here if r["kind_after"] != "severs"])
            was = total(here)
            line += f"{100 * fixed / max(was, 1):9.1f}%"
        print(line)

    print("\n=== P2: does that account for the betti-0 improvement? ===")
    print(f"{'radius':>7}{'severs/img':>12}{'resolved/img':>14}"
          f"{'betti0 raw':>12}{'betti0 br':>11}{'delta':>8}{'ratio':>8}")
    for radius in RADII:
        here = at_radius[radius]
        severing = [r for r in here if r["kind_before"] == "severs"]
        resolved = total([r for r in severing if r["kind_after"] != "severs"])
        # betti errors are per (run, image) but repeated on every row, so
        # average over the distinct pairs rather than over rows.
        seen = {}
        for row in here:
            seen[(row["run"], row["image"])] = (row["betti0_err_raw"],
                                                row["betti0_err_bridged"])
        raw = np.mean([v[0] for v in seen.values()])
        bridged = np.mean([v[1] for v in seen.values()])
        per_image = resolved / images
        delta = raw - bridged
        print(f"{radius:7d}{total(severing) / images:12.1f}{per_image:14.1f}"
              f"{raw:12.1f}{bridged:11.1f}{delta:8.1f}"
              f"{per_image / delta if delta else float('nan'):8.2f}")

    print("\n=== P3: what separates a resolved severance from a stuck one? ===")
    radius = RADII[-1]
    # Only breaks that severed BEFORE bridging. An intact break trivially does
    # not sever afterwards, so including them would inflate every rate and hide
    # the comparison this prediction is about.
    severing = [r for r in at_radius[radius] if r["kind_before"] == "severs"]
    print(f"at radius {radius} px, among the {total(severing)} breaks that "
          f"severed before bridging:")

    def rates_by(field: str, order: list[str]) -> float:
        """Resolution rate per bin. Returns the spread the predictor achieves,
        which is how the two candidate explanations get compared."""
        seen = []
        for name in order:
            sub = [r for r in severing if r[field] == name]
            if not sub:
                continue
            fixed = total([r for r in sub if r["kind_after"] != "severs"])
            seen.append(100 * fixed / total(sub))
            print(f"    {name:>8}{total(sub):9d}{seen[-1]:9.1f}%")
        return max(seen) - min(seen) if seen else 0.0

    lengths = [f"{low}-{high}" if high < 10 ** 6 else f"{low}+"
               for low, high in BUCKETS]
    thicknesses = [name for _, name in THICKNESS_BINS]
    print(f"  by LENGTH (E9b's explanation) {'n':>9}{'resolved':>10}")
    length_spread = rates_by("bucket", lengths)
    print(f"  by THICKNESS (the selftest's) {'n':>9}{'resolved':>10}")
    thick_spread = rates_by("thickness", thicknesses)
    winner = "thickness" if thick_spread > length_spread else "length"
    print(f"\n  spread across bins: length {length_spread:.1f} points, "
          f"thickness {thick_spread:.1f} points -> {winner} separates better")

    print("\n=== what the unreachable severing breaks look like ===")
    stuck = [r for r in severing if r["kind_after"] == "severs"]
    if stuck:
        print(f"{'length':>8}{'stuck':>9}{'share':>9}     "
              f"{'thick':>8}{'stuck':>9}{'share':>9}")
        left = [(n, total([r for r in stuck if r["bucket"] == n]))
                for n in lengths]
        right = [(n, total([r for r in stuck if r["thickness"] == n]))
                 for n in thicknesses]
        for (ln, lc), (tn, tc) in zip(left, right):
            print(f"{ln:>8}{lc:9d}{100 * lc / total(stuck):8.1f}%     "
                  f"{tn:>8}{tc:9d}{100 * tc / total(stuck):8.1f}%")


if __name__ == "__main__":
    main()
