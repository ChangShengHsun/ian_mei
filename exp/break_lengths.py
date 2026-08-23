"""E10: what is a "break", actually?

E2 counted breaks -- maximal runs of ground-truth centreline pixels the model
missed -- and used the count as the topology quantity that localises, since a
Betti number does not. E9b then guessed at why bridging recovers so few loops:
"the rest are gaps wider than two vessel widths". Both are assumptions about
what a break IS, and neither had been checked.

This script checks them, and the answer changes how the metric should be read:
only about 7% of breaks sever the prediction's connectivity. The other 93% have
predicted structure connected around them -- the run is missed centreline, not
a disconnection.

The classification is direct rather than a proxy. Dilate the run and see which
predicted components it touches: two or more means it sits between separate
pieces, exactly one means both sides are the same component, none means nothing
was predicted nearby. An earlier version used "how far the run gets from any
predicted foreground" instead; that quantity correlates with betti-0 error at
0.09 while this classification correlates at 0.46, so the proxy was measuring
something else and was dropped. Both are still recorded per break so the
comparison stays checkable.

Measured on top of E4's component filter, so the numbers line up with E9b.

  python exp/break_lengths.py
  python exp/break_lengths.py --selftest

Writes results/break_lengths.csv, one row per break.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breaks
import cross_dataset
import hole_sweep
import speckle
import stratify
import train

torch.set_num_threads(2)  # the trainer already holds six; see CLAUDE.md

RESULTS = Path(__file__).resolve().parent / "results"


# Retained only to report the failed proxy alongside the validated one; see
# the module docstring. Do not use it to define a disconnection.
DISCONNECTION_DEPTH = 3.0


CONN8 = np.ones((3, 3), dtype=bool)


def classify(run: np.ndarray, pred: np.ndarray, pieces: np.ndarray) -> str:
    """What this missed run does to the PREDICTION's connectivity.

    Dilate the run by one pixel and see which predicted components it touches:

      >= 2 distinct  the run sits between two separate pieces -- a severance
      exactly 1      both sides are the same component -- connectivity intact
      0              nothing predicted nearby -- a missed segment

    This is the definition rather than a proxy, which matters: an earlier
    version of this script used "how far the run gets from any predicted
    foreground" instead, and that quantity correlates with betti-0 error at
    0.09 while this classification correlates at 0.46. The proxy was measuring
    something else.
    """
    near = ndimage.binary_dilation(run, CONN8) & pred
    touched = set(np.unique(pieces[near])) - {0}
    if len(touched) >= 2:
        return "severs"
    return "intact" if touched else "absent"


def break_runs(pred: np.ndarray, skel_gt: np.ndarray,
               band: np.ndarray) -> list[tuple[int, int, float, str]]:
    """Every maximal run of missed centreline pixels,
    as (length, band, depth, kind).

    kind is the validated measure (see classify). depth is kept beside it as
    the proxy that failed, so the two stay comparable rather than the failure
    being invisible once it is out of the report.

    Connected components of the missed skeleton, 8-connected so a diagonal
    step counts as one run rather than two. The band is the majority band of
    the run's own pixels, which is how E2 charged a break to a contrast
    territory; a break that straddles two bands goes to the one it mostly
    sits in rather than being split or dropped.
    """
    missed = skel_gt & ~pred
    labels, count = ndimage.label(missed, structure=CONN8)
    if count == 0:
        return []
    away = ndimage.distance_transform_edt(~pred)
    pieces, _ = ndimage.label(pred, structure=CONN8)
    # Work inside each run's bounding box. Dilating a 2 px run across a whole
    # 584x565 image, 70k times, is a ten-minute job; on the crop it is seconds.
    boxes = ndimage.find_objects(labels)
    out = []
    for index, box in enumerate(boxes, start=1):
        grown = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
        pixels = labels[grown] == index
        codes = band[grown][pixels]
        out.append((int(pixels.sum()), int(np.bincount(codes).argmax()),
                    float(away[grown][pixels].max()),
                    classify(pixels, pred[grown], pieces[grown])))
    return out


def selftest() -> None:
    """Two breaks of known length on one straight vessel."""
    skel = np.zeros((20, 40), dtype=bool)
    skel[10, 2:38] = True
    pred = skel.copy()
    pred[10, 5:8] = False       # a 3 px break
    pred[10, 20:29] = False     # a 9 px break
    band = np.zeros_like(skel, dtype=int)

    found = break_runs(pred, skel, band)
    runs = sorted(length for length, _, _, _ in found)
    print(f"break lengths found: {runs}")
    assert runs == [3, 9], runs

    # Depth, and why it is not length. A clean cut through a 1 px wide line
    # puts the deepest missed pixel about half the run length from the
    # surviving structure; the same run beside an intact prediction is 1 px
    # deep no matter how long it is.
    depths = sorted(round(d, 2) for _, _, d, _ in found)
    print(f"depths of a 3px and a 9px clean cut: {depths}")
    offset = np.zeros((20, 40), dtype=bool)
    offset[10, 2:38] = True
    shifted = np.zeros_like(offset)
    shifted[11, 2:38] = True          # prediction covers the vessel, one row over
    misaligned = break_runs(shifted, offset, band)
    assert len(misaligned) == 1 and misaligned[0][0] == 36, misaligned
    assert misaligned[0][2] == 1.0, misaligned
    assert misaligned[0][3] == "intact", misaligned
    print(f"a 36px run beside an intact prediction one row over: "
          f"length 36, depth {misaligned[0][2]}, kind {misaligned[0][3]!r}"
          f" -- length alone would call this a huge break")

    # A run that straddles two bands is charged to its majority band, not
    # split. Six pixels in band 2 against three in band 0.
    band[:, 20:26] = 2
    runs = break_runs(pred, skel, band)
    straddler = [b for length, b, _, _ in runs if length == 9]
    print(f"a 9px break spanning bands 0 and 2 is charged to band "
          f"{straddler[0]}")
    assert straddler == [2], straddler
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    import drive
    val = drive.load_split("val")
    stacked = train.stack_split("train")
    inside = stacked["images"][stacked["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())
    width = cross_dataset.median_width(val)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))

    geometry = []
    for item in val:
        gt = item["label"] & item["fov"]
        geometry.append({"gt": gt, "skel": skeletonize(gt),
                         "contrast": breaks.local_contrast(item["image"])})
    edges = np.percentile(
        np.concatenate([g["contrast"][g["gt"]] for g in geometry]),
        [25, 50, 75])
    for geo in geometry:
        geo["band"] = stratify.band_map(geo["gt"], geo["contrast"], edges)

    runs = sys.argv[1:] or train.trained_runs()
    rows = []
    for run_name in runs:
        weights = RESULTS / run_name / "final.pt"
        if not weights.exists():
            print(f"[{run_name}] no checkpoint, skipping", flush=True)
            continue
        model = train.build_model(run_name.rsplit("_s", 1)[0])
        model.load_state_dict(
            torch.load(weights, weights_only=False)["model"])
        model.eval()
        for item, geo in zip(val, geometry):
            prob = train.predict_full(model, item["image"], mean, std)
            pred = speckle.drop_small((prob >= 0.5) & item["fov"], component_px)
            for length, code, depth, kind in break_runs(
                    pred, geo["skel"], geo["band"]):
                rows.append({"run": run_name,
                             "config": run_name.rsplit("_s", 1)[0],
                             "image": item["name"], "length": length,
                             "depth": round(depth, 3), "kind": kind,
                             "band": stratify.BANDS[code]})
        print(f"[{run_name}] done", flush=True)

    out = RESULTS / "break_lengths.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} breaks to {out}")

    lengths = np.array([r["length"] for r in rows])
    near_free = int(round(0.75 * width))
    expensive = int(round(2.0 * width))
    print(f"\nmedian structure width {width:.1f} px, so the E9b radii are "
          f"{near_free} px (near free) and {expensive} px (costly)")
    print(f"breaks: n={len(lengths)}, median {np.median(lengths):.0f} px, "
          f"mean {lengths.mean():.1f}, p90 {np.percentile(lengths, 90):.0f}")
    for radius in (1, near_free, expensive, 2 * expensive):
        share = 100 * (lengths <= radius).mean()
        print(f"  length <= {radius:2d} px: {share:5.1f}% of breaks")

    print(f"\n{'config':13}{'breaks/img':>12}{'median':>9}{'mean':>8}"
          f"{'p90':>7}{'<=2px':>8}{'<=6px':>8}")
    for config in sorted({r["config"] for r in rows}):
        picked = np.array([r["length"] for r in rows if r["config"] == config])
        images = len({(r["run"], r["image"]) for r in rows
                      if r["config"] == config})
        print(f"{config:13}{len(picked) / images:12.1f}"
              f"{np.median(picked):9.0f}{picked.mean():8.1f}"
              f"{np.percentile(picked, 90):7.0f}"
              f"{100 * (picked <= 2).mean():7.1f}%"
              f"{100 * (picked <= 6).mean():7.1f}%")

    print("\n=== how many of these actually sever the prediction? ===")
    for kind in ("severs", "intact", "absent"):
        sel = [r for r in rows if r["kind"] == kind]
        print(f"  {kind:8}{len(sel):9d}{100 * len(sel) / len(rows):7.1f}%")

    # The failed proxy, kept visible. depth <= 1 was supposed to mean "the
    # prediction covers this vessel, its skeleton is just offset", but it
    # tracks betti-0 error far worse than the direct classification does.
    depths = np.array([r["depth"] for r in rows])
    print(f"  (for comparison, the proxy that failed: "
          f"{100 * (depths <= 1.0).mean():.1f}% of runs are 1 px deep)")

    print(f"\n{'config':13}{'all breaks/img':>16}{'severing/img':>15}")
    for config in sorted({r["config"] for r in rows}):
        picked = [r for r in rows if r["config"] == config]
        images = len({(r["run"], r["image"]) for r in picked})
        severing = sum(r["kind"] == "severs" for r in picked)
        print(f"{config:13}{len(picked) / images:16.1f}{severing / images:15.1f}")

    print(f"\n{'band':>14}{'breaks':>9}{'share':>8}{'median':>9}{'mean':>8}"
          f"{'p90':>7}{'<=6px':>8}{'sever%':>9}")
    for band in stratify.BANDS:
        picked = np.array([r["length"] for r in rows if r["band"] == band])
        print(f"{band:>14}{len(picked):9d}{100 * len(picked) / len(rows):7.1f}%"
              f"{np.median(picked):9.0f}{picked.mean():8.1f}"
              f"{np.percentile(picked, 90):7.0f}"
              f"{100 * (picked <= 6).mean():7.1f}%"
              f"{100 * np.mean([r['kind'] == 'severs' for r in rows if r['band'] == band]):8.1f}%")


if __name__ == "__main__":
    main()
