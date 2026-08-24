"""Expected Run Length: the connectomics metric this series re-invented.

E10 built a per-break classification (`severs` / `intact` / `absent`) and
validated it twice. The literature check afterwards found that connectomics has
had the same idea since the flood-filling networks paper (2018): a ground-truth
skeleton edge whose endpoints land in DIFFERENT predicted segments is a SPLIT,
and one whose endpoint lands in no segment is an OMIT. Allen Institute's
`segmentation-skeleton-metrics` implements exactly that.

Their headline number is better than our count, for two reasons:

  - it is length-weighted, so a break that isolates two pixels of twig is not
    charged the same as one that severs half the tree, which a count does;
  - it has a unit. "You can trace 47 px before hitting an error" is a sentence
    a clinician can check. "18.8 severing breaks per image" is not.

ERL is the expected length of an error-free trace from a uniformly random point
on the ground-truth skeleton. Decompose the skeleton into fragments that lie
inside a single predicted component; with fragment lengths l_i and total
skeleton length L,

    ERL = sum(l_i^2) / L

Omitted pixels contribute nothing to the numerator and still count in L, which
is right: starting there, you trace zero.

  python exp/erl.py            # score every checkpoint
  python exp/erl.py --selftest # the mechanism

Writes results/erl.csv, one row per (run, image).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import cross_dataset
import hole_sweep
import speckle
import stratify
import train

torch.set_num_threads(2)  # the trainer already holds six; see CLAUDE.md

RESULTS = Path(__file__).resolve().parent / "results"


def fragments(skel_gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Lengths of the maximal skeleton runs lying in one predicted component.

    The labelling does the work. `pieces` is 8-connected, so two predicted
    pixels that are 8-adjacent are by construction in the SAME component --
    which means labelling the covered skeleton with the same connectivity can
    never merge two fragments that belong to different components. No
    per-component loop is needed, and there is no way for a split to be missed.
    """
    pieces = ndimage.label(pred, structure=break_lengths.CONN8)[0]
    covered = skel_gt & (pieces > 0)
    labels, count = ndimage.label(covered, structure=break_lengths.CONN8)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(labels.ravel())[1:]


def expected_run_length(skel_gt: np.ndarray, pred: np.ndarray) -> float:
    total = int(skel_gt.sum())
    if total == 0:
        return 0.0
    lengths = fragments(skel_gt, pred)
    return float((lengths.astype(np.float64) ** 2).sum() / total)


def selftest() -> None:
    """A straight skeleton cut into k pieces must give ERL = L / k."""
    skel = np.zeros((20, 100), dtype=bool)
    skel[10, 10:90] = True                 # 80 px of centreline
    total = int(skel.sum())
    assert total == 80, total

    perfect = np.zeros_like(skel)
    perfect[9:12, 10:90] = True            # a 3 px wide vessel covering it all
    got = expected_run_length(skel, perfect)
    print(f"fully covered skeleton of {total} px: ERL {got:.1f}")
    assert abs(got - 80.0) < 1e-9, got

    # A cut is two pixels WIDE, so it does not merely divide the skeleton, it
    # also deletes length. Writing the expectation out rather than reaching for
    # "L/k" is the point: the first draft of this test asserted L/2 and L/4 and
    # was wrong about the metric, not about the code.
    def expected(*pieces: int) -> float:
        return sum(piece ** 2 for piece in pieces) / total

    cut = perfect.copy()
    cut[:, 49:51] = False                  # removes skeleton cols 49 and 50
    got = expected_run_length(skel, cut)
    want = expected(39, 39)
    print(f"  one 2px cut in the middle -> 39 + 39 covered: "
          f"ERL {got:.2f} (expected {want:.2f})")
    assert abs(got - want) < 1e-9, (got, want)

    cuts = perfect.copy()
    for at in (30, 50, 70):
        cuts[:, at:at + 2] = False
    got = expected_run_length(skel, cuts)
    want = expected(20, 18, 18, 18)
    print(f"  three 2px cuts -> 20 + 18 + 18 + 18: "
          f"ERL {got:.2f} (expected {want:.2f})")
    assert abs(got - want) < 1e-9, (got, want)

    # The property a COUNT does not have: where the break falls matters. Two
    # predictions with one break each, one near the end and one in the middle.
    near_end, middle = perfect.copy(), perfect.copy()
    near_end[:, 15:17] = False
    middle[:, 49:51] = False
    end_erl = expected_run_length(skel, near_end)
    mid_erl = expected_run_length(skel, middle)
    print(f"  one break near the end: ERL {end_erl:.1f}; "
          f"same break in the middle: ERL {mid_erl:.1f}")
    assert end_erl > mid_erl + 10, (end_erl, mid_erl)
    print("  -> a count charges these equally; ERL does not")

    # Omitted skeleton (nothing predicted there) must lower ERL, not be ignored.
    partial = np.zeros_like(skel)
    partial[9:12, 10:50] = True            # only half the vessel predicted
    got = expected_run_length(skel, partial)
    print(f"  half the vessel never predicted: ERL {got:.1f} "
          f"(40^2/80 = 20)")
    assert abs(got - 20.0) < 1e-9, got
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

    import breaks
    geometry = []
    for item in test_items:
        truth = item["label"] & item["fov"]
        contrast = breaks.local_contrast(item["image"])
        geometry.append({"skel": skeletonize(truth), "contrast": contrast,
                         "gt": truth})
    edges = np.percentile(
        np.concatenate([g["contrast"][g["gt"]] for g in geometry]),
        [25, 50, 75])
    for geo in geometry:
        geo["band"] = stratify.band_map(geo["gt"], geo["contrast"], edges)
        # The best any prediction could score on this band: feed the ground
        # truth in as the prediction. Without it a band's ERL is unreadable,
        # because masking the skeleton to a contrast band shatters it and the
        # cap comes from the mask rather than the model.
        for index, band in enumerate(stratify.BANDS):
            mask = geo["skel"] & (geo["band"] == index)
            geo[f"ceiling_{band}"] = expected_run_length(mask, geo["gt"])

    runs = sys.argv[1:] or default_runs
    print(f"drive: {len(test_items)} images, component filter {component_px} px",
          flush=True)

    rows = []
    for run_name in runs:
        weights = out_dir / run_name / "final.pt"
        if not weights.exists():
            print(f"[{run_name}] no checkpoint, skipping", flush=True)
            continue
        model = train.build_model(run_name.rsplit("_s", 1)[0])
        model.load_state_dict(torch.load(weights, weights_only=False)["model"])
        model.eval()
        mean, std = train.normalisation(run_name, stacked)
        for item, geo in zip(test_items, geometry):
            prob = train.predict_full(model, item["image"], mean, std)
            pred = speckle.drop_small((prob >= 0.5) & item["fov"], component_px)
            row = {"run": run_name,
                   "config": run_name.rsplit("_s", 1)[0],
                   "image": item["name"],
                   "erl": round(expected_run_length(geo["skel"], pred), 3),
                   "skel_px": int(geo["skel"].sum())}
            # Per contrast band, because E2's whole lesson is that one average
            # hides two opposite effects. The share of the ceiling is what
            # makes the bands comparable, and it is a check rather than a
            # decoration: Q3 and Q4 turn out to sit at 99% of achievable, so
            # their raw numbers say nothing about the model at all.
            for index, band in enumerate(stratify.BANDS):
                mask = geo["skel"] & (geo["band"] == index)
                value = expected_run_length(mask, pred)
                ceiling = geo[f"ceiling_{band}"]
                row[f"erl_{band}"] = round(value, 3)
                row[f"share_{band}"] = (round(value / ceiling, 4)
                                        if ceiling else 0.0)
            rows.append(row)
        print(f"[{run_name}] done", flush=True)

    if not rows:
        raise SystemExit("no checkpoints found")
    out = RESULTS / "erl.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    configs = sorted({r["config"] for r in rows})
    print("\n=== expected run length over the whole skeleton, px ===")
    for config in configs:
        picked = [r for r in rows if r["config"] == config]
        print(f"{config:14}{np.mean([r['erl'] for r in picked]):9.1f}")

    print("\n=== share of the ACHIEVABLE run length, per contrast band ===")
    print("(1.000 means a perfect prediction could not trace further either)")
    print(f"{'config':14}"
          + "".join(f"{b.split('_')[0]:>9}" for b in stratify.BANDS))
    for config in configs:
        picked = [r for r in rows if r["config"] == config]
        print(f"{config:14}" + "".join(
            f"{np.mean([r[f'share_{band}'] for r in picked]):9.3f}"
            for band in stratify.BANDS))

    print(f"\n{'band':>14}{'ceiling px':>12}   what a perfect prediction scores")
    for band in stratify.BANDS:
        print(f"{band:>14}"
              f"{np.mean([g[f'ceiling_{band}'] for g in geometry]):12.2f}")


if __name__ == "__main__":
    main()
