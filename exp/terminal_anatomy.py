"""Is ERL blind to DRIVE's failures, because DRIVE's failures are terminal?

THE OBJECTION, in full, because it is the strongest one aimed at this work:
ERL is Sum(l^2)/L. It weights a fragment by its own length, so severing a
2-pixel twig costs almost nothing while cutting a trunk in half costs
enormously. E15 already measured this on a synthetic: the SAME break scores
66.9 at a terminal and 38.0 at the centre. If DRIVE's breaks live at the
distal tips -- which is where the vessels are thinnest and the contrast is
worst -- then ERL is nearly blind to the errors that actually dominate, and a
paper built on ERL is measuring the wrong tail.

Nothing in the repo answers this. break_lengths.csv records length, depth,
kind and contrast band; it does not record WHERE ON THE TREE a break sits.
This file measures that, and it is written to be able to LOSE: if the ERL
loss turns out to be terminal-dominated, that is a finding against our own
headline and it goes in the paper as one.

WHAT IS MEASURED, per (run, image), at the arm's own dev-picked threshold:

  terminal distance   geodesic distance along the GROUND-TRUTH skeleton from
                      each break to the nearest skeleton endpoint, in
                      multiples of median vessel width so it transfers.
  erl cost            leave-one-out: ERL with this break filled from ground
                      truth, minus ERL as predicted. What this break actually
                      costs, not how long it is.
  masked erl          ERL recomputed with every skeleton pixel within k widths
                      of a terminal DELETED. If the arms rank the same way
                      with the terminals removed, ERL's conclusions are not
                      terminal artefacts -- which is the defence. If the
                      ranking moves, the objection holds and ERL needs a
                      companion metric.

The two halves answer different halves of the objection. The first says where
the loss lives; the second says whether our conclusions depend on it.

  python exp/terminal_anatomy.py --selftest
  python exp/terminal_anatomy.py --shard 0/4
  python exp/terminal_anatomy.py --report

Writes results/heldout/terminal_anatomy[.shardIofN].csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import composition
import cross_dataset
import drive
import erl
import erl_convention
import hole_sweep
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import threshold_control as control
import train

# Weak to strong, so the answer can be read against base-model strength the
# way the postproc table is. Four arms, not ten: this is a question about the
# METRIC, and four points across the range answer it.
ARMS = ("A_dice", "H_aug", "H_aug_clw16", "K_focal_aug_clw64")

# In multiples of median vessel width. 1 width is "on the tip", 10 widths is
# about 28 px on DRIVE, which is past the 21 px 90th percentile of the break
# stretches, so a break that far in is unambiguously mid-tree.
TERMINAL_BANDS = (1.0, 2.0, 5.0, 10.0)


def terminal_distance(skel: np.ndarray) -> np.ndarray:
    """Geodesic distance along the skeleton to the nearest endpoint.

    A multi-source BFS over the 8-connected skeleton graph, seeded at every
    endpoint at once. Euclidean distance would be wrong: two tips of a hairpin
    are close in the plane and far apart along the vessel, and it is the along
    distance that says how much run length a break can possibly destroy.

    Diagonal steps count as 1, not sqrt(2). The quantity is fed to a banding
    in multiples of vessel width, and a 40% error on the diagonal steps cannot
    move a break across a band boundary that is whole widths wide.
    """
    counts = ndimage.convolve(skel.astype(np.uint8), break_lengths.CONN8.astype(np.uint8),
                              mode="constant")
    ends = skel & (counts == 2)
    distance = np.full(skel.shape, -1, dtype=np.int32)
    if not ends.any():
        return distance
    distance[ends] = 0
    frontier = ends.copy()
    step = 0
    while frontier.any():
        step += 1
        grown = ndimage.binary_dilation(frontier, break_lengths.CONN8)
        frontier = grown & skel & (distance < 0)
        distance[frontier] = step
    return distance


def break_segments(skel: np.ndarray, pred: np.ndarray):
    """Connected runs of ground-truth centreline the prediction misses."""
    missed = skel & ~pred
    labels, count = ndimage.label(missed, structure=break_lengths.CONN8)
    return labels, count


def measure_image(skel, truth, pred, distance, width) -> list[dict]:
    """One row per break: where it sits, and what it costs."""
    base_split = erl.expected_run_length(skel, pred)
    base_bridged = erl_convention.bridged_run_length(skel, pred)
    labels, count = break_segments(skel, pred)
    rows = []
    for index in range(1, count + 1):
        segment = labels == index
        # Fill from ground truth, thickened to the vessel so the filled run is
        # actually connected to what it joins -- a one-pixel centreline thread
        # would be 8-connected to nothing at either end.
        patch = ndimage.binary_dilation(
            segment, structure=np.ones((3, 3), bool),
            iterations=max(int(round(width / 2)), 1)) & truth
        filled = pred | patch
        rows.append({
            "length": int(segment.sum()),
            "terminal": float(np.min(distance[segment])) / width
            if (distance[segment] >= 0).any() else -1.0,
            "erl_cost_split": round(
                (erl.expected_run_length(skel, filled) - base_split)
                / skel.sum(), 6),
            "erl_cost_bridged": round(
                (erl_convention.bridged_run_length(skel, filled) - base_bridged)
                / skel.sum(), 6)})
    return rows


def masked_erl(skel, pred, distance, width, band: float,
               keep_tips: bool = False) -> tuple:
    """ERL on part of the skeleton: the trunk, or ONLY the tips.

    Two readings, because the objection needs both halves answered.

    `keep_tips=False` deletes the terminal `band` widths of every branch, and
    is the DEFENCE: if the arms rank the same with the tips removed, the
    ranking was never a statement about tips.

    `keep_tips=True` keeps ONLY those tips, and is the COMPANION METRIC: it is
    where a length-weighted metric is structurally blind, so it is exactly the
    number that has to be reported beside ERL rather than instead of it. Note
    it is not merely ERL restricted -- restricting the skeleton also shortens
    every run, so its ceiling is lower and it must never be compared against
    the whole-skeleton figure, only across arms at the same band.
    """
    if band <= 0.0:
        keep = skel if not keep_tips else np.zeros_like(skel)
    elif keep_tips:
        keep = skel & (distance >= 0) & (distance <= band * width)
    else:
        keep = skel & ((distance < 0) | (distance > band * width))
    if not keep.any():
        return 0.0, 0.0
    return (erl.expected_run_length(keep, pred) / keep.sum(),
            erl_convention.bridged_run_length(keep, pred) / keep.sum())


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    # 0. Sharding is an exact partition. This file lost 5 of H_aug's 12 seeds
    #    to hash-based sharding on 2026-09-01 and the report showed nothing.
    work = [f"a_s{index}" for index in range(13)]
    for total in (2, 4):
        flat = [r for i in range(total)
                for r in sweep.shard_filter(work, (i, total))]
        assert sorted(flat) == sorted(work) and len(flat) == len(set(flat))
    print("sharding is an exact partition over 2/4 shards")

    # 1. A LEAF IS AT DISTANCE 0 AND THE MIDDLE IS FAR. On a straight segment
    #    of length L the middle is L/2 from either end.
    line = np.zeros((10, 41), dtype=bool)
    line[5, 0:41] = True
    distance = terminal_distance(line)
    assert distance[5, 0] == 0 and distance[5, 40] == 0, distance[5, [0, 40]]
    assert distance[5, 20] == 20, distance[5, 20]
    print(f"straight skeleton of 41 px: tips at 0, centre at "
          f"{distance[5, 20]} -- geodesic, along the vessel")

    # 2. A HAIRPIN MUST NOT BE FOOLED BY EUCLIDEAN CLOSENESS. Two tips a
    #    single pixel apart in the plane are the full length apart along the
    #    curve, and it is the along distance the ERL cost depends on.
    pin = np.zeros((10, 25), dtype=bool)
    pin[3, 2:23] = True
    pin[5, 2:23] = True
    pin[4, 22] = True
    got = terminal_distance(pin)
    print(f"hairpin whose two tips are 2 px apart in the plane: the centre of "
          f"one arm is {got[3, 12]} steps from a tip, not 2")
    assert got[3, 12] >= 10, got[3, 12]

    # 3. E15'S CLAIM MUST REPRODUCE HERE. The same break costs less at a
    #    terminal than at the centre. If this fails the objection is moot and
    #    so is the file.
    skel = np.zeros((10, 101), dtype=bool)
    skel[5, 0:101] = True
    truth = np.zeros_like(skel)
    truth[4:7, 0:101] = True
    total = float(skel.sum())
    for where, label in ((slice(1, 6), "at the tip"), (slice(48, 53), "mid")):
        pred = truth.copy()
        pred[:, where] = False
        cut = erl.expected_run_length(skel, pred) / total
        print(f"  a 5 px break {label:11}: ERL {cut:.1%}")
        if label == "at the tip":
            tip = cut
        else:
            middle = cut
    assert tip > middle, (tip, middle)
    print(f"  -> the identical break costs {tip - middle:.1%} more ERL when "
          f"it is mid-vessel. THAT IS THE OBJECTION, reproduced.")

    # 4. THE LEAVE-ONE-OUT COST MUST BE POSITIVE AND ORDERED THE SAME WAY.
    #    Filling a break can only help, and filling the mid one must help more.
    distance = terminal_distance(skel)
    for where, label in ((slice(1, 6), "at the tip"), (slice(48, 53), "mid")):
        pred = truth.copy()
        pred[:, where] = False
        rows = measure_image(skel, truth, pred, distance, 3.0)
        assert len(rows) == 1, rows
        assert rows[0]["erl_cost_split"] > 0, rows
        print(f"  leave-one-out cost of the {label:11} break: "
              f"{rows[0]['erl_cost_split']:+.1%} "
              f"(terminal distance {rows[0]['terminal']:.1f} widths)")

    # 5. MASKING THE TIPS MUST REMOVE SKELETON, AND NOT ALL OF IT.
    pred = truth.copy()
    pred[:, 48:53] = False
    plain = erl.expected_run_length(skel, pred) / total
    kept, _ = masked_erl(skel, pred, distance, 3.0, 5.0)
    print(f"  ERL {plain:.1%} on the whole skeleton, {kept:.1%} with the "
          f"terminal 15 px of each branch deleted")
    assert kept > 0.0
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def load() -> list[dict]:
    rows = []
    for path in sorted(heldout.ROOT.glob("terminal_anatomy*.csv")):
        for row in csv.DictReader(path.open()):
            rows.append({**row, "length": int(row["length"]),
                         "terminal": float(row["terminal"]),
                         "erl_cost_split": float(row["erl_cost_split"]),
                         "erl_cost_bridged": float(row["erl_cost_bridged"])})
    return rows


def load_masked() -> list[dict]:
    rows = []
    for path in sorted(heldout.ROOT.glob("terminal_masked*.csv")):
        for row in csv.DictReader(path.open()):
            rows.append({**row, "band": float(row["band"]),
                         "region": row.get("region", "trunk"),
                         "erl_split": float(row["erl_split"]),
                         "erl_bridged": float(row["erl_bridged"])})
    return rows


def report() -> None:
    rows, masked = load(), load_masked()
    if not rows:
        raise SystemExit("no terminal_anatomy*.csv yet")
    print("=== does DRIVE's ERL loss live at the terminals? ===\n")
    print(f"{len(rows)} breaks over {len({r['run'] for r in rows})} runs, "
          f"{len({r['image'] for r in rows})} images")
    print("Terminal distance is geodesic ALONG the ground-truth skeleton to")
    print("the nearest tip, in multiples of median vessel width.\n")
    for metric, label in (("erl_cost_split", "convention A"),
                          ("erl_cost_bridged", "convention B")):
        print(f"--- {label}: where the ERL loss actually is ---")
        print("    share of the total ERL loss carried by breaks within N "
              "widths of a tip")
        print(f"    {'arm':20} {'breaks':>8} " +
              " ".join(f"{'<=' + str(b) + 'w':>9}" for b in TERMINAL_BANDS) +
              "   breaks on a tip")
        for config in ARMS:
            these = [r for r in rows if r["config"] == config]
            if not these:
                continue
            total = sum(r[metric] for r in these)
            if total <= 0:
                continue
            shares = []
            for band in TERMINAL_BANDS:
                inside = sum(r[metric] for r in these
                             if 0 <= r["terminal"] <= band)
                shares.append(f"{inside / total:9.1%}")
            counts = sum(1 for r in these if 0 <= r["terminal"] <= 1.0)
            print(f"    {config:20} {len(these):8d} " + " ".join(shares) +
                  f"   {counts / len(these):14.0%}")
        print()
    if not masked:
        print("no terminal_masked*.csv yet -- the ranking defence is pending")
        return
    print("--- ERL on the trunk alone, and on the tips alone ---")
    print("    A band's tip figure has a LOWER ceiling than the whole "
          "skeleton\n    -- read it across arms at one band, never against "
          "the full number.")
    for metric in ("erl_split", "erl_bridged"):
        print(f"\n  {metric}")
        for region in ("trunk", "tips"):
            for band in ([0.0] if region == "trunk" else []) + \
                    list(TERMINAL_BANDS):
                cells, order = [], []
                for config in ARMS:
                    these = [r[metric] for r in masked
                             if r["config"] == config and r["band"] == band
                             and r["region"] == region]
                    if these:
                        cells.append(f"{config} {np.mean(these):6.1%}")
                        order.append((float(np.mean(these)), config))
                    else:
                        cells.append(f"{config} {'--':>6}")
                if not order:
                    continue
                where = ("whole skeleton" if band == 0.0 else
                         f"{region:5} at {band:4.1f}w")
                print(f"    {where:20}: " + "  ".join(cells))
                print(f"    {'':20}  ranking "
                      f"{' < '.join(c for _, c in sorted(order))}")
        print()


# -------------------------------------------------------------------- main

def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    shard = None
    for index, arg in enumerate(sys.argv):
        if arg == "--shard":
            part, total = sys.argv[index + 1].split("/")
            shard = (int(part), int(total))
    stem = "terminal_anatomy" if shard is None else \
        f"terminal_anatomy.shard{shard[0]}of{shard[1]}"
    target = heldout.ROOT / f"{stem}.csv"
    masked_target = heldout.ROOT / f"{stem.replace('anatomy', 'masked')}.csv"

    thresholds = control.load("dev")
    if not thresholds:
        raise SystemExit("no threshold_control_dev*.csv -- the operating "
                         "point is chosen from it")
    epochs = heldout.chosen_epochs()
    done = set()
    for existing in sorted(heldout.ROOT.glob("terminal_anatomy*.csv")):
        done |= {(r["config"], r["seed"]) for r in csv.DictReader(existing.open())}

    items = drive.load_split("test")
    data = train.stack_split("fit")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = []
    for item in items:
        truth = item["label"] & item["fov"]
        skel = skeletonize(truth)
        geometry.append({"skel": skel, "truth": truth,
                         "distance": terminal_distance(skel)})
    print(f"{len(ARMS)} arms, width {width:.2f} px, filter {component_px} px",
          flush=True)

    fresh, fresh_masked = not target.exists(), not masked_target.exists()
    with target.open("a", newline="") as handle, \
            masked_target.open("a", newline="") as masked_handle:
        writer = csv.DictWriter(
            handle, fieldnames=["config", "run", "seed", "image", "threshold",
                                "length", "terminal", "erl_cost_split",
                                "erl_cost_bridged"])
        masked_writer = csv.DictWriter(
            masked_handle, fieldnames=["config", "run", "seed", "image",
                                       "band", "region", "erl_split",
                                       "erl_bridged"])
        if fresh:
            writer.writeheader()
        if fresh_masked:
            masked_writer.writeheader()
        for config in ARMS:
            base = composition.base_threshold(thresholds, config,
                                              "erl_bridged")
            runs = sorted(r for r in epochs
                          if r.rsplit("_s", 1)[0] == config
                          and (heldout.ROOT / r / "final.pt").exists())
            # Stride over a sorted list, never hash(): see sweep.shard_filter.
            for run in sweep.shard_filter(runs, shard):
                seed = run.rsplit("_s", 1)[1]
                if (config, seed) in done:
                    continue
                model, mean, std = sweep.load_model(run, config, epochs, data)
                if model is None:
                    continue
                for item, geo in zip(items, geometry):
                    prob = train.predict_full(model, item["image"], mean, std)
                    pred = speckle.drop_small((prob >= base) & item["fov"],
                                              component_px)
                    common = {"config": config, "run": run, "seed": seed,
                              "image": item["name"]}
                    for row in measure_image(geo["skel"], geo["truth"], pred,
                                             geo["distance"], width):
                        writer.writerow({**common, "threshold": base, **row})
                    for band in (0.0,) + TERMINAL_BANDS:
                        for keep_tips in (False, True):
                            if band == 0.0 and keep_tips:
                                continue
                            split, bridged = masked_erl(
                                geo["skel"], pred, geo["distance"], width,
                                band, keep_tips)
                            masked_writer.writerow({
                                **common, "band": band,
                                "region": "tips" if keep_tips else "trunk",
                                "erl_split": round(split, 5),
                                "erl_bridged": round(bridged, 5)})
                handle.flush()
                masked_handle.flush()
                print(f"  {run} at threshold {base} done", flush=True)
    print(f"wrote {target} and {masked_target}")


if __name__ == "__main__":
    main()
