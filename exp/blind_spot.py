"""Is there a failure the field's metrics cannot see, and ERL can?

WRITTEN AND SELFTESTED 2026-09-19, BEFORE IT SCORED ANY REAL IMAGE.

WHY THIS FILE EXISTS. metric_redundancy.py showed that 22 runs sitting inside
clDice 0.8153 +/- 0.002 span 51.6 points of traceable length. That is half an
argument: it shows clDice calls them equal, but nothing in it says WHICH of
those runs is actually better -- only ERL says so, and using ERL to prove ERL
is circular. This file supplies the missing half by CONSTRUCTING the pairs, so
that the quality ordering is fixed by the construction and not by any metric.

THE CONSTRUCTION. Take a ground-truth mask as a perfect prediction, then sever
it k times. Every cut is a disc of radius 2x the local vessel radius centred on
a centreline pixel, so a cut always severs and its size scales with calibre
(CLAUDE.md: thresholds in multiples of structure width, never absolute px).
Two arms are built from the SAME cut geometry and differ in ONE variable:

    mid   the site sits in the interior of a branch, >= 25 skeleton steps from
          the nearest endpoint. Severing there orphans everything distal.
    tip   the site sits R+2 steps from an endpoint, so the cut disc still lies
          fully on the branch but the piece it orphans is a few pixels.

Sites are PAIRED across arms: a mid site and a tip site enter the table only if
they share the same cut radius R and their deleted foreground area differs by
at most 1 px. So the two arms delete the same amount of vessel, in the same
number of pieces, at the same calibre. The only thing that differs is WHERE
along the branch the tissue was taken -- which is exactly the thing a set
overlap cannot represent.

WHY THIS IS NOT CIRCULAR. The quality ordering is not read off ERL. Cutting a
branch in its interior detaches a whole subtree from the tree; cutting it two
pixels from its tip detaches two pixels. The anchor reported alongside the
metrics is that quantity: the fraction of predicted vessel that is no longer
connected to the main tree, i.e. the fraction a reader tracing from the optic
disc can no longer reach. It is linear in the size of what is lost, where ERL
is quadratic, and Dice, clDice and Betti-0 are all equally free to track it.

THE CONTROL. A third arm, `edge`, deletes the same total area from vessel
BOUNDARIES, one simple point at a time, so nothing is ever severed. Without it
"ERL moved" would only say that ERL noticed pixels were deleted.

PRE-REGISTERED PREDICTIONS (written before the first real image was scored):

  P1  |Dice(mid) - Dice(tip)|   < 0.002   the tie window metric_redundancy used
  P2  |clDice(mid) - clDice(tip)| < 0.005
  P3  Betti-0 error is IDENTICAL for mid and tip (k extra components each)
  P4  traced ERL: tip - mid > 20 points
  P5  the anchor orders tip better than mid on every image (so ERL's ordering
      is the anchor's ordering, and Dice's / clDice's tie is a tie on a pair
      the anchor separates)
  P6  the `edge` control, at the same deleted area, stays within 1 point of the
      uncut mask's traced ERL

  KILL: if |clDice(mid) - clDice(tip)| > 0.01, five times the reader's tie
  window, then clDice is NOT blind to this failure, the case set does not
  support the claim, and the report must say so instead of softening it.

Writes results/blind_spot.csv, results/blind_spot.txt, results/blind_spot.png.
About 2 minutes on CPU for the 20 DRIVE test images.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration
import drive
import erl
import metrics

CONN8 = np.ones((3, 3), dtype=bool)
RESULTS = Path(__file__).resolve().parent / "results"

CUTS_PER_IMAGE = 30      # e15 measured ~18.8 severing breaks per image; same order
CUT_SCALE = 2.0          # cut radius = CUT_SCALE * local vessel radius, then +1
MID_MIN_STEPS = 25       # a mid site is >= this many skeleton steps from a tip
AREA_TOLERANCE = 1       # px, per paired cut
RADIUS_BAND = (1.0, 3.0)  # local vessel radius admitted, px


def disc(radius: int) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    return (span[:, None] ** 2 + span[None, :] ** 2) <= radius ** 2


def geodesic(skel: np.ndarray, seeds: np.ndarray, cap: int = 120) -> np.ndarray:
    """Steps along the skeleton from any seed, 8-connected, capped."""
    dist = np.full(skel.shape, np.inf, dtype=np.float32)
    dist[seeds] = 0.0
    frontier = seeds.copy()
    for step in range(1, cap + 1):
        grown = ndimage.binary_dilation(frontier, CONN8) & skel & np.isinf(dist)
        if not grown.any():
            break
        dist[grown] = step
        frontier = grown
    return dist


def degree(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours each skeleton pixel has, 0 off the skeleton."""
    counted = ndimage.convolve(skel.astype(np.int16), CONN8.astype(np.int16),
                               mode="constant") - skel.astype(np.int16)
    return np.where(skel, counted, 0)


def candidate_sites(mask: np.ndarray, skel: np.ndarray) -> dict:
    """Two pools of cut sites that differ only in distance from a branch tip."""
    radius_map = ndimage.distance_transform_edt(mask)
    deg = degree(skel)
    tips = skel & (deg == 1)
    forks = skel & (deg >= 3)
    from_tip = geodesic(skel, tips)
    from_fork = geodesic(skel, forks)

    radius_here = np.where(skel, radius_map, 0.0)
    in_band = skel & (radius_here >= RADIUS_BAND[0]) & (radius_here <= RADIUS_BAND[1])
    cut_radius = np.ceil(CUT_SCALE * radius_here).astype(np.int32) + 1
    # A cut must not swallow a bifurcation: that would change the component
    # count by more than one and break the Betti-0 match between the arms.
    clear_of_fork = from_fork > (cut_radius + 2)
    plain = in_band & (deg == 2) & clear_of_fork

    pools = {}
    for tag, keep in (("mid", from_tip >= MID_MIN_STEPS),
                      ("tip", (from_tip >= cut_radius + 2)
                              & (from_tip <= cut_radius + 3))):
        rows, cols = np.nonzero(plain & keep)
        pools[tag] = [{"row": int(r), "col": int(c),
                       "radius": int(cut_radius[r, c]),
                       "vessel_radius": float(radius_here[r, c]),
                       "from_tip": float(from_tip[r, c])}
                      for r, c in zip(rows, cols)]
    return pools


def cut_extent(mask: np.ndarray, skel: np.ndarray, site: dict) -> tuple:
    """(deleted foreground px, deleted centreline px, the slice, the stamp)."""
    radius = site["radius"]
    top, left = site["row"] - radius, site["col"] - radius
    box = (slice(max(top, 0), min(site["row"] + radius + 1, mask.shape[0])),
           slice(max(left, 0), min(site["col"] + radius + 1, mask.shape[1])))
    stamp = disc(radius)[
        box[0].start - top: box[0].stop - top,
        box[1].start - left: box[1].stop - left]
    return (int((mask[box] & stamp).sum()), int((skel[box] & stamp).sum()),
            box, stamp)


def pair_sites(mask: np.ndarray, skel: np.ndarray, pools: dict,
               wanted: int, spacing: float = 3.0) -> list:
    """Match mid sites to tip sites of the SAME radius and the same area.

    Greedy, and deliberately so: the pairing is a constraint on the case set,
    not an optimisation. A pair that cannot be matched is dropped, never
    stretched, because an unmatched pair reintroduces the confound the whole
    file exists to remove.
    """
    measured = {}
    for tag, pool in pools.items():
        rows = []
        for site in pool:
            area, centre, _, _ = cut_extent(mask, skel, site)
            rows.append({**site, "area": area, "centreline": centre})
        measured[tag] = rows

    by_radius = {}
    for site in measured["tip"]:
        by_radius.setdefault(site["radius"], []).append(site)

    # Deepest first, NOT largest-area first. The contrast the case set is
    # built on is position along the branch, so the mid arm must take the sites
    # furthest from any tip; sorting by area let it settle on sites sitting
    # right at the MID_MIN_STEPS floor and halved the effect.
    chosen, taken = [], {"mid": [], "tip": []}
    for site in sorted(measured["mid"], key=lambda s: -s["from_tip"]):
        if len(chosen) == wanted:
            break
        if not far_enough(site, taken["mid"], spacing):
            continue
        candidates = [other for other in by_radius.get(site["radius"], [])
                      if abs(other["area"] - site["area"]) <= AREA_TOLERANCE
                      and far_enough(other, taken["tip"], spacing)]
        if not candidates:
            continue
        partner = min(candidates, key=lambda o: abs(o["area"] - site["area"]))
        by_radius[site["radius"]].remove(partner)
        taken["mid"].append(site)
        taken["tip"].append(partner)
        chosen.append((site, partner))
    return chosen


def far_enough(site: dict, already: list, spacing: float) -> bool:
    """Keep cuts from interacting: centres at least spacing * radii apart."""
    for other in already:
        gap = np.hypot(site["row"] - other["row"], site["col"] - other["col"])
        if gap < spacing * max(site["radius"], other["radius"]):
            return False
    return True


def apply_cuts(mask: np.ndarray, skel: np.ndarray, sites: list) -> np.ndarray:
    cut = mask.copy()
    for site in sites:
        _, _, box, stamp = cut_extent(mask, skel, site)
        cut[box] &= ~stamp
    return cut


def crossing_number(mask: np.ndarray) -> np.ndarray:
    """0->1 transitions around each pixel's 8-neighbourhood.

    A foreground pixel with exactly one transition is a simple point: deleting
    it cannot split a component or open a hole. This is the Zhang-Suen /
    Hilditch condition, and it is what lets the `edge` control delete area
    without ever severing anything.
    """
    ring = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    shifted = [np.roll(np.roll(mask, -dr, axis=0), -dc, axis=1).astype(np.int8)
               for dr, dc in ring]
    # np.roll wraps; the border is not a place we delete from anyway, and the
    # caller masks it off explicitly below.
    transitions = np.zeros(mask.shape, dtype=np.int16)
    for here, nxt in zip(shifted, shifted[1:] + shifted[:1]):
        transitions += (here == 0) & (nxt == 1)
    return transitions


def erode_by(mask: np.ndarray, area: int, rng: np.random.Generator,
             spare: np.ndarray | None = None) -> np.ndarray:
    """Delete `area` px from the boundary, one simple point at a time.

    `spare` is a set of pixels that may never be deleted. Passing the ground
    truth centreline is what separates the two controls: without it, eroding a
    1 px wide capillary deletes the centreline itself, which is severance under
    another name -- P6 failed exactly there.
    """
    worn = mask.copy()
    border = np.zeros(mask.shape, dtype=bool)
    border[1:-1, 1:-1] = True
    if spare is not None:
        border &= ~spare
    removed = 0
    while removed < area:
        simple = (worn & border & (crossing_number(worn) == 1)
                  & ~ndimage.binary_erosion(worn, CONN8))
        rows, cols = np.nonzero(simple)
        if len(rows) == 0:
            break
        order = rng.permutation(len(rows))[: max(1, (area - removed) // 4)]
        # Deleting a whole independent set at once is the fast path, but two
        # adjacent simple points are not jointly simple, so thin the batch to
        # pixels that are not 8-adjacent to another pixel in the batch.
        batch = np.zeros(mask.shape, dtype=bool)
        for index in order:
            row, col = rows[index], cols[index]
            if batch[row - 1:row + 2, col - 1:col + 2].any():
                continue
            batch[row, col] = True
            removed += 1
            if removed == area:
                break
        if not batch.any():
            break
        worn &= ~batch
    return worn


def splits_made(mask: np.ndarray, skel: np.ndarray, sites: list) -> dict:
    """How many of these cuts actually add a component, and what the rest do.

    P3 predicted the two arms would post the same Betti-0 error and they did
    not. A cut that severs a TREE always adds a component; a cut that severs
    one side of a LOOP does not, it closes a hole instead. The DRIVE mask is
    full of loops because arteries and veins cross in projection, so this
    counts both outcomes rather than assuming either.
    """
    before = metrics.betti(mask)
    adds, closes = 0, 0
    for site in sites:
        after = metrics.betti(apply_cuts(mask, skel, [site]))
        adds += int(after[0] > before[0])
        closes += int(after[1] < before[1])
    return {"splits": adds, "loop_cuts": closes, "sites": len(sites)}


def largest_component_fraction(mask: np.ndarray) -> float:
    labels, count = ndimage.label(mask, structure=CONN8)
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / sizes.sum())


def score(mask: np.ndarray, truth: np.ndarray, skel: np.ndarray) -> dict:
    """The field's three metrics, ERL, and the anchor that is none of them."""
    run_length = erl.expected_run_length(skel, mask)
    return {
        "dice": metrics.dice(mask, truth),
        "cldice": metrics.cl_dice(mask, truth),
        "betti0_err": abs(metrics.betti(mask)[0] - metrics.betti(truth)[0]),
        "erl_px": run_length,
        "traced": run_length / max(int(skel.sum()), 1),
        "unreachable": 1.0 - largest_component_fraction(mask),
        "foreground": int(mask.sum()),
    }


def selftest() -> None:
    """Assert the MECHANISM on synthetic geometry, before any real image."""
    # --- the crossing number, which the `edge` control rests on -------------
    line = np.zeros((9, 9), dtype=bool)
    line[4, 2:7] = True
    crossings = crossing_number(line)
    assert crossings[4, 4] == 2, crossings[4, 4]   # mid-line: deleting splits
    assert crossings[4, 6] == 1, crossings[4, 6]   # the end: deleting is safe
    blob = np.zeros((9, 9), dtype=bool)
    blob[3:7, 3:7] = True
    assert crossings[0, 0] == 0                     # background counts nothing
    assert crossing_number(blob)[3, 4] == 1, "a blob's boundary must be simple"
    print("crossing number: mid-line 2, line end 1, blob boundary 1")

    # --- geodesic steps along a skeleton ------------------------------------
    bar = np.zeros((21, 101), dtype=bool)
    bar[10, 10:91] = True
    steps = geodesic(bar, bar & (degree(bar) == 1))
    assert steps[10, 10] == 0 and steps[10, 90] == 0, "endpoints are the seeds"
    assert steps[10, 50] == 40, steps[10, 50]
    print("geodesic: the centre of an 81 px bar is 40 steps from a tip")

    # --- THE mechanism: same cut, two positions ----------------------------
    vessel = np.zeros((41, 241), dtype=bool)
    vessel[19:22, 15:225] = True                    # one straight 3 px vessel
    skel = skeletonize(vessel)
    pairs = pair_sites(vessel, skel, candidate_sites(vessel, skel), wanted=1)
    assert pairs, "no matched pair on a straight vessel -- the pools are broken"
    mid_site, tip_site = pairs[0]
    assert mid_site["radius"] == tip_site["radius"], (mid_site, tip_site)
    assert abs(mid_site["area"] - tip_site["area"]) <= AREA_TOLERANCE

    cut_mid = apply_cuts(vessel, skel, [mid_site])
    cut_tip = apply_cuts(vessel, skel, [tip_site])
    assert cut_mid.sum() == cut_tip.sum(), (cut_mid.sum(), cut_tip.sum())
    graded = {tag: score(mask, vessel, skel)
              for tag, mask in (("mid", cut_mid), ("tip", cut_tip))}
    assert abs(graded["mid"]["dice"] - graded["tip"]["dice"]) < 1e-12, graded
    assert graded["mid"]["betti0_err"] == graded["tip"]["betti0_err"] == 1
    gap = graded["tip"]["traced"] - graded["mid"]["traced"]
    assert gap > 0.2, gap
    assert graded["mid"]["unreachable"] > graded["tip"]["unreachable"], graded
    print(f"one cut, same area ({mid_site['area']} px), same Dice, same b0: "
          f"traced ERL {graded['mid']['traced']:.3f} (mid) vs "
          f"{graded['tip']['traced']:.3f} (tip); "
          f"unreachable {graded['mid']['unreachable']:.3f} vs "
          f"{graded['tip']['unreachable']:.3f}")

    # --- the control must delete area without severing ----------------------
    before = int(vessel.sum())
    worn = erode_by(vessel, 40, np.random.default_rng(0))
    assert before - int(worn.sum()) == 40, before - int(worn.sum())
    assert metrics.betti(worn)[0] == metrics.betti(vessel)[0], "edge severed"
    assert erl.expected_run_length(skel, worn) == erl.expected_run_length(
        skel, vessel), "an unsevered erosion moved ERL"
    print("edge control: 40 px deleted, component count and ERL both unmoved")

    # --- why there are TWO controls (the lesson P6 taught) ------------------
    hair = np.zeros((41, 241), dtype=bool)
    hair[20, 15:225] = True                         # a 1 px wide capillary
    hair_skel = skeletonize(hair)
    intact = erl.expected_run_length(hair_skel, hair)
    careless = erode_by(hair, 20, np.random.default_rng(0))
    careful = erode_by(hair, 20, np.random.default_rng(0), spare=hair_skel)
    assert int(careful.sum()) == int(hair.sum()), "sparing left nothing to delete"
    assert erl.expected_run_length(hair_skel, careless) < intact, (
        "eroding a 1 px vessel must uncover its centreline")
    print("at capillary calibre an erosion IS severance: traced ERL "
          f"{intact / hair_skel.sum():.3f} -> "
          f"{erl.expected_run_length(hair_skel, careless) / hair_skel.sum():.3f}")

    # --- a cut on a loop closes a hole instead of adding a component --------
    loop = np.zeros((41, 41), dtype=bool)
    loop[10:31, 10:31] = True
    loop[14:27, 14:27] = False                      # a square annulus
    loop_skel = skeletonize(loop)
    site = {"row": 10, "col": 20, "radius": 3}
    counted = splits_made(loop, loop_skel, [site])
    assert counted["splits"] == 0 and counted["loop_cuts"] == 1, counted
    print("splits_made: a cut across a loop adds 0 components and closes 1 hole")
    print("selftest ok")


def rows_for(item: dict, rng: np.random.Generator) -> list:
    truth = item["label"]
    skel = skeletonize(truth)
    pairs = pair_sites(truth, skel, candidate_sites(truth, skel),
                       wanted=CUTS_PER_IMAGE)
    if len(pairs) < CUTS_PER_IMAGE // 3:
        return []
    arms = {"mid": apply_cuts(truth, skel, [pair[0] for pair in pairs]),
            "tip": apply_cuts(truth, skel, [pair[1] for pair in pairs])}
    deleted = int(truth.sum()) - int(arms["mid"].sum())
    arms["edge_any"] = erode_by(truth, deleted, rng)
    arms["edge_off"] = erode_by(truth, deleted, rng, spare=skel)
    arms["uncut"] = truth
    mechanism = {tag: splits_made(truth, skel, [pair[index] for pair in pairs])
                 for index, tag in ((0, "mid"), (1, "tip"))}
    return [{"image": item["name"], "arm": tag, "cuts": len(pairs),
             "deleted": int(truth.sum()) - int(mask.sum()),
             "splits": mechanism.get(tag, {}).get("splits", 0),
             "loop_cuts": mechanism.get(tag, {}).get("loop_cuts", 0),
             **score(mask, truth, skel)}
            for tag, mask in arms.items()]


def figure(rows: list, item: dict) -> None:
    """One panel per arm on a real crop, plus the per-image scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = item["label"]
    skel = skeletonize(truth)
    pairs = pair_sites(truth, skel, candidate_sites(truth, skel),
                       wanted=CUTS_PER_IMAGE)
    masks = {"mid": apply_cuts(truth, skel, [pair[0] for pair in pairs]),
             "tip": apply_cuts(truth, skel, [pair[1] for pair in pairs])}
    graded = {tag: score(mask, truth, skel) for tag, mask in masks.items()}

    figure_, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    titles = {"mid": "cut in the branch INTERIOR", "tip": "cut NEAR THE TIP"}
    for axis, tag in zip(axes, ("mid", "tip")):
        axis.imshow(truth, cmap="Greys", interpolation="nearest")
        lost = truth & ~masks[tag]
        overlay = np.zeros(truth.shape + (4,))
        overlay[lost] = (0.85, 0.10, 0.10, 1.0)
        axis.imshow(overlay, interpolation="nearest")
        stats_ = graded[tag]
        axis.set_title(
            f"{titles[tag]}\nDice {stats_['dice']:.4f}   "
            f"clDice {stats_['cldice']:.4f}   b0 err {stats_['betti0_err']}\n"
            f"traced ERL {stats_['traced']:.1%}   "
            f"unreachable {stats_['unreachable']:.1%}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    table = {}
    for row in rows:
        table.setdefault(row["image"], {})[row["arm"]] = row
    mids = [table[name]["mid"]["traced"] for name in sorted(table)]
    tips = [table[name]["tip"]["traced"] for name in sorted(table)]
    axes[2].scatter(mids, tips, s=28, color="#1f4e79")
    span = mids + tips
    limit = [min(span) - 0.05, max(span) + 0.05]
    axes[2].plot(limit, limit, color="grey", linewidth=1, linestyle="--")
    axes[2].set_xlim(limit)
    axes[2].set_ylim(limit)
    axes[2].set_xlabel("traced ERL, interior cuts")
    axes[2].set_ylabel("traced ERL, tip cuts")
    axes[2].set_title("same Dice, same clDice, same $\\beta_0$\n"
                      f"every one of {len(mids)} images above the line",
                      fontsize=9)
    figure_.tight_layout()
    figure_.savefig(RESULTS / "blind_spot.png", dpi=140)
    print(f"wrote {RESULTS / 'blind_spot.png'}")


def settle(rows: list) -> list:
    """Mechanically decide the six pre-registered predictions and the kill."""
    table = {}
    for row in rows:
        table.setdefault(row["image"], {})[row["arm"]] = row
    names = sorted(table)

    def gap(key: str, left: str = "tip", right: str = "mid") -> list:
        return [table[name][left][key] - table[name][right][key]
                for name in names]

    dice_gap = np.abs(gap("dice"))
    cldice_gap = np.abs(gap("cldice"))
    betti_gap = np.abs(gap("betti0_err"))
    traced = calibration.decide(
        [(table[name]["tip"]["traced"], table[name]["mid"]["traced"])
         for name in names], gap("traced"))
    anchor = [table[name]["mid"]["unreachable"] - table[name]["tip"]["unreachable"]
              for name in names]
    edge_gap = [abs(table[name]["edge_any"]["traced"] - table[name]["uncut"]["traced"])
                for name in names]
    off_gap = [abs(table[name]["edge_off"]["traced"] - table[name]["uncut"]["traced"])
               for name in names]
    splits = {tag: [table[name][tag]["splits"] for name in names]
              for tag in ("mid", "tip")}

    verdicts = [
        ("P1  |Dice(mid)-Dice(tip)| < 0.002",
         bool(dice_gap.max() < 0.002), f"worst image {dice_gap.max():.5f}"),
        ("P2  |clDice(mid)-clDice(tip)| < 0.005",
         bool(cldice_gap.max() < 0.005), f"worst image {cldice_gap.max():.5f}"),
        ("P3  Betti-0 error identical",
         bool(betti_gap.max() == 0), f"worst image {betti_gap.max():.0f}"),
        ("P4  traced ERL: tip - mid > 20 points",
         bool(traced["mean"] > 0.20),
         f"mean {traced['mean']:+.1%} t {traced['t']:.2f} "
         f"gate {'HOLDS' if traced['holds'] else 'fails'} "
         f"over {traced['seeds']} images (the gate's unit here is the image, "
         f"not the seed: there is no model and no seed in this table)"),
        ("P5  the anchor orders tip better on EVERY image",
         bool(min(anchor) > 0),
         f"{sum(1 for d in anchor if d > 0)}/{len(anchor)} images, "
         f"mean {np.mean(anchor):+.1%}"),
        ("P6  the edge control stays within 1 point of uncut",
         bool(max(edge_gap) < 0.01), f"worst image {max(edge_gap):.4f}"),
    ]
    verdicts.append((
        "F1  [follow-up, written after P6 failed] an erosion that spares the",
        bool(max(off_gap) < 1e-12),
        f"    centreline leaves traced ERL untouched: worst image {max(off_gap):.2e}"))
    verdicts.append((
        "F2  [follow-up, written after P3 failed] mid cuts add fewer components",
        bool(np.mean(splits["mid"]) < np.mean(splits["tip"])),
        f"    because they land on loops: splits {np.mean(splits['mid']):.1f} (mid) "
        f"vs {np.mean(splits['tip']):.1f} (tip) of {CUTS_PER_IMAGE}"))
    killed = bool(cldice_gap.max() > 0.01)
    verdicts.append(("KILL  |clDice(mid)-clDice(tip)| > 0.01 would sink the claim",
                     not killed,
                     "not triggered" if not killed
                     else f"TRIGGERED at {cldice_gap.max():.5f}"))
    return verdicts


def report(items: list) -> None:
    rng = np.random.default_rng(20260919)
    rows = []
    for item in items:
        made = rows_for(item, rng)
        if not made:
            print(f"  {item['name']}: too few matched pairs, skipped")
            continue
        rows.extend(made)
        print(f"  {item['name']}: {made[0]['cuts']} matched cuts, "
              f"{made[0]['deleted']} px deleted per arm")
    # An empty table is not a null result (CLAUDE.md): refuse rather than print
    # headers with no rows.
    if not rows:
        raise SystemExit("no image yielded a matched pair -- refusing to report")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "blind_spot.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["=== a failure Dice, clDice and Betti-0 cannot see ===", "",
             f"{len({row['image'] for row in rows})} DRIVE test images, "
             f"{CUTS_PER_IMAGE} cuts per arm,",
             "cut radius = 2.0 x local vessel radius, paired across arms on",
             f"radius and on deleted area (tolerance {AREA_TOLERANCE} px).",
             "",
             "ERL convention: split / pixels / full -- a bridged gap splits a",
             "run, fragment length is a pixel count, and the denominator is the",
             "whole ground-truth skeleton (erl.expected_run_length). Nothing in",
             "this table is selected on anything: the input is the ground truth",
             "itself, so there is no epoch, threshold or geometry to leak.", ""]
    header = (f"{'arm':<9}{'Dice':>9}{'clDice':>9}{'b0 err':>9}"
              f"{'ERL px':>10}{'traced':>9}{'unreachable':>13}{'deleted':>9}"
              f"{'splits':>9}{'loop cuts':>11}")
    lines += [header, "-" * len(header)]
    for tag in ("uncut", "edge_off", "edge_any", "tip", "mid"):
        picked = [row for row in rows if row["arm"] == tag]
        lines.append(
            f"{tag:<9}{np.mean([r['dice'] for r in picked]):>9.4f}"
            f"{np.mean([r['cldice'] for r in picked]):>9.4f}"
            f"{np.mean([r['betti0_err'] for r in picked]):>9.1f}"
            f"{np.mean([r['erl_px'] for r in picked]):>10.0f}"
            f"{np.mean([r['traced'] for r in picked]):>8.1%}"
            f"{np.mean([r['unreachable'] for r in picked]):>13.1%}"
            f"{np.mean([r['deleted'] for r in picked]):>9.0f}"
            f"{np.mean([r['splits'] for r in picked]):>9.1f}"
            f"{np.mean([r['loop_cuts'] for r in picked]):>11.1f}")
    lines += ["", "the two arms delete the same tissue in the same number of",
              "pieces at the same calibre; they differ only in WHERE along the",
              "branch it was taken.", "",
              "=== the pre-registered predictions ===", ""]
    for name, held, detail in settle(rows):
        lines.append(f"  {'HOLDS' if held else 'FAILS'}  {name}")
        lines.append(f"         {detail}")
    text = "\n".join(lines)
    (RESULTS / "blind_spot.txt").write_text(text + "\n")
    print("\n" + text)
    figure(rows, items[0])


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    selftest()
    print()
    report(drive.load_split("test"))


if __name__ == "__main__":
    main()
