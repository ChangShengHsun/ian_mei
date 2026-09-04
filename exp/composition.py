"""Does the direction field add anything ON TOP of simply lowering the
threshold -- and does restricting it to endpoints fix its cost?

WRITTEN 2026-09-01, after threshold_control.py settled that the oriented
dilation layer LOSES to a dev-picked lower threshold on all ten arms under
BOTH ERL conventions, by 2.1-6.4 points (convention B) and 4.5-12.2 points
(convention A). Robust to a +0.005 and +0.010 Dice margin: 0/10 either way.

So the layer is not a competitor to thresholding. Two questions remain, and
they are the ones worth GPU-free hours:

  1. COMPOSITION. Thresholding and oriented dilation spend the same currency
     (Dice) on different things. Applied at the arm's dev-picked threshold
     rather than at 0.5, does the field still buy ERL over isotropic dilation
     -- and does threshold+layer beat threshold alone? If yes, the field is a
     complement and the paper is about the pair. If no, the field is a worse
     way to spend Dice and this line is finished.

  2. SCOPE. The layer dilates EVERY foreground pixel. Breaks live at skeleton
     endpoints, which are 0.395% of ground-truth foreground (measured on five
     DRIVE test images: 31,477 px of foreground, 124 px of endpoints). We pay
     100% of the Dice to fix 0.4% of the image. `endpoint` grows only from
     the endpoints of the predicted skeleton, along the same field, so its
     Dice cost is O(endpoints) instead of O(foreground).

Every source is read at the SAME base threshold, so the comparison isolates
the operator. Geometry is chosen on the 5 dev images and read on the 20 test
images -- the third level of the protocol leak, caught 2026-09-01.

  python exp/composition.py --selftest
  python exp/composition.py --dev --shard 0/4
  python exp/composition.py --shard 0/4
  python exp/composition.py --report

Writes results/heldout/composition[_dev][.shardIofN].csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anisotropic
import cross_dataset
import drive
import hole_sweep
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import threshold_control as control
import train

ARMS = control.ARMS
DEFAULT_FIELD = sweep.DEFAULT_FIELD

# The same grid postproc_ceiling used, so a row here is comparable to a row
# there by construction. Both are in multiples of median vessel width.
ALONG, ACROSS = sweep.ALONG, sweep.ACROSS
ISOTROPIC = sweep.ISOTROPIC

# The Dice a base threshold is allowed to give up against the arm at 0.5.
# The same 0.02 the postproc table's tightest budget uses, so "lower the
# threshold" and "dilate" are handed the same allowance and the comparison
# is about what each does with it.
BASE_BUDGET = 0.02


def endpoints_of(mask: np.ndarray) -> np.ndarray:
    """Skeleton pixels with exactly one neighbour -- where a break shows.

    Counted on the 8-neighbourhood with the pixel itself included, so an
    endpoint has a count of 2. An isolated speck counts 1 and is NOT an
    endpoint: it has no direction to continue and growing it is how a
    dilation baseline buys ERL without knowing anything.
    """
    skel = skeletonize(mask)
    counts = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                              mode="constant")
    return skel & (counts == 2)


def endpoint_growth(mask, ends, sin2, cos2, along, across, fov):
    """The mask, plus oriented growth FROM ITS ENDPOINTS ONLY."""
    if along < 0.5 and across < 0.5:
        return mask.copy()
    grown = anisotropic.oriented_dilation(ends, sin2, cos2, along, across)
    return (mask | grown) & fov


def base_threshold(dev_rows, config: str, metric: str) -> float:
    """The arm's own operating point, chosen on dev: the threshold that
    traces furthest while giving up at most BASE_BUDGET of Dice against the
    same arm at 0.5.

    This is the configuration that BEAT the layer, so it is the base every
    operator here is applied on top of. Picking it on test would hand the
    baseline the same advantage the layer was just corrected for.
    """
    curve = control.curve(dev_rows, config, metric)
    if 0.5 not in curve:
        return 0.5
    floor = curve[0.5][0] - BASE_BUDGET
    chosen = control.best_within(curve, floor)
    return 0.5 if chosen is None else chosen


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    for config in ARMS:
        assert config in train.CONFIGS, config
    assert 0.0 in ALONG and 0.0 in ACROSS

    # 1. AN ENDPOINT IS AN ENDPOINT. A straight segment has exactly two; a
    #    closed ring has none; a single speck has none either.
    line = np.zeros((20, 40), dtype=bool)
    line[10, 5:35] = True
    assert endpoints_of(line).sum() == 2, endpoints_of(line).sum()
    speck = np.zeros((20, 20), dtype=bool)
    speck[10, 10] = True
    assert endpoints_of(speck).sum() == 0
    ring = np.zeros((40, 40), dtype=bool)
    yy, xx = np.mgrid[0:40, 0:40]
    radius = np.hypot(yy - 20, xx - 20)
    ring |= (radius > 11) & (radius < 14)
    assert endpoints_of(ring).sum() == 0, endpoints_of(ring).sum()
    print("endpoints: straight segment 2, ring 0, isolated speck 0")

    # 2. THE WHOLE POINT: at the SAME geometry, growing from endpoints must
    #    cost far less foreground than growing from every pixel. If it does
    #    not, `endpoint` is just `predicted` with extra steps.
    #
    #    A HORIZONTAL vessel on purpose. On a 45-degree one the 16 orientation
    #    bins put the axis at 50.6 degrees, and at long reach that 5.6-degree
    #    error walks the growth off the vessel and SHATTERS it -- measured
    #    here 2026-09-01: whole-mask dilation of a broken diagonal goes from
    #    2 components to 5 between reach 6 and 7. That is a real property of
    #    the operator and it belongs in the sweep, not inside a test whose
    #    job is to check the endpoint restriction.
    import break_lengths
    import direction
    size = 81
    vessel = np.zeros((size, size), dtype=bool)
    vessel[39:42, 15:65] = True
    broken = vessel.copy()
    broken[:, 38:44] = False
    sin2, cos2, _ = direction.tangent_field(vessel)
    fov = np.ones_like(vessel)
    full = anisotropic.oriented_dilation(broken, sin2, cos2, 4.0, 0.0) & fov
    ends = endpoints_of(broken)
    tips = endpoint_growth(broken, ends, sin2, cos2, 4.0, 0.0, fov)
    added_full = int(full.sum() - broken.sum())
    added_tips = int(tips.sum() - broken.sum())
    print(f"a 6px gap in a horizontal vessel, reach 4: whole-mask dilation "
          f"adds {added_full} px, endpoint-only adds {added_tips} px "
          f"({added_tips / max(added_full, 1):.1%} of the cost)")
    assert 0 < added_tips < added_full / 2, (added_tips, added_full)

    # 3. AND IT MUST STILL REACH. A cheaper operator that no longer closes
    #    the gap has answered a different question. Counted on CONN8, the
    #    repo's connectivity everywhere (erl.py:63, link_ceiling.py:113) --
    #    the default 4-connectivity calls every pixel of a 1 px diagonal its
    #    own component, which read as "endpoint growth shatters the vessel"
    #    on the first run of this test.
    before = ndimage.label(broken, structure=break_lengths.CONN8)[1]
    after = ndimage.label(tips, structure=break_lengths.CONN8)[1]
    print(f"components {before} -> {after} at {added_tips} px of new "
          f"foreground (whole-mask needs {added_full} px for the same close)")
    assert after < before, (before, after)

    # 4. THE BASE THRESHOLD IS PICKED ON DEV, AND IT IS THE ARM'S OWN. Build
    #    a curve where 0.5 is not the best and check it moves.
    rows = []
    for threshold, dice, traced in ((0.5, 0.820, 0.40), (0.3, 0.805, 0.55),
                                    (0.1, 0.780, 0.70)):
        rows.append({"config": "X", "threshold": threshold, "dice": dice,
                     "erl_split": traced, "erl_bridged": traced})
    got = base_threshold(rows, "X", "erl_split")
    print(f"base threshold on this dev curve: {got} "
          f"(0.1 costs {0.820 - 0.780:.3f} Dice, over the {BASE_BUDGET} budget)")
    assert got == 0.3, got
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def load(split: str = "test") -> list[dict]:
    stem = "composition" if split == "test" else "composition_dev"
    rows = []
    found = sorted(heldout.ROOT.glob("composition*.csv"))
    is_dev = [p for p in found if p.name.startswith("composition_dev")]
    for path in (is_dev if split == "dev" else
                 [p for p in found if p not in is_dev]):
        for row in csv.DictReader(path.open()):
            rows.append({**row,
                         "along": float(row["along"]),
                         "across": float(row["across"]),
                         "dice": float(row["dice"]),
                         "erl_split": float(row["erl_split"]),
                         "erl_bridged": float(row["erl_bridged"])})
    return rows


def raw_of(rows, config):
    these = [r for r in rows if r["config"] == config and r["source"] == "raw"]
    if not these:
        return None
    return float(np.mean([r["dice"] for r in these]))


def pick(rows, config, source, floor, metric, budget):
    """Best geometry for one source within a Dice budget, on the DEV rows."""
    best = None
    for along in ALONG:
        for across in ACROSS:
            cells = [r for r in rows if r["config"] == config
                     and r["source"] == source
                     and r["along"] == along and r["across"] == across]
            if not cells:
                continue
            dice = float(np.mean([r["dice"] for r in cells]))
            if dice < floor - budget:
                continue
            traced = float(np.mean([r[metric] for r in cells]))
            if best is None or traced > best[1]:
                best = ((along, across), traced)
    return None if best is None else best[0]


def report() -> None:
    import calibration
    rows, dev_rows = load("test"), load("dev")
    if not rows or not dev_rows:
        raise SystemExit("need composition*.csv AND composition_dev*.csv")
    print("=== the field applied on top of the threshold that beat it ===\n")
    print(f"{len(rows)} test rows, {len(dev_rows)} dev rows")
    print("Every source is read at the SAME base threshold -- the arm's own,")
    print("chosen on dev at a 0.02 Dice budget. Geometry chosen on the 5 dev")
    print("images, read on the 20 test. `raw` is the threshold ALONE, which")
    print("is the baseline that has to be beaten, not `isotropic`.\n")
    for metric, label in (("erl_split", "convention A (a bridged gap splits)"),
                          ("erl_bridged", "convention B (it does not)")):
        print(f"--- {label} ---")
        for config in ARMS:
            floor = raw_of(dev_rows, config)
            if floor is None or raw_of(rows, config) is None:
                continue
            base = sorted({r["threshold"] for r in rows
                           if r["config"] == config})
            traced = float(np.mean([r[metric] for r in rows
                                    if r["config"] == config
                                    and r["source"] == "raw"]))
            print(f"    {config}  threshold {','.join(base)}  "
                  f"raw {traced:.1%} traced at Dice "
                  f"{raw_of(rows, config):.4f}")
            for budget in (0.02, 0.05):
                cells = []
                for source in ("predicted", "endpoint", "isotropic",
                               "shuffled"):
                    setting = pick(dev_rows, config, source, floor, metric,
                                   budget)
                    if setting is None:
                        cells.append(f"{source} {'--':>18}")
                        continue
                    mine = {(r["seed"], r["image"]): r[metric] for r in rows
                            if r["config"] == config
                            and r["source"] == source
                            and (r["along"], r["across"]) == setting}
                    theirs = {(r["seed"], r["image"]): r[metric] for r in rows
                              if r["config"] == config
                              and r["source"] == "raw"}
                    keys = sorted(set(mine) & set(theirs))
                    seeds = sorted({s for s, _ in keys})
                    if len(seeds) < 3:
                        cells.append(f"{source} {'--':>18}")
                        continue
                    got = calibration.decide(
                        [(mine[k], theirs[k]) for k in keys],
                        [float(np.mean([mine[k] - theirs[k]
                                        for k in keys if k[0] == s]))
                         for s in seeds])
                    cells.append(f"{source} {got['mean']:+.1%} "
                                 f"t{got['t']:5.2f} "
                                 f"{'HOLDS' if got['holds'] else 'fails'}")
                print(f"      -{budget:.2f}  " + "   ".join(cells))
        print()
    print("Read as `source - raw`: what the operator adds ON TOP of the")
    print("threshold, not on top of 0.5. `shuffled` is the control and must")
    print("fail; `isotropic` is what direction has to beat; `raw` winning")
    print("everywhere means the field spends Dice worse than the threshold.")


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
    split = "dev" if "--dev" in sys.argv else "test"
    stem = "composition" if split == "test" else "composition_dev"
    target = heldout.ROOT / (f"{stem}.csv" if shard is None else
                             f"{stem}.shard{shard[0]}of{shard[1]}.csv")

    thresholds = control.load("dev")
    if not thresholds:
        raise SystemExit("no threshold_control_dev*.csv -- the base threshold "
                         "is chosen from it. Run exp/threshold_control.py "
                         "--dev first.")
    epochs = heldout.chosen_epochs()
    found = sorted(heldout.ROOT.glob(f"{stem}*.csv"))
    is_dev = [p for p in found if p.name.startswith("composition_dev")]
    done = set()
    for existing in (is_dev if split == "dev" else
                     [p for p in found if p not in is_dev]):
        done |= {(r["config"], r["seed"]) for r in csv.DictReader(existing.open())}

    items = drive.load_split(split)
    data = train.stack_split("fit")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(i["label"] & i["fov"]),
                 "truth": i["label"] & i["fov"]} for i in items]
    seeds = sorted({r.rsplit("_s", 1)[1] for r in epochs
                    if r.rsplit("_s", 1)[0] == DEFAULT_FIELD
                    and (heldout.ROOT / r / "final.pt").exists()})
    print(f"{split}: {len(ARMS)} arms, field from {DEFAULT_FIELD} "
          f"({len(seeds)} seeds), width {width:.2f} px", flush=True)
    fields = sweep.shared_fields(DEFAULT_FIELD, epochs, items, data, seeds)

    fresh = not target.exists()
    with target.open("a", newline="") as handle:
        writer = None
        for config in ARMS:
            base = base_threshold(thresholds, config, "erl_bridged")
            runs = sorted(r for r in epochs
                          if r.rsplit("_s", 1)[0] == config
                          and (heldout.ROOT / r / "final.pt").exists())
            for run in runs:
                seed = run.rsplit("_s", 1)[1]
                if (config, seed) in done or seed not in seeds:
                    continue
                if shard is not None and \
                        abs(hash((config, seed))) % shard[1] != shard[0]:
                    continue
                model, mean, std = sweep.load_model(run, config, epochs, data)
                if model is None:
                    continue
                out = []
                for item, geo in zip(items, geometry):
                    prob = train.predict_full(model, item["image"], mean, std)
                    pred = speckle.drop_small((prob >= base) & item["fov"],
                                              component_px)
                    ends = endpoints_of(pred)
                    common = {"config": config, "run": run, "seed": seed,
                              "threshold": base, "image": item["name"],
                              "field_arm": DEFAULT_FIELD}
                    out.append({**common, "source": "raw", "along": 0.0,
                                "across": 0.0,
                                **sweep.measure(pred, geo["skel"],
                                                geo["truth"])})
                    for radius in ISOTROPIC:
                        if radius == 0.0:
                            continue
                        grown = anisotropic.isotropic_dilation(
                            pred, radius * width) & item["fov"]
                        out.append({**common, "source": "isotropic",
                                    "along": radius, "across": radius,
                                    **sweep.measure(grown, geo["skel"],
                                                    geo["truth"])})
                    predicted = fields.get((seed, item["name"]))
                    if predicted is None:
                        continue
                    shuffled = anisotropic.shuffled_field(
                        item["label"].shape,
                        seed=abs(hash((run, item["name"]))) % 2**31)
                    for source, (sin2, cos2) in (("predicted", predicted),
                                                 ("shuffled", shuffled)):
                        for along in ALONG:
                            for across in ACROSS:
                                if along == 0.0 and across == 0.0:
                                    continue
                                grown = anisotropic.oriented_dilation(
                                    pred, sin2, cos2, along * width,
                                    across * width) & item["fov"]
                                out.append({**common, "source": source,
                                            "along": along, "across": across,
                                            **sweep.measure(grown, geo["skel"],
                                                            geo["truth"])})
                    sin2, cos2 = predicted
                    for along in ALONG:
                        for across in ACROSS:
                            if along == 0.0 and across == 0.0:
                                continue
                            grown = endpoint_growth(
                                pred, ends, sin2, cos2, along * width,
                                across * width, item["fov"])
                            out.append({**common, "source": "endpoint",
                                        "along": along, "across": across,
                                        **sweep.measure(grown, geo["skel"],
                                                        geo["truth"])})
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(out[0]))
                    if fresh:
                        writer.writeheader()
                writer.writerows(out)
                handle.flush()
                print(f"  {run} at threshold {base} done", flush=True)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
