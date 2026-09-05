"""D1's ceiling: how much of the budget can knowing the vessel's axis reach?

WRITTEN AND SELFTESTED 2026-08-27, BEFORE IT SCORED A SINGLE PREDICTION. This
is the same move that retired C1 in two hours: measure the bound before
building the method.

THE BUDGET, decomposed this morning on K_focal_aug at rule (iv):

    raw, as erl.py scores it                            47.4%
    raw, not splitting runs the prediction bridges      67.3%   (+19.9 is a
                                                                 convention)
    every intact break filled from ground truth         84.0%   (+16.7 real)
    every break filled from ground truth                97.8%   (+ 5.1 severs)

So the reachable prize is about 21.8 points, and the shape of the error is
known: the prediction runs BESIDE the centreline, offset a median 1.4 px on
2.8 px vessels, over stretches whose 90th percentile is 21 px.

THE FOUR CONDITIONS, and what each one rules out.

  oracle      oriented dilation driven by the GROUND TRUTH tangent field.
              The ceiling for any method built on direction, however good the
              predictor gets.
  predicted   the same, driven by a trained _dir head. The gap to `oracle` is
              how much of the prize is lost to the predictor rather than to
              the mechanism.
  shuffled    the same, driven by a random per-pixel axis field. Oriented
              dilation ADDS FOREGROUND, and adding foreground raises ERL on
              its own -- that is exactly how the closing baseline beat the C1
              oracle until its Dice cost was matched. If shuffled scores like
              oracle, this measures dilation and not direction.
  isotropic   plain dilation, swept over radii, reported AT MATCHED
              FOREGROUND against the oriented conditions. This, not `raw`, is
              what the mechanism has to beat.

Both radii are swept, in multiples of the median vessel width, and the sweep
IS the experiment: if `along` alone captures the budget the fix is extending
vessels lengthways; if `across` is needed the vessel is drawn in the wrong
place. A design that could only express one would answer by assuming.

Every number is reported under BOTH erl.py conventions, because 19.9 of the
36.6 points turned out to be the splitting rule and a table that hides which
is which cannot be acted on.

Radii are chosen on the SELECTION half (odd images) and reported on the other.

  python exp/direction_ceiling.py --selftest
  python exp/direction_ceiling.py                 # every arm
  python exp/direction_ceiling.py K_focal_aug

Writes results/selection_sweep/direction_ceiling.csv.
"""
import csv
import sys
import zlib
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anisotropic
import cross_dataset
import direction
import drive
import erl
import erl_convention
import hole_sweep
import link_ceiling
import score_direction
import select_checkpoint as rules_module
import speckle
import summarize_selection as selection
import train

OUT = selection.SWEEP / "direction_ceiling.csv"
RULE = "(iv) best clDice"

# In multiples of the median vessel width (2.8 px on DRIVE). Pre-registered.
# `along` reaches to 2 widths because the measured stretches run to 21 px,
# about 7 widths, and a dilation only has to close half a gap from each side.
# `across` stops at 1 width because the measured offset is half a width, and
# a correction reaching further than the vessel is wide is thickening, not
# repositioning -- the Dice column would show it, but it should not be in the
# grid at all.
ALONG = (0.0, 0.5, 1.0, 1.5, 2.0)
ACROSS = (0.0, 0.25, 0.5, 1.0)
ISOTROPIC = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def stable_seed(*parts) -> int:
    """A reproducible seed for the shuffled control.

    zlib.crc32, not hash(): Python randomises str/tuple hashes per process, so
    the same run scored twice drew a DIFFERENT random field and the control
    column could not be reproduced. Same defect as the sharding bug of
    2026-09-01, in a place where it degraded reproducibility rather than
    coverage.
    """
    return zlib.crc32("|".join(str(part) for part in parts).encode())


def fields_for(run: str, config: str, item: dict, model, data,
               mean, std) -> dict:
    """The three axis fields this run is scored under."""
    truth = direction.tangent_field(item["label"] & item["fov"])
    out = {"oracle": (truth[0], truth[1]),
           "shuffled": anisotropic.shuffled_field(
               item["label"].shape, seed=stable_seed(run, item["name"]))}
    if train.uses_direction(config):
        out["predicted"] = score_direction.predict_field(
            model, item["image"], mean, std)
    return out


def measure(mask, skel, truth) -> dict:
    return {"erl_split": round(
                erl.expected_run_length(skel, mask) / skel.sum(), 5),
            "erl_bridged": round(
                erl_convention.bridged_run_length(skel, mask) / skel.sum(), 5),
            "dice": round(link_ceiling.dice(mask, truth), 5),
            "fg": int(mask.sum())}


def selftest() -> None:
    # The grid must contain the do-nothing point, or "did it help" has no
    # reference inside the sweep itself.
    assert 0.0 in ALONG and 0.0 in ACROSS and 0.0 in ISOTROPIC
    print(f"grid: along {ALONG}, across {ACROSS}, isotropic {ISOTROPIC} "
          f"-- {len(ALONG) * len(ACROSS)} oriented + {len(ISOTROPIC)} "
          f"isotropic settings per image")

    # Both ERL conventions must be reported, and they must differ on a
    # prediction that detours around the centreline -- otherwise the split of
    # the budget into "real" and "convention" cannot be read off this table.
    skel = np.zeros((30, 100), dtype=bool)
    skel[15, 5:95] = True
    detour = np.zeros_like(skel)
    detour[14:17, 5:95] = True
    detour[:, 45:55] = False
    detour[17:20, 44:56] = True
    got = measure(detour, skel, skel)
    print(f"a bridged detour scores split {got['erl_split']:.1%} vs bridged "
          f"{got['erl_bridged']:.1%} -- both are reported, always")
    assert got["erl_bridged"] > got["erl_split"], got

    # Oriented dilation on a real-shaped vessel must beat isotropic dilation
    # AT MATCHED FOREGROUND. If this fails on a clean synthetic vessel it
    # cannot possibly hold on a retina, and the experiment is not worth
    # running.
    size = 81
    yy, xx = np.mgrid[0:size, 0:size]
    vessel = np.abs(xx - yy) <= 1.0
    vessel &= (xx > 15) & (xx < 65)
    sin2, cos2, _ = direction.tangent_field(vessel)
    broken = vessel.copy()
    broken[:, 38:44] = False          # a gap along the vessel
    truth = vessel

    oriented = anisotropic.oriented_dilation(broken, sin2, cos2, 4.0, 0.0)
    o = measure(oriented, skeletonize(truth), truth)

    # Matched at EQUAL TRACED DISTANCE, not equal foreground. Equal foreground
    # is not reachable on a grid of integer-ish radii -- the first attempt at
    # this test "matched" 161 px of oriented dilation against a 129 px radius
    # that dilates nothing at all, and then read the no-op's untouched Dice as
    # isotropic winning. Asking how much paint each needs to trace the same
    # distance is well defined at every radius and is the question anyway.
    needed = None
    for radius in np.arange(0.5, 10.0, 0.25):
        got_mask = anisotropic.isotropic_dilation(broken, radius)
        got = measure(got_mask, skeletonize(truth), truth)
        if got["erl_split"] >= o["erl_split"]:
            needed = (radius, got)
            break
    print(f"a 6px gap in a 45-degree vessel:")
    print(f"  oriented  traced {o['erl_split']:.1%}  Dice {o['dice']:.3f}  "
          f"{o['fg']} px of foreground")
    assert needed is not None, "no isotropic radius reached the oriented trace"
    radius, i = needed
    print(f"  isotropic needs radius {radius} to trace as far: "
          f"{i['erl_split']:.1%}  Dice {i['dice']:.3f}  {i['fg']} px")
    assert i["fg"] > o["fg"], (o, i)
    assert i["dice"] < o["dice"], (o, i)
    print(f"  -> isotropic pays {i['fg'] / o['fg']:.2f}x the foreground and "
          f"{o['dice'] - i['dice']:.3f} more Dice for the same trace")

    # And the shuffled control must NOT reproduce it, or the gain is dilation.
    s_sin, s_cos = anisotropic.shuffled_field(broken.shape, seed=0)
    shuffled = anisotropic.oriented_dilation(broken, s_sin, s_cos, 4.0, 0.0)
    s = measure(shuffled, skeletonize(truth), truth)
    print(f"  shuffled  traced {s['erl_split']:.1%}  Dice {s['dice']:.3f} "
          f"({s['fg']} px) -- the control that says it is direction, not "
          f"foreground")
    assert o["erl_split"] > s["erl_split"], (o, s)
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    # Four arms, not six. The ceiling is a question about the MECHANISM, and
    # K_focal_aug (the best arm, and the one the decision is about) plus
    # A_dice (the baseline) answer it; the two _dir arms are here because
    # they are the only runs that can supply a PREDICTED field. Scoring
    # G_focal and H_aug as well would add two hours and no new answer.
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or \
        ["K_focal_aug", "A_dice", "A_dice_dir", "H_aug_dir"]

    points = selection.selection_points(selection.load())
    rule = dict(rules_module.rules())[RULE]
    items = drive.load_split("val")
    data = train.stack_split("train")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(item["label"] & item["fov"]),
                 "truth": item["label"] & item["fov"]} for item in items]
    print(f"{len(wanted)} arm(s), width {width:.2f} px, component filter "
          f"{component_px} px", flush=True)

    rows = []
    for config in wanted:
        runs = sorted(r for r in points if r.rsplit("_s", 1)[0] == config)
        if not runs:
            print(f"[{config}] not in the scores CSV; skipping", flush=True)
            continue
        for run in runs:
            epoch = rule(points[run])["epoch"]
            model = train.build_model(config)
            model.load_state_dict(train.load_checkpoint(
                selection.SWEEP / run / f"epoch{epoch:03d}.pt")["model"])
            model.eval()
            mean, std = train.normalisation(run, data)
            for item, geo in zip(items, geometry):
                prob = train.predict_full(model, item["image"], mean, std)
                pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                          component_px)
                common = {"config": config, "run": run, "epoch": epoch,
                          "seed": run.rsplit("_s", 1)[1],
                          "image": item["name"]}
                rows.append({**common, "source": "raw", "along": 0.0,
                             "across": 0.0,
                             **measure(pred, geo["skel"], geo["truth"])})
                for radius in ISOTROPIC:
                    if radius == 0.0:
                        continue
                    grown = anisotropic.isotropic_dilation(
                        pred, radius * width) & item["fov"]
                    rows.append({**common, "source": "isotropic",
                                 "along": radius, "across": radius,
                                 **measure(grown, geo["skel"], geo["truth"])})
                fields = fields_for(run, config, item, model, data, mean, std)
                for source, (sin2, cos2) in fields.items():
                    for along in ALONG:
                        for across in ACROSS:
                            if along == 0.0 and across == 0.0:
                                continue
                            grown = anisotropic.oriented_dilation(
                                pred, sin2, cos2, along * width,
                                across * width) & item["fov"]
                            rows.append({**common, "source": source,
                                         "along": along, "across": across,
                                         **measure(grown, geo["skel"],
                                                   geo["truth"])})
            print(f"  {run} epoch {epoch} done", flush=True)

    if not rows:
        raise SystemExit("nothing scored")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
