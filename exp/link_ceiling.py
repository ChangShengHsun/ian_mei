"""C1.0: how much run length is a perfect fragment linker even worth?

WRITTEN AND SELFTESTED 2026-08-27, BEFORE THE FIRST LINKER LINE OF CODE. This
is the cheap step that can kill C1 outright, and it exists because E15's
ceiling calculation was the best-value hour this series has spent.

C1 proposes to link predicted fragments across the breaks that actually sever
connectivity -- E10 measured that only about 7% of breaks do, and
break_lengths.classify names which ones. Before building a linker, measure
what linking them PERFECTLY would buy. Four conditions, per run, per image:

  raw          the prediction as it stands, after E4's component filter.
  closing      E9b's blind morphological closing, at the radius that was best
               on the SELECTION half. This, NOT `raw`, is what any linker has
               to beat: E4 claimed a loss beat filtering and E9's hole sweep
               overturned it, and no method claim in this repo has survived
               without a strong post-processing baseline beside it.
  oracle_sever every SEVERING break filled in from the ground truth. The
               ceiling for C1 as specified: a linker that finds every real
               disconnection and joins it along the true path.
  oracle_all   every missed centreline pixel filled in. The ceiling for any
               post-processing at all, linking or not.

Reading the result:

  raw -> oracle_sever small        C1 is not worth building. The tree is not
                                   fragmented so much as unseen, and the
                                   budget belongs to D1 or B1.
  raw -> oracle_sever large, and
  closing already takes most of it C1 is worth little MORE than what one
                                   morphological operation already does.
  a large gap that closing misses  C1 is worth building, and this number is
                                   what it is chasing.

The oracle uses ground truth by construction and is not a method -- it is an
upper bound, and every table here labels it as one.

  python exp/link_ceiling.py --selftest
  python exp/link_ceiling.py                 # every arm, rule (iv) weights
  python exp/link_ceiling.py K_focal_aug

Writes results/selection_sweep/link_ceiling.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import bridge_sweep
import cross_dataset
import drive
import erl
import hole_sweep
import select_checkpoint as rules_module
import speckle
import summarize_selection as selection
import train

OUT = selection.SWEEP / "link_ceiling.csv"
# The weights every condition is measured on: task A3's best rule. Fixed here
# because plan_next.md section 0 says a new method must beat the best
# selection rule, not the protocol it happened to be developed against.
RULE = "(iv) best clDice"


def fill(pred: np.ndarray, skel_gt: np.ndarray, band: np.ndarray,
         kinds: tuple) -> np.ndarray:
    """Add back the missed centreline runs of the given kinds.

    The run's own pixels are enough to reconnect: a run is a MAXIMAL set of
    consecutive missed skeleton pixels, so the skeleton pixel at each end of
    it is covered and therefore already in `pred`, and 8-connectivity carries
    through. Nothing is dilated, so the oracle adds the least foreground that
    does the job and cannot flatter itself by thickening the prediction.
    """
    missed = skel_gt & ~pred
    labels, count = ndimage.label(missed, structure=break_lengths.CONN8)
    if count == 0:
        return pred
    pieces, _ = ndimage.label(pred, structure=break_lengths.CONN8)
    out = pred.copy()
    boxes = ndimage.find_objects(labels)
    for index, box in enumerate(boxes, start=1):
        grown = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
        pixels = labels[grown] == index
        kind = break_lengths.classify(pixels, pred[grown], pieces[grown])
        if kind in kinds:
            out[grown] |= pixels
    return out


def best_radius(per_radius: dict) -> int:
    """The closing radius with the best ERL on the SELECTION half.

    Chosen on the half no result is reported from, for the same reason the
    checkpoint is: picking the radius on the reported images would make the
    baseline the strongest of seven attempts and the comparison meaningless.
    """
    return max(per_radius, key=lambda radius: np.mean(
        [value for image, value in per_radius[radius].items()
         if rules_module.is_selection_image(image)]))


def selftest() -> None:
    # A straight vessel with two gaps: one that severs it, and one beside an
    # intact prediction that does not. Filling only the severing gap must
    # recover the run length; filling both must not do more.
    skel = np.zeros((30, 100), dtype=bool)
    skel[15, 5:95] = True
    pred = np.zeros_like(skel)
    pred[14:17, 5:95] = True
    pred[:, 40:44] = False               # a real cut
    band = np.zeros_like(skel, dtype=int)

    kinds = [k for _, _, _, k in break_lengths.break_runs(pred, skel, band)]
    assert kinds == ["severs"], kinds
    raw = erl.expected_run_length(skel, pred)
    healed = erl.expected_run_length(skel, fill(pred, skel, band, ("severs",)))
    whole = erl.expected_run_length(skel, skel)
    print(f"one severing cut: ERL {raw:.0f} -> {healed:.0f} "
          f"(uncut vessel {whole:.0f})")
    assert healed > raw and abs(healed - whole) < 1e-6, (raw, healed, whole)

    # A break the prediction is connected around: filling severing breaks
    # must leave it alone, because it costs no run length to begin with.
    beside = np.zeros_like(skel)
    beside[16:19, 5:95] = True           # covers the vessel, one row over
    found = break_lengths.break_runs(beside, skel, band)
    assert [k for _, _, _, k in found] == ["intact"], found
    only_severs = fill(beside, skel, band, ("severs",))
    assert np.array_equal(only_severs, beside), "an intact break was filled"
    print("  a break the prediction is connected around is left alone by "
          "oracle_sever, as it must be -- it costs no run length")
    everything = fill(beside, skel, band, ("severs", "intact", "absent"))
    assert everything.sum() > beside.sum()
    print("  oracle_all does fill it, which is why the two bounds differ")

    # best_radius must read the selection half only.
    per_radius = {1: {}, 2: {}}
    for index in range(1, 21):
        name = f"{index:02d}"
        odd = rules_module.is_selection_image(name)
        per_radius[1][name] = 100.0 if odd else 0.0
        per_radius[2][name] = 50.0 if odd else 9999.0
    assert best_radius(per_radius) == 1, "the radius was picked on the report"
    print("the closing radius is picked on the selection half; a radius that "
          "only wins on the reported images is not chosen")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] \
        or list(selection.ARMS)

    points = selection.selection_points(selection.load())
    rule = dict(rules_module.rules())[RULE]

    items = drive.load_split("val")
    data = train.stack_split("train")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(item["label"] & item["fov"]),
                 "band": np.zeros(item["label"].shape, dtype=int)}
                for item in items]
    radii = sorted({int(round(f * width))
                    for f in bridge_sweep.WIDTH_FRACTIONS if f > 0})
    print(f"{len(wanted)} arm(s), {len(items)} images, weights = {RULE}, "
          f"component filter {component_px} px, closing radii {radii} px",
          flush=True)

    rows = []
    for config in wanted:
        for run in sorted(r for r in points if r.rsplit("_s", 1)[0] == config):
            epoch = rule(points[run])["epoch"]
            state = train.load_checkpoint(
                selection.SWEEP / run / f"epoch{epoch:03d}.pt")
            model = train.build_model(config)
            model.load_state_dict(state["model"])
            model.eval()
            mean, std = train.normalisation(run, data)

            preds, closings = [], {radius: {} for radius in radii}
            for item, geo in zip(items, geometry):
                prob = train.predict_full(model, item["image"], mean, std)
                pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                          component_px)
                preds.append(pred)
                for radius in radii:
                    closed = bridge_sweep.bridge(pred, radius) & item["fov"]
                    closings[radius][item["name"]] = \
                        erl.expected_run_length(geo["skel"], closed)
            radius = best_radius(closings)

            for index, (item, geo) in enumerate(zip(items, geometry)):
                pred = preds[index]
                conditions = {
                    "raw": pred,
                    "closing": bridge_sweep.bridge(pred, radius) & item["fov"],
                    "oracle_sever": fill(pred, geo["skel"], geo["band"],
                                         ("severs",)),
                    "oracle_all": fill(pred, geo["skel"], geo["band"],
                                       ("severs", "intact", "absent")),
                }
                for name, mask in conditions.items():
                    rows.append({
                        "config": config, "run": run, "epoch": epoch,
                        "seed": run.rsplit("_s", 1)[1], "condition": name,
                        "radius_px": radius, "image": item["name"],
                        "erl": round(erl.expected_run_length(geo["skel"],
                                                             mask), 3),
                        "skel_px": int(geo["skel"].sum()),
                        "foreground": int(mask.sum())})
            print(f"  {run} epoch {epoch} done (closing radius {radius} px)",
                  flush=True)

    if not rows:
        raise SystemExit("nothing scored")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
