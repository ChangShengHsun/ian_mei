"""Phase 3: does the geometry transfer, or is it a pixel count in disguise?

WRITTEN AND SELFTESTED 2026-08-27, BEFORE IT SCORED ANYTHING.

CLAUDE.md's rule is that a threshold which must transfer between datasets is
expressed in multiples of the median structure width, never in pixels. For D1
that rule is load-bearing rather than tidy: the propagation layer's along and
across radii ARE the architecture, and the three datasets here span

    DRIVE   2.83 px      STARE   ~3 px       HRF   4.00 px
    VessMAP 5.66 px, 28% of the frame vessel, and not a retina at all

If the setting phase 1 chose on DRIVE is also the best setting on VessMAP
when both are read in vessel widths, the units are doing their job. If each
dataset wants a different multiple, then "1.0 widths" is a DRIVE pixel count
wearing a unit, and the layer will silently become a different operator the
first time it is transferred.

REVISED 2026-08-27, after the first run answered nothing. It used
DRIVE-trained models on every dataset, on the argument that the geometry
question does not need a good segmentation. It does need a segmentation:
raw traced run length came out at 3.4% on HRF and 0.0% on VessMAP, and a
correction cannot be measured on a prediction that has nothing to correct.
What that run measured was the transfer of the SEGMENTATION, which is E4's
question and was already answered.

Each dataset now uses models trained on itself, by train.py -- the same
trainer, the same augmentation tuples, the same 100 epochs -- so the arms are
comparable across datasets in everything but the data.

THE CHECKPOINT is best.pt, the highest whole-val Dice, on every dataset
including DRIVE. That is optimistic: these datasets have no third split, so
it is chosen on the images it is reported on. It is stated rather than hidden,
and it cannot bias the question being asked -- the same checkpoint serves
every setting and every source within a dataset, so it cannot favour one
geometry over another. The dilation setting itself is still chosen on half
the images and reported on the other.

  python exp/transfer_ceiling.py --selftest
  python exp/transfer_ceiling.py

Writes results/selection_sweep/transfer_ceiling.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anisotropic
import cross_dataset
import direction
import direction_ceiling
import drive
import erl
import hole_sweep
import link_ceiling
import speckle
import summarize_selection as selection
import train

OUT = selection.SWEEP / "transfer_ceiling.csv"
# Three seeds, not six: this is a question about geometry, and the geometry
# does not have a seed. Six would double an hour of CPU to sharpen an error
# bar on a quantity nobody is claiming a t-test about.
SEEDS = (0, 1, 2)
# Two arms, not one: a geometry that only holds for the arm it was found on
# is an arm-specific accident, which is the same standard gate_task_b applied.
CONFIGS = ("K_focal_aug", "A_dice")
TRANSFER = Path(__file__).resolve().parent / "results" / "transfer"


def datasets() -> dict:
    """{name: (test items, median width)} -- loaded once, they are not small."""
    import stare_agreement
    out = {}
    out["DRIVE"] = drive.load_split("val")
    out["HRF"] = cross_dataset.load_hrf()[1]
    out["VessMAP"] = cross_dataset.load_vessmap()[1]
    # STARE's loader carries two annotators; `ah` is the one E4 scored
    # against, so it is the one used here.
    stare = []
    for item in stare_agreement.load_stare():
        stare.append({"name": item["name"], "image": item["image"],
                      "fov": item["fov"], "label": item["ah"]})
    out["STARE"] = stare
    return out


def selftest() -> None:
    # The grid must be the SAME one phase 1 swept, or "the same setting wins
    # everywhere" is a claim about two different grids.
    assert direction_ceiling.ALONG and direction_ceiling.ACROSS
    print(f"reuses phase 1's grid exactly: along {direction_ceiling.ALONG}, "
          f"across {direction_ceiling.ACROSS}")

    # The whole point, in miniature: the same width-relative setting must
    # behave the same way on a vessel of a different width. A 2 px vessel and
    # a 6 px one, each with a gap of one width, must both be closed by the
    # same multiple.
    for width in (2.0, 6.0):
        size = 121
        yy, xx = np.mgrid[0:size, 0:size]
        vessel = np.abs(yy - 60) <= width / 2
        vessel &= (xx > 20) & (xx < 100)
        broken = vessel.copy()
        broken[:, 58:58 + int(round(width))] = False   # a gap of one width
        sin2, cos2, _ = direction.tangent_field(vessel)
        skel = skeletonize(vessel)
        before = erl.expected_run_length(skel, broken) / skel.sum()
        healed = anisotropic.oriented_dilation(
            broken, sin2, cos2, 1.0 * width, 0.25 * width)
        after = erl.expected_run_length(skel, healed) / skel.sum()
        print(f"  width {width:.0f} px, gap of one width: along=1.0 widths "
              f"takes it {before:.0%} -> {after:.0%}")
        assert after > before + 0.3, (width, before, after)
    print("  the same multiple closes the same relative gap at both scales -- "
          "which is what the unit is for")

    # And a PIXEL count does not: a radius tuned on the 2 px vessel must fail
    # on the 6 px one, or the unit would not matter and neither would this.
    size = 121
    yy, xx = np.mgrid[0:size, 0:size]
    wide = np.abs(yy - 60) <= 3.0
    wide &= (xx > 20) & (xx < 100)
    broken = wide.copy()
    broken[:, 58:64] = False
    sin2, cos2, _ = direction.tangent_field(wide)
    skel = skeletonize(wide)
    tuned_on_narrow = anisotropic.oriented_dilation(broken, sin2, cos2,
                                                    1.0 * 2.0, 0.25 * 2.0)
    got = erl.expected_run_length(skel, tuned_on_narrow) / skel.sum()
    print(f"  the same reach in PIXELS (tuned on the 2 px vessel) leaves the "
          f"6 px one at {got:.0%}")
    assert got < 0.5, got
    print("all checks passed")


def model_root(dataset: str) -> Path:
    """Where this dataset's own models live."""
    return (selection.SWEEP if dataset == "DRIVE"
            else TRANSFER / dataset.lower())


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]

    rows = []
    for name, items in datasets().items():
        if wanted and name not in wanted:
            continue
        width = cross_dataset.median_width(items)
        component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE
                                 * width * width))
        geometry = [{"skel": skeletonize(i["label"] & i["fov"]),
                     "truth": i["label"] & i["fov"]} for i in items]
        fields = [direction.tangent_field(g["truth"]) for g in geometry]
        root = model_root(name)
        stacked = train.stack_split(
            "train", None if name == "DRIVE"
            else cross_dataset.loader_for(name.lower())()[0])
        inside = stacked["images"][stacked["fovs"]]
        mean, std = float(inside.mean()), float(inside.std())
        print(f"{name}: {len(items)} images, width {width:.2f} px, filter "
              f"{component_px} px, models from {root}", flush=True)

        for config in CONFIGS:
            for seed in SEEDS:
                run = f"{config}_s{seed}"
                weights = root / run / "best.pt"
                if not weights.exists():
                    print(f"  WARNING {name}/{run}: no best.pt, skipping",
                          flush=True)
                    continue
                model = train.build_model(config)
                model.load_state_dict(
                    train.load_checkpoint(weights)["model"])
                model.eval()
                for item, geo, field in zip(items, geometry, fields):
                    prob = train.predict_full(model, item["image"], mean, std)
                    pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                              component_px)
                    common = {"dataset": name, "width": round(width, 3),
                              "config": config, "run": run, "seed": str(seed),
                              "image": item["name"]}

                    def record(source, along, across, mask):
                        rows.append({
                            **common, "source": source, "along": along,
                            "across": across,
                            "erl": round(erl.expected_run_length(
                                geo["skel"], mask) / geo["skel"].sum(), 5),
                            "dice": round(link_ceiling.dice(mask,
                                                            geo["truth"]), 5),
                            "fg": int(mask.sum())})

                    record("raw", 0.0, 0.0, pred)
                    for radius in direction_ceiling.ISOTROPIC:
                        if radius == 0.0:
                            continue
                        record("isotropic", radius, radius,
                               anisotropic.isotropic_dilation(
                                   pred, radius * width) & item["fov"])
                    for along in direction_ceiling.ALONG:
                        for across in direction_ceiling.ACROSS:
                            if along == 0.0 and across == 0.0:
                                continue
                            record("oracle", along, across,
                                   anisotropic.oriented_dilation(
                                       pred, field[0], field[1], along * width,
                                       across * width) & item["fov"])
                print(f"  {name}/{run} done", flush=True)

    if not rows:
        raise SystemExit("nothing scored -- are the transfer models trained?")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
