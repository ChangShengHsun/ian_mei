"""Zero-shot transfer: train on DRIVE, deploy on STARE and HRF unchanged.

E4 asked whether a FINDING transfers by retraining on each dataset. This asks
the different and more practical question: does the MODEL transfer? Nothing is
retrained, nothing is fine-tuned, and the normalisation constants are the ones
each checkpoint was trained with -- a model that peeked at the target dataset's
statistics would not be zero-shot.

Why now: E16's whole premise is that LIOT deletes contrast dependence, and
contrast dependence is exactly what a change of scanner produces. A same-
dataset test cannot see that. LIOT's own paper claims cross-dataset
generalisation, so this is the setting its claim lives in.

It is also worth running for the arms that have nothing to do with LIOT.
Twenty experiments in, no result in this series has ever been checked for
whether it survives a change of camera.

Inference only: no training, roughly a minute per run per dataset.

  python exp/transfer.py                    # every finished DRIVE run
  python exp/transfer.py stare A_dice_s0    # one dataset, one run
  python exp/transfer.py --selftest

Writes results/transfer.csv, one row per (run, dataset, image).
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
import erl
import hole_sweep
import metrics
import speckle
import train

torch.set_num_threads(2)  # a training job may hold the other four

RESULTS = Path(__file__).resolve().parent / "results"


def load_targets(name: str) -> list[dict]:
    """Test images from a dataset the models have never seen.

    STARE has two annotators; `ah` is used because E3' established that the
    two disagree enough to change a topology metric, and picking the one the
    rest of the repo reports keeps this comparable.
    """
    if name == "stare":
        import stare_agreement
        return [{"name": item["name"], "image": item["image"],
                 "fov": item["fov"], "label": item["ah"]}
                for item in stare_agreement.load_stare()]
    if name == "hrf":
        return cross_dataset.load_hrf()[1]
    raise SystemExit(f"unknown dataset {name!r}")


def constants(run_name: str, drive_train: dict) -> tuple:
    """The normalisation the checkpoint was TRAINED with, not the target's.

    This is the whole point of the word zero-shot. Recomputing mean and std on
    STARE would hand the model a summary of the test set, and would also hide
    the failure mode being measured: a grey model transfers badly precisely
    because the target's intensities sit somewhere its normalisation does not
    expect.
    """
    if train.uses_liot(run_name.rsplit("_s", 1)[0]):
        return train.liot_stats(drive_train)
    inside = drive_train["images"][drive_train["fovs"]]
    return float(inside.mean()), float(inside.std())


def selftest() -> None:
    """The one thing that can silently invalidate every number here."""
    drive_train = train.stack_split("train")
    inside = drive_train["images"][drive_train["fovs"]]
    grey = (float(inside.mean()), float(inside.std()))

    got = constants("A_dice_s0", drive_train)
    assert got == grey, got
    print(f"grey runs keep DRIVE's constants: mean {got[0]:.4f} "
          f"std {got[1]:.4f}")

    mean, std = constants("J_liot_s0", drive_train)
    assert mean.shape == (4, 1, 1) and (std > 0).all(), mean.shape
    print(f"LIOT runs get 4-channel constants: "
          f"mean {np.round(mean.ravel(), 1)}")

    # And they are DRIVE's, not the target's. Compute STARE's and require they
    # differ -- if the two datasets happened to have identical statistics the
    # experiment would have nothing to measure.
    stare = load_targets("stare")
    stare_inside = np.concatenate(
        [item["image"][item["fov"]] for item in stare[:4]])
    assert abs(float(stare_inside.mean()) - grey[0]) > 0.01, \
        "STARE and DRIVE have the same mean -- nothing to transfer"
    print(f"STARE mean {stare_inside.mean():.4f} against DRIVE's "
          f"{grey[0]:.4f}: the shift the models must survive")
    print("all checks passed")


def score(model, item, mean, std, component_px: int) -> dict:
    prob = train.predict_full(model, item["image"], mean, std)
    truth = item["label"] & item["fov"]
    pred = speckle.drop_small((prob >= 0.5) & item["fov"], component_px)
    skeleton = skeletonize(truth)
    row = metrics.evaluate(prob, truth, item["fov"])
    row["erl"] = round(erl.expected_run_length(skeleton, pred), 2)
    row["erl_ceiling"] = round(erl.expected_run_length(skeleton, truth), 2)
    # Severing breaks, E10's metric: a missed centreline run whose two sides
    # land in different predicted components. Reported unstratified because
    # the contrast quartiles are defined per dataset and would not be
    # comparable across the two targets.
    missed = skeleton & ~pred
    labels, count = ndimage.label(missed, structure=break_lengths.CONN8)
    pieces = ndimage.label(pred, structure=break_lengths.CONN8)[0]
    severs = 0
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        grown = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in box)
        if break_lengths.classify(labels[grown] == index, pred[grown],
                                  pieces[grown]) == "severs":
            severs += 1
    row["breaks"] = count
    row["severs"] = severs
    return row


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    datasets = [args[0]] if args and args[0] in ("stare", "hrf") \
        else ["stare", "hrf"]
    runs = args[1:] if args and args[0] in ("stare", "hrf") else args
    runs = runs or train.trained_runs()

    drive_train = train.stack_split("train")
    rows = []
    for dataset in datasets:
        items = load_targets(dataset)
        width = cross_dataset.median_width(items)
        component_px = int(round(
            hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
        print(f"{dataset}: {len(items)} images, median width {width:.2f} px, "
              f"component filter {component_px} px", flush=True)
        for run_name in runs:
            weights = RESULTS / run_name / "final.pt"
            if not weights.exists():
                continue
            model = train.build_model(run_name.rsplit("_s", 1)[0])
            model.load_state_dict(
                torch.load(weights, weights_only=False)["model"])
            model.eval()
            mean, std = constants(run_name, drive_train)
            for item in items:
                rows.append({"run": run_name,
                             "config": run_name.rsplit("_s", 1)[0],
                             "dataset": dataset, "image": item["name"],
                             **score(model, item, mean, std, component_px)})
            print(f"  [{run_name}] done", flush=True)

    if not rows:
        raise SystemExit("no checkpoints found")
    out = RESULTS / "transfer.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")

    configs = sorted({r["config"] for r in rows})
    for dataset in datasets:
        print(f"\n=== {dataset}, zero-shot from DRIVE ===")
        print(f"{'config':14}{'dice':>8}{'clDice':>8}{'b0err':>9}"
              f"{'severs':>8}{'ERL':>9}{'ERL/ceil':>10}")
        for config in configs:
            picked = [r for r in rows if r["config"] == config
                      and r["dataset"] == dataset]
            if not picked:
                continue
            def avg(key):
                return float(np.nanmean([r[key] for r in picked]))
            print(f"{config:14}{avg('dice'):8.4f}{avg('cldice'):8.4f}"
                  f"{avg('betti0_err'):9.1f}{avg('severs'):8.1f}"
                  f"{avg('erl'):9.1f}"
                  f"{avg('erl') / avg('erl_ceiling'):10.3f}")


if __name__ == "__main__":
    main()
