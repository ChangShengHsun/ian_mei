"""E4: is "filtering beats the topology loss" a DRIVE fact or a general one?

stage 0 found that dropping predicted components below 20 px closes most of the
gap between BCE+Dice and the clDice loss on DRIVE. Two objections stand:

  1. one dataset, and 20 px is an absolute number that means nothing on a
     different sampling grid;
  2. the comparison used the 2021 loss, not the current one. stage_1 section
     0.2 requires cbDice (MICCAI 2024) as a control.

Both are answered here. The threshold is expressed in units of the median
structure width squared, so it transfers: DRIVE's 20 px is 2.1 * w^2, and the
same 2.1 becomes ~24 px on HRF and ~90 px on TopoMortar.

Measured widths (skeleton distance transform x 2). Per-image medians averaged
over 8 images, which is what fixes the scale:

    DRIVE          3.10 px    p90 6.16    8.9% foreground
    HRF full res   6.75 px    p90 17.11   7.2%
    HRF halved     3.4  px    p90 8.6     7.2%
    TopoMortar     6.54 px    p90 8.24   26.7%

median_width() below pools all skeleton radii before taking the median, which
lands on the quantisation grid of the distance transform and reads 4.00 px for
halved HRF and 8.00 px for TopoMortar. Both estimates are coarse and the sweep
covers a 50x range around them, so nothing here rests on the second decimal.

HRF is halved rather than used at full resolution so that a 48 px training
patch spans the same number of structure widths as it does on DRIVE. Otherwise
the network sees a different amount of context and "the same architecture" is
not the same experiment. This costs the high-resolution angle, which E4 is not
asking about.

TopoMortar earns its place by being unlike both: mortar lines form a grid, so
the interesting Betti number is b1 (loops) rather than b0, its structures are
nearly uniform in width (p90/median 1.26 against DRIVE's 1.99), and it ships
accurate, pseudo and noisy labels for the same images, which E5 needs.

  python exp/cross_dataset.py hrf              # all runs for one dataset
  python exp/cross_dataset.py hrf A_dice_s0    # one run

Writes results/cross/<dataset>/<run>/{final.pt,scores.csv}.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
import speckle
import train

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent / "results" / "cross"
LOSSES = ("A_dice", "B_cldice", "E_cbdice")
SEEDS = (0, 1)
# 60 rather than the 100 used on DRIVE and STARE. Measured step cost on an
# uncontended machine is 0.12 s for BCE+Dice, 0.18 for clDice and 0.19 for
# cbDice, so 100 epochs x 24 runs is about 34 hours of CPU on a laptop that is
# also Ivan's coursework machine. Every arm gets the identical budget, so the
# within-dataset comparison E4 asks for is unaffected; what it costs is the
# right to compare these absolute scores against the 100-epoch DRIVE numbers.
EPOCHS, VAL_EVERY = 60, 20
# The DRIVE sweep (0, 2, 5, 10, 20, 50, 100) px divided by DRIVE's w^2 = 9.6.
# Every dataset is swept over the same multiples of its own w^2.
WIDTH_MULTIPLES = (0.0, 0.21, 0.52, 1.04, 2.08, 5.21, 10.42)


def prepare(gray: np.ndarray, label: np.ndarray, fov: np.ndarray,
            name: str) -> dict:
    """One item in the same shape every other loader in this repo produces."""
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return {"name": name, "image": enhanced.astype(np.float32) / 255.0,
            "label": label & fov, "fov": fov}


def load_hrf() -> tuple[list[dict], list[dict]]:
    """45 fundus images, halved. Ten of each pathology train, five test."""
    items = []
    for label_path in sorted((DATA / "HRF" / "manual1").glob("*.tif")):
        stem = label_path.stem
        image_path = next((DATA / "HRF" / "images").glob(f"{stem}.[jJ][pP][gG]"))
        rgb = np.asarray(Image.open(image_path))[::2, ::2]
        label = np.asarray(Image.open(label_path))[::2, ::2] > 127
        mask = np.asarray(Image.open(DATA / "HRF" / "mask" / f"{stem}_mask.tif"))
        fov = (mask[::2, ::2, 0] if mask.ndim == 3 else mask[::2, ::2]) > 127
        items.append(prepare(rgb[..., 1], label, fov, stem))
    # Split on the index inside each pathology, so train and test both contain
    # healthy, glaucomatous and diabetic-retinopathy eyes.
    train_items = [i for i in items if int(i["name"].split("_")[0]) <= 10]
    test_items = [i for i in items if int(i["name"].split("_")[0]) > 10]
    return train_items, test_items


def load_topomortar(label_kind: str = "accurate") -> tuple[list[dict], list[dict]]:
    """The authors' own 50/350 split, read from their splits.yaml."""
    root = DATA / "TopoMortar" / "dataset"
    splits = yaml.safe_load((root / "splits.yaml").open())[1]

    def build(paths: list[str], kind: str) -> list[dict]:
        out = []
        for relative in paths:
            stem = Path(relative).name
            split = Path(relative).parts[1]
            rgb = np.asarray(Image.open(root.parent / relative))
            label = np.asarray(
                Image.open(root / split / kind / stem)) > 127
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            out.append(prepare(gray, label, np.ones(label.shape, bool),
                               f"{split}_{stem}"))
        return out

    # Test labels only exist as "accurate"; the label_kind switch is for E5,
    # which varies the TRAINING labels and always scores against the truth.
    return build(splits["train"], label_kind), build(splits["test"], "accurate")


def loader_for(dataset: str):
    """`topomortar:noisy` trains on the noisy labels and still scores against
    the accurate ones. That is E5: how fast does each loss degrade as the
    training labels get worse, on labels somebody else designed rather than
    noise we invented for ourselves."""
    if dataset.startswith("topomortar"):
        kind = dataset.partition(":")[2] or "accurate"
        return lambda: load_topomortar(kind)
    return load_hrf


def median_width(items: list[dict]) -> float:
    radii = []
    for item in items[:8]:
        mask = item["label"]
        if mask.any():
            radii.append(ndimage.distance_transform_edt(mask)[skeletonize(mask)])
    return float(np.median(np.concatenate(radii)) * 2)


def stack(items: list[dict]) -> dict:
    labels = np.stack([i["label"] for i in items]).astype(np.float32)
    return {
        "images": np.stack([i["image"] for i in items]),
        "labels": labels,
        "fovs": np.stack([i["fov"] for i in items]),
        # Only the boundary loss reads this and E4 does not use it; kept zero
        # rather than computing 45 distance transforms nobody looks at.
        "dists": np.zeros_like(labels),
        "names": [i["name"] for i in items],
    }


def score(model, test_items: list[dict], mean: float, std: float,
          width: float) -> list[dict]:
    rows = []
    for item in test_items:
        prob = train.predict_full(model, item["image"], mean, std)
        raw = (prob >= 0.5) & item["fov"]
        truth = item["label"]
        b0_gt, b1_gt = metrics.betti(truth)
        for multiple in WIDTH_MULTIPLES:
            min_size = int(round(multiple * width * width))
            pred = speckle.drop_small(raw, min_size)
            b0, b1 = metrics.betti(pred)
            rows.append({
                "image": item["name"], "width_multiple": multiple,
                "min_size": min_size,
                "dice": round(metrics.dice(pred, truth), 5),
                "cldice": round(metrics.cl_dice(pred, truth), 5),
                "betti0_err": abs(b0 - b0_gt), "betti1_err": abs(b1 - b1_gt),
            })
    return rows


def out_root(dataset: str) -> Path:
    """Windows forbids ':' in a path, so `topomortar:noisy` needs sanitising.

    Learned the hard way: the smoke test wrote `dataset.replace(":", "_")` at
    the call site while main() passed the raw string, so the test diverged from
    the real path at exactly the point that breaks. Do the substitution here,
    where both go through it.
    """
    return RESULTS / dataset.replace(":", "_")


def write_scores(out_dir: Path, rows: list[dict]) -> None:
    with (out_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(dataset: str, run_name: str, data: dict, test_items: list[dict],
              width: float) -> None:
    config_name, seed_tag = run_name.rsplit("_s", 1)
    out_dir = out_root(dataset) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "scores.csv").exists():
        print(f"[{dataset}/{run_name}] already finished, skipping", flush=True)
        return

    _, extra = train.CONFIGS[config_name]
    # Scoring 350 images is the long tail after training and has twice been
    # where an interrupted run died, throwing away weights that were already
    # saved. If they are on disk, skip straight to scoring.
    if (out_dir / "final.pt").exists():
        state = torch.load(out_dir / "final.pt", weights_only=False)
        model = train.TinyUNet()
        model.load_state_dict(state["model"])
        model.eval()
        print(f"[{dataset}/{run_name}] weights found, scoring only", flush=True)
        write_scores(out_dir, score(model, test_items, state["mean"],
                                   state["std"], width))
        return
    inside = data["images"][data["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())

    torch.manual_seed(int(seed_tag))
    rng = np.random.default_rng(int(seed_tag))
    model = train.TinyUNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = train.PATCHES_PER_EPOCH // train.BATCH
    started = time.time()

    for epoch in range(EPOCHS):
        running = 0.0
        for _ in range(steps):
            images, labels, dists = train.sample_batch(data, rng, mean, std)
            optimiser.zero_grad()
            loss = train.compute_loss(model(images), labels, dists, extra)
            loss.backward()
            optimiser.step()
            running += loss.item()
        if (epoch + 1) % VAL_EVERY == 0:
            print(f"[{dataset}/{run_name}] epoch {epoch + 1:3d} "
                  f"loss {running / steps:.4f} "
                  f"({(time.time() - started) / 60:.0f} min)", flush=True)

    torch.save({"model": model.state_dict(), "mean": mean, "std": std},
               out_dir / "final.pt")
    rows = score(model, test_items, mean, std, width)
    write_scores(out_dir, rows)
    baseline = np.mean([r["dice"] for r in rows if r["width_multiple"] == 0])
    print(f"[{dataset}/{run_name}] done in {(time.time() - started) / 60:.0f} "
          f"min · unfiltered dice {baseline:.4f}", flush=True)


def main() -> None:
    dataset = sys.argv[1]
    train_items, test_items = loader_for(dataset)()
    # From the TEST labels, never the training ones. E5 varies the training
    # labels, and the noisy set is thinner (width 4.0 against 8.0), so a
    # threshold derived from them would change between the very arms E5
    # compares -- the filter would become a confound instead of a control.
    width = median_width(test_items)
    print(f"{dataset}: {len(train_items)} train / {len(test_items)} test, "
          f"median width {width:.2f} px, filter sweep "
          f"{[int(round(m * width * width)) for m in WIDTH_MULTIPLES]} px",
          flush=True)

    data = stack(train_items)
    names = sys.argv[2:] or [f"{loss}_s{seed}"
                             for seed in SEEDS for loss in LOSSES]
    for run_name in names:
        train_one(dataset, run_name, data, test_items, width)


if __name__ == "__main__":
    main()
