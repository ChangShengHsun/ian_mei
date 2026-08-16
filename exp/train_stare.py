"""Train on one annotator, score against the other: how much is style?

STARE labels the same 20 images twice. If a model trained on ah scores far
worse against vk than against ah, then part of what it learned is that
annotator's personal threshold for "is this faint thing a vessel", not vessel
anatomy. The human-human Dice of 0.740 is the reference: a model trained on ah
that reaches 0.740 against vk is exactly as close to vk as ah is -- a faithful
ah-clone, and no further.

Two-fold (STARE has no official split, so this declares its own) x three targets
(ah, vk, and the soft consensus) x two seeds = 12 runs of the plain BCE + Dice
baseline. The loss is held fixed because the variable under test is the
annotation, not the objective.

  python exp/train_stare.py            # all 12, resumable
  python exp/train_stare.py ah_f0_s0   # one run

~45 min per run, ~9 h total on 6 cores.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
import speckle
import stare_agreement
import train

RESULTS = Path(__file__).resolve().parent / "results" / "stare_cross"
EPOCHS, VAL_EVERY = 100, 20
MIN_SIZE = 20                      # the post-processing settled on in speckle.py
FOLDS = {0: (slice(0, 10), slice(10, 20)), 1: (slice(10, 20), slice(0, 10))}
TARGETS = ("ah", "vk")            # the two annotators; also the scoring targets
# "soft" is the consensus arm added for E3': 1.0 where both annotators marked a
# vessel, 0.5 where exactly one did. E0 confirmed nobody has asked what happens
# to TOPOLOGY when you train on a soft consensus instead of one person's mask.
TRAIN_TARGETS = ("ah", "vk", "soft")
SEEDS = (0, 1)


def build(items: list[dict], target: str) -> dict:
    if target == "soft":
        labels = np.stack([(item["ah"].astype(np.float32)
                            + item["vk"].astype(np.float32)) / 2
                           for item in items])
    else:
        labels = np.stack([item[target] for item in items]).astype(np.float32)
    return {
        "images": np.stack([item["image"] for item in items]),
        "labels": labels,
        "fovs": np.stack([item["fov"] for item in items]),
        "dists": np.stack([
            np.clip(ndimage.distance_transform_edt(~(mask > 0.5))
                    - ndimage.distance_transform_edt(mask > 0.5),
                    -train.DIST_CLIP, train.DIST_CLIP) for mask in labels
        ]).astype(np.float32) / train.DIST_CLIP,
        "names": [item["name"] for item in items],
    }


def score(model, held_out: list[dict], mean: float, std: float) -> list[dict]:
    """Score the held-out fold against BOTH annotators, after speckle removal."""
    rows = []
    for item in held_out:
        prob = train.predict_full(model, item["image"], mean, std)
        pred = speckle.drop_small((prob >= 0.5) & item["fov"], MIN_SIZE)
        for annotator in TARGETS:
            rows.append({
                "image": item["name"], "scored_against": annotator,
                **metrics.evaluate(pred.astype(float), item[annotator],
                                   item["fov"]),
            })
    return rows


def train_one(run_name: str, items: list[dict]) -> None:
    target, fold_tag, seed_tag = run_name.split("_")
    fold, seed = int(fold_tag[1:]), int(seed_tag[1:])
    train_slice, test_slice = FOLDS[fold]
    out_dir = RESULTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keyed on the weights, not on scores.csv: the first eight runs wrote scores
    # but never saved a model, so every later re-measurement had to retrain.
    if (out_dir / "final.pt").exists():
        print(f"[{run_name}] already finished, skipping", flush=True)
        return

    train_items, held_out = items[train_slice], items[test_slice]
    data = build(train_items, target)
    inside = data["images"][data["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = train.TinyUNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = train.PATCHES_PER_EPOCH // train.BATCH
    started = time.time()

    for epoch in range(EPOCHS):
        running = 0.0
        for _ in range(steps):
            images, labels, dists = train.sample_batch(data, rng, mean, std)
            optimiser.zero_grad()
            loss = train.compute_loss(model(images), labels, dists, None)
            loss.backward()
            optimiser.step()
            running += loss.item()
        if (epoch + 1) % VAL_EVERY == 0:
            print(f"[{run_name}] epoch {epoch + 1:3d} loss {running / steps:.4f} "
                  f"({(time.time() - started) / 60:.0f} min)", flush=True)

    torch.save({"model": model.state_dict(), "mean": mean, "std": std},
               out_dir / "final.pt")

    rows = score(model, held_out, mean, std)
    # The original eight runs are already cited in stage 0; a rerun writes
    # beside them so the published numbers stay auditable against the repeat.
    out_name = ("scores.csv" if not (out_dir / "scores.csv").exists()
                else "scores_rerun.csv")
    with (out_dir / out_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    own = np.mean([r["dice"] for r in rows if r["scored_against"] == target])
    other = np.mean([r["dice"] for r in rows if r["scored_against"] != target])
    print(f"[{run_name}] done in {(time.time() - started) / 60:.0f} min · "
          f"dice vs {target} {own:.4f} · vs other {other:.4f}", flush=True)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    items = stare_agreement.load_stare()
    print(f"STARE: {len(items)} images, folds 0-9 / 10-19", flush=True)
    names = [f"{t}_f{f}_s{s}"
             for s in SEEDS for f in FOLDS for t in TRAIN_TARGETS]
    for run_name in (sys.argv[1:] or names):
        train_one(run_name, items)


if __name__ == "__main__":
    main()
