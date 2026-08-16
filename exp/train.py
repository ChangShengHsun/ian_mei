"""The 4-loss x 3-seed matrix from the plan, CPU-only, resumable.

Runs are ordered seed-major: all four losses at seed 0 finish first, so an
interrupted job still leaves a complete (if noisy) comparison rather than one
over-measured arm. A finished run is skipped, a half-finished one resumes from
its checkpoint, so re-running this file after a crash or a reboot is safe.

  python exp/train.py            # the whole matrix
  python exp/train.py A_dice_s0  # one run

~70 min per run on 6 cores, ~14 h for the matrix.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive
import metrics

RESULTS = Path(__file__).resolve().parent / "results"
PATCH, BATCH, PATCHES_PER_EPOCH, EPOCHS, VAL_EVERY = 48, 32, 10_000, 100, 10
DIST_CLIP = 20.0  # px; caps the boundary loss's reach into open background

torch.set_num_threads(6)


# --------------------------------------------------------------- model

def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class BlurPool(nn.Module):
    """Anti-aliased downsampling (Zhang, ICML 2019).

    Plain stride-2 pooling samples below the Nyquist rate of a 1-2 px vessel,
    so whether a vessel survives a downsample depends on its sub-pixel phase.
    Low-pass filtering before decimating removes that lottery. This is the
    cheapest of the detail-preservation tools in 2_methods.md section D.
    """

    def __init__(self, channels: int):
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel",
                             kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return F.conv2d(x, self.kernel, stride=2, padding=1,
                        groups=self.channels)


class TinyUNet(nn.Module):
    """3-level U-Net, 16 base channels, 117k params -- sized for CPU."""

    def __init__(self, base: int = 16, blurpool: bool = False):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.down1 = BlurPool(base) if blurpool else nn.MaxPool2d(2)
        self.down2 = BlurPool(base * 2) if blurpool else nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.down1(skip1))
        bottom = self.enc3(self.down2(skip2))
        x = self.dec2(torch.cat([self.up2(bottom), skip2], 1))
        x = self.dec1(torch.cat([self.up1(x), skip1], 1))
        return self.head(x)


# --------------------------------------------------------------- losses

def soft_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = 2 * (prob * target).sum((1, 2, 3)) + 1.0
    union = prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1.0
    return 1 - (intersection / union).mean()


def soft_skeleton(mask: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Differentiable skeleton by iterated morphological thinning (clDice).

    min-pool = erosion, max-pool = dilation; erode-then-dilate is an opening,
    and what an opening removes is exactly the thin ridge, i.e. the skeleton.
    """
    def erode(x):
        return -F.max_pool2d(-x, 3, 1, 1)

    def opened(x):
        return F.max_pool2d(erode(x), 3, 1, 1)

    skeleton = F.relu(mask - opened(mask))
    for _ in range(iterations):
        mask = erode(mask)
        delta = F.relu(mask - opened(mask))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cl_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    skel_pred, skel_true = soft_skeleton(prob), soft_skeleton(target)
    precision = ((skel_pred * target).sum((1, 2, 3)) + 1.0) / (
        skel_pred.sum((1, 2, 3)) + 1.0)
    sensitivity = ((skel_true * prob).sum((1, 2, 3)) + 1.0) / (
        skel_true.sum((1, 2, 3)) + 1.0)
    return 1 - (2 * precision * sensitivity / (precision + sensitivity)).mean()


CONFIGS = {
    # name -> (blurpool, extra loss term applied on top of BCE + soft Dice)
    "A_dice": (False, None),
    "B_cldice": (False, "cldice"),
    "C_boundary": (False, "boundary"),
    "D_blurpool": (True, None),
}


def compute_loss(logits, target, dist, extra):
    prob = torch.sigmoid(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    if extra == "cldice":
        # alpha = 0.5, the split used in the clDice paper (CVPR 2021).
        loss = loss + 0.5 * soft_dice(prob, target) + 0.5 * soft_cl_dice(prob, target)
    elif extra == "boundary":
        # ponytail: fixed weight 0.1. Kervadec ramps alpha 0.01 -> 0.99 over
        # training; add the ramp if the fixed weight under- or over-shoots.
        loss = loss + soft_dice(prob, target) + 0.1 * (dist * prob).mean()
    else:
        loss = loss + soft_dice(prob, target)
    return loss


# --------------------------------------------------------------- data

def stack_split(split: str) -> dict:
    items = drive.load_split(split)
    images = np.stack([item["image"] for item in items])
    labels = np.stack([item["label"] for item in items]).astype(np.float32)
    fovs = np.stack([item["fov"] for item in items])
    # Signed distance to the vessel boundary: negative inside, positive out.
    dists = np.stack([
        ndimage.distance_transform_edt(~item["label"])
        - ndimage.distance_transform_edt(item["label"]) for item in items])
    dists = np.clip(dists, -DIST_CLIP, DIST_CLIP).astype(np.float32) / DIST_CLIP
    return {"images": images, "labels": labels, "fovs": fovs, "dists": dists,
            "names": [item["name"] for item in items]}


def sample_batch(data: dict, rng: np.random.Generator, mean: float, std: float):
    height, width = data["images"].shape[1:]
    images, labels, dists = [], [], []
    while len(images) < BATCH:
        index = rng.integers(len(data["images"]))
        top = rng.integers(height - PATCH)
        left = rng.integers(width - PATCH)
        window = (slice(top, top + PATCH), slice(left, left + PATCH))
        # Skip patches whose centre lies outside the aperture: they are pure
        # black corner and teach the network nothing.
        if not data["fovs"][index][top + PATCH // 2, left + PATCH // 2]:
            continue
        images.append(data["images"][index][window])
        labels.append(data["labels"][index][window])
        dists.append(data["dists"][index][window])
    batch = (np.stack(images) - mean) / std
    return (torch.from_numpy(batch)[:, None],
            torch.from_numpy(np.stack(labels))[:, None],
            torch.from_numpy(np.stack(dists))[:, None])


@torch.no_grad()
def predict_full(model: nn.Module, image: np.ndarray, mean: float,
                 std: float) -> np.ndarray:
    """Whole-image inference; width 565 is padded to 568 for the two /2 levels."""
    model.eval()
    height, width = image.shape
    pad_h, pad_w = (-height) % 4, (-width) % 4
    tensor = torch.from_numpy((image - mean) / std)[None, None]
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    prob = torch.sigmoid(model(tensor))[0, 0].numpy()
    model.train()
    return prob[:height, :width]


def validate(model, val, mean, std) -> tuple[dict, list[dict]]:
    rows = []
    for index, name in enumerate(val["names"]):
        prob = predict_full(model, val["images"][index], mean, std)
        rows.append({"image": name, **metrics.evaluate(
            prob, val["labels"][index] > 0.5, val["fovs"][index])})
    keys = ("dice", "cldice", "betti0_err", "betti1_err", "hd95")
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}, rows


# --------------------------------------------------------------- run

def train_one(run_name: str, train, val, mean: float, std: float) -> None:
    config_name, seed = run_name.rsplit("_s", 1)
    blurpool, extra = CONFIGS[config_name]
    out_dir = RESULTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path, ckpt_path = out_dir / "final.pt", out_dir / "ckpt.pt"

    if final_path.exists():
        print(f"[{run_name}] already finished, skipping", flush=True)
        return

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = TinyUNet(blurpool=blurpool)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0

    if ckpt_path.exists():
        state = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        rng = state["rng"]
        start_epoch = state["epoch"]
        print(f"[{run_name}] resuming from epoch {start_epoch}", flush=True)

    log_path = out_dir / "log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["epoch", "loss", "dice", "cldice", "betti0_err",
                 "betti1_err", "hd95", "minutes"])

    steps = PATCHES_PER_EPOCH // BATCH
    started = time.time()
    for epoch in range(start_epoch, EPOCHS):
        running = 0.0
        for _ in range(steps):
            images, labels, dists = sample_batch(train, rng, mean, std)
            optimiser.zero_grad()
            loss = compute_loss(model(images), labels, dists, extra)
            loss.backward()
            optimiser.step()
            running += loss.item()
        running /= steps

        last = (epoch + 1) == EPOCHS
        if (epoch + 1) % VAL_EVERY == 0 or last:
            scores, rows = validate(model, val, mean, std)
            minutes = (time.time() - started) / 60
            with log_path.open("a", newline="") as handle:
                csv.writer(handle).writerow(
                    [epoch + 1, round(running, 4)]
                    + [round(scores[k], 4) for k in
                       ("dice", "cldice", "betti0_err", "betti1_err", "hd95")]
                    + [round(minutes, 1)])
            print(f"[{run_name}] epoch {epoch + 1:3d} loss {running:.4f} "
                  f"dice {scores['dice']:.4f} clDice {scores['cldice']:.4f} "
                  f"b0err {scores['betti0_err']:.1f} "
                  f"95HD {scores['hd95']:.2f} ({minutes:.0f} min)", flush=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "optimiser": optimiser.state_dict(), "rng": rng},
                       ckpt_path)
            if last:
                with (out_dir / "val_final.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

    torch.save({"model": model.state_dict(), "epoch": EPOCHS}, final_path)
    print(f"[{run_name}] done in {(time.time() - started) / 60:.0f} min",
          flush=True)


def main() -> None:
    train, val = stack_split("train"), stack_split("val")
    inside = train["images"][train["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())
    print(f"train norm mean {mean:.4f} std {std:.4f}", flush=True)

    wanted = sys.argv[1:]
    run_names = [f"{name}_s{seed}" for seed in (0, 1, 2) for name in CONFIGS]
    for run_name in (wanted or run_names):
        train_one(run_name, train, val, mean, std)


if __name__ == "__main__":
    main()
