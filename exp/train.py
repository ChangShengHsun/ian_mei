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
import augment
import drive
import liot
import metrics

RESULTS = Path(__file__).resolve().parent / "results"
PATCH, BATCH, PATCHES_PER_EPOCH, EPOCHS, VAL_EVERY = 48, 32, 10_000, 100, 10
CKPT_EVERY = 5   # must divide VAL_EVERY; see the note in train_one
DIST_CLIP = 20.0  # px; caps the boundary loss's reach into open background

torch.set_num_threads(6)


# --------------------------------------------------------------- model

def trained_runs() -> list[str]:
    """Every run directory that actually holds a finished checkpoint.

    The seed range used to be written out as `(0, 1, 2)` in each analysis
    script. E12 trained seeds 3-5, both scripts silently scored the old three,
    and the verdict file they fed was WRONG while looking complete -- which
    then opened the gate on the next queue. Enumerate the disk instead: adding
    a seed must never require editing a script that reads the results.
    """
    return sorted(path.parent.name for path in RESULTS.glob("*_s*/final.pt"))


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


def base_width(config_name: str) -> int:
    """Base channel count, read off the config NAME.

    Capacity lives in the name so that every script which loads a checkpoint
    builds the matching architecture from the run name alone. Get this wrong
    and load_state_dict fails loudly rather than silently scoring the wrong
    model, but only if there is one place that decides -- hence build_model
    below, which all the analysis scripts call.

    `A_dice_w32` is 32 base channels; anything without the suffix is the
    original 16.
    """
    head, sep, tail = config_name.rpartition("_w")
    return int(tail) if sep and tail.isdigit() else 16


def uses_liot(config_name: str) -> bool:
    """Whether this config feeds the network LIOT's 4 channels instead of grey.

    Same principle as base_width: the input representation is encoded in the
    name, so a script that only knows a run name still builds a model whose
    first convolution matches the checkpoint.
    """
    return "liot" in config_name


def build_model(config_name: str) -> "TinyUNet":
    """The only place architecture is derived from a config name."""
    blurpool, _ = CONFIGS[config_name]
    return TinyUNet(base=base_width(config_name), blurpool=blurpool,
                    in_channels=len(liot.DIRECTIONS) if uses_liot(config_name)
                    else 1)


class TinyUNet(nn.Module):
    """3-level U-Net. 16 base channels is 117k params, sized for CPU; the
    _w32 configs are 467k. Cost grows sub-quadratically with base: 4x the
    width is 16x the parameters but 9x the measured step time."""

    def __init__(self, base: int = 16, blurpool: bool = False,
                 in_channels: int = 1):
        super().__init__()
        self.enc1 = conv_block(in_channels, base)
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


def weighted_cl_dice(prob: torch.Tensor, target: torch.Tensor,
                     weight: torch.Tensor | float = 1.0) -> torch.Tensor:
    """clDice with a per-pixel weight on every term.

    Every summand below is already masked by skel_pred, skel_true, prob or
    target, so `weight` in empty background only reaches the loss where the
    model hallucinates a skeleton there. That is what lets the weight be a
    plain image-derived map with no structure mask of its own.

    weight = 1.0 reproduces the CVPR 2021 loss exactly.
    """
    skel_pred, skel_true = soft_skeleton(prob), soft_skeleton(target)
    precision = ((weight * skel_pred * target).sum((1, 2, 3)) + 1.0) / (
        (weight * skel_pred).sum((1, 2, 3)) + 1.0)
    sensitivity = ((weight * skel_true * prob).sum((1, 2, 3)) + 1.0) / (
        (weight * skel_true).sum((1, 2, 3)) + 1.0)
    return 1 - (2 * precision * sensitivity / (precision + sensitivity)).mean()


def soft_cl_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return weighted_cl_dice(prob, target)


GATE_GAMMA = 1.0  # weight range [1, 1 + gamma]; 1.0 = at most double


def contrast_weight(image: torch.Tensor, size: int = 15,
                    gamma: float = GATE_GAMMA) -> torch.Tensor:
    """Per-pixel topology weight from local image contrast. Dim structure
    weighs more.

    The signal is the black top-hat (grey closing minus image), which is large
    where a dark thin structure sits inside a brighter surround, i.e. where the
    structure is easy to see. E2 measured that clDice's gain lives entirely in
    the dimmest contrast quartile (+0.0213 Dice) while its cost lives in the
    brightest (-0.0070), so the weight is 1 + gamma where the top-hat is low
    and 1 where it is high: pay the topology price only where it buys something.

    Carries no gradient by construction -- it is a property of the input, not
    of the prediction. That is the whole point of the arm; see contrast_weight
    against confidence_weight.

    Known side-effect: flat background has no top-hat and so draws the maximum
    weight (1.87 measured against 1.05 on the clearest vessels). It only bites
    where the model hallucinates a skeleton in open background, so in practice
    this is extra speckle pressure -- the thing E4 found post-filtering already
    handles. Worth watching in the results, not worth a structure mask that
    would make the weight depend on the labels.
    """
    with torch.no_grad():
        pad = size // 2
        # Grey closing = dilation then erosion; min-pool is erosion of x
        # written as -maxpool(-x), the same identity soft_skeleton uses.
        dilated = F.max_pool2d(image, size, 1, pad)
        closed = -F.max_pool2d(-dilated, size, 1, pad)
        tophat = closed - image
        # Per-patch robust scale: the top-hat is in normalised-image units,
        # which differ between datasets and between CLAHE settings. p99 rather
        # than max so one specular highlight cannot flatten the map, and not
        # p90: vessels are only ~9% of a DRIVE image, so p90 lands at the
        # BOTTOM of the vessel distribution and every vessel pixel clamps to
        # weight 1. Measured on three DRIVE images, p90 gives Q1..Q4 weights
        # 1.26/1.00/1.00/1.00 -- a dead gate -- while p99 gives 1.75/1.54/
        # 1.33/1.05, monotone in E2's contrast bands, which is the point.
        scale = torch.quantile(tophat.flatten(1), 0.99, dim=1)
        scale = scale.clamp(min=1e-3)[:, None, None, None]
        visibility = (tophat / scale).clamp(0.0, 1.0)
        return 1.0 + gamma * (1.0 - visibility)


def confidence_weight(prob: torch.Tensor,
                      gamma: float = GATE_GAMMA) -> torch.Tensor:
    """The discriminating control: same weight range, model-derived signal.

    Hesitation 1 - 2|p - 0.5| is the focal-loss family's notion of a hard pixel,
    and it is exactly the quantity E1' scored against human disagreement: AUROC
    0.881 in the brightest band but 0.373 in the dimmest, where 45.6% of the
    disagreement actually lives. So this arm should help where it is not needed
    and mislead where it is. If F_gated and G_focal come out equal, the claim
    is "weighting helps" and not "the signal must come from the image".

    Detached on purpose: an attached weight lets the model cut the loss by
    growing confident rather than by being right, which is a different
    experiment.
    """
    return 1.0 + gamma * (1.0 - 2 * (prob.detach() - 0.5).abs())


def _cb_weights(mask: torch.Tensor, skeleton: torch.Tensor,
                soft_mask: torch.Tensor, soft_skel: torch.Tensor) -> tuple:
    """The three radius weights of cbDice, ported from the authors' loss/.

    Every weight is built from the Euclidean distance transform of the HARD
    mask, so it carries no gradient; the gradient enters only through the soft
    tensors it multiplies at the end. That is what the reference does too, and
    it is why scipy can be used here inside no_grad.

    Reference: github.com/PengchengShi1220/cbDice, loss/cbdice_loss.py.
    """
    with torch.no_grad():
        binary = mask.detach().cpu().numpy() > 0.5
        distances = torch.from_numpy(
            np.stack([ndimage.distance_transform_edt(item)
                      for item in binary[:, 0]])).float()[:, None]
        distances = distances * (mask > 0.5)
        radius = distances * (skeleton > 0.5)

        # The reference takes the min over the whole radius map, which is 0
        # everywhere off the skeleton, so this clamp makes r_min exactly 1.
        # Kept as-is rather than "fixed": changing it changes the loss.
        r_max = radius.amax((1, 2, 3), keepdim=True).clamp(min=1.0)
        r_min = radius.amin((1, 2, 3), keepdim=True).clamp(min=1.0)

        dist_norm = distances.clamp(max=float(r_max.max())) / r_max
        skel_norm = radius / r_max
        inverse = (r_max - radius + r_min) / r_max
        inverse = inverse * (skeleton > 0.5)
    return dist_norm * soft_mask, skel_norm * soft_mask, inverse * soft_skel


def _combine(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """B*C, except where A is on and B is off, where it falls back to A*C."""
    out = b * c
    fallback = (a != 0) & (b == 0)
    return torch.where(fallback, a * c, out)


def soft_cb_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Centreline-boundary Dice (MICCAI 2024): clDice rebalanced by radius.

    The authors' complaint is not that clDice ignores width but that clDice
    *combined with* Dice ends up "favoring larger vessels": the distance-weighted
    terms scale with diameter, so a thick vessel dominates the sum. cbDice
    multiplies each centreline pixel by an inverse-radius weight so that thin
    and thick vessels contribute comparably. It is the control stage_1 section
    0.2 requires for any claim about vessel width.
    """
    with torch.no_grad():
        hard = (prob > 0.5).float()
        skel_pred_hard = soft_skeleton(hard)
        skel_true = soft_skeleton(target)
    skel_pred = skel_pred_hard * prob

    q_vl, q_slvl, q_sl = _cb_weights(target, skel_true, target, skel_true)
    q_vp, q_spvp, q_sp = _cb_weights(hard, skel_pred_hard, prob, skel_pred)

    precision = ((q_sp * q_vl).sum() + 1.0) / (_combine(q_spvp, q_slvl, q_sp).sum() + 1.0)
    sensitivity = ((q_sl * q_vp).sum() + 1.0) / (_combine(q_slvl, q_spvp, q_sl).sum() + 1.0)
    return 1 - 2 * precision * sensitivity / (precision + sensitivity)


CONFIGS = {
    # name -> (blurpool, extra loss term applied on top of BCE + soft Dice)
    "A_dice": (False, None),
    "B_cldice": (False, "cldice"),
    "C_boundary": (False, "boundary"),
    "D_blurpool": (True, None),
    "E_cbdice": (False, "cbdice"),
    # F and G differ in one thing only: where the per-pixel weight on the
    # clDice term comes from. F reads the image, G reads the model.
    "F_gated": (False, "gated"),
    "G_focal": (False, "focal"),
    # E13. Same two losses at 4x the base width (467k params), to ask whether
    # the +0.0213 dim-band gap between them grows, holds, or closes as the
    # model stops being tiny. Every conclusion in this series so far was
    # measured on 117k parameters against a field standard near 30M, so
    # "the loss barely matters" is currently indistinguishable from
    # "the model is too small for the loss to matter".
    "A_dice_w32": (False, None),
    "B_cldice_w32": (False, "cldice"),
    # E14. Both are BCE+Dice with the loss untouched; the only change is what
    # the network is shown. Twenty experiments compared losses on a pipeline
    # with no augmentation at all, while the CoLeTra paper measures plain
    # augmentation as a 63% cut in Betti error on this same dataset.
    "H_aug": (False, None),
    "I_coletra": (False, None),
    # E13's third arm, added after E14 made it the interesting one: if
    # augmentation is partly compensating for a model too small, its advantage
    # should shrink at 4x the width. If it holds, the input side matters
    # independently of capacity.
    "H_aug_w32": (False, None),
    # E16. Identical to H_aug in loss and in augmentation; the only difference
    # is that the network is shown LIOT's four contrast-invariant channels
    # instead of grey. Every intervention so far tried to make the model care
    # more about dim vessels; this one deletes the difference between dim and
    # bright before the model sees it. E15 measured 82% of the achievable run
    # length still missing in the dimmest band and under 2% in the two
    # brightest, so that band is the entire remaining budget.
    "J_liot": (False, None),
}

# Which augmentations each config gets. Keeping this beside CONFIGS rather than
# inside it leaves the (blurpool, extra) shape that eleven call sites unpack.
AUGMENTS = {
    "H_aug": ("dihedral", "jitter"),
    "I_coletra": ("dihedral", "jitter", "coletra"),
    # Deliberately the same tuple as H_aug so the two arms differ in exactly
    # one thing. Keeping jitter is not an oversight: LIOT is invariant to any
    # increasing map of the intensities, so jitter is close to a no-op here,
    # and how close is a measurement (test_liot_pipeline.py) rather than an
    # assumption. Dropping it would confound the representation with the
    # augmentation set.
    "J_liot": ("dihedral", "jitter"),
    # Must match H_aug exactly. AUGMENTS is keyed on the FULL config name, so
    # a width variant that is not listed here trains with no augmentation at
    # all and nothing complains -- it just quietly stops being the arm it is
    # named after. test_capacity.py asserts every _w config agrees with its
    # base, which is the check that would have caught this.
    "H_aug_w32": ("dihedral", "jitter"),
}


def compute_loss(logits, target, dist, extra, image=None):
    prob = torch.sigmoid(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    if extra == "cldice":
        # alpha = 0.5, the split used in the clDice paper (CVPR 2021).
        loss = loss + 0.5 * soft_dice(prob, target) + 0.5 * soft_cl_dice(prob, target)
    elif extra in ("gated", "focal"):
        # Same 0.5/0.5 split as B_cldice, so B, F and G differ only in the
        # weight map and any difference between them is attributable to it.
        weight = (contrast_weight(image) if extra == "gated"
                  else confidence_weight(prob))
        loss = loss + 0.5 * soft_dice(prob, target) + 0.5 * weighted_cl_dice(
            prob, target, weight)
    elif extra == "cbdice":
        # Same 0.5/0.5 split as clDice so the two are directly comparable; the
        # only variable between B and E is whether the centreline is weighted.
        loss = loss + 0.5 * soft_dice(prob, target) + 0.5 * soft_cb_dice(prob, target)
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


def normalisation(name: str, data: dict):
    """The mean and std a run was trained with, chosen from its name.

    This exists because splitting one decision across two places cost E16 a
    whole verdict. predict_full inspects the model to decide whether to encode
    the image as LIOT, but the constants stayed an argument, so stratify.py,
    break_lengths.py and erl.py encoded correctly and then normalised 0-255
    byte codes with grey statistics (mean 0.42, std 0.14). The inputs landed
    some 800 standard deviations out, every LIOT model predicted nothing, and
    the analysis reported Dice 0.0000 as though it were a result about LIOT.

    So representation and constants are decided together, here, once.
    """
    if uses_liot(name.rsplit("_s", 1)[0]):
        return liot_stats(data)
    inside = data["images"][data["fovs"]]
    return float(inside.mean()), float(inside.std())


def liot_stats(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of the LIOT code over the training aperture.

    Per channel, not one number for all four: the four directions do not have
    the same distribution on a retina, where vessels run predominantly out of
    the disc. Statistics are taken inside the FOV only, for the same reason the
    grey ones are -- the black surround is a third of the frame and would drag
    both numbers toward a value no real pixel takes.
    """
    inside = np.concatenate(
        [liot.liot(image)[:, fov] for image, fov
         in zip(data["images"], data["fovs"])], axis=1).astype(np.float32)
    return (inside.mean(1)[:, None, None], inside.std(1)[:, None, None])


def sample_batch(data: dict, rng: np.random.Generator, mean, std,
                 augments: tuple = (), inpainted: np.ndarray | None = None,
                 use_liot: bool = False):
    """One batch of random crops, optionally augmented.

    Order matters. CoLeTra reads from the inpainted copy of the SAME crop, so
    it runs before any geometry is applied; the symmetry acts on image, label
    and distance map together; the photometric jitter is last because it is the
    only step that must not touch the label.

    LIOT comes after all of them and needs a wider crop than it returns. Its
    code at a pixel reads up to liot.MARGIN away, so computing it on a bare
    48 px patch would give every patch a border of pixels whose rays were
    clamped at the crop edge -- an artefact that only exists in training,
    never at inference on a whole image. Crop wide, encode, then trim back.
    """
    pad = liot.MARGIN if use_liot else 0
    size = PATCH + 2 * pad
    inner = slice(pad, pad + PATCH)
    height, width = data["images"].shape[1:]
    images, labels, dists = [], [], []
    while len(images) < BATCH:
        index = rng.integers(len(data["images"]))
        top = rng.integers(height - size)
        left = rng.integers(width - size)
        window = (slice(top, top + size), slice(left, left + size))
        # Skip patches whose centre lies outside the aperture: they are pure
        # black corner and teach the network nothing.
        if not data["fovs"][index][top + size // 2, left + size // 2]:
            continue
        image = data["images"][index][window]
        label = data["labels"][index][window]
        dist = data["dists"][index][window]
        if "coletra" in augments:
            image = augment.coletra(image, label, rng,
                                    inpainted=inpainted[index][window])
        if "dihedral" in augments:
            image, label, dist = augment.dihedral(rng, image, label, dist)
        if "jitter" in augments:
            image = augment.jitter(image, rng)
        if use_liot:
            image = liot.liot(image).astype(np.float32)[:, inner, inner]
        else:
            image = image[inner, inner]
        images.append(image)
        labels.append(label[inner, inner])
        dists.append(dist[inner, inner])
    batch = np.stack(images)
    if batch.ndim == 3:
        batch = batch[:, None]
    # mean/std are scalars for grey input and shape (C, 1, 1) for LIOT, so the
    # same expression normalises both.
    batch = ((batch - mean) / std).astype(np.float32)
    return (torch.from_numpy(batch),
            torch.from_numpy(np.stack(labels))[:, None],
            torch.from_numpy(np.stack(dists))[:, None])


@torch.no_grad()
def predict_full(model: nn.Module, image: np.ndarray, mean, std) -> np.ndarray:
    """Whole-image inference; width 565 is padded to 568 for the two /2 levels.

    Whether to encode is read off the model rather than passed in, because
    every analysis script calls this with just a checkpoint and an image. A
    LIOT model fed raw grey would not raise -- a 4-channel first convolution
    given 1 channel does raise, which is the point of deciding here.
    """
    model.eval()
    height, width = image.shape
    channels = model.enc1[0].in_channels
    # The constants must match the representation. Getting this wrong does not
    # raise on its own -- it normalises byte codes with grey statistics and the
    # model silently predicts nothing, which is how E16's first verdict came
    # out as Dice 0.0000 and was very nearly published as a fact about LIOT.
    if channels > 1 and np.ndim(mean) == 0:
        raise ValueError(
            f"a {channels}-channel model was given scalar normalisation "
            f"constants; use train.normalisation(run_name, train_split)")
    if channels == 1 and np.ndim(mean) != 0:
        raise ValueError("a 1-channel model was given per-channel constants")
    if channels == len(liot.DIRECTIONS):
        # On a whole image there is no crop border, so no margin is needed.
        encoded = liot.liot(image).astype(np.float32)
    else:
        encoded = image[None]
    pad_h, pad_w = (-height) % 4, (-width) % 4
    tensor = torch.from_numpy(((encoded - mean) / std).astype(np.float32))
    tensor = F.pad(tensor[None], (0, pad_w, 0, pad_h), mode="reflect")
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
    _, extra = CONFIGS[config_name]  # build_model owns the architecture half
    augments = AUGMENTS.get(config_name, ())
    use_liot = uses_liot(config_name)
    if use_liot:
        # The grey mean/std main() computed are meaningless for a byte-code
        # input, so recompute here rather than making every caller know.
        mean, std = liot_stats(train)
    # Inpaint the whole training split once rather than per crop: it is
    # deterministic, and grey_closing on 20 full images costs seconds
    # against 31,200 crops per run.
    inpainted = (np.stack([augment.remove_structures(image)
                           for image in train["images"]])
                 if "coletra" in augments else None)
    out_dir = RESULTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path, ckpt_path = out_dir / "final.pt", out_dir / "ckpt.pt"

    if final_path.exists():
        print(f"[{run_name}] already finished, skipping", flush=True)
        return

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = build_model(config_name)
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
            images, labels, dists = sample_batch(
                train, rng, mean, std, augments, inpainted, use_liot)
            optimiser.zero_grad()
            loss = compute_loss(model(images), labels, dists, extra, images)
            loss.backward()
            optimiser.step()
            running += loss.item()
        running /= steps

        last = (epoch + 1) == EPOCHS
        # Checkpointing is decoupled from validation: validation costs a pass
        # over 20 full images, saving 117k parameters costs nothing, and the
        # thing we are insuring against is the machine sleeping mid-run, which
        # it did twice on 2026-08-19. VAL_EVERY is a multiple of CKPT_EVERY, so
        # a validated epoch is also a saved one.
        if (epoch + 1) % CKPT_EVERY == 0 or last:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "optimiser": optimiser.state_dict(), "rng": rng},
                       ckpt_path)
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
