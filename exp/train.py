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
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment
import direction
import drive
import propagate
import liot
import metrics

RESULTS = Path(__file__).resolve().parent / "results"
PATCH, BATCH, PATCHES_PER_EPOCH, EPOCHS, VAL_EVERY = 48, 32, 10_000, 100, 10
CKPT_EVERY = 5   # must divide VAL_EVERY; see the note in train_one
# Set by main() from --keep-epochs. When on, every validated epoch's weights
# are kept as epoch{N:03d}.pt instead of only the last and the best.
#
# Why it exists: best.pt is chosen by whole-image validation Dice, and
# K_focal_aug's Dice peaks at epoch 10 (median over six seeds) while H_aug's
# peaks at 65 -- with identical augmentation, so it is the confidence-gated
# clDice loss doing it. E13b's ERL then measured that K gets WORSE under
# early stopping (46.1% -> 38.1% of the tree traced) while the baseline gets
# better. So K keeps improving topologically through the epochs where its
# Dice is falling, and selecting on Dice deliberately discards exactly that.
# Deciding which epoch to keep is a question about the data, not a constant,
# so the runs keep all of them and the selection rule is fitted afterwards on
# the VALIDATION numbers only.
KEEP_EPOCHS = False

# Which images the model is fitted on and which ones choose its checkpoint.
#
#   legacy   fit on all 20 of DRIVE's train directory, select best.pt by Dice
#            on the val directory -- which is DRIVE's official TEST set. Every
#            run in exp/results predating 2026-08-28 was trained this way, so
#            it stays the default until they are all superseded; changing it
#            under a running queue would give one comparison two protocols.
#   heldout  fit on 15 images, select on the 5 held out from the SAME train
#            directory, and let the test set be read once, at scoring time.
#
# The leak legacy carries is not that the test set is scored -- it must be --
# but that it CHOOSES. Selecting the best of ten epochs on the set you then
# report makes the reported number the maximum of ten draws, not the model's
# score. log.csv holds every validated epoch, so under heldout the same
# selection rules can be recomputed honestly from the dev column.
PROTOCOL = "legacy"
PROTOCOL_SPLITS = {"legacy": ("train", "val"), "heldout": ("fit", "dev")}
DIST_CLIP = 20.0  # px; caps the boundary loss's reach into open background

torch.set_num_threads(6)

# One module-level decision, imported by the analysis scripts rather than
# re-derived at each call site. That is E16's lesson (stage-report/README.md,
# lesson seven): half a decision moved is how a script once encoded LIOT
# correctly, normalised it with grey constants, and published Dice 0.0000.
# WHICH card is not decided here -- CUDA_VISIBLE_DEVICES does that, so the
# queue can pin one job per GPU without the code knowing there are two.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    """Every checkpoint read goes through here, so map_location is set once.

    Checkpoints written on the CPU laptop have to load on this box and back
    again. Without map_location torch.load restores each tensor to the device
    it was saved from, which inside a GPU script is a silently-CPU model that
    still runs -- at laptop speed, with no error to notice.
    """
    return torch.load(path, map_location=DEVICE, weights_only=False)


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


class ShapeNameError(ValueError):
    """A run name carries an architecture suffix that cannot be parsed."""


def _shape_suffix(config_name: str, letter: str, default: int) -> int:
    """Read one architecture number out of a run name, e.g. `w64` -> 64.

    Tokenised rather than rpartitioned so a name can carry more than one
    suffix and the order between them does not matter: `A_dice_w64_d5` and
    `A_dice_d5_w64` are the same architecture.

    THE FAILURE MODE THIS RAISES FOR. The original parser returned the
    default whenever the suffix did not parse. Name a 31M config
    `A_dice_w64d5` and `64d5` is not a digit string, so it answered 16: a
    117k model trains for fifteen hours, saves under a 30M name, and every
    number on the capacity curve is wrong with nothing printed anywhere. That
    is the shape of the two bugs this repo has already paid for -- E12's
    hand-written seed range and E16's split normalisation constants -- and
    both were silent in exactly this way.

    So the three cases are kept apart deliberately:

      `w64`       a suffix that parses            -> 64
      `w64d5`     a suffix that does NOT parse    -> raise
      `weighted`  not a suffix at all             -> keep looking

    The third is why the test cannot simply be "starts with the letter":
    `X_weighted` is a legitimate config name and has always meant base 16.
    The discriminator is a digit -- a token carrying digits after the letter
    was meant to be read as a number, and failing to read it is an error, not
    a default. A repeated suffix (`A_dice_w32_w64`) is the same kind of
    ambiguity and raises for the same reason.
    """
    found = None
    for token in config_name.split("_"):
        if not token.startswith(letter):
            continue
        if token == letter:
            # `B_cldice_w_64`: the marker and its number split by a stray
            # underscore. Nothing downstream would ever see the 64, so this
            # is the same fifteen-hour bug wearing a different typo.
            raise ShapeNameError(
                f"{config_name!r} has a bare {letter!r} token with no number "
                f"attached; write it as {letter}<digits>.")
        rest = token[len(letter):]
        if not any(character.isdigit() for character in rest):
            continue                      # a word, not a number: `weighted`
        if not rest.isdigit():
            raise ShapeNameError(
                f"{config_name!r} carries {token!r}, which is not "
                f"{letter!r} followed by digits. Returning the default "
                f"{default} here would train the wrong architecture under "
                f"the right name; rename the config instead.")
        if found is not None:
            raise ShapeNameError(
                f"{config_name!r} sets {letter!r} more than once "
                f"({found} and {rest}); one name must mean one architecture.")
        found = rest
    return int(found) if found is not None else default


def base_width(config_name: str) -> int:
    """Base channel count, read off the config NAME.

    Capacity lives in the name so that every script which loads a checkpoint
    builds the matching architecture from the run name alone. Get this wrong
    and load_state_dict fails loudly rather than silently scoring the wrong
    model, but only if there is one place that decides -- hence build_model
    below, which all the analysis scripts call.

    `A_dice_w32` is 32 base channels; anything without the suffix is the
    original 16. A suffix that is present but unparseable raises
    ShapeNameError rather than quietly falling back -- see _shape_suffix.
    """
    return _shape_suffix(config_name, "w", 16)


def net_depth(config_name: str) -> int:
    """Number of U-Net levels, read off the config NAME. Default 3.

    Same mechanism as base_width and for the same reason. Depth changes the
    module list itself, so getting it from anywhere other than the name means
    a checkpoint can load into a net with the wrong number of levels -- or,
    worse, into one where the names happen to line up and only some tensors
    are restored. Unparseable suffixes raise here too.
    """
    return _shape_suffix(config_name, "d", 3)


def uses_liot(config_name: str) -> bool:
    """Whether this config feeds the network LIOT's 4 channels instead of grey.

    Same principle as base_width: the input representation is encoded in the
    name, so a script that only knows a run name still builds a model whose
    first convolution matches the checkpoint.
    """
    return "liot" in config_name


# D1's auxiliary loss weight. The same 0.5 the clDice arms give their second
# term, so it is a weight this repo already uses rather than a new number
# tuned on a result. PRE-REGISTERED 2026-08-27, before the first _dir run.
DIRECTION_WEIGHT = 0.5


def uses_direction(config_name: str) -> bool:
    """Whether this config carries D1's auxiliary tangent-direction head.

    Encoded in the name for the same reason width and depth are: an analysis
    script holding only a run name has to build the architecture the
    checkpoint was saved from. `_dir` adds a second 1x1 head and changes
    nothing else -- same backbone, same segmentation loss, same augmentation
    as the arm it is named after.
    """
    return "dir" in config_name.split("_")


_WIDTH_CACHE: dict = {}


def vessel_width() -> float:
    """Median vessel width of the TRAINING split, in pixels. Cached.

    The name carries MULTIPLES of this, never pixels -- CLAUDE.md's rule, and
    here it is load-bearing rather than tidy: HRF is about six times DRIVE's
    resolution, so a layer built from a pixel count would silently become a
    different operator on transfer while keeping the same name. The conversion
    happens here, once, at the boundary.
    """
    if "width" not in _WIDTH_CACHE:
        import cross_dataset
        _WIDTH_CACHE["width"] = cross_dataset.median_width(
            drive.load_split("train"))
    return _WIDTH_CACHE["width"]


def propagation_geometry(config_name: str) -> tuple[float, float]:
    """(along, across) IN PIXELS, read from the config NAME.

    `a` and `c` are HUNDREDTHS of a vessel width, so `_a100_c025` is one
    width along the vessel and a quarter of one across it. Same mechanism as
    `w64` and `d5`, same parser, and unparseable suffixes raise for the same
    reason.

    THIS USED TO LIVE IN A FILE, exp/results/d1_geometry.txt, written by the
    gate. That was wrong for the reason base_width exists: one config name has
    to mean one architecture. With the geometry in a file, training the same
    arm at two reaches produced two different networks under one name, into
    one directory, and the second silently replaced the first. It also cost a
    run: the file held width multiples and this function handed them to
    build_model as pixels, so `along=1.0` built a one-pixel kernel and an
    entire 24-run queue measured a layer that was the identity.
    """
    along = _shape_suffix(config_name, "a", 0)
    across = _shape_suffix(config_name, "c", 0)
    if along == 0 and across == 0:
        # The same failure _shape_suffix raises for, one level up. A missing
        # reach used to default to zero, which builds a one-pixel kernel: the
        # layer is the identity, the arm trains for hours, and it reports that
        # propagation does nothing. Refusing here is the whole point.
        raise ShapeNameError(
            f"{config_name!r} asks for the propagation layer but carries no "
            f"reach. Write it as ..._prop_a<hundredths>_c<hundredths>, e.g. "
            f"_prop_a100_c025 for one vessel width along and a quarter "
            f"across. A missing reach would build a one-pixel kernel and the "
            f"layer would be the identity.")
    return (along / 100.0 * vessel_width(),
            across / 100.0 * vessel_width())


def uses_propagation(config_name: str) -> bool:
    """D-B: the oriented propagation layer, inside the network."""
    return "prop" in config_name.split("_")


def uses_shuffled_direction(config_name: str) -> bool:
    """D-B's ablation: the same layer driven by a MEANINGLESS axis field.

    The control the whole architecture rests on. Oriented propagation adds
    evidence, and adding evidence moves ERL on its own; if the shuffled arm
    matches the real one, the layer is a dilation with a direction-shaped
    parameter and D-B has measured nothing.
    """
    return "shuf" in config_name.split("_")


# D-E's weight on ground-truth centreline pixels, on top of the 1.0 every
# pixel already carries. DERIVED, not chosen: DRIVE's median vessel is 2.83 px
# across, so a cross-section is about three pixels of which one is centreline.
# At weight 2 the centreline counts for as much as the rest of the vessel put
# together, which is the strongest form of "cover the centreline" that does
# not simply outvote the body.
CENTRELINE_WEIGHT = 2.0


def uses_centreline_weight(config_name: str) -> bool:
    """D-E: the competitor that could make this whole line unnecessary.

    The measured error is that the prediction covers the vessel but misses its
    centreline. The cheapest possible answer to that is to weight centreline
    pixels in the loss -- no direction field, no extra head, no new layer. If
    this arm captures the budget, D1 is an expensive route to something one
    weight map already does, and that has to be known before D-B is believed.
    """
    return any(token == "clw" or (token.startswith("clw")
                                  and token[3:].isdigit())
               for token in config_name.split("_"))


def centreline_weight(config_name: str) -> float:
    """The extra weight on centreline pixels, read from the config NAME.

    `clw` alone is 2.0, the value the first twelve D-E runs were trained at,
    so those runs keep meaning what they meant. `clw4` is 4.0. Same rule as
    propagation_geometry: a number that changes the operator belongs in the
    name, or one name means two models writing into one directory.
    """
    for token in config_name.split("_"):
        if token == "clw":
            return CENTRELINE_WEIGHT
        if token.startswith("clw") and token[3:].isdigit():
            return float(token[3:])
    raise ShapeNameError(f"{config_name!r} has no clw token")


def build_model(config_name: str) -> "TinyUNet":
    """The only place architecture is derived from a config name."""
    blurpool, _ = CONFIGS[config_name]
    return TinyUNet(base=base_width(config_name), blurpool=blurpool,
                    in_channels=len(liot.DIRECTIONS) if uses_liot(config_name)
                    else 1, depth=net_depth(config_name),
                    direction=uses_direction(config_name),
                    propagation=(propagation_geometry(config_name)
                                 if uses_propagation(config_name)
                                 else None),
                    shuffle=uses_shuffled_direction(config_name)).to(DEVICE)


class TinyUNet(nn.Module):
    """U-Net of `depth` levels. 3 levels at 16 base channels is 117k params,
    the size everything up to E18 was measured at; the _w32 configs are 467k.

    Depth 5 at base=64 is ~31M, the field-standard shape. It is not the same
    31M as widening this net to base=256: at depth 5 the extra capacity sits
    at 6x6 and 3x3, where a parameter costs almost no compute, instead of at
    full patch resolution where every one of them is paid for on every pixel.

    Module names are built to match the hand-written 3-level version exactly
    (enc1..encN, down1.., up1.., dec1.., head), so a checkpoint from before
    this generalisation still loads.
    """

    def __init__(self, base: int = 16, blurpool: bool = False,
                 in_channels: int = 1, depth: int = 3,
                 direction: bool = False, propagation: tuple | None = None,
                 shuffle: bool = False):
        super().__init__()
        self.depth = depth
        channels = [base * 2 ** level for level in range(depth)]
        previous = in_channels
        for level, width in enumerate(channels, start=1):
            setattr(self, f"enc{level}", conv_block(previous, width))
            previous = width
        for level, width in enumerate(channels[:-1], start=1):
            setattr(self, f"down{level}",
                    BlurPool(width) if blurpool else nn.MaxPool2d(2))
        for level in range(depth - 1, 0, -1):
            wide, narrow = channels[level], channels[level - 1]
            setattr(self, f"up{level}",
                    nn.ConvTranspose2d(wide, narrow, 2, 2))
            setattr(self, f"dec{level}", conv_block(narrow * 2, narrow))
        self.head = nn.Conv2d(base, 1, 1)
        # D1. Two channels, not one: the target is (sin 2theta, cos 2theta),
        # because a vessel tangent is an axis and a single angle tears at the
        # wrap-around. See exp/direction.py for why, and for the test.
        self.dir_head = nn.Conv2d(base, 2, 1) if direction else None
        # D-B. Between the trunk and the head, driven by dir_head's own
        # output, so a wrong field costs segmentation accuracy directly.
        self.propagation = (propagate.OrientedPropagation(*propagation)
                            if propagation is not None else None)
        # D-B's ablation. The layer runs on a random axis field instead of the
        # head's, so the head predicts direction (the auxiliary loss still
        # sees it) but nothing downstream uses it. Set on the MODEL rather
        # than at a call site because inference has to be ablated too -- an
        # ablation that only holds during training is not one.
        self.shuffle_field = shuffle

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Everything up to the 1x1 heads. Shared by both of them."""
        skips = []
        for level in range(1, self.depth):
            skips.append(getattr(self, f"enc{level}")(x))
            x = getattr(self, f"down{level}")(skips[-1])
        x = getattr(self, f"enc{self.depth}")(x)
        for level in range(self.depth - 1, 0, -1):
            x = getattr(self, f"dec{level}")(torch.cat(
                [getattr(self, f"up{level}")(x), skips[level - 1]], 1))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segmentation logits, after propagation if this model has it.

        Deliberately unchanged in shape by D1: twenty analysis scripts call
        model(image) and index [0, 0]. A _dir model reached through forward()
        is exactly the segmentation model it would have been, so every
        existing script scores it correctly with no edit.
        """
        if self.propagation is None:
            return self.head(self.features(x))
        # A _prop model's segmentation is not defined without its field, so
        # forward() computes both and returns only the logits -- every
        # analysis script that calls model(image) then scores the real thing
        # rather than the pre-propagation logits.
        return self.forward_direction(x)[0]

    def forward_direction(self, x: torch.Tensor) -> tuple:
        """(segmentation logits, tangent field). Training and D1 only."""
        if self.dir_head is None:
            raise ValueError("this model was built without a direction head")
        shared = self.features(x)
        logits, field = self.head(shared), self.dir_head(shared)
        if self.propagation is not None:
            driving = field
            if self.shuffle_field:
                angle = torch.rand_like(field[:, :1]) * torch.pi
                driving = torch.cat([torch.sin(2 * angle),
                                     torch.cos(2 * angle)], dim=1)
            logits = self.propagation(logits, driving)
        return logits, field


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
                      for item in binary[:, 0]])).float()[:, None].to(
            mask.device)
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
    # E18. The two best arms in this series have never been run together.
    # G_focal wins the dim band and PAYS 1.85 extra severing breaks (E12);
    # H_aug removes 3.2 of them (E14); E15 puts them level on ERL. Opposite
    # mechanisms at the same score is the definition of worth combining.
    # Not a novelty claim: CoLeTra already crossed augmentation with six
    # losses. What is untested is this gate, whose weight comes from the
    # model's own hesitancy, together with augmentation.
    "K_focal_aug": (False, "focal"),
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
    # Task 1 of prompt.md: the five arms that carry the series' conclusions,
    # re-run on a 5-level base=64 U-Net (~31M params) instead of the 117k one
    # every earlier number was measured on. Nothing but the backbone changes;
    # the losses and augmentation tuples below are the same objects the
    # narrow arms use, so any difference is attributable to capacity.
    # A_dice/B_cldice/H_aug at this width are also E13's third point.
    "A_dice_w64_d5": (False, None),
    "B_cldice_w64_d5": (False, "cldice"),
    "H_aug_w64_d5": (False, None),
    "G_focal_w64_d5": (False, "focal"),
    "K_focal_aug_w64_d5": (False, "focal"),
    # D1 (2026-08-27). Identical to the arm each is named after -- same
    # backbone, same segmentation loss, same augmentation tuple -- plus a
    # 2-channel 1x1 head predicting the vessel tangent axis. 34 extra
    # parameters out of 117,393, so a difference is the auxiliary TASK and
    # not capacity. Two questions: does predicting direction improve
    # segmentation, and is the predicted field a better cost map for C1's
    # linker than 1-p.
    "A_dice_dir": (False, None),
    "H_aug_dir": (False, None),
    # D-E. No direction at all: BCE with the ground-truth centreline weighted
    # up. The cheap competitor that could make the whole line unnecessary.
    "A_dice_clw": (False, None),
    "H_aug_clw": (False, None),
}

# D-B, swept over the layer's REACH. The first attempt handed over a single
# geometry, chosen at the tightest Dice budget of the post-hoc sweep, and it
# built a 5x5 kernel holding three pixels per orientation. The gate opened for
# a real field (0.15) and shut for a random one (0.03), so the network could
# tell them apart -- but a three-pixel operator had nothing to give. The Dice
# constraint had been applied twice: once when choosing the reach, and again
# by the training loss, which is what the gate is for.
#
# So the reach is swept instead of chosen, and it is in the NAME, because one
# config name must mean one architecture.
#
#   a050  half a vessel width along the vessel   (5x5 kernel)
#   a100  one width                              (7x7)
#   a200  two widths                             (13x13)
#   c025  a quarter width across it, at every reach -- the post-hoc sweep
#         chose 0.25 at every budget it was asked about
#
# Each reach gets its shuffled control: the same layer on a random axis field,
# which is the comparison that says this is direction and not dilation.
PROPAGATION_REACHES = ("a050", "a100", "a200")
for _base, _extra in (("A_dice", None), ("H_aug", None)):
    for _reach in PROPAGATION_REACHES:
        for _shuffle in ("", "_shuf"):
            CONFIGS[f"{_base}_dir_prop{_shuffle}_{_reach}_c025"] = (False,
                                                                   _extra)
        # D-B x D-E: both interventions at the same reach. The only question
        # worth asking once each works alone -- do they compose, or are they
        # two routes to the same points?
        CONFIGS[f"{_base}_clw_dir_prop_{_reach}_c025"] = (False, _extra)

# D-E swept over its one number. The first twelve runs fixed the weight at 2
# by an argument from vessel geometry -- a 2.83 px cross-section is about
# three pixels of which one is centreline -- and that argument says nothing
# about whether 2 is the best of the reachable values, only that it is not
# absurd. H_aug_clw is the single intervention in this series that passed the
# seed gate, and it is the least swept thing in it.
#
# K_focal_aug joins the sweep because it is the strongest arm measured and has
# never been crossed with D-E at all. clw2 duplicates the published `clw` on
# purpose: under the held-out protocol every arm is retrained, so the sweep
# needs its own weight-2 point rather than a legacy one measured differently.
# 2026-08-29: extended to 16 and 32. The first sweep ran 1/2/4/8 and the
# response was MONOTONE INCREASING to its endpoint -- 8 was the best weight on
# all three bases, at matched Dice as well as at rule (iv). A sweep whose best
# value is its largest value has not found a peak, it has found the edge of
# the range. The pre-registered prediction that the response would be
# single-peaked and that 8 would cost Dice without buying run length was
# simply wrong, and the honest repair is to keep going until it turns.
#
# 64 is in the list to BRACKET the peak rather than to win. At 64 the
# centreline outweighs the vessel body 65:1, so the model should collapse
# toward drawing the skeleton and nothing else. A sweep that ends before the
# collapse cannot say where the peak is; one that contains the collapse can.
CENTRELINE_WEIGHTS = (1, 2, 4, 8, 16, 32, 64)
for _base, _extra in (("A_dice", None), ("H_aug", None),
                      ("K_focal_aug", "focal")):
    for _weight in CENTRELINE_WEIGHTS:
        CONFIGS[f"{_base}_clw{_weight}"] = (False, _extra)

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
    # Exactly H_aug's tuple: K differs from H_aug only in the loss, and from
    # G_focal only in the augmentation, so either comparison isolates one thing.
    "K_focal_aug": ("dihedral", "jitter"),
    # Must match their narrow namesakes exactly; test_capacity.py asserts it.
    "H_aug_w64_d5": ("dihedral", "jitter"),
    "K_focal_aug_w64_d5": ("dihedral", "jitter"),
    # Must match H_aug exactly; test_capacity.py asserts every variant does.
    "H_aug_dir": ("dihedral", "jitter"),
    # Every H_aug variant must carry H_aug's tuple exactly; test_capacity.py
    # and gpu_queue's selftest both assert it, because a variant missing from
    # here trains unaugmented and still answers to the augmented arm's name.
    "H_aug_dir_prop": ("dihedral", "jitter"),
    "H_aug_dir_prop_shuf": ("dihedral", "jitter"),
    "H_aug_clw": ("dihedral", "jitter"),
}

# Every H_aug variant must carry H_aug's tuple exactly. Generated beside the
# configs rather than typed out, because a name missing from here trains with
# no augmentation at all and still answers to the augmented arm's name -- the
# trap E13 already paid for once, and test_capacity.py asserts against it.
# K_focal_aug is in the prefix list because D-E's sweep is the first thing to
# make a variant of it. Its tuple is H_aug's; the arm differs in its loss.
for _name in list(CONFIGS):
    for _prefix in ("H_aug_", "K_focal_aug_"):
        if _name.startswith(_prefix) and _name not in AUGMENTS:
            AUGMENTS[_name] = AUGMENTS["H_aug"]


def direction_loss(field, target) -> torch.Tensor:
    """Coherence-weighted MSE on (sin 2theta, cos 2theta), on vessel pixels.

    Weighted, not plain: a junction has no single tangent, and coherence is
    near zero exactly there, so the head is not charged for failing to invent
    one. Restricted to the vessel because the tangent of a background pixel
    is not a quantity -- charging it would spend most of the loss on the 88%
    of the frame that has no direction at all.

    The head is NOT asked to produce a unit vector. Its magnitude is free to
    fall where the target's coherence is low, which is the same information
    the weight carries and costs nothing to allow.
    """
    weight = target[:, 2:3]
    error = (field - target[:, :2]) ** 2
    total = weight.sum() * 2 + 1e-6
    return (error * weight).sum() / total


def centreline_loss(logits, target, skeleton, weight: float) -> torch.Tensor:
    """D-E. BCE with ground-truth centreline pixels weighted up.

    No direction anywhere in it. That is the point: it is the cheapest thing
    that answers the measurement -- the prediction covers the vessel and
    misses its centreline -- and if it captures the budget then D1's field,
    head and layer are an expensive route to what one weight map does.
    """
    per_pixel = 1.0 + weight * skeleton
    return F.binary_cross_entropy_with_logits(logits, target, weight=per_pixel)


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

def stack_split(split: str, items: list | None = None) -> dict:
    """DRIVE's split by name, or an already-loaded list of items.

    The `items` argument is what lets this trainer run on HRF, STARE and
    VessMAP without a second copy of the training loop -- cross_dataset.py
    has one, from before augmentation and the direction head existed, and a
    transfer arm trained by it would differ from its DRIVE namesake in more
    than the dataset.
    """
    if items is None:
        items = drive.load_split(split)
    images = np.stack([item["image"] for item in items])
    labels = np.stack([item["label"] for item in items]).astype(np.float32)
    fovs = np.stack([item["fov"] for item in items])
    # Signed distance to the vessel boundary: negative inside, positive out.
    dists = np.stack([
        ndimage.distance_transform_edt(~item["label"])
        - ndimage.distance_transform_edt(item["label"]) for item in items])
    dists = np.clip(dists, -DIST_CLIP, DIST_CLIP).astype(np.float32) / DIST_CLIP
    # D1's target: the tangent axis at every pixel, in double-angle form,
    # with the coherence that says where it means anything. Built here rather
    # than per crop because it is deterministic and three gaussian filters on
    # 20 full images cost seconds against 1M crops per run.
    fields = [direction.tangent_field(item["label"] > 0.5) for item in items]
    # D-E's target: the ground-truth centreline. Precomputed because
    # skeletonize on a 48 px crop, a million times a run, is not affordable.
    skeletons = np.stack([skeletonize(item["label"] > 0.5).astype(np.float32)
                          for item in items])
    return {"images": images, "labels": labels, "fovs": fovs, "dists": dists,
            "skel": skeletons,
            "dir_sin": np.stack([f[0] for f in fields]),
            "dir_cos": np.stack([f[1] for f in fields]),
            "dir_weight": np.stack([f[2] for f in fields]),
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
                 use_liot: bool = False, use_direction: bool = False,
                 use_skeleton: bool = False):
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
    sines, cosines, weights, skeletons = [], [], [], []
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
        # Read only when wanted: cross_dataset, train_stare, bench_step and
        # the LIOT tests each build their own data dict, and none of them
        # carries these planes. An unconditional read makes D1's target a
        # requirement of every caller instead of of the one that asked.
        sine = cosine = weight = skeleton = None
        if use_skeleton:
            skeleton = data["skel"][index][window]
        if use_direction:
            sine = data["dir_sin"][index][window]
            cosine = data["dir_cos"][index][window]
            weight = data["dir_weight"][index][window]
        if "coletra" in augments:
            image = augment.coletra(image, label, rng,
                                    inpainted=inpainted[index][window])
        if "dihedral" in augments:
            # The tangent field's VALUES move too, not just its pixels. Moving
            # the planes alone leaves a field a quarter turn off the vessel it
            # is drawn on, which trains to a plausible-looking nothing; the
            # transform and the test that catches it are in direction.py.
            turns, flip = augment.dihedral_choice(rng)
            image, label, dist = augment.apply_dihedral(
                turns, flip, image, label, dist)
            if use_skeleton:
                skeleton, = augment.apply_dihedral(turns, flip, skeleton)
            if use_direction:
                weight, = augment.apply_dihedral(turns, flip, weight)
                sine, cosine = direction.dihedral(turns, flip, sine, cosine)
        if "jitter" in augments:
            image = augment.jitter(image, rng)
        if use_liot:
            image = liot.liot(image).astype(np.float32)[:, inner, inner]
        else:
            image = image[inner, inner]
        images.append(image)
        labels.append(label[inner, inner])
        dists.append(dist[inner, inner])
        if use_direction:
            sines.append(sine[inner, inner])
            cosines.append(cosine[inner, inner])
            weights.append(weight[inner, inner])
        if use_skeleton:
            skeletons.append(skeleton[inner, inner])
    batch = np.stack(images)
    if batch.ndim == 3:
        batch = batch[:, None]
    # mean/std are scalars for grey input and shape (C, 1, 1) for LIOT, so the
    # same expression normalises both.
    batch = ((batch - mean) / std).astype(np.float32)
    out = (torch.from_numpy(batch).to(DEVICE),
           torch.from_numpy(np.stack(labels))[:, None].to(DEVICE),
           torch.from_numpy(np.stack(dists))[:, None].to(DEVICE))
    # Three values unless an extra target was ASKED for. Six call sites in
    # this repo unpack exactly three, and a fourth slot appearing under them
    # would break every one of them at import time -- for targets five of the
    # six have no use for.
    #
    # When there IS a fourth it is a DICT, not a bare tensor. With two
    # optional planes a positional fourth slot would mean the direction field
    # for one caller and the skeleton for another, and a caller that asked
    # for the wrong one would get a correctly shaped tensor of the wrong
    # quantity -- silent, and exactly the class of bug this repo keeps paying
    # for. test_direction.py pins all three shapes.
    if not (use_direction or use_skeleton):
        return out
    extras = {}
    if use_direction:
        # (sin 2theta, cos 2theta, coherence) in one tensor, so the loss gets
        # its target and its weight together and cannot be handed one alone.
        field = np.stack([np.stack(sines), np.stack(cosines)], axis=1)
        extras["field"] = torch.from_numpy(np.concatenate(
            [field, np.stack(weights)[:, None]], axis=1)).to(DEVICE)
    if use_skeleton:
        extras["skel"] = torch.from_numpy(
            np.stack(skeletons))[:, None].to(DEVICE)
    return out + (extras,)


@torch.no_grad()
def predict_full(model: nn.Module, image: np.ndarray, mean, std) -> np.ndarray:
    """Whole-image inference; the frame is padded up to the net's stride.

    565 pads to 568 for a 3-level net and to 576 for a 5-level one. The stride
    is read off the model for the same reason the channel count is: every
    analysis script calls this with a checkpoint and an image and nothing else.

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
    stride = 2 ** (getattr(model, "depth", 3) - 1)
    pad_h, pad_w = (-height) % stride, (-width) % stride
    tensor = torch.from_numpy(((encoded - mean) / std).astype(np.float32))
    tensor = F.pad(tensor[None], (0, pad_w, 0, pad_h), mode="reflect")
    prob = torch.sigmoid(model(tensor.to(DEVICE)))[0, 0].cpu().numpy()
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

def standing_best(out_dir: Path) -> float:
    """The best validation Dice already on disk for this run, else -1.0.

    Read from best.pt itself rather than carried along in ckpt.pt. The
    checkpoint is written BEFORE the epoch's validation runs, so its idea of
    the best is always one validation stale: resume from it and a later,
    WORSE epoch can clear the bar and overwrite a better best.pt that is
    sitting right there on disk. CLAUDE.md's rule covers this exactly -- gate
    on the artifact, not on a bookkeeping value that can drift from it.

    Found by test_best_checkpoint.py on the first run of the mechanism, not
    by reading it.
    """
    path = out_dir / "best.pt"
    return float(load_checkpoint(path)["dice"]) if path.exists() else -1.0


def rerun_path(out_dir: Path, name: str) -> Path:
    """Where this run writes `name`, moved aside if a finished one is there.

    Checkpoints are gitignored and the CSVs are not, so a fresh clone of this
    repo holds 54 runs' worth of published measurements and zero weights.
    Recovering the weights means retraining runs whose log.csv is already
    complete -- and train_one APPENDS, which would leave one file holding ten
    rows from the laptop and ten from this box with nothing to tell them
    apart, silently rewriting a published result.

    A half-finished log is a different case and must still be appended to:
    that is a resume, not a rerun, and it is the reason this checks the last
    epoch rather than mere existence.

    The search walks log.csv -> log_rerun.csv -> log_rerun2.csv rather than
    stopping at the first rerun name. A_dice_s0 already had both a published
    laptop log and a finished GPU rerun beside it when a third training was
    queued; returning log_rerun.csv there would have appended twenty rows from
    two different runs into one file -- the very failure this function exists
    to prevent, one level up from where it was being prevented.

    Precedent is train_stare.py:126, which writes scores_rerun.csv beside
    scores.csv so stage 0's published numbers stay auditable against the
    repeat. Being able to diff the two IS the finding; overwriting is not.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    for index in range(50):
        tag = "" if index == 0 else f"_rerun{index if index > 1 else ''}"
        candidate = out_dir / f"{stem}{tag}{suffix}"
        if not candidate.exists():
            return candidate
        with candidate.open() as handle:
            rows = list(csv.DictReader(handle))
        if not (rows and int(rows[-1]["epoch"]) >= EPOCHS):
            return candidate          # unfinished: this is a resume
    raise RuntimeError(f"{out_dir}/{name}: 50 finished reruns is not a rerun")


def train_one(run_name: str, train, val, mean: float, std: float) -> None:
    config_name, seed = run_name.rsplit("_s", 1)
    _, extra = CONFIGS[config_name]  # build_model owns the architecture half
    augments = AUGMENTS.get(config_name, ())
    use_liot = uses_liot(config_name)
    use_direction = uses_direction(config_name)
    use_skeleton = uses_centreline_weight(config_name)
    skel_weight = centreline_weight(config_name) if use_skeleton else 0.0
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
    # PRE-REGISTERED 2026-08-26, before the first run judged by it.
    #
    # best.pt is the epoch with the highest whole-image validation Dice. It
    # exists because at 31M parameters the fixed 100-epoch protocol stops
    # being neutral: A_dice_w64_d5_s0 drove its training loss from 0.4164 to
    # 0.2386 while validation Dice fell 0.8202 -> 0.8014 and Betti-0 error
    # rose 59 -> 142, i.e. the baseline arm at 31M is scored after it has
    # already overfitted, while the augmented arm has not. Reporting only
    # epoch 100 measures that as an augmentation advantage; reporting only
    # the best epoch throws away the fact that it happened. Both get reported,
    # from one training run.
    #
    # The rule is Dice, not a topology metric, and it is fixed HERE rather
    # than chosen later at scoring time -- picking the selection metric after
    # seeing which one flatters an arm is precisely the post-hoc threshold
    # this repo's pre-registration rule exists to prevent. Every validated
    # epoch's full metric row stays in log.csv, so the choice is auditable,
    # but a different rule needs a retrain: only one best.pt is kept.
    best_path = out_dir / "best.pt"

    # A directory carries the protocol it was trained under. Resuming a
    # legacy ckpt.pt with heldout data would produce one run fitted on two
    # different image sets and named as if it were one -- silently, because
    # nothing downstream reads anything but the weights.
    stamp = out_dir / "protocol.txt"
    if stamp.exists() and stamp.read_text().strip() != PROTOCOL:
        raise SystemExit(
            f"{out_dir} was trained under {stamp.read_text().strip()!r}, "
            f"not {PROTOCOL!r}; use a different --results root")
    stamp.write_text(PROTOCOL + "\n")

    if final_path.exists():
        print(f"[{run_name}] already finished, skipping", flush=True)
        return

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = build_model(config_name)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    best_dice = -1.0

    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        rng = state["rng"]
        start_epoch = state["epoch"]
        best_dice = standing_best(out_dir)
        print(f"[{run_name}] resuming from epoch {start_epoch} "
              f"(best dice so far {best_dice:.4f})", flush=True)

    log_path = rerun_path(out_dir, "log.csv")
    # Decided HERE, with log_path, and not at the end of training. Asking
    # rerun_path again after the last epoch asks it about a log this very run
    # has just completed, so every first-time run classified itself as a
    # rerun and wrote val_final_rerun.csv while val_final.csv -- the file
    # summarize.py:24 opens by name -- was never created at all.
    # Derived from the log's own name so the two files always carry the same
    # tag: log_rerun2.csv pairs with val_final_rerun2.csv, never with a
    # val_final_rerun.csv left behind by the previous attempt.
    val_final_path = out_dir / f"val_final{log_path.stem[len('log'):]}.csv"
    if log_path.name != "log.csv":
        print(f"[{run_name}] log.csv is already complete; this rerun writes "
              f"{log_path.name}", flush=True)
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
            optimiser.zero_grad()
            if use_direction or use_skeleton:
                images, labels, dists, extras = sample_batch(
                    train, rng, mean, std, augments, inpainted, use_liot,
                    use_direction, use_skeleton)
                if use_direction:
                    logits, predicted = model.forward_direction(images)
                else:
                    logits = model(images)
                loss = compute_loss(logits, labels, dists, extra, images)
                if use_direction:
                    loss = loss + DIRECTION_WEIGHT * direction_loss(
                        predicted, extras["field"])
                if use_skeleton:
                    # Replaces nothing: it is added on top of the arm's own
                    # loss, so _clw differs from its namesake in exactly this
                    # term and the comparison isolates it.
                    loss = loss + centreline_loss(logits, labels,
                                                  extras["skel"], skel_weight)
            else:
                images, labels, dists = sample_batch(
                    train, rng, mean, std, augments, inpainted, use_liot)
                loss = compute_loss(model(images), labels, dists, extra,
                                    images)
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
            if KEEP_EPOCHS:
                # 117k weights are 0.5 MB, so ten per run is 5 MB; at 31M
                # they are 124 MB and the caller has to have checked the disk.
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "dice": scores["dice"],
                            "betti0_err": scores["betti0_err"],
                            "cldice": scores["cldice"]},
                           out_dir / f"epoch{epoch + 1:03d}.pt")
            if scores["dice"] > best_dice:
                best_dice = scores["dice"]
                # Written the moment it exists, not at the end: CLAUDE.md's
                # rule is to save the expensive artifact as soon as it is
                # there, and this one cannot be reconstructed afterwards --
                # ckpt.pt rolls forward and final.pt is epoch 100.
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "dice": best_dice}, best_path)
            print(f"[{run_name}] epoch {epoch + 1:3d} loss {running:.4f} "
                  f"dice {scores['dice']:.4f} clDice {scores['cldice']:.4f} "
                  f"b0err {scores['betti0_err']:.1f} "
                  f"95HD {scores['hd95']:.2f} ({minutes:.0f} min)", flush=True)
            if last:
                with val_final_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

    torch.save({"model": model.state_dict(), "epoch": EPOCHS}, final_path)
    print(f"[{run_name}] done in {(time.time() - started) / 60:.0f} min; "
          f"best dice {best_dice:.4f}", flush=True)


def main() -> None:
    global RESULTS, KEEP_EPOCHS, PROTOCOL
    argv = list(sys.argv[1:])
    if "--results" in argv:
        # A sweep must not overwrite the published runs: retraining into the
        # same directories would replace the very final.pt and best.pt that
        # stratify.csv and erl.csv were just computed from, and those CSVs
        # would stop being reproducible from what is on disk.
        index = argv.index("--results")
        RESULTS = Path(argv[index + 1])
        RESULTS.mkdir(parents=True, exist_ok=True)
        del argv[index:index + 2]
    if "--keep-epochs" in argv:
        KEEP_EPOCHS = True
        argv.remove("--keep-epochs")
    print(f"results -> {RESULTS}"
          f"{', keeping every validated epoch' if KEEP_EPOCHS else ''}",
          flush=True)

    if "--protocol" in argv:
        index = argv.index("--protocol")
        PROTOCOL = argv[index + 1]
        del argv[index:index + 2]
        if PROTOCOL not in PROTOCOL_SPLITS:
            raise SystemExit(f"--protocol must be one of "
                             f"{sorted(PROTOCOL_SPLITS)}, got {PROTOCOL!r}")

    if "--dataset" in argv:
        index = argv.index("--dataset")
        name = argv[index + 1]
        del argv[index:index + 2]
        import cross_dataset
        train_items, test_items = cross_dataset.loader_for(name)()
        if PROTOCOL == "heldout":
            # Same defect as DRIVE's, one level up: this branch was handing
            # the TEST list in as the validation set, so best.pt was chosen
            # on the images the transfer numbers are reported from. The
            # split rule is cross_dataset's, shared with DRIVE, so the two
            # cannot drift apart.
            train_items, val_items = cross_dataset.fit_dev(train_items)
        else:
            val_items = test_items
        print(f"dataset {name} ({PROTOCOL}): {len(train_items)} fit / "
              f"{len(val_items)} select / {len(test_items)} test, "
              f"median width "
              f"{cross_dataset.median_width(test_items):.2f} px", flush=True)
        train = stack_split("train", train_items)
        val = stack_split("val", val_items)
    else:
        fit_split, dev_split = PROTOCOL_SPLITS[PROTOCOL]
        train, val = stack_split(fit_split), stack_split(dev_split)
        print(f"protocol {PROTOCOL}: fit on {fit_split} "
              f"({len(train['images'])} images), select on {dev_split} "
              f"({len(val['images'])} images)", flush=True)
    inside = train["images"][train["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())
    print(f"train norm mean {mean:.4f} std {std:.4f}", flush=True)

    wanted = argv
    run_names = [f"{name}_s{seed}" for seed in (0, 1, 2) for name in CONFIGS]
    for run_name in (wanted or run_names):
        train_one(run_name, train, val, mean, std)


if __name__ == "__main__":
    main()
