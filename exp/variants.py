"""Tier-1 inference-time methods: SWA, TTA and seed ensembling.

WRITTEN AND SELFTESTED 2026-08-27, BEFORE ANY OF THE THREE WAS SCORED. The
predictions each one is judged against are in summarize_variants.py, written
the same day; nothing here may be tuned after seeing a number.

None of the three is novel. They are the standard inference-time methods this
series never ran, and their job is to say where the line a NEW method has to
beat actually sits -- stage-report/plan_next.md section 0. If TTA alone
recovers most of what augmentation buys, then E14's headline is partly a
statement about test-time invariance and has to be written that way.

  A1 SWA         average the weights along the tail of the trajectory instead
                 of picking one epoch off a validation metric.
  A3 TTA         average the predictions over the eight symmetries of the
                 square, which is the group H_aug and K_focal_aug already
                 train under and A_dice and G_focal do not.
  A2 ensemble    average the predictions of a config's six seeds.

THE ONE THING THAT MUST NOT BE FORGOTTEN. Every conv_block here carries a
BatchNorm, and BatchNorm's running_mean/running_var are NOT parameters of a
function being averaged -- they are statistics of activations produced by one
particular weight vector. Average the weights of five epochs and the averaged
network's activations have a distribution none of the five had, so the
inherited statistics are wrong and the model predicts noise. The SWA paper
requires a pass over the training data to recompute them, and update_bn below
is that pass. Skipping it does not raise; it produces a bad number that reads
as "SWA does not help on this problem".

  python exp/variants.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

# PRE-REGISTERED. The last five validation points of the 100-epoch schedule.
# Chosen so it coincides with NO selection rule -- an average over a fixed
# tail is not a disguised pick, which is the whole claim SWA makes here. E13b
# R.1 measured Dice peaking at epoch 10 for K_focal_aug and 65 for H_aug, so
# any window chosen to contain "the good epochs" would mean different things
# for different arms.
SWA_EPOCHS = (60, 70, 80, 90, 100)

# Batches of 48 px crops used to recompute BatchNorm statistics. 200 x 32 =
# 6400 patches, against the 10,000 an epoch of training draws, so the
# statistics are estimated on the same order of data they were trained with.
BN_BATCHES = 200


def average_weights(states: list[dict]) -> dict:
    """Elementwise mean of several state dicts.

    Integer buffers -- num_batches_tracked -- are carried through rather than
    averaged, because they are counts and their value is about to be reset by
    update_bn anyway.
    """
    if not states:
        raise ValueError("nothing to average")
    keys = set(states[0])
    for other in states[1:]:
        if set(other) != keys:
            raise ValueError("state dicts do not have the same keys")
    out = {}
    for key, first in states[0].items():
        if not first.is_floating_point():
            out[key] = first.clone()
            continue
        stacked = torch.stack([state[key].float() for state in states])
        out[key] = stacked.mean(0).to(first.dtype)
    return out


@torch.no_grad()
def update_bn(model, data: dict, mean, std, augments: tuple,
              batches: int = BN_BATCHES, seed: int = 0) -> int:
    """Recompute every BatchNorm's running statistics for these weights.

    Momentum is set to None so each layer accumulates a cumulative average
    over the whole pass rather than an exponential one that mostly remembers
    the last few batches. The counters are reset first, or the fresh
    statistics would be blended into the ones inherited from the averaged
    checkpoints -- which are the very numbers that need discarding.

    Returns the number of BatchNorm layers it touched, so a caller can assert
    it was not zero. A model with no BatchNorm would make this a no-op and
    SWA would look like it needed no recalibration at all.
    """
    layers = [module for module in model.modules()
              if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not layers:
        return 0
    saved = []
    for layer in layers:
        saved.append(layer.momentum)
        layer.reset_running_stats()
        layer.momentum = None
    was_training = model.training
    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(batches):
        images, _, _ = train.sample_batch(data, rng, mean, std, augments)
        model(images)
    if not was_training:
        model.eval()
    for layer, momentum in zip(layers, saved):
        layer.momentum = momentum
    return len(layers)


def swa_model(config: str, run_dir: Path, data: dict, mean, std,
              epochs: tuple = SWA_EPOCHS):
    """The weight-averaged model, with its BatchNorm statistics recomputed."""
    paths = [run_dir / f"epoch{epoch:03d}.pt" for epoch in epochs]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir.name} is missing {missing}")
    states = [train.load_checkpoint(path)["model"] for path in paths]
    model = train.build_model(config)
    model.load_state_dict(average_weights(states))
    touched = update_bn(model, data, mean, std,
                        train.AUGMENTS.get(config, ()))
    if touched == 0:
        raise RuntimeError(
            f"{config} has no BatchNorm layers; averaging weights without "
            f"recalibration was silently fine here, which it is not in "
            f"general -- check the architecture before trusting this")
    model.eval()
    return model


@torch.no_grad()
def predict_tta(model, image: np.ndarray, mean, std) -> np.ndarray:
    """Mean probability over the eight symmetries of the square.

    The image is moved, predicted, and the PREDICTION moved back, so all
    eight agree pixel for pixel before they are averaged. Probabilities are a
    scalar field, so only the pixels move -- unlike D1's tangent field, whose
    values move too.
    """
    total = np.zeros(image.shape, dtype=np.float64)
    for turns in range(4):
        for flip in (False, True):
            moved = np.rot90(image, turns)
            if flip:
                moved = np.fliplr(moved)
            prob = train.predict_full(model, np.ascontiguousarray(moved),
                                      mean, std)
            if flip:
                prob = np.fliplr(prob)
            total += np.rot90(prob, -turns)
    return (total / 8.0).astype(np.float32)


def selftest() -> None:
    # average_weights is the mean, and it refuses mismatched dicts.
    a = {"w": torch.ones(3), "n": torch.tensor(4)}
    b = {"w": torch.full((3,), 3.0), "n": torch.tensor(9)}
    got = average_weights([a, b])
    assert torch.allclose(got["w"], torch.full((3,), 2.0)), got["w"]
    assert got["n"].item() == 4, got["n"]
    print("average_weights: floats averaged, integer counters carried through")
    try:
        average_weights([a, {"w": torch.ones(3)}])
    except ValueError as error:
        print(f"  and mismatched state dicts are refused: {error}")
    else:
        raise AssertionError("mismatched dicts must raise")

    # THE BATCHNORM CLAIM, measured rather than asserted. Two models trained
    # on visibly different inputs; the average of their weights inherits
    # statistics belonging to neither, and recomputing them changes the
    # output. If this printed "no change", update_bn would be doing nothing
    # and every SWA number would be quietly wrong.
    size = 96
    yy, xx = np.mgrid[0:size, 0:size]
    label = (np.abs((xx - yy)) <= 1) | (np.abs((xx + yy) - size) <= 1)
    data = {"images": np.stack([(0.3 + 0.4 * label).astype(np.float32)] * 2),
            "labels": np.stack([label.astype(np.float32)] * 2),
            "fovs": np.ones((2, size, size), dtype=bool),
            "dists": np.zeros((2, size, size), dtype=np.float32),
            "names": ["00", "01"]}
    states = []
    for scale in (0.5, 4.0):
        model = train.build_model("A_dice")
        optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
        rng = np.random.default_rng(int(scale * 10))
        for _ in range(30):
            images, labels, dists = train.sample_batch(data, rng, 0.0, 1.0)
            optimiser.zero_grad()
            loss = train.compute_loss(model(images * scale), labels, dists,
                                      None, images)
            loss.backward()
            optimiser.step()
        states.append({k: v.clone() for k, v in model.state_dict().items()})

    averaged = train.build_model("A_dice")
    averaged.load_state_dict(average_weights(states))
    averaged.eval()
    probe = data["images"][0]
    stale = train.predict_full(averaged, probe, 0.0, 1.0)
    touched = update_bn(averaged, data, 0.0, 1.0, (), batches=40)
    averaged.eval()
    fresh = train.predict_full(averaged, probe, 0.0, 1.0)
    shift = float(np.abs(fresh - stale).mean())
    print(f"update_bn recalibrated {touched} BatchNorm layers; mean |change| "
          f"in the averaged model's output {shift:.4f}")
    expected = sum(1 for m in train.build_model("A_dice").modules()
                   if isinstance(m, nn.modules.batchnorm._BatchNorm))
    assert touched == expected, (touched, expected)
    assert shift > 1e-3, (
        f"recalibration changed nothing ({shift:.2e}); either update_bn is a "
        f"no-op or the two models were too similar to make the point")

    # And the counters really were reset rather than blended.
    for module in averaged.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert int(module.num_batches_tracked) == 40, \
                module.num_batches_tracked
    print("  every layer's counter reads 40, so the inherited statistics were "
          "discarded and not averaged into")

    # TTA on a model is exactly the model when the model is equivariant, and
    # an identity "model" is. Checked on a NON-square image, because the
    # rotations change the frame's shape and a wrong inverse would show up as
    # a transpose that a square image hides.
    class Identity(nn.Module):
        depth = 3

        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(1, 1, 1))
            with torch.no_grad():
                self.enc1[0].weight.fill_(1.0)
                self.enc1[0].bias.zero_()

        def forward(self, x):
            return x

    rng = np.random.default_rng(3)
    frame = rng.random((37, 23)).astype(np.float32)
    model = Identity().to(train.DEVICE)
    single = train.predict_full(model, frame, 0.0, 1.0)
    eight = predict_tta(model, frame, 0.0, 1.0)
    assert eight.shape == frame.shape, eight.shape
    gap = float(np.abs(eight - single).max())
    print(f"TTA over a 37x23 frame returns the frame's own shape and agrees "
          f"with a single pass on an equivariant model to {gap:.2e}")
    assert gap < 1e-5, gap

    # The averaging must actually be over eight DIFFERENT passes: a wrong
    # inverse transform would still return the right shape, and would blur.
    class Corner(nn.Module):
        """Fires only in the top-left corner -- maximally not equivariant."""
        depth = 3

        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(1, 1, 1))

        def forward(self, x):
            out = torch.full_like(x, -10.0)
            out[..., :5, :5] = 10.0
            return out

    corner = Corner().to(train.DEVICE)
    spread = predict_tta(corner, frame, 0.0, 1.0)
    lit = (spread > 0.05).sum()
    print(f"a corner-only model under TTA lights {lit} pixels in "
          f"{spread.shape} -- the eight passes land in different places")
    assert lit > 25, lit
    print("all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
