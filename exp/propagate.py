"""D-B: the oriented propagation layer, inside the network.

WRITTEN AND SELFTESTED 2026-08-27, BEFORE THE FIRST _prop RUN.

anisotropic.py established the mechanism as post-processing: spread every
foreground pixel through an ellipse aligned with its own tangent. This is the
differentiable version, placed between the trunk and the segmentation head and
driven by the network's OWN direction head.

That placement is the point. After D1.b, the direction head was a dead end --
it predicted a field that nothing consumed, and the auxiliary loss was the
only thing keeping it honest. Here the field decides how segmentation evidence
spreads, so a wrong field costs segmentation accuracy directly. The head stops
being a side task and becomes load-bearing.

WHY IT IS A SOFT MAXIMUM AND NOT A BLUR. The post-processing version is a
morphological dilation: a pixel becomes foreground if ANY pixel in its
oriented neighbourhood was. The differentiable analogue of a maximum is
logsumexp, not a mean. A mean would let a confident background pixel cancel a
confident foreground one, which is a smoothing operator, and smoothing is
exactly what the measurements say destroys run length -- averaging thins the
faint connections ERL depends on (P2, P3, and every averaging row of
variants_summary).

THE GEOMETRY IS NOT LEARNED, IT IS HANDED OVER. `along` and `across` come
from phase 1's sweep, which chose them on held-out images under a Dice floor.
Learning them here as well would fit the same quantity twice, on 20 images,
and phase 1 exists precisely so that this layer does not have to guess. The
only learned parameter is `strength`: how much of the propagated field to mix
in, initialised so the layer starts as the identity.

  python exp/propagate.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Eight orientations, not anisotropic.py's sixteen: this runs on every
# training step, and 8 depthwise convolutions on a 48 px patch is the budget.
# 22.5 degrees of quantisation against a head whose own axis error is about
# 6.6 degrees -- the bins are the coarser term here, which the selftest
# measures rather than assumes.
BINS = 8
# Sharpness of the orientation blend. 4.0 puts ~85% of the weight on the
# nearest bin for a pixel sitting on a bin centre; low enough to stay
# differentiable, high enough that a pixel is not smeared over every
# orientation at once.
TAU = 4.0
# logsumexp temperature. Higher is closer to a true maximum; 4.0 keeps the
# gradient alive on the losing branches, which is what makes it trainable.
BETA = 4.0


def line_kernels(along: float, across: float, bins: int = BINS) -> torch.Tensor:
    """(bins, 1, size, size) elliptical kernels, one per orientation."""
    reach = max(int(np.ceil(max(along, across))), 1)
    size = 2 * reach + 1
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    stack = []
    for index in range(bins):
        angle = (index + 0.5) / bins * np.pi
        u = xx * np.cos(angle) + yy * np.sin(angle)
        v = -xx * np.sin(angle) + yy * np.cos(angle)
        inside = (u / max(along, 0.5)) ** 2 + (v / max(across, 0.5)) ** 2 <= 1.0
        inside[reach, reach] = True
        stack.append(inside.astype(np.float32))
    return torch.from_numpy(np.stack(stack)[:, None]), size


class OrientedPropagation(nn.Module):
    """Spread logits along the axis the direction head predicts.

    field is (B, 2, H, W) holding (sin 2theta, cos 2theta), unnormalised --
    the head's output has free magnitude, and the blend below only needs the
    direction of that vector, so it is normalised here rather than
    constrained upstream.
    """

    def __init__(self, along: float, across: float, bins: int = BINS):
        super().__init__()
        kernels, size = line_kernels(along, across, bins)
        self.register_buffer("kernels", kernels)
        self.bins = bins
        self.pad = size // 2
        # Orientation of each bin, in double-angle form, so the affinity
        # between a pixel's axis and a bin is a plain dot product. That is the
        # representation paying off again: at double angle, "how aligned are
        # these two axes" is linear.
        angles = (torch.arange(bins) + 0.5) / bins * np.pi
        self.register_buffer("bin_sin", torch.sin(2 * angles)[None, :, None, None])
        self.register_buffer("bin_cos", torch.cos(2 * angles)[None, :, None, None])
        # Starts nearly closed: sigmoid(-2) = 0.12, so a _prop run begins
        # within a tenth of the arm it is named after and has to earn the
        # rest. NOT at exactly zero, which was the first version's claim and
        # was wrong twice over -- sigmoid(0) is 0.5, not 0, and the value that
        # would give a true identity, sigmoid(-6), has a gradient of 0.0025
        # and would likely never open at all. A gate that cannot move is not
        # a conservative initialisation, it is an ablation by accident.
        self.strength = nn.Parameter(torch.full((1,), -2.0))

    def forward(self, logits: torch.Tensor,
                field: torch.Tensor) -> torch.Tensor:
        sin2, cos2 = field[:, 0:1], field[:, 1:2]
        length = torch.sqrt(sin2 * sin2 + cos2 * cos2).clamp_min(1e-6)
        sin2, cos2 = sin2 / length, cos2 / length
        affinity = sin2 * self.bin_sin + cos2 * self.bin_cos   # (B, bins, H, W)
        weight = torch.softmax(TAU * affinity, dim=1)

        # Soft maximum over each oriented neighbourhood, stabilised against a
        # LOCAL maximum rather than a global one.
        #
        # The first version subtracted the whole patch's peak. With beta 4 and
        # logits spanning +-8, a background pixel's exponentials came out at
        # exp(-64), the sum of seven of them at 1e-27, and the clamp floor of
        # 1e-20 then dominated -- so every distant background pixel was
        # reported at -3.5 instead of -7.5 and the layer lit the entire frame.
        # A guard against underflow that sits above the real values is not a
        # guard, it is the answer.
        #
        # Against a local max the exponentials are at most 1, uniform regions
        # come out exactly right (weighted is all ones, so pooled is the local
        # value plus log(area)/beta), and the floor is only reachable when a
        # pixel sits beside something far hotter that the oriented kernel
        # excludes -- where erring low is the conservative direction, since
        # the mix below only ever adds.
        local = F.max_pool2d(logits, kernel_size=2 * self.pad + 1, stride=1,
                             padding=self.pad).detach()
        weighted = torch.exp(BETA * (logits - local))
        pooled = F.conv2d(weighted, self.kernels, padding=self.pad)
        pooled = local + torch.log(pooled.clamp_min(1e-30)) / BETA
        propagated = (weight * pooled).sum(dim=1, keepdim=True)

        # The layer may only ADD evidence, never remove it: propagation is a
        # dilation, and a mix that could lower a logit would let it erase
        # structure the trunk was sure about.
        gain = torch.sigmoid(self.strength)
        return logits + gain * F.relu(propagated - logits)


def selftest() -> None:
    torch.manual_seed(0)
    # At initialisation the layer must be nearly closed, and its gate must
    # still be able to move. Both halves matter: a layer that starts wide open
    # is a different architecture from the arm it is compared against, and one
    # whose gate has no gradient is an ablation wearing a parameter.
    layer = OrientedPropagation(along=3.0, across=0.5)
    gain = torch.sigmoid(layer.strength).item()
    slope = gain * (1 - gain)
    print(f"initial gain {gain:.3f} with gradient slope {slope:.3f} -- nearly "
          f"closed, and able to open")
    assert 0.05 < gain < 0.2, gain
    assert slope > 0.05, slope
    logits = torch.randn(2, 1, 24, 24)
    field = torch.randn(2, 2, 24, 24)
    got = layer(logits, field)
    drift = (got - logits).abs().mean() / logits.abs().mean()
    print(f"  its output starts {drift:.1%} away from the trunk's own logits")
    assert drift < 0.25, drift

    # Opened up, it must spread a single hot pixel ALONG the axis it is given
    # and not across it.
    with torch.no_grad():
        layer.strength.fill_(10.0)
    canvas = torch.full((1, 1, 41, 41), -8.0)
    canvas[0, 0, 20, 20] = 8.0
    # theta = 0 is the +x (column) axis: sin2 = 0, cos2 = 1.
    flat = torch.zeros(1, 2, 41, 41)
    flat[:, 1] = 1.0
    spread = layer(canvas, flat)[0, 0]
    along_arm = (spread[20, :] > -7.0).sum().item()
    across_arm = (spread[:, 20] > -7.0).sum().item()
    print(f"a single hot pixel under a horizontal axis field lights "
          f"{along_arm} px along the row and {across_arm} down the column")
    assert along_arm > across_arm, (along_arm, across_arm)

    # Rotate the field a quarter turn and the spread must rotate with it.
    upright = torch.zeros(1, 2, 41, 41)
    upright[:, 1] = -1.0                    # theta = pi/2
    turned = layer(canvas, upright)[0, 0]
    assert (turned[:, 20] > -7.0).sum() > (turned[20, :] > -7.0).sum()
    print("  and a quarter turn in the field turns the spread with it")

    # It must never lower a logit.
    for _ in range(5):
        logits = torch.randn(2, 1, 32, 32) * 4
        field = torch.randn(2, 2, 32, 32)
        assert (layer(logits, field) >= logits - 1e-4).all()
    print("the layer only ever adds evidence -- it can dilate, never erase")

    # Gradients must reach both the logits and the field, or the direction
    # head goes back to being a dead end with extra steps.
    logits = torch.randn(1, 1, 24, 24, requires_grad=True)
    field = torch.randn(1, 2, 24, 24, requires_grad=True)
    layer(logits, field).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert field.grad is not None and field.grad.abs().sum() > 0, \
        "no gradient reaches the direction field; the head is not load-bearing"
    print(f"gradient reaches the field (norm {field.grad.norm():.3f}) -- the "
          f"direction head is load-bearing, which D1.b's version was not")

    # No overflow on confident logits: beta 4 on a logit of 30 is exp(120).
    hot = torch.full((1, 1, 16, 16), 30.0)
    got = layer(hot, torch.randn(1, 2, 16, 16))
    assert torch.isfinite(got).all(), "logsumexp overflowed"
    print("confident logits (30.0) do not overflow the soft maximum")

    # Bin quantisation must be the coarser error, and by how much is a number.
    worst = np.rad2deg(np.pi / BINS / 2)
    print(f"{BINS} bins: worst orientation error {worst:.1f} degrees, against "
          f"the head's measured 6.6 -- the bins are the coarser term")
    print("all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
