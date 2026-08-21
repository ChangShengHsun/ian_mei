"""Does the contrast gate actually redistribute topology pressure?

Same shape as test_cbdice.py: assert the mechanism, not the output. The claim
F_gated makes is spatial -- an identical break should cost more where the
structure is dim than where it is clear -- and the claim G_focal is there to
falsify is that any per-pixel weight would do. Both are checkable on two bars.

  python exp/test_gated.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

BRIGHT_ROW, DIM_ROW, BAR = 12, 34, 5
GAP = slice(20, 28)


def two_bars() -> tuple[torch.Tensor, torch.Tensor]:
    """A 48 px patch with one clearly darker bar and one barely darker bar.

    Same width, same length, same GT: the only difference between them is how
    far the intensity sits below the background.
    """
    image = torch.ones(1, 1, 48, 48)
    target = torch.zeros(1, 1, 48, 48)
    for row, depth in ((BRIGHT_ROW, 0.8), (DIM_ROW, 0.2)):
        image[:, :, row:row + BAR, :] = 1.0 - depth
        target[:, :, row:row + BAR, :] = 1.0
    return image, target


def with_gap(target: torch.Tensor, row: int) -> torch.Tensor:
    """A near-perfect prediction broken at one place on one bar."""
    prob = target.clone() * 0.9 + 0.05
    prob[:, :, row:row + BAR, GAP] = 0.05
    return prob


def main() -> None:
    image, target = two_bars()

    weight = train.contrast_weight(image)
    on_bright = weight[:, :, BRIGHT_ROW:BRIGHT_ROW + BAR, :].mean().item()
    on_dim = weight[:, :, DIM_ROW:DIM_ROW + BAR, :].mean().item()
    assert 1.0 <= on_bright < on_dim <= 1.0 + train.GATE_GAMMA, (on_bright, on_dim)
    print(f"contrast weight: bright bar {on_bright:.3f} < dim bar {on_dim:.3f}")

    # Regression: the two bars above cover 21% of the patch, but vessels are
    # ~9% of a DRIVE image, and the first version of the gate normalised by
    # the p90 of the top-hat. At real sparsity p90 sits BELOW the vessel
    # distribution, every vessel pixel clamps, and the gate goes flat -- which
    # the fat-bar case cannot see. Four 1 px bars at graded depth catch it:
    # under the broken normaliser the top three all read exactly 1.000.
    sparse = torch.ones(1, 1, 48, 48)
    depths = (0.8, 0.6, 0.4, 0.2)
    for index, depth in enumerate(depths):
        sparse[:, :, 8 + index * 8, :] = 1.0 - depth
    sparse_weight = train.contrast_weight(sparse)
    graded = [sparse_weight[:, :, 8 + i * 8, :].mean().item()
              for i in range(len(depths))]
    assert all(a < b for a, b in zip(graded, graded[1:])), graded
    print("gate stays graded at 4% foreground: " +
          " < ".join(f"{value:.3f}" for value in graded))

    broken_dim = with_gap(target, DIM_ROW)
    broken_bright = with_gap(target, BRIGHT_ROW)

    # The two breaks are geometrically identical, so an unweighted clDice has
    # no way to tell them apart. Anything above the tolerance would mean the
    # patch is not symmetric and the rest of the test proves nothing.
    plain_dim = train.soft_cl_dice(broken_dim, target).item()
    plain_bright = train.soft_cl_dice(broken_bright, target).item()
    assert abs(plain_dim - plain_bright) < 1e-6, (plain_dim, plain_bright)
    print(f"clDice charges both breaks the same: {plain_dim:.5f}")

    gated_dim = train.weighted_cl_dice(broken_dim, target, weight).item()
    gated_bright = train.weighted_cl_dice(broken_bright, target, weight).item()
    assert gated_dim > gated_bright, (gated_dim, gated_bright)
    # Redistribution, not just amplification: the dim break gets dearer AND
    # the bright one gets cheaper. The second half is what E2 asks for, since
    # clDice's measured cost was -0.0070 Dice in the clearest quartile.
    assert gated_dim > plain_dim > gated_bright, (gated_dim, gated_bright)
    print(f"gated moves the dim break to {gated_dim:.5f} "
          f"({gated_dim - plain_dim:+.5f}) and the bright one to "
          f"{gated_bright:.5f} ({gated_bright - plain_bright:+.5f})")

    # G_focal reads the prediction, and both predictions here are equally
    # confident, so its weight map cannot separate the two breaks. That is the
    # failure E1' predicts, made mechanical: hesitation is a property of the
    # model, and the model is equally sure about a dim vessel it has already
    # dropped as about a clear one.
    focal_dim = train.weighted_cl_dice(
        broken_dim, target, train.confidence_weight(broken_dim)).item()
    focal_bright = train.weighted_cl_dice(
        broken_bright, target, train.confidence_weight(broken_bright)).item()
    assert abs(focal_dim - focal_bright) < 1e-6, (focal_dim, focal_bright)
    print(f"focal charges both breaks the same: {focal_dim:.5f}")

    # The weight must not become a second gradient path: G could otherwise cut
    # its loss by growing confident instead of by being right.
    prob = torch.full((1, 1, 48, 48), 0.4, requires_grad=True)
    assert not train.confidence_weight(torch.sigmoid(prob)).requires_grad
    print("confidence weight is detached: no gradient through the weight")

    # A weight of 1 must leave the CVPR 2021 loss exactly where it was, or
    # every earlier B_cldice number stops being comparable to F and G.
    assert train.weighted_cl_dice(broken_dim, target, 1.0).item() == plain_dim
    print("weight 1.0 reproduces clDice exactly")

    for name in ("F_gated", "G_focal"):
        loss = train.compute_loss(
            torch.logit(broken_dim.clamp(1e-4, 1 - 1e-4)), target,
            torch.zeros_like(target), train.CONFIGS[name][1], image)
        assert torch.isfinite(loss), name
        print(f"{name} end to end: loss {loss.item():.5f}")


if __name__ == "__main__":
    main()
