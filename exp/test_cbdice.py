"""Does the ported cbDice actually balance thin against thick vessels?

That is the entire claim of the loss and the one thing a port can get backwards
while still running and producing plausible numbers. The scene is two bars of
the same length, one 3 px wide and one 11 px wide.

First attempt at this test asserted "a break in the thin bar must cost more
than a break in the thick bar" and it failed. The assertion was wrong, not the
port: the paper's complaint is that clDice combined with Dice ends up
"favoring larger vessels", and the fix is to make the two contribute *equally*,
not to make thin vessels dominate. cbDice also keeps a boundary term, so
destroying 44 px of thick bar legitimately costs more than 12 px of thin bar.
The test below therefore measures the balancing mechanism directly.

  python exp/test_cbdice.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

SIZE = 64
THIN_ROWS, THICK_ROWS = slice(14, 17), slice(40, 51)   # 3 px and 11 px wide


def scene() -> torch.Tensor:
    canvas = np.zeros((SIZE, SIZE), dtype=np.float32)
    canvas[THIN_ROWS, 4:60] = 1.0
    canvas[THICK_ROWS, 4:60] = 1.0
    return torch.from_numpy(canvas)[None, None]


def split_mass(tensor: torch.Tensor) -> tuple[float, float]:
    """Total weight sitting on the thin bar vs on the thick bar."""
    return (float(tensor[0, 0, THIN_ROWS].sum()),
            float(tensor[0, 0, THICK_ROWS].sum()))


def main() -> None:
    truth = scene()
    skeleton = train.soft_skeleton(truth)
    distance_weight, _, inverse_weight = train._cb_weights(
        truth, skeleton, truth, skeleton)

    # The imbalance cbDice is meant to fix: the distance-transform term alone
    # puts most of its mass on the thick bar.
    thin_raw, thick_raw = split_mass(distance_weight)
    raw_ratio = thin_raw / thick_raw

    # What the loss actually sums over the centreline: skeleton x inverse radius.
    thin_weighted, thick_weighted = split_mass(inverse_weight * distance_weight)
    balanced_ratio = thin_weighted / thick_weighted

    print(f"distance term only     thin {thin_raw:9.1f}  thick {thick_raw:9.1f}"
          f"  ratio {raw_ratio:.3f}")
    print(f"after inverse radius   thin {thin_weighted:9.1f}  "
          f"thick {thick_weighted:9.1f}  ratio {balanced_ratio:.3f}")

    assert raw_ratio < 0.2, (
        "the scene must actually contain the imbalance being corrected; got "
        f"ratio {raw_ratio:.3f}")
    assert balanced_ratio > 4 * raw_ratio, (
        "the inverse-radius weight must move the thin bar's share up by a "
        f"large factor; got {raw_ratio:.3f} -> {balanced_ratio:.3f}")

    perfect = float(train.soft_cb_dice(truth, truth))
    print(f"\ncbDice perfect prediction {perfect:.5f}")
    assert perfect < 0.01, "a perfect prediction must cost nothing"

    empty = float(train.soft_cb_dice(torch.zeros_like(truth), truth))
    print(f"cbDice empty prediction   {empty:.5f}")
    assert empty > 0.9, "predicting nothing must cost nearly everything"

    # The loss is useless if no gradient reaches the network. The prediction
    # has to be imperfect for this to mean anything: at a perfect prediction
    # the loss is exactly 0, which is its minimum, so the gradient is legitimately
    # zero there and the test would pass for the wrong reason.
    broken = scene()
    broken[0, 0, THIN_ROWS, 30:34] = 0.0
    probability = (0.4 + 0.4 * broken).clone().requires_grad_(True)
    cost = train.soft_cb_dice(probability, truth)
    cost.backward()
    print(f"cbDice imperfect          {float(cost.detach()):.5f}")
    assert float(cost.detach()) > 0.01, "the imperfect prediction must cost something"
    assert probability.grad is not None and probability.grad.abs().sum() > 0, \
        "no gradient flowed back through cbDice"

    # Documented dead zone, not a bug: every cbDice weight is built from the
    # prediction thresholded at 0.5, so a model that is below 0.5 everywhere
    # gets exactly zero gradient from this term. The reference implementation
    # behaves the same way. It is safe here only because cbDice is always added
    # on top of BCE + soft Dice, which do drive the early epochs.
    flat = torch.full((1, 1, SIZE, SIZE), 0.4, requires_grad=True)
    train.soft_cb_dice(flat, truth).backward()
    assert flat.grad.abs().sum() == 0, (
        "the all-below-threshold dead zone changed; re-read the port before "
        "trusting any cbDice run")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
