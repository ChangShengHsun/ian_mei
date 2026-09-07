"""The flux head end to end: target, augmentation, loss, and the model.

WRITTEN 2026-09-07 alongside the head. The one test that matters is the third:
a displacement field's VALUES must move under augmentation, and every cheaper
check passes when they do not. np.rot90 preserves magnitude, so a bound on
|displacement| cannot see the bug; only asking whether the field still POINTS
AT THE VESSEL can, and this file proves that by breaking the transform on
purpose and requiring the check to collapse.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flux
import train


def landing_rate(batch) -> tuple[float, int]:
    """Fraction of banded pixels whose displacement lands on the vessel."""
    _, labels, _, extras = batch
    field = extras["flux"].cpu().numpy()
    label = labels[:, 0].cpu().numpy() > 0.5
    hit = total = 0
    for index in range(field.shape[0]):
        for row, col in np.argwhere(field[index, 2] > 0.5):
            landed_y = int(round(row + field[index, 0, row, col]))
            landed_x = int(round(col + field[index, 1, row, col]))
            if (0 <= landed_y < label.shape[1]
                    and 0 <= landed_x < label.shape[2]):
                total += 1
                hit += bool(label[index, landed_y, landed_x])
    return hit / max(total, 1), total


def main() -> None:
    # 1. THE ARCHITECTURE. `_flux` adds a head and nothing else; its namesake
    #    must not grow one, or the comparison stops isolating the head.
    for name in sorted(c for c in train.CONFIGS if c.endswith("_flux")):
        model = train.build_model(name)
        assert model.flux_head is not None, name
        base = name[: -len("_flux")]
        assert train.build_model(base).flux_head is None, base
    print(f"{len([c for c in train.CONFIGS if c.endswith('_flux')])} _flux "
          f"configs build a flux head; their namesakes do not")

    # 2. THE STORED TARGET IS THE CANONICAL ONE. stack_split precomputes it;
    #    if it ever drifts from flux.target the head trains on one definition
    #    and flux_ceiling prices another.
    data = train.stack_split("fit")
    same = total = 0
    for index in range(len(data["images"])):
        (want_y, want_x), band = flux.target(data["skel"][index] > 0.5,
                                             train.FLUX_RADIUS)
        agree = ((data["flux_y"][index] == want_y)
                 & (data["flux_x"][index] == want_x))
        same += int(agree[band].sum())
        total += int(band.sum())
    assert same == total, (same, total)
    print(f"stack_split's target is flux.target exactly: {same}/{total} px")

    # 3. THE VECTOR TRAP, END TO END. Not a bound on the magnitude -- rot90
    #    preserves that, so the bound passes on a field rotated wrongly. Ask
    #    instead whether the field still points at the vessel, and prove the
    #    question bites by breaking the transform.
    mean, std = train.normalisation("A_dice_flux_s0", data)

    def batch(augments):
        return train.sample_batch(data, np.random.default_rng(1), mean, std,
                                  augments, None, False, False, False, True)

    plain, plain_n = landing_rate(batch(()))
    turned, turned_n = landing_rate(batch(("dihedral",)))
    assert plain > 0.99, plain
    assert turned > 0.99, (turned, "the dihedral is not moving the values")

    real = flux.dihedral
    flux.dihedral = lambda turns, flip, delta_y, delta_x: tuple(
        np.ascontiguousarray(np.fliplr(np.rot90(plane, turns)) if flip
                             else np.rot90(plane, turns))
        for plane in (delta_y, delta_x))
    try:
        broken, _ = landing_rate(batch(("dihedral",)))
    finally:
        flux.dihedral = real
    assert broken < turned - 0.2, (broken, turned, "the check does not bite")
    print(f"landings on the vessel: {plain:.4f} plain ({plain_n} px), "
          f"{turned:.4f} rotated ({turned_n} px), {broken:.4f} with the "
          f"transform deliberately broken -- so the check bites")

    # 4. THE LOSS reads only inside the band and is zero on the truth.
    import torch
    field = batch(("dihedral",))[3]["flux"]
    assert float(train.flux_loss(field[:, :2], field)) == 0.0
    assert float(train.flux_loss(torch.zeros_like(field[:, :2]), field)) > 0.5
    outside = field.clone()
    outside[:, :2] = torch.where(outside[:, 2:3] > 0.5, outside[:, :2],
                                 torch.full_like(outside[:, :2], 99.0))
    assert float(train.flux_loss(outside[:, :2], field)) == 0.0, \
        "the loss read a value outside the band"
    print("flux_loss: zero on the truth, positive on a zero field, and blind "
          "to anything outside the band")
    print("all checks passed")


if __name__ == "__main__":
    main()
