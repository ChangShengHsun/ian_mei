"""D1's wiring, checked where it can fail silently.

Four things, each of which would produce a plausible-looking training run and
a wrong result:

  1. sample_batch's arity follows its argument. Six call sites in this repo
     unpack three values; a fourth appearing under them is an immediate
     TypeError, and a THIRD where four were wanted is a silent mis-unpack.
  2. The batch's tangent target is the tangent of the batch's own augmented
     label -- not of the label before augmentation. This is the double-angle
     trap arriving through the pipeline instead of through direction.py.
  3. forward() of a _dir model is byte-identical to the segmentation model it
     would have been, so the twenty analysis scripts that call model(image)
     score it correctly with no edit.
  4. The auxiliary loss is weighted by coherence and confined to the vessel:
     a head that is perfect on vessels and garbage on background pays zero.

  python exp/test_direction.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction
import train


def fake_split(count: int = 3, size: int = 96) -> dict:
    """A split of diagonal bars, so the tangent target is not degenerate."""
    rng = np.random.default_rng(0)
    images, labels = [], []
    for index in range(count):
        label = np.zeros((size, size), dtype=bool)
        yy, xx = np.mgrid[0:size, 0:size]
        for offset in (-24, 0, 24):
            label |= np.abs((xx - yy) - offset) <= 1
        labels.append(label)
        images.append((0.4 + 0.2 * label + 0.01 * rng.standard_normal(
            (size, size))).astype(np.float32))
    from skimage.morphology import skeletonize
    fields = [direction.tangent_field(label) for label in labels]
    return {"images": np.stack(images),
            "labels": np.stack(labels).astype(np.float32),
            "fovs": np.ones((count, size, size), dtype=bool),
            "dists": np.zeros((count, size, size), dtype=np.float32),
            "dir_sin": np.stack([f[0] for f in fields]),
            "dir_cos": np.stack([f[1] for f in fields]),
            "dir_weight": np.stack([f[2] for f in fields]),
            "skel": np.stack([skeletonize(l).astype(np.float32)
                              for l in labels]),
            "names": [f"{i:02d}" for i in range(count)]}


def check_arity(data) -> None:
    rng = np.random.default_rng(1)
    plain = train.sample_batch(data, rng, 0.4, 0.2)
    assert len(plain) == 3, len(plain)
    withdir = train.sample_batch(data, np.random.default_rng(1), 0.4, 0.2,
                                 use_direction=True)
    assert len(withdir) == 4, len(withdir)
    assert set(withdir[3]) == {"field"}, withdir[3].keys()
    withskel = train.sample_batch(data, np.random.default_rng(1), 0.4, 0.2,
                                  use_skeleton=True)
    assert set(withskel[3]) == {"skel"}, withskel[3].keys()
    both = train.sample_batch(data, np.random.default_rng(1), 0.4, 0.2,
                              use_direction=True, use_skeleton=True)
    assert set(both[3]) == {"field", "skel"}, both[3].keys()
    # The extras are a DICT for exactly this reason: two optional planes in
    # one positional slot would hand a caller the wrong quantity in the right
    # shape.
    assert not torch.equal(both[3]["field"][:, :1], both[3]["skel"])
    # Same seed, same first three tensors: asking for the target must not
    # perturb the crops, or a _dir arm would not be comparable to its namesake.
    for left, right in zip(plain, withdir):
        assert torch.equal(left, right), "asking for direction moved the crops"
    field = withdir[3]["field"]
    assert field.shape == (train.BATCH, 3, train.PATCH, train.PATCH), field.shape
    print(f"sample_batch returns 3 tensors by default and 4 when asked; the "
          f"crops are identical either way, target {tuple(field.shape)}")


def check_augmented_target(data) -> None:
    """The target must be the tangent of the AUGMENTED label.

    Precomputing the field and moving only its pixels leaves a field pointing
    across the vessel -- the exact bug direction.dihedral exists to prevent,
    here checked end to end through sample_batch rather than in isolation.
    """
    _, labels, _, extras = train.sample_batch(
        data, np.random.default_rng(4), 0.4, 0.2,
        augments=("dihedral",), use_direction=True)
    field = extras["field"]
    # The reference is recomputed on the CROP, whose four edges the batch's
    # field does not have -- it was cut out of a whole image. A ridge that
    # runs off the edge of a 48 px tile is smoothed against nothing there, so
    # the recomputed tangent near the border is an artefact of the tile and
    # the comparison is made in the interior only.
    margin = 8
    inside = np.zeros((train.PATCH, train.PATCH), dtype=bool)
    inside[margin:-margin, margin:-margin] = True
    worst, checked = 0.0, 0
    for index in range(labels.shape[0]):
        mask = labels[index, 0].cpu().numpy() > 0.5
        want_sin, want_cos, _ = direction.tangent_field(mask)
        got = field[index].cpu().numpy()
        core = mask & inside & (got[2] > 0.8)
        if core.sum() < 20:
            continue
        checked += 1
        worst = max(worst, float(direction.axis_gap(
            got[0], got[1], want_sin, want_cos)[core].max()))
    assert checked >= 5, checked
    assert worst < 0.35, worst
    print(f"the target is the tangent of the augmented label on {checked} "
          f"dihedral-augmented crops (worst axis gap {worst:.3f} of 2.0, "
          f"{margin} px in from the tile edge)")

    # The check has to be able to fail, or it is decoration. Moving the
    # planes and NOT the values -- the bug -- must be caught by it.
    caught = 0.0
    for index in range(labels.shape[0]):
        mask = labels[index, 0].cpu().numpy() > 0.5
        want_sin, want_cos, _ = direction.tangent_field(mask)
        got = field[index].cpu().numpy()
        core = mask & inside & (got[2] > 0.8)
        if core.sum() < 20:
            continue
        # A quarter turn is the whole error, so simulating it is one sign.
        caught = max(caught, float(direction.axis_gap(
            -got[0], -got[1], want_sin, want_cos)[core].max()))
    assert caught > 1.5, caught
    print(f"  a quarter-turn error in the same place scores {caught:.3f}, "
          f"so the threshold above is not vacuous")


def check_forward_unchanged() -> None:
    plain = train.build_model("A_dice")
    withdir = train.build_model("A_dice_dir")
    # Same backbone, same names: a _dir state dict is a segmentation state
    # dict plus two tensors, so the difference is exactly the head.
    extra = set(withdir.state_dict()) - set(plain.state_dict())
    assert extra == {"dir_head.weight", "dir_head.bias"}, extra
    missing = withdir.load_state_dict(plain.state_dict(), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    assert set(missing.missing_keys) == extra, missing.missing_keys

    image = torch.randn(1, 1, 48, 48, device=train.DEVICE)
    plain.eval(), withdir.eval()
    with torch.no_grad():
        assert torch.allclose(plain(image), withdir(image), atol=1e-6)
        logits, field = withdir.forward_direction(image)
        assert torch.allclose(plain(image), logits, atol=1e-6)
    assert field.shape == (1, 2, 48, 48), field.shape
    print("forward() of a _dir model is the segmentation model it would have "
          "been; the head adds "
          f"{sum(p.numel() for p in withdir.dir_head.parameters())} of "
          f"{sum(p.numel() for p in withdir.parameters()):,} parameters")

    try:
        plain.forward_direction(image)
    except ValueError as error:
        print(f"  and a model without the head refuses: {error}")
    else:
        raise AssertionError("forward_direction must raise without a head")


def check_loss_masking() -> None:
    target = torch.zeros(2, 3, 8, 8)
    target[:, 1] = 1.0                        # cos 2theta = 1 everywhere
    target[:, 2, :, :4] = 1.0                 # coherent on the left half only
    perfect_where_it_counts = torch.zeros(2, 2, 8, 8)
    perfect_where_it_counts[:, 1, :, :4] = 1.0
    perfect_where_it_counts[:, 0, :, 4:] = 5.0   # garbage where weight is 0
    assert float(train.direction_loss(perfect_where_it_counts, target)) < 1e-6
    print("the auxiliary loss ignores pixels with zero coherence -- garbage "
          "off the vessel is free, as intended")

    wrong = torch.zeros(2, 2, 8, 8)
    wrong[:, 1] = -1.0                        # a quarter turn off, everywhere
    charged = float(train.direction_loss(wrong, target))
    assert charged > 1.0, charged
    print(f"  and a field a quarter turn off the vessel is charged "
          f"{charged:.2f}")


def main() -> None:
    data = fake_split()
    check_arity(data)
    check_augmented_target(data)
    check_forward_unchanged()
    check_loss_masking()
    print("all checks passed")


if __name__ == "__main__":
    main()
