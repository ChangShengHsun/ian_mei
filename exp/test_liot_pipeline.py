"""LIOT through the real batch path, not just the transform in isolation.

liot.py checks the encoding. What it cannot check is the plumbing, and the
plumbing is where the silent failures live: a 4-channel batch that arrives
transposed, a normalisation computed over the wrong axis, a label that no
longer lines up with its image after the wider crop is trimmed back. None of
those raise. They just train a worse model.

  python exp/test_liot_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment
import liot
import train


def fake_split(rng, count=4, size=120) -> dict:
    """An image that IS its own label, so misalignment is visible as a number.

    Same trick augment.py uses. If any step transforms the image without
    transforming the label the same way, the two stop being equal and the
    assertion below fails; a real image would let that pass unnoticed.
    """
    labels = (rng.random((count, size, size)) > 0.7).astype(np.float32)
    return {"images": labels.copy(),
            "labels": labels,
            "fovs": np.ones((count, size, size), dtype=bool),
            "dists": np.zeros((count, size, size), dtype=np.float32),
            "names": [f"img{i}" for i in range(count)]}


def main() -> None:
    rng = np.random.default_rng(0)
    data = fake_split(rng)

    assert train.uses_liot("J_liot") and not train.uses_liot("H_aug")
    model = train.build_model("J_liot")
    assert model.enc1[0].in_channels == 4, model.enc1[0].in_channels
    grey_params = sum(p.numel() for p in train.build_model("H_aug").parameters())
    liot_params = sum(p.numel() for p in model.parameters())
    print(f"J_liot is 4-channel: {liot_params} params against "
          f"{grey_params} for grey (+{liot_params - grey_params})")

    mean, std = train.liot_stats(data)
    assert mean.shape == (4, 1, 1) and std.shape == (4, 1, 1), mean.shape
    assert (std > 0).all(), std
    print(f"per-channel stats: mean {np.round(mean.ravel(), 1)}, "
          f"std {np.round(std.ravel(), 1)}")

    images, labels, dists = train.sample_batch(
        data, rng, mean, std, ("dihedral", "jitter"), None, use_liot=True)
    assert images.shape == (train.BATCH, 4, train.PATCH, train.PATCH), images.shape
    assert labels.shape == (train.BATCH, 1, train.PATCH, train.PATCH), labels.shape
    assert dists.shape == labels.shape
    print(f"batch shapes: images {tuple(images.shape)}, "
          f"labels {tuple(labels.shape)}")

    # The normalisation has to leave the batch roughly standardised. Loose
    # bounds: one batch of 32 crops is a small sample and the aperture here is
    # the whole frame, so this catches an axis mistake, not a subtle bias.
    per_channel = images.mean(dim=(0, 2, 3))
    assert per_channel.abs().max() < 1.5, per_channel
    print(f"normalised per-channel means: "
          f"{np.round(per_channel.cpu().numpy(), 2)}")

    # The crop margin actually did something: a batch taken with use_liot must
    # not contain the clamped-ray border. Encode a patch the naive way (crop
    # then encode) and confirm it DIFFERS at the border from the pipeline's,
    # which is the artefact MARGIN exists to remove.
    patch = data["images"][0][:train.PATCH + 2 * liot.MARGIN,
                              :train.PATCH + 2 * liot.MARGIN]
    inner = slice(liot.MARGIN, liot.MARGIN + train.PATCH)
    wide = liot.liot(patch)[:, inner, inner]
    naive = liot.liot(patch[inner, inner])
    edge_differs = (wide[:, 0, :] != naive[:, 0, :]).any()
    assert edge_differs, "the margin changed nothing -- it is not being used"
    assert np.array_equal(wide[:, train.PATCH // 2, train.PATCH // 2],
                          naive[:, train.PATCH // 2, train.PATCH // 2]), \
        "the centre must not depend on the margin"
    print("the wider crop changes the patch border and not its centre")

    # Alignment through the whole path. The image IS the label, and structure
    # here is BRIGHT (1.0) on a 0.0 field, so a background pixel is never
    # strictly greater than any neighbour and must code exactly 0 in all four
    # channels. If a geometric step moved the image without moving the label,
    # some label-background position would land on image-structure and code
    # non-zero. Passed mean=0, std=1 so these are raw codes.
    raw, labels, _ = train.sample_batch(
        data, np.random.default_rng(1), 0.0, 1.0, ("dihedral",), None,
        use_liot=True)
    assert raw.shape[1] == 4
    background = labels[:, 0] < 0.5
    off_structure = raw.permute(1, 0, 2, 3)[:, background]
    assert float(off_structure.max()) == 0.0, float(off_structure.max())
    # and the check is not vacuous: structure pixels DO code non-zero.
    on_structure = raw.permute(1, 0, 2, 3)[:, ~background]
    assert float(on_structure.max()) > 0.0
    print(f"image and label stay aligned through LIOT + dihedral "
          f"({int(background.sum())} background pixels, all four channels 0)")

    # How much of a no-op jitter is under LIOT. Not asserted as exactly zero:
    # jitter clips to [0, 1], and clipping can tie two pixels that used to
    # differ, which flips a strict comparison. Measuring beats assuming.
    grey = data["images"][0]
    jittered = augment.jitter(grey, np.random.default_rng(2))
    changed = (liot.liot(grey) != liot.liot(jittered)).mean()
    print(f"jitter changes {changed:.4%} of LIOT bits "
          f"(a monotone map would change none; clipping creates ties)")
    assert changed < 0.05, changed

    # predict_full must pick the representation off the model, since every
    # analysis script calls it with nothing but a checkpoint and an image.
    prob = train.predict_full(model, data["images"][0], mean, std)
    assert prob.shape == data["images"][0].shape, prob.shape
    assert 0.0 <= prob.min() and prob.max() <= 1.0
    print(f"predict_full on a LIOT model returns {prob.shape} in "
          f"[{prob.min():.3f}, {prob.max():.3f}]")

    # And the grey path still works unchanged -- this file's edits touched it.
    grey_model = train.build_model("A_dice")
    grey_prob = train.predict_full(grey_model, data["images"][0], 0.5, 0.2)
    assert grey_prob.shape == data["images"][0].shape
    grey_batch = train.sample_batch(data, rng, 0.5, 0.2)[0]
    assert grey_batch.shape == (train.BATCH, 1, train.PATCH, train.PATCH), \
        grey_batch.shape
    print("the 1-channel path is unchanged:", tuple(grey_batch.shape))

    # The bug that cost E16 its first verdict: the representation was decided
    # from the model, the constants were still passed in, and the two halves
    # disagreed. Encoding to LIOT and then normalising byte codes with grey
    # statistics does not raise -- it puts the input ~800 sigma out, the model
    # predicts nothing, and the analysis reports Dice 0.0000 as a finding.
    grey_mean, grey_std = train.normalisation("A_dice_s0", data)
    liot_mean, liot_std = train.normalisation("J_liot_s0", data)
    assert np.ndim(grey_mean) == 0 and np.shape(liot_mean) == (4, 1, 1)
    for bad_model, bad_mean, bad_std, why in (
            (model, grey_mean, grey_std, "4-channel model, scalar constants"),
            (grey_model, liot_mean, liot_std, "1-channel model, per-channel")):
        try:
            train.predict_full(bad_model, data["images"][0], bad_mean, bad_std)
        except ValueError:
            print(f"  refused: {why}")
        else:
            raise AssertionError(f"scored silently with {why}")

    # A checkpoint round trip, because the name is the only record of how many
    # channels a run had.
    state = model.state_dict()
    train.build_model("J_liot").load_state_dict(state)
    try:
        train.build_model("H_aug").load_state_dict(state)
    except RuntimeError:
        print("a grey model refuses a LIOT checkpoint rather than scoring it")
    else:
        raise AssertionError("channel mismatch loaded without complaint")

    # J_liot must not use a loss that reads the input image as grey.
    assert train.CONFIGS["J_liot"][1] is None, train.CONFIGS["J_liot"]
    assert train.AUGMENTS["J_liot"] == train.AUGMENTS["H_aug"], \
        "the arms must differ only in the input representation"
    print("J_liot differs from H_aug in exactly one thing: the input")
    print("all checks passed")


if __name__ == "__main__":
    main()
