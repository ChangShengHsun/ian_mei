"""E13's architecture-from-name mechanism.

Capacity is encoded in the config name so that eleven call sites which load a
checkpoint all build the matching architecture. The failure this guards against
is quiet: build the wrong width and load_state_dict raises, but build the RIGHT
width by accident (because the suffix was misparsed to the default 16) and a
w32 run would be scored with a 117k model that happens to load nothing. So the
parse and the round trip are both asserted.

  python exp/test_capacity.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train


def main() -> None:
    for name, expected in (("A_dice", 16), ("B_cldice", 16),
                           ("D_blurpool", 16), ("A_dice_w32", 32),
                           ("B_cldice_w32", 32), ("A_dice_w64_d5", 64)):
        got = train.base_width(name)
        assert got == expected, (name, got, expected)
    print("base_width parses every config in CONFIGS:",
          {n: train.base_width(n) for n in train.CONFIGS})

    # A name with no digits after _w must fall back rather than crash, since
    # a future config could legitimately contain the letters "_w".
    assert train.base_width("X_weighted") == 16, train.base_width("X_weighted")
    print("  a word after _w (X_weighted) falls back to 16, not an exception")

    # But a suffix that was CLEARLY meant to be a number and is not must
    # raise. This is the fifteen-hour bug: name a 31M config A_dice_w64d5 and
    # the old parser answered 16, so a 117k model trained under a 30M name,
    # every point of the capacity curve moved, and nothing printed a warning.
    # Same silent shape as E12's hand-written seed range and E16's split
    # normalisation constants; the repo has paid for it twice already.
    for bad in ("A_dice_w64d5", "A_dice_w32x", "B_cldice_w_64",
                "A_dice_w32_w64"):
        try:
            train.base_width(bad)
        except train.ShapeNameError:
            continue
        raise AssertionError(f"{bad} parsed silently instead of raising")
    print("  an unparseable _w suffix raises ShapeNameError, never returns 16")

    for bad in ("A_dice_d5b", "A_dice_w64_d5x", "A_dice_d3_d5"):
        try:
            train.net_depth(bad)
        except train.ShapeNameError:
            continue
        raise AssertionError(f"{bad} parsed silently instead of raising")
    print("  and the same rule holds for _d")

    # The names actually queued must all parse, or the guard above turns a
    # silent wrong curve into a crash at step one -- better, but still a
    # wasted queue if nothing checks it before launch.
    for name in train.CONFIGS:
        train.base_width(name), train.net_depth(name)
    print(f"  every one of the {len(train.CONFIGS)} names in CONFIGS parses")

    # Depth gets the same treatment, and the two suffixes must not interfere:
    # rpartition("_w") on "A_dice_w64_d5" returns "64_d5", which is not a
    # digit string, so the old parser would have called this net base=16 and
    # then loaded a 31M checkpoint into 117k of parameters.
    for name, expected in (("A_dice", 3), ("A_dice_w32", 3),
                           ("A_dice_w64_d5", 5), ("A_dice_d5_w64", 5)):
        got = train.net_depth(name)
        assert got == expected, (name, got, expected)
    assert train.base_width("A_dice_d5_w64") == 64
    assert train.net_depth("X_deep") == 3, train.net_depth("X_deep")
    print("  net_depth parses either suffix order; _d with no digits falls "
          "back to 3")

    # A width variant must be its base arm at another width, nothing else.
    # AUGMENTS is keyed on the full config name, so H_aug_w32 missing from it
    # trains with no augmentation while still being called an augmentation
    # arm: a silently different experiment, reported under the wrong name.
    for name in train.CONFIGS:
        # Strip every architecture suffix, not just a trailing _w32: the
        # 5-level arms carry two, and rpartition("_w") skips them silently --
        # which is exactly this check failing open on the runs it was written
        # to protect.
        base = "_".join(
            token for token in name.split("_")
            if not (token[:1] in "wd" and token[1:].isdigit()))
        if base == name or base not in train.CONFIGS:
            continue
        assert train.AUGMENTS.get(name, ()) == train.AUGMENTS.get(base, ()), (
            name, train.AUGMENTS.get(name, ()), train.AUGMENTS.get(base, ()))
        assert train.CONFIGS[name] == train.CONFIGS[base], name
        print(f"  {name} matches {base}: loss {train.CONFIGS[base][1]}, "
              f"augments {train.AUGMENTS.get(base, ()) or 'none'}")

    counts = {}
    for name in ("A_dice", "A_dice_w32", "A_dice_w64_d5"):
        model = train.build_model(name)
        counts[name] = sum(p.numel() for p in model.parameters())
    print(f"parameters: A_dice {counts['A_dice']:,}, "
          f"A_dice_w32 {counts['A_dice_w32']:,} "
          f"({counts['A_dice_w32'] / counts['A_dice']:.1f}x)")
    # Just under 4x, not exactly: the 1x1 head and the biases are linear in
    # width while the conv blocks are quadratic.
    assert 3.5 < counts["A_dice_w32"] / counts["A_dice"] < 4.0, counts
    # The point of depth 5 at base 64 rather than depth 3 at base 256: both
    # are ~30M, only one puts the parameters where they are cheap to run.
    deep = counts["A_dice_w64_d5"]
    assert 25e6 < deep < 40e6, deep
    print(f"  A_dice_w64_d5 {deep:,} params "
          f"({deep / counts['A_dice']:.0f}x the original net)")

    # Depth changes the module list, so a depth mismatch must raise for the
    # same reason a width mismatch does.
    try:
        train.build_model("A_dice_w64_d5").load_state_dict(
            train.build_model("A_dice_w32").state_dict())
    except RuntimeError:
        print("  and a depth mismatch raises rather than scoring silently")
    else:
        raise AssertionError("depth mismatch loaded without complaint")

    # blurpool still has to survive the refactor: it is the one architecture
    # flag that lived in CONFIGS before capacity did.
    plain = {k for k, _ in train.build_model("A_dice").named_modules()}
    blurred = {k for k, _ in train.build_model("D_blurpool").named_modules()}
    assert plain == blurred, "module names should match; only the type differs"
    assert isinstance(train.build_model("D_blurpool").down1, train.BlurPool)
    assert not isinstance(train.build_model("A_dice").down1, train.BlurPool)
    print("  build_model still honours the blurpool flag")

    # The round trip that the eleven call sites depend on: a checkpoint saved
    # from one config must load into a model built from its name alone.
    state = train.build_model("A_dice_w32").state_dict()
    train.build_model("A_dice_w32").load_state_dict(state)
    print("a w32 checkpoint loads into a model built from the name alone")

    try:
        train.build_model("A_dice").load_state_dict(state)
    except RuntimeError:
        print("  and a width mismatch raises rather than scoring silently")
    else:
        raise AssertionError("width mismatch loaded without complaint")

    # Shape check: the wider model must still return one channel at input size.
    image = torch.randn(1, 1, train.PATCH, train.PATCH,
                        device=train.DEVICE)
    for name in ("B_cldice_w32", "B_cldice_w64_d5"):
        out = train.build_model(name)(image)
        assert out.shape == (1, 1, train.PATCH, train.PATCH), (name, out.shape)
        print(f"forward pass at {train.PATCH}x{train.PATCH} returns "
              f"{tuple(out.shape)} for {name}")

    # 48 px survives four halvings (48/24/12/6/3); whole-image inference pads
    # 584x565 up to the stride, which is what predict_full now reads off the
    # model rather than assuming 4.
    import numpy as np
    prob = train.predict_full(train.build_model("A_dice_w64_d5"),
                              np.zeros((584, 565), np.float32), 0.0, 1.0)
    assert prob.shape == (584, 565), prob.shape
    print("  predict_full returns the original 584x565 through five levels")
    print("all checks passed")


if __name__ == "__main__":
    main()
