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
                           ("B_cldice_w32", 32)):
        got = train.base_width(name)
        assert got == expected, (name, got, expected)
    print("base_width parses every config in CONFIGS:",
          {n: train.base_width(n) for n in train.CONFIGS})

    # A name with no digits after _w must fall back rather than crash, since
    # a future config could legitimately contain the letters "_w".
    assert train.base_width("X_weighted") == 16, train.base_width("X_weighted")
    print("  a non-numeric _w suffix falls back to 16, not an exception")

    counts = {}
    for name in ("A_dice", "A_dice_w32"):
        model = train.build_model(name)
        counts[name] = sum(p.numel() for p in model.parameters())
    print(f"parameters: A_dice {counts['A_dice']:,}, "
          f"A_dice_w32 {counts['A_dice_w32']:,} "
          f"({counts['A_dice_w32'] / counts['A_dice']:.1f}x)")
    # Just under 4x, not exactly: the 1x1 head and the biases are linear in
    # width while the conv blocks are quadratic.
    assert 3.5 < counts["A_dice_w32"] / counts["A_dice"] < 4.0, counts

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
    image = torch.randn(1, 1, train.PATCH, train.PATCH)
    out = train.build_model("B_cldice_w32")(image)
    assert out.shape == (1, 1, train.PATCH, train.PATCH), out.shape
    print(f"forward pass at {train.PATCH}x{train.PATCH} returns {tuple(out.shape)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
