"""LIOT: throw the contrast away at the input, before the network sees it.

Shi et al., TIP 2021 (arXiv:2202.12587). Every experiment in this series says
the same thing about where the failures are: E2 stratified by local contrast
and found the dim quartile is where methods differ, E15 put a ceiling on each
band and found Q3 and Q4 already sit at 99% of achievable while Q1 sits at 18%.
Every intervention so far has tried to make the network care more about dim
vessels. LIOT instead removes the thing that makes them dim.

For each of four directions and each distance 1..8, compare the centre pixel
with its neighbour at that distance; set a bit if the centre is brighter. Pack
the eight bits into one byte, one channel per direction:

    channel_d(p) = sum_i  2^(i-1) * [ I(p) > I(p + i*d) ]

The result depends only on the ORDER of pixel intensities along four rays, so
any strictly increasing map applied to the image leaves it unchanged. A vessel
at 5% contrast and the same vessel at 50% contrast produce identical LIOT.

E0's literature check found LIOT has never been evaluated on a topology metric,
which is the gap this arm is aimed at.

  python exp/liot.py --selftest
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DISTANCES = 8
# (dy, dx) per channel. Order is fixed forever: it is baked into every
# checkpoint trained with it.
DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))

# A LIOT pixel reads up to DISTANCES away, so a crop computed in isolation has
# a border of that width where the ray runs off the edge and gets clamped.
MARGIN = DISTANCES


def shifted(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """image translated by (dy, dx), with the edge pixel repeated outward.

    Edge replication rather than zeros: a zero border would read as "darker
    than everything", which sets every bit along that ray and paints a bright
    frame the network can learn to key on. Replication makes the comparison
    with an off-image neighbour compare the pixel against itself, giving 0.
    """
    padded = np.pad(image, ((abs(dy),) * 2, (abs(dx),) * 2), mode="edge")
    top = abs(dy) + dy
    left = abs(dx) + dx
    return padded[top:top + image.shape[0], left:left + image.shape[1]]


def liot(image: np.ndarray) -> np.ndarray:
    """(4, H, W) uint8. Input may be any real-valued 2-D array."""
    out = np.empty((len(DIRECTIONS), *image.shape), dtype=np.uint8)
    for channel, (dy, dx) in enumerate(DIRECTIONS):
        acc = np.zeros(image.shape, dtype=np.uint8)
        for distance in range(1, DISTANCES + 1):
            brighter = image > shifted(image, dy * distance, dx * distance)
            acc |= brighter.astype(np.uint8) << (distance - 1)
        out[channel] = acc
    return out


def selftest() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((40, 40)).astype(np.float32)

    coded = liot(image)
    assert coded.shape == (4, 40, 40), coded.shape
    assert coded.dtype == np.uint8
    print(f"liot returns {coded.shape} uint8, range "
          f"{coded.min()}-{coded.max()}")

    # THE property. Any strictly increasing map of the intensities must leave
    # the code identical, because only the comparisons survive. If this fails
    # the transform is not contrast-invariant and the whole arm is pointless.
    for name, mapped in (("gain x0.1", image * 0.1),
                         ("gain + bias", image * 3 + 7),
                         ("gamma 2.2", image ** 2.2),
                         ("sqrt", np.sqrt(image))):
        assert np.array_equal(liot(mapped), coded), name
    print("  invariant under gain, bias, gamma and sqrt")

    # And it is not invariant to something that CHANGES the ordering, or the
    # check above would pass on a constant output.
    assert not np.array_equal(liot(-image), coded)
    print("  but inverting the image does change it")

    # A vessel is dark on a bright field. Along a ray crossing it, the code
    # must differ between vessel and background; a flat field must give zero.
    flat = np.full((20, 20), 0.5, dtype=np.float32)
    assert liot(flat).max() == 0, "a flat field must produce no bits"
    vessel = flat.copy()
    vessel[10, :] = 0.2
    coded_vessel = liot(vessel)
    on = coded_vessel[:, 10, 10]
    off = coded_vessel[:, 5, 10]
    assert on.max() == 0, ("inside a dark vessel nothing is brighter", on)
    assert off.max() > 0, ("beside it the vessel is darker", off)
    print(f"  flat field -> 0; beside a dark line -> {list(off)}")

    # The vertical channels must react to a horizontal line and the horizontal
    # channels must not, which is what makes four channels worth having.
    assert coded_vessel[0, 5, 10] == 0 and coded_vessel[1, 5, 10] == 0, \
        "a horizontal line should be invisible along a horizontal ray"
    assert coded_vessel[3, 5, 10] > 0, "it must be visible looking down"
    print("  a horizontal line is seen by the vertical channels only")

    # Edge replication, not zero padding: the top-left corner reads a ray that
    # runs off the image. It must compare against itself, giving no bit.
    ramp = np.tile(np.arange(20, dtype=np.float32), (20, 1))
    coded_ramp = liot(ramp)
    assert coded_ramp[0, 0, 0] == 0, ("clamped ray set a bit",
                                      coded_ramp[0, 0, 0])
    print("  a ray running off the edge sets no bit")

    # The margin claim MARGIN == DISTANCES, checked rather than asserted in
    # prose: computing LIOT on a crop and on the full image must agree
    # everywhere except within MARGIN of the crop border.
    full = liot(image)
    crop = liot(image[8:32, 8:32])
    inner = slice(MARGIN, 24 - MARGIN)
    assert np.array_equal(crop[:, inner, inner],
                          full[:, 8 + MARGIN:32 - MARGIN,
                               8 + MARGIN:32 - MARGIN])
    print(f"  a crop agrees with the full image beyond {MARGIN} px of border")
    print("all checks passed")


if __name__ == "__main__":
    selftest()
