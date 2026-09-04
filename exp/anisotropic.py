"""The mechanism D1 exists to drive: correction that knows the vessel's axis.

WRITTEN AND SELFTESTED 2026-08-27, BEFORE IT WAS RUN ON A SINGLE PREDICTION.

WHAT IT IS FOR. Measured this morning on K_focal_aug: the prediction runs
BESIDE the true centreline, offset a median 1.4 px on vessels 2.8 px wide,
for stretches whose 90th percentile is 21 px. Covering that centreline is
worth 16.7 points of traced tree once erl.py's splitting rule is accounted
for -- against 5.1 for linking every severing break, which is why C1 was
retired and this was not.

The same measurement says the correction must be ANISOTROPIC. Dilating one
pixel in every direction reaches 67.9% traced and drops Dice from 0.80 to
0.66; putting the same foreground only where it belongs reaches 84.0% and
RAISES Dice to 0.82. The difference between those two rows is the whole
argument for predicting direction at all.

HOW IT WORKS. Source-driven oriented dilation. Every foreground pixel spreads
through an ellipse aligned with ITS OWN tangent: semi-axis `along` down the
vessel, semi-axis `across` through it. Orientation is quantised into K axes
(K bins over [0, pi), because a tangent is an axis and theta = theta + pi),
each bin dilated with its own structuring element, and the results unioned.

The two radii are swept rather than chosen, and the sweep is the experiment:
if the budget is captured by `along` alone, the fix is extending vessels
lengthways; if it needs `across`, the fix is that the vessel is drawn in the
wrong place. Nobody knows which yet, and a design that can only express one
of them would answer the question by assuming it.

Radii are in multiples of the median vessel width, never pixels -- CLAUDE.md's
rule, so the same setting means the same thing on a retina at six times the
resolution.

  python exp/anisotropic.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Orientation bins over [0, pi). 16 gives 11.25 degrees of quantisation error,
# which is under the angular error the direction head itself makes (axis gap
# 0.23 is about 6.6 degrees), so the bins are not the limiting factor.
BINS = 16


def ellipse(along: float, across: float, angle: float) -> np.ndarray:
    """A structuring element: semi-axis `along` at `angle`, `across` normal.

    Always contains its own centre, so a dilation with it can only add
    foreground and never erase a pixel -- which is what makes every condition
    in the sweep a superset of the raw prediction and the comparison monotone.
    """
    reach = int(np.ceil(max(along, across)))
    if reach < 1:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    # Rotate the sample grid into the ellipse's frame. x is column, y is row,
    # matching direction.py's convention for theta.
    u = xx * np.cos(angle) + yy * np.sin(angle)      # along the axis
    v = -xx * np.sin(angle) + yy * np.cos(angle)     # across it
    a = max(along, 0.5)
    b = max(across, 0.5)
    inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    inside[reach, reach] = True
    return inside


def axis_element(along: float, across: float, angle: float) -> np.ndarray:
    """A structuring element that ACTUALLY reaches `along` at `angle`.

    REPLACES ellipse() inside oriented_dilation on 2026-09-01. The ellipse
    tested each lattice point for `(u/a)^2 + (v/b)^2 <= 1`, which is exact in
    continuous space and badly wrong on a grid: the lattice point nearest the
    axis at distance `a` sits up to 0.71 px off it, so with a thin `b` the tip
    fails the test and the element is silently shorter than asked. Measured
    over the sweep's own grid at DRIVE's 2.83 px width:

        ALONG=0.5 ACROSS=0.25 (the geometry the sweep PICKED): asked 1.41 px,
        delivered a mean of 0.71, and EXACTLY 0.00 in 4 of the 16 orientation
        bins -- the two diagonal bands. Diagonal vessels received no growth
        along their axis at all.
        ALONG=2.0 ACROSS=0.25: asked 5.66 px, delivered 3.98-5.00 (78%).

    So `along` did not mean what it said, the shortfall depended on the
    vessel's orientation, and it was worst exactly where it was smallest.
    Every postproc/direction_ceiling CSV written before 2026-09-01 carries it.

    The fix is to rasterise the axis instead of testing membership: step along
    the segment at half-pixel intervals and round, which puts a lattice point
    within 0.5 px of every point on it, then thicken by `across`. Reach is
    then exact by construction rather than by luck, at every angle.
    """
    reach = int(np.ceil(max(along, across)))
    if reach < 1:
        return np.ones((1, 1), dtype=bool)
    size = 2 * reach + 1
    out = np.zeros((size, size), dtype=bool)
    out[reach, reach] = True
    if along >= 0.5:
        steps = max(int(np.ceil(along * 2)), 1)
        for step in range(-steps, steps + 1):
            distance = along * step / steps
            row = int(round(reach + distance * np.sin(angle)))
            column = int(round(reach + distance * np.cos(angle)))
            out[row, column] = True
    if across >= 0.5:
        yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
        out = ndimage.binary_dilation(
            out, structure=(xx * xx + yy * yy) <= across * across)
    return out


def bin_index(sin2: np.ndarray, cos2: np.ndarray) -> np.ndarray:
    """Which orientation bin each pixel's axis falls in.

    theta is recovered from the double angle and wrapped into [0, pi), which
    is the range an axis actually lives in; a vector representation would
    split one vessel across two opposite bins.
    """
    theta = (0.5 * np.arctan2(sin2, cos2)) % np.pi
    return np.minimum((theta / np.pi * BINS).astype(np.int64), BINS - 1)


def oriented_dilation(mask: np.ndarray, sin2: np.ndarray, cos2: np.ndarray,
                      along: float, across: float) -> np.ndarray:
    """Dilate every foreground pixel along its own tangent axis.

    Source-driven: the orientation used for a pixel is the one AT that pixel,
    so a vessel extends down its own length rather than through the direction
    of whatever happens to be nearby. Doing it the other way round -- reading
    the orientation at the destination -- would let a strong local direction
    pull in foreground from an unrelated vessel crossing it.
    """
    if along < 0.5 and across < 0.5:
        return mask.copy()
    bins = bin_index(sin2, cos2)
    out = np.zeros_like(mask)
    for index in range(BINS):
        source = mask & (bins == index)
        if not source.any():
            continue
        angle = (index + 0.5) / BINS * np.pi
        out |= ndimage.binary_dilation(
            source, structure=axis_element(along, across, angle))
    return out


def isotropic_dilation(mask: np.ndarray, radius: float) -> np.ndarray:
    """The control. Same operation with no direction in it."""
    if radius < 0.5:
        return mask.copy()
    reach = int(np.ceil(radius))
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    return ndimage.binary_dilation(
        mask, structure=(xx * xx + yy * yy) <= radius * radius)


def shuffled_field(shape: tuple, seed: int) -> tuple:
    """A per-pixel axis field carrying no information about the image.

    THE CONTROL THAT MATTERS. Oriented dilation adds foreground, and adding
    foreground raises ERL on its own -- that is how the closing baseline beat
    the C1 oracle before its Dice cost was matched. If a random field does as
    well as the real one, this mechanism is dilation wearing a direction
    field, and the whole D1 line is measuring nothing.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, np.pi, size=shape)
    return np.sin(2 * theta), np.cos(2 * theta)


def _bin_angles():
    return [(index + 0.5) / BINS * np.pi for index in range(BINS)]


def _reach_of(element: np.ndarray, angle: float) -> float:
    """How far the element actually extends along `angle`."""
    radius = element.shape[0] // 2
    rows, columns = np.nonzero(element)
    return float(np.max((columns - radius) * np.cos(angle)
                        + (rows - radius) * np.sin(angle)))


def selftest() -> None:
    # THE REACH MUST BE THE REACH, AT EVERY ORIENTATION. This is the check
    # that was missing while the sweep ran, and its absence is why the
    # picked geometry delivered zero growth on diagonal vessels.
    print("delivered reach vs asked, worst orientation bin:")
    for along, across in ((1.41, 0.71), (2.83, 0.71), (5.66, 0.71),
                          (5.66, 0.0)):
        worst = min(_reach_of(axis_element(along, across, ang), ang)
                    for ang in _bin_angles())
        old = min(_reach_of(ellipse(along, across, ang), ang)
                  for ang in _bin_angles())
        print(f"  along {along:5.2f} across {across:4.2f}: "
              f"axis_element {worst:5.2f}  (ellipse was {old:5.2f})")
        assert worst >= along - 1.0, (along, across, worst)
        assert worst >= old, (worst, old)

    # An ellipse aligned with the x axis must be wide in x and thin in y.
    flat = ellipse(4.0, 1.0, 0.0)
    assert flat.shape == (9, 9), flat.shape
    rows = flat.any(1).sum()
    cols = flat.any(0).sum()
    print(f"ellipse(along=4, across=1, angle=0): {cols} px wide, {rows} tall")
    assert cols > rows, (cols, rows)
    # Rotated a quarter turn it must swap, and it must always hold its centre.
    upright = ellipse(4.0, 1.0, np.pi / 2)
    assert upright.any(0).sum() < upright.any(1).sum()
    assert flat[4, 4] and upright[4, 4]
    print("  a quarter turn swaps its axes, and it always contains its centre")

    # A diagonal vessel must extend ALONG itself and not sideways.
    size = 81
    yy, xx = np.mgrid[0:size, 0:size]
    line = np.abs((xx - yy)) <= 0.6
    line &= (xx > 20) & (xx < 60)
    import direction
    sin2, cos2, _ = direction.tangent_field(line)
    grown = oriented_dilation(line, sin2, cos2, along=6.0, across=0.0)
    # It should reach further along the diagonal than perpendicular to it.
    along_reach = grown[yy == xx].sum() - line[yy == xx].sum()
    across_reach = grown[(xx - yy) == 4].sum()
    print(f"a 45-degree vessel dilated along=6 across=0: {along_reach} px "
          f"gained on its own diagonal, {across_reach} px four rows off it")
    assert along_reach > 0
    assert across_reach < along_reach, (along_reach, across_reach)

    # And the isotropic control at a comparable radius must spill sideways.
    round_grown = isotropic_dilation(line, 6.0)
    round_across = round_grown[(xx - yy) == 4].sum()
    print(f"  isotropic dilation radius 6 puts {round_across} px there "
          f"instead -- that spill is what costs Dice")
    assert round_across > across_reach, (round_across, across_reach)

    # Foreground cost: for the SAME added foreground, the oriented version
    # must stay closer to the structure. This is the whole claim.
    oriented_fg = grown.sum() / line.sum()
    print(f"oriented adds {oriented_fg:.2f}x foreground, isotropic "
          f"{round_grown.sum() / line.sum():.2f}x, for the same reach")
    assert grown.sum() < round_grown.sum()

    # Dilation must never remove a pixel, whatever the radii.
    for along, across in ((0.0, 0.0), (3.0, 0.0), (0.0, 2.0), (5.0, 2.0)):
        got = oriented_dilation(line, sin2, cos2, along, across)
        assert (got | line == got).all(), (along, across)
    print("every setting is a superset of the input, so the sweep is monotone")

    # The shuffled control must be a genuine axis field and must NOT track
    # the structure.
    s_sin, s_cos = shuffled_field(line.shape, seed=0)
    assert np.allclose(np.hypot(s_sin, s_cos), 1.0)
    agree = np.abs(direction.axis_gap(s_sin, s_cos, sin2, cos2))[line].mean()
    real = np.abs(direction.axis_gap(sin2, cos2, sin2, cos2))[line].mean()
    print(f"shuffled field: axis gap {agree:.2f} against the true tangent "
          f"(the true field scores {real:.2f}) -- it carries no direction")
    assert agree > 1.0, agree

    # Bins must cover [0, pi) and treat theta and theta+pi as one axis.
    # Away from the seam this is exact.
    # Angles deliberately off a bin boundary: at a boundary the wrap is a
    # coin flip on the last bit, which the seam check below covers instead.
    theta = np.array([[0.3, 1.0, 2.5, 2.9]])
    same = bin_index(np.sin(2 * theta), np.cos(2 * theta))
    wrapped = bin_index(np.sin(2 * (theta + np.pi)),
                        np.cos(2 * (theta + np.pi)))
    assert (same == wrapped).all(), (same, wrapped)
    span = bin_index(np.sin(2 * np.linspace(0, np.pi, 200)),
                     np.cos(2 * np.linspace(0, np.pi, 200)))
    assert set(span.tolist()) == set(range(BINS)), sorted(set(span.tolist()))
    print(f"{BINS} bins span [0, pi) and theta+pi lands in the same bin")

    # AT the seam, theta = 0 and theta = pi are one axis that rounds to bin 0
    # or bin 15 depending on the last bit of the arctangent. That is not a
    # bug to fix but a fact to bound: the two are ADJACENT orientations under
    # the mod-pi topology the bins live in, 11.25 degrees apart, so the worst
    # a seam pixel suffers is one bin of quantisation like any other pixel.
    seam = bin_index(np.sin(2 * np.array([[0.0, np.pi - 1e-12]])),
                     np.cos(2 * np.array([[0.0, np.pi - 1e-12]])))
    gap = min(abs(int(seam[0, 0]) - int(seam[0, 1])),
              BINS - abs(int(seam[0, 0]) - int(seam[0, 1])))
    print(f"  at the seam theta=0 and theta=pi land in bins {seam.tolist()[0]},"
          f" which are {gap} apart on the mod-pi circle -- adjacent, not "
          f"opposite")
    assert gap <= 1, (seam, gap)
    # And adjacent bins really are adjacent orientations, which is what makes
    # that harmless: the elements they dilate with differ by one step.
    first = (0 + 0.5) / BINS * np.pi
    last = (BINS - 1 + 0.5) / BINS * np.pi
    separation = min(abs(first - last), np.pi - abs(first - last))
    assert abs(np.rad2deg(separation) - 180.0 / BINS) < 1e-6, separation
    print(f"  and their structuring elements differ by "
          f"{np.rad2deg(separation):.2f} degrees, one quantisation step")

    print("all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
