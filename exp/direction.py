"""D1: the vessel tangent field, in the only representation that survives.

WHY DOUBLE ANGLE. A vessel's tangent is an AXIS, not a vector: theta and
theta+pi describe the same piece of vessel, and nothing in the data can
distinguish them. Regressing theta, or a unit vector (cos theta, sin theta),
therefore asks the network to break a tie that has no answer -- and the loss
tears at the wrap-around, because two pixels one step apart along the same
vessel can be assigned theta = 179 and theta = 181 degrees, or equivalently
179 and -179, and the second pair is charged a huge error for agreeing.

The standard fix is to regress the DOUBLE angle: (sin 2theta, cos 2theta) is a
genuine function of the axis, identical at theta and theta+pi, and continuous
everywhere. Two channels, a plain MSE, no wrap-around.

The failure this avoids is not loud. It looks like "the auxiliary head does not
learn", because on any straight vessel about half the pixels are pulled one way
and half the other and the gradients cancel. It would be read as evidence
against D1 rather than as a bug in D1.

HOW THE TARGET IS BUILT. The structure tensor of the smoothed ground-truth
mask, J = G_sigma * (grad I)(grad I)^T. Its dominant eigenvector is the
gradient direction, i.e. ACROSS the vessel; the tangent is perpendicular to it.
In double-angle form both are closed-form -- no per-pixel eigendecomposition:

    D          = sqrt((Jxx - Jyy)^2 + 4 Jxy^2)
    gradient   (cos 2phi, sin 2phi) = ((Jxx - Jyy)/D, 2 Jxy/D)
    tangent    (cos 2theta, sin 2theta) = -(gradient), since theta = phi + pi/2
    coherence  D / (Jxx + Jyy), 1 where the structure is locally a clean ridge
               and 0 where it is isotropic (a junction, a blob, the background)

Coherence is carried alongside as the per-pixel confidence, because a junction
has no single tangent and the loss should not pretend otherwise.

  python exp/direction.py --selftest
"""
import sys

import numpy as np
from scipy import ndimage

# One vessel width. DRIVE's median is 2.8 px, and the tensor must average over
# enough of the profile to see the ridge without merging two vessels that pass
# close by. Fixed here, in dataset-relative reasoning, not tuned on a result.
SIGMA = 1.5


def structure_tensor(mask: np.ndarray, sigma: float = SIGMA) -> tuple:
    """(Jxx, Jyy, Jxy) of a smoothed binary mask. x is column, y is row."""
    smooth = ndimage.gaussian_filter(mask.astype(np.float64), sigma)
    grad_y, grad_x = np.gradient(smooth)
    blur = lambda plane: ndimage.gaussian_filter(plane, sigma)
    return blur(grad_x * grad_x), blur(grad_y * grad_y), blur(grad_x * grad_y)


def tangent_field(mask: np.ndarray, sigma: float = SIGMA) -> tuple:
    """(sin 2theta, cos 2theta, coherence) of the vessel tangent.

    theta is measured from the +x (column) axis toward +y (row), which is the
    convention np.gradient hands back and the one dihedral() below is derived
    against. All three planes are float32 and defined everywhere; coherence is
    what says where they mean anything.
    """
    jxx, jyy, jxy = structure_tensor(mask, sigma)
    difference = jxx - jyy
    spread = np.sqrt(difference * difference + 4.0 * jxy * jxy)
    trace = jxx + jyy
    safe = np.where(spread > 1e-12, spread, 1.0)
    # Gradient orientation, then rotated a quarter turn onto the ridge: at
    # double angle a quarter turn is a half turn, hence the single sign flip.
    cos2 = -difference / safe
    sin2 = -2.0 * jxy / safe
    isotropic = spread <= 1e-12
    cos2[isotropic] = 0.0
    sin2[isotropic] = 0.0
    coherence = np.where(trace > 1e-12, spread / np.where(trace > 1e-12,
                                                          trace, 1.0), 0.0)
    return (sin2.astype(np.float32), cos2.astype(np.float32),
            np.clip(coherence, 0.0, 1.0).astype(np.float32))


def dihedral(turns: int, flip: bool, sin2: np.ndarray,
             cos2: np.ndarray) -> tuple:
    """Move a double-angle field through one symmetry of the square.

    The planes must be moved AND their values transformed; moving the arrays
    alone leaves a field that points across the vessel it is drawn on.

    np.rot90 by one turn sends a displacement (dy, dx) to (-dx, dy), so
    theta -> theta + pi/2 and at double angle BOTH components negate.
    np.fliplr sends (dy, dx) to (dy, -dx), so theta -> -theta and only the
    sine negates. Both are verified against a recomputed structure tensor in
    the selftest rather than asserted here.
    """
    moved_sin = np.rot90(sin2, turns)
    moved_cos = np.rot90(cos2, turns)
    if flip:
        moved_sin, moved_cos = np.fliplr(moved_sin), np.fliplr(moved_cos)
    sign = -1.0 if turns % 2 else 1.0
    out_sin = sign * moved_sin * (-1.0 if flip else 1.0)
    out_cos = sign * moved_cos
    return (np.ascontiguousarray(out_sin, dtype=np.float32),
            np.ascontiguousarray(out_cos, dtype=np.float32))


def axis_gap(sin_a, cos_a, sin_b, cos_b) -> np.ndarray:
    """Distance between two axes, in double-angle chord units.

    0 when the axes agree, 2 when they are a quarter turn apart -- which is
    the failure a direction head actually makes, and is exactly what
    sin(2*delta) cannot see, being zero at both ends.
    """
    return np.hypot(sin_a - sin_b, cos_a - cos_b)


def selftest() -> None:
    # A horizontal bar: the tangent runs along +x, so theta = 0 and the double
    # angle is (sin, cos) = (0, 1). Read in the middle, away from the ends.
    canvas = np.zeros((60, 60), dtype=bool)
    canvas[29:32, 5:55] = True
    sin2, cos2, coherence = tangent_field(canvas)
    core = (slice(29, 32), slice(20, 40))
    print(f"horizontal bar: sin2 {sin2[core].mean():+.3f} "
          f"cos2 {cos2[core].mean():+.3f} coherence "
          f"{coherence[core].mean():.3f}")
    assert abs(sin2[core].mean()) < 0.02, sin2[core].mean()
    assert cos2[core].mean() > 0.98, cos2[core].mean()
    assert coherence[core].mean() > 0.9, coherence[core].mean()

    # A vertical bar is theta = pi/2, i.e. the OPPOSITE double angle. If the
    # tangent had been left as the gradient direction these two would swap,
    # so this pins the quarter-turn.
    vertical = np.zeros((60, 60), dtype=bool)
    vertical[5:55, 29:32] = True
    v_sin, v_cos, _ = tangent_field(vertical)
    print(f"vertical bar:   sin2 {v_sin[core[::-1]].mean():+.3f} "
          f"cos2 {v_cos[core[::-1]].mean():+.3f}")
    assert v_cos[core[::-1]].mean() < -0.98, v_cos[core[::-1]].mean()

    # THE TRAP, stated as the thing that goes wrong. Bars at +89 and -89
    # degrees are the SAME axis to within 2 degrees. A unit-vector target
    # places them at opposite ends of the circle; the double angle places
    # them next to each other, which is what the data actually says.
    def bar_at(degrees: float) -> np.ndarray:
        plane = np.zeros((81, 81), dtype=bool)
        yy, xx = np.mgrid[0:81, 0:81]
        radians = np.deg2rad(degrees)
        across = (xx - 40) * np.sin(radians) - (yy - 40) * np.cos(radians)
        along = (xx - 40) * np.cos(radians) + (yy - 40) * np.sin(radians)
        return (np.abs(across) <= 1.2) & (np.abs(along) <= 30)

    middle = (slice(38, 43), slice(38, 43))
    readings = {}
    for degrees in (89.0, -89.0):
        s, c, _ = tangent_field(bar_at(degrees))
        readings[degrees] = (float(s[middle].mean()), float(c[middle].mean()))
    double = np.hypot(readings[89.0][0] - readings[-89.0][0],
                      readings[89.0][1] - readings[-89.0][1])
    naive = np.hypot(np.cos(np.deg2rad(89.0)) - np.cos(np.deg2rad(-89.0)),
                     np.sin(np.deg2rad(89.0)) - np.sin(np.deg2rad(-89.0)))
    print(f"two bars 2 degrees apart (+89 and -89): a unit-vector target "
          f"charges {naive:.2f}, the double angle charges {double:.2f}")
    assert double < 0.2 < naive, (double, naive)
    print(f"  a raw-angle MSE would see them {abs(89.0 - -89.0):.0f} degrees "
          f"apart and pull half of every vessel the wrong way")

    # The augmentation transform, checked the only way worth checking it:
    # against the field recomputed from the transformed mask. A sign error
    # here trains the head on a field rotated 90 degrees off the vessel, and
    # nothing else in the pipeline would notice.
    diagonal = bar_at(30.0)
    base_sin, base_cos, _ = tangent_field(diagonal)
    worst = 0.0
    for turns in range(4):
        for flip in (False, True):
            moved = np.ascontiguousarray(
                np.fliplr(np.rot90(diagonal, turns)) if flip
                else np.rot90(diagonal, turns))
            want_sin, want_cos, _ = tangent_field(moved)
            got_sin, got_cos = dihedral(turns, flip, base_sin, base_cos)
            gap = float(axis_gap(got_sin, got_cos,
                                 want_sin, want_cos)[moved].max())
            worst = max(worst, gap)
            assert gap < 0.05, (turns, flip, gap)
    print(f"all 8 symmetries: field transformed == field recomputed "
          f"(worst axis gap {worst:.4f} of a possible 2.0)")

    # The naive version, to show the test above can fail. Moving the planes
    # and NOT transforming the values leaves a field a quarter turn off the
    # vessel it is drawn on -- gap 2.0, the maximum.
    rotated = np.ascontiguousarray(np.rot90(diagonal, 1))
    want_sin, want_cos, _ = tangent_field(rotated)
    naive = float(axis_gap(np.rot90(base_sin, 1), np.rot90(base_cos, 1),
                           want_sin, want_cos)[rotated].max())
    print(f"  moving the planes WITHOUT the value transform: gap "
          f"{naive:.3f} -- a quarter turn off, the bug this test catches")
    assert naive > 1.9, naive

    # Coherence must fall at a junction: a crossing has no single tangent.
    cross = np.zeros((60, 60), dtype=bool)
    cross[29:32, 5:55] = True
    cross[5:55, 29:32] = True
    _, _, junction = tangent_field(cross)
    centre = junction[28:33, 28:33].mean()
    limb = junction[29:32, 10:20].mean()
    print(f"coherence at a crossing {centre:.3f} vs along a limb {limb:.3f}")
    assert centre < 0.5 * limb, (centre, limb)
    print("all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
