"""The flux representation: geometry as the network's OUTPUT, not its teacher.

WRITTEN AND SELFTESTED 2026-09-07, before any head consumes it.

WHY A SEPARATE MODULE. exp/flux_ceiling.py priced this representation and
exp/train.py will learn it; two copies of the target arithmetic is how two
tables quietly stop agreeing. This is the one definition, and both import it.

THE REPRESENTATION. For every pixel within `radius` of the ground-truth
centreline, the DISPLACEMENT to its nearest centreline pixel, in pixels. The
centreline is decoded by voting: every banded pixel points somewhere, and a
pixel called centreline is one that enough pixels point at. DeepFlux
(arXiv:1811.12608) uses the unit vector and recovers the skeleton with
morphology; the displacement is kept unnormalised here because the magnitude
is exactly what turns the decode into a vote instead of a search.

WHY IT MIGHT HELP, stated so it can be checked later. A 4 px wide vessel
contributes about 4 binary pixels of positive signal per unit length to a
mask loss, against a background of tens of thousands. Under this
representation every pixel within `radius` of that vessel carries a target
pointing at it -- about 2*radius per unit length, and directional, which is
harder to explain away as noise. The failure this is aimed at is measured:
on HRF 5.8% of the centreline sits below p=0.01 and 40.8% of that is in runs
of 20 px or more, whole vessels the model never sees.

WHY IT MIGHT NOT. D1's tangent head LEARNED its target (axis gap 0.243
against a constant 1.205) and segmentation did not improve. The diagnosis was
that an auxiliary head can be ignored by the mask head. That diagnosis only
protects this line if the flux is LOAD-BEARING -- if the reported mask is
rendered from the decode rather than read off a separate mask head. A flux
head bolted beside a mask head is D1 again, and would deserve D1's result.

THE VECTOR TRAP, which is why this module exists at all. Under augmentation
the field's VALUES move, not just its pixels. augment.dihedral says so in its
own docstring and refuses to handle it. np.rot90 by one turn sends a
displacement (dy, dx) to (-dx, dy); np.fliplr sends it to (dy, -dx). Rather
than assert that reasoning, the selftest RECOMPUTES the target from the
transformed skeleton and compares -- the same discipline direction.py uses
for the tangent field.

TIES, found by that selftest on its first run, 2026-09-07. The comparison is
NOT exact equality, because the target is ambiguous where a pixel is
equidistant from two centreline pixels: distance_transform_edt picks one
arbitrarily and that choice is not equivariant. On a 33x33 test skeleton 65
of 1089 pixels differed, and ALL 65 landed at the same distance -- ties, not
errors. So the invariant asserted is the one that actually holds: every
landing is on the centreline, and at the same distance as the recomputed
target, with exact agreement wherever the nearest point is unique.

That ambiguity is also a real property of the representation and not only of
the test. Under augmentation a network sees a different, equally correct
target at those pixels. They sit on the medial axis BETWEEN two vessels, so
they are at the outer edge of the band; a narrower `radius` has fewer of
them. If a flux head is ever seen to be unstable, this is the first place to
look.

  python exp/flux.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment


def target(skel: np.ndarray, radius: float) -> tuple:
    """((dy, dx), band) -- displacement to the nearest centreline pixel.

    Outside the band the displacement is meaningless and a loss must not read
    it; the band is returned rather than baked in so the caller cannot forget.
    """
    distance, (near_y, near_x) = ndimage.distance_transform_edt(
        ~skel, return_indices=True)
    rows, cols = np.indices(skel.shape)
    return ((near_y - rows).astype(np.float32),
            (near_x - cols).astype(np.float32)), distance <= radius


def dihedral(turns: int, flip: bool, delta_y: np.ndarray,
             delta_x: np.ndarray) -> tuple:
    """Move a displacement field through one symmetry of the square.

    Planes AND values. One quarter turn sends (dy, dx) to (-dx, dy); the flip
    sends (dy, dx) to (dy, -dx). Verified in the selftest by recomputing the
    target from the transformed skeleton, not by trusting this comment.
    """
    moved_y, moved_x = np.rot90(delta_y, turns), np.rot90(delta_x, turns)
    for _ in range(turns % 4):
        moved_y, moved_x = -moved_x, moved_y
    if flip:
        moved_y, moved_x = np.fliplr(moved_y), -np.fliplr(moved_x)
    return (np.ascontiguousarray(moved_y, dtype=np.float32),
            np.ascontiguousarray(moved_x, dtype=np.float32))


def decode(delta_y: np.ndarray, delta_x: np.ndarray, band: np.ndarray,
           votes: int = 1) -> np.ndarray:
    """Recover the centreline: every banded pixel votes for where it points.

    Rounding is what makes this tolerant -- a displacement wrong by less than
    half a pixel lands on the same target. flux_ceiling.py measured that ERL
    is flat in displacement noise up to 2 px and that the cost of noise is
    paint, not tracing.
    """
    shape = delta_y.shape
    rows, cols = np.indices(shape)
    target_y = np.clip(np.rint(rows + delta_y), 0, shape[0] - 1).astype(int)
    target_x = np.clip(np.rint(cols + delta_x), 0, shape[1] - 1).astype(int)
    flat = np.bincount((target_y * shape[1] + target_x)[band],
                       minlength=shape[0] * shape[1])
    return flat.reshape(shape) >= votes


def render(centreline: np.ndarray, radius_map: np.ndarray) -> np.ndarray:
    """The mask a centreline-plus-radius output produces."""
    out = np.zeros_like(centreline)
    for value in np.unique(np.rint(radius_map[centreline])):
        if value < 1:
            continue
        seeds = centreline & (np.rint(radius_map) == value)
        if not seeds.any():
            continue
        out |= ndimage.binary_dilation(
            seeds, ndimage.generate_binary_structure(2, 2),
            iterations=int(value))
    return out | centreline


def selftest() -> None:
    # An ASYMMETRIC skeleton on purpose: a symmetric one passes every sign
    # error in the dihedral, which is exactly the bug this test exists for.
    skel = np.zeros((33, 33), dtype=bool)
    skel[16, 6:26] = True          # a horizontal bar
    skel[8:17, 20] = True          # a branch going up from near its right end
    for step in range(6):          # and a diagonal tail, so no axis is spared
        skel[16 + step, 6 + step] = True

    # 1. THE VECTOR TRAP. For every element of the group, transforming the
    #    TARGET must agree with recomputing the target from the transformed
    #    skeleton -- exactly where the nearest centreline pixel is unique, and
    #    up to a tie elsewhere. See TIES in the module docstring.
    total_ties = 0
    for turns in range(4):
        for flip in (False, True):
            (base_y, base_x), _ = target(skel, 7.0)
            moved_skel, = augment.apply_dihedral(turns, flip, skel)
            (want_y, want_x), _ = target(moved_skel, 7.0)
            got_y, got_x = dihedral(turns, flip, base_y, base_x)
            differs = (got_y != want_y) | (got_x != want_x)
            total_ties += int(differs.sum())
            # Every disagreement must be a tie: same distance, and the
            # transformed field must still land ON the centreline.
            rows, cols = np.nonzero(differs)
            for row, col in zip(rows, cols):
                landed = (int(round(row + got_y[row, col])),
                          int(round(col + got_x[row, col])))
                assert moved_skel[landed], (turns, flip, row, col,
                                            "landed off the centreline")
                assert abs(np.hypot(got_y[row, col], got_x[row, col])
                           - np.hypot(want_y[row, col], want_x[row, col])
                           ) < 1e-6, (turns, flip, row, col, "not a tie")
    print(f"dihedral: agrees with recomputation on all 8 symmetries; the "
          f"{total_ties} disagreeing pixels are all ties (same distance, "
          f"still on the centreline)")

    # 2. AND THE TRAP ITSELF MUST BE REAL. If moving the planes alone happened
    #    to be right, the module would be pointless -- so assert that the
    #    naive version is WRONG, or a future simplification will look safe.
    (base_y, base_x), _ = target(skel, 7.0)
    naive_y, naive_x = np.rot90(base_y, 1), np.rot90(base_x, 1)
    moved_skel, = augment.apply_dihedral(1, False, skel)
    (want_y, want_x), _ = target(moved_skel, 7.0)
    assert not (np.array_equal(naive_y, want_y)
                and np.array_equal(naive_x, want_x)), \
        "moving the planes alone was enough -- this module has no purpose"
    print("moving the planes without transforming the values IS wrong, as "
          "augment.dihedral warns")

    # 3. THE BAND IS THE BAND, and the displacement inside it is the true nearest
    #    centreline pixel, checked against a brute-force search on a few
    #    points rather than against the same distance transform that built it.
    (delta_y, delta_x), band = target(skel, 5.0)
    coords = np.argwhere(skel)
    rng = np.random.default_rng(0)
    for row, col in rng.choice(np.argwhere(band), size=25, replace=False):
        best = coords[np.argmin(np.hypot(coords[:, 0] - row,
                                         coords[:, 1] - col))]
        landed = (row + delta_y[row, col], col + delta_x[row, col])
        assert skel[int(landed[0]), int(landed[1])], (row, col)
        assert abs(np.hypot(*(landed - np.array([row, col])))
                   - np.hypot(*(best - np.array([row, col])))) < 1e-6
    print("25 random banded pixels point at a true centreline pixel, at the "
          "brute-force nearest distance")

    # 4. THE DECODE IS EXACT ON A PERFECT FIELD -- which is why flux_ceiling
    #    measures tolerance instead of running an oracle test.
    got = decode(delta_y, delta_x, band)
    assert np.array_equal(got, skel), "a perfect field did not decode exactly"
    print("a perfect field decodes to the skeleton exactly")

    # 5. RENDER MUST WIDEN, and must contain the centreline it was given.
    radius_map = np.full(skel.shape, 2.0)
    mask = render(skel, radius_map)
    assert mask[skel].all() and mask.sum() > skel.sum() * 3
    print(f"render at radius 2 turns {skel.sum()} centreline px into "
          f"{mask.sum()} mask px, containing all of them")
    print("all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        raise SystemExit("pass --selftest")
