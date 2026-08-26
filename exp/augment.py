"""E14: the two cheapest interventions the literature says actually move topology.

Twenty experiments in and this pipeline had NO data augmentation -- random crops
and nothing else. That matters more than any loss we have compared. CoLeTra's
DRIVE table (arXiv:2503.05541, Table 1) reports Betti error 3.687 -> 1.354 for
plain data augmentation and 1.354 -> 1.282 for their method on top of it, so
the augmentation itself is a 63% reduction and dwarfs every loss-level effect
this series has measured (E5: on clean labels the losses sit closer together
than two seeds of one loss).

Two transforms live here:

  dihedral + jitter   standard augmentation. The retina has no canonical
                      orientation at patch scale, so the full symmetry group of
                      the square is available, and gamma/gain/bias covers the
                      illumination variation a fundus camera actually produces.

  coletra             Valverde et al. 2025. Paste an INPAINTED (vessel-removed)
                      patch over random foreground pixels while leaving the
                      LABEL untouched, so the model is taught that structures
                      which look broken are still connected. The paper uses LaMa
                      for the inpainting and states the method is agnostic to
                      the inpainter; for a dark thin structure on a smooth
                      background a grey closing does the same job in one call,
                      and E7's contrast gate already leans on exactly that
                      image.

The label staying unchanged is the whole mechanism, so it is asserted below
rather than assumed.

  python exp/augment.py        # runnable checks
"""
import numpy as np
from scipy import ndimage

# Wider than any DRIVE vessel (median width 2.8 px), so the closing removes the
# structure rather than thinning it. Same size E7's contrast gate uses.
INPAINT_SIZE = 15

# CoLeTra's two hyper-parameters. The paper swept patch sizes 11/15/19 on DRIVE
# "since the structures have a similar width" and chose the count to cover
# 40-60% of the structure; PATCHES is set to land in that band on a 48 px crop,
# which the selftest measures rather than trusts.
COLETRA_PATCH = 15
COLETRA_PATCHES = 2
COLETRA_SIGMA = COLETRA_PATCH / 4.0


def dihedral(rng, *planes: np.ndarray) -> tuple:
    """One of the eight symmetries of the square, applied to every plane.

    Image, label and distance map must receive the SAME element or the batch
    silently teaches the network a mirrored target, which no loss would flag.
    """
    turns = int(rng.integers(4))
    flip = rng.random() < 0.5
    out = []
    for plane in planes:
        moved = np.rot90(plane, turns)
        out.append(np.ascontiguousarray(np.fliplr(moved) if flip else moved))
    return tuple(out)


def jitter(image: np.ndarray, rng, gamma=(0.7, 1.4), gain=(0.85, 1.15),
           bias=0.05) -> np.ndarray:
    """Gamma, gain and offset on a [0, 1] image.

    Monotone in the input, so it changes how visible a vessel is without ever
    reordering two pixels -- which is the point. E2 measured that visibility is
    what decides whether a vessel is found, so this is the augmentation aimed
    at the failure this series actually diagnosed.
    """
    adjusted = np.clip(image, 1e-6, 1.0) ** rng.uniform(*gamma)
    adjusted = adjusted * rng.uniform(*gain) + rng.uniform(-bias, bias)
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


def remove_structures(image: np.ndarray, size: int = INPAINT_SIZE) -> np.ndarray:
    """The inpainted image CoLeTra pastes from: thin dark structures filled in.

    A grey closing is a dilation followed by an erosion, so anything darker
    than its surroundings and thinner than the element is swallowed while the
    background is left where it was.
    """
    return ndimage.grey_closing(image, size=size)


def coletra(image: np.ndarray, label: np.ndarray, rng,
            count: int = COLETRA_PATCHES, patch: int = COLETRA_PATCH,
            sigma: float = COLETRA_SIGMA,
            inpainted: np.ndarray | None = None) -> np.ndarray:
    """Blend vessel-free content over random foreground pixels; label untouched.

    Equation (1) of the paper: at each chosen centre c, every pixel p inside the
    patch is replaced by a Gaussian-weighted mix of the inpainted image and the
    original, g(p,c,sigma) * inpainted + (1 - g) * image. The Gaussian is what
    keeps the seam invisible; a hard paste would teach the network to find
    rectangles.
    """
    rows, cols = np.nonzero(label)
    if len(rows) == 0:
        return image
    if inpainted is None:
        inpainted = remove_structures(image)

    out = image.copy()
    half = patch // 2
    grid = np.arange(-half, half + 1)
    weight = np.exp(-0.5 * (grid[:, None] ** 2 + grid[None, :] ** 2)
                    / sigma ** 2)
    for index in rng.choice(len(rows), size=min(count, len(rows)),
                            replace=False):
        row, col = int(rows[index]), int(cols[index])
        top, left = row - half, col - half
        # Clip the window at the border and take the matching slice of the
        # kernel, so a centre near the edge still gets the right weights.
        r0, r1 = max(top, 0), min(top + patch, out.shape[0])
        c0, c1 = max(left, 0), min(left + patch, out.shape[1])
        kernel = weight[r0 - top:r1 - top, c0 - left:c1 - left]
        window = (slice(r0, r1), slice(c0, c1))
        out[window] = (kernel * inpainted[window]
                       + (1 - kernel) * out[window]).astype(np.float32)
    return out


def _checks() -> None:
    rng = np.random.default_rng(0)

    # 1. dihedral must move every plane together.
    image = rng.random((12, 12)).astype(np.float32)
    label = (rng.random((12, 12)) > 0.8).astype(np.float32)
    dist = rng.random((12, 12)).astype(np.float32)
    for trial in range(20):
        local = np.random.default_rng(trial)
        moved = dihedral(local, image, label, dist)
        expected = dihedral(np.random.default_rng(trial), image)[0]
        assert np.array_equal(moved[0], expected), trial
        # The label must be the same symmetry of the label, which is only
        # checkable by applying the transform to a plane we can identify.
        marker = np.arange(144, dtype=np.float32).reshape(12, 12)
        a, b = dihedral(np.random.default_rng(trial), marker, marker)
        assert np.array_equal(a, b), trial
    print("dihedral applies one symmetry to every plane (20 draws)")

    # 2. jitter stays in range and never reorders two pixels.
    ramp = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
    for trial in range(50):
        out = jitter(ramp, np.random.default_rng(trial))
        assert out.min() >= 0.0 and out.max() <= 1.0, (out.min(), out.max())
        flat = out.ravel()
        assert np.all(np.diff(flat) >= -1e-6), "jitter reordered pixels"
    print("jitter stays in [0,1] and is monotone (50 draws)")

    # 3. the inpainter must actually erase a thin dark line.
    canvas = np.full((40, 40), 0.8, dtype=np.float32)
    canvas[20, 5:35] = 0.2
    filled = remove_structures(canvas)
    print(f"a dark line at 0.20 on a 0.80 field becomes "
          f"{filled[20, 20]:.2f} after closing")
    assert filled[20, 20] > 0.75, filled[20, 20]

    # 4. CoLeTra's whole premise: the image changes, the label does not.
    vessel = np.zeros((48, 48), dtype=np.float32)
    vessel[:] = 0.8
    vessel[22:26, 4:44] = 0.2
    mask = np.zeros((48, 48), dtype=np.float32)
    mask[22:26, 4:44] = 1.0
    before = mask.copy()
    out = coletra(vessel, mask, np.random.default_rng(3))
    assert np.array_equal(mask, before), "coletra modified the label"
    changed = np.abs(out - vessel) > 1e-3
    assert changed.any(), "coletra changed nothing"
    print(f"coletra altered {changed.sum()} of {vessel.size} pixels "
          f"and left the label byte-identical")

    # 5. the alteration must make the vessel look BROKEN, i.e. brighter where
    #    it was dark. If it darkened instead, the augmentation would be
    #    teaching the opposite lesson.
    on_vessel = mask > 0
    lifted = (out - vessel)[on_vessel & changed]
    print(f"on-vessel change: min {lifted.min():+.3f}, "
          f"mean {lifted.mean():+.3f}, max {lifted.max():+.3f}")
    assert lifted.mean() > 0.05, lifted.mean()

    # 6. coverage, so the hyper-parameter is measured rather than asserted.
    #    The paper aims for 40-60% of the structure covered.
    covered = []
    for trial in range(40):
        out = coletra(vessel, mask, np.random.default_rng(trial))
        touched = (np.abs(out - vessel) > 0.02) & on_vessel
        covered.append(touched.sum() / on_vessel.sum())
    share = 100 * float(np.mean(covered))
    print(f"coletra covers {share:.1f}% of the structure "
          f"(paper's band is 40-60%)")
    assert 25.0 < share < 75.0, share

    # 7. The wiring, which is where the dangerous bug would live: if the batch
    #    sampler applied the symmetry to the image but not the label, the model
    #    would train on mirrored targets and no loss would ever complain.
    #    Feeding an image that IS the label makes the misalignment exact.
    import train  # local: train imports this module at load time
    marker = rng.random((80, 80)).astype(np.float32)
    data = {"images": marker[None], "labels": marker[None],
            "dists": marker[None], "fovs": np.ones((1, 80, 80), bool)}
    for name, augments in train.AUGMENTS.items():
        geometry = tuple(a for a in augments if a != "jitter")
        images, labels, _ = train.sample_batch(
            data, np.random.default_rng(1), 0.0, 1.0, geometry,
            inpainted=marker[None] if "coletra" in geometry else None)
        assert np.allclose(images.cpu().numpy(), labels.cpu().numpy(),
                       atol=1e-5), name
        print(f"  {name}: image and label stay aligned through {geometry}")

    # And the augmentation must actually change something, or the arm is a
    # relabelled copy of the baseline.
    plain, _, _ = train.sample_batch(data, np.random.default_rng(1), 0.0, 1.0)
    moved, _, _ = train.sample_batch(data, np.random.default_rng(1), 0.0, 1.0,
                                     ("dihedral", "jitter"))
    assert not np.allclose(plain.cpu().numpy(), moved.cpu().numpy()), \
        "augmentation is a no-op"
    print("  augmented batches differ from unaugmented ones")
    print("all checks passed")


if __name__ == "__main__":
    _checks()
