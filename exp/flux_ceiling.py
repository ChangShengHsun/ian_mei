"""How accurate would a flux head have to be, before it is worth building one?

WRITTEN AND SELFTESTED 2026-09-06, BEFORE ANY FLUX HEAD EXISTS.

WHY. exp/hysteresis.py closed the last square of the post-hoc map: losses,
operators, linking, readout/ensembling and local decision rules are each shut
with a measurement. What is left is changing what the model predicts. The
geometry family in this repo has been tried three ways and only one worked:

    as an auxiliary head (D1)      the tangent head LEARNED the axis (0.243
                                   against a constant 1.205 and classical
                                   0.322) and segmentation did not improve
                                   (+37.9 t 0.37, +59.2 t 0.93, both fail)
    as a post-hoc operator (D-B)   dead, and beaten by its shuffled control
    as a loss weight (D-E = clw)   works -- and is prior art, Skeleton Recall
                                   Loss, ECCV 2024

The untried form is geometry as the OUTPUT: the network predicts a flux field
pointing at the centreline, the centreline is decoded from it, and the mask is
rendered. D1's failure has a diagnosis -- an auxiliary head can be ignored by
the mask head -- and in this form it cannot be, because the output IS the
geometry. Prior art is recorded in stage-report/flux_novelty_check.md
(DeepFlux, Bouix & Siddiqi 2005, VesselPose); this is a port, not an
invention.

WHAT THIS FILE DOES *NOT* TEST. Decoding a PERFECT flux field is trivially
exact: every pixel votes for its own nearest skeleton point, so every skeleton
pixel receives at least its own vote. An oracle test here would return 100%
and prove nothing. That is the trap this file is written to avoid.

WHAT IT TESTS INSTEAD. How much error the representation tolerates. The field
is degraded with Gaussian displacement noise of standard deviation sigma and
decoded, and the question is:

    at what sigma does the decoded tree fall to what the BEST EXISTING ARM
    already achieves?

That number is a requirement on a head that does not exist yet. If a flux head
would have to predict displacement to a fraction of a pixel, it is not worth
building. If a couple of pixels of slop is enough, it is.

THE COMPARISON POINT is `H_aug_clw` on HRF, read at threshold 0.01: ERL 0.708,
measured 2026-09-06. HRF because that is where the room is -- 5.8% of its
centreline sits below p=0.01 and 40.8% of that is in runs of 20 px or more,
whole vessels never seen, against DRIVE's 1.4% and 5.3%.

  python exp/flux_ceiling.py --selftest
  python exp/flux_ceiling.py --report

Writes results/flux_ceiling.txt via --report. Reads no checkpoints.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import drive
import erl
import erl_length

# DeepFlux uses a context band of 7 px around the skeleton on natural images.
# Retinal vessels are 4.00 px wide at the median on DRIVE, STARE and HRF, so
# the band is swept rather than assumed: too narrow and thin vessels get few
# voters, too wide and two vessels compete for the same pixels.
RADII = (3.0, 5.0, 7.0, 10.0)
SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
# H_aug_clw on HRF at threshold 0.01, measured 2026-09-06 by the headroom
# scan. This is the number a flux head has to beat to be worth training.
HRF_BAR = 0.708
DRIVE_BAR = 0.930


def flux_target(skel: np.ndarray, radius: float):
    """(displacement field, band) -- the exact target a flux head would learn.

    For every pixel within `radius` of the skeleton, the displacement to its
    nearest skeleton pixel. This is DeepFlux's representation kept in pixels
    rather than normalised, because the magnitude is what makes the decode a
    vote rather than a search.
    """
    dist, (near_y, near_x) = ndimage.distance_transform_edt(
        ~skel, return_indices=True)
    rows, cols = np.indices(skel.shape)
    return (near_y - rows, near_x - cols), (dist <= radius)


def decode(displacement, band: np.ndarray, shape, votes: int = 1):
    """Recover the centreline: every banded pixel votes for where it points.

    A pixel is called centreline when at least `votes` pixels point at it.
    Rounding is what makes this robust: a displacement wrong by less than half
    a pixel lands on the same target.
    """
    delta_y, delta_x = displacement
    rows, cols = np.indices(shape)
    target_y = np.clip(np.rint(rows + delta_y), 0, shape[0] - 1).astype(int)
    target_x = np.clip(np.rint(cols + delta_x), 0, shape[1] - 1).astype(int)
    flat = np.bincount((target_y * shape[1] + target_x)[band],
                       minlength=shape[0] * shape[1])
    return flat.reshape(shape) >= votes


def corrupt(displacement, band, sigma: float, seed: int):
    """The field a real head would produce: the target plus Gaussian error.

    Isotropic noise on both components, applied only inside the band -- what
    the head predicts outside it is irrelevant to the decode.
    """
    if sigma <= 0:
        return displacement
    rng = np.random.default_rng(seed)
    delta_y, delta_x = displacement
    noise_y = rng.normal(0.0, sigma, delta_y.shape) * band
    noise_x = rng.normal(0.0, sigma, delta_x.shape) * band
    return (delta_y + noise_y, delta_x + noise_x)


def render(centreline: np.ndarray, radius_map: np.ndarray) -> np.ndarray:
    """The mask a centreline-plus-radius output would produce.

    The radius comes from the ground truth here on purpose: this file prices
    the CENTRELINE representation, and giving the radius away isolates it. A
    head that got the centreline right and the radius wrong is a different
    experiment.
    """
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
    # 1. THE TRAP, STATED AS AN ASSERTION. A perfect field must decode
    #    EXACTLY, and that is why the perfect case is not the experiment.
    line = np.zeros((41, 41), dtype=bool)
    line[20, 5:36] = True
    field, band = flux_target(line, 7.0)
    got = decode(field, band, line.shape)
    assert np.array_equal(got, line), "a perfect field did not decode exactly"
    print("a perfect field decodes exactly -- which is why the oracle case "
          "is not the experiment")

    # 2. THE BAND MUST BE THE BAND. Everything within the radius, nothing
    #    beyond it; a band that quietly covers the image would make the vote
    #    count meaningless.
    for radius in RADII:
        _, got_band = flux_target(line, radius)
        rows = np.flatnonzero(got_band[:, 20])
        assert rows.min() == 20 - int(radius) and rows.max() == 20 + int(radius)
    print(f"the context band is exactly the radius, checked at {RADII}")

    # 3. NOISE MUST ACTUALLY DEGRADE IT, and monotonically enough to read.
    #    A decode that survives any sigma is not measuring anything.
    scores = []
    for sigma in (0.0, 1.0, 3.0):
        noisy = corrupt(field, band, sigma, seed=0)
        out = decode(noisy, band, line.shape)
        scores.append(float((out & line).sum()) / float(line.sum()))
    assert scores[0] == 1.0 and scores[-1] < scores[0], scores
    print(f"recall of the true centreline at sigma 0 / 1 / 3: "
          f"{scores[0]:.2f} / {scores[1]:.2f} / {scores[2]:.2f}")

    # 4. NOISE MUST BE REPRODUCIBLE. A curve that moves between processes
    #    cannot be quoted.
    first = decode(corrupt(field, band, 1.0, 7), band, line.shape)
    again = decode(corrupt(field, band, 1.0, 7), band, line.shape)
    assert np.array_equal(first, again)
    print("the corruption is seeded and reproducible")

    # 5. THE RENDERER MUST PRODUCE A VESSEL, not a centreline. A 3 px radius
    #    on a straight line has to come back about 7 px wide.
    radius_map = np.full(line.shape, 3.0)
    mask = render(line, radius_map)
    width = int(mask[:, 20].sum())
    assert 5 <= width <= 9, width
    print(f"rendering a 3 px radius gives a {width} px wide vessel")
    print("all checks passed")


def report() -> None:
    print("=== how accurate would a flux head have to be? ===\n")
    print("A perfect field decodes exactly, so the question is tolerance.")
    print("The field is degraded with Gaussian displacement noise of s.d.")
    print("sigma and decoded; the number to find is the sigma at which the")
    print("decoded tree falls to what the best EXISTING arm already gets.\n")
    print(f"The bars: HRF {HRF_BAR:.3f} and DRIVE {DRIVE_BAR:.3f}, both")
    print("`H_aug_clw`/`H_aug_clw64` read at threshold 0.01 (2026-09-06).")
    print("A flux head only earns its build if it clears the bar at a sigma")
    print("a real network could plausibly reach.\n")

    sets = [("drive", drive.load_split("test")[:4], DRIVE_BAR)]
    for name in ("stare", "hrf"):
        _, test_items = cross_dataset.loader_for(name)()
        sets.append((name, test_items[:4],
                     HRF_BAR if name == "hrf" else None))

    for name, items, bar in sets:
        geometry = []
        for item in items:
            truth = item["label"] & item["fov"]
            skel = skeletonize(truth)
            radius = ndimage.distance_transform_edt(truth)
            geometry.append((skel, radius))
        print(f"--- {name}: {len(items)} images"
              + (f", bar {bar:.3f}" if bar else "") + " ---")
        print(f"    {'radius':>7}" + "".join(f"{s:>8}" for s in SIGMAS))
        # ERL does not punish extra foreground -- link_ceiling.py was caught
        # by exactly this in 2026-08-27, when a closing that painted 31% more
        # foreground beat the oracle and would have killed C1 on an artefact.
        # A noisy decode emits spurious centreline points, the renderer
        # dilates them, and ERL can RISE. So every cell carries its foreground
        # against the true mask, and a cell only counts as clearing the bar if
        # it does so without buying the difference in paint.
        FG_LIMIT = 1.10
        best = {}
        for radius_px in RADII:
            row = []
            for sigma in SIGMAS:
                num = den = paint = truth_paint = 0.0
                for index, (skel, radius_map) in enumerate(geometry):
                    field, band = flux_target(skel, radius_px)
                    noisy = corrupt(field, band, sigma, seed=index)
                    got = decode(noisy, band, skel.shape)
                    mask = render(got, radius_map)
                    num += erl.expected_run_length(skel, mask)
                    den += float(skel.sum())
                    paint += float(mask.sum())
                    truth_paint += float(render(skel, radius_map).sum())
                score, ratio = num / den, paint / truth_paint
                row.append((score, ratio))
                if ratio <= FG_LIMIT:
                    best[sigma] = max(best.get(sigma, 0.0), score)
                else:
                    best.setdefault(sigma, 0.0)
            print(f"    {radius_px:>7.1f}" +
                  "".join(f"{v:>7.3f}{'*' if r > FG_LIMIT else ' '}"
                          for v, r in row))
        print(f"    (* = painted more than {FG_LIMIT:.2f}x the true mask; "
              f"those cells bought ERL with paint and do not count)")
        if bar:
            over = [s for s in SIGMAS if best[s] >= bar]
            edge = max(over) if over else None
            print(f"\n    Best over radii, against the {bar:.3f} bar:")
            print("      " + "  ".join(
                f"s={s}:{'PASS' if best[s] >= bar else 'fail'}"
                for s in SIGMAS))
            if edge is None:
                print("      clears the bar at NO sigma -- the representation "
                      "cannot reach\n      what the current arm already does, "
                      "and is not worth building")
            else:
                print(f"      clears the bar up to sigma = {edge} px.")
                print(f"      A flux head must predict the displacement to "
                      f"within about {edge} px\n      for this to be worth "
                      f"training. Judge that against what a\n      network "
                      f"plausibly achieves before writing the head.")
        print()


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    raise SystemExit("pass --selftest or --report")


if __name__ == "__main__":
    main()
