"""D1.a: how well does the tangent head predict the axis?

One row per (run, image): the mean axis gap on ground-truth vessel pixels,
for the head and for two references that involve no learning at all.

  head       the network's own field.
  constant   one fixed axis everywhere, the one that fits this image best.
             A head that does not beat this did not learn direction; it is
             the floor, and it is not zero, because vessels in a retina are
             not uniformly distributed in angle.
  classical  the tangent read straight off the RAW IMAGE by the same
             structure tensor that builds the target -- a Hessian filter,
             no training, no labels at inference. This is the real bar. D1's
             premise is that a network can infer direction where the image
             does not show it; a head that only matches this has learned to
             recompute a filter that costs three gaussian blurs.

The gap is measured in double-angle chord units: 0 when the axes agree, 2
when they are a quarter turn apart. Averaged over vessel pixels weighted by
the ground truth's coherence, so junctions -- which have no single tangent --
do not dominate a score about how well tangents are predicted.

  python exp/score_direction.py --selftest
  python exp/score_direction.py

Writes results/selection_sweep/direction_quality.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction
import drive
import select_checkpoint as rules_module
import summarize_selection as selection
import train

OUT = selection.SWEEP / "direction_quality.csv"
RULE = "(iv) best clDice"


def weighted_gap(sin_pred, cos_pred, sin_gt, cos_gt, weight) -> float:
    """Coherence-weighted mean axis gap, over the pixels weight covers."""
    total = weight.sum()
    if total <= 0:
        return float("nan")
    gap = direction.axis_gap(sin_pred, cos_pred, sin_gt, cos_gt)
    return float((gap * weight).sum() / total)


def normalise(sin_raw, cos_raw):
    """Project a free 2-vector onto the unit circle, where the axes live."""
    length = np.hypot(sin_raw, cos_raw)
    safe = np.where(length > 1e-8, length, 1.0)
    return sin_raw / safe, cos_raw / safe


def best_constant(sin_gt, cos_gt, weight) -> tuple:
    """The single axis that fits this image best, as full planes.

    The weighted mean of the double-angle vectors, renormalised: at double
    angle the circular mean IS the arithmetic mean of the unit vectors, which
    is the whole reason this representation was chosen.
    """
    total = weight.sum()
    if total <= 0:
        return np.zeros_like(sin_gt), np.ones_like(cos_gt)
    mean_sin = (sin_gt * weight).sum() / total
    mean_cos = (cos_gt * weight).sum() / total
    length = float(np.hypot(mean_sin, mean_cos))
    if length < 1e-8:
        return np.zeros_like(sin_gt), np.ones_like(cos_gt)
    return (np.full_like(sin_gt, mean_sin / length),
            np.full_like(cos_gt, mean_cos / length))


@torch.no_grad()
def predict_field(model, image: np.ndarray, mean, std) -> tuple:
    """The head's (sin 2theta, cos 2theta) over a whole image."""
    model.eval()
    height, width = image.shape
    stride = 2 ** (getattr(model, "depth", 3) - 1)
    tensor = torch.from_numpy(((image[None] - mean) / std).astype(np.float32))
    padded = torch.nn.functional.pad(
        tensor[None], (0, (-width) % stride, 0, (-height) % stride),
        mode="reflect")
    _, field = model.forward_direction(padded.to(train.DEVICE))
    got = field[0, :, :height, :width].cpu().numpy()
    return normalise(got[0], got[1])


def selftest() -> None:
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    label = np.abs(xx - yy) <= 1
    sin_gt, cos_gt, weight = direction.tangent_field(label)
    on = weight * label

    # A perfect prediction scores 0; a quarter turn off scores 2.
    assert weighted_gap(sin_gt, cos_gt, sin_gt, cos_gt, on) < 1e-6
    turned = weighted_gap(-sin_gt, -cos_gt, sin_gt, cos_gt, on)
    print(f"perfect field 0.000, a quarter turn off {turned:.3f} of 2.0")
    assert turned > 1.9, turned

    # The constant reference is the best SINGLE axis, so on an image with one
    # straight vessel it is nearly perfect -- which is exactly why it is the
    # floor a head has to clear rather than a straw man.
    c_sin, c_cos = best_constant(sin_gt, cos_gt, on)
    single = weighted_gap(c_sin, c_cos, sin_gt, cos_gt, on)
    print(f"best constant axis on ONE straight vessel: {single:.3f} -- near "
          f"perfect, as it should be")
    assert single < 0.05, single

    # On two vessels at right angles no single axis fits, so the floor rises.
    both = label | (np.abs((xx + yy) - size) <= 1)
    s2, c2, w2 = direction.tangent_field(both)
    on2 = w2 * both
    c_sin, c_cos = best_constant(s2, c2, on2)
    crossed = weighted_gap(c_sin, c_cos, s2, c2, on2)
    print(f"  on two vessels at right angles: {crossed:.3f} -- the floor "
          f"depends on the image, which is why it is measured per image")
    assert crossed > 0.5, crossed

    # normalise must put a free vector on the unit circle without changing
    # its direction: the head's output is not constrained to unit length.
    s, c = normalise(np.array([3.0, 0.0]), np.array([4.0, 0.0]))
    assert abs(np.hypot(s[0], c[0]) - 1.0) < 1e-6, (s, c)
    assert abs(s[0] - 0.6) < 1e-6 and abs(c[0] - 0.8) < 1e-6, (s, c)
    assert np.isfinite(s[1]) and np.isfinite(c[1]), "a zero vector must not "\
        "produce NaN -- an untrained head outputs zeros"
    print("normalise projects onto the unit circle and survives a zero vector")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    points = selection.selection_points(selection.load())
    rule = dict(rules_module.rules())[RULE]
    runs = sorted(run for run in points
                  if train.uses_direction(run.rsplit("_s", 1)[0]))
    if not runs:
        raise SystemExit("no _dir runs in checkpoint_scores.csv yet")

    items = drive.load_split("val")
    data = train.stack_split("train")
    truth = [direction.tangent_field(item["label"] & item["fov"])
             for item in items]
    print(f"{len(runs)} _dir run(s), {len(items)} images", flush=True)

    rows = []
    for run in runs:
        config = run.rsplit("_s", 1)[0]
        epoch = rule(points[run])["epoch"]
        state = train.load_checkpoint(
            selection.SWEEP / run / f"epoch{epoch:03d}.pt")
        model = train.build_model(config)
        model.load_state_dict(state["model"])
        mean, std = train.normalisation(run, data)
        for item, (sin_gt, cos_gt, coherence) in zip(items, truth):
            on = coherence * (item["label"] & item["fov"])
            sin_head, cos_head = predict_field(model, item["image"], mean, std)
            c_sin, c_cos = best_constant(sin_gt, cos_gt, on)
            # The classical reference reads the RAW IMAGE, not the label:
            # inverted, because a vessel is dark on a fundus and the target
            # is built from a bright mask.
            k_sin, k_cos, _ = direction.tangent_field(1.0 - item["image"])
            rows.append({
                "run": run, "config": config, "seed": run.rsplit("_s", 1)[1],
                "epoch": epoch, "image": item["name"],
                "head": round(weighted_gap(sin_head, cos_head, sin_gt, cos_gt,
                                           on), 4),
                "constant": round(weighted_gap(c_sin, c_cos, sin_gt, cos_gt,
                                               on), 4),
                "classical": round(weighted_gap(k_sin, k_cos, sin_gt, cos_gt,
                                                on), 4)})
        print(f"  {run} epoch {epoch} done", flush=True)

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
