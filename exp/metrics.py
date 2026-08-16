"""The four metrics from 3_datasets_metrics.md section 3, on binary masks.

Dice measures pixel overlap and is nearly blind to a vessel snapping in two.
clDice, Betti-0 error and 95HD are the three that are not.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import euler_number
from skimage.morphology import skeletonize

_CONN8 = np.ones((3, 3), dtype=bool)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * (pred & gt).sum() / total)


def cl_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Overlap of each mask's 1-px-wide skeleton with the other mask.

    A break in a vessel deletes skeleton on both sides of the gap, so this
    reacts to disconnection where Dice loses a fraction of a percent.
    """
    skel_pred, skel_gt = skeletonize(pred), skeletonize(gt)
    precision = (skel_pred & gt).sum() / skel_pred.sum() if skel_pred.any() else 0.0
    sensitivity = (skel_gt & pred).sum() / skel_gt.sum() if skel_gt.any() else 0.0
    denominator = precision + sensitivity
    return 0.0 if denominator == 0 else float(2 * precision * sensitivity / denominator)


def betti(mask: np.ndarray) -> tuple[int, int]:
    """(b0, b1) = (connected components, holes), 8-connected foreground."""
    components = ndimage.label(mask, structure=_CONN8)[1]
    return components, int(components - euler_number(mask, connectivity=2))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile of the symmetric boundary-to-boundary distances."""
    pred_border = np.argwhere(pred & ~ndimage.binary_erosion(pred))
    gt_border = np.argwhere(gt & ~ndimage.binary_erosion(gt))
    if len(pred_border) == 0 or len(gt_border) == 0:
        return float("nan")
    forward = cKDTree(gt_border).query(pred_border)[0]
    backward = cKDTree(pred_border).query(gt_border)[0]
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def evaluate(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray,
             threshold: float = 0.5) -> dict:
    """Score one image. Everything outside the FOV is forced to background."""
    pred = (prob >= threshold) & fov
    gt = gt & fov
    b0_pred, b1_pred = betti(pred)
    b0_gt, b1_gt = betti(gt)
    return {
        "dice": dice(pred, gt),
        "cldice": cl_dice(pred, gt),
        "betti0_err": abs(b0_pred - b0_gt),
        "betti1_err": abs(b1_pred - b1_gt),
        "b0_pred": b0_pred,
        "b0_gt": b0_gt,
        "hd95": hd95(pred, gt),
    }


if __name__ == "__main__":
    # The whole argument of the survey, as a runnable check: take a clean
    # vessel and punch a 3-px gap in it. Dice barely notices, the topology
    # metrics do.
    whole = np.zeros((64, 64), dtype=bool)
    whole[30:33, 5:60] = True
    broken = whole.copy()
    broken[30:33, 30:33] = False

    d, c = dice(broken, whole), cl_dice(broken, whole)
    print(f"one 3-px gap in a 55-px vessel:  dice {d:.4f}  cldice {c:.4f}")
    print(f"betti0 whole {betti(whole)[0]} -> broken {betti(broken)[0]}")
    assert d > 0.94, "Dice should stay high -- that is the point"
    assert c < d, "clDice must punish the break harder than Dice"
    assert betti(broken)[0] == 2 and betti(whole)[0] == 1

    ring = np.zeros((64, 64), dtype=bool)
    ring[10:50, 10:50] = True
    ring[20:40, 20:40] = False
    assert betti(ring) == (1, 1), betti(ring)
    assert hd95(whole, whole) == 0.0
    print("metrics self-check passed")
