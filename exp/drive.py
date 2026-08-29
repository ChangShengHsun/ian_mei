"""DRIVE loading shared by the classical and learned baselines.

This HuggingFace copy ships no *_mask.gif, so the FOV (field of view -- the
circular region the camera actually imaged) is derived from the image itself.
Evaluating outside the FOV inflates every metric, so nothing is scored there.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent / "data" / "DRIVE"


def fov_mask(rgb: np.ndarray) -> np.ndarray:
    """Circular camera aperture: bright pixels, filled, then grown by 2 px.

    Deliberately grown rather than eroded. The annotators traced vessels right
    up to the aperture rim, so the intensity threshold alone cuts off labelled
    vessel pixels (measured over all 40 images: 26 lost at 0 px of dilation,
    3 at 2 px). Eroding instead -- the usual reflex, to avoid the rim's
    intensity ramp -- costs 140 labelled pixels per image.
    """
    aperture = ndimage.binary_fill_holes(rgb.sum(-1) > 30)
    return ndimage.binary_dilation(aperture, np.ones((3, 3)), iterations=2)


def green_clahe(rgb: np.ndarray) -> np.ndarray:
    """Green channel + CLAHE, the standard DRIVE preprocessing.

    Green carries the highest vessel/background contrast of the three channels;
    CLAHE (contrast limited adaptive histogram equalisation) removes the slow
    illumination gradient across the retina without blowing up noise.
    """
    import cv2
    green = rgb[..., 1]
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
    return equalised.astype(np.float32) / 255.0


# Five of the twenty DRIVE training images, held out for model selection.
# Every fourth image so the choice cannot follow acquisition order. Fixed by
# id and not by a seed: a selection split that moves between runs is not a
# selection split, it is another source of variance.
DEV_IDS = ("24", "28", "32", "36", "40")

# 'train'/'val' name the two DIRECTORIES DRIVE ships. 'val' is DRIVE's
# official TEST set, so selecting a checkpoint on it leaks the test set into
# the result. Training reads 'fit' and selects on 'dev'; only the final score
# reads 'test'. The two directory names still work because every scoring
# script uses them, and scoring on the test set is exactly right -- the sin
# was ever CHOOSING on it.
_ALIASES = {"fit": "train", "dev": "train", "test": "val"}


def load_split(split: str) -> list[dict]:
    """'fit' (15 imgs), 'dev' (5, for selection), 'test' (20, report only).

    'train' (all 20 of images 21-40) and 'val' (all 20 of images 01-20) are
    the raw directories and stay available for scoring.
    """
    if split not in _ALIASES and split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    directory = _ALIASES.get(split, split)
    suffix = "" if directory == "train" else "_manual1"
    keep = {"fit": lambda stem: stem not in DEV_IDS,
            "dev": lambda stem: stem in DEV_IDS}.get(split, lambda stem: True)
    items = []
    for image_path in sorted((ROOT / directory / "input").glob("*.tif")):
        if not keep(image_path.stem):
            continue
        stem = image_path.stem
        rgb = np.asarray(Image.open(image_path))
        label = np.asarray(
            Image.open(ROOT / directory / "label" / f"{stem}{suffix}.png")) > 127
        items.append({
            "name": stem,
            "rgb": rgb,
            "image": green_clahe(rgb),
            "label": label,
            "fov": fov_mask(rgb),
        })
    return items


if __name__ == "__main__":
    for split in ("train", "val"):
        items = load_split(split)
        first = items[0]
        inside = np.mean([i["label"][i["fov"]].mean() for i in items])
        print(f"{split}: {len(items)} images, shape {first['image'].shape}, "
              f"fov {first['fov'].mean():.3f}, vessel-inside-fov {inside:.3f}")
        assert len(items) == 20
        assert first["image"].shape == first["label"].shape == first["fov"].shape
        # A wrong FOV shows up as either leaked labels or an implausible area.
        for item in items:
            assert (item["label"] & ~item["fov"]).sum() < 10, item["name"]
            assert 0.70 < item["fov"].mean() < 0.78, item["name"]
        assert 0.10 < inside < 0.14, inside

    # The selection split. What must hold is not "fit has 15 rows" -- that
    # passes for any 15 -- but that fit and dev PARTITION the training
    # directory and that neither can ever see a test image.
    names = {s: [i["name"] for i in load_split(s)] for s in
             ("fit", "dev", "test", "train", "val")}
    for split in ("fit", "dev", "test"):
        print(f"{split}: {len(names[split])} images "
              f"[{', '.join(names[split][:6])}{'...' if len(names[split]) > 6 else ''}]")
    assert set(names["dev"]) == set(DEV_IDS), names["dev"]
    assert not set(names["fit"]) & set(names["dev"])
    assert sorted(names["fit"] + names["dev"]) == sorted(names["train"])
    assert names["test"] == names["val"]
    # The whole point: no image the model is fitted or selected on may appear
    # in the reported score.
    assert not (set(names["fit"]) | set(names["dev"])) & set(names["test"])
    assert load_split("dev")[0]["image"].shape == (584, 565)
    for bad in ("valid", "Dev", ""):
        try:
            load_split(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not have loaded")
    print("all checks passed")
