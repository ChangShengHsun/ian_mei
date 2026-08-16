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


def load_split(split: str) -> list[dict]:
    """split is 'train' (images 21-40) or 'val' (images 01-20)."""
    suffix = "" if split == "train" else "_manual1"
    items = []
    for image_path in sorted((ROOT / split / "input").glob("*.tif")):
        stem = image_path.stem
        rgb = np.asarray(Image.open(image_path))
        label = np.asarray(
            Image.open(ROOT / split / "label" / f"{stem}{suffix}.png")) > 127
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
