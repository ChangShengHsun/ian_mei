"""Download HRF (High-Resolution Fundus) for the E4 cross-dataset check.

45 images at 3504x2336, six times DRIVE's linear resolution, with vessel
ground truth and field-of-view masks supplied by the authors. That resolution
is the point: E4 asks whether "drop components under 20 px" beats a topology
loss on datasets other than DRIVE, and 20 px means something completely
different when a vessel is six times wider. HRF is the dataset that forces the
threshold to be expressed in vessel widths rather than in pixels.

  python exp/fetch_hrf.py

Writes data/HRF/{images,manual1,mask}. ~73 MB, one zip.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "HRF"
URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"
# The authors ship 15 healthy + 15 glaucomatous + 15 diabetic-retinopathy
# images, each with one manual segmentation and one aperture mask.
EXPECTED = {"images": 45, "manual1": 45, "mask": 45}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "all.zip"
    if not archive.exists():
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    print(f"{archive.name}: {archive.stat().st_size:,} bytes")

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(ROOT)

    problems = []
    for folder, count in EXPECTED.items():
        found = len(list((ROOT / folder).glob("*"))) if (ROOT / folder).is_dir() else 0
        print(f"  {folder:10} {found:3d} files (expected {count})")
        if found != count:
            problems.append(folder)
    if problems:
        # Loud rather than silent: a partial extract that looks like a dataset
        # is the failure mode that costs a whole day of training on bad data.
        print(f"\nFAILED: wrong file count in {problems}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nHRF ready at {ROOT}")


if __name__ == "__main__":
    main()
