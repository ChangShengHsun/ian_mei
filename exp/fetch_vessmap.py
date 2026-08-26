"""Fetch VessMAP: 100 fluorescence-microscopy images of mouse cortex vessels.

Task D2. Every cross-dataset check in this series so far has been a change of
CAMERA -- DRIVE to STARE to HRF are all colour fundus photography. VessMAP is
a change of MODALITY, and it was built for exactly the question E17 asks: its
own paper reports a performance drop of 0.13 on DRIVE against 0.55 on VessMAP.
If augmentation's 4-6x transfer advantage survives here, that finding stops
being "it survives a different camera" and becomes "it survives a different
imaging physics".

Source: Zenodo record 10045265, "VessMAP - Feature-Mapped Cortex Vasculature
Dataset", CC-BY-4.0 (verified 2026-08-26). 4.4 MB.
Paper: Comin et al., PLOS One 2025, PMC12112280.

  python exp/fetch_vessmap.py

Writes data/VessMAP/. 100 images at 256x256 with binary masks.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "VessMAP"
URL = "https://zenodo.org/records/10045265/files/VessMAP.zip?download=1"
EXPECTED_MB = 4.4


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "VessMAP.zip"
    if not archive.exists():
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    size_mb = archive.stat().st_size / 1e6
    print(f"{archive.name}: {size_mb:.1f} MB (expected ~{EXPECTED_MB})")
    if size_mb < 1.0:
        raise SystemExit("download is too small to be the dataset; check the "
                         "URL rather than unzipping a error page")

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(ROOT)
    # Report the layout rather than assuming one: the loader has to be written
    # against what is actually in the archive, and guessing that is how a
    # fetch script silently produces an empty split.
    folders = sorted(p for p in ROOT.rglob("*") if p.is_dir())
    for folder in folders[:10]:
        count = len([p for p in folder.iterdir() if p.is_file()])
        if count:
            suffixes = sorted({p.suffix for p in folder.iterdir()
                               if p.is_file()})
            print(f"  {folder.relative_to(ROOT)}: {count} files {suffixes}")
    print(f"\nVessMAP ready at {ROOT}")


if __name__ == "__main__":
    main()
