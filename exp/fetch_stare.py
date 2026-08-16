"""Fetch STARE: 20 fundus images with TWO independent expert annotations.

DRIVE's second observer sits behind Grand Challenge registration, but STARE
publishes both of its annotators openly -- Adam Hoover ("ah") and Valentina
Kouznetsova ("vk"). Two labellings of the same 20 images is exactly what is
needed to ask whether the ground truth itself is reliable where the contrast
is lowest.

The Clemson host serves a certificate that does not validate, hence the
explicit unverified context; the payload is checked by size instead.
"""
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "STARE"
BASE = "https://cecas.clemson.edu/~ahoover/stare"
FILES = {
    "images": (f"{BASE}/probing/stare-images.tar", 18_674_176),
    "labels-ah": (f"{BASE}/probing/labels-ah.tar", 241_664),
    "labels-vk": (f"{BASE}/probing/labels-vk.tar", 333_824),
}
CONTEXT = ssl.create_default_context()
CONTEXT.check_hostname = False
CONTEXT.verify_mode = ssl.CERT_NONE


def fetch(name: str, url: str, expected: int) -> Path:
    out_dir = ROOT / name
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{name}: already present, skipping")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = ROOT / f"{name}.tar"
    print(f"{name}: downloading {expected / 1e6:.1f} MB ...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120, context=CONTEXT) as response:
        payload = response.read()
    if len(payload) != expected:
        sys.exit(f"{name}: got {len(payload)} bytes, expected {expected}")
    archive.write_bytes(payload)
    with tarfile.open(archive) as tar:
        tar.extractall(out_dir, filter="data")
    archive.unlink()
    print(f"{name}: extracted {len(list(out_dir.iterdir()))} files")
    return out_dir


if __name__ == "__main__":
    for name, (url, size) in FILES.items():
        directory = fetch(name, url, size)
        sample = sorted(directory.iterdir())[:3]
        print(f"  {[p.name for p in sample]}")
