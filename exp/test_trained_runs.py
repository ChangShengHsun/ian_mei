"""The enumeration bug that produced a confident, wrong verdict.

E12 trained seeds 3-5. stratify.py and break_lengths.py each wrote the seed
range out as `(0, 1, 2)`, so they scored the old three, and
summarize_confidence.py printed a full verdict from them -- which then released
the queue gate for the next experiment. Nothing errored. The file existed, it
was recent, and it was wrong.

The check therefore asserts the property that failed: a run directory that
exists on disk is returned, whatever its seed number.

  python exp/test_trained_runs.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train


def main() -> None:
    real = train.trained_runs()
    assert real, "no finished runs on disk -- cannot check enumeration"
    print(f"{len(real)} finished runs on disk")

    for seed in ("3", "4", "5"):
        present = [name for name in real if name.endswith(f"_s{seed}")]
        print(f"  seed {seed}: {present or 'none trained'}")

    with tempfile.TemporaryDirectory() as tmp:
        original = train.RESULTS
        try:
            train.RESULTS = Path(tmp)
            # A seed well outside any range a script might hardcode, and a
            # half-finished run beside it: only the one with final.pt counts,
            # because scoring a checkpoint that is still being written is the
            # other way to get a wrong number quietly.
            for run, files in (("Z_new_s9", ("final.pt",)),
                               ("Z_new_s7", ("ckpt.pt",))):
                (Path(tmp) / run).mkdir()
                for name in files:
                    (Path(tmp) / run / name).write_bytes(b"")
            got = train.trained_runs()
        finally:
            train.RESULTS = original
    assert got == ["Z_new_s9"], got
    print("a seed-9 run is found; a run with only ckpt.pt is not:", got)
    print("all checks passed")


if __name__ == "__main__":
    main()
