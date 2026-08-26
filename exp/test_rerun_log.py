"""A rerun must not append to a published log.

The bug this guards against is silent and destructive in the same step: *.pt
is gitignored and the CSVs are not, so retraining to recover a checkpoint hits
54 directories that already hold a complete log.csv from the laptop. train_one
appends, so without rerun_path the file ends up with ten rows from one machine
and ten from another, in one column layout, with nothing marking the seam --
a published result rewritten in place, which CLAUDE.md forbids outright.

Asserts the mechanism, not the output: no training happens here.

  python exp/test_rerun_log.py
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

HEADER = ["epoch", "loss", "dice", "cldice", "betti0_err", "betti1_err",
          "hd95", "minutes"]


def write_log(path: Path, last_epoch: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for epoch in range(train.VAL_EVERY, last_epoch + 1, train.VAL_EVERY):
            writer.writerow([epoch, 0.4, 0.81, 0.82, 90.0, 30.0, 4.3, 1.0])


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        out_dir = Path(raw)

        # 1. Nothing on disk: the run writes the normal name.
        assert train.rerun_path(out_dir, "log.csv").name == "log.csv"
        print("a fresh run writes log.csv")

        # 2. A half-finished log is a RESUME, and resuming has to keep
        #    appending to the same file or the run's history is split in two.
        write_log(out_dir / "log.csv", train.EPOCHS - train.VAL_EVERY)
        assert train.rerun_path(out_dir, "log.csv").name == "log.csv", \
            "an unfinished log must be appended to, not moved aside"
        print(f"a log stopped at epoch {train.EPOCHS - train.VAL_EVERY} is "
              f"resumed in place")

        # 3. A complete log is published. The rerun goes beside it.
        write_log(out_dir / "log.csv", train.EPOCHS)
        assert train.rerun_path(out_dir, "log.csv").name == "log_rerun.csv"
        print("a complete log.csv is left alone; the rerun writes "
              "log_rerun.csv")

        # 4. And the published file is genuinely untouched -- the check above
        #    would pass just as well if rerun_path had truncated it first.
        before = (out_dir / "log.csv").read_bytes()
        train.rerun_path(out_dir, "log.csv")
        assert (out_dir / "log.csv").read_bytes() == before, \
            "rerun_path must not modify the published log"
        rows = list(csv.DictReader((out_dir / "log.csv").open()))
        assert len(rows) == train.EPOCHS // train.VAL_EVERY, len(rows)
        print(f"  and still holds exactly its {len(rows)} original rows")

        # 5. A finished rerun is skipped too, not appended to. A_dice_s0 hit
        #    this for real: a published laptop log AND a finished GPU rerun
        #    were both on disk when a third training was queued, and stopping
        #    at the first rerun name would have merged two runs into one file.
        write_log(out_dir / "log_rerun.csv", train.EPOCHS)
        assert train.rerun_path(out_dir, "log.csv").name == "log_rerun2.csv"
        write_log(out_dir / "log_rerun2.csv", train.EPOCHS)
        assert train.rerun_path(out_dir, "log.csv").name == "log_rerun3.csv"
        print("a finished log_rerun.csv is skipped too, not appended to")

        # An UNFINISHED rerun is still a resume and must be continued.
        (out_dir / "log_rerun2.csv").unlink()
        write_log(out_dir / "log_rerun2.csv", train.EPOCHS - train.VAL_EVERY)
        assert train.rerun_path(out_dir, "log.csv").name == "log_rerun2.csv"
        print("  but an unfinished one is resumed rather than skipped")

        for name in ("log_rerun.csv", "log_rerun2.csv"):
            (out_dir / name).unlink()

        # 6. An empty file with only a header is not a finished run.
        write_log(out_dir / "log.csv", 0)
        assert train.rerun_path(out_dir, "log.csv").name == "log.csv"
        print("a header-only log is treated as unfinished, not as published")

    print("all checks passed")


if __name__ == "__main__":
    main()
