"""best.pt must be the best validated epoch, not the last one.

At 31M parameters the fixed 100-epoch protocol stops being neutral: the
un-augmented arm overfits well before epoch 100 while the augmented arm does
not, so scoring only the final epoch measures overfitting as an augmentation
advantage. best.pt is the second protocol, and the property that makes it
worth anything is exactly the one asserted here -- that the weights on disk
belong to the epoch log.csv says was best, and not to whichever epoch
happened to be saved last.

Trains a deliberately tiny run (3 validated epochs, 2 steps each) in a
temporary results directory, so this asserts the mechanism in seconds without
touching exp/results.

  python exp/test_best_checkpoint.py
"""
import csv
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train


def main() -> None:
    data, val = train.stack_split("train"), train.stack_split("val")
    inside = data["images"][data["fovs"]]
    mean, std = float(inside.mean()), float(inside.std())

    original = (train.RESULTS, train.EPOCHS, train.VAL_EVERY,
                train.CKPT_EVERY, train.PATCHES_PER_EPOCH)
    with tempfile.TemporaryDirectory() as raw:
        train.RESULTS = Path(raw)
        train.EPOCHS, train.VAL_EVERY, train.CKPT_EVERY = 3, 1, 1
        train.PATCHES_PER_EPOCH = 2 * train.BATCH
        try:
            train.train_one("A_dice_s0", data, val, mean, std)
            out_dir = train.RESULTS / "A_dice_s0"

            rows = list(csv.DictReader((out_dir / "log.csv").open()))
            assert len(rows) == 3, len(rows)
            best_row = max(rows, key=lambda row: float(row["dice"]))
            print(f"log.csv dice by epoch: "
                  f"{[(r['epoch'], r['dice']) for r in rows]}")

            assert (out_dir / "best.pt").exists(), "best.pt was never written"
            best = train.load_checkpoint(out_dir / "best.pt")
            assert best["epoch"] == int(best_row["epoch"]), (
                best["epoch"], best_row["epoch"])
            assert abs(best["dice"] - float(best_row["dice"])) < 5e-5, (
                best["dice"], best_row["dice"])
            print(f"best.pt holds epoch {best['epoch']} "
                  f"(dice {best['dice']:.4f}), which is log.csv's argmax")

            # The two protocols must be separable, so the final checkpoint has
            # to survive alongside the best one rather than be overwritten by
            # it. When the last epoch IS the best they agree, and that is a
            # real outcome, not a failure -- assert both files, not that they
            # differ.
            final = train.load_checkpoint(out_dir / "final.pt")
            assert final["epoch"] == train.EPOCHS, final["epoch"]
            print(f"final.pt still holds epoch {final['epoch']} separately")

            # A FIRST-TIME run must write val_final.csv, the name summarize.py
            # opens directly. Deciding that name after the last epoch asks
            # rerun_path about a log this run has just completed, so the run
            # classifies itself as a repeat of itself and val_final.csv is
            # never created. Caught in the queue, after one run had already
            # written the wrong name.
            assert (out_dir / "val_final.csv").exists(), \
                "a first-time run must write val_final.csv, not _rerun"
            assert not (out_dir / "val_final_rerun.csv").exists(), \
                "a first-time run must not write val_final_rerun.csv"
            assert not (out_dir / "log_rerun.csv").exists(), \
                "a first-time run must not write log_rerun.csv"
            print("a first-time run wrote val_final.csv and log.csv, "
                  "neither of them under a _rerun name")

            # A resume must not forget the standing best, or a later WORSE
            # epoch clears a stale bar and overwrites a better best.pt.
            # The value has to come from best.pt itself: ckpt.pt is written
            # before the epoch's validation, so anything recorded there is one
            # validation stale. This assertion failed on the first run of the
            # mechanism, which is how that ordering bug was found.
            assert abs(train.standing_best(out_dir) - best["dice"]) < 5e-5, \
                train.standing_best(out_dir)
            state = train.load_checkpoint(out_dir / "ckpt.pt")
            assert "best_dice" not in state, \
                "ckpt.pt must not carry a value that can drift from best.pt"
            print("a resume reads the standing best from best.pt, not ckpt.pt")

            # And with no best.pt there is nothing to regress from.
            assert train.standing_best(train.RESULTS / "no_such_run") == -1.0

            same = torch.equal(
                best["model"]["head.weight"], final["model"]["head.weight"])
            print(f"  (best and final weights identical: {same} -- true only "
                  f"when the last epoch was also the best)")
        finally:
            (train.RESULTS, train.EPOCHS, train.VAL_EVERY,
             train.CKPT_EVERY, train.PATCHES_PER_EPOCH) = original

    # --keep-epochs must keep EVERY validated epoch, each carrying the
    # metrics that any later selection rule could be built from. A rule that
    # needs betti0_err and finds only dice would have to re-score 240
    # checkpoints to get it, so the numbers travel with the weights.
    original = (train.RESULTS, train.EPOCHS, train.VAL_EVERY,
                train.CKPT_EVERY, train.PATCHES_PER_EPOCH, train.KEEP_EPOCHS)
    with tempfile.TemporaryDirectory() as raw:
        train.RESULTS = Path(raw)
        train.EPOCHS, train.VAL_EVERY, train.CKPT_EVERY = 3, 1, 1
        train.PATCHES_PER_EPOCH = 2 * train.BATCH
        train.KEEP_EPOCHS = True
        try:
            train.train_one("A_dice_s0", data, val, mean, std)
            out_dir = train.RESULTS / "A_dice_s0"
            kept = sorted(p.name for p in out_dir.glob("epoch*.pt"))
            assert kept == ["epoch001.pt", "epoch002.pt", "epoch003.pt"], kept
            for name in kept:
                state = train.load_checkpoint(out_dir / name)
                for key in ("epoch", "dice", "betti0_err", "cldice"):
                    assert key in state, (name, key, list(state))
            rows = list(csv.DictReader((out_dir / "log.csv").open()))
            for row, name in zip(rows, kept):
                state = train.load_checkpoint(out_dir / name)
                assert state["epoch"] == int(row["epoch"])
                assert abs(state["dice"] - float(row["dice"])) < 5e-5
                assert abs(state["betti0_err"]
                           - float(row["betti0_err"])) < 5e-3
            print(f"--keep-epochs kept {len(kept)} checkpoints, each carrying "
                  f"the same dice/betti0 the log recorded for that epoch")
            assert (out_dir / "best.pt").exists() and (out_dir / "final.pt").exists()
            print("  and best.pt / final.pt are still written alongside them")
        finally:
            (train.RESULTS, train.EPOCHS, train.VAL_EVERY, train.CKPT_EVERY,
             train.PATCHES_PER_EPOCH, train.KEEP_EPOCHS) = original

    print("all checks passed")


if __name__ == "__main__":
    main()
