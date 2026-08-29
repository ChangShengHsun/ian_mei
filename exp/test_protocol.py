"""Does --protocol change which images the model is fitted and selected on?

The bug this exists for: exp/drive.py's 'val' split is DRIVE's official TEST
set, and best.pt was chosen by Dice on it. A checkpoint picked as the best of
ten epochs on the set it is then reported on is the maximum of ten draws, not
the model's score. This asserts the mechanism -- which images each protocol
touches -- not any number that comes out of it.

Runtime: ~40 s, CPU only, no GPU and no training.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive
import train


def test_splits_partition() -> None:
    names = {s: [i["name"] for i in drive.load_split(s)]
             for s in ("fit", "dev", "test", "train", "val")}
    assert set(names["fit"]) | set(names["dev"]) == set(names["train"])
    assert not set(names["fit"]) & set(names["dev"])
    assert names["test"] == names["val"]
    print(f"  fit {len(names['fit'])} + dev {len(names['dev'])} "
          f"partition train {len(names['train'])}")


def test_only_legacy_selects_on_test() -> None:
    """The whole point. legacy must be shown to have the defect; heldout not."""
    test = {i["name"] for i in drive.load_split("test")}
    for name, (fit, dev) in train.PROTOCOL_SPLITS.items():
        selects_on = {i["name"] for i in drive.load_split(dev)}
        fits_on = {i["name"] for i in drive.load_split(fit)}
        assert not fits_on & test, f"{name} TRAINS on test images"
        leaks = bool(selects_on & test)
        assert leaks == (name == "legacy"), f"{name}: leak={leaks}"
        print(f"  {name:8s} selects on the test set: {leaks}")


def test_unknown_protocol_refused() -> None:
    out = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "train.py"),
         "--protocol", "nope", "A_dice_s0"], capture_output=True, text=True)
    assert out.returncode != 0, "an unknown protocol trained something"
    assert "must be one of" in out.stdout + out.stderr
    print("  unknown --protocol refused")


def test_directory_refuses_a_second_protocol() -> None:
    """Resuming a legacy ckpt.pt with heldout data fits one run on two sets."""
    root = Path(tempfile.mkdtemp())
    try:
        (root / "A_dice_s0").mkdir()
        (root / "A_dice_s0" / "protocol.txt").write_text("legacy\n")
        out = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "train.py"),
             "--results", str(root), "--protocol", "heldout", "A_dice_s0"],
            capture_output=True, text=True, timeout=900)
        combined = out.stdout + out.stderr
        assert "was trained under" in combined, combined[-2000:]
        # It must also have got far enough to have loaded the heldout data,
        # or the refusal proves nothing about the data path.
        assert "fit on fit (15 images), select on dev (5 images)" in combined
        print("  a run directory refuses a second protocol")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_centreline_weight_comes_from_the_name() -> None:
    """One config name must mean one model; a swept number lives in the name."""
    assert train.centreline_weight("H_aug_clw") == train.CENTRELINE_WEIGHT
    for weight in train.CENTRELINE_WEIGHTS:
        for base in ("A_dice", "H_aug", "K_focal_aug"):
            name = f"{base}_clw{weight}"
            assert name in train.CONFIGS, name
            assert train.centreline_weight(name) == float(weight)
            assert train.uses_centreline_weight(name)
            # The trap E13 paid for: an augmented arm's variant missing from
            # AUGMENTS trains unaugmented and still answers to its name.
            if base != "A_dice":
                assert train.AUGMENTS.get(name) == train.AUGMENTS["H_aug"], name
            assert train.CONFIGS[name][1] == train.CONFIGS[base][1], name
    assert not train.uses_centreline_weight("A_dice")
    try:
        train.centreline_weight("A_dice")
    except train.ShapeNameError:
        pass
    else:
        raise AssertionError("a name with no clw token returned a weight")
    print(f"  {len(train.CENTRELINE_WEIGHTS) * 3} swept arms carry their "
          f"base's loss and augmentation")


def test_one_split_rule_for_every_dataset() -> None:
    """drive.DEV_IDS and cross_dataset.dev_indices must not drift apart.

    A split rule written twice is a split rule that will disagree in one of
    the two places, silently, and the disagreement looks like a result.
    """
    import cross_dataset
    names = [item["name"] for item in drive.load_split("train")]
    held = tuple(names[i] for i in cross_dataset.dev_indices(len(names)))
    assert held == drive.DEV_IDS, (held, drive.DEV_IDS)
    print(f"  DRIVE's DEV_IDS come back out of the shared rule: {held}")

    for name in ("stare", "vessmap", "hrf"):
        train_items, test_items = cross_dataset.loader_for(name)()
        fit, dev = cross_dataset.fit_dev(train_items)
        assert len(dev) >= cross_dataset.DEV_MINIMUM, (name, len(dev))
        assert len(fit) + len(dev) == len(train_items)
        fit_names = {item["name"] for item in fit}
        dev_names = {item["name"] for item in dev}
        test_names = {item["name"] for item in test_items}
        assert not fit_names & dev_names, name
        # The whole point, again: the test half is never fitted or selected on.
        assert not (fit_names | dev_names) & test_names, name
        print(f"  {name:8s} fit {len(fit):3d} / select {len(dev):2d} / "
              f"test {len(test_items):3d}, disjoint")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            print(name)
            function()
    print("all checks passed")
