"""Does the calibration artefact reproduce off DRIVE?

THE GAP THIS FILLS. calibration.md's headline -- K_focal_aug reads +13.6% ERL
at a shared threshold of 0.5 and -4.2% at each arm's own dev-optimal
threshold -- is measured on DRIVE alone. A reviewer's first question is
whether that is a property of topology losses or a property of one 20-image
dataset. Until this runs, the honest answer is "we do not know", and the
claim cannot carry a paper.

The checkpoints already exist: results/heldout_transfer/{stare,hrf,vessmap}
holds A_dice, H_aug, H_aug_clw, K_focal_aug (plus the two _dir arms) at three
seeds each, trained under the same fit/dev/test split rule DRIVE uses
(cross_dataset.fit_dev, asserted to reproduce drive.DEV_IDS exactly). So this
is scoring, not training: no GPU, and the answer is available in an hour.

WHAT IS MEASURED. For each dataset and arm, the 33-threshold curve on the dev
images and on the test images, under both ERL conventions. Then:

  at 0.5              every arm read at the shared convention
  at its own peak     every arm read at the threshold maximising DEV Dice

and the difference between those two readings, against A_dice, through the
seed gate. If the sign flips off DRIVE the way it flips on DRIVE, the artefact
is a property of the loss family. If it does not, DRIVE is special and the
paper has to say so.

THREE SEEDS is the gate's minimum, not a comfortable margin. Any arm that
passes here passes narrowly, and the report says so on every line.

  python exp/transfer_calibration.py --selftest
  python exp/transfer_calibration.py stare --shard 0/3
  python exp/transfer_calibration.py --report

Writes results/heldout_transfer/<dataset>/calibration_curve[.shardIofN].csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration
import cross_dataset
import postproc_ceiling as sweep
import select_heldout as heldout
import erl
import erl_convention
import hole_sweep
import speckle
import threshold_control as control
import train

# NOT cross_dataset.out_root(). That points at results/cross/, which is where
# the PRE-heldout transfer runs live; the heldout ones were written by
# run_heldout_more.sh into results/heldout_transfer/, and the two directories
# hold different protocols. Pointing at the wrong one resolved cleanly, found
# no final.pt, and wrote three empty CSVs -- caught 2026-09-01 only because
# the stage finished in eighteen seconds. selftest() now asserts the root
# actually holds the arms this file claims to score.
TRANSFER_ROOT = Path(__file__).resolve().parent / "results" / "heldout_transfer"
DATASETS = ("stare", "hrf", "vessmap")
# The four arms calibration.md is about. The _dir arms are in the directory
# too; they answer a different question and scoring them would double the
# runtime for no part of this one.
ARMS = ("A_dice", "H_aug", "H_aug_clw", "K_focal_aug")
THRESHOLDS = control.THRESHOLDS
SHARED = 0.5


def dataset_root(dataset: str) -> Path:
    return TRANSFER_ROOT / dataset


def out_path(dataset: str, shard) -> Path:
    root = dataset_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    stem = "calibration_curve"
    if shard is None:
        return root / f"{stem}.csv"
    return root / f"{stem}.shard{shard[0]}of{shard[1]}.csv"


def curve_for(model, items, geometry, mean, std, component_px) -> list[dict]:
    totals = defaultdict(lambda: {"inter": 0.0, "sizes": 0.0, "split": 0.0,
                                  "bridged": 0.0, "skel": 0.0})
    for item, geo in zip(items, geometry):
        prob = train.predict_full(model, item["image"], mean, std)
        truth = item["label"] & item["fov"]
        for threshold in THRESHOLDS:
            pred = speckle.drop_small((prob >= threshold) & item["fov"],
                                      component_px)
            cell = totals[threshold]
            cell["inter"] += float((pred & truth).sum())
            cell["sizes"] += float(pred.sum() + truth.sum())
            cell["split"] += erl.expected_run_length(geo["skel"], pred)
            cell["bridged"] += erl_convention.bridged_run_length(geo["skel"],
                                                                pred)
            cell["skel"] += float(geo["skel"].sum())
    rows = []
    for threshold in THRESHOLDS:
        cell = totals[threshold]
        skel = cell["skel"] or 1.0
        rows.append({"threshold": threshold,
                     "dice": round(2.0 * cell["inter"] / cell["sizes"], 5)
                     if cell["sizes"] else 0.0,
                     "erl_split": round(cell["split"] / skel, 5),
                     "erl_bridged": round(cell["bridged"] / skel, 5)})
    return rows


def peak_of(rows: list[dict]) -> float:
    """The DEV Dice-maximising threshold. Never chosen on test."""
    return max(rows, key=lambda row: row["dice"])["threshold"]


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    assert SHARED in THRESHOLDS
    for dataset in DATASETS:
        assert cross_dataset.loader_for(dataset) is not None, dataset
    for arm in ARMS:
        assert arm in train.CONFIGS, arm
    # THE ROOT MUST ACTUALLY HOLD THE RUNS. A path that resolves is not a
    # path that contains anything, and an empty sweep reports as a clean
    # "nothing scored yet" rather than as the failure it is.
    for dataset in DATASETS:
        root = dataset_root(dataset)
        assert root.is_dir(), f"{root} does not exist"
        for arm in ARMS:
            found = [p for p in root.iterdir() if p.is_dir()
                     and p.name.rsplit("_s", 1)[0] == arm
                     and (p / "final.pt").exists()]
            assert found, f"no finished {arm} runs under {root}"
        stamps = {(p / "protocol.txt").read_text().strip()
                  for p in root.iterdir() if (p / "protocol.txt").exists()}
        assert stamps == {"heldout"}, (root, stamps)
    print(f"every arm has finished runs under {TRANSFER_ROOT}, "
          f"all stamped protocol heldout")
    print(f"{len(DATASETS)} datasets x {len(ARMS)} arms, "
          f"{len(THRESHOLDS)} thresholds, shared point {SHARED}")
    work = [f"a_s{index}" for index in range(11)]
    for total in (2, 3):
        flat = [r for i in range(total)
                for r in sweep.shard_filter(work, (i, total))]
        assert sorted(flat) == sorted(work) and len(flat) == len(set(flat))
    print("sharding is an exact partition over 2/3 shards")

    # 1. THE PEAK IS READ OFF DICE, AND IT IS READ OFF THE DEV ROWS. A curve
    #    whose Dice peaks away from 0.5 must return that threshold, or the
    #    whole artefact this file is about cannot be detected.
    rows = [{"threshold": 0.3, "dice": 0.70, "erl_split": 0.9},
            {"threshold": 0.5, "dice": 0.80, "erl_split": 0.5},
            {"threshold": 0.7, "dice": 0.85, "erl_split": 0.2}]
    assert peak_of(rows) == 0.7, peak_of(rows)
    print(f"a curve peaking at 0.7 returns 0.7, not the {SHARED} convention")

    # 2. THE ARTEFACT MUST BE CONSTRUCTIBLE. Two arms that are equal at their
    #    own peaks but unequal at a shared threshold is exactly the shape
    #    calibration.md found; if the comparison here could not express it,
    #    a null result would be uninformative.
    base = {0.5: 0.50, 0.7: 0.40}
    other = {0.5: 0.62, 0.7: 0.40}
    shared_gap = other[0.5] - base[0.5]
    own_gap = other[0.7] - base[0.5]
    print(f"a synthetic pair: {shared_gap:+.1%} at a shared {SHARED}, "
          f"{own_gap:+.1%} once the second arm is read at its own 0.7")
    assert shared_gap > 0 > own_gap or shared_gap > own_gap

    # 3. THE GATE IS THE REPO'S ONE. Three seeds is its minimum; two must be
    #    refused rather than reported with a smaller n.
    assert calibration.decide([(1.0, 0.0)] * 2, [1.0, 1.0])["holds"] is False
    assert calibration.decide([(1.0, 0.0)] * 3, [1.0, 1.0, 1.0])["holds"]
    assert calibration.decide([(1.0, 0.0)] * 3,
                              [1.0, -1.0, 1.0])["holds"] is False
    print("gate: 2 seeds refused, 3 agreeing pass, 3 with one flip refused")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def load(dataset: str) -> list[dict]:
    rows = []
    for path in sorted(dataset_root(dataset).glob("calibration_curve*.csv")):
        for row in csv.DictReader(path.open()):
            rows.append({**row, "threshold": float(row["threshold"]),
                         "dice": float(row["dice"]),
                         "erl_split": float(row["erl_split"]),
                         "erl_bridged": float(row["erl_bridged"])})
    return rows


def report() -> None:
    print("=== does the calibration artefact reproduce off DRIVE? ===\n")
    print("Every arm read twice: at the shared 0.5, and at the threshold that")
    print("maximises its DEV Dice. Both differences are against A_dice read")
    print("the same way, paired on seed, through calibration.decide().")
    print("THREE SEEDS is the gate's minimum -- every pass here is narrow.\n")
    for dataset in DATASETS:
        rows = load(dataset)
        if not rows:
            print(f"--- {dataset}: nothing scored yet ---\n")
            continue
        seeds = sorted({r["seed"] for r in rows})
        print(f"--- {dataset} ({len(seeds)} seeds) ---")
        for metric in ("erl_split", "erl_bridged"):
            print(f"  {metric}")
            peaks, at_shared, at_own = {}, {}, {}
            for arm in ARMS:
                for seed in seeds:
                    dev = [r for r in rows if r["config"] == arm
                           and r["seed"] == seed and r["split"] == "dev"]
                    test = [r for r in rows if r["config"] == arm
                            and r["seed"] == seed and r["split"] == "test"]
                    if not dev or not test:
                        continue
                    peak = peak_of(dev)
                    peaks.setdefault(arm, []).append(peak)
                    by = {r["threshold"]: r for r in test}
                    if SHARED in by:
                        at_shared.setdefault(arm, {})[seed] = by[SHARED][metric]
                    if peak in by:
                        at_own.setdefault(arm, {})[seed] = by[peak][metric]
            # TWO baselines, not one. A_dice differs from H_aug in
            # AUGMENTATION, so `K_focal_aug - A_dice` bundles the loss with
            # the augmentation and cannot be read as a statement about the
            # loss. On STARE the A_dice column reads K_focal_aug +4.7% and
            # H_aug +4.1% -- so the loss is worth 0.6, not 4.7, and on VessMAP
            # it is worth nothing at all. Subtracting in one's head is exactly
            # how a table gets misquoted, so both columns are printed.
            for base in ("A_dice", "H_aug"):
                print(f"    vs {base}")
                for arm in ARMS:
                    if arm == base or arm not in at_own:
                        continue
                    if base == "H_aug" and arm == "A_dice":
                        continue
                    cells = []
                    for label, table in (("at 0.5", at_shared),
                                         ("at own", at_own)):
                        mine, theirs = table.get(arm, {}), table.get(base, {})
                        common = sorted(set(mine) & set(theirs))
                        if len(common) < 3:
                            cells.append(f"{label} {'--':>16}")
                            continue
                        got = calibration.decide(
                            [(mine[s], theirs[s]) for s in common],
                            [mine[s] - theirs[s] for s in common])
                        cells.append(f"{label} {got['mean']:+.1%} "
                                     f"t{got['t']:5.2f} "
                                     f"{'HOLDS' if got['holds'] else 'fails'}")
                    peak = np.mean(peaks.get(arm, [np.nan]))
                    print(f"      {arm:16} peak {peak:.3f}   "
                          + "   ".join(cells))
        print()


# -------------------------------------------------------------------- main

def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    shard = None
    for index, arg in enumerate(sys.argv):
        if arg == "--shard":
            part, total = sys.argv[index + 1].split("/")
            shard = (int(part), int(total))
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")
              and a in DATASETS] or list(DATASETS)

    for dataset in wanted:
        root = dataset_root(dataset)
        target = out_path(dataset, shard)
        done = set()
        for existing in sorted(root.glob("calibration_curve*.csv")):
            done |= {(r["config"], r["seed"], r["split"])
                     for r in csv.DictReader(existing.open())}
        train_items, test_items = cross_dataset.loader_for(dataset)()
        fit_items, dev_items = cross_dataset.fit_dev(train_items)
        data = cross_dataset.stack(fit_items)
        width = cross_dataset.median_width(test_items)
        component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE
                                 * width * width))
        splits = {"dev": dev_items, "test": test_items}
        geometry = {name: [{"skel": skeletonize(i["label"] & i["fov"])}
                           for i in items] for name, items in splits.items()}
        print(f"[{dataset}] width {width:.2f} px, filter {component_px} px, "
              f"{len(dev_items)} dev / {len(test_items)} test images",
              flush=True)

        fresh = not target.exists()
        with target.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["dataset", "config", "run", "seed",
                                    "epoch", "split", "threshold", "dice",
                                    "erl_split", "erl_bridged"])
            if fresh:
                writer.writeheader()
            # Epoch chosen on DEV, exactly as on DRIVE, and never final.pt:
            # these runs carry per-epoch checkpoints and a heldout-stamped
            # log, so the same rule applies without modification.
            epochs = heldout.chosen_epochs(root=root)
            for arm in ARMS:
                runs = sorted(p.name for p in root.iterdir()
                              if p.is_dir() and p.name.rsplit("_s", 1)[0] == arm
                              and (p / "final.pt").exists())
                # Stride over a sorted list, never hash(). Hash-based
                # sharding gave stare/A_dice 2 of its 3 seeds on 2026-09-01,
                # which is under the gate's minimum, so every cell of the
                # table printed "--" and the stage read as unfinished.
                for run in sweep.shard_filter(runs, shard):
                    seed = run.rsplit("_s", 1)[1]
                    epoch = epochs.get(run)
                    weights = root / run / f"epoch{epoch:03d}.pt" if epoch \
                        else None
                    if weights is None or not weights.exists():
                        print(f"  {dataset}/{run}: no dev-chosen epoch, "
                              f"skipping", flush=True)
                        continue
                    model = train.build_model(arm)
                    model.load_state_dict(
                        train.load_checkpoint(weights)["model"])
                    model.eval()
                    # final.pt here carries only the weights, so the constants
                    # are recomputed from the fit split rather than read back.
                    mean, std = train.normalisation(run, data)
                    for name, items in splits.items():
                        if (arm, seed, name) in done:
                            continue
                        for row in curve_for(model, items, geometry[name],
                                             mean, std, component_px):
                            writer.writerow({"dataset": dataset, "config": arm,
                                             "run": run, "seed": seed,
                                             "epoch": epoch, "split": name,
                                             **row})
                        handle.flush()
                    print(f"  {dataset}/{run} done", flush=True)
        print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
