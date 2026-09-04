"""The specification table, off DRIVE. Same three axes, three more datasets.

WRITTEN AND SELFTESTED 2026-09-04, BEFORE IT SCORED ANYTHING.

WHY THIS FILE EXISTS. exp/erl_spec.py measured the paper's first table -- the
two splitting rules x three length conventions x two denominators -- on DRIVE
at twelve seeds. It found a per-arm spread of 38.6 to 42.0 percentage points:
one set of predictions can be reported as an ERL anywhere in that range
without changing the model, the data, the threshold or the checkpoint.

A reviewer's first question is whether that is a property of ERL or a property
of DRIVE. DRIVE is 20 test images at 4.00 px median vessel width; VessMAP is
5.66 px and a completely different imaging setup. If the spread collapses off
DRIVE, the specification claim is much weaker than the DRIVE table suggests
and the paper has to say so.

WHAT IS NOT RE-IMPLEMENTED. Everything that computes a number is imported
from erl_spec: bridged_labels, fragment_sizes, skeleton_total, measure. That
file's selftest anchors them against erl_length.run_length and
erl_convention.bridged_run_length on real retinal images, and a second copy of
that arithmetic is exactly how the two tables would quietly drift apart. This
file is plumbing only.

THE OPERATING POINT is each run's own DEV Dice peak on ITS OWN dataset, read
from that dataset's calibration_curve*.csv through calib.peak_of -- the same
rule DRIVE uses via frontier_dev.csv, from the source that exists here. Dice,
not ERL: choosing the threshold with an ERL is circular when ERL is the
quantity under test, and it would let each column pick its own operating
point.

FOUR ARMS, not ten: calib.ARMS, the arms transfer_calibration reports. An arm
scored here but not there could not be read against the calibration table.

PRE-REGISTERED 2026-09-04, before this file has scored a single run. The DRIVE
table's three findings, restated as predictions for the other three datasets:

  1. Every dataset shows a per-arm spread above 20 percentage points. DRIVE's
     was 38.6-42.0. A dataset coming in under 20 would mean the spread is
     partly a DRIVE property and the headline has to be a range across
     datasets, not one number.
  2. The splitting rule's SIGN REVERSES with the length convention, on every
     dataset: bridged > split for all four arms under `pixels` and `edges`,
     and for none of them under `diameter`. This is the sharpest thing the
     DRIVE table found (10/10 and 10/10 against 0/10) and the easiest to
     falsify here. A dataset where diameter does not flip it means the
     interaction is not a property of the definitions.
  3. `full` < `covered` in every cell of every dataset, and
     `full == covered x coverage` to floating point. This is an identity, not
     a finding; it is here so that a violation is caught as a bug rather than
     read as a result.

  python exp/erl_spec_transfer.py --selftest
  python exp/erl_spec_transfer.py stare --shard 0/4
  python exp/erl_spec_transfer.py --report

Writes results/heldout_transfer/<dataset>/erl_spec[.shardIofN].csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import erl_length
import erl_spec
import hole_sweep
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import train
import transfer_calibration as calib

DATASETS = calib.DATASETS
ARMS = calib.ARMS
SPLIT_RULES = erl_spec.SPLIT_RULES
LENGTHS = erl_spec.LENGTHS
DENOMINATORS = erl_spec.DENOMINATORS
FIELDS = ["dataset"] + erl_spec.FIELDS


def out_path(dataset: str, shard) -> Path:
    root = calib.dataset_root(dataset)
    return root / ("erl_spec.csv" if shard is None else
                   f"erl_spec.shard{shard[0]}of{shard[1]}.csv")


def dev_peaks(dataset: str) -> dict:
    """{run: its own dev Dice-maximising threshold} on this dataset.

    Per RUN, not per arm: erl_spec does the same on DRIVE, and averaging the
    dev curve over seeds before taking the argmax would give every seed a
    threshold chosen partly by the others.
    """
    per_run = defaultdict(list)
    for row in calib.load(dataset):
        if row["split"] == "dev":
            per_run[row["run"]].append(row)
    return {run: calib.peak_of(rows) for run, rows in per_run.items()}


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    for arm in ARMS:
        assert arm in train.CONFIGS, arm
    assert set(DATASETS) == {"stare", "hrf", "vessmap"}, DATASETS

    # 1. THE PARTITION. Four of this repo's tables silently lost seeds to
    #    hash-based sharding on 2026-09-01; every file that shards asserts it.
    work = [f"a_s{index}" for index in range(17)]
    for total in (2, 3, 4, 5):
        flat = [r for i in range(total)
                for r in sweep.shard_filter(work, (i, total))]
        assert sorted(flat) == sorted(work) and len(flat) == len(set(flat))
    print("sharding is an exact partition over 2/3/4/5 shards")

    # 2. DEV AND TEST MUST NOT OVERLAP, on every dataset, by name. The
    #    operating point is chosen on dev and every number is read on test;
    #    if the two share an image nothing in the output would look wrong.
    for dataset in DATASETS:
        train_items, test_items = cross_dataset.loader_for(dataset)()
        fit_items, dev_items = cross_dataset.fit_dev(train_items)
        names = {"fit": {i["name"] for i in fit_items},
                 "dev": {i["name"] for i in dev_items},
                 "test": {i["name"] for i in test_items}}
        assert not (names["dev"] & names["test"]), dataset
        assert not (names["fit"] & names["dev"]), dataset
        assert not (names["fit"] & names["test"]), dataset
    print("fit / dev / test disjoint by name on all three datasets")

    # 3. THE ANCHORS, RE-CHECKED ON A NON-DRIVE IMAGE. erl_spec asserts these
    #    on DRIVE. The imported code is the same code, but the geometry is
    #    not: VessMAP is 5.66 px median width against DRIVE's 4.00, and a
    #    convention that agreed at one scale and not the other would be a
    #    real defect. VessMAP because its images are the small ones -- HRF
    #    would make this selftest cost minutes.
    _, test_items = cross_dataset.loader_for("vessmap")()
    item = test_items[0]
    skel = skeletonize(item["label"] & item["fov"])
    rng = np.random.default_rng(0)
    pred = (item["label"] & item["fov"]
            & (rng.random(item["label"].shape) < 0.85))
    for convention in LENGTHS:
        got = erl_spec.measure(skel, pred, "split", convention)[0]
        # run_length returns (ERL, total skeleton length); the ERL is [0].
        want = erl_length.run_length(skel, pred, convention)[0]
        assert abs(got - want) < 1e-9, (convention, got, want)
    print(f"anchor 1 on vessmap/{item['name']}: (split, *, full) matches "
          f"erl_length.run_length in all {len(LENGTHS)} conventions")

    import erl_convention
    got = erl_spec.measure(skel, pred, "bridged", "pixels")[0]
    want = erl_convention.bridged_run_length(skel, pred)
    assert abs(got - want) < 1e-9, (got, want)
    print(f"anchor 2 on vessmap/{item['name']}: (bridged, pixels, full) "
          f"matches erl_convention.bridged_run_length ({got:.4f})")

    # 4. PREDICTION 3 IS AN IDENTITY, so it is asserted here rather than
    #    discovered in the table: the two denominators differ by exactly the
    #    covered fraction, in every cell.
    for rule in SPLIT_RULES:
        for convention in LENGTHS:
            full, covered, coverage = erl_spec.measure(skel, pred, rule,
                                                       convention)
            assert abs(full - covered * coverage) < 1e-9, (rule, convention)
            assert covered > full - 1e-9, (rule, convention, full, covered)
    print(f"identity: full == covered x coverage in all "
          f"{len(SPLIT_RULES) * len(LENGTHS)} cells, and full <= covered")

    # 5. THE OPERATING POINT MUST COME OFF DEV ROWS ONLY. A peak read from a
    #    mixed dev+test pool is the leak this whole paper is about, and it
    #    would be invisible in the output.
    fake = [{"run": "a_s0", "split": "dev", "threshold": 0.3, "dice": 0.9,
             "config": "A_dice", "erl_split": 0.1, "erl_bridged": 0.1},
            {"run": "a_s0", "split": "test", "threshold": 0.7, "dice": 0.99,
             "config": "A_dice", "erl_split": 0.1, "erl_bridged": 0.1}]
    per_run = defaultdict(list)
    for row in fake:
        if row["split"] == "dev":
            per_run[row["run"]].append(row)
    assert calib.peak_of(per_run["a_s0"]) == 0.3
    print("the dev peak ignores a higher-Dice test row (0.3, not 0.7)")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def load(dataset: str) -> list[dict]:
    rows = []
    for path in sorted(calib.dataset_root(dataset).glob("erl_spec*.csv")):
        for row in csv.DictReader(path.open()):
            rows.append({**row, "erl": float(row["erl"]),
                         "coverage": float(row["coverage"]),
                         "oracle": float(row["oracle"])})
    return rows


def scaled_cells(rows: list[dict]) -> dict:
    """{(arm, rule, length, denominator): mean fraction of the ceiling}.

    ERL is a LENGTH. Every value is divided by what a perfect prediction
    scores on the SAME image under the SAME length convention, which is the
    same divisor for both denominators -- that is what keeps
    `full = covered x coverage` readable across the two columns.
    """
    cells = defaultdict(list)
    for row in rows:
        if row["oracle"] <= 0:
            continue
        cells[(row["config"], row["split_rule"], row["length"],
               row["denominator"])].append(row["erl"] / row["oracle"])
    return {key: float(np.mean(values)) for key, values in cells.items()}


def report() -> None:
    print("=== the specification table, off DRIVE ===\n")
    print("Same three axes as erl_spec.py, same measurement code, three more")
    print("datasets. Epoch chosen on that dataset's dev split, threshold at")
    print("each run's own dev Dice peak, every number read on test. Every")
    print("cell is a fraction of what a PERFECT prediction scores on the same")
    print("image: ERL is a length, not a ratio.\n")
    print("`edges` and `covered` are what the field's reference")
    print("implementation uses; `pixels`, `full` and `split` are what")
    print("exp/erl.py has always used. Neither is wrong -- the point is that")
    print("a paper reporting one number has silently picked a cell.\n")

    spreads = {}
    flips = {}
    for dataset in DATASETS:
        rows = load(dataset)
        if not rows:
            print(f"--- {dataset}: no rows ---\n")
            continue
        seeds = sorted({r["seed"] for r in rows})
        cells = scaled_cells(rows)
        print(f"--- {dataset}: {len(rows)} rows, {len(seeds)} seeds ---")
        header = "".join(f"{rule[:3]}/{den[:3]:<4}".rjust(11)
                         for rule in SPLIT_RULES for den in DENOMINATORS)
        for convention in LENGTHS:
            print(f"  fragment length: {convention}")
            print(f"    {'arm':16}{header}   coverage")
            for arm in ARMS:
                got = []
                for rule in SPLIT_RULES:
                    for den in DENOMINATORS:
                        value = cells.get((arm, rule, convention, den))
                        got.append(f"{100 * value:9.1f}%" if value is not None
                                   else f"{'--':>10}")
                cover = [r["coverage"] for r in rows if r["config"] == arm
                         and r["length"] == convention
                         and r["split_rule"] == "split"]
                tail = f"{100 * float(np.mean(cover)):9.1f}%" if cover else ""
                print(f"    {arm:16}" + "".join(got) + f"   {tail}")
        # Prediction 1: the per-arm spread.
        per_arm = {}
        for arm in ARMS:
            got = {k: v for k, v in cells.items() if k[0] == arm}
            if got:
                per_arm[arm] = 100 * (max(got.values()) - min(got.values()))
        if per_arm:
            spreads[dataset] = per_arm
            print(f"\n    spread per arm: " +
                  ", ".join(f"{a} {v:.1f}" for a, v in per_arm.items()))
        # Prediction 2: does diameter reverse the splitting rule's sign?
        for convention in LENGTHS:
            wins = sum(1 for arm in ARMS
                       if cells.get((arm, "bridged", convention, "covered"),
                                    0.0)
                       > cells.get((arm, "split", convention, "covered"), 0.0)
                       and (arm, "split", convention, "covered") in cells)
            have = sum(1 for arm in ARMS
                       if (arm, "split", convention, "covered") in cells)
            flips[(dataset, convention)] = (wins, have)
        print()

    if not spreads:
        raise SystemExit("no erl_spec*.csv on any transfer dataset -- "
                         "refusing to print an empty table; an empty table "
                         "is not a null result")

    print("--- prediction 1: is the spread a property of ERL or of DRIVE? ---")
    print("    DRIVE, twelve seeds, ten arms: 38.6 to 42.0 points.")
    print(f"    {'dataset':10}{'lowest arm':>14}{'highest arm':>14}"
          f"   above 20 points?")
    for dataset, per_arm in spreads.items():
        low, high = min(per_arm.values()), max(per_arm.values())
        print(f"    {dataset:10}{low:>13.1f} {high:>13.1f}"
              f"   {'yes' if low > 20 else 'NO -- prediction 1 falsified'}")
    print()

    print("--- prediction 2: does `diameter` reverse the splitting rule? ---")
    print("    DRIVE: bridged beat split in 10/10 arms under pixels and")
    print("    edges, and 0/10 under diameter.")
    print(f"    {'dataset':10}" +
          "".join(f"{c:>12}" for c in LENGTHS) + "   verdict")
    for dataset in spreads:
        got = [flips[(dataset, c)] for c in LENGTHS]
        cells = "".join(f"{w}/{h:<11}".rjust(12) for w, h in got)
        held = (got[0][0] == got[0][1] and got[1][0] == got[1][1]
                and got[2][0] == 0)
        print(f"    {dataset:10}{cells}   "
              f"{'holds' if held else 'FALSIFIED on this dataset'}")
    print()
    print("Read the spread as: the range of ERLs one set of predictions can")
    print("be reported as, without changing the model, the data, the")
    print("threshold or the checkpoint.")


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
        root = calib.dataset_root(dataset)
        target = out_path(dataset, shard)
        peaks = dev_peaks(dataset)
        if not peaks:
            print(f"[{dataset}] no calibration_curve*.csv -- the operating "
                  f"point is the dev Dice peak and must be read from it; "
                  f"skipping", flush=True)
            continue
        done = set()
        for existing in sorted(root.glob("erl_spec*.csv")):
            done |= {(r["config"], r["seed"])
                     for r in csv.DictReader(existing.open())}

        train_items, test_items = cross_dataset.loader_for(dataset)()
        fit_items, _ = cross_dataset.fit_dev(train_items)
        data = cross_dataset.stack(fit_items)
        width = cross_dataset.median_width(test_items)
        component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE
                                 * width * width))
        geometry = [skeletonize(i["label"] & i["fov"]) for i in test_items]
        oracles = [{c: erl_length.oracle_run_length(s, c) for c in LENGTHS}
                   for s in geometry]
        epochs = heldout.chosen_epochs(root=root)
        print(f"[{dataset}] width {width:.2f} px, filter {component_px} px, "
              f"{len(test_items)} test images, {len(peaks)} runs with a dev "
              f"peak", flush=True)

        fresh = not target.exists()
        with target.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if fresh:
                writer.writeheader()
            for arm in ARMS:
                runs = sorted(p.name for p in root.iterdir()
                              if p.is_dir()
                              and p.name.rsplit("_s", 1)[0] == arm
                              and (p / "final.pt").exists())
                # Stride over a sorted list, never hash(): see shard_filter.
                for run in sweep.shard_filter(runs, shard):
                    seed = run.rsplit("_s", 1)[1]
                    if (arm, seed) in done or run not in peaks:
                        continue
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
                    mean, std = train.normalisation(run, data)
                    base = peaks[run]
                    # Accumulated per run, then written in one go: a kill
                    # mid-run would otherwise leave a partial set of images
                    # that the resume key reads as complete.
                    out = []
                    for item, skel, oracle in zip(test_items, geometry,
                                                  oracles):
                        prob = train.predict_full(model, item["image"],
                                                  mean, std)
                        pred = speckle.drop_small((prob >= base) & item["fov"],
                                                  component_px)
                        common = {"dataset": dataset, "run": run,
                                  "config": arm, "seed": seed, "epoch": epoch,
                                  "threshold": base, "image": item["name"]}
                        for rule in SPLIT_RULES:
                            for convention in LENGTHS:
                                full, covered, coverage = erl_spec.measure(
                                    skel, pred, rule, convention)
                                for name, value in (("full", full),
                                                    ("covered", covered)):
                                    out.append({**common, "split_rule": rule,
                                                "length": convention,
                                                "denominator": name,
                                                "erl": value,
                                                "coverage": coverage,
                                                "oracle": oracle[convention]})
                    writer.writerows(out)
                    handle.flush()
                    print(f"  {dataset}/{run} at threshold {base:g} done",
                          flush=True)
        print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
