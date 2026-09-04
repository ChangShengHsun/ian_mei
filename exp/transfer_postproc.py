"""Off DRIVE, does lowering the threshold still beat the post-processing layer?

WRITTEN AND SELFTESTED 2026-09-03, BEFORE IT SCORED ANYTHING.

THE GAP THIS FILLS. threshold_control settled on DRIVE that a dev-picked lower
threshold beats the oriented dilation layer on all ten arms under both ERL
conventions, 10 of 10. composition.py then took the layer's best remaining
form -- growth restricted to skeleton endpoints, 0.395% of the foreground --
and the controls came back dead on 2026-09-02: `endpoint_shuf`, which keeps
the endpoints and replaces the field with noise, matched or beat `endpoint` in
7 of 10 arms. So what the endpoint arm bought was the RESTRICTION, and the
open question is whether even the restriction survives being priced against
the threshold. `lower` in composition.py asks that on DRIVE. This file asks it
on STARE, HRF and VessMAP, which is the difference between a property of one
20-image dataset and a property of the measurement.

WHAT IS MEASURED, and what is deliberately not. Four sources, all FIELD-FREE:

    raw           the arm at its own dev-picked base threshold
    lower         the same arm, threshold moved down again
    isotropic     a disc on every foreground pixel
    endpoint_iso  a disc on the predicted skeleton's endpoints only

There is no `predicted` and no `shuffled` here. The direction field is not an
open question any more -- it lost on DRIVE and its endpoint form lost to its
own shuffled control -- and carrying it across three datasets would multiply
the sweep fivefold to re-answer something already answered. What is still open
is the PLACE (whole mask vs endpoints) and the CURRENCY (morphology vs
threshold), and those need no field. Stating the omission here rather than
letting a reader infer it from a missing column.

THE SPLIT. cross_dataset.fit_dev, the same rule DRIVE uses and asserted
against drive.DEV_IDS. Radii and thresholds are chosen on the dev images and
read on the test images; the checkpoint is the dev-chosen epoch, never
final.pt and never best.pt. All three levels of the protocol leak caught on
2026-09-01 are closed here by construction.

THE GRID IS EXTENDED DOWNWARD, for the reason composition.py documents: the
standard grid stops at 0.10 and several arms' base threshold already sits on
that floor, so an unextended comparator would report "the threshold has
nothing left to give" when the true cause is that the grid ran out.

  python exp/transfer_postproc.py --selftest
  python exp/transfer_postproc.py stare --shard 0/4
  python exp/transfer_postproc.py --report

Writes results/heldout_transfer/<dataset>/postproc_curve[.shardIofN].csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anisotropic
import calibration
import composition
import cross_dataset
import hole_sweep
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import threshold_control as control
import train
import transfer_calibration as calib

DATASETS = calib.DATASETS
# The arms transfer_calibration reports, and only those: an arm scored here
# but not there could not be read against the calibration table.
ARMS = calib.ARMS
# In multiples of the dataset's own median vessel width, never in pixels --
# the three datasets span 4.00 px (STARE, HRF) to 5.66 px (VessMAP), and a
# radius in pixels would be a different operator on each.
RADII = sweep.ISOTROPIC
BASE_BUDGET = composition.BASE_BUDGET
BUDGETS = (0.02, 0.05)
SOURCES = ("lower", "isotropic", "endpoint_iso")
FIELDS = ["dataset", "config", "run", "seed", "epoch", "split", "source",
          "threshold", "radius", "image", "erl_split", "erl_bridged", "dice",
          "fg"]


def out_path(dataset: str, shard) -> Path:
    root = calib.dataset_root(dataset)
    return root / ("postproc_curve.csv" if shard is None else
                   f"postproc_curve.shard{shard[0]}of{shard[1]}.csv")


def base_threshold(dataset: str, config: str, metric: str) -> float:
    """The arm's own operating point on THIS dataset, chosen on its dev split.

    Read out of transfer_calibration's curve, which already measured every
    threshold on every dataset -- so the base here and the base in that table
    are the same number by construction rather than by coincidence.
    """
    dev = [r for r in calib.load(dataset) if r["split"] == "dev"]
    curve = control.curve(dev, config, metric)
    if 0.5 not in curve:
        return 0.5
    chosen = control.best_within(curve, curve[0.5][0] - BASE_BUDGET)
    return 0.5 if chosen is None else chosen


def sources_for(prob, item, geo, base: float, width: float,
                component_px: int) -> list[dict]:
    """Every row for one image: the base mask, and the three ways to spend
    more Dice on top of it."""
    fov = item["fov"]
    pred = speckle.drop_small((prob >= base) & fov, component_px)
    out = [{"source": "raw", "threshold": base, "radius": 0.0,
            **sweep.measure(pred, geo["skel"], geo["truth"])}]
    for value in composition.lower_grid(base):
        down = speckle.drop_small((prob >= value) & fov, component_px)
        out.append({"source": "lower", "threshold": value, "radius": 0.0,
                    **sweep.measure(down, geo["skel"], geo["truth"])})
    ends = composition.endpoints_of(pred)
    for radius in RADII:
        if radius == 0.0:
            continue
        grown = anisotropic.isotropic_dilation(pred, radius * width) & fov
        out.append({"source": "isotropic", "threshold": base,
                    "radius": radius,
                    **sweep.measure(grown, geo["skel"], geo["truth"])})
        tips = composition.endpoint_disc(pred, ends, radius * width, fov)
        out.append({"source": "endpoint_iso", "threshold": base,
                    "radius": radius,
                    **sweep.measure(tips, geo["skel"], geo["truth"])})
    return out


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

    # 2. DEV AND TEST MUST NOT OVERLAP, on every dataset, by name. The whole
    #    file is a claim about choosing on one split and reading on another;
    #    if the splits share an image the claim is void, and nothing in the
    #    numbers would look wrong.
    for dataset in DATASETS:
        train_items, test_items = cross_dataset.loader_for(dataset)()
        fit_items, dev_items = cross_dataset.fit_dev(train_items)
        names = {"fit": {i["name"] for i in fit_items},
                 "dev": {i["name"] for i in dev_items},
                 "test": {i["name"] for i in test_items}}
        assert not (names["dev"] & names["test"]), dataset
        assert not (names["fit"] & names["dev"]), dataset
        assert not (names["fit"] & names["test"]), dataset
        print(f"{dataset}: {len(names['fit'])} fit / {len(names['dev'])} dev "
              f"/ {len(names['test'])} test, all disjoint by name")

    # 3. THE OPERATORS MUST DELIVER WHAT THEIR NAMES PROMISE. The rule this
    #    repo adopted on 2026-09-02 after three silent instrument bugs found
    #    in one night, none of which made the results look wrong.
    line = np.zeros((60, 60), dtype=bool)
    line[30, 10:50] = True
    broken = line.copy()
    broken[:, 28:33] = False
    fov = np.ones_like(line)
    ends = composition.endpoints_of(broken)
    assert ends.sum() == 4, ends.sum()
    disc_all = anisotropic.isotropic_dilation(broken, 2.0) & fov
    disc_tip = composition.endpoint_disc(broken, ends, 2.0, fov)
    added_all = int(disc_all.sum() - broken.sum())
    added_tip = int(disc_tip.sum() - broken.sum())
    assert 0 < added_tip < added_all, (added_tip, added_all)
    print(f"a disc of radius 2 on a broken line: whole mask adds {added_all} "
          f"px, endpoints only {added_tip} px "
          f"({added_tip / added_all:.1%} of the cost)")

    # 4. `lower` MUST ACTUALLY LOWER, AND MUST REACH BELOW THE STANDARD FLOOR.
    for base in (0.1, 0.3, 0.5):
        grid = composition.lower_grid(base)
        assert grid and max(grid) < base, (base, grid)
        assert list(grid) == sorted(set(grid)), grid
    assert min(composition.lower_grid(0.1)) < min(control.THRESHOLDS)
    print(f"lower grid below the 0.1 floor: {composition.lower_grid(0.1)}")

    # 5. AND THE WHOLE ROW BUILDER MUST RUN, on a synthetic image, producing
    #    one row per source-setting and no silent duplicates.
    prob = np.zeros((60, 60), dtype=np.float32)
    prob[29:32, 10:50] = 0.9
    prob[30, 28:33] = 0.2
    geo = {"skel": skeletonize(line), "truth": line}
    rows = sources_for(prob, {"fov": fov}, geo, 0.5, 3.0, 4)
    keys = [(r["source"], r["threshold"], r["radius"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate source-setting rows"
    counts = {s: sum(1 for r in rows if r["source"] == s) for r in rows
              for s in [r["source"]]}
    assert counts["raw"] == 1, counts
    assert counts["lower"] == len(composition.lower_grid(0.5)), counts
    assert counts["isotropic"] == counts["endpoint_iso"] == len(RADII) - 1
    assert set(rows[0]) == {"source", "threshold", "radius", "erl_split",
                            "erl_bridged", "dice", "fg"}, sorted(rows[0])
    print(f"one image builds {len(rows)} rows: " +
          ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    # 6. LOWERING MUST BUY FOREGROUND. If it does not, the source is measuring
    #    something other than a threshold.
    lows = [r for r in rows if r["source"] == "lower"]
    lows.sort(key=lambda r: r["threshold"])
    assert lows[0]["fg"] >= lows[-1]["fg"], [r["fg"] for r in lows]
    print(f"foreground from threshold {lows[0]['threshold']} to "
          f"{lows[-1]['threshold']}: {lows[0]['fg']} -> {lows[-1]['fg']}")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def load(dataset: str) -> list[dict]:
    rows = []
    for path in sorted(calib.dataset_root(dataset).glob("postproc_curve*.csv")):
        for row in csv.DictReader(path.open()):
            rows.append({**row, "threshold": float(row["threshold"]),
                         "radius": float(row["radius"]),
                         "dice": float(row["dice"]),
                         "erl_split": float(row["erl_split"]),
                         "erl_bridged": float(row["erl_bridged"])})
    return rows


def setting_key(source: str) -> str:
    """Which column carries this source's setting.

    `lower` is swept over thresholds and the other two over radii. Keying off
    the wrong column returns an empty match set, which reads as "-- " in the
    table and is indistinguishable from an unfinished run.
    """
    return "threshold" if source == "lower" else "radius"


def pick(rows, config, source, floor, metric, budget):
    """Best setting for one source within a Dice budget, on the DEV rows."""
    key = setting_key(source)
    best = None
    for value in sorted({r[key] for r in rows if r["config"] == config
                         and r["source"] == source}):
        cells = [r for r in rows if r["config"] == config
                 and r["source"] == source and r[key] == value]
        if float(np.mean([r["dice"] for r in cells])) < floor - budget:
            continue
        traced = float(np.mean([r[metric] for r in cells]))
        if best is None or traced > best[1]:
            best = (value, traced)
    return None if best is None else best[0]


def report() -> None:
    print("=== off DRIVE: the threshold against the post-processing layer "
          "===\n")
    print("Every source is read at the SAME base threshold -- the arm's own,")
    print("chosen on that dataset's dev split at a 0.02 Dice budget. Settings")
    print("are chosen on dev and read on test. `raw` is the threshold ALONE.")
    print("`lower` spends the budget by moving the threshold down again")
    print("instead of dilating; it is not an operator, it is the question of")
    print("whether an operator is needed. No direction field is scored here")
    print("-- it lost on DRIVE and lost to its own shuffled control.\n")
    for dataset in DATASETS:
        rows = load(dataset)
        dev = [r for r in rows if r["split"] == "dev"]
        test = [r for r in rows if r["split"] == "test"]
        if not dev or not test:
            print(f"--- {dataset}: no rows ---\n")
            continue
        seeds = sorted({r["seed"] for r in test})
        print(f"--- {dataset}: {len(test)} test rows, {len(dev)} dev rows, "
              f"{len(seeds)} seeds ---")
        for metric, label in (("erl_split", "convention A (a bridged gap "
                                            "splits)"),
                              ("erl_bridged", "convention B (it does not)")):
            print(f"  {label}")
            for config in ARMS:
                mine_raw = [r for r in dev if r["config"] == config
                            and r["source"] == "raw"]
                theirs_raw = [r for r in test if r["config"] == config
                              and r["source"] == "raw"]
                if not mine_raw or not theirs_raw:
                    continue
                floor = float(np.mean([r["dice"] for r in mine_raw]))
                base = sorted({r["threshold"] for r in theirs_raw})
                traced = float(np.mean([r[metric] for r in theirs_raw]))
                print(f"    {config:14} threshold "
                      f"{','.join(f'{b:g}' for b in base):8} "
                      f"raw {traced:.1%} traced at Dice "
                      f"{float(np.mean([r['dice'] for r in theirs_raw])):.4f}")
                for budget in BUDGETS:
                    cells = []
                    for source in SOURCES:
                        value = pick(dev, config, source, floor, metric,
                                     budget)
                        if value is None:
                            cells.append(f"{source} {'--':>17}")
                            continue
                        key = setting_key(source)
                        mine = {(r["seed"], r["image"]): r[metric]
                                for r in test if r["config"] == config
                                and r["source"] == source and r[key] == value}
                        theirs = {(r["seed"], r["image"]): r[metric]
                                  for r in theirs_raw}
                        pairs = sorted(set(mine) & set(theirs))
                        got_seeds = sorted({s for s, _ in pairs})
                        if len(got_seeds) < 3:
                            cells.append(f"{source} {'--':>17}")
                            continue
                        got = calibration.decide(
                            [(mine[k], theirs[k]) for k in pairs],
                            [float(np.mean([mine[k] - theirs[k]
                                            for k in pairs if k[0] == s]))
                             for s in got_seeds])
                        cells.append(f"{source} @{value:<5g} "
                                     f"{got['mean']:+.1%} t{got['t']:5.2f} "
                                     f"{'HOLDS' if got['holds'] else 'fails'}")
                    print(f"      -{budget:.2f}  " + "   ".join(cells))
        print()
    print("Read as `source - raw`. `lower` matching or beating both dilation")
    print("arms on a dataset means the post-processing layer buys nothing")
    print("there that the threshold does not buy more cheaply -- the DRIVE")
    print("finding reproducing, rather than a property of one 20-image set.")


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
        done = set()
        for existing in sorted(root.glob("postproc_curve*.csv")):
            done |= {(r["config"], r["seed"], r["split"])
                     for r in csv.DictReader(existing.open())}
        train_items, test_items = cross_dataset.loader_for(dataset)()
        fit_items, dev_items = cross_dataset.fit_dev(train_items)
        data = cross_dataset.stack(fit_items)
        width = cross_dataset.median_width(test_items)
        component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE
                                 * width * width))
        splits = {"dev": dev_items, "test": test_items}
        geometry = {name: [{"skel": skeletonize(i["label"] & i["fov"]),
                            "truth": i["label"] & i["fov"]} for i in items]
                    for name, items in splits.items()}
        bases = {arm: base_threshold(dataset, arm, "erl_bridged")
                 for arm in ARMS}
        print(f"[{dataset}] width {width:.2f} px, filter {component_px} px, "
              f"{len(dev_items)} dev / {len(test_items)} test images", flush=True)
        for arm in ARMS:
            print(f"  base {arm}: {bases[arm]:g} "
                  f"({len(composition.lower_grid(bases[arm]))} lower steps)",
                  flush=True)

        fresh = not target.exists()
        with target.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if fresh:
                writer.writeheader()
            epochs = heldout.chosen_epochs(root=root)
            for arm in ARMS:
                runs = sorted(p.name for p in root.iterdir()
                              if p.is_dir() and p.name.rsplit("_s", 1)[0] == arm
                              and (p / "final.pt").exists())
                # Stride over a sorted list, never hash(): see shard_filter.
                for run in sweep.shard_filter(runs, shard):
                    seed = run.rsplit("_s", 1)[1]
                    todo = [n for n in splits if (arm, seed, n) not in done]
                    if not todo:
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
                    for name in todo:
                        # Accumulated per split, then written in one go: a
                        # kill mid-split loses that split entirely rather
                        # than leaving a half curve the resume set would
                        # read as complete.
                        out = []
                        for item, geo in zip(splits[name], geometry[name]):
                            prob = train.predict_full(model, item["image"],
                                                      mean, std)
                            for row in sources_for(prob, item, geo,
                                                   bases[arm], width,
                                                   component_px):
                                out.append({"dataset": dataset, "config": arm,
                                            "run": run, "seed": seed,
                                            "epoch": epoch, "split": name,
                                            "image": item["name"], **row})
                        writer.writerows(out)
                        handle.flush()
                    print(f"  {dataset}/{run} done ({len(todo)} splits)",
                          flush=True)
        print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
