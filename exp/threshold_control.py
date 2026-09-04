"""The control the post-processing table is missing: just lower the threshold.

WRITTEN 2026-09-01, after `postproc_verdict.txt` showed the oriented-dilation
layer beating isotropic dilation on every arm. `isotropic` answers "is the
gain direction or is it paint". It does NOT answer the cheaper question a
reviewer asks first:

    the layer buys ERL by spending Dice. So does moving the threshold.
    At the same Dice, which buys more?

Nothing in the repo compared those two until now. A first look using the
existing `frontier.csv` said thresholding wins on all ten arms by 4.5 to 12.7
points -- but `frontier.csv` records only `erl.expected_run_length`, the
convention where a bridged gap SPLITS a run, while the postproc headline is
read under the other convention. Comparing our bridged number against their
split number is not a comparison, and that is the whole reason this file
exists: it scores the threshold sweep under BOTH conventions so the two
tables can be read against each other.

The grid, the component filter and the epoch rule are frontier.py's, so a row
here is comparable to a row there by construction rather than by hope.

  python exp/threshold_control.py --selftest
  python exp/threshold_control.py --shard 0/6           # test images
  python exp/threshold_control.py --dev --shard 0/6     # the 5 held out
  python exp/threshold_control.py --report

Writes results/heldout/threshold_control[.shardIofN][_dev].csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import drive
import erl
import erl_convention
import frontier
import hole_sweep
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import train

# Exactly frontier.py's grid. Importing it rather than restating it is the
# point: two copies of a threshold list drift, and then two tables that look
# comparable are not.
THRESHOLDS = frontier.THRESHOLDS

# The arms the postproc table reports, and only those. Scoring the other 28
# configs would add hours and answer nothing this control is asked.
ARMS = tuple(sweep.CONTROL) + tuple(sweep.FRONTIER)


def files_for(split: str) -> list[Path]:
    """Every csv of one split, and NONE of the other.

    CAUGHT 2026-09-01, on the first launch: the test sweep globbed
    `threshold_control*.csv`, which also matches `threshold_control_dev*.csv`,
    read the five dev runs as already done, and finished in two seconds having
    scored nothing. A prefix that is a prefix of another stem is not a filter.
    """
    found = sorted(heldout.ROOT.glob("threshold_control*.csv"))
    is_dev = [p for p in found if p.name.startswith("threshold_control_dev")]
    return is_dev if split == "dev" else [p for p in found if p not in is_dev]


def out_path(shard, split: str = "test") -> Path:
    stem = "threshold_control" if split == "test" else "threshold_control_dev"
    if shard is None:
        return heldout.ROOT / f"{stem}.csv"
    return heldout.ROOT / f"{stem}.shard{shard[0]}of{shard[1]}.csv"


def curve_for(model, items, geometry, mean: float, std: float,
              component_px: int) -> list[dict]:
    """One row per threshold, under both ERL conventions.

    The probability map is computed once per image and thresholded 33 times,
    which is what makes this affordable; the two ERL calls per threshold are
    the cost, not the forward pass.
    """
    totals = defaultdict(lambda: {"inter": 0.0, "sizes": 0.0, "split": 0.0,
                                  "bridged": 0.0, "skel": 0.0, "fg": 0.0})
    for item, geo in zip(items, geometry):
        prob = train.predict_full(model, item["image"], mean, std)
        truth = item["label"] & item["fov"]
        for threshold in THRESHOLDS:
            pred = speckle.drop_small((prob >= threshold) & item["fov"],
                                      component_px)
            cell = totals[threshold]
            cell["inter"] += float((pred & truth).sum())
            cell["sizes"] += float(pred.sum() + truth.sum())
            cell["fg"] += float(pred.sum())
            cell["split"] += erl.expected_run_length(geo["skel"], pred)
            cell["bridged"] += erl_convention.bridged_run_length(geo["skel"],
                                                                pred)
            cell["skel"] += float(geo["skel"].sum())
    rows = []
    for threshold in THRESHOLDS:
        cell = totals[threshold]
        skel = cell["skel"] or 1.0
        rows.append({
            "threshold": threshold,
            "dice": round(2.0 * cell["inter"] / cell["sizes"], 5)
            if cell["sizes"] else 0.0,
            "erl_split": round(cell["split"] / skel, 5),
            "erl_bridged": round(cell["bridged"] / skel, 5),
            "fg": int(cell["fg"])})
    return rows


# ------------------------------------------------------------------ reading

def load(split: str = "test") -> list[dict]:
    rows = []
    for path in files_for(split):
        for row in csv.DictReader(path.open()):
            rows.append({**row,
                         "threshold": float(row["threshold"]),
                         "dice": float(row["dice"]),
                         "erl_split": float(row["erl_split"]),
                         "erl_bridged": float(row["erl_bridged"])})
    return rows


def curve(rows: list[dict], config: str, metric: str) -> dict:
    """{threshold: (mean dice, mean metric)} over the seeds of one arm."""
    cells = defaultdict(list)
    for row in rows:
        if row["config"] == config:
            cells[row["threshold"]].append((row["dice"], row[metric]))
    return {t: (float(np.mean([d for d, _ in v])),
                float(np.mean([e for _, e in v]))) for t, v in cells.items()}


def best_within(dev_curve: dict, dice_floor: float):
    """The threshold with the most traced length whose DEV Dice clears floor.

    Chosen on dev, read on test -- the same rule the postproc geometry now
    follows, and for the same reason: a threshold picked on the images it is
    reported on reads high by an amount nobody can bound.
    """
    best = None
    for threshold, (dice, traced) in dev_curve.items():
        if dice >= dice_floor and (best is None or traced > best[1]):
            best = (threshold, traced)
    return None if best is None else best[0]


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    assert THRESHOLDS is frontier.THRESHOLDS
    assert 0.5 in THRESHOLDS, "the operating point must be on the grid"
    print(f"{len(THRESHOLDS)} thresholds, {THRESHOLDS[0]} to {THRESHOLDS[-1]}, "
          f"{len(ARMS)} arms -- the same grid frontier.py used")

    # 1. THE TWO CONVENTIONS MUST DISAGREE on a prediction that bridges a gap.
    #    If they agreed there would be nothing to re-score and this file would
    #    be a copy of frontier.py.
    skel = np.zeros((30, 100), dtype=bool)
    skel[15, 5:95] = True
    bridged = np.zeros_like(skel)
    bridged[14:17, 5:95] = True
    bridged[:, 45:55] = False
    bridged[17:20, 44:56] = True
    total = float(skel.sum())
    split = erl.expected_run_length(skel, bridged) / total
    joined = erl_convention.bridged_run_length(skel, bridged) / total
    print(f"a bridged detour: split {split:.1%} vs bridged {joined:.1%} "
          f"-- both columns are written, always")
    assert joined > split

    # 2. THE PICK IS MADE ON DEV. Build a case where dev and test disagree
    #    about the best threshold and show the size of the difference, so a
    #    reader can see what picking on test would have bought.
    dev = {0.2: (0.79, 0.60), 0.3: (0.81, 0.50), 0.4: (0.82, 0.40)}
    test = {0.2: (0.79, 0.52), 0.3: (0.81, 0.50), 0.4: (0.82, 0.62)}
    chosen = best_within(dev, 0.80)
    assert chosen == 0.3, chosen
    on_test = test[chosen][1]
    flattering = max(e for d, e in test.values() if d >= 0.80)
    print(f"dev picks threshold {chosen} and reads {on_test:.1%} on test; "
          f"picking on test itself would read {flattering:.1%}")
    assert flattering > on_test

    # 3. THE TWO SPLITS MUST NOT SEE EACH OTHER'S FILES. `threshold_control`
    #    is a prefix of `threshold_control_dev`, which is exactly how the test
    #    sweep scored nothing on its first launch.
    dev_names = {p.name for p in files_for("dev")}
    test_names = {p.name for p in files_for("test")}
    assert not (dev_names & test_names), sorted(dev_names & test_names)
    assert all(n.startswith("threshold_control_dev") for n in dev_names)
    assert not any(n.startswith("threshold_control_dev") for n in test_names)
    print(f"split filter: {len(dev_names)} dev file(s), {len(test_names)} "
          f"test file(s), no overlap")

    # 4. THE FLOOR IS A FLOOR. A threshold that traces further but costs more
    #    Dice than the layer did must not be selectable -- that is exactly the
    #    error that would let thresholding "win" for free.
    assert best_within({0.1: (0.70, 0.99)}, 0.80) is None
    print("a cheaper-Dice threshold is refused however far it traces")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def report() -> None:
    rows, dev_rows = load("test"), load("dev")
    if not rows or not dev_rows:
        raise SystemExit("need both threshold_control*.csv and "
                         "threshold_control_dev*.csv -- run with and "
                         "without --dev")
    import summarize_postproc as summary
    post = [r for r in summary.load() if r["field_arm"] == "H_aug_dir"]
    post_dev = summary.load("postproc_dev")
    # An empty table is not a result, and this repo has twice mistaken one for
    # a result. Refuse rather than print a header with no rows under it.
    if not post or not post_dev:
        raise SystemExit(
            "need postproc_ceiling*.csv AND postproc_dev*.csv -- the layer's\n"
            "geometry is chosen on dev, so without the dev sweep there is no\n"
            "layer column to compare a threshold against. Run:\n"
            "  exp/postproc_ceiling.py --dev --field H_aug_dir")
    print("=== the layer against simply lowering the threshold ===\n")
    print(f"{len(rows)} test rows, {len(dev_rows)} dev rows, "
          f"{len(post)} postproc rows")
    print("Both sides pick on the 5 dev images and are read on the 20 test\n"
          "images. The layer's Dice is the floor the threshold must clear.\n")
    for metric, label in (("erl_split", "convention A (a bridged gap splits)"),
                          ("erl_bridged", "convention B (it does not)")):
        print(f"--- {label} ---")
        print(f"    {'arm':20} {'+layer':>8} {'Dice':>7} "
              f"{'threshold':>10} {'ERL':>8} {'Dice':>7}   verdict")
        for config in ARMS:
            raw = summary.raw_of(post, config)
            dev_raw = summary.raw_of(post_dev, config)
            # `curve()` is keyed by THRESHOLD, so testing `config in curve(...)`
            # is always false and silently empties the table. Test the curve
            # for being empty instead.
            dev_curve = curve(dev_rows, config, metric)
            if raw is None or dev_raw is None or not dev_curve:
                continue
            setting = summary.pick(post_dev, config, "predicted",
                                   dev_raw[1], metric, 0.02)
            if setting is None:
                continue
            cells = [r for r in post if r["config"] == config
                     and r["source"] == "predicted"
                     and (r["along"], r["across"]) == setting]
            layer = float(np.mean([r[metric] for r in cells]))
            floor = float(np.mean([r["dice"] for r in cells]))
            chosen = best_within(dev_curve, floor)
            if chosen is None:
                print(f"    {config:20} {layer:8.1%} {floor:7.4f} "
                      f"{'--':>10}   no dev threshold that cheap")
                continue
            test_curve = curve(rows, config, metric)
            if chosen not in test_curve:
                print(f"    {config:20} {layer:8.1%} {floor:7.4f} "
                      f"{chosen:10.3f}   not scored on test yet")
                continue
            dice, traced = test_curve[chosen]
            verdict = "LAYER" if layer > traced else "threshold"
            print(f"    {config:20} {layer:8.1%} {floor:7.4f} "
                  f"{chosen:10.3f} {traced:8.1%} {dice:7.4f}   {verdict} "
                  f"({layer - traced:+.1%})")
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
    split = "dev" if "--dev" in sys.argv else "test"
    target = out_path(shard, split)

    epochs = heldout.chosen_epochs()
    # final.pt only, for frontier.py's reason: chosen_epochs() reads log.csv,
    # and a run still training has a PARTIAL log, so without this the sweep
    # silently scores models at whatever epoch they had reached.
    runs = sorted(r for r in epochs
                  if (heldout.ROOT / r / "final.pt").exists()
                  and r.rsplit("_s", 1)[0] in ARMS)
    done = set()
    for existing in files_for(split):
        done |= {row["run"] for row in csv.DictReader(existing.open())}
    runs = [r for r in runs if r not in done]
    if shard is not None:
        runs = [r for index, r in enumerate(runs) if index % shard[1] == shard[0]]
    if not runs:
        print("nothing to do")
        return

    items = drive.load_split(split)
    data = train.stack_split("fit")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(i["label"] & i["fov"])} for i in items]
    print(f"{split}: {len(runs)} run(s), {len(items)} images, "
          f"{len(THRESHOLDS)} thresholds, component filter {component_px} px",
          flush=True)

    fresh = not target.exists()
    with target.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run", "config", "seed", "epoch", "threshold",
                                "dice", "erl_split", "erl_bridged", "fg"])
        if fresh:
            writer.writeheader()
        for run in runs:
            config, seed = run.rsplit("_s", 1)
            weights = heldout.ROOT / run / f"epoch{epochs[run]:03d}.pt"
            if not weights.exists():
                print(f"  {run}: no epoch{epochs[run]:03d}.pt, skipping",
                      flush=True)
                continue
            model = train.build_model(config)
            model.load_state_dict(train.load_checkpoint(weights)["model"])
            model.eval()
            mean, std = train.normalisation(run, data)
            for row in curve_for(model, items, geometry, mean, std,
                                 component_px):
                writer.writerow({"run": run, "config": config, "seed": seed,
                                 "epoch": epochs[run], **row})
            handle.flush()
            print(f"  {run} epoch {epochs[run]} done", flush=True)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
