"""Does spending the Dice budget LOCALLY beat spending it globally?

WRITTEN AND SELFTESTED 2026-09-06, BEFORE IT SCORED ANYTHING.

WHY THIS FILE EXISTS. Two measurements in this repo point the same way and
neither has been acted on.

  1. exp/link_ceiling.py (C1.0, 2026-08-27) priced a PERFECT fragment linker
     before any linker was written: filling every severing break from the
     ground truth buys +2.9 to +4.7 points, while filling every missed
     centreline pixel buys +58 to +65 at 1.05x the foreground. Its own
     sentence: "the tree is not fragmented so much as unseen". Repairing
     topology is a small prize; SEEING the vessel is a large one.
  2. exp/composition.py, now at 24 seeds, says every operator loses to
     `lower` -- simply moving the threshold down -- on all ten arms at both
     budgets. `endpoint_shuf` matches `endpoint`, so even the endpoint result
     was about restricting WHERE to spend, not about the direction field.

Put together: the win is in recovering unseen vessel, and the best tool for
that so far is a GLOBAL threshold drop. But a global drop is indiscriminate.
It accepts a faint pixel in the middle of nowhere on the same terms as a
faint pixel continuing a confident vessel, and pays false positives for both.

HYSTERESIS is the smallest local rule that separates those two cases, and
grep says this repo has never tried it. Two thresholds: a pixel above `base`
is a seed; a pixel above `low` is kept ONLY IF its connected component
contains a seed. Same probability map, same budget, no new training, no
geometry, no direction field -- the only new ingredient is "is this faint
pixel attached to something the model was already sure about".

It is also the cheapest stand-in for the tracing methods this problem
eventually points at (NETracer, ICCV 2025, and the RoadTracer line): a tracer
follows a vessel into low contrast because it has context from where it came
from. Hysteresis has that context in its crudest possible form. If the crude
version buys nothing, a tracer is a large build on a small prize and should
be priced before it is written -- which is the C1.0 lesson.

THE BAR IS `lower`, NOT `raw`. `raw` is a strict subset of hysteresis (every
seed survives), so beating `raw` is arithmetic, not a result.

THE CONTROL IS NOT OPTIONAL. This repo has been burned twice by an operator
that turned out to be about how much foreground it painted rather than where:
`endpoint` fell to `endpoint_shuf`, and the whole-mask field fell to
`shuffled`. So `hyst_rand` keeps components of the SAME low-threshold mask,
chosen at random until it has painted as much foreground as `hyst` did,
instead of choosing the ones containing a seed. If `hyst_rand` matches
`hyst`, hysteresis is a foreground-count effect and the direction is dead.

PRE-REGISTERED 2026-09-06, before this file has scored a single run:

  1. `hyst` beats `lower` at matched Dice on at least 8 of 10 arms under BOTH
     ERL conventions, at the 0.02 budget. This is the whole hypothesis. If it
     fails, the local-decision-rule direction is dead and the next step is to
     price a tracer, not to build one.
  2. `hyst_rand` gains less than half of what `hyst` gains, on at least 8 of
     10 arms. If it matches instead, the mechanism is foreground volume and
     prediction 1 -- whatever it says -- means nothing.
  3. At matched Dice `hyst` paints LESS foreground than `lower`. That is its
     claimed mechanism stated as a number: the same Dice cost bought in a
     more useful place rather than in more places.
  4. The gap between `hyst` and `lower` is larger at the 0.05 budget than at
     0.02. A bigger budget buys a lower `low`, which is where an
     indiscriminate rule should start collecting noise fastest.

  python exp/hysteresis.py --selftest
  python exp/hysteresis.py --dev --shard 0/4
  python exp/hysteresis.py --shard 0/4
  python exp/hysteresis.py --report

Writes results/heldout/hysteresis[_dev][.shardIofN].csv.
"""
import csv
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import calibration
import composition
import cross_dataset
import drive
import hole_sweep
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import threshold_control as control
import train

ARMS = sweep.CONTROL + sweep.FRONTIER
SOURCES = ("lower", "hyst", "hyst_rand")
BUDGETS = (0.02, 0.05)
FIELDS = ["config", "run", "seed", "threshold", "image", "source",
          "erl_split", "erl_bridged", "dice", "fg"]


def hysteresis(prob: np.ndarray, fov: np.ndarray, base: float,
               low: float) -> np.ndarray:
    """Components of `prob >= low` that contain at least one `prob >= base`.

    Written out rather than taken from skimage.filters so that the
    connectivity is break_lengths.CONN8 -- the same 8-adjacency every other
    fragment decomposition in this repo uses. A rule that silently used
    4-connectivity here would be measured against tables built on 8.
    """
    seed = (prob >= base) & fov
    grow = (prob >= low) & fov
    if not seed.any():
        return seed
    labels, count = ndimage.label(grow, structure=break_lengths.CONN8)
    if count == 0:
        return seed
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[seed])] = True
    keep[0] = False
    return keep[labels]


def hysteresis_random(prob: np.ndarray, fov: np.ndarray, base: float,
                      low: float, run: str, name: str) -> np.ndarray:
    """The control: same mask, same foreground budget, components chosen at
    random instead of by whether they hold a seed.

    Seeded with zlib.crc32, never hash(): Python randomises str/tuple hashes
    per process, so the same run scored twice would draw a different control
    (the defect caught 2026-09-01 in direction_ceiling).
    """
    grow = (prob >= low) & fov
    target = int(hysteresis(prob, fov, base, low).sum())
    labels, count = ndimage.label(grow, structure=break_lengths.CONN8)
    if count == 0:
        return grow
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    rng = np.random.default_rng(
        zlib.crc32(f"{run}|{name}|{low:g}".encode()))
    order = rng.permutation(np.arange(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    painted = 0
    for label in order:
        if painted >= target:
            break
        keep[label] = True
        painted += int(sizes[label])
    return keep[labels]


def rows_for(prob, item, geo, base: float, component_px: int,
             run: str) -> list[dict]:
    """Every row for one image: the base mask and the three ways to spend."""
    fov = item["fov"]
    pred = speckle.drop_small((prob >= base) & fov, component_px)
    out = [{"source": "raw", "threshold": base,
            **sweep.measure(pred, geo["skel"], geo["truth"])}]
    for low in composition.lower_grid(base):
        down = speckle.drop_small((prob >= low) & fov, component_px)
        out.append({"source": "lower", "threshold": low,
                    **sweep.measure(down, geo["skel"], geo["truth"])})
        held = speckle.drop_small(hysteresis(prob, fov, base, low),
                                  component_px)
        out.append({"source": "hyst", "threshold": low,
                    **sweep.measure(held, geo["skel"], geo["truth"])})
        control_mask = speckle.drop_small(
            hysteresis_random(prob, fov, base, low, run, item["name"]),
            component_px)
        out.append({"source": "hyst_rand", "threshold": low,
                    **sweep.measure(control_mask, geo["skel"],
                                    geo["truth"])})
    return out


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    for arm in ARMS:
        assert arm in train.CONFIGS, arm

    # 1. THE PARTITION. Four tables in this repo silently lost seeds to
    #    hash-based sharding on 2026-09-01; every file that shards asserts it.
    work = [f"a_s{index}" for index in range(17)]
    for total in (2, 3, 4, 5):
        flat = [r for i in range(total)
                for r in sweep.shard_filter(work, (i, total))]
        assert sorted(flat) == sorted(work) and len(flat) == len(set(flat))
    print("sharding is an exact partition over 2/3/4/5 shards")

    # 2. THE OPERATOR MUST DO WHAT ITS NAME SAYS, on a case built by hand.
    #    A confident bar, a faint continuation of it, and a faint blob that
    #    touches nothing. `lower` takes both faint things; hysteresis takes
    #    only the continuation. If this does not hold the comparison below is
    #    measuring something else.
    prob = np.zeros((40, 40), dtype=np.float32)
    fov = np.ones((40, 40), dtype=bool)
    prob[20, 5:20] = 0.9        # a confident vessel
    prob[20, 20:30] = 0.3       # its faint continuation, touching it
    prob[5:9, 30:34] = 0.3      # a faint blob, touching nothing
    held = hysteresis(prob, fov, base=0.5, low=0.2)
    down = (prob >= 0.2) & fov
    assert held[20, 5:30].all(), "the continuation was not taken"
    assert not held[5:9, 30:34].any(), "the detached blob was taken"
    assert down[5:9, 30:34].all(), "the control mask should hold the blob"
    assert held.sum() < down.sum(), (held.sum(), down.sum())
    print(f"hysteresis takes the attached continuation and drops the "
          f"detached blob: {held.sum()} px against lower's {down.sum()}")

    # 3. IT MUST BE A SUPERSET OF `raw`, ALWAYS. Every seed is in a component
    #    containing a seed. This is why the bar is `lower` and not `raw`.
    seed = (prob >= 0.5) & fov
    assert held[seed].all(), "hysteresis lost a seed pixel"
    print("hysteresis contains every seed, so beating `raw` is arithmetic")

    # 4. THE CONTROL MUST BE MATCHED IN FOREGROUND AND REPRODUCIBLE. If it
    #    paints a different amount it is not a control, and if it changes
    #    between processes the column cannot be reproduced.
    first = hysteresis_random(prob, fov, 0.5, 0.2, "a_s0", "01")
    again = hysteresis_random(prob, fov, 0.5, 0.2, "a_s0", "01")
    assert np.array_equal(first, again), "control is not reproducible"
    other = hysteresis_random(prob, fov, 0.5, 0.2, "a_s1", "01")
    assert first.sum() > 0
    # Components are whole, so the match is to within one component.
    assert abs(int(first.sum()) - int(held.sum())) <= int(held.sum())
    print(f"control is reproducible ({first.sum()} px against hysteresis' "
          f"{held.sum()}) and moves with the run name "
          f"({other.sum()} px for a_s1)")

    # 5. LOWERING MUST BUY FOREGROUND MONOTONICALLY, for `lower` and for
    #    `hyst` alike. If it does not, the sweep is not a budget sweep.
    grid = [0.4, 0.3, 0.2, 0.1]
    lows = [int(((prob >= t) & fov).sum()) for t in grid]
    hysts = [int(hysteresis(prob, fov, 0.5, t).sum()) for t in grid]
    assert lows == sorted(lows), lows
    assert hysts == sorted(hysts), hysts
    assert all(h <= l for h, l in zip(hysts, lows)), (hysts, lows)
    print(f"foreground down the grid {grid}: lower {lows}, hyst {hysts} "
          f"-- monotone, and hyst never exceeds lower")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def load(split: str) -> list[dict]:
    stem = "hysteresis_dev" if split == "dev" else "hysteresis"
    rows = []
    for path in sorted(heldout.ROOT.glob(f"{stem}*.csv")):
        if split != "dev" and path.name.startswith("hysteresis_dev"):
            continue
        for row in csv.DictReader(path.open()):
            rows.append({**row, "threshold": float(row["threshold"]),
                         "dice": float(row["dice"]),
                         "fg": float(row["fg"]),
                         "erl_split": float(row["erl_split"]),
                         "erl_bridged": float(row["erl_bridged"])})
    return rows


def pick(dev_rows, config, source, floor, metric, budget):
    """Best threshold for one source within a Dice budget, chosen on dev."""
    best = None
    for value in sorted({r["threshold"] for r in dev_rows
                         if r["config"] == config
                         and r["source"] == source}):
        cells = [r for r in dev_rows if r["config"] == config
                 and r["source"] == source and r["threshold"] == value]
        if float(np.mean([r["dice"] for r in cells])) < floor - budget:
            continue
        traced = float(np.mean([r[metric] for r in cells]))
        if best is None or traced > best[1]:
            best = (value, traced)
    return None if best is None else best[0]


def report() -> None:
    dev, test = load("dev"), load("test")
    if not dev or not test:
        raise SystemExit("no hysteresis rows -- refusing to print an empty "
                         "table; an empty table is not a null result")
    seeds = sorted({r["seed"] for r in test})
    print("=== spending the Dice budget locally instead of globally ===\n")
    print(f"{len(test)} test rows, {len(dev)} dev rows, {len(seeds)} seeds.")
    print("`lower` drops the threshold everywhere. `hyst` keeps a faint pixel")
    print("only where its component holds a pixel the model was already sure")
    print("about. `hyst_rand` keeps as much foreground as `hyst` but picks")
    print("the components at random -- if it matches, the effect is volume,")
    print("not location. The bar is `lower`; `raw` is a subset of `hyst` and")
    print("beating it is arithmetic.\n")
    for metric, label in (("erl_split", "convention A (a bridged gap splits)"),
                          ("erl_bridged", "convention B (it does not)")):
        print(f"--- {label} ---")
        for config in ARMS:
            mine_dev = [r for r in dev if r["config"] == config
                        and r["source"] == "raw"]
            theirs = [r for r in test if r["config"] == config
                      and r["source"] == "raw"]
            if not mine_dev or not theirs:
                continue
            floor = float(np.mean([r["dice"] for r in mine_dev]))
            base = sorted({r["threshold"] for r in theirs})
            print(f"    {config:20} threshold "
                  f"{','.join(f'{b:g}' for b in base):10} raw "
                  f"{float(np.mean([r[metric] for r in theirs])):.1%}")
            for budget in BUDGETS:
                cells = []
                for source in SOURCES:
                    value = pick(dev, config, source, floor, metric, budget)
                    if value is None:
                        cells.append(f"{source} {'--':>18}")
                        continue
                    mine = {(r["seed"], r["image"]): r[metric] for r in test
                            if r["config"] == config
                            and r["source"] == source
                            and r["threshold"] == value}
                    base_by = {(r["seed"], r["image"]): r[metric]
                               for r in theirs}
                    pairs = sorted(set(mine) & set(base_by))
                    got_seeds = sorted({s for s, _ in pairs})
                    if len(got_seeds) < 3:
                        cells.append(f"{source} {'--':>18}")
                        continue
                    got = calibration.decide(
                        [(mine[k], base_by[k]) for k in pairs],
                        [float(np.mean([mine[k] - base_by[k]
                                        for k in pairs if k[0] == s]))
                         for s in got_seeds])
                    fg = float(np.mean(
                        [r["fg"] for r in test if r["config"] == config
                         and r["source"] == source
                         and r["threshold"] == value]))
                    cells.append(f"{source} @{value:<5g} {got['mean']:+.1%} "
                                 f"t{got['t']:5.1f} "
                                 f"{'HOLDS' if got['holds'] else 'fails'} "
                                 f"fg {fg:.0f}")
                print(f"      -{budget:.2f}  " + "   ".join(cells))
        print()
    print("Read as `source - raw`, paired on (seed, image). The question is")
    print("not whether `hyst` beats `raw` -- it must -- but whether it beats")
    print("`lower` at the same Dice budget, and whether `hyst_rand` fails to.")


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
    stem = "hysteresis_dev" if split == "dev" else "hysteresis"
    target = heldout.ROOT / (f"{stem}.csv" if shard is None else
                             f"{stem}.shard{shard[0]}of{shard[1]}.csv")

    done = set()
    for existing in sorted(heldout.ROOT.glob(f"{stem}*.csv")):
        if split != "dev" and existing.name.startswith("hysteresis_dev"):
            continue
        done |= {(r["config"], r["seed"])
                 for r in csv.DictReader(existing.open())}

    dev_rows = [r for r in control.load("dev")]
    # drive.load_split understands "dev" (the 5 selection images) and "test"
    # (all 20 report images) directly; both are named splits of the held-out
    # protocol and neither overlaps the 15 the models were fit on.
    items = drive.load_split(split)
    data = train.stack_split("fit")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(i["label"] & i["fov"]),
                 "truth": i["label"] & i["fov"]} for i in items]
    epochs = heldout.chosen_epochs()
    bases = {arm: composition.base_threshold(dev_rows, arm, "erl_bridged")
             for arm in ARMS}
    print(f"{len(ARMS)} arms, {len(items)} {split} images, "
          f"filter {component_px} px", flush=True)

    fresh = not target.exists()
    with target.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for arm in ARMS:
            runs = sorted(r for r in epochs if r.rsplit("_s", 1)[0] == arm
                          and (heldout.ROOT / r / "final.pt").exists())
            for run in sweep.shard_filter(runs, shard):
                seed = run.rsplit("_s", 1)[1]
                if (arm, seed) in done:
                    continue
                model, mean, std = sweep.load_model(run, arm, epochs, data)
                if model is None:
                    continue
                out = []
                for item, geo in zip(items, geometry):
                    prob = train.predict_full(model, item["image"], mean, std)
                    for row in rows_for(prob, item, geo, bases[arm],
                                        component_px, run):
                        out.append({"config": arm, "run": run, "seed": seed,
                                    "image": item["name"], **row})
                writer.writerows(out)
                handle.flush()
                print(f"  {run} done ({len(out)} rows)", flush=True)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
