"""How many seeds does a verdict need before it stops changing its mind?

WRITTEN AND SELFTESTED 2026-09-03, BEFORE IT REPORTED ANYTHING.

WHAT FORCED IT. run_seeds6.sh pre-registered that no cell should move from
HOLDS to fails by adding seeds. Going 3 -> 6 moved eight. The written
consequence was that the 6-seed table becomes primary. Going 6 -> 12 then
moved five more -- but this time the effect sizes did NOT shrink (median
|effect| ratio 0.97) while |t| grew about as fast as sqrt(n) predicts (median
1.60 against sqrt(2) = 1.41). So the flips at twelve seeds are not the effect
settling down. Three of the four HOLDS -> fails flips happened with a LARGER
t than the pass they replaced.

THE MECHANISM, and the reason this file exists. calibration.decide requires
mean > 0 AND t > 2 AND *every* per-seed difference positive. The first two
get easier with more seeds; the third gets strictly harder, because each new
seed is another chance to draw a negative. So the gate is NOT monotone in the
seed count, and "this cell passed" is partly a statement about how few seeds
were run. That is a property of the instrument, not of the arms, and it is
measurable without a single new training run: resample the twelve seeds
already on disk at k = 3..12 and count how often each cell passes.

WHAT IS REPORTED. For every cell of the transfer_calibration table and of the
transfer_postproc table, the fraction of subsamples of size k that pass the
gate. A cell at 100% is a claim. A cell at 55% is a coin flip that a paper
would report as a finding.

SCOPE, stated rather than left to be inferred. On the postproc side the
geometry and the threshold are held at the choice made on the FULL dev set;
only the seeds entering the gate are resampled. That isolates the gate's seed
dependence from the setting choice's, which is the flip that was observed.

  python exp/seed_stability.py --selftest
  python exp/seed_stability.py --report

Writes nothing; prints. Reads the csvs the two tables already wrote.
"""
import itertools
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration
import composition
import transfer_calibration as calib
import transfer_postproc as tpp

KS = (3, 4, 5, 6, 8, 10, 12)
MAX_DRAWS = 2000
SHARED = calib.SHARED


def draws(seeds: list, k: int) -> list:
    """Every subset of size k, or a reproducible sample of MAX_DRAWS of them.

    zlib.crc32, never hash(): Python randomises str/tuple hashes per process,
    so a table built twice would resample differently and the column could not
    be reproduced. Same defect as the sharding bug of 2026-09-01.
    """
    if k > len(seeds):
        return []
    total = 1
    for step in range(k):
        total = total * (len(seeds) - step) // (step + 1)
    if total <= MAX_DRAWS:
        return [list(c) for c in itertools.combinations(seeds, k)]
    rng = np.random.default_rng(zlib.crc32(f"{sorted(seeds)}|{k}".encode()))
    seen, out = set(), []
    while len(out) < MAX_DRAWS:
        pick = tuple(sorted(rng.choice(len(seeds), size=k, replace=False)))
        if pick in seen:
            continue
        seen.add(pick)
        out.append([seeds[index] for index in pick])
    return out


def pass_rate(per_seed: dict, k: int) -> float:
    """Fraction of k-seed subsamples where this cell passes the gate.

    `per_seed` maps a seed to its list of (mine, theirs) pairs -- one pair per
    seed for the calibration table, one per image for the postproc table.
    """
    subsets = draws(sorted(per_seed), k)
    if not subsets:
        return float("nan")
    passed = 0
    for chosen in subsets:
        paired = [pair for seed in chosen for pair in per_seed[seed]]
        means = [float(np.mean([a - b for a, b in per_seed[seed]]))
                 for seed in chosen]
        if calibration.decide(paired, means)["holds"]:
            passed += 1
    return passed / len(subsets)


# ------------------------------------------------------- calibration table

def calibration_cells(dataset: str) -> dict:
    """{(metric, base, arm, label): {seed: [(mine, theirs)]}}.

    Rebuilt here rather than imported because transfer_calibration.report
    prints instead of returning. The selftest checks this reconstruction
    against that report's own output, so the duplication cannot drift
    silently -- which is the only reason duplicating it is allowed.
    """
    rows = calib.load(dataset)
    seeds = sorted({r["seed"] for r in rows})
    out = {}
    for metric in ("erl_split", "erl_bridged"):
        at_shared, at_own = {}, {}
        for arm in calib.ARMS:
            for seed in seeds:
                dev = [r for r in rows if r["config"] == arm
                       and r["seed"] == seed and r["split"] == "dev"]
                test = [r for r in rows if r["config"] == arm
                        and r["seed"] == seed and r["split"] == "test"]
                if not dev or not test:
                    continue
                by = {r["threshold"]: r for r in test}
                peak = calib.peak_of(dev)
                if SHARED in by:
                    at_shared.setdefault(arm, {})[seed] = by[SHARED][metric]
                if peak in by:
                    at_own.setdefault(arm, {})[seed] = by[peak][metric]
        for base in ("A_dice", "H_aug"):
            for arm in calib.ARMS:
                if arm == base or (base == "H_aug" and arm == "A_dice"):
                    continue
                for label, table in (("at 0.5", at_shared), ("at own", at_own)):
                    mine, theirs = table.get(arm, {}), table.get(base, {})
                    common = sorted(set(mine) & set(theirs))
                    if len(common) < 3:
                        continue
                    out[(metric, base, arm, label)] = {
                        seed: [(mine[seed], theirs[seed])] for seed in common}
    return out


# ---------------------------------------------------------- postproc table

def postproc_cells(dataset: str, budget: float) -> dict:
    """{(metric, source, arm): {seed: [(mine, theirs) per image]}}."""
    rows = tpp.load(dataset)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    out = {}
    for metric in ("erl_split", "erl_bridged"):
        for arm in tpp.ARMS:
            raw = [r for r in test if r["config"] == arm
                   and r["source"] == "raw"]
            dev_raw = [r for r in dev if r["config"] == arm
                       and r["source"] == "raw"]
            if not raw or not dev_raw:
                continue
            floor = float(np.mean([r["dice"] for r in dev_raw]))
            theirs = {(r["seed"], r["image"]): r[metric] for r in raw}
            for source in tpp.SOURCES:
                value = tpp.pick(dev, arm, source, floor, metric, budget)
                if value is None:
                    continue
                key = tpp.setting_key(source)
                mine = {(r["seed"], r["image"]): r[metric] for r in test
                        if r["config"] == arm and r["source"] == source
                        and r[key] == value}
                per_seed = {}
                for pair_key in sorted(set(mine) & set(theirs)):
                    per_seed.setdefault(pair_key[0], []).append(
                        (mine[pair_key], theirs[pair_key]))
                if len(per_seed) >= 3:
                    out[(metric, source, arm)] = per_seed
    return out


# ------------------------------------------------------------ DRIVE table

def drive_cells(budget: float) -> dict:
    """{(metric, source, arm): {seed: [(mine, theirs) per image]}}.

    The DRIVE side of the same question. composition.py holds the headline --
    the post-processing layer priced against simply lowering the threshold --
    and a headline that has not been seed-audited is exactly the kind of cell
    this file exists to find.
    """
    rows, dev_rows = composition.load("test"), composition.load("dev")
    out = {}
    for metric in ("erl_split", "erl_bridged"):
        for arm in composition.ARMS:
            floor = composition.raw_of(dev_rows, arm)
            theirs = {(r["seed"], r["image"]): r[metric] for r in rows
                      if r["config"] == arm and r["source"] == "raw"}
            if floor is None or not theirs:
                continue
            for source in ("lower", "endpoint", "endpoint_shuf",
                           "endpoint_iso", "predicted"):
                if source == "lower":
                    value = composition.pick_lower(dev_rows, arm, floor,
                                                   metric, budget)
                    match = (lambda r, v=value: r["threshold"] == v)
                else:
                    value = composition.pick(dev_rows, arm, source, floor,
                                             metric, budget)
                    match = (lambda r, v=value: (r["along"], r["across"]) == v)
                if value is None:
                    continue
                mine = {(r["seed"], r["image"]): r[metric] for r in rows
                        if r["config"] == arm and r["source"] == source
                        and match(r)}
                per_seed = {}
                for key in sorted(set(mine) & set(theirs)):
                    per_seed.setdefault(key[0], []).append(
                        (mine[key], theirs[key]))
                if len(per_seed) >= 3:
                    out[(metric, source, arm)] = per_seed
    return out


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    # 1. THE DRAW MUST BE A DRAW. Exhaustive below the cap, reproducible
    #    above it, never a duplicate, always the right size.
    small = draws(list(range(6)), 3)
    assert len(small) == 20 and len(set(map(tuple, small))) == 20, len(small)
    big = draws([str(n) for n in range(30)], 5)
    assert len(big) == MAX_DRAWS and len(set(map(tuple, big))) == MAX_DRAWS
    assert all(len(d) == 5 for d in big)
    again = draws([str(n) for n in range(30)], 5)
    assert [tuple(d) for d in big] == [tuple(d) for d in again], "not stable"
    assert draws(list(range(4)), 9) == []
    print(f"draws: exhaustive at C(6,3)={len(small)}, capped and reproducible "
          f"at C(30,5) -> {len(big)}")

    # 2. THE GATE'S SIGN RULE MUST BITE, and this is the whole claim of the
    #    file: a cell that passes on every 3-subset of a set containing one
    #    negative seed must NOT pass at k = all.
    per_seed = {s: [(1.0, 0.0)] for s in range(5)}
    per_seed[5] = [(-1.0, 0.0)]
    assert pass_rate(per_seed, 6) == 0.0
    partial = pass_rate(per_seed, 3)
    assert 0.0 < partial < 1.0, partial
    expected = 1 - (10 / 20)   # C(5,3)=10 all-positive of C(6,3)=20
    assert abs(partial - expected) < 1e-9, (partial, expected)
    print(f"one negative seed in six: passes {partial:.0%} of 3-subsets, "
          f"{pass_rate(per_seed, 6):.0%} of 6-subsets -- the gate is not "
          f"monotone in k")

    # 3. THE RECONSTRUCTION MUST MATCH THE SHIPPED TABLE. This file rebuilds
    #    transfer_calibration's cells; if the rebuild drifts, every column
    #    here is wrong in a way no number would look odd about. Checked
    #    against that script's own report text, cell by cell.
    verdict = Path("exp/results/transfer_calibration_verdict.txt")
    if not verdict.exists():
        print("no transfer_calibration_verdict.txt -- reconstruction check "
              "SKIPPED (run exp/transfer_calibration.py --report first)")
        return
    import re
    shipped, dataset, metric, base = {}, None, None, None
    for line in verdict.read_text().splitlines():
        found = re.match(r"--- (\w+) \(", line)
        if found:
            dataset = found.group(1)
        elif line.strip() in ("erl_split", "erl_bridged"):
            metric = line.strip()
        elif line.strip().startswith("vs "):
            base = line.strip()[3:]
        else:
            found = re.match(r"\s+(\w+)\s+peak\s+[\d.]+\s+"
                             r"at 0\.5 ([+-][\d.]+)% t\s*[-\d.]+ (\w+)\s+"
                             r"at own ([+-][\d.]+)% t\s*[-\d.]+ (\w+)", line)
            if found:
                arm = found.group(1)
                shipped[(dataset, metric, base, arm, "at 0.5")] = (
                    float(found.group(2)), found.group(3))
                shipped[(dataset, metric, base, arm, "at own")] = (
                    float(found.group(4)), found.group(5))
    checked = 0
    for dataset in calib.DATASETS:
        for key, per_seed in calibration_cells(dataset).items():
            metric, base, arm, label = key
            want = shipped.get((dataset, metric, base, arm, label))
            if want is None:
                continue
            got = calibration.decide(
                [p for seed in sorted(per_seed) for p in per_seed[seed]],
                [float(np.mean([a - b for a, b in per_seed[seed]]))
                 for seed in sorted(per_seed)])
            assert abs(got["mean"] * 100 - want[0]) < 0.06, (key, got, want)
            assert ("HOLDS" if got["holds"] else "fails") == want[1], (key,
                                                                      want)
            checked += 1
    assert checked >= 50, f"only {checked} cells cross-checked"
    print(f"reconstruction matches the shipped verdict on {checked} cells, "
          f"mean and verdict both")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def bar(rate: float) -> str:
    if rate != rate:
        return "  -- "
    return f"{rate:4.0%} "


def report() -> None:
    print("=== how many seeds does a verdict need? ===\n")
    print("Each row resamples the seeds ALREADY on disk. The number under k")
    print("is the fraction of k-seed subsamples in which that cell passes")
    print("calibration.decide -- mean > 0, t > 2, and every per-seed")
    print("difference positive. The third condition gets HARDER with more")
    print("seeds, so a column that falls as k grows is the gate tightening,")
    print("not the effect vanishing. A cell that a paper would print as a")
    print("finding should read 100% at the seed count the paper ran.\n")
    header = "  " + " ".join(f" k={k:<3}" for k in KS)

    print("--- transfer_calibration: arm vs baseline, paired on seed ---")
    for dataset in calib.DATASETS:
        cells = calibration_cells(dataset)
        if not cells:
            continue
        have = max(len(v) for v in cells.values())
        print(f"  {dataset} ({have} seeds on disk)")
        print(f"    {'cell':46}{header}")
        for key in sorted(cells):
            metric, base, arm, label = key
            rates = [pass_rate(cells[key], k) for k in KS]
            if all(r != r or r == 0.0 for r in rates):
                continue
            name = f"{metric[4:]:8} {arm:12} vs {base:8} {label}"
            print(f"    {name:46}" + " ".join(bar(r) for r in rates))
        print()

    print("--- DRIVE composition: source vs raw at -0.02 Dice, paired on "
          "(seed, image) ---")
    cells = drive_cells(0.02)
    if cells:
        have = max(len(v) for v in cells.values())
        print(f"  drive ({have} seeds on disk)")
        print(f"    {'cell':46}{header}")
        for key in sorted(cells):
            metric, source, arm = key
            rates = [pass_rate(cells[key], k) for k in KS]
            name = f"{metric[4:]:8} {arm:18} {source}"
            print(f"    {name:46}" + " ".join(bar(r) for r in rates))
        print()

    print("--- transfer_postproc: source vs raw at -0.02 Dice, paired on "
          "(seed, image) ---")
    for dataset in calib.DATASETS:
        cells = postproc_cells(dataset, 0.02)
        if not cells:
            continue
        have = max(len(v) for v in cells.values())
        print(f"  {dataset} ({have} seeds on disk)")
        print(f"    {'cell':46}{header}")
        for key in sorted(cells):
            metric, source, arm = key
            rates = [pass_rate(cells[key], k) for k in KS]
            name = f"{metric[4:]:8} {arm:12} {source}"
            print(f"    {name:46}" + " ".join(bar(r) for r in rates))
        print()

    print("A column that reads 100% at every k is a result. A column that")
    print("climbs with k is an effect the earlier tables were underpowered")
    print("to see. A column that FALLS with k is the sign rule tightening --")
    print("that cell's pass at the smaller k was a statement about how few")
    print("seeds were run, and it is the one shape that cannot be fixed by")
    print("running more of them.")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    report()


if __name__ == "__main__":
    main()
