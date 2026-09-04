"""Does the ERL convention change the NUMBERS, or does it change the WINNER?

WRITTEN AND SELFTESTED 2026-09-03, BEFORE IT REPORTED ANYTHING.

WHAT IS ALREADY KNOWN. erl_reference.py settled that 19.9 of a measured
36.6-point gap came from the splitting rule alone -- whether a bridged gap
counts as breaking a run. CLAUDE.md's standing rule is therefore to report
both conventions and to treat a disagreement as the result.

WHAT IS NOT KNOWN, and what this file measures. That earlier finding is about
MAGNITUDE. A reader can accept it and still assume the two conventions rank
methods the same way, so that picking one is a presentation choice. The
composition and transfer_postproc tables of 2026-09-03 suggest otherwise: on
DRIVE, `lower` beats every operator 10 of 10 under convention A and only 6 of
10 under B; off DRIVE it is 24 of 24 under A and 13 of 24 under B. If the
WINNER changes, the convention is not a presentation choice -- it decides
whether a method works, and every paper that reports one convention has made
that decision without saying so.

HOW A WINNER IS DECIDED. Per arm and per Dice budget, among the real
candidates -- never the shuffled controls, which exist to fail -- the source
with the largest mean gain over `raw` that also passes calibration.decide.
If nothing passes, the winner is `raw`: the threshold alone, no operator.
That is a legitimate winner and the most common honest answer.

  python exp/convention_flip.py --selftest
  python exp/convention_flip.py --report

Reads only csvs that already exist. No GPU.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration
import composition
import transfer_postproc as tpp

BUDGETS = (0.02, 0.05)
METRICS = (("erl_split", "A"), ("erl_bridged", "B"))
# The controls are excluded on purpose: `shuffled` and `endpoint_shuf` are
# there to fail, and letting a control "win" a cell would turn an instrument
# check into a method claim.
DRIVE_SOURCES = ("lower", "predicted", "endpoint", "endpoint_iso",
                 "isotropic")
NONE = "raw"


def decide_cell(mine: dict, theirs: dict) -> dict | None:
    """The gate, on a {(seed, image): value} pair of tables."""
    keys = sorted(set(mine) & set(theirs))
    seeds = sorted({seed for seed, _ in keys})
    if len(seeds) < 3:
        return None
    return calibration.decide(
        [(mine[k], theirs[k]) for k in keys],
        [float(np.mean([mine[k] - theirs[k] for k in keys if k[0] == s]))
         for s in seeds])


def drive_winner(rows, dev_rows, config, metric, budget):
    """(winner, gain) for one DRIVE cell of composition.py's table."""
    floor = composition.raw_of(dev_rows, config)
    theirs = {(r["seed"], r["image"]): r[metric] for r in rows
              if r["config"] == config and r["source"] == "raw"}
    if floor is None or not theirs:
        return None
    best = (NONE, 0.0)
    for source in DRIVE_SOURCES:
        if source == "lower":
            value = composition.pick_lower(dev_rows, config, floor, metric,
                                           budget)
            match = (lambda r, v=value: r["threshold"] == v)
        else:
            value = composition.pick(dev_rows, config, source, floor, metric,
                                     budget)
            match = (lambda r, v=value: (r["along"], r["across"]) == v)
        if value is None:
            continue
        mine = {(r["seed"], r["image"]): r[metric] for r in rows
                if r["config"] == config and r["source"] == source
                and match(r)}
        got = decide_cell(mine, theirs)
        if got and got["holds"] and got["mean"] > best[1]:
            best = (source, got["mean"])
    return best


def transfer_winner(rows, config, metric, budget):
    """(winner, gain) for one cell of the transfer_postproc table."""
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    dev_raw = [r for r in dev if r["config"] == config
               and r["source"] == "raw"]
    theirs = {(r["seed"], r["image"]): r[metric] for r in test
              if r["config"] == config and r["source"] == "raw"}
    if not dev_raw or not theirs:
        return None
    floor = float(np.mean([r["dice"] for r in dev_raw]))
    best = (NONE, 0.0)
    for source in tpp.SOURCES:
        value = tpp.pick(dev, config, source, floor, metric, budget)
        if value is None:
            continue
        key = tpp.setting_key(source)
        mine = {(r["seed"], r["image"]): r[metric] for r in test
                if r["config"] == config and r["source"] == source
                and r[key] == value}
        got = decide_cell(mine, theirs)
        if got and got["holds"] and got["mean"] > best[1]:
            best = (source, got["mean"])
    return best


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    # 1. NO CONTROL MAY BE A CANDIDATE. If a shuffled arm can win a cell, a
    #    disagreement between conventions could be manufactured by noise in
    #    the control rather than by the operators.
    for source in DRIVE_SOURCES + tuple(tpp.SOURCES):
        assert "shuf" not in source, source
    assert "shuffled" not in DRIVE_SOURCES
    print(f"candidates carry no control arm: DRIVE {DRIVE_SOURCES}, "
          f"transfer {tuple(tpp.SOURCES)}")

    # 2. THE GATE MUST STILL BE THE GATE, and must refuse under three seeds
    #    rather than answering. That refusal is what stops an under-covered
    #    dataset from silently reporting a winner.
    two = {("0", "a"): 1.0, ("1", "a"): 1.0}
    assert decide_cell(two, {k: 0.0 for k in two}) is None
    three = {(str(s), "a"): 1.0 for s in range(3)}
    got = decide_cell(three, {k: 0.0 for k in three})
    assert got and got["holds"], got
    flipped = dict(three); flipped[("2", "a")] = -1.0
    assert not decide_cell(flipped, {k: 0.0 for k in three})["holds"]
    print("gate: refuses at 2 seeds, passes at 3, fails when one seed "
          "disagrees in sign")

    # 3. A CELL WITH NOTHING PASSING MUST NAME `raw`, not None. An empty
    #    winner column and a "the threshold alone won" column mean opposite
    #    things, and CLAUDE.md's rule is that an empty table is not a null
    #    result.
    rows = composition.load("test")
    dev_rows = composition.load("dev")
    assert rows and dev_rows, "no composition rows on disk"
    got = drive_winner(rows, dev_rows, composition.ARMS[0], "erl_split", 0.02)
    assert got is not None and isinstance(got[0], str), got
    print(f"DRIVE cell {composition.ARMS[0]}/A/-0.02 resolves to "
          f"{got[0]!r} (+{100 * got[1]:.1f} points)")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def report() -> None:
    print("=== does the convention change the numbers, or the winner? ===\n")
    print("Per arm and budget, the source with the largest gain over `raw`")
    print("that passes the gate. `raw` wins a cell when nothing passes --")
    print("the threshold alone, no operator, which is a real answer.")
    print("Controls are excluded from the candidates by construction.\n")

    flips = agree = 0
    rows, dev_rows = composition.load("test"), composition.load("dev")
    if rows and dev_rows:
        print("--- DRIVE (composition.py) ---")
        print(f"    {'arm':20}{'budget':>8}   {'convention A':22}"
              f"{'convention B':22}")
        for config in composition.ARMS:
            for budget in BUDGETS:
                got = {}
                for metric, label in METRICS:
                    got[label] = drive_winner(rows, dev_rows, config, metric,
                                              budget)
                if any(v is None for v in got.values()):
                    continue
                same = got["A"][0] == got["B"][0]
                agree, flips = agree + same, flips + (not same)
                cells = [f"{got[k][0]} {100 * got[k][1]:+.1f}"
                         for k in ("A", "B")]
                mark = "" if same else "  <-- FLIP"
                print(f"    {config:20}{-budget:>8.2f}   {cells[0]:22}"
                      f"{cells[1]:22}{mark}")
        print()

    for dataset in tpp.DATASETS:
        rows = tpp.load(dataset)
        if not rows:
            continue
        print(f"--- {dataset} (transfer_postproc.py) ---")
        print(f"    {'arm':20}{'budget':>8}   {'convention A':22}"
              f"{'convention B':22}")
        for config in tpp.ARMS:
            for budget in BUDGETS:
                got = {}
                for metric, label in METRICS:
                    got[label] = transfer_winner(rows, config, metric, budget)
                if any(v is None for v in got.values()):
                    continue
                same = got["A"][0] == got["B"][0]
                agree, flips = agree + same, flips + (not same)
                cells = [f"{got[k][0]} {100 * got[k][1]:+.1f}"
                         for k in ("A", "B")]
                mark = "" if same else "  <-- FLIP"
                print(f"    {config:20}{-budget:>8.2f}   {cells[0]:22}"
                      f"{cells[1]:22}{mark}")
        print()

    total = agree + flips
    if not total:
        raise SystemExit("no cells resolved -- refusing to print an empty "
                         "table; an empty table is not a null result")
    print("--- the count ---")
    print(f"    cells resolved      {total}")
    print(f"    same winner         {agree}  ({agree / total:.0%})")
    print(f"    winner FLIPS        {flips}  ({flips / total:.0%})")
    print()
    print("A flip means the two conventions disagree about which operator to")
    print("use, on the same predictions, at the same Dice budget, through the")
    print("same gate. Reporting one convention is then not a presentation")
    print("choice; it is an undeclared decision about what the method is.")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    report()


if __name__ == "__main__":
    main()
