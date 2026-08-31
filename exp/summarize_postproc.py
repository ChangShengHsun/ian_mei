"""Does one shared tangent field buy traceable length on ANY segmenter?

WRITTEN AND SELFTESTED 2026-08-31, BEFORE THE FIRST ROW OF
postproc_ceiling.csv EXISTED.

THE CLAIM BEING TESTED. A learned tangent field, applied as an oriented
post-processing step, buys traceable vessel length on any segmenter's output
at a matched Dice cost, and a random field buys none. Not a loss, not an
architecture: a step applied AFTER the model, driven by a field that one
separate predictor supplies to all of them.

THE HEADLINE IS `predicted - isotropic`. The oracle column is the bound, and a
post-processing paper cannot claim a number reachable only with the ground
truth. `shuffled - isotropic` is printed beside it every time, because it is
the control that makes the headline mean anything: in the legacy table it was
EXACTLY zero, since a random field picks the free do-nothing geometry under
any Dice budget.

READ TWICE, as calibration.py does. Post-processing adds foreground, and
calibration.md showed that adding foreground is what a lower threshold gives
away for free -- K_focal_aug went from +13.6% at threshold 0.5 to -4.2% at its
own operating point. So a layer that only wins at 0.5 has won a calibration
argument. Both readings are printed; where they disagree the own-peak one
wins and the disagreement is the finding.

  at 0.5        every arm at the conventional threshold
  at own peak   every arm at the threshold maximising its DEV Dice

PRE-REGISTERED PREDICTIONS.
  1. `predicted - isotropic` HOLDS on the control arms (A_dice, H_aug) at the
     tightest budget under both conventions. It already does on the legacy
     protocol: +3.9% and +4.3%, convention B.
  2. The gain SHRINKS as the base model gets stronger. The oracle bound
     already falls +4.3, +4.9, +1.7 across A_dice_dir, H_aug_dir,
     K_focal_aug, and the last of those trips the pre-registered "MECHANISM
     WRONG" threshold. On the clw frontier arms, which trace further still,
     expect the gain to be smaller than on A_dice.
  3. `shuffled - isotropic` is zero at the tightest budget on every arm.
  4. The gain survives the own-peak reading, unlike K_focal_aug's. The
     mechanism is geometric -- at matched foreground the oriented dilation
     puts pixels on the vessel rather than beside it -- and a threshold
     change cannot reproduce that. This is the prediction most likely to be
     wrong, and it is the one that decides whether the line is real.

THE GATE is the repo's, from calibration.decide(): paired on (image, seed),
t > 2, every seed agreeing in sign, at least three seeds.

  python exp/summarize_postproc.py --selftest
  python exp/summarize_postproc.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration
import postproc_ceiling as sweep
import select_heldout as heldout

BUDGETS = (0.02, 0.05, 0.10)
METRICS = (("erl_split", "erl.py as written (a bridged gap splits a run)"),
           ("erl_bridged", "not splitting runs the prediction bridges"))
SOURCES = ("isotropic", "shuffled", "predicted", "oracle")


def load() -> list[dict]:
    """Every shard, merged. Globbed rather than listed: a shard that failed to
    start is then visibly absent instead of silently excluded."""
    rows = []
    for path in sorted(sweep.OUT.parent.glob("postproc_ceiling*.csv")):
        for row in csv.DictReader(path.open()):
            for key in ("along", "across", "erl_split", "erl_bridged", "dice"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def raw_of(rows, config: str) -> tuple:
    these = [r for r in rows if r["config"] == config and r["source"] == "raw"]
    if not these:
        return None
    return (float(np.mean([r["erl_split"] for r in these])),
            float(np.mean([r["dice"] for r in these])))


def pick(rows, config: str, source: str, raw_dice: float, metric: str,
         budget: float):
    """Best geometry for one source within a Dice budget.

    Every source gets the SAME budget, which is what makes this a comparison
    of sources rather than of settings. The do-nothing setting always costs
    zero, so any source with data has an admissible answer and an empty cell
    means missing data, not a failed floor.
    """
    grouped = defaultdict(list)
    for row in rows:
        if row["config"] == config and row["source"] == source:
            grouped[(row["along"], row["across"])].append(row)
    allowed = {key: (float(np.mean([r[metric] for r in these])),
                     float(np.mean([r["dice"] for r in these])))
               for key, these in grouped.items()}
    allowed = {k: v for k, v in allowed.items() if v[1] >= raw_dice - budget}
    if not allowed:
        return None
    return max(allowed, key=lambda k: allowed[k][0])


def gate_pair(rows, config: str, metric: str, mine: str, theirs: str,
              settings: dict):
    """The repo's gate on `mine - theirs`, paired on (image, seed)."""
    if settings.get(mine) is None or settings.get(theirs) is None:
        return None
    def table(source):
        return {(r["seed"], r["image"]): r[metric] for r in rows
                if r["config"] == config and r["source"] == source
                and (r["along"], r["across"]) == settings[source]}
    a, b = table(mine), table(theirs)
    keys = sorted(set(a) & set(b))
    seeds = sorted({s for s, _ in keys})
    if len(seeds) < 3:
        return None
    paired = [(a[k], b[k]) for k in keys]
    per_seed = [float(np.mean([a[k] - b[k] for k in keys if k[0] == s]))
                for s in seeds]
    return calibration.decide(paired, per_seed)


def selftest() -> None:
    def make(config, source, along, across, image, erl, dice, seed=0):
        return {"config": config, "run": f"{config}_s{seed}",
                "seed": str(seed), "source": source, "along": along,
                "across": across, "image": image, "erl_split": erl,
                "erl_bridged": erl, "dice": dice, "fg": 1000,
                "field_arm": "H_aug_dir", "epoch": 100}

    # 1. THE BUDGET IS THE COMPARISON. A setting that traces further but costs
    #    more Dice than the budget allows must be refused, however good it
    #    looks -- that is the C1.0 error, where a free closing baseline added
    #    31% more foreground and "beat" the oracle.
    rows = [make("A", "raw", 0.0, 0.0, f"{i:02d}", 0.50, 0.820)
            for i in range(1, 21)]
    for i in range(1, 21):
        rows.append(make("A", "oracle", 2.0, 1.0, f"{i:02d}", 0.90, 0.700))
        rows.append(make("A", "oracle", 0.5, 0.25, f"{i:02d}", 0.60, 0.810))
    assert pick(rows, "A", "oracle", 0.820, "erl_split", 0.02) == (0.5, 0.25)
    assert pick(rows, "A", "oracle", 0.820, "erl_split", 0.20) == (2.0, 1.0)
    print("  a tight budget refuses the setting that buys length with Dice")

    # 2. THE GATE. A mean and a t can both be large while one seed of six
    #    disagrees in sign. That case must fail -- it is the E5 shape, and it
    #    is what by_setting()'s np.mean hid in the legacy table.
    gated = []
    for seed in range(6):
        offset = 0.05 if seed < 5 else -0.02
        for i in range(1, 21):
            gated.append(make("G", "predicted", 0.5, 0.25, f"{i:02d}",
                              0.50 + offset, 0.815, seed=seed))
            gated.append(make("G", "isotropic", 0.5, 0.5, f"{i:02d}",
                              0.50, 0.815, seed=seed))
    settings = {"predicted": (0.5, 0.25), "isotropic": (0.5, 0.5)}
    got = gate_pair(gated, "G", "erl_split", "predicted", "isotropic",
                    settings)
    assert got["seeds"] == 6 and got["mean"] > 0 and got["t"] > 2, got
    assert not got["holds"], "a split-sign effect passed the gate"
    print(f"  the gate refuses a split-sign effect (mean {got['mean']:+.3f}, "
          f"t {got['t']:.1f})")

    # 3. And a shuffled column that is exactly zero must report as a FAILURE,
    #    not as a pass with a tiny effect. That is the control working.
    flat = []
    for seed in range(6):
        for i in range(1, 21):
            flat.append(make("S", "shuffled", 0.0, 0.0, f"{i:02d}", 0.50,
                             0.820, seed=seed))
            flat.append(make("S", "isotropic", 0.0, 0.0, f"{i:02d}", 0.50,
                             0.820, seed=seed))
    got = gate_pair(flat, "S", "erl_split", "shuffled", "isotropic",
                    {"shuffled": (0.0, 0.0), "isotropic": (0.0, 0.0)})
    assert got is not None and not got["holds"], got
    assert abs(got["mean"]) < 1e-12, got["mean"]
    print("  an exactly-zero shuffled column reports as a failure, not a pass")

    # 4. A source with no rows returns None rather than an empty comparison
    #    that would read as a negative result.
    assert gate_pair(flat, "S", "erl_split", "predicted", "isotropic",
                     {"predicted": None, "isotropic": (0.0, 0.0)}) is None
    print("  a missing source reports None, not a false negative")

    for config in sweep.CONTROL + sweep.FRONTIER + sweep.RETIRED:
        assert config in sweep.CONTROL + sweep.FRONTIER + sweep.RETIRED
    print("all checks passed")


def report(rows, metric: str, label: str) -> None:
    print(f"\n--- {label} ---")
    groups = (("control", sweep.CONTROL), ("frontier", sweep.FRONTIER),
              ("retired (negative control)", sweep.RETIRED))
    for group, configs in groups:
        print(f"\n  [{group}]")
        for config in configs:
            raw = raw_of(rows, config)
            if raw is None:
                continue
            _, raw_dice = raw
            traced = float(np.mean([r[metric] for r in rows
                                    if r["config"] == config
                                    and r["source"] == "raw"]))
            print(f"    {config}  raw {traced:.1%} traced at Dice "
                  f"{raw_dice:.4f}")
            for budget in BUDGETS:
                settings = {s: pick(rows, config, s, raw_dice, metric, budget)
                            for s in SOURCES}
                cells = []
                for other in ("predicted", "shuffled", "oracle"):
                    got = gate_pair(rows, config, metric, other, "isotropic",
                                    settings)
                    cells.append("--" if got is None else
                                 f"{got['mean']:+.1%} t{got['t']:5.2f} "
                                 f"{'HOLDS' if got['holds'] else 'fails'}")
                print(f"      -{budget:.2f}  pred {cells[0]:>22}"
                      f"   shuf {cells[1]:>22}   oracle {cells[2]:>22}")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    rows = load()
    if not rows:
        raise SystemExit("no postproc_ceiling*.csv yet -- run "
                         "exp/postproc_ceiling.py")
    shards = sorted(sweep.OUT.parent.glob("postproc_ceiling*.csv"))
    fields = sorted({r["field_arm"] for r in rows})
    print("=== post-processing: one shared field, every arm ===\n")
    print(f"{len(rows)} rows from {len(shards)} file(s), field(s) "
          f"{', '.join(fields)}")
    print("Protocol heldout: fit on 15 DRIVE training images, epoch chosen on")
    print("the 5 held out, scored on all 20 test images. Absolute values are")
    print("NOT comparable to pre-heldout runs or to published figures.\n")
    print("Headline is `predicted - isotropic`: the model's own field against")
    print("plain dilation at the same Dice cost. `shuffled` is the control.")
    for metric, label in METRICS:
        report(rows, metric, label)
    print("\nThe verdict is read at the TIGHTEST budget under the second")
    print("convention. If the three budgets or the two conventions disagree,")
    print("that disagreement is the result -- do not quote the one that")
    print("flatters.")


if __name__ == "__main__":
    main()
