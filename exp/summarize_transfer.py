"""E17's verdict, with an honest split between what is pre-registered and what
is not.

I smoke-tested transfer.py on seed 0 before writing this file, to check the
scoring worked at all. That smoke test showed the augmentation arm transferring
far better than the baseline. So:

  - Criterion A (augmentation transfers better) is NOT pre-registered. I have
    seen one seed of it. It is written here as a confirmation on the seeds I
    had not seen, and it is labelled that way in the output rather than being
    quietly presented as a prediction.
  - Criterion B (LIOT beats grey under transfer) IS pre-registered: no J_liot
    checkpoint existed when this file was written. This is the criterion that
    matters, because cross-dataset generalisation is LIOT's published claim
    and a same-dataset test cannot see it.

Pretending A was a prediction would cost nothing today and make every future
pre-registration in this series worth less.

  python exp/summarize_transfer.py --selftest
  python exp/summarize_transfer.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = Path(__file__).resolve().parent / "results"
BASELINE, GREY, LIOT = "A_dice", "H_aug", "J_liot"
DATASETS = ("stare", "hrf")


def load() -> list[dict]:
    path = RESULTS / "transfer.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run transfer.py first")
    rows = []
    for row in csv.DictReader(path.open()):
        for key in ("dice", "cldice", "betti0_err", "severs", "erl",
                    "erl_ceiling"):
            row[key] = float(row[key])
        row["seed"] = row["run"].rsplit("_s", 1)[1]
        rows.append(row)
    return rows


def compare(rows, better: str, worse: str, dataset: str, key: str) -> dict:
    """Paired by (seed, image), plus the per-seed means the gate needs.

    Pairing on the seed as well as the image is what E5 forced: 400 image
    pairs drawn from 3 trainings will report a tiny p-value for a difference
    that flips sign between trainings.
    """
    left = {(r["seed"], r["image"]): r[key] for r in rows
            if r["config"] == better and r["dataset"] == dataset}
    right = {(r["seed"], r["image"]): r[key] for r in rows
             if r["config"] == worse and r["dataset"] == dataset}
    shared = sorted(set(left) & set(right))
    if not shared:
        return {}
    diffs = [left[k] - right[k] for k in shared]
    per_seed = []
    for seed in sorted({k[0] for k in shared}):
        picked = [left[k] - right[k] for k in shared if k[0] == seed]
        per_seed.append(float(np.mean(picked)))
    return {"mean": float(np.mean(diffs)),
            "t": float(stats.ttest_rel([left[k] for k in shared],
                                       [right[k] for k in shared]).statistic),
            "per_seed": per_seed,
            "gate": (all(d > 0 for d in per_seed)
                     or all(d < 0 for d in per_seed))}


def report(label: str, result: dict, want: str, registered: bool) -> None:
    tag = "pre-registered" if registered else "NOT pre-registered (seed 0 seen)"
    if not result:
        print(f"  [----] {label}\n         no paired data yet [{tag}]")
        return
    sign_ok = result["mean"] > 0 if want == "better" else result["mean"] < 0
    t_ok = result["t"] > 2 if want == "better" else result["t"] < -2
    verdict = "PASS" if (sign_ok and t_ok and result["gate"]) else "FAIL"
    per = " ".join(f"{d:+.4f}" for d in result["per_seed"])
    print(f"  [{verdict}] {label}")
    print(f"         {result['mean']:+.4f}  t={result['t']:.2f}  "
          f"n_seeds={len(result['per_seed'])}  per seed: {per}")
    print(f"         [{tag}]")


def selftest() -> None:
    """A difference that is large and consistent per image but flips between
    seeds must be rejected -- the E5 failure, in this file's own pairing."""
    rows = []
    rng = np.random.default_rng(0)
    for seed, offset in (("0", +0.05), ("1", -0.04), ("2", +0.06)):
        for index in range(20):
            base = rng.normal()
            rows.append({"config": GREY, "seed": seed, "dataset": "stare",
                         "image": f"im{index}", "dice": base + offset})
            rows.append({"config": BASELINE, "seed": seed, "dataset": "stare",
                         "image": f"im{index}", "dice": base})
    result = compare(rows, GREY, BASELINE, "stare", "dice")
    assert not result["gate"], result
    assert abs(result["t"]) > 3, result
    print(f"a seed-flipping difference gives |t|={abs(result['t']):.1f} "
          f"and is still rejected by the gate")

    for row in rows:
        if row["config"] == GREY:
            row["dice"] += 0.09          # now every seed is positive
    agreed = compare(rows, GREY, BASELINE, "stare", "dice")
    assert agreed["gate"], agreed
    print(f"with every seed agreeing it passes: per seed "
          f"{[round(d, 3) for d in agreed['per_seed']]}")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return

    rows = load()
    configs = sorted({r["config"] for r in rows})
    for dataset in DATASETS:
        picked = [r for r in rows if r["dataset"] == dataset]
        if not picked:
            continue
        print(f"\n=== {dataset}: zero-shot from DRIVE ===")
        print(f"{'config':14}{'dice':>8}{'clDice':>8}{'b0err':>9}"
              f"{'severs':>8}{'ERL/ceiling':>13}")
        for config in configs:
            group = [r for r in picked if r["config"] == config]
            if not group:
                continue
            def avg(key):
                return float(np.nanmean([r[key] for r in group]))
            print(f"{config:14}{avg('dice'):8.4f}{avg('cldice'):8.4f}"
                  f"{avg('betti0_err'):9.1f}{avg('severs'):8.1f}"
                  f"{avg('erl') / avg('erl_ceiling'):13.3f}")

    print("\n=== E17 criteria ===\n")
    for dataset in DATASETS:
        print(f"-- {dataset} --")
        report(f"A. augmentation beats the baseline under transfer",
               compare(rows, GREY, BASELINE, dataset, "dice"), "better",
               registered=False)
        report(f"B. LIOT beats grey under transfer (LIOT's published claim)",
               compare(rows, LIOT, GREY, dataset, "dice"), "better",
               registered=True)
        report(f"C. LIOT cuts severing breaks under transfer",
               compare(rows, LIOT, GREY, dataset, "severs"), "fewer",
               registered=True)
        print()


if __name__ == "__main__":
    main()
