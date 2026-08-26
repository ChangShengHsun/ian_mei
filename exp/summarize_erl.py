"""The paired ERL comparison, with the seed gate. Written 2026-08-26.

E18's headline -- "traces 44.7% of the tree against the baseline's 26.4%" --
was computed ad hoc, with no script and no selftest, on three seeds. E15's own
rule makes ERL the arbiter whenever Dice and the topology metrics disagree, so
the arbiter has been the one number in this series with no verdict script
behind it. This file is that script.

Two things it must do that the ad hoc version did not:

  - enumerate seeds from erl.csv itself, never from the checkpoints on disk
    (README lesson six, and its cross-machine form in summarize_gated.
    paired_seeds);
  - report under either training protocol, because E13b measured that the
    fixed 100-epoch schedule scores un-augmented arms 40-70 epochs deeper into
    overfitting than augmented ones, and every K-versus-baseline comparison
    here is exactly that shape.

The gate is the repo's: paired t over (image, seed) AND every seed agreeing in
sign. The t alone is not evidence -- 700 image pairs from two trainings is two
trainings, which is how E5 reported p = 3e-4 on a difference that flipped sign
between its only two seeds.

  python exp/summarize_erl.py --selftest
  python exp/summarize_erl.py                      # both protocols
  python exp/summarize_erl.py K_focal_aug A_dice   # one pair, both protocols
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_gated as gated

RESULTS = Path(__file__).resolve().parent / "results"
PROTOCOLS = (("erl.csv", "last epoch (the published protocol)"),
             ("erl_best.csv", "best validated epoch"))
# E18's headline pair, and the two arms it has to beat to earn its claim:
# A_dice is the baseline the 26.4% -> 44.7% sentence is about, H_aug is the
# arm that isolates what the gate adds on top of augmentation alone.
PAIRS = (("K_focal_aug", "A_dice", "E18 headline: the combination vs baseline"),
         ("K_focal_aug", "H_aug", "what the gate adds on top of augmentation"),
         ("K_focal_aug", "G_focal", "what augmentation adds on top of the gate"),
         ("H_aug_w64_d5", "A_dice_w64_d5", "31M: augmentation vs baseline"),
         ("B_cldice_w64_d5", "A_dice_w64_d5", "31M: topology loss vs baseline"),
         ("H_aug_w64_d5", "B_cldice_w64_d5", "31M: augmentation vs topology"))
KEY = "erl"


def load(name: str) -> list[dict]:
    rows = list(csv.DictReader((RESULTS / name).open()))
    for row in rows:
        for key in row:
            if key.startswith(("erl", "share")) or key == "skel_px":
                row[key] = float(row[key])
    return rows


def series(rows, config: str, key: str) -> dict:
    """One value per (image, seed), keyed so two configs line up."""
    return {(r["image"], r["run"].rsplit("_s", 1)[1]): r[key]
            for r in rows if r["config"] == config}


def compare(rows, better: str, worse: str, seeds, key: str = KEY) -> dict:
    left, right = series(rows, better, key), series(rows, worse, key)
    shared = [k for k in left if k in right and k[1] in seeds]
    diffs = np.array([left[k] - right[k] for k in shared])
    per_seed = []
    for seed in seeds:
        inside = [left[k] - right[k] for k in shared if k[1] == seed]
        per_seed.append(float(np.mean(inside)) if inside else float("nan"))
    t = stats.ttest_rel([left[k] for k in shared],
                        [right[k] for k in shared])
    return {"mean": float(diffs.mean()), "t": float(t.statistic),
            "n": len(shared), "per_seed": per_seed,
            "holds": bool(diffs.mean() > 0 and t.statistic > 2
                          and all(d > 0 for d in per_seed)
                          and len(per_seed) >= 3)}


def ceiling_share(rows, config: str, seeds) -> float:
    """Mean traced fraction of the achievable run length, whole skeleton.

    E18 quotes this, not raw ERL, because a length in pixels means nothing
    without the length a perfect prediction would trace on the same image.
    """
    picked = [r for r in rows if r["config"] == config
              and r["run"].rsplit("_s", 1)[1] in seeds]
    if not picked:
        return float("nan")
    # The whole-skeleton ceiling is the skeleton itself: a perfect prediction
    # traces every pixel, so ERL would equal total length. share = erl/skel_px.
    return float(np.mean([r["erl"] / r["skel_px"] for r in picked]))


def report(rows, better: str, worse: str, label: str) -> None:
    seeds = gated.paired_seeds(rows, better, worse)
    print(f"  {label}")
    print(f"    {better} vs {worse}")
    if len(seeds) < 2:
        print(f"    only {len(seeds)} paired seed(s); not reported\n")
        return
    result = compare(rows, better, worse, seeds)
    # A sign gate over fewer than three seeds is not a gate: with two, one
    # cannot disagree with the other often enough to mean anything.
    verdict = ("HOLDS" if result["holds"] else
               f"only {len(seeds)} seeds, NOT gated"
               if len(seeds) < 3 else "fails gate")
    print(f"    ERL diff {result['mean']:+9.1f} px  t={result['t']:6.2f}  "
          f"{len(seeds)} seeds  {verdict}")
    print(f"    per seed [{' '.join(f'{d:+.0f}' for d in result['per_seed'])}]"
          f"  (seeds {', '.join(seeds)})")
    for config in (better, worse):
        share = ceiling_share(rows, config, seeds)
        print(f"      {config:<22} traces {share:6.1%} of the tree")
    print()


def selftest() -> None:
    # The gate must need BOTH halves. A difference that is large and
    # significant but flips sign between seeds is E5's failure and must fail.
    def build(offsets):
        rows = []
        for seed, offset in enumerate(offsets):
            for image in range(20):
                # A little per-image spread, or the paired t is a division by
                # zero and scipy warns about catastrophic cancellation.
                jitter = (image % 5) - 2
                rows.append({"run": f"X_s{seed}", "config": "X",
                             "image": str(image),
                             "erl": 1000.0 + offset + jitter,
                             "skel_px": 4000.0})
                rows.append({"run": f"Y_s{seed}", "config": "Y",
                             "image": str(image), "erl": 1000.0 + jitter,
                             "skel_px": 4000.0})
        return rows

    fake = build((+100.0, -100.0, +100.0))
    seeds = gated.paired_seeds(fake, "X", "Y")
    assert seeds == ("0", "1", "2"), seeds
    result = compare(fake, "X", "Y", seeds)
    assert not result["holds"], "a sign flip between seeds must fail the gate"
    assert result["mean"] > 0 and result["t"] > 2, result
    print(f"a difference that is large (mean {result['mean']:+.0f}) and "
          f"significant (t={result['t']:.1f}) still fails the gate when one "
          f"seed disagrees in sign")

    # And one that is consistent must pass -- a gate that never passes is
    # not a gate either.
    result = compare(build((+100.0, +90.0, +110.0)), "X", "Y", seeds)
    assert result["holds"], result
    print("a consistent difference passes it")

    # Two seeds is not a gate: one seed cannot disagree with itself often
    # enough for the sign condition to mean anything, so it must not pass.
    two = build((+100.0, +100.0))
    assert gated.paired_seeds(two, "X", "Y") == ("0", "1")
    assert not compare(two, "X", "Y", ("0", "1"))["holds"], \
        "two seeds must never pass the sign gate"
    print("two consistent seeds still do not pass -- the gate needs three")

    # Seeds come from the data. A run present on disk but absent from the CSV
    # must not be counted, and the reverse must not crash.
    assert gated.paired_seeds(fake, "X", "Z") == ()
    print("seeds are enumerated from the CSV, not from the checkpoints")

    # share is erl / skeleton length, and must be a fraction.
    share = ceiling_share(fake, "X", seeds)
    assert 0.0 < share < 1.0, share
    print(f"traced share is a fraction of the achievable length ({share:.3f})")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    pairs = ((argv[0], argv[1], "requested"),) if len(argv) >= 2 else PAIRS

    for source, protocol in PROTOCOLS:
        if not (RESULTS / source).exists():
            print(f"### {source} not built yet "
                  f"(python exp/erl.py"
                  f"{' --checkpoint best.pt' if 'best' in source else ''})\n")
            continue
        rows = load(source)
        print(f"=== expected run length, {protocol} ({source}) ===")
        gated.report_scope(rows, source)
        print()
        for better, worse, label in pairs:
            if any(r["config"] == better for r in rows) and \
                    any(r["config"] == worse for r in rows):
                report(rows, better, worse, label)
            else:
                print(f"  {label}\n    {better} vs {worse}: not in {source}\n")

    print("The gate is a paired t over (image, seed) AND every seed agreeing")
    print("in sign. The t alone is not evidence: 700 image pairs from two")
    print("trainings is two trainings, which is how E5 reported p = 3e-4 on a")
    print("difference that flipped sign between its only two seeds.")


if __name__ == "__main__":
    main()
