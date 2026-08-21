"""E8: does seed disagreement fix the inversion E1' found in per-pixel confidence?

E1' measured that the model's own hesitation predicts human disagreement at
AUROC 0.881 in the clearest contrast band and 0.373 in the dimmest -- below
0.5, i.e. actively misleading -- and the dimmest band is where 45.6% of the
annotator disagreement lives. The mechanism was p = 0.203 on contested pixels
against 0.574 on agreed ones: the model is confidently wrong about the dim
vessels it dropped.

The LLM hallucination literature hit the same wall and stepped around it the
same way twice: post-softmax probability is unreliable, so measure the spread
across independent samples instead (semantic entropy, Nature 2024). The
segmentation analogue of "sample the model again" is "train it again with a
different seed", and we already have two seeds for every STARE configuration,
so this costs inference and no training at all.

Deep ensembles as an uncertainty estimate are old news in medical imaging;
the question E8 asks is not whether they work but whether they repair this
specific failure, in the specific band where confidence inverts. Nobody has
reported that because the failure only appears once results are stratified by
local contrast, which the failure-detection benchmarks do not do
(arXiv:2406.03323 aggregates pixels to whole images, on blob-shaped organs).

Three signals, same pixels, same evaluation as E1', so the numbers are
directly comparable to results/stare_hesitation.csv:

    single     hesitation of one model            (E1' reproduced)
    ens_mean   hesitation of the averaged map     (does averaging alone do it?)
    disagree   |p_seed0 - p_seed1|                (the new signal)

ens_mean is the control that matters. If ensembling helps only through a
smoother mean probability, then this is calibration and not disagreement, and
the semantic-entropy analogy does not hold.

  python exp/ensemble_uncertainty.py             # all six (target, fold) pairs
  python exp/ensemble_uncertainty.py soft_f0     # one pair
  python exp/ensemble_uncertainty.py --selftest  # check the signals, no data

Writes results/ensemble_uncertainty.csv. ~10 min on 6 cores.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breaks
import stare_agreement
import stare_stratify
import stratify
import train
import train_stare

RESULTS = Path(__file__).resolve().parent / "results"
SIGNALS = ("single", "ens_mean", "disagree")
ANNOTATOR_MEMBERS = ("ah", "vk")


def signals(probs: list[np.ndarray]) -> dict:
    """The three uncertainty maps, from the per-seed probability maps.

    Hesitation is 1 - 2|p - 0.5|, peaking at p = 0.5 and falling to 0 at either
    certainty -- the same definition E1' used, kept identical on purpose.
    Disagreement is the spread between seeds, which is 0 when they agree
    regardless of WHAT they agree on. That difference is the whole experiment:
    two models can both be confidently wrong and still disagree with each
    other, and only the second signal can see it.
    """
    mean = np.mean(probs, axis=0)
    return {
        "single": 1.0 - 2.0 * np.abs(probs[0] - 0.5),
        "ens_mean": 1.0 - 2.0 * np.abs(mean - 0.5),
        # Range rather than std: with two members std is just range/2, and the
        # range keeps meaning if a third seed is ever added.
        "disagree": np.max(probs, axis=0) - np.min(probs, axis=0),
    }


def main() -> None:
    items = stare_agreement.load_stare()
    pairs = sys.argv[1:] or [f"{target}_f{fold}"
                             for fold in range(len(train_stare.FOLDS))
                             for target in train_stare.TRAIN_TARGETS]

    geometry = []
    for item in items:
        union = (item["ah"] | item["vk"]) & item["fov"]
        geometry.append({"union": union,
                         "contrast": breaks.local_contrast(item["image"]),
                         "disagree": (item["ah"] ^ item["vk"]) & item["fov"]})
    # Same binning as E1', from the union of both annotators over all images,
    # so an AUROC here sits in the same coordinate system as stare_hesitation.
    edges = np.percentile(
        np.concatenate([g["contrast"][g["union"]] for g in geometry]),
        [25, 50, 75])
    for geo in geometry:
        geo["band"] = stratify.band_map(geo["union"], geo["contrast"], edges)

    # Two kinds of ensemble member, and the contrast between them is the point.
    # "seed" members differ only in initialisation, so they share whatever the
    # data and the loss taught them. "annotator" members were trained on ah and
    # on vk, so they disagree about the thing the humans disagree about.
    #
    # The annotator arm is NOT a general uncertainty method and must not be read
    # as one: it needs two annotators at training time, which is exactly the
    # scarce thing, and it is partly circular because a model trained on ah
    # leans toward ah. It is here to answer a different question -- is the
    # information about contested pixels learnable at all, or is it absent from
    # anything a network trained this way can produce?
    jobs = [("seed", pair, [f"{pair}_s{seed}" for seed in train_stare.SEEDS])
            for pair in pairs]
    jobs += [("annotator", f"ahvk_f{fold}_s{seed}",
              [f"{a}_f{fold}_s{seed}" for a in ANNOTATOR_MEMBERS])
             for fold in range(len(train_stare.FOLDS))
             for seed in train_stare.SEEDS]

    rows = []
    for kind, pair, run_names in jobs:
        target, fold_tag = pair.split("_f")
        _, test_slice = train_stare.FOLDS[int(fold_tag[0])]
        members = [stare_stratify.load(name) for name in run_names]

        for index in range(len(items))[test_slice]:
            item, geo = items[index], geometry[index]
            probs = [train.predict_full(model, item["image"], mean, std)
                     for model, mean, std in members]
            maps = signals(probs)

            for code, band in enumerate(stratify.BANDS):
                population = geo["union"] & (geo["band"] == code)
                if not population.any():
                    continue
                contested = geo["disagree"][population]
                row = {"kind": kind, "pair": pair, "trained_on": target,
                       "image": item["name"], "band": band,
                       "n_px": int(population.sum()),
                       "contested_frac": round(float(contested.mean()), 5)}
                for name in SIGNALS:
                    values = maps[name][population]
                    row[f"auroc_{name}"] = round(
                        stare_stratify.auroc(values, contested), 5)
                    # The mechanism behind whichever AUROC wins or loses: a
                    # signal can only rank contested above agreed if it is
                    # actually larger there.
                    row[f"mean_{name}_contested"] = round(
                        float(values[contested].mean()), 5)
                    row[f"mean_{name}_agreed"] = round(
                        float(values[~contested].mean()), 5)
                rows.append(row)
        print(f"{kind}/{pair} done", flush=True)

    out = RESULTS / "ensemble_uncertainty.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    # Split by member kind. Pooling them hides the whole result: seed members
    # and annotator members behave completely differently outside Q1.
    for kind in ("seed", "annotator"):
        print(f"\n=== AUROC by band, {kind} members ===")
        print(f"{'band':>14}" + "".join(f"{name:>12}" for name in SIGNALS))
        for band in stratify.BANDS:
            picked = [r for r in rows
                      if r["band"] == band and r["kind"] == kind]
            cells = "".join(
                f"{np.nanmean([r[f'auroc_{name}'] for r in picked]):12.3f}"
                for name in SIGNALS)
            print(f"{band:>14}{cells}")


def selftest() -> None:
    """Build the case the experiment is looking for and check each signal
    reports what it should.

    Two models that are both confidently wrong on the contested pixels but
    wrong in OPPOSITE directions: hesitation sees nothing (both are far from
    0.5), disagreement sees everything. If `signals` ever loses that
    distinction the experiment silently becomes a calibration study.
    """
    rng = np.random.default_rng(0)
    contested = np.zeros(1000, dtype=bool)
    contested[:200] = True

    prob0 = np.where(contested, 0.05, 0.5 + rng.uniform(-0.02, 0.02, 1000))
    prob1 = np.where(contested, 0.95, 0.5 + rng.uniform(-0.02, 0.02, 1000))
    maps = signals([prob0, prob1])

    single = stare_stratify.auroc(maps["single"], contested)
    ensemble = stare_stratify.auroc(maps["ens_mean"], contested)
    disagree = stare_stratify.auroc(maps["disagree"], contested)
    print(f"confidently-wrong-in-opposite-directions case: "
          f"single {single:.3f}, ens_mean {ensemble:.3f}, "
          f"disagree {disagree:.3f}")
    # Both members sit at 0.05 / 0.95 on contested pixels and near 0.5
    # elsewhere, so hesitation ranks contested pixels BELOW agreed ones: this
    # is E1's inversion, reproduced by construction.
    assert single < 0.1, single
    assert disagree > 0.99, disagree
    # Averaging 0.05 and 0.95 lands on 0.5, which looks maximally hesitant, so
    # the mean map gets this case right too. That is exactly why ens_mean has
    # to be in the experiment: without it, a win for disagreement could just be
    # a win for averaging.
    assert ensemble > 0.99, ensemble

    # The other half: two models that agree, and are both wrong. Disagreement
    # must go blind here -- it is not a correctness detector, and a reader who
    # assumes it is will over-read the result.
    both_wrong = np.where(contested, 0.05, 0.5)
    maps = signals([both_wrong, both_wrong + rng.normal(0, 1e-6, 1000)])
    blind = stare_stratify.auroc(maps["disagree"], contested)
    assert 0.4 < blind < 0.6, blind
    print(f"agreed-and-both-wrong case: disagree {blind:.3f} (blind, as it "
          f"must be)")
    print("all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
