"""Q1: does ERL rank a model ABOVE the second human? If so, the ruler is bent.

WRITTEN AND SELFTESTED 2026-09-08, BEFORE IT SCORED ANYTHING.

THE TEST. A dataset with two independent annotators gives a free sanity
check on any segmentation metric: take annotator A as ground truth, and score
annotator B's mask as if it were a prediction. B is a careful human doing the
same task the model does, so B's score is the level at which "as good as a
person" sits. A metric that puts a model ABOVE it is either measuring
something other than what a person would call a good tracing, or is being
gamed.

This is not a rhetorical check. Dice puts models above the second annotator
routinely and the field has learned to shrug at it. The question here is
whether ERL does the same, because a paper telling this field to adopt ERL
has to know.

WHICH DATASET. STARE, because its two readings (`ah` and `vk`) both cover all
20 images and the test split is the last 10 by sorted name -- 10 test images
with two annotators and no overlap with anything trained on. DRIVE's
2nd_manual is not on this machine. VessMAP's annotator2 covers 20 images of
which only THREE are in its test split and seventeen are in its train split,
so a VessMAP comparison would be a model scored on its own training data; it
is reported separately and marked, never mixed in.

THE CONFOUND, and the control for it. ERL is gameable by predicting MORE:
covering more ground-truth centreline with more foreground raises it, which
is why link_ceiling.py had to introduce a foreground-matched condition in
August. A model that beats `vk` while painting 1.4x the foreground has not
beaten a person, it has spent Dice. So every model is reported twice:

  own point      each arm at its own dev-chosen threshold
  matched fg     each arm thresholded so its foreground equals vk's, per
                 image, which is the only comparison in which "the model
                 traces further" means the model's ORDERING of pixels is
                 better rather than its budget being larger

BOTH DIRECTIONS. Which annotator is "the truth" is arbitrary, so the table is
computed with ah as truth and again with vk as truth. If the verdict flips,
that is itself the finding: the metric would be measuring which human is
being used as the target, not who traced better.

PRE-REGISTERED 2026-09-08, before this file has read a pixel:

  1. At its own operating point at least one arm beats the second annotator
     on ERL. Models are tuned on a Dice-like objective and ERL rewards
     foreground; if none of them manages it the metric is stricter than
     expected and the paper says so.
  2. At MATCHED foreground no arm beats the second annotator on ERL under
     convention A. This is the prediction that matters. If an arm beats a
     careful human at equal paint, then either ERL is not measuring
     traceability the way a person would judge it, or these models really are
     superhuman at connectivity -- and the second is not credible, so the
     paper would have to report the first.
  3. The verdict is the same with ah as truth and with vk as truth.
  4. Dice puts models ABOVE the second annotator even where ERL does not.
     That contrast is the argument for reporting ERL at all.

  python exp/human_baseline.py --selftest
  python exp/human_baseline.py --report

Writes results/human_baseline.txt via --report.
"""
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import erl
import erl_convention
import hole_sweep
import metrics
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import stare_agreement
import train
import transfer_calibration as calib

ARMS = calib.ARMS
SEEDS = 6          # of the 24 on disk; the spread between seeds is reported
READINGS = ("ah", "vk")


def stare_pairs() -> list[dict]:
    """The 10 STARE TEST images, each with both readings and its aperture.

    Aligned by name against cross_dataset's own split rather than re-derived,
    so this file cannot disagree with the split every other table used.
    """
    _, test_items = cross_dataset.loader_for("stare")()
    wanted = {item["name"]: item for item in test_items}
    out = []
    for row in stare_agreement.load_stare():
        if row["name"] in wanted:
            item = wanted[row["name"]]
            out.append({"name": row["name"], "image": item["image"],
                        "fov": item["fov"], "ah": row["ah"], "vk": row["vk"]})
    return out


def score(truth: np.ndarray, pred: np.ndarray, fov: np.ndarray) -> dict:
    """ERL under both conventions, plus Dice, clDice and the paint."""
    skel = skeletonize(truth & fov)
    pred = pred & fov
    total = float(skel.sum()) or 1.0
    got = metrics.evaluate(pred.astype(np.float32), truth & fov, fov)
    return {"erl_split": erl.expected_run_length(skel, pred) / total,
            "erl_bridged": erl_convention.bridged_run_length(skel, pred) / total,
            "dice": got["dice"], "cldice": got["cldice"],
            "fg": float(pred.sum())}


def match_foreground(prob: np.ndarray, fov: np.ndarray, target: int,
                     component_px: int) -> np.ndarray:
    """Threshold `prob` so its foreground is as close to `target` as the grid
    allows. Bisection on the threshold, not on the count: the count is
    monotone in the threshold, which is what makes this well defined."""
    low, high = 0.0, 1.0
    best = None
    for _ in range(24):
        middle = 0.5 * (low + high)
        mask = speckle.drop_small((prob >= middle) & fov, component_px)
        count = int(mask.sum())
        if best is None or abs(count - target) < abs(best[1] - target):
            best = (mask, count)
        if count > target:
            low = middle
        else:
            high = middle
    return best[0]


def selftest() -> None:
    pairs = stare_pairs()
    assert len(pairs) == 10, len(pairs)
    print(f"{len(pairs)} STARE test images carry both readings")

    # 1. THE TWO READINGS MUST ACTUALLY DIFFER. If they were identical the
    #    whole test would be vacuous and would silently report a tie.
    overlap = []
    for pair in pairs:
        both = float((pair["ah"] & pair["vk"]).sum())
        overlap.append(2 * both / float(pair["ah"].sum() + pair["vk"].sum()))
    assert 0.5 < np.mean(overlap) < 0.95, np.mean(overlap)
    print(f"the two readings agree at Dice {np.mean(overlap):.3f} "
          f"(range {min(overlap):.3f}-{max(overlap):.3f}) -- different enough "
          f"to be a real baseline, close enough to be the same task")

    # 2. THE SECOND READING MUST NOT SCORE PERFECTLY against the first, or the
    #    comparison has no headroom and every model would trivially lose.
    got = score(pairs[0]["ah"], pairs[0]["vk"], pairs[0]["fov"])
    assert 0.0 < got["erl_split"] < 1.0, got
    print(f"vk against ah on {pairs[0]['name']}: ERL(A) {got['erl_split']:.3f}, "
          f"Dice {got['dice']:.3f}")

    # 3. FOREGROUND MATCHING MUST MATCH. A control that misses its target by a
    #    lot is not a control, and the whole prediction 2 rests on it.
    rng = np.random.default_rng(0)
    prob = rng.random(pairs[0]["fov"].shape).astype(np.float32)
    for target in (5000, 20000, 60000):
        got_mask = match_foreground(prob, pairs[0]["fov"], target, 0)
        assert abs(int(got_mask.sum()) - target) / target < 0.02, \
            (target, int(got_mask.sum()))
    print("foreground matching lands within 2% of the target at 5k, 20k, 60k")

    # 4. ERL MUST BE GAMEABLE BY PAINT, which is the whole reason the matched
    #    condition exists. Assert it rather than assume it.
    thin = pairs[0]["ah"] & pairs[0]["fov"]
    from scipy import ndimage
    fat = ndimage.binary_dilation(thin, np.ones((5, 5))) & pairs[0]["fov"]
    lean, heavy = score(pairs[0]["ah"], thin, pairs[0]["fov"]), \
        score(pairs[0]["ah"], fat, pairs[0]["fov"])
    assert heavy["erl_split"] >= lean["erl_split"] - 1e-9
    assert heavy["fg"] > lean["fg"] * 1.5
    print(f"painting {heavy['fg'] / lean['fg']:.2f}x the foreground does not "
          f"lower ERL ({lean['erl_split']:.3f} -> {heavy['erl_split']:.3f}) "
          f"-- so the matched condition is not optional")
    print("all checks passed")


def report() -> None:
    pairs = stare_pairs()
    data = cross_dataset.stack(cross_dataset.fit_dev(
        cross_dataset.loader_for("stare")()[0])[0])
    root = calib.dataset_root("stare")
    epochs = heldout.chosen_epochs(root=root)
    width = cross_dataset.median_width(
        [{"label": p["ah"], "fov": p["fov"]} for p in pairs])
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))

    print("=== does ERL rank a model above the second human? ===\n")
    print(f"STARE, {len(pairs)} test images, both readings. `ah` and `vk` are")
    print("independent tracings of the same 20 fundus images. One is taken as")
    print("truth and the other is scored as if it were a prediction: that is")
    print("the level at which 'as good as a person' sits.\n")
    print("ERL is gameable by painting more, so every arm is shown twice: at")
    print("its own dev-chosen threshold, and thresholded to the SECOND")
    print("ANNOTATOR'S foreground per image. Only the matched row answers")
    print("whether the model's ordering of pixels is better than a human's.\n")

    for truth_name in READINGS:
        other = "vk" if truth_name == "ah" else "ah"
        print(f"--- truth = {truth_name}, human baseline = {other} ---")
        human = [score(p[truth_name], p[other], p["fov"]) for p in pairs]
        human_fg = [int(row["fg"]) for row in human]

        def show(tag, rows):
            print(f"    {tag:26}"
                  f"{np.mean([r['erl_split'] for r in rows]):>10.1%}"
                  f"{np.mean([r['erl_bridged'] for r in rows]):>11.1%}"
                  f"{np.mean([r['dice'] for r in rows]):>9.3f}"
                  f"{np.mean([r['cldice'] for r in rows]):>9.3f}"
                  f"{np.mean([r['fg'] for r in rows]) / np.mean(human_fg):>9.2f}x")

        print(f"    {'':26}{'ERL(A)':>10}{'ERL(B)':>11}{'Dice':>9}"
              f"{'clDice':>9}{'paint':>10}")
        show(f"HUMAN {other}", human)
        for arm in ARMS:
            runs = sorted(r for r in epochs
                          if r.rsplit("_s", 1)[0] == arm
                          and (root / r / "final.pt").exists())[:SEEDS]
            own, matched = [], []
            for run in runs:
                # load_model hardcodes heldout.ROOT, which is DRIVE's; the
                # transfer roots need the explicit form transfer_postproc
                # uses, or this would silently score a DRIVE checkpoint.
                weights = root / run / f"epoch{epochs[run]:03d}.pt"
                if not weights.exists():
                    continue
                model = train.build_model(arm)
                model.load_state_dict(
                    train.load_checkpoint(weights)["model"])
                model.eval()
                mean, std = train.normalisation(run, data)
                for index, pair in enumerate(pairs):
                    prob = train.predict_full(model, pair["image"], mean, std)
                    at_half = speckle.drop_small((prob >= 0.5) & pair["fov"],
                                                 component_px)
                    own.append(score(pair[truth_name], at_half, pair["fov"]))
                    matched.append(score(
                        pair[truth_name],
                        match_foreground(prob, pair["fov"], human_fg[index],
                                         component_px),
                        pair["fov"]))
            if own:
                show(f"{arm} @0.5", own)
                show(f"{arm} @matched fg", matched)
        print()


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    raise SystemExit("pass --selftest or --report")


if __name__ == "__main__":
    main()
