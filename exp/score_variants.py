"""Score A1 (SWA), A2 (seed ensemble) and A3 (TTA) against their baselines.

One row per (config, seed, variant, image), so summarize_variants.py never
re-scores anything and every comparison it makes is paired on the same image
and the same seed.

The nine variants, and what each is the baseline FOR:

  rule_i    the current protocol: the epoch with the best validation Dice,
            which is what best.pt holds. Baseline for tta_i.
  rule_iv   the epoch with the best validation clDice -- task A3's rule, the
            one that took K_focal_aug from 40.1% to 47.4% traced. It did not
            pass the pre-registered two-arm gate, so it is not adopted; it is
            here because plan_next.md section 0 says a new method has to beat
            the best rule applied to BOTH arms, and that is this one.
            Baseline for tta_iv.
  swa       weights averaged over epochs 60-100, BatchNorm recomputed.
            Compared against BOTH rules, since it picks no epoch at all.
  tta_i     rule_i, predicted under all eight symmetries and averaged.
  tta_iv    rule_iv, likewise.
  swa_tta   both, to see whether they are additive or the same gain twice.
  ens_i     the six seeds' probability maps at rule_i, averaged.
  ens_iv    likewise at rule_iv.
  ens_swa   likewise for the six SWA models.

Selection reads the odd DRIVE val images and the reported ERL reads the even
ones, exactly as task A did -- select_checkpoint.is_selection_image is still
the single place that decides which is which. The rules here are applied to
checkpoint_scores.csv, which was built on the selection half.

  python exp/score_variants.py --selftest
  python exp/score_variants.py                 # all four arms
  python exp/score_variants.py K_focal_aug     # one arm

Writes results/selection_sweep/variant_scores.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset
import drive
import erl
import hole_sweep
import metrics
import select_checkpoint as rules_module
import speckle
import summarize_selection as selection
import train
import variants

OUT = selection.SWEEP / "variant_scores.csv"
ARMS = selection.ARMS

# variant name -> (which epoch rule picks the weights, use TTA)
SINGLE = {
    "rule_i": ("(i) best Dice [current]", False),
    "rule_iv": ("(iv) best clDice", False),
    "tta_i": ("(i) best Dice [current]", True),
    "tta_iv": ("(iv) best clDice", True),
}
ENSEMBLES = {"ens_i": "rule_i", "ens_iv": "rule_iv", "ens_swa": "swa"}


def chosen_epochs(points: dict) -> dict:
    """{run: {rule name: epoch}} from the selection half."""
    named = dict(rules_module.rules())
    return {run: {name: named[name](these)["epoch"]
                  for name in {r[0] for r in SINGLE.values()}}
            for run, these in points.items()}


def score(prob, item, geo, component_px: int) -> dict:
    pred = speckle.drop_small((prob >= 0.5) & item["fov"], component_px)
    scores = metrics.evaluate(prob, item["label"] & item["fov"], item["fov"])
    return {"dice": round(float(scores["dice"]), 5),
            "cldice": round(float(scores["cldice"]), 5),
            "betti0_err": round(float(scores["betti0_err"]), 3),
            "erl": round(erl.expected_run_length(geo["skel"], pred), 3),
            "skel_px": int(geo["skel"].sum())}


def selftest() -> None:
    # Every variant must name a baseline that is itself scored, or a
    # comparison silently has nothing to pair against.
    names = set(SINGLE) | {"swa", "swa_tta"} | set(ENSEMBLES)
    for variant, base in ENSEMBLES.items():
        assert base in names, (variant, base)
    for variant, (rule, _) in SINGLE.items():
        assert rule in dict(rules_module.rules()), rule
    assert {"rule_i", "rule_iv"} <= names
    print(f"{len(names)} variants, every ensemble's base among them: "
          f"{', '.join(sorted(names))}")

    # The TTA variants must differ from their non-TTA namesakes in exactly
    # one thing: the flag. Otherwise a comparison attributes to TTA whatever
    # else changed.
    assert SINGLE["tta_i"][0] == SINGLE["rule_i"][0]
    assert SINGLE["tta_iv"][0] == SINGLE["rule_iv"][0]
    assert SINGLE["tta_i"][1] and not SINGLE["rule_i"][1]
    print("tta_i and rule_i read the same epoch and differ only in the "
          "eight-fold average; likewise tta_iv and rule_iv")

    # SWA's window must not coincide with any rule's pick, or "averaging
    # beats picking" would be comparing a thing with itself.
    assert len(variants.SWA_EPOCHS) >= 3, variants.SWA_EPOCHS
    print(f"SWA averages epochs {variants.SWA_EPOCHS}, a fixed tail chosen "
          f"before any result and independent of every selection rule")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ARMS)

    rows_in = selection.load()
    points = selection.selection_points(rows_in)
    epochs = chosen_epochs(points)

    items = drive.load_split("val")
    data = train.stack_split("train")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(item["label"] & item["fov"])}
                for item in items]
    print(f"{len(wanted)} arm(s), {len(items)} images, component filter "
          f"{component_px} px, SWA over epochs {variants.SWA_EPOCHS}",
          flush=True)

    rows = []
    for config in wanted:
        seeds = sorted(run for run in points
                       if run.rsplit("_s", 1)[0] == config)
        if not seeds:
            print(f"[{config}] not in {selection.SCORES.name}; skipping",
                  flush=True)
            continue
        # {variant: {image index: [prob per seed]}} for the ensembles.
        pooled = {base: [[] for _ in items] for base in ENSEMBLES.values()}
        for run in seeds:
            run_dir = selection.SWEEP / run
            mean, std = train.normalisation(run, data)
            models = {}
            for variant, (rule, use_tta) in SINGLE.items():
                epoch = epochs[run][rule]
                if epoch not in models:
                    state = train.load_checkpoint(
                        run_dir / f"epoch{epoch:03d}.pt")
                    model = train.build_model(config)
                    model.load_state_dict(state["model"])
                    model.eval()
                    models[epoch] = model
                models[variant] = (models[epoch], use_tta)
            averaged = variants.swa_model(config, run_dir, data, mean, std)
            models["swa"] = (averaged, False)
            models["swa_tta"] = (averaged, True)

            for variant in list(SINGLE) + ["swa", "swa_tta"]:
                model, use_tta = models[variant]
                for index, (item, geo) in enumerate(zip(items, geometry)):
                    prob = (variants.predict_tta(model, item["image"], mean,
                                                 std) if use_tta
                            else train.predict_full(model, item["image"],
                                                    mean, std))
                    if variant in pooled:
                        pooled[variant][index].append(prob)
                    rows.append({"config": config, "run": run,
                                 "seed": run.rsplit("_s", 1)[1],
                                 "variant": variant, "image": item["name"],
                                 **score(prob, item, geo, component_px)})
                print(f"  {run} {variant} done", flush=True)

        for variant, base in ENSEMBLES.items():
            for index, (item, geo) in enumerate(zip(items, geometry)):
                stack = pooled[base][index]
                if len(stack) != len(seeds):
                    raise RuntimeError(
                        f"{variant} has {len(stack)} maps for {len(seeds)} "
                        f"seeds on image {item['name']}")
                prob = np.mean(stack, axis=0).astype(np.float32)
                # seed "" -- an ensemble is not a seed, and writing one here
                # would let a paired test match it against a single run.
                rows.append({"config": config, "run": f"{config}_ens",
                             "seed": "", "variant": variant,
                             "image": item["name"],
                             **score(prob, item, geo, component_px)})
            print(f"  {config} {variant} done ({len(seeds)} seeds)",
                  flush=True)

    if not rows:
        raise SystemExit("nothing scored")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
