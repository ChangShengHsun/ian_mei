"""The post-processing layer, under the clean protocol, with a SHARED field.

WRITTEN AND SELFTESTED 2026-08-31, BEFORE THE FIRST ROW WAS SCORED.

WHAT IS DIFFERENT FROM exp/direction_ceiling.py, and why this is a new file
rather than an edit. That script produced the published legacy table; CLAUDE.md
forbids rewriting a published result in place, and its numbers stay
reproducible from what is on disk. Three changes:

  1. THE CLEAN PROTOCOL. It reads exp/results/heldout, picks each run's epoch
     from the DEV curve, and scores all 20 test images. The legacy table
     picked on odd test images and reported on the even ten.

  2. ONE FIELD FOR EVERY ARM. The legacy script read `predicted` from the
     arm's OWN direction head, so only the two _dir arms had the column at
     all. Here a separate direction model supplies the field to every arm's
     output. That is the whole claim: a tangent field is a property of the
     IMAGE, not of the segmenter, so one predictor should correct anything.
     Reading it from each arm's own head would make this an auxiliary-task
     paper again, which is the framing being retired.

  3. THE ARMS THAT SURVIVED CALIBRATION. K_focal_aug and G_focal led the
     legacy table and calibration.md then showed both are calibration
     artifacts (+13.6% at threshold 0.5, -4.2% at their own operating point).
     They are kept here only as the negative control. The frontier arms are
     the clw family, and not one of them was in the old table.

THE FIELD IS PAIRED BY SEED. Arm seed 3 is corrected by the field model's seed
3. Pairing them removes a systematic artifact -- one lucky field model
flattering every arm -- at no cost, since both have six seeds.

THE CONTROL THAT MAKES ANY NUMBER MEAN SOMETHING is `shuffled`: the same
oriented dilation on a random per-pixel axis field. In the legacy table it
scored EXACTLY `raw` in every cell of both conventions, because a random field
picks the free do-nothing geometry under any Dice budget. If that ever stops
being true here, the sweep is measuring dilation and not direction.

  python exp/postproc_ceiling.py --selftest
  python exp/postproc_ceiling.py [config ...] [--field A_dice_dir]
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anisotropic
import cross_dataset
import direction
import drive
import erl
import erl_convention
import hole_sweep
import link_ceiling
import score_direction
import select_heldout as heldout
import speckle
import train

OUT = heldout.ROOT / "postproc_ceiling.csv"


def out_path(shard, split: str = "test") -> Path:
    """One file per shard and split. Merged by summarize_postproc.py."""
    stem = "postproc_ceiling" if split == "test" else "postproc_dev"
    if shard is None:
        return OUT.with_name(f"{stem}.csv")
    return OUT.with_name(f"{stem}.shard{shard[0]}of{shard[1]}.csv")
# The same grid the legacy sweep used, so the two tables are comparable in
# geometry even though their protocols differ. Both are in multiples of median
# vessel width, which is what lets task 3 apply the same numbers to HRF at six
# times DRIVE's resolution.
ALONG = (0.0, 0.5, 1.0, 1.5, 2.0)
ACROSS = (0.0, 0.25, 0.5, 1.0)
ISOTROPIC = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
DEFAULT_FIELD = "H_aug_dir"

# The arms, grouped by what calibration.md said about them.
CONTROL = ("A_dice", "H_aug")
FRONTIER = ("H_aug_clw2", "H_aug_clw8", "H_aug_clw16", "H_aug_clw64",
            "K_focal_aug_clw32", "K_focal_aug_clw64", "A_dice_clw64",
            "H_aug_w64_d5")
RETIRED = ("K_focal_aug", "G_focal")


def measure(mask, skel, truth) -> dict:
    return {"erl_split": round(
                erl.expected_run_length(skel, mask) / skel.sum(), 5),
            "erl_bridged": round(
                erl_convention.bridged_run_length(skel, mask) / skel.sum(), 5),
            "dice": round(link_ceiling.dice(mask, truth), 5),
            "fg": int(mask.sum())}


def load_model(run: str, config: str, epochs: dict, data: dict):
    weights = heldout.ROOT / run / f"epoch{epochs[run]:03d}.pt"
    if not weights.exists():
        return None, None, None
    model = train.build_model(config)
    model.load_state_dict(train.load_checkpoint(weights)["model"])
    model.eval()
    mean, std = train.normalisation(run, data)
    return model, mean, std


def shared_fields(field_arm: str, epochs: dict, items: list, data: dict,
                  seeds: list) -> dict:
    """{(seed, image): (sin2, cos2)} from ONE direction model per seed.

    Computed once and reused across every arm being corrected. That is both
    the saving -- twelve arms would otherwise each pay for the same forward
    pass -- and the claim: the field does not know which segmenter it is
    about to correct.
    """
    out = {}
    for seed in seeds:
        run = f"{field_arm}_s{seed}"
        model, mean, std = load_model(run, field_arm, epochs, data)
        if model is None:
            print(f"  no checkpoint for {run}; that seed has no field",
                  flush=True)
            continue
        for item in items:
            out[(seed, item["name"])] = score_direction.predict_field(
                model, item["image"], mean, std)
        print(f"  field from {run}", flush=True)
    return out


def selftest() -> None:
    # 1. The arm lists must name real configs, or the sweep silently covers
    #    fewer arms than the table claims.
    for config in CONTROL + FRONTIER + RETIRED:
        assert config in train.CONFIGS, config
    assert DEFAULT_FIELD in train.CONFIGS
    assert train.uses_direction(DEFAULT_FIELD), DEFAULT_FIELD
    # And the field model must be a PLAIN head: a _dir arm carrying a
    # propagation or snake layer predicts a field that its own layer has
    # already consumed, which is a different quantity.
    assert not train.uses_propagation(DEFAULT_FIELD)
    assert not train.uses_snake(DEFAULT_FIELD)
    print(f"  {len(CONTROL + FRONTIER + RETIRED)} arms and the field model "
          f"{DEFAULT_FIELD} are all real configs")

    # 2. THE POINT OF THE FILE. A field from a DIFFERENT run must reach the
    #    dilation, and must give a different answer from the arm's own field.
    #    If the plumbing quietly fell back to the arm's own head, every number
    #    would look fine and the paper's claim would be false.
    rng = np.random.default_rng(0)
    mask = np.zeros((60, 60), dtype=bool)
    mask[30, 10:50] = True
    mask = np.asarray(mask)
    truth = np.zeros_like(mask)
    truth[29:32, 10:50] = True
    horizontal = direction.tangent_field(truth)
    vertical = direction.tangent_field(np.ascontiguousarray(truth.T))
    a = anisotropic.oriented_dilation(mask, horizontal[0], horizontal[1],
                                      4.0, 1.0)
    b = anisotropic.oriented_dilation(mask, vertical[0], vertical[1],
                                      4.0, 1.0)
    assert a.sum() != b.sum(), (a.sum(), b.sum())
    print(f"  two different fields give two different dilations "
          f"({a.sum()} vs {b.sum()} px) -- the field is load-bearing")

    # 3. WHAT THE MECHANISM ACTUALLY IS, asserted rather than assumed. The
    #    naive story -- "the true axis covers more centreline at the same
    #    reach" -- is FALSE, and this test was written the wrong way round
    #    first. Probed on the measured error mode (the prediction running one
    #    pixel beside the centreline), at along=0.5 across=0.25 the true axis
    #    covers 2 centreline px and a RANDOM field covers 32: a thin ellipse
    #    along a horizontal vessel barely reaches the row above it, while
    #    random orientations sometimes point straight at it.
    #
    #    The true axis wins on COST, not on reach. At a reach where both cover
    #    the same centreline, the oriented dilation spends far less foreground
    #    doing it, because it adds pixels ON the vessel instead of beside it.
    #    That is the same matched-cost logic every verdict in this repo now
    #    uses, and it is why the sweep compares under a Dice budget.
    beside = np.zeros((60, 60), dtype=bool)
    beside[31, 5:55] = True                    # prediction, one px off
    line = np.zeros((60, 60), dtype=bool)
    line[30, 5:55] = True                      # the centreline it misses
    body = np.zeros((60, 60), dtype=bool)
    body[30:33, 5:55] = True
    axis = direction.tangent_field(body)
    random_field = anisotropic.shuffled_field(beside.shape, seed=1)
    true_grown = anisotropic.oriented_dilation(beside, axis[0], axis[1],
                                               6.0, 3.0)
    rand_grown = anisotropic.oriented_dilation(beside, random_field[0],
                                               random_field[1], 6.0, 3.0)
    assert (line & true_grown).sum() == (line & rand_grown).sum(), "not matched"
    true_cost = float(true_grown.sum())
    rand_cost = float(rand_grown.sum())
    assert true_cost < rand_cost, (true_cost, rand_cost)
    print(f"  at equal centreline coverage the true axis spends "
          f"{true_cost:.0f} foreground px against {rand_cost:.0f} for a "
          f"random one -- the mechanism is cost, not reach")

    # 4. The geometry grid must contain the free do-nothing setting, or a
    #    tight Dice budget has no admissible answer and the column comes back
    #    empty -- the bug that made gate_d1 skip four GPU hours.
    assert 0.0 in ALONG and 0.0 in ACROSS and 0.0 in ISOTROPIC
    print("  the grid contains the free do-nothing setting on every axis")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    argv = sys.argv[1:]
    field_arm = DEFAULT_FIELD
    if "--field" in argv:
        index = argv.index("--field")
        field_arm = argv[index + 1]
        del argv[index:index + 2]
    shard = None
    if "--shard" in argv:
        index = argv.index("--shard")
        part, total = argv[index + 1].split("/")
        shard = (int(part), int(total))
        del argv[index:index + 2]
    wanted = [a for a in argv if not a.startswith("--")] or \
        list(CONTROL + FRONTIER + RETIRED)

    epochs = heldout.chosen_epochs()
    # The geometry has to be CHOSEN somewhere other than where it is
    # reported. This project has now made that mistake three times at three
    # levels -- the checkpoint on the test set, the threshold on the test
    # curve, and (caught 2026-09-01, after the first sweep had already run)
    # the dilation geometry on the reported images. Scoring dev as well is
    # what lets summarize_postproc.py pick on dev and read on test.
    split = "dev" if "--dev" in sys.argv else "test"
    items = drive.load_split(split)
    data = train.stack_split("fit")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(item["label"] & item["fov"]),
                 "truth": item["label"] & item["fov"]} for item in items]
    oracle = {item["name"]: direction.tangent_field(item["label"] & item["fov"])
              for item in items}

    target = out_path(shard, split)
    stem = "postproc_ceiling" if split == "test" else "postproc_dev"
    done = set()
    for existing in sorted(OUT.parent.glob(f"{stem}*.csv")):
        done |= {(r["config"], r["seed"], r["field_arm"])
                 for r in csv.DictReader(existing.open())}
    seeds = sorted({run.rsplit("_s", 1)[1] for run in epochs
                    if run.rsplit("_s", 1)[0] == field_arm})
    print(f"{split}: {len(wanted)} arm(s), field from {field_arm} "
          f"({len(seeds)} seeds), width {width:.2f} px, filter "
          f"{component_px} px", flush=True)
    fields = shared_fields(field_arm, epochs, items, data, seeds)

    fresh = not target.exists()
    with target.open("a", newline="") as handle:
        writer = None
        for config in wanted:
            runs = sorted(r for r in epochs
                          if r.rsplit("_s", 1)[0] == config)
            if not runs:
                print(f"[{config}] no runs under {heldout.ROOT}; skipping",
                      flush=True)
                continue
            for run in runs:
                seed = run.rsplit("_s", 1)[1]
                if (config, seed, field_arm) in done:
                    continue
                # Sharded on (config, seed) so each process writes disjoint
                # rows into its OWN file. Two processes appending to one CSV
                # interleave partial lines under load, and a half-written row
                # is worse than a missing one -- it parses.
                if shard is not None:
                    key = abs(hash((config, seed))) % shard[1]
                    if key != shard[0]:
                        continue
                model, mean, std = load_model(run, config, epochs, data)
                if model is None:
                    continue
                rows = []
                for item, geo in zip(items, geometry):
                    prob = train.predict_full(model, item["image"], mean, std)
                    pred = speckle.drop_small(
                        (prob >= 0.5) & item["fov"], component_px)
                    common = {"config": config, "run": run,
                              "epoch": epochs[run], "seed": seed,
                              "field_arm": field_arm, "image": item["name"]}
                    rows.append({**common, "source": "raw", "along": 0.0,
                                 "across": 0.0,
                                 **measure(pred, geo["skel"], geo["truth"])})
                    for radius in ISOTROPIC:
                        if radius == 0.0:
                            continue
                        grown = anisotropic.isotropic_dilation(
                            pred, radius * width) & item["fov"]
                        rows.append({**common, "source": "isotropic",
                                     "along": radius, "across": radius,
                                     **measure(grown, geo["skel"],
                                               geo["truth"])})
                    sources = {
                        "oracle": oracle[item["name"]][:2],
                        "shuffled": anisotropic.shuffled_field(
                            item["label"].shape,
                            seed=abs(hash((run, item["name"]))) % 2**31)}
                    got = fields.get((seed, item["name"]))
                    if got is not None:
                        sources["predicted"] = got
                    for source, (sin2, cos2) in sources.items():
                        for along in ALONG:
                            for across in ACROSS:
                                if along == 0.0 and across == 0.0:
                                    continue
                                grown = anisotropic.oriented_dilation(
                                    pred, sin2, cos2, along * width,
                                    across * width) & item["fov"]
                                rows.append({
                                    **common, "source": source,
                                    "along": along, "across": across,
                                    **measure(grown, geo["skel"],
                                              geo["truth"])})
                if writer is None:
                    writer = csv.DictWriter(handle,
                                            fieldnames=list(rows[0]))
                    if fresh:
                        writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                print(f"  {run} done ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
