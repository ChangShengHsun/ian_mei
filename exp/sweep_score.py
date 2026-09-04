"""Task A2: score every kept checkpoint of the selection sweep.

One row per (checkpoint, image), carrying both the metrics a selection rule
could read and the metric the report is judged on, so that A3 never has to
re-score anything and can never accidentally select on the reported half.

  python exp/sweep_score.py --selftest
  python exp/sweep_score.py                    # every run in the sweep dir
  python exp/sweep_score.py A_dice_s0          # one run

Writes results/selection_sweep/checkpoint_scores.csv.

TWO PROTOCOLS, TWO NORMALISATIONS. `--results` points this at the held-out
retrain's root, whose runs were fit on 15 images with 5 held back for
selection; the sweep's own runs were trained on all 20. The inference
normalisation must come from the images the run was FIT on, so the stack is
chosen from the root rather than hardwired.

CAUGHT 2026-09-04. This was hardwired to stack_split("train") and never
updated when the held-out protocol landed on 09-01 -- so every held-out row
was normalised with constants computed partly from the 5 selection images.
frontier.py, composition.py, threshold_control.py and postproc_ceiling.py all
use "fit"; this file was the only one left. Measured cost on three runs at
threshold 0.5: A_dice +0.40, H_aug_clw64 +0.06, K_focal_aug_clw64 +0.14 ERL
points, always in favour of the leaked stack, against a +1.4-point bar for
the largest honest effect in this repo. The GAPS the leak ledger reports are
nearly unaffected (both sides shared the constant); the absolute columns were
optimistic by up to 0.4.

The geometry -- skeleton, contrast bands, component filter -- is built exactly
as erl.py builds it, by calling into erl.py, so a number here is comparable
with erl.csv rather than merely similar to it. The component filter is E4's
threshold in dataset-relative units (multiples of median vessel width
squared), not a pixel count, for the reason CLAUDE.md gives: 20 px means
something different on a retina six times the resolution.
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
import select_heldout as heldout
import speckle
import train

SWEEP = Path(__file__).resolve().parent / "results" / "selection_sweep"
OUT = SWEEP / "checkpoint_scores.csv"


def stack_for(root: Path) -> str:
    """Which images a run under `root` was fit on.

    The held-out root's runs saw 15; the sweep's saw all 20. Normalising with
    images a run never trained on is a protocol impurity, and in this repo's
    case it is one that flatters the score.
    """
    return "fit" if root.resolve() == heldout.ROOT.resolve() else "train"


def checkpoints(run_dir: Path) -> list[Path]:
    """Every kept validation checkpoint, in epoch order.

    epoch*.pt only: final.pt and best.pt are the same weights under other
    names, and counting them twice would weight two epochs double in any
    later aggregate.
    """
    return sorted(run_dir.glob("epoch*.pt"))


def score_run(run_name: str, items, geometry, component_px: int,
              stacked) -> list[dict]:
    run_dir = SWEEP / run_name
    config = run_name.rsplit("_s", 1)[0]
    seed = run_name.rsplit("_s", 1)[1]
    mean, std = train.normalisation(run_name, stacked)
    rows = []
    for path in checkpoints(run_dir):
        state = train.load_checkpoint(path)
        model = train.build_model(config)
        model.load_state_dict(state["model"])
        model.eval()
        for item, geo in zip(items, geometry):
            prob = train.predict_full(model, item["image"], mean, std)
            scores = metrics.evaluate(prob, item["label"] & item["fov"],
                                      item["fov"])
            pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                      component_px)
            rows.append({
                "run": run_name, "config": config, "seed": seed,
                "epoch": int(state["epoch"]), "image": item["name"],
                "dice": round(float(scores["dice"]), 5),
                "cldice": round(float(scores["cldice"]), 5),
                "betti0_err": round(float(scores["betti0_err"]), 3),
                "erl": round(erl.expected_run_length(geo["skel"], pred), 3),
                "skel_px": int(geo["skel"].sum()),
            })
        print(f"  {run_name} {path.name} done", flush=True)
    return rows


def selftest() -> None:
    # The checkpoint list must be epoch-ordered and must not pick up final.pt
    # or best.pt, which are copies of epochs already in the list.
    import tempfile
    with tempfile.TemporaryDirectory() as raw:
        run_dir = Path(raw)
        for name in ("epoch100.pt", "epoch010.pt", "epoch090.pt",
                     "final.pt", "best.pt", "ckpt.pt"):
            (run_dir / name).write_bytes(b"")
        got = [p.name for p in checkpoints(run_dir)]
        assert got == ["epoch010.pt", "epoch090.pt", "epoch100.pt"], got
        print(f"checkpoints() returns {got} -- epoch order, no duplicates of "
              f"final/best")

    # Zero-padding is what makes the sort an epoch sort rather than a string
    # sort; without it epoch100 would come before epoch20.
    assert f"epoch{20:03d}.pt" < f"epoch{100:03d}.pt"
    print("  and the zero-padded names sort as numbers, not as strings")
    print("all checks passed")


def main() -> None:
    global SWEEP, OUT
    if "--selftest" in sys.argv:
        selftest()
        return
    argv = sys.argv[1:]
    if "--results" in argv:
        # The held-out retrain writes to its own root, and its checkpoints
        # must not be scored into the sweep's table: the two roots were
        # trained under different protocols, so one row per checkpoint with
        # no protocol column would silently mix them.
        index = argv.index("--results")
        SWEEP = Path(argv[index + 1])
        OUT = SWEEP / "checkpoint_scores.csv"
        del argv[index:index + 2]
    wanted = [a for a in argv if not a.startswith("--")]
    runs = wanted or sorted(
        path.name for path in SWEEP.iterdir()
        if path.is_dir() and checkpoints(path))
    # A run directory whose config is no longer in CONFIGS is a superseded
    # architecture, not an error to crash on -- and not something to score
    # silently either, since its weights are still on disk and a reader would
    # assume the table covers everything present. Named, then skipped.
    known, retired = [], []
    for run_name in runs:
        (known if run_name.rsplit("_s", 1)[0] in train.CONFIGS
         else retired).append(run_name)
    if retired:
        print(f"skipping {len(retired)} run(s) whose config is retired: "
              f"{', '.join(sorted({r.rsplit('_s', 1)[0] for r in retired}))}",
              flush=True)
    runs = known
    if not runs:
        raise SystemExit(f"no swept runs under {SWEEP}")

    items = drive.load_split("val")
    split = stack_for(SWEEP)
    stacked = train.stack_split(split)
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [{"skel": skeletonize(item["label"] & item["fov"])}
                for item in items]
    print(f"{len(runs)} run(s), {len(items)} images, "
          f"component filter {component_px} px, "
          f"normalised on the {split} split "
          f"({stacked['images'].shape[0]} images)", flush=True)

    # THE TRAP THIS GUARD EXISTS FOR. This file takes run names on the
    # command line, which reads as "score these and add them" -- but it
    # writes with "w". On 2026-09-03 exp/run_paper.sh called it in a loop,
    # once per run, and each call truncated the table; 460 runs became 1.
    # Naming a subset while a table already exists is now refused unless the
    # caller says which it means.
    if wanted and OUT.exists() and "--merge" not in sys.argv:
        raise SystemExit(
            f"{OUT} already exists and you named {len(wanted)} run(s).\n"
            f"This file REBUILDS the table; it does not append. Either pass\n"
            f"every run you want in the table (one invocation, no --merge),\n"
            f"or pass --merge to keep the rows already there.")

    rows = []
    for run_name in runs:
        rows.extend(score_run(run_name, items, geometry, component_px,
                              stacked))
    if not rows:
        raise SystemExit("no checkpoints scored")
    if "--merge" in sys.argv and OUT.exists():
        rescored = {row["run"] for row in rows}
        kept = [row for row in csv.DictReader(OUT.open())
                if row["run"] not in rescored]
        print(f"merging: {len(kept)} rows kept, {len(rows)} rescored")
        rows = kept + rows
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT}")
    print(f"  {len({r['run'] for r in rows})} runs, "
          f"{len({(r['run'], r['epoch']) for r in rows})} checkpoints")


if __name__ == "__main__":
    main()
