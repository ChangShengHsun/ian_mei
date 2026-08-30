"""The GPU queue: which runs, in what order, on which card.

One place decides the run list, because the alternative is a seed range typed
by hand into a shell script -- the exact mistake CLAUDE.md's E12 lesson is
about. run_queue.sh asks this file what to run and does nothing else.

Stages, in prompt.md's priority order:

  gate   the two runs task 1 requires before the other 23 (section 5)
  e13    E13's third capacity point: three arms at 31M, seeds 0-4. This is
         the 15-run experiment the whole curve turns on.
  task1  the five arms on the 31M backbone, 5 seeds; a superset of e13
  recover  every run that has a committed log.csv but no checkpoint on this
           machine, plus task 2's extra K_focal_aug seeds. Ordered so E13's
           OWN narrow arms come first: the two existing points of the capacity
           curve cannot be recomputed without their checkpoints, so those 27
           runs are what turns a one-column table into a three-column one.

`recover` exists because *.pt is gitignored: this clone has 53 runs' worth of
results and zero weights, so every analysis script that loads a checkpoint --
erl, transfer, stratify, break_lengths -- is dead until they are retrained.
It is last because it is recovery, not new science, and prompt.md is explicit
that task 1 is the reason for moving machines.

Ordering is seed-major inside each stage, matching train.py: all arms at seed
0 finish before any arm reaches seed 1, so an interrupted queue leaves a
complete-but-noisy comparison rather than one over-measured arm.

Named gpu_queue and not queue: exp/ goes on sys.path, and a module called
queue.py there shadows the standard library's, which torch's DataLoader
imports at load time. `import torch` then fails with a circular import that
names this file and not the collision.

  python exp/gpu_queue.py task1              # print the list
  python exp/gpu_queue.py task1 --shard 0/2  # this card's half of it
  python exp/gpu_queue.py --selftest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

RESULTS = train.RESULTS
SEEDS = (0, 1, 2, 3, 4)
# E13's three arms lead: they complete the capacity curve, which prompt.md
# section 4.4 calls the best value on the page, and they need no new code.
E13_ARMS = ("A_dice_w64_d5", "B_cldice_w64_d5", "H_aug_w64_d5")
REST_ARMS = ("G_focal_w64_d5", "K_focal_aug_w64_d5")
GATE = ("A_dice_w64_d5_s0", "H_aug_w64_d5_s0")
# Task 2: three cannot support the 44.7% headline; six can.
TASK2_EXTRA = tuple(f"K_focal_aug_s{seed}" for seed in (3, 4, 5))
# The three bases the held-out retrain and the D-E sweep are built on: the
# plain loss, the augmented one, and the strongest arm in the series.
HELDOUT_BASES = ("A_dice", "H_aug", "K_focal_aug")
# Methods with a paper behind them, for the comparison table. B_cldice is
# clDice (CVPR 2021), E_cbdice is cbDice, I_coletra is CoLeTra, J_liot is
# LIOT; G_focal is ours and is here because the table needs it beside them.
PUBLISHED_ARMS = ("B_cldice", "E_cbdice", "I_coletra", "J_liot", "G_focal")


def existing_runs() -> tuple[str, ...]:
    """Runs with a committed log.csv, in seed-major order.

    Globbed off disk rather than listed here for the same reason
    train.trained_runs() is: adding a seed must never require editing a script
    that reads the results. `_cpu_reference/` and the moved-aside baseline are
    excluded by requiring the directory name to parse as config + seed.
    """
    found = []
    for path in sorted(RESULTS.glob("*_s*/log.csv")):
        run_name = path.parent.name
        config, _, seed = run_name.rpartition("_s")
        if not (config in train.CONFIGS and seed.isdigit()):
            continue
        # Skip this experiment's OWN runs. They land here the moment they
        # write a log.csv, and recovering them is the e13 stage's job -- if
        # both stages claimed them, the two cards would train the same run at
        # the same time, into the same directory, and the later save would
        # win silently.
        if train.net_depth(config) != 3:
            continue
        found.append((int(seed), config, run_name))
    return tuple(name for _, _, name in sorted(found))


# The narrow arms of E13's first two capacity points. Their checkpoints are
# what summarize_capacity.py needs to fill columns 1 and 2; stratify.csv
# already holds every one of their measurements.
CURVE_ARMS = ("A_dice", "B_cldice", "H_aug",
              "A_dice_w32", "B_cldice_w32", "H_aug_w32")


def is_curve_arm(run_name: str) -> bool:
    return run_name.rsplit("_s", 1)[0] in CURVE_ARMS


def _seed_major(run_name: str) -> tuple:
    return int(run_name.rsplit("_s", 1)[1]), run_name


def contaminated_runs(path: Path = RESULTS / "erl_best.csv") -> tuple[str, ...]:
    """Every run whose published number was read off best.pt.

    Enumerated from the CSV that holds those numbers, so the repair cannot
    cover fewer runs than the damage. A run whose config has since been
    retired is dropped and named, not crashed on.
    """
    import csv
    if not path.exists():
        return ()
    found = sorted({row["run"] for row in csv.DictReader(path.open())})
    known = tuple(r for r in found if r.rsplit("_s", 1)[0] in train.CONFIGS)
    missing = sorted({r.rsplit("_s", 1)[0] for r in found} -
                     {r.rsplit("_s", 1)[0] for r in known})
    if missing:
        print(f"# retired configs not requeued: {', '.join(missing)}",
              file=sys.stderr)
    return known


def stage(name: str) -> tuple[str, ...]:
    if name == "gate":
        return GATE
    if name == "e13":
        return tuple(f"{arm}_s{seed}" for seed in SEEDS for arm in E13_ARMS)
    if name == "task1":
        runs = [f"{arm}_s{seed}" for seed in SEEDS
                for arm in E13_ARMS + REST_ARMS]
        # E13's arms lead at every seed, so a queue stopped early still has
        # the capacity curve at more seeds than it has the other two arms.
        runs.sort(key=lambda r: (int(r.rsplit("_s", 1)[1]),
                                 r.rsplit("_s", 1)[0] not in E13_ARMS))
        return tuple(runs)
    if name == "recover":
        # TASK2_EXTRA are named ahead of their runs, so once they are trained
        # they also come back from existing_runs() and the concatenation holds
        # each of them twice. A duplicate here is not cosmetic: it would put
        # the same run on both cards at once, into the same directory.
        found = existing_runs()
        runs = found + tuple(r for r in TASK2_EXTRA if r not in found)
        # E13's own arms first, seed-major inside each group. summarize_
        # capacity.py reads seeds off train.trained_runs(), which globs
        # final.pt, so base=16 and base=32 both print "not trained yet" on a
        # fresh clone even though stratify.csv already holds every one of
        # their measurements. These 27 runs are the difference between a
        # verdict and a blank column.
        return tuple(sorted(runs, key=lambda run: (
            not is_curve_arm(run), int(run.rsplit("_s", 1)[1]))))
    # Split out so one card can finish the whole capacity curve as a
    # contiguous block. Dealt round-robin across four shards instead, the 27
    # runs that unblock the verdict would finish scattered through the night
    # for no gain -- the rest of `recover` unblocks nothing until it is all
    # there.
    # Task B: the 31M arms again, keeping every validated epoch so A3's
    # winning selection rule can be applied at real capacity. Written into a
    # separate results root by the runner -- retraining over the published
    # w64_d5 directories would replace the weights stratify.csv and erl.csv
    # were computed from.
    if name == "taskb":
        return tuple(f"{arm}_s{seed}" for seed in range(5)
                     for arm in E13_ARMS)
    # Task C: seeds 6-11 of the 117k arms, to settle the two ERL comparisons
    # that sit on the gate's edge and to give e13b section 3 a THIRD batch of
    # six seeds. B_cldice is in the list although the work order's C1 omits
    # it: e13b's unreproduced verdict IS B_cldice minus A_dice, so without it
    # C3 cannot be answered at all.
    if name == "taskc":
        arms = ("A_dice", "B_cldice", "H_aug", "G_focal", "K_focal_aug")
        return tuple(f"{arm}_s{seed}" for seed in range(6, 12) for arm in arms)
    # D1: the tangent-direction head, six seeds of each of the two arms it
    # can be compared against. Trained into the selection sweep's results
    # root with every validated epoch kept, so the SAME selection rule that
    # picks a checkpoint for A_dice and H_aug picks one for A_dice_dir and
    # H_aug_dir -- comparing a new arm under one protocol against a baseline
    # under another is the confound e13b R.3 already paid for.
    if name == "d1":
        return tuple(f"{arm}_s{seed}" for seed in range(6)
                     for arm in ("A_dice_dir", "H_aug_dir"))
    # D-B and its ablation: the propagation layer driven by the model's own
    # direction head, against the same layer driven by noise. Paired so a
    # queue stopped early still has both sides of the comparison at the seeds
    # it reached, rather than one arm measured and its control missing.
    if name == "d1b":
        arms = tuple(f"{base}_dir_prop{shuffle}_{reach}_c025"
                     for base in ("A_dice", "H_aug")
                     for reach in train.PROPAGATION_REACHES
                     for shuffle in ("", "_shuf"))
        return tuple(f"{arm}_s{seed}" for seed in range(6) for arm in arms)
    # D-E: the cheap competitor, no direction anywhere in it.
    if name == "d1e":
        return tuple(f"{arm}_s{seed}" for seed in range(6)
                     for arm in ("A_dice_clw", "H_aug_clw"))
    # D-B x D-E: both interventions in one arm, six seeds.
    if name == "d1f":
        arms = tuple(f"{base}_clw_dir_prop_{reach}_c025"
                     for base in ("A_dice", "H_aug")
                     for reach in train.PROPAGATION_REACHES)
        return tuple(f"{arm}_s{seed}" for seed in range(6) for arm in arms)
    # ------------------------------------------------ the held-out protocol
    # Everything below is retrained under --protocol heldout, into its own
    # results root. The defect being repaired: drive.load_split("val") reads
    # DRIVE's official TEST images, and best.pt was the epoch with the highest
    # Dice on them. A checkpoint chosen as the best of ten epochs on the set
    # it is then reported on is the maximum of ten draws.
    #
    # Every arm a conclusion rests on has to come back under one protocol --
    # comparing a new arm measured honestly against a baseline measured with
    # the leak is worse than either, because the difference then mixes the
    # intervention with the protocol.
    #
    # HELDOUT_BASES are the three the D-E sweep is built on. K_focal_aug is
    # the strongest arm measured in the series and has never been crossed with
    # D-E at all; that cell is the one new question in this batch.
    if name == "heldout":
        arms = tuple(f"{base}_clw{weight}" for base in HELDOUT_BASES
                     for weight in train.CENTRELINE_WEIGHTS)
        return tuple(f"{arm}_s{seed}" for seed in range(6)
                     for arm in HELDOUT_BASES + arms)
    # The published methods, under the same protocol, so the comparison table
    # has one row per method and one protocol for all of them. E_cbdice has no
    # checkpoint on this machine at all, so it is retrained rather than scored.
    # The cleanup half: every run whose number was read off best.pt, which is
    # the one path that really leaks -- best.pt is the highest-Dice epoch over
    # all 20 test images, so a result taken from it is the maximum of ten
    # draws on the set it is reported on. erl.csv and stratify.csv are read
    # off final.pt at a fixed epoch 100 and are NOT affected; erl_best.csv and
    # stratify_best.csv are.
    #
    # The run list is READ OFF the contaminated CSV rather than typed here.
    # A hand-written seed range is the E12 mistake this repo already paid for
    # once, and it is worse in a repair: a cleanup that silently covers 45 of
    # 72 runs looks exactly like one that covers all of them.
    if name == "heldout_series":
        runs = contaminated_runs()
        already = set(stage("heldout"))
        # E_cbdice has no checkpoint on this machine at all, so it never
        # reached erl_best.csv and has to be added rather than recovered.
        runs = runs + tuple(f"E_cbdice_s{seed}" for seed in range(6))
        # The 117k arms go to six seeds whatever they had before. Half of
        # them were measured at three, and a paired t on three seeds has two
        # degrees of freedom -- it satisfies the gate's "at least three" and
        # supports almost nothing. Since every one of them is being retrained
        # anyway, the marginal cost of seeds 3-5 buys a table that can carry
        # a claim. The wide arms keep the seeds they had: they cost five
        # times as much each, and they are points on the capacity curve
        # rather than arms a conclusion rests on.
        narrow_arms = sorted({r.rsplit("_s", 1)[0] for r in runs
                              if train.base_width(r.rsplit("_s", 1)[0]) == 16})
        cheap = tuple(f"{arm}_s{seed}" for arm in narrow_arms
                      for seed in range(6))
        cheap = tuple(r for r in cheap if r not in already)
        wide = tuple(r for r in runs if r not in already
                     and train.base_width(r.rsplit("_s", 1)[0]) != 16)
        # Narrow arms first: 45 of them cost what 9 wide ones do, and they
        # carry every conclusion in the series up to E13.
        return tuple(sorted(cheap, key=_seed_major)
                     + sorted(wide, key=_seed_major))
    # Seeds 6-11 of the D-E sweep. Queued because the gate that matters here
    # is the SIGN rule, and six seeds is where it is weakest: on 2026-08-28
    # A_dice_clw beat A_dice by +240.0 ERL at t 3.12 -- an effect the same
    # size as H_aug_clw's +249.5, which HELD -- and failed on one seed of six
    # coming back -201. That is not "no effect", it is "not enough seeds to
    # tell". Twelve seeds settles it either way.
    #
    # Runs last, so it costs nothing if the first six seeds are unambiguous.
    if name == "heldout_seeds":
        arms = tuple(f"{base}_clw{weight}" for base in HELDOUT_BASES
                     for weight in train.CENTRELINE_WEIGHTS)
        return tuple(f"{arm}_s{seed}" for seed in range(6, 12)
                     for arm in HELDOUT_BASES + arms)
    # D-C, the redesign of D-B. Every length carries its two controls, so a
    # queue stopped early still has curvature and direction isolated at the
    # lengths it reached rather than one arm measured and its controls
    # missing -- the shape D-B's queue already used, for the same reason.
    if name == "snake":
        out = []
        for seed in range(6):
            for base in ("A_dice", "H_aug"):
                for taps in train.SNAKE_TAPS:
                    for kind in ("snake", "snkstr", "snkshf"):
                        out.append(f"{base}_dir_{kind}_k{taps:02d}_t1_s{seed}")
                for kind in ("snake", "snkshf"):
                    out.append(f"{base}_dir_{kind}_k16_t2_s{seed}")
        return tuple(out)
    if name == "heldout_published":
        return tuple(f"{arm}_s{seed}" for seed in range(6)
                     for arm in PUBLISHED_ARMS)
    if name == "curve":
        return tuple(run for run in stage("recover") if is_curve_arm(run))
    if name == "recover_rest":
        return tuple(run for run in stage("recover")
                     if not is_curve_arm(run))
    raise SystemExit(f"unknown stage {name!r}; try gate, task1, recover "
                     f"or heldout")


def shard(runs: tuple[str, ...], index: int, total: int) -> tuple[str, ...]:
    """Deal the list round-robin across cards.

    Round-robin rather than a split down the middle: the arms differ in cost
    (a topology loss is ~30% slower than BCE+Dice), so contiguous halves would
    leave one card idle while the other finishes. Dealing also keeps both
    cards inside the same seed, which is what makes an interrupted queue
    balanced across arms.
    """
    return tuple(runs[index::total])


def pending(runs: tuple[str, ...], root: Path = RESULTS) -> tuple[str, ...]:
    """Drop runs whose final.pt is already on disk under `root`.

    Gate on the artifact, never on a PID: CLAUDE.md's long-running-jobs note
    is that Wait-Process on a dead PID returns instantly and starts the next
    job against a half-finished predecessor.

    `root` is a parameter because not every stage writes to exp/results. D1
    and task B train into their own results roots, and asking about the
    default one would report every run of theirs as pending forever -- a
    retry loop reading that would never see its own work finish.
    """
    return tuple(name for name in runs
                 if not (root / name / "final.pt").exists())


def selftest() -> None:
    task1 = stage("task1")
    assert len(task1) == len(SEEDS) * 5 == 25, len(task1)
    assert len(set(task1)) == len(task1), "duplicate run in task1"
    for run_name in task1:
        config = run_name.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
        assert train.base_width(config) == 64 and train.net_depth(config) == 5
    # Seed-major, and E13's three arms before the other two inside each seed.
    seeds = [int(r.rsplit("_s", 1)[1]) for r in task1]
    assert seeds == sorted(seeds), seeds
    assert task1[:3] == tuple(f"{arm}_s0" for arm in E13_ARMS), task1[:3]
    for name in GATE:
        assert name in task1, name

    # Every shard together must be the whole list and nothing twice: a card
    # silently skipping a run is a queue that looks finished and is not.
    for total in (1, 2, 3):
        parts = [shard(task1, index, total) for index in range(total)]
        assert sorted(sum(parts, ())) == sorted(task1), total
        assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1

    e13 = stage("e13")
    assert len(e13) == 15, len(e13)
    assert set(e13) <= set(task1), "e13 must be a subset of task1"
    assert e13[:3] == tuple(f"{arm}_s0" for arm in E13_ARMS), e13[:3]
    assert [int(r.rsplit("_s", 1)[1]) for r in e13] == sorted(
        int(r.rsplit("_s", 1)[1]) for r in e13), "e13 must be seed-major"
    for run_name in e13:
        config = run_name.rsplit("_s", 1)[0]
        assert train.base_width(config) == 64, config
        assert train.net_depth(config) == 5, config
        # The trap E13 already paid for once: AUGMENTS is keyed on the FULL
        # config name, so a width variant missing from it trains with no
        # augmentation at all and still answers to the augmented arm's name.
        base = "_".join(token for token in config.split("_")
                        if not (token[:1] in "wd" and token[1:].isdigit()))
        assert train.AUGMENTS.get(config, ()) == train.AUGMENTS.get(base, ()), \
            f"{config} does not carry {base}'s augmentation tuple"

    recover = stage("recover")
    assert len(set(recover)) == len(recover), "duplicate run in recover"
    # The two cards run e13 and recover concurrently, so the stages must be
    # disjoint or both will train the same run into the same directory.
    assert not set(recover) & set(e13), sorted(set(recover) & set(e13))
    assert not set(recover) & set(task1), sorted(set(recover) & set(task1))
    # The 27 runs the capacity curve's first two columns need must lead.
    taskb, taskc = stage("taskb"), stage("taskc")
    assert len(taskb) == 15, len(taskb)
    assert len(taskc) == 30, len(taskc)
    for run_name in taskb:
        assert train.net_depth(run_name.rsplit("_s", 1)[0]) == 5, run_name
    for run_name in taskc:
        config = run_name.rsplit("_s", 1)[0]
        assert train.net_depth(config) == 3 and train.base_width(config) == 16
        assert int(run_name.rsplit("_s", 1)[1]) >= 6, run_name
    # Task C must not collide with the runs that were already PUBLISHED, or
    # it would retrain over the weights every earlier verdict was measured
    # from. Those are seeds 0-5; task C is 6-11 by construction.
    #
    # Written first as "must not collide with existing_runs()", which passed
    # until task C started and then failed on task C's own directories -- a
    # self-check that goes red the moment the thing it checks starts working
    # is a check on the wrong quantity. The quantity is the seed number.
    published = {run for run in existing_runs()
                 if int(run.rsplit("_s", 1)[1]) < 6}
    assert not (set(taskc) & published), sorted(set(taskc) & published)
    assert all(int(run.rsplit("_s", 1)[1]) >= 6 for run in taskc)
    print(f"  taskb {len(taskb)} runs @31M, taskc {len(taskc)} runs @117k "
          f"seeds 6-11")

    d1 = stage("d1")
    assert len(d1) == 12, len(d1)
    assert len(set(d1)) == len(d1), "duplicate run in d1"
    for run_name in d1:
        config = run_name.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
        assert train.uses_direction(config), config
        assert train.net_depth(config) == 3 and train.base_width(config) == 16
        # The trap E13 paid for, in its D1 shape: AUGMENTS is keyed on the
        # FULL config name, so H_aug_dir missing from it would train with no
        # augmentation at all and still answer to H_aug's name.
        base = config[:-len("_dir")]
        assert train.AUGMENTS.get(config, ()) == train.AUGMENTS.get(base, ()),\
            f"{config} does not carry {base}'s augmentation tuple"
        # And it must not collide with a run that already exists under a
        # different protocol.
        assert run_name not in taskc and run_name not in existing_runs()
    print(f"  d1 {len(d1)} runs: {d1[0]} .. {d1[-1]}, each carrying its "
          f"namesake's augmentation")

    d1b, d1e = stage("d1b"), stage("d1e")
    reaches = len(train.PROPAGATION_REACHES)
    assert len(d1b) == 2 * reaches * 2 * 6, len(d1b)
    assert len(d1e) == 12, len(d1e)
    for run_name in d1b + d1e:
        config = run_name.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
        assert train.net_depth(config) == 3 and train.base_width(config) == 16
        # The AUGMENTS trap again, in its D-B shape. Four of these six names
        # start with H_aug, and one missing from AUGMENTS trains unaugmented
        # while still answering to the augmented arm's name.
        base = "_".join(config.split("_")[:2])
        assert train.AUGMENTS.get(config, ()) == train.AUGMENTS.get(base, ()),\
            f"{config} does not carry {base}'s augmentation tuple"
        # A propagation arm must name a reach that builds a kernel bigger than
        # one pixel. Checked here, before six hours of training, because the
        # failure it guards is silent: the layer becomes the identity and the
        # arm reports that propagation does nothing.
        if train.uses_propagation(config):
            along, _ = train.propagation_geometry(config)
            assert along > 1.0, (config, along)
    # Every reach must have its shuffled control at every seed, or the
    # ablation is missing exactly where the answer would be read.
    for run_name in d1b:
        if "shuf" not in run_name:
            assert run_name.replace("_prop_", "_prop_shuf_") in d1b, run_name
    assert sum("shuf" in r for r in d1b) == len(d1b) // 2
    d1f = stage("d1f")
    assert len(d1f) == 2 * reaches * 6, len(d1f)
    for run_name in d1f:
        config = run_name.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
        # The combination arm must actually carry BOTH interventions, or it
        # is one of them under a name claiming two.
        assert train.uses_centreline_weight(config), config
        assert train.uses_propagation(config), config
        assert train.uses_direction(config), config
        base = "_".join(config.split("_")[:2])
        assert train.AUGMENTS.get(config, ()) == train.AUGMENTS.get(base, ())
        assert train.propagation_geometry(config)[0] > 1.0, config
    print(f"  d1b {len(d1b)} runs over reaches {train.PROPAGATION_REACHES} "
          f"(half the shuffled control), d1e {len(d1e)}, d1f {len(d1f)}")

    curve, rest = stage("curve"), stage("recover_rest")
    # The 27 published runs of E13's narrow arms are what turn a one-column
    # capacity table into a three-column one, so they must all be here and
    # must lead. Not "== 27": that was true until task C added seeds 6-11 of
    # the same arms, and a count is the wrong thing to pin -- what matters is
    # that nothing published is missing.
    assert len(curve) >= 27, len(curve)
    assert {run for run in curve
            if int(run.rsplit("_s", 1)[1]) < 6} >= {
        run for run in existing_runs()
        if is_curve_arm(run) and int(run.rsplit("_s", 1)[1]) < 6}
    assert recover[:len(curve)] == curve, "the curve arms must lead recover"
    assert set(curve) | set(rest) == set(recover), "curve + rest != recover"
    assert not set(curve) & set(rest), "a run is in both curve and rest"
    for run_name in curve:
        assert train.net_depth(run_name.rsplit("_s", 1)[0]) == 3, run_name
    for run_name in recover:
        config = run_name.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
    assert "A_dice_s0" in recover, "the existing runs should be enumerated"
    assert not any("cpu_baseline" in r or "reference" in r for r in recover), \
        "the moved-aside CPU baseline must not be retrained over"
    for name in TASK2_EXTRA:
        assert name in recover, name
    # An empty stage must print nothing at all, not a blank line: every gate
    # in the shell scripts counts lines with `wc -l`, so one stray newline
    # reads as one pending run and the gate never opens. This cost five hours
    # of two idle cards on 2026-08-26 -- every run had finished and both
    # post-processing scripts were still polling.
    import io, contextlib
    for sample in ((), ("a_s0", "b_s1")):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            if sample:
                print("\n".join(sample))
        assert len(captured.getvalue().splitlines()) == len(sample), sample
    print("  an empty queue prints zero lines, not one blank one")

    # The held-out batch. Pinned to the quantities that carry the meaning --
    # a baseline for every base, every swept weight present, and each seed a
    # COMPLETE grid -- not to a total, which stays true while the grid rots.
    heldout = stage("heldout")
    assert len(set(heldout)) == len(heldout), "duplicate run in heldout"
    per_seed = len(HELDOUT_BASES) * (1 + len(train.CENTRELINE_WEIGHTS))
    for seed in range(6):
        block = [r for r in heldout if r.endswith(f"_s{seed}")]
        assert len(block) == per_seed, (seed, len(block))
        # Every arm at this seed, and its own baseline with it: a sweep whose
        # baseline lands three hours later cannot be read while it runs.
        assert set(block[:len(HELDOUT_BASES)]) == {
            f"{base}_s{seed}" for base in HELDOUT_BASES}, block[:3]
    seeds = [int(r.rsplit("_s", 1)[1]) for r in heldout]
    assert seeds == sorted(seeds), "heldout must be seed-major"
    for run_name in heldout + stage("heldout_published"):
        config = run_name.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
        # The E13 trap: an augmented arm's variant missing from AUGMENTS
        # trains unaugmented and still answers to the augmented arm's name.
        base = config.split("_clw")[0]
        assert train.AUGMENTS.get(config, ()) == train.AUGMENTS.get(base, ()), \
            config
    print(f"  heldout: {len(heldout)} runs, {per_seed} per seed, "
          f"every arm carries its base's augmentation")

    # The cleanup stage must cover the damage. Pinned to "every contaminated
    # ARM comes back" rather than to a run count, which stays true while an
    # arm silently drops out.
    series = stage("heldout_series")
    assert len(set(series)) == len(series), "duplicate run in heldout_series"
    damaged = {r.rsplit("_s", 1)[0] for r in contaminated_runs()}
    covered = {r.rsplit("_s", 1)[0] for r in series + stage("heldout")}
    assert damaged <= covered, sorted(damaged - covered)
    assert not set(series) & set(stage("heldout")), "a run queued twice"
    # Six seeds for everything cheap enough to have them.
    for arm in covered:
        if train.base_width(arm) != 16:
            continue
        seeds = {r.rsplit("_s", 1)[1] for r in series + stage("heldout")
                 if r.rsplit("_s", 1)[0] == arm}
        assert seeds == {str(s) for s in range(6)}, (arm, sorted(seeds))
    print(f"  heldout_series: {len(series)} runs covering all "
          f"{len(damaged)} best.pt-contaminated arms, 117k arms at 6 seeds")

    # The seed extension must continue the sweep, not restart it: the same
    # arms at seeds the first batch does not have.
    more = stage("heldout_seeds")
    assert not set(more) & set(heldout), "a seed queued twice"
    assert {r.rsplit("_s", 1)[0] for r in more} == \
        {r.rsplit("_s", 1)[0] for r in heldout}, "different arms"
    assert {int(r.rsplit("_s", 1)[1]) for r in more} == set(range(6, 12))
    print(f"  heldout_seeds: {len(more)} runs, seeds 6-11 of the same "
          f"{len(set(r.rsplit('_s', 1)[0] for r in more))} arms")

    # D-C. Pinned to "every arm has both of its controls at the same seed",
    # which is what makes the comparison readable, not to a total.
    snake_runs = stage("snake")
    assert len(set(snake_runs)) == len(snake_runs), "duplicate in snake"
    for seed in range(6):
        block = {r for r in snake_runs if r.endswith(f"_s{seed}")}
        for run in list(block):
            if "_snake_" not in run:
                continue
            assert run.replace("_snake_", "_snkshf_") in block, run
            if "_t1" in run:
                assert run.replace("_snake_", "_snkstr_") in block, run
    for run in snake_runs:
        config = run.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
        base = config.split("_dir_")[0]
        assert train.AUGMENTS.get(config, ()) == train.AUGMENTS.get(base, ())
    print(f"  snake: {len(snake_runs)} runs, every arm with its controls at "
          f"the same seed")

    # pending() must answer about the root the runs are actually written to.
    import tempfile
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / d1[0]).mkdir(parents=True)
        (root / d1[0] / "final.pt").write_bytes(b"")
        assert pending(d1, root) == d1[1:], pending(d1, root)
        # And the default root knows nothing about them: D1 trains into the
        # sweep, so asking about exp/results would call all 12 pending for
        # ever and a retry loop would never see its own work land.
        assert pending(d1) == d1, "d1 must not be found under RESULTS"
    print("  pending() answers about the root it is given, not a fixed one")

    print(f"gate {len(GATE)} runs, e13 {len(e13)} runs, "
          f"task1 {len(task1)} runs, recover {len(recover)} runs "
          f"({len(curve)} of them E13's narrow arms, first)")
    print(f"  task1 seed 0: {', '.join(task1[:5])}")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = stage(argv[0] if argv else "task1")
    if "--shard" in sys.argv:
        index, total = sys.argv[sys.argv.index("--shard") + 1].split("/")
        runs = shard(runs, int(index), int(total))
    if "--pending" in sys.argv:
        root = RESULTS
        if "--results" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--results") + 1])
        runs = pending(runs, root)
    # Only when there is something to print. "\n".join(()) is the empty
    # string, and print() still emits a newline, so an empty queue came out as
    # one blank line and `... | wc -l` answered 1. Both gates that wait for a
    # stage to reach zero -- sweep_missing_best.sh and post_e13.sh -- then wait
    # forever on a queue that finished hours ago. Cost five hours of two idle
    # cards on 2026-08-26.
    if runs:
        print("\n".join(runs))


if __name__ == "__main__":
    main()
