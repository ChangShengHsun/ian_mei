# The held-out protocol, and what it repairs

2026-08-28. Queued as `exp/run_heldout.sh`, behind the running `d1all` job.

## The defect

`exp/drive.py`'s `load_split("val")` reads images 01-20. Those are DRIVE's
official **test** set. Two things in this repo chose a checkpoint on them.

| Path | What it selects on | Leaks? |
|---|---|---|
| `best.pt` | highest Dice over **all 20 test images** | **Yes.** The number read from it is the maximum of ten validated epochs on the set it is then reported on. |
| rules (i)-(iv) in `checkpoint_scores.csv` | odd test images; reports on even ones (`select_checkpoint.py:61`) | No leak into the reported half. But it still touches the test set before reporting, and it halves the reporting set to 10 images. |
| `final.pt` | nothing -- fixed epoch 100 | No. |

So the contamination is narrower than "every number in the series". Concretely:

- **Contaminated:** `exp/results/erl_best.csv` and `exp/results/stratify_best.csv`
  (72 runs, 16 arms), and anything downstream quoting the "best.pt protocol"
  column, including `transfer_ceiling.py`, which uses `best.pt` on every dataset.
- **Not contaminated:** `erl.csv`, `stratify.csv` (both `final.pt`), and the
  D-B / D-E verdicts in `interim_reach.txt`, which are computed at rule (iv)
  on the report half.

That last line matters: the negative D-B result and the positive D-E result
both stand. They were not produced by the leak.

## The repair

`--protocol heldout` (`exp/train.py`). Five of the twenty **training** images
(`drive.DEV_IDS = 24, 28, 32, 36, 40`, every fourth so the choice cannot follow
acquisition order) are held out. The model is fitted on the other 15, every
selection rule reads the 5, and the test set is read once, whole.

| | legacy | heldout |
|---|---|---|
| fitted on | 20 train images | 15 train images |
| checkpoint chosen on | 20 test images (`best.pt`) or 10 of them (rules) | 5 held-out train images |
| reported on | 10 test images (the even half) | **20 test images** |

Two consequences, one of them a gain: no selection of any kind touches a
reported image, and the paired test runs over twice the images.

The cost is real and is not hidden. 15 training images instead of 20 lowers
absolute Dice. Every arm pays it equally, so comparisons between arms are
unaffected -- but absolute numbers are **not** comparable to the legacy runs
or to published DRIVE results, and any table mixing the two protocols is
wrong. `protocol.txt` is written into every run directory and `train.py`
refuses to resume a directory stamped with a different protocol.

`legacy` stays the default. The `d1all` queue was mid-flight when this landed,
and changing a default under a running queue gives one comparison two
protocols. The default flips once that queue drains.

## What was queued

Into `exp/results/heldout/`, seed-major, `--keep-epochs`:

- **`heldout`** (90 runs) -- `A_dice`, `H_aug`, `K_focal_aug`, each with
  centreline weight 1, 2, 4, 8, at 6 seeds. D-E is the only intervention in
  this series that passed the seed gate (`H_aug_clw` +249.5 ERL, t 5.23, 6/6
  seeds) and its weight was fixed at 2 by an argument from vessel geometry,
  never swept. `K_focal_aug`, the strongest arm measured, has never been
  crossed with D-E at all.
- **`heldout_series`** (72 runs) -- every arm whose published number came off
  `best.pt`, enumerated from `erl_best.csv` rather than typed out, plus
  `E_cbdice`, which has no checkpoint on this machine. The 117k arms go to six
  seeds: half of them were measured at three, and a paired t on three seeds has
  two degrees of freedom.

The weight moved from a module constant into the config name (`clw4` = 4.0),
the same rule `propagation_geometry` already follows: one config name must mean
one model, or two models write into one directory.

## Pre-registered predictions

Written before the first held-out run started, in `exp/select_heldout.py`:

1. The D-E effect survives the protocol change on `H_aug`.
2. The response to weight is single-peaked, not monotone -- weight 8 makes the
   centreline outvote the vessel body and costs Dice without buying run length.
3. The effect is **smaller** on `K_focal_aug` than on `H_aug`; focal loss
   already up-weights hard pixels and the centreline is where the model
   hesitates, so the two overlap.
4. `A_dice`, unaugmented, stays weakest at every weight.

Gate unchanged: paired t over (image, seed) with t > 2, every seed agreeing in
sign, at least three seeds.

## Checks that run before anything is queued

`exp/test_protocol.py`, `exp/drive.py`, `exp/select_heldout.py --selftest` and
`exp/gpu_queue.py --selftest`, all of which `run_heldout.sh` runs and refuses
to queue without. They assert the mechanism, not an outcome:

- `fit` and `dev` partition the training directory and neither meets `test`.
- `legacy` **does** select on the test set and `heldout` does not -- the
  defect is asserted to exist in the thing being replaced.
- A run directory refuses a second protocol.
- Every swept arm carries its base's loss and augmentation (the E13 trap: a
  variant missing from `AUGMENTS` trains unaugmented under the augmented arm's
  name).
- The verdict reports on all 20 test images, not the even half.
- The gate refuses a split-sign effect (mean +190, t 9.4) -- sign agreement,
  not just significance.
