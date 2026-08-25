# prompt.md — picking this up on the lab GPU machine

Written 2026-08-25 on the laptop this work has run on so far. Read this first,
then `CLAUDE.md` (repo rules), then `stage-report/README.md` (what is known).

Everything below assumes you are on a machine with an NVIDIA GPU. That is the
one thing this project has never had, and it is the reason this file exists.

---

## Start here

In order. Do not skip ahead; task 0 gates everything after it.

1. **Set the machine up** — section 6. Ends with `torch.cuda.is_available()`
   printing `True` and every `--selftest` passing.
2. **Make the code use the GPU** — section 3. It currently does not, at all.
   Ends with one training run reproducing an existing `log.csv`.
3. **Train the core five configs on a real ~30M U-Net** — section 5, task 1.
   This is the actual reason for moving machines.

If you only get through step 3, the trip was worth it. Tasks 2 to 5 are the
rest of the queue, in priority order.

---

## 1. What this repo is, in one minute

Curvilinear-structure segmentation, mostly retinal vessels on DRIVE, with
STARE / HRF / TopoMortar for cross-dataset checks.

The problem the whole series is about: **pixel-overlap metrics cannot see
topology.** A model at Dice 0.81 can cut a vessel tree into ninety pieces, and
Dice barely moves. Everything here is built to measure and fix that.

- `exp/*.py` — one script per experiment, docstring states the question.
- `exp/results/*.csv` — the measurements. Checkpoints (`*.pt`) are gitignored.
- `stage-report/*.md` — one report per experiment, `README.md` is the index.

Nineteen experiments so far, all on 6 CPU cores, no GPU.

## 2. Where the work stands

The current best model is `K_focal_aug` = confidence-gated clDice + geometric
and photometric augmentation. On DRIVE it traces **44.7%** of the vessel tree
without error against the baseline's 26.4% (ERL 3968 vs 2341, seed-gated,
t=13.76). Betti-0 error halves, 89.7 -> 35.8.

Four findings that still stand, in rough order of how much they should shape
what you do next:

1. **The loss function's advantage is a small-model artifact.** At 4x width
   (117k -> 467k params) clDice's dim-band advantage over BCE+Dice collapses
   from +0.0186 (gated, 6 seeds) to -0.0040 (not gated), while augmentation's
   grows from +0.0185 to +0.0399. See `stage-report/e13_capacity.md`.
2. **Single-dataset evaluation underestimates augmentation 4-6x.** DRIVE
   internal +0.0106; zero-shot STARE +0.0476, HRF +0.0678.
   `stage-report/e17_transfer.md`.
3. **93% of visible "breaks" do not sever connectivity.** Raw break counts
   measure coverage, not topology. `stage-report/e10_break_anatomy.md`.
4. **Capacity only pays when paired with augmentation.** Without augmentation,
   4x width makes zero-shot Betti-0 *worse*, 4 of 4 cells.

Honest caveats, all of which the GPU is supposed to fix:

- 3 seeds on `K_focal_aug`, and seed 1 has four documented sign flips.
- K's ERL advantage does **not** transfer (t=0.63 / 0.57 on STARE / HRF). Only
  its Betti-0 advantage does. "Fewer pieces" travels, "longer runs" does not.
- Whole-image Dice drops 0.8109 -> 0.7989. The trade is real and must stay in
  every report of this model.
- 467k is still ~60x smaller than the field standard. Finding 1 above rules out
  one order of magnitude, not the gap.

## 3. Task 0 — the code does not use the GPU yet

**Nothing in `exp/` touches CUDA.** `train.py:35` is `torch.set_num_threads(6)`
and that is the whole device story. Opening this on a GPU box and running
`train.py` trains on CPU at laptop speed. Fix this before anything else.

The installed torch is the CPU build (`2.13.0+cpu`). Install the CUDA build for
whatever driver the box has, then:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Must print `True` before you touch the training code.

Every tensor boundary that needs a device move, verified by reading the file
on 2026-08-25:

| Location | What is needed |
|---|---|
| `train.py:526-528` | `sample_batch` returns three CPU tensors; move all three |
| `train.py:559,561` | `predict_full` builds a tensor, then calls `.numpy()`; needs `.to(device)` and `.cpu().numpy()` |
| `train.py:275-278` | **the landmine.** `_cb_weights` round-trips through numpy for the distance transform and rebuilds a **CPU** tensor, then multiplies it by `mask`, which is on GPU. Device mismatch, throws at runtime. Needs `.to(mask.device)` |
| `build_model` callers | `model.to(device)` |
| `torch.load` (`train.py:608`, and every analysis script) | pass `map_location` |

`BlurPool`'s kernel is a registered buffer (`train.py:73`), so `model.to()`
carries it. That one is already fine.

Suggested shape: one module-level `DEVICE` in `train.py` resolved from
`torch.cuda.is_available()`, imported by the analysis scripts rather than each
deciding for itself. **Repeat of the E16 lesson (`stage-report/README.md`
lesson seven): move the whole decision, not half of it.** If "which device"
lives in `train.py` but "which normalisation constants" or "map_location" stays
at each call site, you will get the same class of silent bug that once reported
Dice 0.0000 as a finding.

**Acceptance for task 0:** `python exp/train.py A_dice_s0` reproduces the
existing `exp/results/A_dice_s0/log.csv` within noise on GPU, and every
`exp/test_*.py` still passes. Do not start task 1 until this holds.

## 4. Budget, architecture, and the number to measure first

### 4.1 What a run costs today

One run is fixed at **31,200 steps** (`PATCHES_PER_EPOCH 10_000 // BATCH 32`
= 312 steps x `EPOCHS 100`). Measured on the laptop, 6 CPU threads, batch 32 at
48x48, on 2026-08-25:

| base | params | CPU ms/step | vs base=16 |
|---|---|---|---|
| 16 | 117k | 135 | 1.0x |
| 32 | 467k | 405 | 3.0x |
| 64 | 1.86M | 1,174 | 2.9x |
| 128 | 7.45M | 3,651 | 3.1x |
| 256 | 29.8M | ~11,000 (extrapolated) | ~81x |

**Doubling `base` costs 4x the parameters and about 3x the step time.**

Also measured: `sample_batch` costs **1.0 ms** without augmentation and
**5.1 ms** with it, against 135-3,651 ms of compute. The data pipeline is 1-2%
of a step, so **do not rewrite it as a DataLoader with workers** when porting
to GPU. It is not the bottleneck and never was.

At base=256 a single run is ~95 hours on this laptop. That is the whole reason
for the move.

### 4.2 Do not just widen the 3-level net

`TinyUNet` is **3 levels**. Reaching 30M parameters with it means `base=256`,
which is an extremely wide, extremely shallow network — and every parameter
sits at full 48x48 resolution, so it is the most expensive possible way to
spend 30M parameters.

The standard U-Net reaches ~31M with **5 levels at base=64**, where the extra
capacity lives at 6x6 and 3x3 and costs almost nothing per parameter.

**Use a 5-level base=64 U-Net, not a widened 3-level one.** It is cheaper per
run, and it is the architecture a reviewer expects. Two consequences:

- `base_width()` reads capacity off the run name (`_w32` -> 32). A depth change
  needs the same treatment, or checkpoints will load into the wrong shape. One
  place decides, as `build_model` already does — see the E16 lesson in §3.
- 48x48 patches through 5 levels is 48/24/12/6/3, which works but is tight.
  128x128 is the conventional choice. **Patch area drives cost linearly**: going
  48 -> 128 is 7x the pixels and therefore roughly 7x the step time. This single
  decision moves the budget more than the choice of GPU does.

### 4.3 Measure before you queue anything

Every hour figure below is **extrapolated from CPU measurements, not measured on
a GPU**. Before committing to a queue:

```python
# exp/bench_step.py -- time one training step at the real batch and patch size.
import sys, time, torch
sys.path.insert(0, "exp"); import train

model = train.TinyUNet(base=64).to(train.DEVICE)      # swap in the 5-level net
opt = torch.optim.Adam(model.parameters())
img = torch.randn(train.BATCH, 1, train.PATCH, train.PATCH, device=train.DEVICE)
lab = (torch.rand_like(img) > 0.9).float()
dist = torch.rand_like(img)

for _ in range(5):                                     # warm up cuDNN autotune
    opt.zero_grad(); train.compute_loss(model(img), lab, dist, None, img).backward(); opt.step()
torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    opt.zero_grad(); train.compute_loss(model(img), lab, dist, None, img).backward(); opt.step()
torch.cuda.synchronize()
ms = 1000 * (time.time() - start) / 50
steps = (train.PATCHES_PER_EPOCH // train.BATCH) * train.EPOCHS
print(f"{ms:.0f} ms/step -> {ms * steps / 3_600_000:.1f} h per run, "
      f"{sum(p.numel() for p in model.parameters()):,} params")
```

`torch.cuda.synchronize()` is not optional: CUDA calls are asynchronous, so
timing without it measures how fast Python queued the work, not how fast the
GPU did it. That mistake reports a 20x speedup that is not there.

Five minutes of timing replaces the whole table below with real numbers. Do it
first and rescale.

### 4.4 The schedule, at ~1 h per 30M run

An RTX 4070 Ti at 30M, batch 32, 48x48 is estimated at 75-150 ms/step, so
**~40-80 minutes per run**; a 5-level base=64 net should land at the fast end,
128x128 patches at the slow end or beyond.

| Work | runs | GPU hours |
|---|---|---|
| **E13 three-point capacity curve @30M** (3 arms x 5 seeds) | 15 | **15** |
| Task 1: five configs x 5 seeds @30M | 25 | 25 |
| Transfer @30M (inference only, no training) | — | ~1 |
| Task 3: published topology losses @30M | 15 | 15 |
| Task 5: CHASE_DB1 @30M | 25 | 25 |

**~80 GPU hours if everything moves to 30M.** Three and a half days continuous,
realistically one to two weeks of overnight queues.

**The first row is the best value on the page.** E13 currently has two points
(117k, 467k). A third at 30M directly answers the question a reviewer will ask
first — does "the topology loss advantage vanishes with capacity" survive at
real scale — and it needs no new code, only another width. Fifteen hours.

## 5. Tasks, in priority order

### Task 1 — put the core comparison on a real architecture

**This is the point of moving machines.** A 117k-parameter TinyUNet is not a
baseline any reviewer accepts, and finding 1 above is precisely the reason to
doubt ourselves at scale: we proved capacity changes conclusions.

Train `A_dice`, `B_cldice`, `H_aug`, `G_focal`, `K_focal_aug` on a 5-level
base=64 U-Net (~31M params, see §4.2), ideally with an ImageNet-pretrained
encoder. Keep every loss, augmentation, and evaluation path identical; change
only the backbone.

Do the three E13 arms (`A_dice`, `B_cldice`, `H_aug`) **first**: they complete
the capacity curve and are the highest-value 15 hours available.

Note before you start: DRIVE has **20 training images**. 30M parameters on 20
images is badly overparameterised, which is why the field pretrains. Expect
augmentation to matter *more* here, not less, which is itself the prediction
finding 1 makes.

**Before queueing 25 runs**, train `A_dice` and `H_aug` at one seed each (~1-2 h)
and check the baseline Dice lands inside the published DRIVE range. If the
baseline is not credible, the other 23 runs are wasted.

**Acceptance:** the three gaps in `exp/summarize_capacity.py` recomputed at the
new width, with the same seed gate. The readable claim is "the trend continues"
or "it stops at 30M". Either is publishable; a missing number is not.

### Task 2 — six seeds on `K_focal_aug`

Three cannot support the 44.7% headline. Cheap on a GPU.

**Acceptance:** `exp/summarize_combo.py` and the ERL comparison rerun at 6
seeds, and `stage-report/e18_gate_plus_augmentation.md` amended with a dated
revision block. Do **not** rewrite the original text; that rule is in CLAUDE.md
and it exists so the record stays auditable.

### Task 3 — compare against published topology losses, as published

We have clDice, cbDice and boundary loss as our own implementations. A paper
needs a table against the published numbers, on the same backbone: TopoLoss,
Betti matching, warping loss.

**Acceptance:** a table where our baseline sits inside the published DRIVE
range. Verify that range against a current paper; do not trust any number
quoted from memory, including the ones in this file.

### Task 4 — is the Dice loss coming from vessel width?

Open question with a concrete answer. Across all four contrast bands
`K_focal_aug` has **higher** topological precision *and* recall than the
baseline while whole-image Dice falls (see the table in
`stage-report/e18_gate_plus_augmentation.md`). That rules out the obvious
confound — K is not simply over-predicting — and leaves the hypothesis that the
centreline is more correct while the vessel **width** is less correct.

Likely mechanism, unverified: the confidence gate weights `1 + gamma*(1 -
2|p-0.5|)`, which peaks where the model hesitates, and the model hesitates at
vessel **boundaries**. So the gate concentrates effort on boundary pixels while
clDice pushes centreline correctness there.

**Acceptance:** measure mean predicted vessel width against ground truth width,
per band, per config. If the hypothesis holds, it points at a real fix (restrict
the gate to a neighbourhood of the centreline) and that fix is a better paper
contribution than the combination is.

### Task 5 — why ERL does not transfer

Two competing explanations, and the answer decides whether the headline result
generalises: HRF's ERL sits at 4% of ceiling, near the floor, where differences
on a tiny base are unstable; or it is a genuine failure to generalise.
Separating them needs a dataset between DRIVE and HRF in difficulty.
CHASE_DB1 is the field's usual fourth dataset and would serve both this and
task 3.

## 6. Setup on the new machine

### 6.1 Check what the box actually has

```
nvidia-smi
```

Read two things off it: the **driver version** and the **CUDA version** in the
top-right, and whether anyone else's job is already occupying the GPU. Note the
free VRAM. If `nvidia-smi` is not found, there is no usable NVIDIA driver and
nothing below will work — stop and sort that out first.

```
python --version
```

This project was developed on **Python 3.11**. 3.10-3.12 are all fine; avoid
3.13+ until you have checked that the torch build you want exists for it.

### 6.2 Clone and isolate

```
git clone https://github.com/ChangShengHsun/ian_mei.git
cd ian_mei
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Use a venv even if the machine "already has torch". A shared lab box usually
has a torch pinned to someone else's project, and `.venv/` is already in
`.gitignore`.

### 6.3 Install torch — the one step that has to match the machine

**Do not `pip install torch` on its own.** The default wheel on many indexes is
the CPU build, which is exactly the state this repo is currently in
(`2.13.0+cpu`) and exactly the problem you came here to fix.

Go to <https://pytorch.org/get-started/locally/>, select Linux / pip / Python /
the CUDA version at or **below** what `nvidia-smi` reported, and run the command
it gives you. It looks like this, with the `cu___` suffix being the part that
matters:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

A CUDA build newer than the driver will install cleanly and then fail at
runtime. When unsure, pick one step lower; the driver is backward compatible.

### 6.4 Everything else

```
pip install numpy scipy scikit-image opencv-python pyyaml
```

There is no `requirements.txt` on purpose: the torch line differs per machine,
and a pinned wheel committed from this laptop would pin the CPU build onto
every future machine. The five packages above have no version constraints that
have ever mattered here.

If `opencv-python` fails to import with a libGL error on a headless box:

```
pip install opencv-python-headless
```

### 6.5 Verify before writing any code

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Must print a version without `+cpu`, and `True`.** If it prints `False`, the
torch build and the driver disagree — go back to 6.3. Nothing after this point
is worth doing until this line is right.

```
python -c "import torch; print(torch.zeros(1).cuda())"
```

Confirms the GPU is actually reachable, not just detected.

### 6.6 Data

```
python exp/fetch_stare.py      # 19 MB
python exp/fetch_hrf.py        # 76 MB
git clone --depth 1 https://github.com/jmlipman/TopoMortar.git data/TopoMortar
```

DRIVE is committed to the repo and needs no fetching. The other three are
gitignored. Skip TopoMortar if you are only doing tasks 1-3; it is used by the
cross-dataset scripts, not the retinal ones.

### 6.7 Confirm the port did not break anything

Every `exp/*.py` with non-trivial logic carries a `--selftest` that asserts the
mechanism rather than the output.

```
python exp/augment.py
python exp/liot.py --selftest
python exp/test_cbdice.py
python exp/test_trained_runs.py
python exp/summarize_capacity.py --selftest
python exp/summarize_combo.py --selftest
```

Run these **twice**: once now on the untouched CPU code to establish that the
environment is sane, and again after the device change in section 3. A failure
in the first pass is an install problem; a failure only in the second is your
port.

Then one real smoke test, which is also the acceptance check for task 0:

```
python exp/train.py A_dice_s0
```

`A_dice_s0` already has a `final.pt`, so the script prints `already finished,
skipping` and exits in seconds. To actually exercise the path, train a name
that does not exist yet, or move the existing directory aside first and compare
the new `log.csv` against `exp/results/A_dice_s0/log.csv`. **Move it, do not
delete it** — deleting results is against the repo rules in `CLAUDE.md`.

## 7. Rules that carry over, and one that does not

From `CLAUDE.md`, still true on any machine:

- Analysis scripts enumerate runs with `train.trained_runs()`. **Never write a
  seed range by hand.** That bug once let a verdict confirm itself on the wrong
  seeds and open the next experiment's queue gate.
- Thresholds that must transfer between datasets go in dataset-relative units
  (multiples of median structure width), never absolute pixels.
- Never rewrite a published `stage-report/*.md` result in place. Amend with a
  dated revision block.
- Pre-register criteria: write and selftest the verdict script **before** the
  first training step of the experiment it judges.
- A claim needs the paired t **and** every seed agreeing in sign. 700 paired
  images from 2 trainings once gave p=3e-4 on a difference that flipped sign
  between those two seeds.

What changes on a Linux GPU box: the CLAUDE.md section "Environment traps that
have actually bitten" is mostly Windows-specific (cp950 console, PowerShell
UTF-16, no colon in paths, `Start-Process` detaching). Ignore those. The traps
that survive are the ones about **jobs**: budget from a measured step cost
rather than a guess, save the expensive artifact as soon as it exists and key
resume on it, and check for competing processes before launching — a shared lab
GPU has other people's jobs on it, which is a stronger version of the same
problem this laptop had.

## 8. What not to redo

These are settled and re-running them wastes GPU time that tasks 1-5 need:

- Whether augmentation beats the plain baseline on DRIVE. It does, 6 seeds.
- Whether filling small holes recovers loops on retinal data. It does not; the
  models have too few loops to begin with (E9, corrected by E9b for the
  bridging case).
- Whether pixel-level uncertainty predicts annotator disagreement in the
  dimmest band. It does not, and three different readouts all fail the same
  way (E1', E8).
- Whether LIOT beats grey input under transfer. Direction is right, seed gate
  fails (E16, E17).

Read `stage-report/README.md` section "這一系列真正學到的東西" before designing
anything new. It is nine lessons, most of them bought with a wrong result.
