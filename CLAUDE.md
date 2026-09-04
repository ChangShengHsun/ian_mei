# CLAUDE.md — ian-mei

Curvilinear-structure segmentation (retinal vessels; DRIVE, STARE, HRF,
VessMAP). Ivan's global rules in `~/.claude/CLAUDE.md` still apply and win on
conflict; this file adds only what is specific to this repo.

Rewritten 2026-09-01 to carry the results, not just the conventions. Findings
still live in `stage-report/`; what is here is the short list a session must
know before it proposes anything, because most of it is a **negative** result
and negative results are what a fresh session re-discovers at four GPU-hours
a time.

## Direction

Primary effort goes into **improving the model**. A new evaluation metric or
protocol is acceptable as a secondary contribution, not as the main one.
Confirmed with Ivan's supervisor, 2026-08-19. Propose work in that order.

The current line is post-processing: an oriented dilation driven by a
predicted tangent field. Its brief is `prompt_postproc.md`.

## What is already settled — do not re-run it

Each line names the report that holds the evidence. If a proposal contradicts
one of these, say so out loud and give the measurement that overturns it.

1. **Changing the loss does not fix connectivity.** `stage-report/calibration.md`.
   Every published topology loss in the repo — clDice, cbDice, boundary,
   contrast-gated, focal/confidence-weighted — lands between −1.8% and +1.4%
   ERL against `A_dice` once each arm is read at **its own** dev-optimal
   threshold, and **none passes the gate**. Do not propose another loss
   without first saying why it escapes this.
2. **A threshold difference can manufacture a whole result.** Same report.
   `K_focal_aug` reads **+13.6%** ERL at a shared 0.5 and **−4.2%** at each
   arm's own threshold, because its probabilities peak at 0.662 and `A_dice`'s
   at 0.377. Any comparison at a shared threshold is a comparison of
   calibration. State the threshold rule in every table.
3. **`clw` is the only intervention that survived, and it is prior art.**
   Centreline-weighted BCE (`train.centreline_loss`, weight in the config name,
   swept 1–64) monotonically trades Dice for ERL and passes the gate. Skeleton
   Recall Loss (ECCV 2024) owns the idea. So `clw` is our **baseline**, never
   our contribution: beating `A_dice` proves nothing, beating `H_aug_clw16`
   is the bar.
4. **Lowering the threshold is a competitor, not a nuisance.** Settled
   2026-09-01 (`threshold_control_verdict.txt`): threshold chosen on dev, read
   on test, at matched Dice, the whole-mask oriented-dilation layer **loses on
   all ten arms under BOTH conventions** -- 4.5 to 12.2 points (A), 2.1 to 6.4
   (B) -- and the verdict is unchanged at a +0.005 and +0.010 Dice margin.
   Any operation that spends Dice to buy ERL is compared against simply moving
   the threshold, at matched Dice. That baseline is `raw` in composition.py.
6. **The gains off DRIVE are augmentation, not the loss.**
   `transfer_calibration_verdict.txt`, 2026-09-02. Against `A_dice` the arms
   read +4 to +15 points on STARE/HRF/VessMAP; against `H_aug`, which differs
   only in augmentation, `K_focal_aug` fails on all three datasets and both
   conventions. Always name the baseline: `- A_dice` bundles the loss with the
   augmentation and is not a statement about the loss.
7. **Restrict the operator before enlarging it.** Endpoints are 0.4% of
   ground-truth foreground, and whole-mask dilation spends 100% of the Dice to
   reach them. First (uncontrolled) reading: endpoint-restricted growth passes
   the gate on 10/10 arms on top of the dev-picked threshold, where whole-mask
   growth cannot afford any setting at all. The `endpoint_iso` /
   `endpoint_shuf` controls that separate "the right place" from "the right
   direction" are what decides this; until they land it is a lead, not a
   result.
5. **D-B (learned anisotropic kernels) is dead** at 6 seeds. The measured
   fault was too little reach, not too much: the kernel deviates only 5.1%
   from straight at 2 vessel widths.
8. **"The convention flips the method ranking" is Berger et al. 2024, not
   ours.** `stage-report/metric_novelty_check.md`, 2026-09-03. "Pitfalls of
   topology-aware image segmentation" (arXiv:2412.14619) published exactly
   that claim, **on DRIVE**, at Spearman ρ = −0.63, by changing pixel
   adjacency (4- vs 8-connectivity) across Dice, Betti, Betti matching, VOI,
   ARE, ARI and clDice. Never present the phenomenon as our finding. What is
   ours: it happens on **ERL**, which they do not touch, along three
   independent axes of ERL's own under-specification, plus the four axes they
   do not cover (test-set selection leak, shared vs per-arm threshold, seed
   count, post-processing against an equal-cost threshold drop).
9. **The gate is not monotone in seed count, and this is measured, not
   theoretical.** `exp/results/seed_stability.txt` + `transfer_calibration_
   verdict.stare24.txt`, 2026-09-04. `calibration.decide`'s third condition —
   every per-seed difference positive — gets strictly HARDER as seeds are
   added while the mean and t conditions get easier. On STARE at 12 → 24
   seeds two cells flipped **HOLDS → fails while both the effect size and t
   GREW**: `H_aug_clw` +10.3% t 5.51 → +10.6% t 6.93, `K_focal_aug` +10.1%
   t 5.79 → +10.3% t 7.51, both erl_bridged vs `A_dice` at the shared 0.5.
   Nothing moved fails → HOLDS. **Strengthened the same day**, once HRF and
   VessMAP also reached 24 (`stage-report/seed_stability.md` revision block):
   across all three datasets HOLDS went **6 cells → 2**, 0 the other way, and
   3 of the 4 deaths had an effect that held or GREW with a larger t. The
   condition survives n seeds with probability (1−p)^n where p is the rate a
   fresh seed dissents; measured over 60 cells (`exp/seed_survival.py`) the
   median p is **0.21**, a half-life of **3.0 seeds**, and only 2 of 60 cells
   still have zero dissenters at 24. So the third condition is **not a
   statistical test, it is a count of dissenters that is guaranteed to fail
   given enough seeds**. Never quote a verdict without its seed count.
   Corollary, learned by having a pre-registered prediction fail: `d = 0` is
   NOT `p = 0`. seed_stability's resampling curve reading 100% only says the
   seeds on disk contain no dissenter; the rule of three caps p at 3/n, so at
   n = 12 a 100% cell still dies at 24 seeds with probability 0.999. Use the
   curve to COUNT dissenters (its ladders are exactly C(n−d,k)/C(n,k)), never
   to plan how many seeds will be enough.
10. **The ERL convention spread scales with coverage; it is not a constant.**
   `stage-report/erl_spec_transfer.md`, 2026-09-04, 3 datasets × 24 seeds.
   The splitting rule's sign reversal under `diameter` reproduces perfectly
   (4/4, 4/4, 4/4 under pixels and edges; **0/4, 0/4, 0/4** under diameter).
   But DRIVE's 38.6–42.0-point spread does NOT: VessMAP gives **14.1–16.8**,
   because its coverage is 96–97% against STARE's 73–79%, and `ours =
   reference × coverage` collapses the denominator axis as coverage → 100%.
   Report the scaling rule and each dataset's coverage, never a single
   "about 40 points". Reading it backwards is the sharper claim: a paper
   reporting ERL on an easy dataset hides less disagreement than one on a
   hard dataset.

## Instruments break silently -- check the instrument

Three separate defects in 2026-09-01's own code each produced a table that
looked entirely normal. None was caught by a result looking wrong.

- **`anisotropic.ellipse` under-reached.** It tested lattice points against a
  continuous ellipse, so `along` delivered 22-47% less than asked and exactly
  0 on diagonal vessels at the geometry the sweep picked. Fixed by
  `axis_element`, which rasterises the axis; the selftest now asserts
  delivered reach at every orientation bin.
- **`abs(hash((config, seed))) % shards` lost runs.** Python randomises
  str/tuple hashes per process, so each shard computed a different partition
  and some runs were claimed by none. Four tables ran with missing seeds; in
  one the table printed "--" and read as unfinished rather than wrong. Fixed
  by `sweep.shard_filter` (stride over a sorted list), asserted to be an exact
  partition in four selftests. The `shuffled` control's RNG seed moved to
  `zlib.crc32` for the same reason.
- **An empty table is not a null result.** `curve()` keyed by threshold was
  queried with a config name, which is always false, and the report printed
  headers with no rows. Any report that can produce an empty table must
  refuse instead.

- **`sweep_score.py` normalised on the wrong split for three days.** It was
  hardwired to `train.stack_split("train")` — all 20 training images,
  including the 5 held back for selection — and never updated when the
  held-out protocol landed on 09-01. `frontier.py`, `composition.py`,
  `threshold_control.py` and `postproc_ceiling.py` all used `"fit"`; this one
  file was the exception, so nothing looked inconsistent. Measured cost at
  threshold 0.5: +0.40 / +0.06 / +0.14 ERL points on three runs, always in
  favour of the leaked stack, against the repo's +1.4-point bar. Fixed
  2026-09-04 by `stack_for(root)`. The gaps the leak ledger reports were
  nearly unaffected (both sides shared the constant); the absolute columns
  were optimistic. `chosen_epochs()` reads each run's own `log.csv`, so no
  selected epoch moved and nothing downstream needed rebuilding.

The rule: a selftest asserts the OPERATOR delivers what its parameter names,
and that a partition is a partition -- not only that the arithmetic downstream
is right.

## The protocol leak, and its three levels

The same mistake has been made three times, one level lower each time. Assume
a fourth exists and look for it.

| Level | Selected on the test images | Fixed |
|---|---|---|
| checkpoint | `best.pt` chose the epoch on test | `--protocol heldout` |
| threshold | the operating point read off the test curve | choose on `frontier_dev.csv` |
| geometry | the dilation radii picked on the reported images | `postproc_ceiling.py --dev`, 2026-09-01 |

**The rule:** every selection names the split it selected on, in the printed
output, next to the number. A script that cannot find its dev rows must
**refuse to run** — `summarize_postproc.py:293` does. Silently falling back to
the old behaviour is what produced the superseded `postproc_verdict.txt`
(00:33 on 2026-09-01; optimistic by an unknown amount, kept for the record).

**The held-out protocol:** fit on the 15 DRIVE training images, select on the
5 held out (`drive.DEV_IDS`), report on all 20 test images. One split rule,
`cross_dataset.fit_dev`, which reproduces `DEV_IDS` exactly and is asserted to.

## The gate — one definition

`calibration.decide(paired, per_seed)`. A claim holds only when the paired
t over (image, seed) exceeds 2, **every** seed agrees in sign, and there are
at least 3 seeds. Never re-implement it; a second copy will drift, and the
sign rule is the one that caught E5.

Companions that must appear in every table that uses it:
- a **matched cost** — compare at equal Dice, never at unmatched foreground;
- a **`shuffled` control** wherever a method adds foreground, because adding
  foreground raises ERL on its own; that is how the closing baseline beat the
  C1 oracle until its Dice was matched.

## ERL conventions — always report both

`pixels` (`erl.py`, a bridged gap splits a run), `edges` (√2 diagonals, the
Allen reference), `diameter` (longest path). `ours = reference × coverage`
exactly (`erl_reference.py`). 19.9 of a measured 36.6-point gap turned out to
be the splitting rule alone, so a table that hides which convention it used
cannot be acted on. Report both; if they disagree, **that disagreement is the
result** — do not quote the flattering one.

## Where things live

| Path | What | Committed? |
|---|---|---|
| `exp/*.py` | One script per experiment. The docstring states the question and the runtime. | yes |
| `exp/results/*.csv`, `*.log` | Measurements. The results worth keeping. | yes |
| `exp/results/**/*.pt` | Checkpoints. Regenerable, ~8 GPU-h for 57 runs. | no |
| `exp/results/heldout/` | Everything under the clean protocol. Pre-heldout numbers are NOT comparable to these. | csv only |
| `data/` | Fetched by `exp/fetch_*.py`. Only DRIVE is committed. | mostly no |
| `stage-report/` | One markdown per experiment; `README.md` is the index. | yes |
| `prompt*.md`, `_raw/` | The briefs and the literature survey this started from. | yes |

New experiment → new `exp/<name>.py` + `stage-report/<name>.md` + a row in
`stage-report/README.md`.

## Pre-registration

The verdict script is written **and selftested before the first training
step**, and the selftest asserts the *mechanism*, not the output. This retired
C1 in two hours and caught four bugs before they reached a table. Write the
prediction down before looking: several have failed, and the failure was the
finding (`clw` is monotone to 64, not single-peaked; two propagation steps do
not beat one).

## Environment

Linux, `rll1011`, migrated from Windows 2026-08-26. The Windows-era traps
(cp950 console, `Start-Process`, UTF-16 logs, `:` in paths) are gone.

- Interpreter is **`.venv/bin/python`**, 3.11.16 (system python is 3.8 and
  cannot parse the repo's annotations). torch 2.5.1+cu121, pinned to the
  530.30.02 driver — a newer cu build installs and then fails at runtime.
- **Two RTX 4070 Ti, 12 GB, SHARED WITH THE LAB.** Other users' jobs appear
  without warning. **Never kill, suspend or renice another user's process**
  — Ivan's absolute, 2026-08-26. Check
  `nvidia-smi --query-compute-apps=pid,used_memory`, then fit inside the free
  VRAM, gate on free memory, or run on CPU (`CUDA_VISIBLE_DEVICES=""`); the
  117k and 467k nets are usable there. Say in the report that the cards were
  shared and what was done.
- Measured step cost (`exp/bench_step.py`, 2026-08-26): 14.9 ms/step at 117k,
  23.5 at 31M, ~30.5 for the topology losses at 31M — so a 31M run is 13–17
  minutes, not the 40–80 `prompt.md` guessed. Budget from these, not from
  `prompt.md` section 4.
- `git push` needs `dangerouslyDisableSandbox: true`; the sandbox blocks DNS
  for `lfs.github.com`.
- `pandoc`, `xelatex`, `node` are absent. `opencv-python-headless`, not
  `opencv-python`.

## Long-running jobs

- Launch detached in **tmux**, one session per queue (`postproc`, `snake`,
  `thrctl`). The Bash tool's background mode dies with the session.
- **Never edit a shell script while it is executing** — bash reads it by byte
  offset and will resume in the middle of a different line. Copy it to a new
  name and edit that. Cost: one corrupted `run_task.sh`, 2026-08-28.
- **A queue never waits on a PID**, and never on another runner's "all done"
  line either. Gate on the artifact on disk, or on `gpu_queue.py <q> --pending`
  being empty. Waiting for a marker that comes *after* CPU scoring cost
  **thirteen idle GPU hours** on 2026-08-28; the runners now carry `SKIP_WAIT`
  and gate on "training pending".
- Measure a sweep's cost on one run before queueing all of it. Shard CPU work
  so each shard writes **its own** csv: two processes appending to one file
  interleave partial lines, and a half-written row is worse than a missing one
  because it parses.
- Save the expensive artifact as soon as it exists and key resume on it —
  resuming on the final CSV instead threw away finished training twice.
- A stale log will lie to you: check its mtime before believing a traceback.

## Code conventions

- Comments answer *why*, not *what*.
- Non-trivial logic leaves one runnable check behind that asserts the
  mechanism (see `exp/test_cbdice.py`, `exp/snake.py:selftest`).
- Thresholds that must transfer between datasets are in **multiples of median
  structure width**, never absolute pixels — this is what lets a DRIVE number
  be applied to HRF at six times the resolution.
- A number that changes the operator belongs **in the config name**
  (`clw32`, `snake_k08_t1`), or one name means two models writing into one
  directory.
- Analysis scripts enumerate runs from the CSV or from
  `train.trained_runs()`. **Never write a seed range by hand**: E12 trained
  seeds 3–5, two scripts scored 0–2, and the verdict file looked complete.
- Scripts take run names on the command line so a partial set can be analysed
  while the rest trains; they skip configs with no checkpoint.
- **Do not snapshot an analysis CSV before regenerating it.** These scripts
  are deterministic and only ever ADD rows (verified 2026-08-23, five
  snapshots, 0 exceptions). The one deliberate lookalike is
  `stare_cross/*/scores_rerun.csv`, byte-identical to its `scores.csv` on
  purpose (`train_stare.py:126`) — being identical is the finding.

## Never

- Never commit or push without Ivan asking.
- Never delete a file unless Ivan asks.
- Never rewrite a published `stage-report/*.md` result in place — amend with a
  dated revision block and leave the original text readable. A superseded
  verdict file is **superseded, not deleted**, and the note saying so names
  what replaced it.
- Never quote a number without the split it was selected on and the ERL
  convention it was read under.
- Never describe the Allen reference implementation from a reading of its
  source alone. It was RUN on 2026-09-05 (`stage-report/erl_reference.md`,
  `exp/erl_reference_check.py`, version 5.9.5) and two of its behaviours are
  not visible in the formula: `compute_graph_erl` measures only the connected
  group containing `nodes[0]` for each label, which on a real retinal
  prediction left 97.8% of covered skeleton unmeasured on DRIVE image 01
  (ERL 21.81 against 6394.21 after relabelling); and `run_length_from` walks
  a DFS spanning tree keeping every diagonal, so a three-pixel L-corner reads
  2.0 or 2.414 depending on the order its nodes were listed. Cite the
  measured behaviour, not the transcription.
- Never call a results-writing script on a SUBSET without first checking
  whether it appends or truncates. `sweep_score.py` takes run names on the
  command line but opens the table with `"w"`: on 2026-09-03 a loop calling it
  once per run truncated `checkpoint_scores.csv` 27 times and left 1 run of
  460. Recovered from git by `exp/run_rescore.sh`; a guard now refuses a
  subset call when the table exists. The sharded, resume-by-key files
  (`composition`, `frontier`, `erl_spec`, `postproc_ceiling`,
  `threshold_control`, `transfer_*`) open with `"a"` and are safe; the
  whole-table rebuilders open with `"w"` and must be given everything at
  once.
