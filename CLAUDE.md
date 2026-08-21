# CLAUDE.md — ian-mei

Project notes for curvilinear-structure segmentation experiments.
Ivan's global rules in `~/.claude/CLAUDE.md` still apply and win on conflict;
this file only adds what is specific to this repo. Findings live in
`stage-report/`, not here.

## Direction

Primary effort goes into **improving the model**. A new evaluation metric or
protocol is acceptable as a secondary contribution, not as the main one.
Confirmed with Ivan's supervisor, 2026-08-19. Propose work in that order.

## Where things live

| Path | What | Committed? |
|---|---|---|
| `exp/*.py` | One script per experiment. The module docstring states the question it answers and the runtime. | yes |
| `exp/results/*.csv`, `*.log` | Measurements. These are the results worth keeping. | yes |
| `exp/results/**/final.pt` | Checkpoints. Regenerable. | no |
| `data/` | Fetched by `exp/fetch_*.py` or a git clone. Only DRIVE is committed. | mostly no |
| `stage-report/` | One markdown per experiment, plus `README.md` as the index and `stage_0`/`stage_1` as background. | yes |
| `_raw/`, `*.md` at root | The literature survey this work started from. | yes |

Published HTML reports are built in the session scratchpad and pushed to an
Artifact URL. They are not in the repo; redeploy by republishing the same file
path, or pass the existing URL.

New experiment → new `exp/<name>.py` + new `stage-report/<name>.md` + a row in
`stage-report/README.md`.

## Environment traps that have actually bitten

- **No `:` in a Windows path.** `topomortar:noisy` as a directory name killed a
  12-run job twice. Anything that turns a CLI argument into a path must
  sanitise it, and the sanitising must live in the function both the real code
  and the test call, not at one call site.
- **The console is cp950.** `print()` of `β₀`, `±`, `→` raises
  `UnicodeEncodeError` and kills the script. Use ASCII in printed output;
  Unicode is fine inside files.
- **PowerShell writes UTF-16 by default.** `Out-File`/`Tee-Object` need
  `-Encoding utf8`, and a log written without it must be read through
  `tr -d '\0'` from bash.
- **`git push` needs `dangerouslyDisableSandbox: true`.** The sandbox blocks
  DNS for `lfs.github.com` and the push fails with "no such host".
- **Bash heredocs have failed on large files** with mixed quoting; fall back to
  the Write tool rather than debugging the quoting.

## Long-running jobs

- Launch detached with `Start-Process`, not the Bash tool's background mode —
  session teardown kills the latter but not the former.
- **`Start-Process` does not survive system sleep.** On 2026-08-19 the machine
  slept at 16:15 and killed a detached queue mid-run; nine hours passed with
  nothing running. Two consequences, both now in the code: every training loop
  writes `ckpt.pt` periodically and resumes from it, and a queue **never waits
  on a PID** — `Wait-Process` on an already-dead PID returns instantly, so the
  next job starts against a half-finished predecessor. Gate on "is the artifact
  on disk", which is what the scripts check anyway.
- A stale log will lie to you. `cross_pseudo.log` still held a traceback from
  the previous day and read as a fresh crash; the give-away was that its line
  numbers no longer matched the file. Check the log's mtime before believing it.
- **This laptop is also Ivan's coursework machine.** Check for competing
  `python.exe` before launching; a run went from 62 to 231 minutes under
  contention. Never kill his processes. Lower our own priority instead if his
  work is interactive.
- **Serial beats parallel here.** Six cores; two training jobs at once finish
  later than one after the other.
- **Save the expensive artifact as soon as it exists**, and key resume on it.
  Checkpoints that were never saved cost a 6-hour retrain; resume keyed on the
  final CSV instead of the checkpoint threw away finished training twice.
- Budget from a measured step cost, not a guess. On a free machine this
  machine does ~0.12 s/step for BCE+Dice and ~0.19 for the topology losses.

## Code conventions

- Comments answer *why*, not *what*.
- Any non-trivial logic leaves one runnable check behind (see
  `exp/test_cbdice.py` for the shape: assert the mechanism, not the output).
- Thresholds that must transfer between datasets are expressed in
  dataset-relative units (multiples of median structure width squared), never
  absolute pixels.
- Adding a name to `train.CONFIGS` breaks every script that enumerates it until
  those runs exist; such scripts skip configs with no checkpoint.
- Scripts take run names on the command line so a partial set can be analysed
  while the rest still trains.

## Never

- Never commit or push without Ivan asking.
- Never delete a file unless Ivan asks.
- Never rewrite a published `stage-report/*.md` result in place — amend with a
  dated revision block and leave the original text readable.
