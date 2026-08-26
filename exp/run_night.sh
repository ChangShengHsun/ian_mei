#!/usr/bin/env bash
# The 2026-08-27 night queue: A1/A2/A3, C1.0 and D1.
#
# THE GPUs DO ONE THING: train D1's twelve arms. Everything else here is
# inference over checkpoints that already exist, and a whole-image pass at
# 117k costs 121 ms on four CPU threads (measured 2026-08-27) against ~20 ms
# on a card. This box has 20 cores and two 4070 Tis that other people also
# use, so the analyses run on the CPU, in parallel with the training, and
# never compete with it or with anyone else for VRAM.
#
# Every stage gates on an ARTIFACT on disk, never on a PID, and a stage that
# fails leaves the ones after it to run: a night that loses one stage must
# not lose the other four. Nothing here kills anything -- this is a shared
# lab machine, and other people's jobs are waited for, never touched.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=exp/results/night.log
SWEEP=exp/results/selection_sweep
# Four of twenty cores per analysis; three analyses can overlap and the
# trainer's own dataloading still has room.
export OMP_NUM_THREADS=4

say () { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }
cpu () { CUDA_VISIBLE_DEVICES="" "$PY" "$@"; }
step () {   # step <name> <output-file> <script> [args...]
    local name="$1" out="$2"; shift 2
    if [ -f "$out" ]; then say "$name already done ($out)"; return 0; fi
    say "$name"
    if cpu "$@" >> "$LOG" 2>&1; then say "$name OK"
    else say "$name FAILED -- continuing"; fi
}

say "night queue starting"
df -h / | tail -1 | tee -a "$LOG"

# ------------------------------------------------ 1. the CPU work, at once
# These depend on NOTHING the GPUs are doing: every checkpoint they read is
# already on disk. They are started first, in the background, because the
# cards on this box belong to everyone -- on the night this was written both
# were held by another user's jobs, and an earlier version of this script had
# the CPU analyses gated behind the GPU queue, so a busy card would have
# meant a night with nothing done at all. Never make work wait on a resource
# it does not use.
(
    # C1.0 leads: it is the one result that can retire weeks of planned work
    # before a line of it is written.
    step "C1.0 ceiling" "$SWEEP/link_ceiling.csv" exp/link_ceiling.py
    cpu exp/summarize_ceiling.py > exp/results/ceiling_summary.txt 2>&1
    cat exp/results/ceiling_summary.txt | tee -a "$LOG"

    step "A1/A2/A3 scoring" "$SWEEP/variant_scores.csv" exp/score_variants.py
    cpu exp/summarize_variants.py > exp/results/variants_summary.txt 2>&1
    cat exp/results/variants_summary.txt | tee -a "$LOG"
) & CPUWORK=$!

# ------------------------------------------------------------- 2. the cards
# Task C's last runs first -- they were queued before this script existed and
# finish the third batch of six seeds e13b section 3 needs. Wait for the
# QUEUE to empty rather than for a process, so an interrupted task C is
# simply resumed by whoever restarts it.
say "waiting for task C to drain"
while [ "$("$PY" exp/gpu_queue.py taskc --pending | wc -l)" -gt 0 ]; do
    sleep 300
done
say "task C drained"

# ------------------------------------------------------------ 3. D1 on GPU
# Twelve 117k runs, ~10 min each, into the selection sweep's results root
# with every validated epoch kept, so rule (iv) picks a checkpoint for the
# _dir arms exactly as it does for the arms they are compared against.
# run_task.sh waits for free VRAM before each run; if a card stays busy all
# night with someone else's job, this simply takes longer. It never kills
# anything.
say "D1: training the tangent-direction arms on both cards"
bash exp/run_task.sh d1 0 0 2 "$SWEEP" 1 & D1A=$!
bash exp/run_task.sh d1 1 1 2 "$SWEEP" 1 & D1B=$!
wait "$D1A"; wait "$D1B"

# --------------------------------------------------------------- 4. D1 done
DONE=$(ls -d "$SWEEP"/*_dir_s*/final.pt 2>/dev/null | wc -l)
say "D1 training finished: $DONE of 12 runs have final.pt"

# Rescore every checkpoint in the sweep, now including the _dir arms.
# sweep_score.py OVERWRITES checkpoint_scores.csv rather than appending, so
# it is called with NO arguments: an explicit subset would silently shrink
# the file to that subset, which cost 54 runs' rows on 2026-08-26. The copy
# is kept so a failed rescore does not leave the morning with neither.
say "waiting for the CPU analyses before the rescore overwrites their input"
wait "$CPUWORK"
say "CPU analyses finished"

if [ "$DONE" -gt 0 ]; then
    say "rescoring the sweep (all arms, including _dir)"
    cp "$SWEEP/checkpoint_scores.csv" "$SWEEP/checkpoint_scores.pre_d1.csv"
    if cpu exp/sweep_score.py >> "$LOG" 2>&1; then
        say "rescore OK"
        step "D1.a field quality" "$SWEEP/direction_quality.csv" \
            exp/score_direction.py
        cpu exp/summarize_direction.py \
            > exp/results/direction_summary.txt 2>&1
        cat exp/results/direction_summary.txt | tee -a "$LOG"
        # The selection table now covers six arms rather than four; rerun it
        # so every number in the morning comes from one CSV.
        cpu exp/summarize_selection.py \
            > exp/results/selection_summary.txt 2>&1
    else
        say "rescore FAILED -- restoring the pre-D1 scores"
        cp "$SWEEP/checkpoint_scores.pre_d1.csv" "$SWEEP/checkpoint_scores.csv"
    fi
else
    say "no _dir run finished; skipping the rescore and D1's verdict"
fi

say "night queue done"
df -h / | tail -1 | tee -a "$LOG"
