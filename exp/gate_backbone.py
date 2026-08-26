"""Pre-registered gate: is the 31M baseline credible before 23 more runs?

Written and selftested 2026-08-26, BEFORE the first w64_d5 training step.

prompt.md section 5, task 1: "train A_dice and H_aug at one seed each and
check the baseline Dice lands inside the published DRIVE range. If the baseline
is not credible, the other 23 runs are wasted." 31M parameters on DRIVE's 20
training images is badly overparameterised, so the failure this gate exists to
catch is real and cheap to catch: a net that memorised the training split and
scores 0.6 on validation, discovered after 25 runs instead of after 2.

The band below is NOT a literature number. It is anchored on this repository's
own 6-seed A_dice measurement (whole-image Dice 0.8105 on the 117k net) with
headroom either side, and prompt.md task 3 is where it gets replaced by a
range verified against a current paper. Anything quoting a published DRIVE
figure from memory is exactly what that task forbids.

  python exp/gate_backbone.py --selftest
  python exp/gate_backbone.py                 # reads the run logs on disk
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train

# Two-sided on purpose. Below the floor the backbone is broken or overfitting;
# above the ceiling something is wrong with the evaluation, not with the model
# -- DRIVE's inter-annotator agreement is itself near 0.80 Dice, so 0.88 would
# mean the validation labels leaked, not that the net is excellent.
FLOOR, CEILING = 0.78, 0.84
GATED = ("A_dice_w64_d5_s0", "H_aug_w64_d5_s0")


def final_dice(run_name: str) -> float | None:
    """Whole-image Dice at the last logged epoch, or None if not trained yet."""
    log_path = train.RESULTS / run_name / "log.csv"
    if not log_path.exists():
        return None
    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]["epoch"]) < train.EPOCHS:
        return None
    return float(rows[-1]["dice"])


def check(run_names=GATED) -> bool:
    verdict = True
    for run_name in run_names:
        dice = final_dice(run_name)
        if dice is None:
            print(f"  {run_name:<24} not finished -- cannot judge")
            verdict = False
            continue
        inside = FLOOR <= dice <= CEILING
        verdict = verdict and inside
        print(f"  {run_name:<24} dice {dice:.4f}  "
              f"{'inside' if inside else 'OUTSIDE'} [{FLOOR}, {CEILING}]")
    return verdict


def selftest() -> None:
    for name in GATED:
        config = name.rsplit("_s", 1)[0]
        assert config in train.CONFIGS, config
        assert train.base_width(config) == 64, config
        assert train.net_depth(config) == 5, config
    assert FLOOR < 0.8105 < CEILING, "the band must contain the 117k baseline"
    # A run that has not reached EPOCHS must not be judged: a mid-run log row
    # is a partially trained model and would fail the gate for the wrong
    # reason, which is how a good backbone gets thrown away.
    assert final_dice("definitely_not_a_run_s9") is None
    print(f"gate: {', '.join(GATED)} must both land in "
          f"[{FLOOR}, {CEILING}] whole-image Dice")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    print(f"backbone gate, band [{FLOOR}, {CEILING}] whole-image Dice")
    if check():
        print("PASS -- the 31M backbone is credible, open the queue")
        return
    print("FAIL -- do not queue the remaining runs; diagnose the backbone "
          "first (overfitting on 20 images is the first thing to rule out)")
    sys.exit(1)


if __name__ == "__main__":
    main()
