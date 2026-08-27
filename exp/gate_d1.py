"""Phase 1 decides what phase 2 builds, and hands over the geometry.

WRITTEN 2026-08-27, BEFORE direction_ceiling.py FINISHED. The routing below is
the pre-registered one from summarize_direction_ceiling.py, written as a
script rather than as a judgement so it cannot drift once the table is up --
the `if cmd | tee` lesson of 2026-08-26, where a CLOSED gate was printed
directly above the fifteen GPU hours it had just forbidden.

WHAT IT DECIDES.

  D-E (centreline-weighted loss) always runs. It carries no direction and is
  the competitor that could make the whole line unnecessary; its result is
  worth having whichever way the ceiling falls.

  D-B (the propagation layer) runs only if knowing the axis is worth at least
  DEAD points over isotropic dilation. Below that the mechanism is wrong, and
  training an architecture built on it would be measuring a foregone
  conclusion for four GPU hours.

WHAT IT HANDS OVER. The along/across radii, in MULTIPLES OF THE MEDIAN VESSEL
WIDTH, into exp/results/d1_geometry.txt. train.propagation_geometry() converts
to pixels at the boundary. Taken from the `predicted` source where it has
data -- that is the field the layer will actually be driven by -- and from
`oracle` otherwise, which is stated loudly rather than silently.

CORRECTED 2026-08-27, after its first run skipped D-B on an empty table. The
routing and thresholds are unchanged. What changed is that both come from
summarize_direction_ceiling's MATCHED-BUDGET comparison instead of its
absolute Dice floor, because no dilation can be Dice-free and the floor
therefore returned nothing for every source -- which this script reported as
"no arm produced a usable oracle setting" and treated as a NO. A gate that
cannot evaluate must not answer; the fix is in the evaluator.

  python exp/gate_d1.py --selftest
  python exp/gate_d1.py          # exit 0 = build D-B, 1 = D-E only
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_direction_ceiling as ceiling
# The arms D-B will be built on; their geometry is the geometry that matters.
BUILD_ON = ("A_dice_dir", "H_aug_dir")


def decide(rows) -> tuple[bool, tuple, str]:
    """(build D-B, (along, across) in widths, why)."""
    selection = ceiling.half(rows, True)
    report = ceiling.half(rows, False)
    gains, picks = [], []
    for config in sorted({r["config"] for r in rows}):
        raw = ceiling.by_setting(report, config, "raw",
                                 "erl_split").get((0.0, 0.0))
        if raw is None:
            continue
        raw_erl, raw_dice = raw
        # The tightest budget, which is the least favourable to the
        # mechanism. A verdict that needs the loose one is not a verdict.
        budget = ceiling.BUDGETS[0]
        chosen = {}
        for source in ("oracle", "isotropic", "predicted"):
            chosen[source] = ceiling.pick(selection, config, source, raw_dice,
                                          "erl_split", budget)
        if chosen["oracle"] is None:
            continue
        oracle = ceiling.by_setting(report, config, "oracle",
                                    "erl_split")[chosen["oracle"]][0]
        base = raw_erl
        if chosen["isotropic"] is not None:
            base = ceiling.by_setting(report, config, "isotropic",
                                      "erl_split")[chosen["isotropic"]][0]
        gains.append(oracle - base)
        if config in BUILD_ON:
            picks.append((chosen["predicted"], chosen["oracle"]))
    if not gains:
        return False, (0.0, 0.0), "no arm produced a usable oracle setting"

    gain = float(np.mean(gains))
    if gain < ceiling.DEAD:
        return False, (0.0, 0.0), (
            f"oracle beats isotropic by only {gain:+.1%}, under the "
            f"pre-registered {ceiling.DEAD:.0%}: the mechanism is wrong")

    # A pick of (0, 0) is the do-nothing setting: under a matched budget every
    # source can always afford it, so "predicted returned something" is no
    # longer evidence that the predicted field can drive the layer. Handing it
    # over would build a layer with a one-pixel kernel -- the same silent
    # no-op that cost the first _prop smoke test its gradient.
    usable = lambda pick: pick is not None and pick != (0.0, 0.0)
    predicted = [p for p, _ in picks if usable(p)]
    if predicted:
        geometry = tuple(float(np.mean([p[i] for p in predicted]))
                         for i in (0, 1))
        why = (f"oracle beats isotropic by {gain:+.1%}; geometry from the "
               f"PREDICTED field, which is what the layer will be driven by")
    else:
        oracles = [o for _, o in picks if usable(o)]
        if not oracles:
            return False, (0.0, 0.0), "no geometry available for the D-B arms"
        geometry = tuple(float(np.mean([o[i] for o in oracles]))
                         for i in (0, 1))
        why = (f"oracle beats isotropic by {gain:+.1%}, BUT no predicted "
               f"field could afford to do anything: geometry taken from the "
               f"oracle, and D-B is being asked to fix by joint training "
               f"what the head could not do post-hoc")
    return True, geometry, why


def selftest() -> None:
    def make(config, source, along, across, image, erl, dice):
        return {"config": config, "source": source, "along": along,
                "across": across, "image": image, "erl_split": erl,
                "erl_bridged": erl, "dice": dice, "fg": 1000}

    def build(oracle_erl, predicted_ok):
        rows = []
        for index in range(1, 21):
            image = f"{index:02d}"
            for config in BUILD_ON:
                rows.append(make(config, "raw", 0.0, 0.0, image, 0.50, 0.820))
                for source in ("isotropic", "oracle", "predicted"):
                    rows.append(make(config, source, 0.0, 0.0, image, 0.50,
                                     0.820))
                rows.append(make(config, "isotropic", 0.5, 0.5, image, 0.55,
                                 0.821))
                rows.append(make(config, "oracle", 1.5, 0.25, image,
                                 oracle_erl, 0.822))
                rows.append(make(config, "predicted", 1.0, 0.5, image,
                                 0.60, 0.823 if predicted_ok else 0.700))
                rows.append(make(config, "oracle", 0.0, 0.0, image, 0.50,
                                 0.820))
        return rows

    build_it, geometry, why = decide(build(0.70, True))
    assert build_it and geometry == (1.0, 0.5), (build_it, geometry)
    print(f"a {0.70 - 0.55:+.0%} oracle gain builds D-B, geometry {geometry} "
          f"from the predicted field")

    build_it, geometry, why = decide(build(0.70, False))
    assert build_it and geometry == (1.5, 0.25), (build_it, geometry)
    assert "could not do post-hoc" in why, why
    print(f"  a predicted field that fails the Dice floor falls to the "
          f"oracle's {geometry} AND says so")

    build_it, _, why = decide(build(0.56, True))
    assert not build_it, why
    print(f"  a {0.56 - 0.55:+.0%} gain does not: {why}")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not ceiling.SCORES.exists():
        print(f"CLOSED: {ceiling.SCORES} does not exist; phase 1 did not run")
        sys.exit(1)
    build_it, geometry, why = decide(ceiling.load())
    print(why)
    if not build_it:
        print("D-B is NOT built. D-E (centreline weighting, no direction) "
              "still runs -- it does not depend on this mechanism.")
        sys.exit(1)
    print(f"D-B builds. The post-hoc sweep liked along={geometry[0]} "
          f"across={geometry[1]} widths at the tightest budget.")
    print("That is REPORTED, not handed over. Handing over one reach was the "
          "2026-08-27 error: the tightest budget's geometry built a "
          "three-pixel kernel, and the Dice constraint had then been applied "
          "twice -- once here and again by the training loss, which is what "
          "the layer's gate is for. The reach is swept in the config NAME "
          "instead; see train.PROPAGATION_REACHES.")
    sys.exit(0)


if __name__ == "__main__":
    main()
