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
to pixels at the boundary. Taken from the `predicted` source where that clears
the Dice floor -- that is the field the layer will actually be driven by --
and from `oracle` otherwise, which is stated loudly rather than silently,
because it means the head is the bottleneck and D-B is being asked to fix by
joint training what it could not do post-hoc.

  python exp/gate_d1.py --selftest
  python exp/gate_d1.py          # exit 0 = build D-B, 1 = D-E only
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_direction_ceiling as ceiling
import train

GEOMETRY = train.GEOMETRY
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
        chosen = {}
        for source in ("oracle", "isotropic", "predicted"):
            chosen[source] = ceiling.pick(selection, config, source, raw_dice,
                                          "erl_split")
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

    predicted = [p for p, _ in picks if p is not None]
    if predicted:
        geometry = tuple(float(np.mean([p[i] for p in predicted]))
                         for i in (0, 1))
        why = (f"oracle beats isotropic by {gain:+.1%}; geometry from the "
               f"PREDICTED field, which is what the layer will be driven by")
    else:
        oracles = [o for _, o in picks if o is not None]
        if not oracles:
            return False, (0.0, 0.0), "no geometry available for the D-B arms"
        geometry = tuple(float(np.mean([o[i] for o in oracles]))
                         for i in (0, 1))
        why = (f"oracle beats isotropic by {gain:+.1%}, BUT no predicted "
               f"field cleared the Dice floor: geometry taken from the "
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
                rows.append(make(config, "isotropic", 0.5, 0.5, image, 0.55,
                                 0.821))
                rows.append(make(config, "oracle", 1.5, 0.25, image,
                                 oracle_erl, 0.822))
                rows.append(make(config, "predicted", 1.0, 0.5, image,
                                 0.60, 0.823 if predicted_ok else 0.700))
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
    GEOMETRY.write_text(f"{geometry[0]} {geometry[1]}\n")
    print(f"D-B builds. Wrote along={geometry[0]} across={geometry[1]} "
          f"widths to {GEOMETRY}")
    sys.exit(0)


if __name__ == "__main__":
    main()
