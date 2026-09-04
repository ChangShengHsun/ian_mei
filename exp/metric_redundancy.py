"""If a reader already reports Dice, clDice and Betti-0, what does ERL add?

WRITTEN AND SELFTESTED 2026-09-04, BEFORE IT MEASURED ANYTHING.

WHY THIS FILE EXISTS -- IT IS THE PAPER'S EXISTENTIAL CHECK. The paper
(framing a) argues this field should adopt ERL and specifies how. The first
question any reviewer asks is the one the paper has not yet answered: this
field already has Dice, clDice and Betti-0. If ERL ranks methods the way
clDice does, the paper is proposing a synonym and has no reason to exist.

So this file is written to be able to KILL the paper, and the predictions
below say exactly what would do it. That is the only honest way to ask a
question whose answer you want to come out one way.

WHAT IS MEASURED, on checkpoint_scores.csv -- 97,400 rows already on disk,
one per (run, epoch, image), carrying dice, cldice, betti0_err and erl
together, so no re-scoring is needed and nothing can drift between metrics.

  1. RANK AGREEMENT. Each arm gets one number per metric, at each run's
     dev-chosen epoch, averaged over its seeds and the 20 test images.
     Spearman rho between ERL's ordering of the arms and each other metric's,
     plus the fraction of arm PAIRS the two metrics order differently. Rho
     alone hides how many actual decisions would change; the discordance rate
     does not.
  2. THE MATCHED-DICE SPREAD. Among runs whose Dice agrees to within a
     tolerance, how far apart can ERL be? This is the argument that Dice
     cannot stand in for ERL: if two models a reader would call equally good
     differ by 20 ERL points, the reader needs ERL. Same for clDice.
  3. ERL IS A LENGTH, and that is an argument no correlation can make. Every
     number here is also reported in pixels, because "you can trace 47 px
     before hitting an error" is a sentence clDice cannot produce at all.

PRE-REGISTERED 2026-09-04, before this file has read a single row:

  1. Spearman rho(ERL, Dice) over the arms is below 0.80. Dice is a volume
     overlap and is not expected to order methods by connectivity. If it is
     above 0.80 the paper's motivating premise -- that the field's default
     metric misses connectivity -- is much weaker than claimed.
  2. rho(ERL, clDice) > rho(ERL, Dice). clDice is the field's connectivity
     metric and should be the closer relative. THE KILL CONDITION: if
     rho(ERL, clDice) > 0.95 AND the pairwise discordance is under 10%, then
     for RANKING PURPOSES clDice is a substitute, and the paper must say so
     and rest its case on interpretability (a length, with a stated unit and
     a stated ceiling) rather than on ordering. Writing that down now so the
     conclusion cannot be quietly softened later.
  3. At Dice matched to within 0.002, the ERL range across runs exceeds 10
     percentage points of the oracle. Under 10 would mean Dice pins ERL well
     enough that a second number buys little.
  4. Betti-0 error is the LEAST like ERL of the three, by rho. It counts
     components without weighting them by length, so one lost capillary and
     one severed arteriole cost the same.

  python exp/metric_redundancy.py --selftest
  python exp/metric_redundancy.py --report

Reads results/heldout/checkpoint_scores.csv. Writes nothing.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_heldout as heldout

# betti0_err is an ERROR: lower is better. Flipped once, here, so that every
# comparison below is "higher is better" and no later line has to remember.
METRICS = (("dice", 1.0), ("cldice", 1.0), ("betti0_err", -1.0))
DICE_TOLERANCE = 0.002
CLDICE_TOLERANCE = 0.002
KILL_RHO = 0.95
KILL_DISCORDANCE = 0.10


def spearman(left: list[float], right: list[float]) -> float:
    """Rank correlation, ties averaged. scipy is available but this is four
    lines and keeps the tie convention explicit and inspectable."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while (stop + 1 < len(order)
                   and values[order[stop + 1]] == values[order[index]]):
                stop += 1
            mean_rank = (index + stop) / 2.0
            for position in range(index, stop + 1):
                out[order[position]] = mean_rank
            index = stop + 1
        return out
    a, b = ranks(left), ranks(right)
    return float(np.corrcoef(a, b)[0, 1])


def discordance(left: list[float], right: list[float]) -> float:
    """Fraction of pairs the two orderings disagree about.

    Ties in either metric are skipped rather than counted as agreement: a tie
    is an absence of an ordering, and scoring it as agreement would make a
    constant metric look perfectly concordant with everything.
    """
    disagree = total = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            if left[i] == left[j] or right[i] == right[j]:
                continue
            total += 1
            if (left[i] < left[j]) != (right[i] < right[j]):
                disagree += 1
    return disagree / total if total else float("nan")


def per_run() -> dict:
    """{run: {metric: mean over the 20 test images at its dev-chosen epoch}}.

    The epoch comes from chosen_epochs(), which reads each run's own log.csv
    and never touches the test images -- picking the epoch by a test metric
    here would make every correlation below an artefact of that choice.
    """
    epochs = heldout.chosen_epochs()
    gathered = defaultdict(lambda: defaultdict(list))
    with heldout.SCORES.open() as handle:
        for row in csv.DictReader(handle):
            if epochs.get(row["run"]) != int(row["epoch"]):
                continue
            skel = float(row["skel_px"])
            if skel <= 0:
                continue
            cell = gathered[row["run"]]
            # ERL as a fraction of the skeleton it could have traced, so that
            # images of different sizes are commensurable. It is still a
            # LENGTH underneath; report() prints the pixels too.
            cell["erl"].append(float(row["erl"]) / skel)
            cell["erl_px"].append(float(row["erl"]))
            for name, _ in METRICS:
                cell[name].append(float(row[name]))
    return {run: {name: float(np.mean(values))
                  for name, values in cell.items()}
            for run, cell in gathered.items()}


def by_arm(runs: dict) -> dict:
    """{arm: {metric: mean over its seeds}}."""
    grouped = defaultdict(lambda: defaultdict(list))
    for run, cell in runs.items():
        arm = run.rsplit("_s", 1)[0]
        for name, value in cell.items():
            grouped[arm][name].append(value)
    return {arm: {name: float(np.mean(v)) for name, v in cell.items()}
            for arm, cell in grouped.items()}


def matched_spread(runs: dict, key: str, tolerance: float) -> list:
    """[(centre, n, erl_low, erl_high)] over windows where `key` is matched.

    One window per run, centred on that run's value: every run within
    `tolerance` is a model a reader comparing on `key` alone would call
    equivalent. The widest such window is the headline.
    """
    items = sorted(runs.items(), key=lambda kv: kv[1][key])
    out = []
    for _, centre in items:
        inside = [cell["erl"] for _, cell in items
                  if abs(cell[key] - centre[key]) <= tolerance]
        if len(inside) < 2:
            continue
        out.append((centre[key], len(inside), min(inside), max(inside)))
    return out


def selftest() -> None:
    # 1. SPEARMAN AGAINST HAND CASES, including the tie convention. A rank
    #    correlation that mishandles ties would silently flatter any metric
    #    with repeated values, and betti0_err has many.
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [1, 3, 2, 4]) - 0.8) < 1e-12
    assert abs(spearman([1, 1, 2, 2], [1, 1, 2, 2]) - 1.0) < 1e-12
    print("spearman: identity 1.0, reversal -1.0, one swap 0.8, ties handled")

    # 2. DISCORDANCE MUST COUNT DECISIONS, and must not score a tie as
    #    agreement -- a constant metric agrees with nothing, it orders
    #    nothing.
    assert discordance([1, 2, 3], [1, 2, 3]) == 0.0
    assert discordance([1, 2, 3], [3, 2, 1]) == 1.0
    assert abs(discordance([1, 2, 3], [1, 3, 2]) - 1 / 3) < 1e-12
    assert np.isnan(discordance([1, 1, 1], [1, 2, 3]))
    print("discordance: 0 for identity, 1 for reversal, 1/3 for one swap, "
          "nan when one side is constant")

    # 3. THE SIGN CONVENTION. betti0_err is an error; if the flip below were
    #    dropped, its correlation with ERL would come out backwards and read
    #    as a finding. Assert the flip exists rather than trusting a comment.
    assert dict(METRICS)["betti0_err"] == -1.0
    assert dict(METRICS)["dice"] == 1.0
    print("betti0_err carries a -1 orientation; dice and cldice carry +1")

    # 4. THE MATCHED-WINDOW READER, on a case built by hand. Two runs with
    #    identical Dice and far apart in ERL must produce a wide window; one
    #    run alone must produce none.
    fake = {"a_s0": {"dice": 0.80, "erl": 0.30},
            "a_s1": {"dice": 0.801, "erl": 0.55},
            "b_s0": {"dice": 0.90, "erl": 0.40}}
    windows = matched_spread(fake, "dice", 0.002)
    assert windows, windows
    widest = max(windows, key=lambda w: w[3] - w[2])
    assert widest[1] == 2 and abs((widest[3] - widest[2]) - 0.25) < 1e-9
    assert all(w[1] >= 2 for w in windows)
    lonely = matched_spread({"b_s0": fake["b_s0"]}, "dice", 0.002)
    assert lonely == [], lonely
    print("matched window: two runs at equal Dice 0.25 apart in ERL, "
          "a lone run yields no window")

    # 5. THE REAL TABLE MUST EXIST AND BE THE PROTOCOL'S. A silent empty read
    #    would print correlations over nothing.
    runs = per_run()
    assert len(runs) > 100, len(runs)
    for cell in runs.values():
        assert set(cell) == {"erl", "erl_px", "dice", "cldice", "betti0_err"}
        assert 0.0 <= cell["erl"] <= 1.0, cell["erl"]
    print(f"read {len(runs)} runs at their dev-chosen epochs, "
          f"{len(by_arm(runs))} arms")
    print("all checks passed")


def report() -> None:
    runs = per_run()
    arms = by_arm(runs)
    print("=== does ERL tell a reader anything Dice, clDice and Betti-0 do "
          "not? ===\n")
    print(f"{len(runs)} runs, {len(arms)} arms, each at its dev-chosen epoch,")
    print("averaged over the 20 DRIVE test images. ERL is shown as a fraction")
    print("of the skeleton it could have traced; it is a LENGTH underneath,")
    print("and the pixel column is the sentence no other metric here can")
    print("produce: 'you can trace this far before hitting an error'.\n")

    import postproc_ceiling as sweep
    ledger = set(sweep.CONTROL + sweep.FRONTIER)

    def panel(label: str, names: list[str]) -> dict:
        erl = [arms[a]["erl"] for a in names]
        print(f"  {label} (n = {len(names)})")
        print(f"    {'metric':14}{'rho vs ERL':>12}"
              f"{'pairs ordered differently':>28}")
        out = {}
        for name, orientation in METRICS:
            other = [orientation * arms[a][name] for a in names]
            rho, disc = spearman(erl, other), discordance(erl, other)
            out[name] = (rho, disc)
            print(f"    {name:14}{rho:>12.3f}{disc:>27.1%}")
        print()
        return out

    print("--- 1. do the metrics ORDER the arms the same way? ---")
    print("    Two arm sets, because they answer different questions and the")
    print("    second one is partly true BY CONSTRUCTION. All 56 includes")
    print("    every config ever trained under this protocol, retired ones")
    print("    among them -- an unselected set. The 10 ledger arms are the")
    print("    paper's, and they deliberately sweep a Dice-for-ERL trade")
    print("    (`clw` weights 1 to 64, settled fact 3), so a NEGATIVE rho")
    print("    there is a demonstration of that trade, not an independent")
    print("    discovery. Quote the 56-arm number as the finding and the")
    print("    10-arm number as the mechanism.\n")
    rhos = panel("all arms", sorted(arms))
    panel("the 10 ledger arms", sorted(a for a in arms if a in ledger))

    # The open question metric_redundancy.md flags: this whole table reads
    # ERL in ONE cell of the specification (split / pixels / full). If the
    # convention flips these correlations, that is an independent result.
    # composition's `raw` rows carry erl_split, erl_bridged AND dice at the
    # SAME threshold and image, so the splitting axis -- the one worth 20
    # points -- can be answered exactly and with no new compute. clDice and
    # Betti-0 are not in that table, so this panel is against Dice only.
    print("--- 1b. does the ERL CONVENTION flip those correlations? ---")
    try:
        import composition
        rows = [r for r in composition.load("test") if r["source"] == "raw"]
    except Exception as error:                      # noqa: BLE001
        rows = []
        print(f"    composition rows unavailable ({error}); panel skipped")
    if rows:
        gathered = defaultdict(lambda: defaultdict(list))
        for row in rows:
            cell = gathered[row["config"]]
            for name in ("erl_split", "erl_bridged", "dice"):
                cell[name].append(float(row[name]))
        means = {arm: {k: float(np.mean(v)) for k, v in cell.items()}
                 for arm, cell in gathered.items()}
        shared = sorted(means)
        dice_col = [means[a]["dice"] for a in shared]
        print(f"    {len(shared)} arms, composition `raw` rows, one threshold")
        print(f"    {'ERL convention':18}{'rho vs Dice':>13}"
              f"{'pairs ordered differently':>28}")
        got = {}
        for name in ("erl_split", "erl_bridged"):
            col = [means[a][name] for a in shared]
            got[name] = spearman(col, dice_col)
            print(f"    {name:18}{got[name]:>13.3f}"
                  f"{discordance(col, dice_col):>27.1%}")
        pair = spearman([means[a]["erl_split"] for a in shared],
                        [means[a]["erl_bridged"] for a in shared])
        print(f"    the two conventions agree with EACH OTHER at rho "
              f"{pair:.3f}")
        flipped = (got["erl_split"] < 0) != (got["erl_bridged"] < 0)
        print(f"    sign against Dice: "
              f"{'FLIPS with the convention' if flipped else 'same under both'}")
    print()

    print("--- 2. at MATCHED Dice / clDice, how far apart is ERL? ---")
    print("    A window is every run within the tolerance of one run's value")
    print("    -- models a reader comparing on that metric alone would call")
    print("    equivalent. The widest window is the headline.\n")
    for key, tolerance in (("dice", DICE_TOLERANCE),
                           ("cldice", CLDICE_TOLERANCE)):
        windows = matched_spread(runs, key, tolerance)
        if not windows:
            print(f"    {key}: no window holds two runs at +-{tolerance}")
            continue
        widest = max(windows, key=lambda w: w[3] - w[2])
        spread = 100 * (widest[3] - widest[2])
        median = float(np.median([100 * (w[3] - w[2]) for w in windows]))
        print(f"    {key} matched to +-{tolerance}: widest window holds "
              f"{widest[1]} runs at {key} {widest[0]:.4f},")
        print(f"      and their ERL runs from {100 * widest[2]:.1f}% to "
              f"{100 * widest[3]:.1f}% -- a spread of {spread:.1f} points.")
        print(f"      Median window spread over all {len(windows)} windows: "
              f"{median:.1f} points.")
    print()

    print("--- 3. ERL in the unit it actually has ---")
    lengths = sorted((cell["erl_px"], run) for run, cell in runs.items())
    print(f"    worst run  {lengths[0][1]:24} {lengths[0][0]:8.0f} px")
    print(f"    best run   {lengths[-1][1]:24} {lengths[-1][0]:8.0f} px")
    print("    Neither Dice nor clDice nor a Betti number can be stated in")
    print("    pixels a reader can picture.\n")

    print("--- the pre-registered predictions ---")
    dice_rho = rhos["dice"][0]
    cldice_rho, cldice_disc = rhos["cldice"]
    betti_rho = rhos["betti0_err"][0]
    print(f"    1. rho(ERL, Dice) < 0.80:            {dice_rho:.3f}  "
          f"{'holds' if dice_rho < 0.80 else 'FALSIFIED'}")
    print(f"    2. rho(ERL, clDice) > rho(ERL, Dice): "
          f"{cldice_rho:.3f} > {dice_rho:.3f}  "
          f"{'holds' if cldice_rho > dice_rho else 'FALSIFIED'}")
    killed = cldice_rho > KILL_RHO and cldice_disc < KILL_DISCORDANCE
    print(f"       KILL CONDITION (rho > {KILL_RHO} and discordance < "
          f"{KILL_DISCORDANCE:.0%}): "
          f"{'TRIGGERED -- clDice is a ranking substitute' if killed else 'not triggered'}")
    print(f"    4. Betti-0 least like ERL:           "
          f"{betti_rho:.3f} vs dice {dice_rho:.3f}, cldice {cldice_rho:.3f}  "
          f"{'holds' if abs(betti_rho) < min(abs(dice_rho), abs(cldice_rho)) else 'FALSIFIED'}")
    print("    (prediction 3 is the matched-window section above.)")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    raise SystemExit("pass --selftest or --report")


if __name__ == "__main__":
    main()
