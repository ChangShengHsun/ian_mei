"""Phase 3's answer: is the geometry a vessel-width multiple or a DRIVE pixel?

One question, and it is checkable rather than arguable: does every dataset
choose the SAME along/across multiple, when each is read in its own median
vessel width? The widths span 2.83 px on DRIVE to 5.66 px on VessMAP, which
is not a retina at all, so a setting that survives that is a setting in
vessel widths and a setting that does not was a pixel count.

Chosen on the selection half by parity of the image index, as everywhere else
in this repo, under the same Dice floor of zero.

  python exp/summarize_transfer.py --selftest
  python exp/summarize_transfer.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_selection as selection

SCORES = selection.SWEEP / "transfer_ceiling.csv"


def load(path: Path = SCORES) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ("along", "across", "erl", "dice", "width"):
            row[key] = float(row[key])
    return rows


def is_selection(name: str) -> bool:
    """Half the images of any dataset, deterministically, by name.

    DRIVE's ids are numeric and select by parity; STARE's are `im0001`,
    VessMAP's are numeric strings. Taking the digits and using their parity
    covers all three with one rule, and any deterministic half would do -- the
    requirement is only that it was not chosen from the numbers.
    """
    digits = "".join(c for c in name if c.isdigit())
    return bool(digits) and int(digits) % 2 == 1


def best(rows, dataset: str, source: str, floor: float):
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] == dataset and row["source"] == source:
            grouped[(row["along"], row["across"])].append(row)
    allowed = {key: (float(np.mean([r["erl"] for r in these])),
                     float(np.mean([r["dice"] for r in these])))
               for key, these in grouped.items()}
    allowed = {k: v for k, v in allowed.items() if v[1] >= floor}
    if not allowed:
        return None
    return max(allowed, key=lambda k: allowed[k][0])


def selftest() -> None:
    for name, want in (("01", True), ("02", False), ("im0003", True),
                       ("10084", False), ("13_g", True)):
        assert is_selection(name) == want, (name, is_selection(name))
    print("the split works on DRIVE's '01', STARE's 'im0003', VessMAP's "
          "'10084' and HRF's '13_g'")

    rows = []
    for index in range(1, 21):
        for dataset, width in (("DRIVE", 2.83), ("VessMAP", 5.66)):
            rows.append({"dataset": dataset, "width": width, "source": "raw",
                         "along": 0.0, "across": 0.0, "image": f"{index:02d}",
                         "erl": 0.40, "dice": 0.80})
            for along, value in ((0.5, 0.45), (1.0, 0.55), (2.0, 0.50)):
                rows.append({"dataset": dataset, "width": width,
                             "source": "oracle", "along": along,
                             "across": 0.25, "image": f"{index:02d}",
                             "erl": value, "dice": 0.81})
    picks = {d: best([r for r in rows if is_selection(r["image"])], d,
                     "oracle", 0.80) for d in ("DRIVE", "VessMAP")}
    assert picks["DRIVE"] == picks["VessMAP"] == (1.0, 0.25), picks
    print(f"two datasets 2x apart in vessel width choosing {picks['DRIVE']} "
          f"is what 'the unit transfers' looks like")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if not SCORES.exists():
        raise SystemExit(f"{SCORES} not built -- run exp/transfer_ceiling.py")
    rows = load()
    chosen_rows = [r for r in rows if is_selection(r["image"])]
    report_rows = [r for r in rows if not is_selection(r["image"])]
    print("=== phase 3: does the geometry transfer? ===\n")
    header = (f"  {'dataset':<10}{'width':>7}{'raw':>8}{'oracle':>9}"
              f"{'isotropic':>11}{'gain':>8}   chosen (along, across)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    picks = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        raw = best(report_rows, dataset, "raw", -1.0)
        raw_erl = float(np.mean([r["erl"] for r in report_rows
                                 if r["dataset"] == dataset
                                 and r["source"] == "raw"]))
        floor = float(np.mean([r["dice"] for r in report_rows
                               if r["dataset"] == dataset
                               and r["source"] == "raw"]))
        width = next(r["width"] for r in rows if r["dataset"] == dataset)
        line = f"  {dataset:<10}{width:6.2f}p{raw_erl:7.1%}"
        values = {}
        for source in ("oracle", "isotropic"):
            pick = best(chosen_rows, dataset, source, floor)
            if pick is None:
                line += f"{'--':>{9 if source == 'oracle' else 11}}"
                values[source] = None
                continue
            got = float(np.mean([r["erl"] for r in report_rows
                                 if r["dataset"] == dataset
                                 and r["source"] == source
                                 and (r["along"], r["across"]) == pick]))
            values[source] = got
            line += f"{got:{8 if source == 'oracle' else 10}.1%}*"
            if source == "oracle":
                picks[dataset] = pick
        if values["oracle"] is not None:
            gain = values["oracle"] - (values["isotropic"] or raw_erl)
            line += f"{gain:+8.1%}   {picks[dataset]}"
        print(line)
    print("  * best setting under the dataset's own raw Dice, chosen on its "
          "selection half.")
    print()
    if len(set(picks.values())) == 1 and picks:
        print(f"EVERY dataset chose {next(iter(picks.values()))} widths. The "
              f"unit transfers: the same multiple is the same operation at "
              f"2.83 px and at 5.66 px.")
    else:
        print("The datasets did NOT agree on a setting:")
        for dataset, pick in picks.items():
            print(f"  {dataset:<10}{pick}")
        print("Read carefully before calling this a failure -- disagreement")
        print("between a retina and a brain micrograph may be a real")
        print("difference in vessel geometry rather than a broken unit. What")
        print("it does rule out is shipping one setting as universal.")


if __name__ == "__main__":
    main()
