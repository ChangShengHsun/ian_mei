"""Every figure and table the paper needs, regenerated from the results files.

WRITTEN AND SELFTESTED 2026-09-20.

WHY A SCRIPT AND NOT A DRAWING PROGRAM. CLAUDE.md's rule is that a number in
the paper carries the split it was selected on and the ERL convention it was
read under. A figure drawn by hand from a report loses both, and drifts the
moment the report is revised. So every mark here is computed from the CSV that
measured it, and the selftest re-derives erl_spec.txt's printed table from the
raw rows and asserts the two agree to the printed precision. If the verdict
file is ever regenerated with different numbers, this script fails instead of
quietly drawing the old ones.

WHAT IS TRANSCRIBED, AND SAID SO. Table 2 (the reference implementation) is the
one set of numbers with no machine-readable result file: exp/erl_reference_
check.py printed them to stdout on 2026-09-05 and the package it needs
(segmentation-skeleton-metrics, pointed at by REFPKG) is not installed on this
machine. Those numbers are typed in below from stage-report/erl_reference.md
and marked TRANSCRIBED in the generated .tex. Everything else is computed.

COLOUR. The categorical hues are the validated default palette from the dataviz
skill, used unchanged and in fixed slot order. Its validator is a node script
and node is not installed here, so the palette was NOT re-validated in this
session -- it is used as shipped, which is what the skill says to do. Every
series also carries a marker and a line style, so identity never rests on
colour alone: these figures are printed, and ISBI reviewers read on paper.

OUTPUTS (into paper/):
    figures/fig1_case_set.{pdf,png}     the constructed case set
    figures/fig2_conventions.{pdf,png}  twelve convention cells, DRIVE
    figures/fig3_coverage.{pdf,png}     the spread scales with coverage
    tables/tab1_case_set.tex            \\input-able fragments
    tables/tab2_reference.tex
    tables/tab3_spread.tex

Run: .venv/bin/python exp/paper_figures.py [--selftest]
About 30 s, CPU only.
"""
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "exp" / "results"
FIGURES = REPO / "paper" / "figures"
TABLES = REPO / "paper" / "tables"

# IEEE two-column geometry. spconf.sty sets a 3.39 in column on US letter.
COLUMN = 3.39
DOUBLE = 7.00

# The dataviz skill's validated light categorical palette, in slot order.
SLOT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK = "#0b0b0b"
MUTED = "#52514e"
CONTEXT = "#c9c9c4"      # the un-highlighted arms
RULE = "#dcdcd7"         # hairline grid

# The twelve cells, in the order they are read across a figure: length outer,
# then the splitting rule, then the denominator. Reference implementation =
# (split, edges, covered); this repository's own default = (split, pixels, full).
LENGTHS = ("pixels", "edges", "diameter")
SPLITS = ("split", "bridged")
DENOMS = ("full", "covered")
DISPLAY = {"stare": "STARE", "hrf": "HRF", "vessmap": "VessMAP"}
REFERENCE_CELL = ("split", "edges", "covered")
OURS_CELL = ("split", "pixels", "full")


def cells_in_order() -> list:
    return [(split, length, denom)
            for length in LENGTHS for split in SPLITS for denom in DENOMS]


def short(cell: tuple) -> str:
    split, length, denom = cell
    return f"{split[:3]}/{denom[:3]}"


def style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",          # matches the paper's body text
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "figure.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def load_spec(root: Path) -> list:
    """Every erl_spec row under `root`, de-duplicated by measurement key.

    The sharded writers open with "a" and resume by key, and vessmap carries
    both a 4-shard and a 48-shard generation of files, so the same measurement
    can appear in two files. Counting it twice would bias any mean, silently.
    """
    seen, rows = set(), []
    for path in sorted(root.glob("erl_spec*.csv")):
        for row in csv.DictReader(path.open()):
            key = (row["run"], row["seed"], row["epoch"], row["threshold"],
                   row["image"], row["split_rule"], row["length"],
                   row["denominator"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def fractions(rows: list) -> dict:
    """{config: {cell: mean of erl/oracle}} -- erl_spec.py:345-348 exactly.

    The aggregation is the mean of per-row ratios, NOT the ratio of sums. They
    differ, and matching the verdict file is the whole point of this function.
    """
    buckets = defaultdict(list)
    for row in rows:
        oracle = float(row["oracle"])
        if oracle <= 0:
            continue
        cell = (row["split_rule"], row["length"], row["denominator"])
        buckets[(row["config"], cell)].append(float(row["erl"]) / oracle)
    table = defaultdict(dict)
    for (config, cell), values in buckets.items():
        table[config][cell] = float(np.mean(values))
    return dict(table)


def coverages(rows: list) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["config"]].append(float(row["coverage"]))
    return {config: float(np.mean(values)) for config, values in buckets.items()}


def spreads(table: dict) -> dict:
    """Per arm: lowest cell, highest cell, and the gap between them, in points."""
    out = {}
    for config, cells in table.items():
        if len(cells) < len(cells_in_order()):
            continue
        low = min(cells.items(), key=lambda item: item[1])
        high = max(cells.items(), key=lambda item: item[1])
        out[config] = {"low": low[1], "high": high[1],
                       "spread": 100 * (high[1] - low[1]), "where": high[0]}
    return out


def printed_spreads(path: Path) -> dict:
    """The spread section of erl_spec.txt, parsed back out for comparison."""
    out, inside = {}, False
    for line in path.read_text().split("\n"):
        if "the spread, per arm" in line:
            inside = True
            continue
        if inside:
            parts = line.split()
            if len(parts) == 5 and parts[1].endswith("%"):
                out[parts[0]] = {"low": float(parts[1].rstrip("%")) / 100,
                                 "high": float(parts[2].rstrip("%")) / 100,
                                 "spread": float(parts[3]),
                                 "where": tuple(parts[4].split("/"))}
    return out


def selftest() -> None:
    """The figures must agree with the verdict file they illustrate."""
    rows = load_spec(RESULTS / "heldout")
    assert len(rows) >= 50000, f"only {len(rows)} spec rows -- is the CSV complete?"
    table = fractions(rows)
    mine = spreads(table)
    printed = printed_spreads(RESULTS / "erl_spec.txt")
    assert printed, "could not parse the spread section of erl_spec.txt"
    assert set(printed) <= set(mine), sorted(set(printed) - set(mine))
    for arm, want in printed.items():
        got = mine[arm]
        assert abs(100 * got["low"] - 100 * want["low"]) < 0.05, (arm, got, want)
        assert abs(100 * got["high"] - 100 * want["high"]) < 0.05, (arm, got, want)
        assert abs(got["spread"] - want["spread"]) < 0.1, (arm, got, want)
        assert got["where"] == want["where"], (arm, got["where"], want["where"])
    print(f"recomputed {len(printed)} arms from {len(rows)} raw rows: "
          f"every low, high, spread and winning cell matches erl_spec.txt")

    # The aggregation rule is the mean of ratios. If someone "simplifies" it to
    # the ratio of sums, this catches it: the two differ on real data.
    sample = [r for r in rows if r["config"] == "A_dice"
              and (r["split_rule"], r["length"], r["denominator"]) == OURS_CELL]
    mean_of_ratios = np.mean([float(r["erl"]) / float(r["oracle"]) for r in sample])
    ratio_of_sums = (sum(float(r["erl"]) for r in sample)
                     / sum(float(r["oracle"]) for r in sample))
    assert abs(mean_of_ratios - ratio_of_sums) > 1e-6, (
        "the two aggregations coincide here, so this guard proves nothing")
    print(f"aggregation guard: mean-of-ratios {mean_of_ratios:.4f} is not "
          f"ratio-of-sums {ratio_of_sums:.4f}")

    duplicated = load_spec(RESULTS / "heldout_transfer" / "vessmap")
    keys = {(r["run"], r["seed"], r["epoch"], r["threshold"], r["image"],
             r["split_rule"], r["length"], r["denominator"]) for r in duplicated}
    assert len(keys) == len(duplicated), "de-duplication did not de-duplicate"
    print(f"vessmap: {len(duplicated)} rows, all distinct after de-duplication")
    print("selftest ok")


def figure_case_set() -> None:
    """Fig 1: the same mask, cut in two places, and what it costs."""
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from skimage.morphology import skeletonize
    import blind_spot
    import drive

    items = drive.load_split("test")
    item = items[0]
    truth = item["label"]
    skel = skeletonize(truth)
    pairs = blind_spot.pair_sites(truth, skel,
                                  blind_spot.candidate_sites(truth, skel),
                                  wanted=blind_spot.CUTS_PER_IMAGE)
    masks = {tag: blind_spot.apply_cuts(truth, skel, [pair[index] for pair in pairs])
             for index, tag in ((0, "mid"), (1, "tip"))}
    graded = {tag: blind_spot.score(mask, truth, skel) for tag, mask in masks.items()}

    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.35))
    titles = {"mid": "(a) cuts in branch interiors",
              "tip": "(b) cuts near branch tips"}
    for axis, tag in zip(axes, ("mid", "tip")):
        mask = masks[tag]
        labels, count = ndimage.label(mask, structure=blind_spot.CONN8)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        orphaned = mask & (labels != int(sizes.argmax()))

        canvas = np.ones(truth.shape + (3,))
        canvas[truth] = (0.72, 0.72, 0.70)           # the vessel tree
        canvas[orphaned] = tuple(int(SLOT[0][i:i + 2], 16) / 255
                                 for i in (1, 3, 5))  # cut off from the tree
        canvas[truth & ~mask] = (0.05, 0.05, 0.05)   # the tissue removed
        axis.imshow(canvas, interpolation="nearest")
        fig_stats = graded[tag]
        axis.set_title(titles[tag], fontsize=7.5)
        # The five numbers used to be three title lines and the panels collided
        # at column width. The fundus leaves its corners empty, so they go inside.
        axis.text(0.01, 0.01,
                  f"Dice {fig_stats['dice']:.4f}\n"
                  f"clDice {fig_stats['cldice']:.4f}\n"
                  f"$\\beta_0$ err {fig_stats['betti0_err']}\n"
                  f"ERL {fig_stats['traced']:.1%}\n"
                  f"unreachable {fig_stats['unreachable']:.1%}",
                  transform=axis.transAxes, fontsize=6, va="bottom",
                  ha="left", color=INK, linespacing=1.4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    rows = list(csv.DictReader((RESULTS / "blind_spot.csv").open()))
    per_image = defaultdict(dict)
    for row in rows:
        per_image[row["image"]][row["arm"]] = float(row["traced"])
    names = sorted(per_image)
    mids = [100 * per_image[name]["mid"] for name in names]
    tips = [100 * per_image[name]["tip"] for name in names]
    axis = axes[2]
    limit = [min(mids + tips) - 4, max(mids + tips) + 4]
    axis.plot(limit, limit, color=MUTED, linewidth=0.5)
    axis.scatter(mids, tips, s=12, color=SLOT[0], linewidths=0, zorder=3)
    axis.set_xlim(limit)
    axis.set_ylim(limit)
    axis.set_xlabel("ERL, interior cuts (%)")
    axis.set_ylabel("ERL, tip cuts (%)")
    axis.set_title(f"(c) {len(names)} images, same Dice\nevery one above the line")
    axis.grid(True, linewidth=0.5)
    axis.set_axisbelow(True)
    save(figure, "fig1_case_set")


def figure_conventions(table: dict) -> None:
    """Fig 2: one set of predictions, read twelve legal ways."""
    import matplotlib.pyplot as plt

    order = cells_in_order()
    xs = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(COLUMN, 2.9))

    highlight = [("A_dice", SLOT[0], "o", "-"),
                 ("K_focal_aug_clw64", SLOT[1], "s", "--")]
    named = {name for name, _, _, _ in highlight}
    for config, cells in sorted(table.items()):
        if config in named or len(cells) < len(order):
            continue
        axis.plot(xs, [100 * cells[cell] for cell in order],
                  color=CONTEXT, linewidth=0.6, zorder=1)
    for config, colour, marker, dash in highlight:
        if config not in table:
            continue
        axis.plot(xs, [100 * table[config][cell] for cell in order],
                  color=colour, linewidth=1.2, marker=marker, markersize=3,
                  linestyle=dash, zorder=3, label=config)

    # Rotated text inside the axes collided with the series lines, so the two
    # cells a reader is most likely to have used are marked with a caret and
    # named in the caption instead.
    for cell in (REFERENCE_CELL, OURS_CELL):
        axis.annotate("", xy=(order.index(cell), 1.0),
                      xycoords=("data", "axes fraction"),
                      xytext=(0, 9), textcoords="offset points",
                      arrowprops={"arrowstyle": "-|>", "color": INK,
                                  "linewidth": 0.7, "shrinkA": 0, "shrinkB": 0})

    axis.set_xticks(xs)
    axis.set_xticklabels([short(cell) for cell in order], rotation=90)
    axis.set_xlim(-0.5, len(order) - 0.5)
    axis.set_ylabel("ERL, fraction of a perfect trace (%)")
    for index, length in enumerate(LENGTHS):
        axis.annotate(length, xy=(4 * index + 1.5, -0.22),
                      xycoords=("data", "axes fraction"),
                      ha="center", fontsize=6.5, color=INK)
        if index:
            axis.axvline(4 * index - 0.5, color=RULE, linewidth=0.6, zorder=0)
    axis.grid(True, axis="y", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                frameon=False, handlelength=2.0, columnspacing=1.2,
                borderaxespad=0.2)
    save(figure, "fig2_conventions")


def figure_coverage(drive_table: dict, drive_cover: dict) -> None:
    """Fig 3: the disagreement is bounded by what the prediction misses."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(COLUMN, 2.3))
    datasets = [("DRIVE", drive_table, drive_cover)]
    for name in ("stare", "hrf", "vessmap"):
        rows = load_spec(RESULTS / "heldout_transfer" / name)
        if not rows:
            continue
        datasets.append((DISPLAY[name], fractions(rows), coverages(rows)))

    for index, (name, table, cover) in enumerate(datasets):
        gaps = spreads(table)
        xs = [100 * cover[arm] for arm in gaps]
        ys = [gaps[arm]["spread"] for arm in gaps]
        axis.scatter(xs, ys, s=18, color=SLOT[index % len(SLOT)],
                     marker=("o", "s", "^", "D")[index % 4], linewidths=0,
                     label=f"{name} ({len(gaps)} arms)", zorder=3)

    axis.set_xlabel("covered fraction of the skeleton (%)")
    axis.set_ylabel("spread across the 12 cells (points)")
    axis.grid(True, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", frameon=False)
    save(figure, "fig3_coverage")


def save(figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"{stem}.{suffix}")
    print(f"  wrote figures/{stem}.pdf and .png")
    import matplotlib.pyplot as plt
    plt.close(figure)


def write_table(name: str, body: str, note: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    header = (f"% {name} -- GENERATED by exp/paper_figures.py. Do not edit here;\n"
              f"% edit the script, or the measurement it reads. {note}\n")
    (TABLES / name).write_text(header + body)
    print(f"  wrote tables/{name}")


def table_case_set() -> None:
    rows = list(csv.DictReader((RESULTS / "blind_spot.csv").open()))
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    images = len({row["image"] for row in rows})

    def mean(arm: str, key: str) -> float:
        return float(np.mean([float(row[key]) for row in by_arm[arm]]))

    labels = {"uncut": "ground truth", "edge_off": "erosion, off centreline",
              "edge_any": "erosion, anywhere", "tip": "cuts near tips",
              "mid": "cuts in interiors"}
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Arm & Dice & clDice & $\beta_0$ err & ERL & Unreach. \\",
             r"\midrule"]
    for arm in ("uncut", "edge_off", "edge_any", "tip", "mid"):
        lines.append(
            f"{labels[arm]} & {mean(arm, 'dice'):.4f} & {mean(arm, 'cldice'):.4f} "
            f"& {mean(arm, 'betti0_err'):.1f} & {100 * mean(arm, 'traced'):.1f}\\% "
            f"& {100 * mean(arm, 'unreachable'):.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab1_case_set.tex", "\n".join(lines) + "\n",
                f"Source: exp/results/blind_spot.csv, {images} DRIVE test images, "
                f"{by_arm['mid'][0]['cuts']} cuts per arm, "
                "ERL convention split/pixels/full, nothing selected on any split.")


def table_reference() -> None:
    # TRANSCRIBED from stage-report/erl_reference.md (the 2026-09-05 run of
    # segmentation-skeleton-metrics 5.9.5). The package is not installed on this
    # machine, so these are the only numbers in the paper not recomputed here.
    measured = [("01", 97.8, 21.81, 6394.21),
                ("02", 13.7, 7874.06, 6929.71),
                ("03", 95.3, 33.47, 1033.64)]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Image & Covered skeleton & Their ERL & After relabelling \\",
             r"      & not measured     & (px)      & (px) \\", r"\midrule"]
    for name, unwalked, theirs, fixed in measured:
        lines.append(f"{name} & {unwalked:.1f}\\% & {theirs:.2f} & {fixed:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab2_reference.tex", "\n".join(lines) + "\n",
                "TRANSCRIBED from stage-report/erl_reference.md (run of "
                "segmentation-skeleton-metrics 5.9.5 on 2026-09-05); the package "
                "is not installed here, so this table is NOT recomputed.")


def table_spread(table: dict) -> None:
    gaps = spreads(table)
    order = sorted(gaps, key=lambda arm: -gaps[arm]["spread"])
    lines = [r"\begin{tabular}{lcccl}", r"\toprule",
             r"Arm & Lowest & Highest & Spread & Highest cell \\",
             r"\midrule"]
    for arm in order:
        row = gaps[arm]
        where = "/".join(row["where"])
        lines.append(
            f"\\texttt{{{arm.replace('_', chr(92) + '_')}}} & "
            f"{100 * row['low']:.1f}\\% & {100 * row['high']:.1f}\\% & "
            f"{row['spread']:.1f} & {where} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab3_spread.tex", "\n".join(lines) + "\n",
                "Source: exp/results/heldout/erl_spec*.csv, recomputed and "
                "asserted equal to exp/results/erl_spec.txt by the selftest.")


def ranking_shift(table: dict) -> dict:
    """How far the arm ranking moves between the two most-used cells.

    ERL_thesis.md quotes rho = -0.103 and a 7-place move, but that was the
    12-seed run; this recomputes both at whatever is on disk now, so the paper
    cannot carry a superseded number.
    """
    from scipy import stats

    arms = sorted(arm for arm, cells in table.items()
                  if len(cells) >= len(cells_in_order()))
    far = ("bridged", "diameter", "covered")
    left = [table[arm][REFERENCE_CELL] for arm in arms]
    right = [table[arm][far] for arm in arms]
    rank_left = stats.rankdata(left)
    rank_right = stats.rankdata(right)
    return {"rho": float(stats.spearmanr(left, right).statistic),
            "move": int(np.max(np.abs(rank_left - rank_right))),
            "arms": len(arms)}


def write_numbers(table: dict, rows: list) -> None:
    """The measured macros the paper uses, generated so they cannot go stale."""
    gaps = spreads(table)
    shift = ranking_shift(table)
    case = list(csv.DictReader((RESULTS / "blind_spot.csv").open()))
    by_arm = defaultdict(list)
    for row in case:
        by_arm[row["arm"]].append(row)

    def mean(arm: str, key: str) -> float:
        return float(np.mean([float(r[key]) for r in by_arm[arm]]))

    values = {
        "csImages": f"{len({r['image'] for r in case})}",
        "csCuts": by_arm["mid"][0]["cuts"],
        "csDice": f"{mean('mid', 'dice'):.4f}",
        "csTracedUncut": f"{100 * mean('uncut', 'traced'):.1f}",
        "csTracedEdgeOff": f"{100 * mean('edge_off', 'traced'):.1f}",
        "csTracedEdgeAny": f"{100 * mean('edge_any', 'traced'):.1f}",
        "csTracedTip": f"{100 * mean('tip', 'traced'):.1f}",
        "csTracedMid": f"{100 * mean('mid', 'traced'):.1f}",
        "csGap": f"{100 * (mean('tip', 'traced') - mean('mid', 'traced')):.1f}",
        "csBettiTip": f"{mean('tip', 'betti0_err'):.1f}",
        "csBettiMid": f"{mean('mid', 'betti0_err'):.1f}",
        "csSplitsMid": f"{mean('mid', 'splits'):.1f}",
        "csLoopsMid": f"{mean('mid', 'loop_cuts'):.1f}",
        "csUnreachMid": f"{100 * mean('mid', 'unreachable'):.1f}",
        "csUnreachTip": f"{100 * mean('tip', 'unreachable'):.1f}",
        "specArms": f"{len(gaps)}",
        "specSeeds": f"{len({r['seed'] for r in rows})}",
        "specSpreadLo": f"{min(g['spread'] for g in gaps.values()):.1f}",
        "specSpreadHi": f"{max(g['spread'] for g in gaps.values()):.1f}",
        "specRho": f"{shift['rho']:.3f}",
        "specMove": f"{shift['move']}",
    }
    for name in ("stare", "hrf", "vessmap"):
        other = load_spec(RESULTS / "heldout_transfer" / name)
        if not other:
            continue
        gap = spreads(fractions(other))
        cover = coverages(other)
        tag = DISPLAY[name]
        values[f"tx{tag}Lo"] = f"{min(g['spread'] for g in gap.values()):.1f}"
        values[f"tx{tag}Hi"] = f"{max(g['spread'] for g in gap.values()):.1f}"
        values[f"tx{tag}Cov"] = (f"{100 * min(cover[a] for a in gap):.0f}--"
                                 f"{100 * max(cover[a] for a in gap):.0f}")
        values[f"tx{tag}Seeds"] = f"{len({r['seed'] for r in other})}"
    lines = ["% GENERATED by exp/paper_figures.py -- every macro here is a",
             "% Coverage macros are the mean over ALL twelve cells; the verdict",
             "% files print it per length convention, so they differ by ~1 point.",
             "% measurement read out of exp/results/. Do not edit: edit the",
             "% measurement. Macros NOT in this file (the transcribed reference-",
             "% implementation numbers) stay in main.tex and say so there.",
             ""]
    lines += [r"\newcommand{\%s}{%s}" % (name, value)
              for name, value in values.items()]
    (REPO / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote numbers.tex ({len(values)} macros)")
    print(f"  ranking between {'/'.join(REFERENCE_CELL)} and bridged/diameter/"
          f"covered: Spearman {shift['rho']:+.3f}, largest move {shift['move']} "
          f"of {shift['arms']} arms")


def main() -> None:
    selftest()
    if "--selftest" in sys.argv:
        return
    style()
    print("\nfigures:")
    rows = load_spec(RESULTS / "heldout")
    table = fractions(rows)
    figure_case_set()
    figure_conventions(table)
    figure_coverage(table, coverages(rows))
    print("\ntables:")
    table_case_set()
    table_reference()
    table_spread(table)
    print("\nmacros:")
    write_numbers(table, rows)
    gaps = spreads(table)
    print(f"\nDRIVE spread across {len(gaps)} arms: "
          f"{min(g['spread'] for g in gaps.values()):.1f}"
          f"--{max(g['spread'] for g in gaps.values()):.1f} points "
          f"({len({r['seed'] for r in rows})} seeds)")


if __name__ == "__main__":
    main()
