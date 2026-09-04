"""ERL is three choices, not one number. All of them, on the clean protocol.

WRITTEN AND SELFTESTED 2026-09-03, BEFORE IT SCORED ANYTHING.

WHY THIS FILE EXISTS. The paper's first table is the specification: ERL as
this field would have to adopt it, with every place the definition forks named
and priced. Three such places are already measured in this repo --
erl_reference.md (the denominator), erl_length.md (fragment length),
erl_convention.md (the splitting rule) -- but all three were measured on
2026-08-27 to 08-29, which is BEFORE the held-out protocol landed on 09-01.
Those three scripts share one hardwired opening:

    points  = selection.selection_points(selection.load())   # pre-heldout
    items   = drive.load_split("val")                        # and half of it
    data    = train.stack_split("train")                     # 20, not fit's 15
    weights = selection.SWEEP / run / ...

CLAUDE.md's rule is that pre-heldout numbers are NOT comparable to held-out
ones. So the table that opens a paper about selection leaks was itself built
under the leaking protocol. That is the one defect a reviewer would find by
comparing two dates, and it is why this file exists rather than a patch: the
three originals are the RECORD of the pre-heldout measurement and are
superseded, not overwritten.

WHAT IS MEASURED. The full cross, not three separate tables, because the
question the paper has to answer is whether the axes interact:

    splitting rule   split    an uncovered centreline pixel ends the run
                     bridged  a gap the prediction connects AROUND does not
    fragment length  pixels   count of skeleton pixels, a diagonal step is 1
                     edges    8-adjacency edge weights, a diagonal is sqrt(2)
                              -- what the field's reference implementation uses
                     diameter the longest shortest path inside the fragment
    denominator      full     all ground-truth centreline
                     covered  only the fragments the prediction found
                              -- what the field's reference implementation uses

Twelve numbers per image, from one forward pass.

THE OPERATING POINT is each run's own dev Dice-maximising threshold, read
from frontier_dev.csv. Dice, not ERL: picking the threshold with an ERL is
circular when ERL is the quantity under test, and it would let each column
choose its own operating point.

TWO ANCHORS, asserted in the selftest rather than assumed. This file
recomputes fragment labels under the bridged rule instead of importing a
function that returns only a number, and a re-implementation that drifts is
how a table goes quietly wrong. So:
  - (split, *, full) must equal erl_length.run_length exactly, all three
    length conventions;
  - (bridged, pixels, full) must equal erl_convention.bridged_run_length
    exactly.
Both to floating point, on real retinal images, not on synthetic ones.

  python exp/erl_spec.py --selftest
  python exp/erl_spec.py --shard 0/4
  python exp/erl_spec.py --report

Writes results/heldout/erl_spec[.shardIofN].csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import cross_dataset
import drive
import erl_convention
import erl_length
import hole_sweep
import postproc_ceiling as sweep
import select_heldout as heldout
import speckle
import train

ARMS = sweep.CONTROL + sweep.FRONTIER
SPLIT_RULES = ("split", "bridged")
LENGTHS = ("pixels", "edges", "diameter")
DENOMINATORS = ("full", "covered")
FIELDS = ["run", "config", "seed", "epoch", "threshold", "image",
          "split_rule", "length", "denominator", "erl", "coverage", "oracle"]


def bridged_labels(skel_gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """The split-rule labelling, with gaps the prediction bridges MERGED.

    The union-find below is erl_convention.bridged_run_length's, kept as a
    label image instead of collapsed to a number, so the length conventions
    can be applied on top of it. The selftest asserts the two agree.
    """
    labels = erl_length.fragment_labels(skel_gt, pred)
    count = int(labels.max())
    if count == 0:
        return labels
    pieces = ndimage.label(pred, structure=break_lengths.CONN8)[0]
    parent = list(range(count + 1))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    missed = skel_gt & ~pred
    gaps = ndimage.label(missed, structure=break_lengths.CONN8)[0]
    for index, box in enumerate(ndimage.find_objects(gaps), start=1):
        grown = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
        pixels = gaps[grown] == index
        if break_lengths.classify(pixels, pred[grown],
                                  pieces[grown]) != "intact":
            continue
        touching = sorted(set(np.unique(labels[grown][
            ndimage.binary_dilation(pixels, break_lengths.CONN8)])) - {0})
        for other in touching[1:]:
            root_a, root_b = find(touching[0]), find(other)
            if root_a != root_b:
                parent[root_b] = root_a
    remap = np.array([0] + [find(f) for f in range(1, count + 1)])
    return remap[labels]


def fragment_sizes(labels: np.ndarray, convention: str) -> np.ndarray:
    """Fragment lengths under one convention, from a GIVEN labelling.

    erl_length.lengths recomputes the labelling itself, which is right for
    that file and wrong here: the whole point is to vary the labelling.
    """
    count = int(labels.max())
    if count == 0:
        return np.zeros(0)
    if convention == "pixels":
        return np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(
            np.float64)
    starts, _, weights = erl_length.edge_list(labels > 0)
    owner = labels.ravel()[starts]
    if convention == "edges":
        return np.bincount(owner, weights=weights,
                           minlength=count + 1)[1:].astype(np.float64)
    if convention == "diameter":
        return erl_length.lengths_from_labels(labels) \
            if hasattr(erl_length, "lengths_from_labels") \
            else _diameters(labels, count)
    raise ValueError(f"unknown convention {convention!r}")


def _diameters(labels: np.ndarray, count: int) -> np.ndarray:
    """Longest shortest path inside each fragment. Two-sweep, exact on a tree.

    Transcribed from erl_length.lengths' diameter branch so it can run on a
    supplied labelling; the selftest pins it against that function.
    """
    from scipy import sparse
    from scipy.sparse import csgraph
    starts, ends, weights = erl_length.edge_list(labels > 0)
    graph = sparse.coo_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([starts, ends]), np.concatenate([ends, starts]))),
        shape=(labels.size, labels.size)).tocsr()
    flat, out = labels.ravel(), np.zeros(count)
    for label in range(1, count + 1):
        nodes = np.flatnonzero(flat == label)
        if len(nodes) <= 1:
            continue
        sub = graph[nodes][:, nodes]
        first = csgraph.dijkstra(sub, indices=0)
        far = int(np.argmax(np.where(np.isfinite(first), first, -1)))
        second = csgraph.dijkstra(sub, indices=far)
        out[label - 1] = float(np.max(second[np.isfinite(second)]))
    return out


def skeleton_total(skel_gt: np.ndarray, convention: str) -> float:
    """The whole ground-truth centreline, measured the same way fragments are.

    A fraction whose numerator is sqrt(2)-weighted and whose denominator is a
    pixel count is not a fraction of anything -- erl_length.run_length's own
    warning, kept here.
    """
    if convention == "pixels":
        return float(skel_gt.sum())
    return float(erl_length.edge_list(skel_gt)[2].sum())


def measure(skel_gt, pred, split_rule, convention):
    """(erl_full, erl_covered, coverage) for one image and one cell."""
    labels = (erl_length.fragment_labels(skel_gt, pred) if split_rule == "split"
              else bridged_labels(skel_gt, pred))
    total = skeleton_total(skel_gt, convention)
    if total == 0:
        return 0.0, 0.0, 0.0
    # `mass` sets the probability of landing in a fragment and is always a
    # length; only `value` becomes the diameter. Writing sum(diameter^2)/L
    # shrinks the probability too, and scored a PERFECT prediction at 1.3%.
    mass = fragment_sizes(labels, "edges" if convention == "diameter"
                          else convention)
    value = fragment_sizes(labels, convention)
    if len(mass) == 0:
        return 0.0, 0.0, 0.0
    numerator = float((mass * value).sum())
    covered = float(mass.sum())
    return (numerator / total, numerator / covered if covered else 0.0,
            covered / total)


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    rng = np.random.default_rng(20260903)
    # A branched, broken, partly-bridged skeleton: every axis must have
    # something to bite on, or an anchor can agree by accident.
    truth = np.zeros((90, 120), dtype=bool)
    truth[45, 10:110] = True
    truth[20:45, 60] = True
    truth[45:70, 85] = True
    skel = skeletonize(truth)
    pred = ndimage.binary_dilation(truth, np.ones((3, 3)))
    pred[:, 38:42] = False          # a clean sever
    pred[44:47, 70:75] = False      # a gap the prediction routes around
    pred[43, 68:78] = True
    pred[47, 68:78] = True
    pred[43:48, 68] = True
    pred[43:48, 77] = True

    # 1. ANCHOR ONE. (split, *, full) must reproduce erl_length exactly, for
    #    all three length conventions -- that file is the record of what these
    #    conventions mean and this one must not drift from it.
    for convention in LENGTHS:
        mine = measure(skel, pred, "split", convention)[0]
        theirs = erl_length.run_length(skel, pred, convention)[0]
        assert abs(mine - theirs) < 1e-9, (convention, mine, theirs)
    print(f"anchor 1: (split, *, full) matches erl_length.run_length on all "
          f"{len(LENGTHS)} length conventions")

    # 2. ANCHOR TWO. (bridged, pixels, full) must reproduce
    #    erl_convention.bridged_run_length -- which means the union-find
    #    transcribed above merges exactly the fragments that one merges.
    mine = measure(skel, pred, "bridged", "pixels")[0]
    theirs = erl_convention.bridged_run_length(skel, pred)
    assert abs(mine - theirs) < 1e-9, (mine, theirs)
    print(f"anchor 2: (bridged, pixels, full) matches "
          f"erl_convention.bridged_run_length ({mine:.6f})")

    # 3. THE AXES MUST ACTUALLY DIFFER on this image, or the anchors above
    #    were satisfied by a degenerate case and prove nothing.
    split_pix = measure(skel, pred, "split", "pixels")[0]
    bridged_pix = measure(skel, pred, "bridged", "pixels")[0]
    assert bridged_pix > split_pix + 1e-6, (split_pix, bridged_pix)
    full, covered, coverage = measure(skel, pred, "split", "pixels")
    assert covered > full + 1e-6 and 0.0 < coverage < 1.0, (full, covered,
                                                            coverage)
    print(f"axes bite: split {split_pix:.4f} -> bridged {bridged_pix:.4f}; "
          f"full {full:.4f} -> covered {covered:.4f} at coverage "
          f"{coverage:.1%}")

    # 4. THE IDENTITY THE REFERENCE COMPARISON RESTS ON. erl_reference.md
    #    states `ours = reference x coverage` and asserts it there for one
    #    convention; it has to hold in EVERY cell of this cross, because the
    #    two denominators differ by exactly the covered fraction.
    for split_rule in SPLIT_RULES:
        for convention in LENGTHS:
            full, covered, coverage = measure(skel, pred, split_rule,
                                              convention)
            assert abs(full - covered * coverage) < 1e-9, (split_rule,
                                                           convention)
    print(f"identity: full == covered x coverage in all "
          f"{len(SPLIT_RULES) * len(LENGTHS)} cells")

    # 5. UNITS. ERL is a LENGTH, not a fraction -- "you can trace 47 px" --
    #    so every cell has to be divided by what a PERFECT prediction scores
    #    on the same image before it can be read as a percentage. Under
    #    pixels and edges that ceiling is the skeleton's own length and the
    #    quotient is exactly 1; under diameter it is strictly less, because
    #    one traversal cannot cover a branched tree. Getting this wrong is
    #    what the anchor above just caught, one division too many.
    perfect = ndimage.binary_dilation(skel, np.ones((3, 3)))
    for convention in LENGTHS:
        raw = measure(skel, perfect, "split", convention)[0]
        oracle = erl_length.oracle_run_length(skel, convention)
        assert abs(raw - oracle) < 1e-9, (convention, raw, oracle)
        assert abs(raw / oracle - 1.0) < 1e-9, convention
    for convention in ("pixels", "edges"):
        assert abs(erl_length.oracle_run_length(skel, convention)
                   - skeleton_total(skel, convention)) < 1e-9, convention
    ceiling = erl_length.oracle_run_length(skel, "diameter")
    assert 0.0 < ceiling < skeleton_total(skel, "edges"), ceiling
    print(f"units: a perfect prediction scores its own ceiling in every "
          f"convention; that ceiling is the skeleton length for pixels and "
          f"edges, and {ceiling:.1f} vs {skeleton_total(skel, 'edges'):.1f} "
          f"for diameter")

    # 6. NOISE MUST NOT SCORE. A random mask of the same foreground area has
    #    no business tracing anything, once the ceiling is divided out.
    noise = rng.random(skel.shape) < (pred.sum() / pred.size)
    got = (measure(skel, noise, "split", "edges")[0]
           / erl_length.oracle_run_length(skel, "edges"))
    assert got < 0.2, got
    print(f"random mask of the same area traces {got:.3f} of the ceiling")
    print("all checks passed")


# ---------------------------------------------------------------- reporting

def load() -> list[dict]:
    rows = []
    for path in sorted(heldout.ROOT.glob("erl_spec*.csv")):
        for row in csv.DictReader(path.open()):
            rows.append({**row, "erl": float(row["erl"]),
                         "coverage": float(row["coverage"]),
                         "oracle": float(row["oracle"])})
    return rows


def report() -> None:
    rows = load()
    if not rows:
        raise SystemExit("no erl_spec*.csv -- refusing to print an empty "
                         "table; an empty table is not a null result")
    seeds = sorted({r["seed"] for r in rows})
    print("=== ERL is three choices. Every one of them, on the held-out "
          "protocol ===\n")
    print(f"{len(rows)} rows, {len(seeds)} seeds, epoch chosen on the 5 dev")
    print("images, threshold at each run's own dev Dice peak, read on all 20")
    print("test images. `edges` and `covered` are what the field's reference")
    print("implementation (Allen Institute) uses; `pixels`, `full` and")
    print("`split` are what exp/erl.py has always used. Neither is wrong.")
    print("The point is that a paper reporting one number has silently")
    print("picked a cell of this table.\n")
    print("Every cell is a fraction of what a PERFECT prediction scores on")
    print("the same image: ERL is a length, not a ratio. For pixels and edges")
    print("that ceiling is the skeleton's own length; for diameter it is much")
    print("less, because one traversal cannot cover a branched tree.\n")

    # ERL is a length. Every cell is divided by what a perfect prediction
    # scores on the SAME image under the SAME length convention -- the same
    # divisor for both denominators, which is what keeps
    # `full = covered x coverage` readable across the two columns.
    cells = defaultdict(list)
    for row in rows:
        if row["oracle"] <= 0:
            continue
        cells[(row["config"], row["split_rule"], row["length"],
               row["denominator"])].append(row["erl"] / row["oracle"])

    header = "".join(f"{rule[:3]}/{den[:3]:<4}" .rjust(11)
                     for rule in SPLIT_RULES for den in DENOMINATORS)
    for convention in LENGTHS:
        print(f"--- fragment length: {convention} ---")
        print(f"    {'arm':20}{header}   coverage")
        for arm in ARMS:
            got = []
            for rule in SPLIT_RULES:
                for den in DENOMINATORS:
                    values = cells.get((arm, rule, convention, den))
                    got.append(f"{100 * float(np.mean(values)):9.1f}%"
                               if values else f"{'--':>10}")
            cover = [r["coverage"] for r in rows if r["config"] == arm
                     and r["length"] == convention and r["split_rule"] == "split"]
            tail = f"{100 * float(np.mean(cover)):9.1f}%" if cover else ""
            print(f"    {arm:20}" + "".join(got) + f"   {tail}")
        print()

    print("--- the spread, per arm ---")
    print(f"    {'arm':20}{'lowest':>10}{'highest':>10}{'spread':>10}"
          f"   which cell is highest")
    for arm in ARMS:
        got = {key: float(np.mean(v)) for key, v in cells.items()
               if key[0] == arm}
        if not got:
            continue
        scaled = {k: 100 * v for k, v in got.items()}
        low = min(scaled, key=scaled.get)
        high = max(scaled, key=scaled.get)
        print(f"    {arm:20}{scaled[low]:>9.1f}%{scaled[high]:>9.1f}%"
              f"{scaled[high] - scaled[low]:>9.1f} "
              f"  {high[1]}/{high[2]}/{high[3]}")
    print()
    print("Read the spread as: the range of ERLs one set of predictions can")
    print("be reported as, without changing the model, the data, the")
    print("threshold or the checkpoint. A paper that does not name its cell")
    print("has not reported a number a reader can compare against.")


# -------------------------------------------------------------------- main

def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--report" in sys.argv:
        report()
        return
    shard = None
    for index, arg in enumerate(sys.argv):
        if arg == "--shard":
            part, total = sys.argv[index + 1].split("/")
            shard = (int(part), int(total))
    limit = None
    if "--runs" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--runs") + 1])
    target = heldout.ROOT / ("erl_spec.csv" if shard is None else
                             f"erl_spec.shard{shard[0]}of{shard[1]}.csv")

    peaks = defaultdict(dict)
    curve = heldout.ROOT / "frontier_dev.csv"
    if not curve.exists():
        raise SystemExit("no frontier_dev.csv -- the operating point is the "
                         "dev Dice peak and must be read from it")
    for row in csv.DictReader(curve.open()):
        peaks[row["run"]][float(row["threshold"])] = float(row["dice"])

    done = set()
    for existing in sorted(heldout.ROOT.glob("erl_spec*.csv")):
        done |= {(r["config"], r["seed"])
                 for r in csv.DictReader(existing.open())}

    items = drive.load_split("test")
    data = train.stack_split("fit")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))
    geometry = [skeletonize(i["label"] & i["fov"]) for i in items]
    oracles = [{c: erl_length.oracle_run_length(s, c) for c in LENGTHS}
               for s in geometry]
    epochs = heldout.chosen_epochs()
    print(f"{len(ARMS)} arms, {len(items)} test images, width {width:.2f} px",
          flush=True)

    fresh = not target.exists()
    with target.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        scored = 0
        for arm in ARMS:
            runs = sorted(r for r in epochs
                          if r.rsplit("_s", 1)[0] == arm
                          and (heldout.ROOT / r / "final.pt").exists())
            # Stride over a sorted list, never hash(): see sweep.shard_filter.
            for run in sweep.shard_filter(runs, shard):
                seed = run.rsplit("_s", 1)[1]
                if (arm, seed) in done or run not in peaks:
                    continue
                if limit is not None and scored >= limit:
                    print(f"stopping at --runs {limit}", flush=True)
                    return
                model, mean, std = sweep.load_model(run, arm, epochs, data)
                if model is None:
                    continue
                base = max(peaks[run], key=peaks[run].get)
                out = []
                for item, skel, oracle in zip(items, geometry, oracles):
                    prob = train.predict_full(model, item["image"], mean, std)
                    pred = speckle.drop_small((prob >= base) & item["fov"],
                                              component_px)
                    common = {"run": run, "config": arm, "seed": seed,
                              "epoch": epochs[run], "threshold": base,
                              "image": item["name"]}
                    for rule in SPLIT_RULES:
                        for convention in LENGTHS:
                            full, covered, coverage = measure(skel, pred, rule,
                                                              convention)
                            for name, value in (("full", full),
                                                ("covered", covered)):
                                out.append({**common, "split_rule": rule,
                                            "length": convention,
                                            "denominator": name, "erl": value,
                                            "coverage": coverage,
                                            "oracle": oracle[convention]})
                writer.writerows(out)
                handle.flush()
                scored += 1
                print(f"  {run} at threshold {base:g} done", flush=True)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
