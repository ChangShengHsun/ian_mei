"""How long is a fragment? Pixel count, edge-weight sum, or subtree diameter.

RAISED BY AN EXTERNAL REVIEW, 2026-08-29. exp/erl.py measures a fragment with
np.bincount -- the NUMBER OF PIXELS in it. On an 8-connected skeleton a
diagonal step therefore counts 1 where its true length is sqrt(2), so every
length in this repo is biased low by however much of the skeleton runs
diagonally. Retinal vessels run in every direction, so this is not a small
correction applied uniformly; it is a correction that depends on orientation.

THREE CONVENTIONS, same fragments:

  pixels    erl.py as written. Count of skeleton pixels in the component.
  edges     Sum of edge weights over the component's 8-adjacency graph:
            1 for a 4-neighbour step, sqrt(2) for a diagonal one. A diagonal
            edge is DROPPED when both of its orthogonal alternatives are also
            present, or a staircase is counted twice -- once along the steps
            and once along the hypotenuse.
  diameter  The longest shortest-path between any two pixels of the component,
            with the same edge weights. Differs from `edges` exactly at
            branches: `edges` adds every branch, `diameter` keeps one path.

WHICH IS RIGHT depends on what ERL is for. ERL asks "how far can a tracer
follow this structure before it must stop". A tracer at a bifurcation takes
ONE branch, which argues for `diameter`. But the Allen reference implementation
sums edge lengths over the whole labelled component (graph_classes.py:430,
run_length += physical_dist(i, j) over a traversal), which is `edges`. So
`edges` is the convention that matches the field, and `diameter` is reported
beside it as the sensitivity analysis.

WHY THIS IS NOT A BUG FIX. erl.py's numbers are not wrong, they are a
different unit. Every comparison in this repo is between two arms measured
the same way, and a monotone change of unit cannot flip a paired sign test.
What it changes is the ABSOLUTE traced fraction, which is the number that
gets compared against other papers.

  python exp/erl_length.py --selftest
  python exp/erl_length.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import cross_dataset
import drive
import erl
import hole_sweep
import select_checkpoint as rules_module
import speckle
import summarize_selection as selection
import train

DIAGONAL = float(np.sqrt(2.0))
# (row step, col step, weight). The four orthogonal moves, then the four
# diagonals. Only one direction of each pair, so every edge is visited once.
STEPS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, DIAGONAL), (1, -1, DIAGONAL))


def fragment_labels(skel_gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """erl.fragments' decomposition, kept as a label image.

    Identical to erl.fragments up to the final bincount, so the three
    conventions below differ ONLY in how a fragment is measured, never in
    what a fragment is.
    """
    pieces = ndimage.label(pred, structure=break_lengths.CONN8)[0]
    covered = skel_gt & (pieces > 0)
    return ndimage.label(covered, structure=break_lengths.CONN8)[0]


def edge_list(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(from, to, weight) over `mask`'s 8-adjacency, as flat pixel indices.

    A diagonal edge is dropped when both orthogonal alternatives exist: on a
    staircase the hypotenuse and the two steps are all present, and counting
    all three measures the same advance twice.
    """
    rows, cols = mask.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    starts, ends, weights = [], [], []
    for row_step, col_step, weight in STEPS:
        here = mask[max(0, -row_step):rows - max(0, row_step),
                    max(0, -col_step):cols - max(0, col_step)]
        there = mask[max(0, row_step):rows - max(0, -row_step),
                     max(0, col_step):cols - max(0, -col_step)]
        link = here & there
        if weight == DIAGONAL:
            # The two corner pixels of this diagonal step. EITHER one gives
            # a two-step orthogonal path covering the same advance, so the
            # hypotenuse is redundant and adding it counts the advance twice.
            # (Requiring both was the first draft and was wrong: an L-shaped
            # corner has only one of them and was measured at 2 + sqrt(2).)
            side_a = mask[max(0, -row_step):rows - max(0, row_step),
                          max(0, col_step):cols - max(0, -col_step)]
            side_b = mask[max(0, row_step):rows - max(0, -row_step),
                          max(0, -col_step):cols - max(0, col_step)]
            link = link & ~(side_a | side_b)
        if not link.any():
            continue
        starts.append(index[max(0, -row_step):rows - max(0, row_step),
                            max(0, -col_step):cols - max(0, col_step)][link])
        ends.append(index[max(0, row_step):rows - max(0, -row_step),
                          max(0, col_step):cols - max(0, -col_step)][link])
        weights.append(np.full(int(link.sum()), weight))
    if not starts:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0, float))
    return (np.concatenate(starts), np.concatenate(ends),
            np.concatenate(weights))


def lengths(skel_gt: np.ndarray, pred: np.ndarray,
            convention: str) -> np.ndarray:
    """Fragment lengths under one convention."""
    labels = fragment_labels(skel_gt, pred)
    count = int(labels.max())
    if count == 0:
        return np.zeros(0)
    if convention == "pixels":
        return np.bincount(labels.ravel())[1:].astype(np.float64)

    starts, ends, weights = edge_list(labels > 0)
    if convention == "edges":
        # Every edge lies inside one fragment, so charging it to the label of
        # either endpoint is the same thing.
        owner = labels.ravel()[starts]
        return np.bincount(owner, weights=weights,
                           minlength=count + 1)[1:].astype(np.float64)
    if convention == "diameter":
        out = np.zeros(count)
        graph = sparse.coo_matrix(
            (np.concatenate([weights, weights]),
             (np.concatenate([starts, ends]), np.concatenate([ends, starts]))),
            shape=(labels.size, labels.size)).tocsr()
        for label in range(1, count + 1):
            nodes = np.flatnonzero(labels.ravel() == label)
            if len(nodes) == 1:
                continue
            sub = graph[nodes][:, nodes]
            # Two sweeps: farthest node from an arbitrary start, then farthest
            # from that one. Exact on a tree, and a skeleton fragment is one
            # up to the small cycles a loop in the vessel tree creates.
            first = csgraph.dijkstra(sub, indices=0)
            far = int(np.argmax(np.where(np.isfinite(first), first, -1)))
            second = csgraph.dijkstra(sub, indices=far)
            out[label - 1] = float(np.max(second[np.isfinite(second)]))
        return out
    raise ValueError(f"unknown convention {convention!r}")


def run_length(skel_gt: np.ndarray, pred: np.ndarray,
               convention: str) -> tuple[float, float]:
    """(ERL, total skeleton length) under one convention.

    ERL is an EXPECTATION: land on a uniformly random skeleton pixel, and
    report how far the fragment containing it lets you go. So it is always

        sum_i  (mass_i / L) * value_i

    where mass_i is how much skeleton fragment i holds -- that is what sets
    the probability of landing in it -- and value_i is how far it takes you.
    For `pixels` and `edges` the two are the SAME quantity, which is why the
    familiar form sum(l^2)/L is correct there.

    For `diameter` they are NOT the same. A branched fragment still holds all
    of its wire, so the probability of landing in it is unchanged; what
    changes is that a tracer leaving that pixel follows ONE path. Writing
    sum(diameter^2)/L would shrink the probability as well as the value, and
    it does: a PERFECT prediction scored 1.3% that way, which is not a low
    score but a broken statistic. mass stays `edges`; only value changes.

    The denominator moves with the numerator: a fraction whose numerator is
    sqrt(2)-weighted length and whose denominator is a pixel count is not a
    fraction of anything.
    """
    if not skel_gt.any():
        return 0.0, 0.0
    if convention == "pixels":
        total = float(skel_gt.sum())
    else:
        _, _, weights = edge_list(skel_gt)
        total = float(weights.sum())
    if total == 0:
        return 0.0, 0.0
    if convention == "diameter":
        mass = lengths(skel_gt, pred, "edges")
        value = lengths(skel_gt, pred, "diameter")
        return float((mass * value).sum() / total), total
    pieces = lengths(skel_gt, pred, convention)
    return float((pieces ** 2).sum() / total), total


def oracle_run_length(skel_gt: np.ndarray, convention: str) -> float:
    """What a PERFECT prediction scores on this image, under `convention`.

    For `pixels` and `edges` this is the skeleton's own length and the raw
    value is already a fraction of it. For `diameter` it is not: a tracer
    follows one path, so even a flawless segmentation of a branched tree
    reaches a fraction of its wire. Dividing by this ceiling is what turns
    `diameter` back into a fraction of something achievable.
    """
    perfect = ndimage.binary_dilation(skel_gt, np.ones((3, 3)))
    return run_length(skel_gt, perfect, convention)[0]


def selftest() -> None:
    # 1. A straight horizontal run: every step is orthogonal, so all three
    #    conventions must agree up to the count/length off-by-one (n pixels
    #    span n-1 edges).
    skel = np.zeros((20, 100), dtype=bool)
    skel[10, 10:90] = True
    full = np.zeros_like(skel)
    full[9:12, 10:90] = True
    got = {c: lengths(skel, full, c)[0] for c in ("pixels", "edges", "diameter")}
    assert got["pixels"] == 80.0, got
    assert abs(got["edges"] - 79.0) < 1e-9, got
    assert abs(got["diameter"] - 79.0) < 1e-9, got
    print(f"  straight run of 80 px: {got}")

    # 2. A pure diagonal. THIS is the case erl.py gets wrong: 40 pixels of
    #    45-degree vessel advance 39*sqrt(2) = 55.2, not 39.
    diagonal = np.zeros((60, 60), dtype=bool)
    for step in range(40):
        diagonal[10 + step, 10 + step] = True
    cover = ndimage.binary_dilation(diagonal, np.ones((3, 3)))
    got = {c: lengths(diagonal, cover, c)[0]
           for c in ("pixels", "edges", "diameter")}
    assert got["pixels"] == 40.0, got
    assert abs(got["edges"] - 39 * DIAGONAL) < 1e-9, got
    assert abs(got["diameter"] - 39 * DIAGONAL) < 1e-9, got
    print(f"  45-degree run of 40 px: {got} -- pixels understates by "
          f"{(got['edges'] / got['pixels'] - 1):.1%}")

    # 3. A staircase, where the diagonal suppression matters. Without it the
    #    hypotenuse is added on top of the two steps it duplicates.
    stair = np.zeros((20, 20), dtype=bool)
    stair[5, 5] = stair[5, 6] = stair[6, 6] = True
    cover = ndimage.binary_dilation(stair, np.ones((3, 3)))
    edges_only = lengths(stair, cover, "edges")[0]
    assert abs(edges_only - 2.0) < 1e-9, edges_only
    print(f"  L-shaped staircase: edges {edges_only:.3f} (2.0, not "
          f"{2 + DIAGONAL:.3f}) -- the hypotenuse is suppressed")

    # 4. A branch. `edges` adds all three arms; `diameter` keeps the longest
    #    path through two of them. This is the ambiguity the review named.
    branch = np.zeros((40, 40), dtype=bool)
    branch[20, 10:31] = True          # 21 px trunk
    branch[10:20, 20] = True          # 10 px arm going up from the middle
    cover = ndimage.binary_dilation(branch, np.ones((3, 3)))
    edges_sum = lengths(branch, cover, "edges")[0]
    diameter = lengths(branch, cover, "diameter")[0]
    assert abs(edges_sum - 30.0) < 1e-9, edges_sum
    assert abs(diameter - 20.0) < 1e-9, diameter
    assert edges_sum > diameter
    print(f"  T-junction: edges {edges_sum:.1f} (all three arms), "
          f"diameter {diameter:.1f} (one path)")

    # 5. The decomposition must be erl.py's, exactly. If these ever disagree
    #    the three conventions are measuring different fragments and none of
    #    the comparisons below mean anything.
    rng = np.random.default_rng(0)
    for _ in range(5):
        truth = np.zeros((40, 40), dtype=bool)
        truth[20, 5:35] = True
        pred = ndimage.binary_dilation(truth, np.ones((3, 3)))
        pred[:, rng.integers(10, 30)] = False
        assert np.array_equal(
            np.sort(lengths(truth, pred, "pixels")),
            np.sort(erl.fragments(truth, pred).astype(np.float64)))
    print("  the fragment decomposition is erl.fragments', unchanged")

    # 6. THE TWO THINGS THIS FILE GOT WRONG, both caught by one property: a
    #    PERFECT prediction must score 1.0. Asserted on a BRANCHED tree,
    #    because a straight line cannot tell any of these apart.
    tree = np.zeros((120, 120), dtype=bool)
    tree[60, 10:110] = True                       # trunk
    for column in range(20, 105, 12):             # ribs, both sides
        tree[45:60, column] = True
        tree[60:76, column] = True
    cover = ndimage.binary_dilation(tree, np.ones((3, 3)))

    #    (a) The first draft weighted the diameter BY the diameter. That
    #        scored a perfect prediction at 1.3% on a real retina -- not a low
    #        score, a broken statistic. Mass must stay `edges`.
    for convention in ("pixels", "edges"):
        value, total = run_length(tree, cover, convention)
        assert 0.90 < value / total <= 1.0, (convention, value / total)

    #    (b) Even weighted correctly, `diameter` has a CEILING BELOW ONE that
    #        depends on how branched the tree is, not on the segmentation: a
    #        tracer follows one path and cannot cover a tree. Here a perfect
    #        prediction reaches 0.34 of the skeleton's length. So `diameter`
    #        is only a fraction of something when it is divided by its own
    #        ceiling on the same image, and that is how main() reports it.
    raw, total = run_length(tree, cover, "diameter")
    assert 0.2 < raw / total < 0.5, raw / total
    ceiling = oracle_run_length(tree, "diameter")
    assert abs(raw / ceiling - 1.0) < 1e-9, raw / ceiling
    print(f"  a perfect prediction scores 1.000 under every convention "
          f"(diameter reaches only {raw / total:.2f} of skeleton length -- "
          f"a tracer cannot cover a tree, so it is normalised by that)")

    #    (c) And a BROKEN prediction must score below 1 under all three, or
    #        the normalisation has flattened the metric into uselessness.
    broken = cover.copy()
    broken[:, 60:63] = False
    for convention in ("pixels", "edges", "diameter"):
        value, _ = run_length(tree, broken, convention)
        assert value < oracle_run_length(tree, convention) * 0.99, convention
    print("  a broken prediction still scores below the ceiling under all "
          "three")
    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    points = selection.selection_points(selection.load())
    rule = dict(rules_module.rules())["(iv) best clDice"]
    items = drive.load_split("val")
    data = train.stack_split("train")
    width = cross_dataset.median_width(items)
    component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE * width * width))

    print("Traced fraction under three length conventions, rule (iv), "
          "report half.")
    print("Every column is a fraction of what a PERFECT prediction scores on "
          "the same image.\n")
    header = (f"  {'arm':<16}{'pixels':>9}{'edges':>9}{'diameter':>10}"
              f"{'edges/px':>10}  runs")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config in selection.ARMS:
        runs = sorted(r for r in points if r.rsplit("_s", 1)[0] == config)
        got = {c: [] for c in ("pixels", "edges", "diameter")}
        for run in runs:
            weights = selection.SWEEP / run / f"epoch{rule(points[run])['epoch']:03d}.pt"
            if not weights.exists():
                continue
            model = train.build_model(config)
            model.load_state_dict(train.load_checkpoint(weights)["model"])
            model.eval()
            mean, std = train.normalisation(run, data)
            for item in items:
                if rules_module.is_selection_image(item["name"]):
                    continue
                skel = skeletonize(item["label"] & item["fov"])
                prob = train.predict_full(model, item["image"], mean, std)
                pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                          component_px)
                for convention in got:
                    value, _ = run_length(skel, pred, convention)
                    ceiling = oracle_run_length(skel, convention)
                    got[convention].append(value / ceiling if ceiling else 0.0)
        if not got["pixels"]:
            print(f"  {config:<16}{'no checkpoints':>38}")
            continue
        means = {c: float(np.mean(v)) for c, v in got.items()}
        print(f"  {config:<16}{means['pixels']:8.1%}{means['edges']:9.1%}"
              f"{means['diameter']:10.1%}"
              f"{means['edges'] / means['pixels']:9.3f}x  {len(runs)}",
              flush=True)
    print()
    print("  `edges` is the field's convention (the Allen implementation sums")
    print("  physical edge lengths). `pixels` is what erl.py has always used;")
    print("  it undercounts diagonal steps, but numerator and denominator")
    print("  take the same correction, so the FRACTION barely moves.")
    print()
    print("  `diameter` asks a different question: not how much wire is")
    print("  connected, but how far ONE traversal gets. A tree cannot be")
    print("  covered by one path, so it is reported against its own ceiling.")
    print("  Where diameter falls much further than edges, the prediction is")
    print("  keeping the tree connected but not keeping any long path clean.")


if __name__ == "__main__":
    main()
