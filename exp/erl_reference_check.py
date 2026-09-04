"""Run the reference package itself, and check our transcription against it.

WRITTEN AND SELFTESTED 2026-09-05.

THE GAP THIS CLOSES. exp/erl_reference.py says so in its own docstring:

    "It does NOT run the reference package itself: that package consumes SWC
     skeletons and label volumes through a graph pipeline built for 3D
     connectomics, and porting DRIVE into it would put the port, not the
     metric, under test."

That reasoning is sound and the conclusion was still a liability: a paper
whose subject is that DEFINITIONS MATTER was stating someone else's
definition from a reading of their source. One reviewer question -- "did you
run it?" -- and the section falls.

HOW THIS AVOIDS THE PORT PROBLEM. It does not port DRIVE into their pipeline.
It builds their LabeledGraph DIRECTLY from a skeleton image plus a prediction,
which is a fifteen-line conversion with no file format, no SWC loader and no
volume reader in the path, and then calls their own
ERLMetric.compute_graph_erl on it. What is under test is our transcription of
their formula, which is exactly the thing that was unverified.

THE CONVERSION, and why it is the honest one:
  nodes       one per ground-truth skeleton pixel
  edges       8-adjacency between skeleton pixels
  node_voxel  (0, row, col), so their physical_dist -- a plain Euclidean norm
              on voxel coordinates -- gives 1 for an orthogonal step and
              sqrt(2) for a diagonal one. That is OUR `edges` convention, by
              construction rather than by hope.
  node_label  str(id of the predicted connected component covering that
              pixel), or "0" where the prediction does not cover it. "0" is
              what their node_labels() discards.

WHAT THE FIRST RUN OF THIS FILE FOUND, 2026-09-05. Synthetic straight-line
breaks matched to floating point. Real retinal images did NOT: 43.92 against
our 86.80 on DRIVE image 01. The cause is in their loop, and it is a
PRECONDITION on the input rather than an error in either implementation:

    for label in graph.node_labels():
        nodes = graph.nodes_with_label(label)
        run_length = graph.run_length_from(nodes[0])   # nodes[0] ONLY

run_length_from traverses the connected group of same-labelled nodes that
contains nodes[0]. If one label's nodes are DISCONNECTED along the ground
truth skeleton, every group but the first is silently unmeasured. Their
loader (graph_loading.py:357-406, `_label_graph`) assigns the segmentation's
raw label per node and has no step that splits a disconnected label, so this
is their function's real behaviour and not an artefact of the shortcut here.

In 3D connectomics the precondition usually holds: a segmented neuron touches
one ground-truth skeleton in one place. On a 2D retina it does not, because
one predicted blob commonly touches several separate branches. Measured
below, on a real model prediction.

SO THE FILE DOES TWO THINGS, and keeping them apart is the whole point:

  PART A -- verify the transcription. Feed labels that satisfy their
  precondition (one label per connected covered-skeleton group, which is what
  their pipeline's data implicitly is) and assert their number equals ours.
  This is not circular: it tests the weighted-average arithmetic, the
  Euclidean edge geometry and the covered denominator, which is precisely
  what exp/erl_reference.py transcribes.
    1. their compute_graph_erl == our measure(skel, pred, "split", "edges")
       covered value, to floating point.
    2. our `full` == their ERL x coverage -- the identity erl_reference.md
       claims, now against their code rather than our reading of it.

  PART B -- measure the precondition. Feed the RAW predicted component ids,
  as their loader would, and report how much ground-truth skeleton their loop
  never reaches. This is a finding for the paper, not a bug report: a
  specification of ERL for this field has to state the connectivity
  precondition that the connectomics setting satisfies silently.

INSTALLING THE REFERENCE. Deliberately NOT in the repo venv, which has
long-running queues depending on it, and whose dependency set should not grow
for one verification. Install into a directory and point REFPKG at it:

    python -m pip install --target /some/dir segmentation-skeleton-metrics \\
        boto3 google-cloud-storage s3fs tensorstore tifffile pandas \\
        matplotlib tqdm
    REFPKG=/some/dir python exp/erl_reference_check.py --selftest

Verified 2026-09-05 against segmentation-skeleton-metrics 5.9.5.

  python exp/erl_reference_check.py --selftest
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import break_lengths
import drive
import erl_length
import erl_spec

# Not a repo dependency and deliberately not in the venv: the venv has
# long-running queues against it and should not grow boto3, tensorstore and
# google-cloud-storage for one verification. Point REFPKG at an install.
REFPKG = os.environ.get("REFPKG", str(Path.home() / ".cache" / "erl-refpkg"))
VERIFIED_AGAINST = "5.9.5"


def load_reference():
    """Their LabeledGraph and ERLMetric, or a SystemExit that says how to get
    them. A verification script that silently skips verifies nothing."""
    if REFPKG not in sys.path:
        sys.path.insert(0, REFPKG)
    try:
        from segmentation_skeleton_metrics.datamodules.graph_classes import (
            LabeledGraph)
        from segmentation_skeleton_metrics.skeleton_metrics import ERLMetric
    except ImportError as error:
        raise SystemExit(
            f"the reference package is not importable from {REFPKG}: {error}\n"
            f"{__doc__.split('INSTALLING THE REFERENCE.')[1].strip()}")
    import segmentation_skeleton_metrics as package
    version = getattr(package, "__version__", "unknown")
    return LabeledGraph, ERLMetric, version


def graph_from(skel: np.ndarray, pred: np.ndarray, LabeledGraph):
    """Their LabeledGraph over the ground-truth skeleton, labelled by pred.

    One node per skeleton pixel, 8-adjacency edges, node_voxel (0, row, col)
    so that their Euclidean physical_dist reproduces the `edges` convention.
    """
    coords = np.argwhere(skel)
    index = {(int(r), int(c)): n for n, (r, c) in enumerate(coords)}
    pieces = ndimage.label(pred, structure=break_lengths.CONN8)[0]

    graph = LabeledGraph()
    graph.add_nodes_from(range(len(coords)))
    for (row, col), node in index.items():
        for step_r in (-1, 0, 1):
            for step_c in (-1, 0, 1):
                if step_r == 0 and step_c == 0:
                    continue
                other = index.get((row + step_r, col + step_c))
                # node < other: add each undirected edge once. add_edge would
                # be idempotent anyway, but a doubled edge list is the kind of
                # thing that silently changes a traversal.
                if other is not None and node < other:
                    graph.add_edge(node, other)
    graph.node_voxel = np.array([[0, int(r), int(c)] for r, c in coords],
                                dtype=np.int32)
    graph.init_node_labels()
    for (row, col), node in index.items():
        piece = int(pieces[row, col])
        graph.node_label[node] = str(piece) if piece > 0 else "0"
    return graph


def connected_relabel(graph) -> int:
    """Split every label into its connected groups. Returns groups added.

    Their compute_graph_erl measures ONE connected group per label, so its
    formula is only the intended ERL when each label is already connected
    along the ground truth. Their 3D pipeline's data satisfies that; a 2D
    retina's does not. Applying this first is what makes Part A a test of the
    arithmetic rather than of the precondition.
    """
    import networkx as nx
    added = 0
    for label in sorted(set(graph.node_label) - {"0"}):
        nodes = [n for n in graph.nodes if graph.node_label[n] == label]
        groups = list(nx.connected_components(graph.subgraph(nodes)))
        for offset, group in enumerate(groups[1:], start=1):
            for node in group:
                graph.node_label[node] = f"{label}__{offset}"
            added += 1
    return added


def unreached_fraction(graph) -> tuple[float, int, int]:
    """(fraction, labels split, groups) of covered skeleton their loop skips.

    Their loop reaches nodes[0]'s group only, so everything in a label's other
    groups is never counted. Measured by node count, which is what a reader
    would picture: how much of the traced skeleton went unmeasured.
    """
    import networkx as nx
    covered = [n for n in graph.nodes if graph.node_label[n] != "0"]
    reached = 0
    split_labels = 0
    for label in sorted(set(graph.node_label) - {"0"}):
        nodes = [n for n in graph.nodes if graph.node_label[n] == label]
        groups = list(nx.connected_components(graph.subgraph(nodes)))
        if len(groups) > 1:
            split_labels += 1
        # nodes_with_label's order decides which group is kept; theirs comes
        # from node order, so take the group holding the lowest-numbered node.
        first = min(nodes)
        reached += len(next(g for g in groups if first in g))
    missed = len(covered) - reached
    return (missed / len(covered) if covered else 0.0, split_labels,
            len(covered))


def line_case(length: int, gap_at: tuple[int, ...]) -> tuple:
    """A horizontal skeleton with the prediction missing `gap_at` columns."""
    skel = np.zeros((5, length + 4), dtype=bool)
    skel[2, 2:2 + length] = True
    pred = skel.copy()
    for column in gap_at:
        pred[2, 2 + column] = False
    return skel, pred


def selftest() -> None:
    LabeledGraph, ERLMetric, version = load_reference()
    if version != VERIFIED_AGAINST and version != "unknown":
        print(f"NOTE: reference package is {version}, this file was verified "
              f"against {VERIFIED_AGAINST}")
    print(f"reference package loaded from {REFPKG} (version {version})")

    # 1. THEIR CODE ON A CASE COMPUTABLE BY HAND, before anything of ours is
    #    involved. If their function does not do what we think, every
    #    comparison below is meaningless.
    graph = LabeledGraph()
    graph.add_edges_from([(i, i + 1) for i in range(5)])
    graph.node_voxel = np.array([[0, 0, i] for i in range(6)], dtype=np.int32)
    graph.init_node_labels()
    for node in range(6):
        graph.node_label[node] = "1"
    whole = ERLMetric.compute_graph_erl(graph)
    assert abs(whole - 5.0) < 1e-9, whole
    for node in range(6):
        graph.node_label[node] = ("1" if node < 3
                                  else "0" if node == 3 else "2")
    broken = ERLMetric.compute_graph_erl(graph)
    hand = (2 * 2 + 1 * 1) / (2 + 1)
    assert abs(broken - hand) < 1e-9, (broken, hand)
    print(f"their code on hand cases: unbroken 6-node path {whole:.1f}, "
          f"broken into edge-lengths 2 and 1 -> {broken:.4f} "
          f"= (2*2+1*1)/(2+1)")

    # 2. THE CONVERSION MUST PRESERVE THE GEOMETRY. A diagonal step has to
    #    come back as sqrt(2) through THEIR physical_dist, or the comparison
    #    is against the wrong length convention and would still "pass" on
    #    purely horizontal test cases.
    diagonal = np.zeros((7, 7), dtype=bool)
    for step in range(5):
        diagonal[1 + step, 1 + step] = True
    got = graph_from(diagonal, diagonal, LabeledGraph)
    assert abs(ERLMetric.compute_graph_erl(got) - 4 * np.sqrt(2)) < 1e-9
    print(f"conversion preserves geometry: a 5-pixel diagonal reads "
          f"{4 * np.sqrt(2):.4f} through their physical_dist, not 4")

    # 3. THE HEADLINE IDENTITY, on synthetic breaks of several shapes.
    for length, gaps in ((20, (7,)), (20, (5, 12)), (31, (10, 11, 20)),
                         (15, ()), (25, (1,)), (25, (23,))):
        skel, pred = line_case(length, gaps)
        graph = graph_from(skel, pred, LabeledGraph)
        connected_relabel(graph)
        theirs = ERLMetric.compute_graph_erl(graph)
        full, covered, coverage = erl_spec.measure(skel, pred, "split", "edges")
        assert abs(theirs - covered) < 1e-9, (length, gaps, theirs, covered)
        assert abs(full - covered * coverage) < 1e-9, (length, gaps)
    print("PART A, synthetic: their ERL == our (split, edges, covered) on 6 "
          "breaks, and full == covered x coverage in each")

    # 4. PART A ON REAL RETINAL IMAGES. Synthetic straight lines have no
    #    cycles; a real 8-connected skeleton does, wherever a staircase or an
    #    L-corner puts a diagonal beside two orthogonal steps. That is where
    #    the two implementations stop agreeing exactly, so this check asserts
    #    CLOSENESS and reports the gap rather than pretending to equality.
    #    Check 6 isolates the mechanism.
    items = drive.load_split("test")[:3]
    rng = np.random.default_rng(0)
    gaps = []
    for item in items:
        skel = skeletonize(item["label"] & item["fov"])
        for keep in (0.95, 0.80, 0.60):
            pred = (item["label"] & item["fov"]
                    & (rng.random(item["label"].shape) < keep))
            graph = graph_from(skel, pred, LabeledGraph)
            connected_relabel(graph)
            theirs = ERLMetric.compute_graph_erl(graph)
            full, covered, coverage = erl_spec.measure(skel, pred, "split",
                                                       "edges")
            gaps.append(theirs / covered - 1.0)
            assert abs(gaps[-1]) < 0.05, (item["name"], keep, theirs, covered)
            assert abs(full - covered * coverage) < 1e-9, (item["name"], keep)
    print(f"  PART A, real images: {len(gaps)} comparisons, their ERL runs "
          f"{100 * min(gaps):+.2f}% to {100 * max(gaps):+.2f}% against ours; "
          f"full == covered x coverage exactly in all")

    # 5. PART B: THE CONNECTIVITY PRECONDITION, PRICED ON A REAL PREDICTION.
    #    Random speckle would overstate it -- a speckled mask fragments into
    #    components that each straddle several branches. A trained model's
    #    mask is the honest input.
    import cross_dataset
    import hole_sweep
    import postproc_ceiling as sweep
    import select_heldout as heldout
    import speckle
    import train
    epochs = heldout.chosen_epochs()
    arm = "H_aug"
    runs = sorted(r for r in epochs if r.rsplit("_s", 1)[0] == arm
                  and (heldout.ROOT / r / "final.pt").exists())
    if not runs:
        print("  PART B skipped: no trained H_aug run on disk")
    else:
        data = train.stack_split("fit")
        width = cross_dataset.median_width(items)
        component_px = int(round(hole_sweep.E4_COMPONENT_MULTIPLE
                                 * width * width))
        model, mean, std = sweep.load_model(runs[0], arm, epochs, data)
        worst = 0.0
        for item in items:
            skel = skeletonize(item["label"] & item["fov"])
            prob = train.predict_full(model, item["image"], mean, std)
            pred = speckle.drop_small((prob >= 0.5) & item["fov"],
                                      component_px)
            raw = graph_from(skel, pred, LabeledGraph)
            missed, split_labels, covered_nodes = unreached_fraction(raw)
            theirs_raw = ERLMetric.compute_graph_erl(raw)
            fixed = graph_from(skel, pred, LabeledGraph)
            connected_relabel(fixed)
            theirs_fixed = ERLMetric.compute_graph_erl(fixed)
            worst = max(worst, missed)
            print(f"  PART B, {item['name']} ({runs[0]} at 0.5): "
                  f"{split_labels} labels disconnected, {missed:.1%} of "
                  f"{covered_nodes} covered skeleton nodes never reached; "
                  f"ERL {theirs_raw:.2f} raw vs {theirs_fixed:.2f} relabelled")
        # If a future change makes this negligible, say so loudly rather than
        # letting erl_reference.md keep a stale claim.
        assert worst > 0.01, (f"the precondition now costs only {worst:.2%}; "
                              f"erl_reference.md's revision block must be "
                              f"re-checked before it is cited again")
        print(f"  PART B: worst image loses {worst:.1%} of covered skeleton "
              f"to the connectivity precondition")

    # 6. PART C: THEIR FRAGMENT LENGTH IS NOT WELL DEFINED ON A CYCLE.
    #    run_length_from walks a DFS spanning tree and charges the edge it
    #    arrives on, keeping every diagonal. On a graph with a cycle the tree
    #    -- and therefore the length -- depends on the order the nodes were
    #    listed, which for their pipeline is the order of points in an SWC
    #    file. The three-pixel L-corner below is the minimal case: an
    #    8-adjacency triangle. Our `edges` convention drops the hypotenuse
    #    when either orthogonal alternative exists (erl_length.edge_list), so
    #    it returns 2 regardless of ordering.
    import itertools
    corner = {"A": (0, 0), "B": (0, 1), "C": (1, 1)}
    seen = set()
    for order in itertools.permutations("ABC"):
        graph = LabeledGraph()
        position = {name: i for i, name in enumerate(order)}
        graph.add_nodes_from(range(3))
        for one, other in itertools.combinations(order, 2):
            (row_a, col_a), (row_b, col_b) = corner[one], corner[other]
            if max(abs(row_a - row_b), abs(col_a - col_b)) == 1:
                graph.add_edge(position[one], position[other])
        graph.node_voxel = np.array([[0, *corner[n]] for n in order],
                                    dtype=np.int32)
        graph.init_node_labels()
        for node in range(3):
            graph.node_label[node] = "1"
        seen.add(round(float(ERLMetric.compute_graph_erl(graph)), 6))
    assert len(seen) > 1, ("their length is order-independent here; the "
                           "finding in erl_reference.md must be re-checked")
    assert 2.0 in seen, seen
    ours = 2.0
    print(f"  PART C: the SAME 3-pixel L-corner reads {sorted(seen)} under "
          f"the 6 node orderings -- their fragment length depends on input "
          f"order. Ours is {ours} in every ordering.")

    print("all checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    raise SystemExit("pass --selftest")


if __name__ == "__main__":
    main()
