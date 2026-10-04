"""The species tree with AleRax 1.4.0's node names and file formats.

AleRax (``PLLRootedTree::ensureUniqueLabels``) walks the tree in post-order and renames an
internal node whose label is empty, parses as a number (a bootstrap value), equals a leaf
label, or equals a label already used, to ``Node_<leftmost leaf of left child>_<leftmost leaf
of right child>_<k>`` with ``k`` counting from 0 until the name is unused.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesTree:
    names: tuple[str, ...]
    lengths: tuple[float, ...]
    child1: tuple[int, ...]
    child2: tuple[int, ...]
    parent: tuple[int, ...]
    root: int
    labels: tuple[str, ...]

    @property
    def leaf_index(self) -> dict[str, int]:
        return {self.labels[i]: i for i in range(len(self.labels)) if self.is_leaf(i)}

    def is_leaf(self, node: int) -> bool:
        return self.child1[node] < 0


def _is_number(label: str) -> bool:
    try:
        float(label)
    except ValueError:
        return False
    return True


def _leftmost_leaf(child1: list[int], node: int) -> int:
    while child1[node] >= 0:
        node = child1[node]
    return node


def _alerax_labels(names: list[str], child1: list[int], child2: list[int]) -> list[str]:
    n = len(names)
    labels = list(names)
    leaf_labels = [names[i] for i in range(n) if child1[i] < 0]
    if len(set(leaf_labels)) != len(leaf_labels):
        raise ValueError("the species tree has two leaves with the same label")
    used = set(leaf_labels)
    for node in range(n):  # gpurec indices are a post-order
        if child1[node] < 0:
            continue
        label = names[node]
        if label == "" or _is_number(label) or label in used:
            prefix = (
                f"Node_{names[_leftmost_leaf(child1, child1[node])]}"
                f"_{names[_leftmost_leaf(child1, child2[node])]}"
            )
            k = 0
            while f"{prefix}_{k}" in used:
                k += 1
            label = f"{prefix}_{k}"
        labels[node] = label
        used.add(label)
    return labels


def species_tree_from_arrays(names, lengths, child1, child2) -> SpeciesTree:
    n = len(names)
    parent = [-1] * n
    for node in range(n):
        for child in (child1[node], child2[node]):
            if child >= 0:
                parent[child] = node
    roots = [node for node in range(n) if parent[node] < 0]
    if len(roots) != 1:
        raise ValueError(f"the species tree must have one root, found {len(roots)}")
    return SpeciesTree(
        names=tuple(names),
        lengths=tuple(float(x) for x in lengths),
        child1=tuple(int(c) for c in child1),
        child2=tuple(int(c) for c in child2),
        parent=tuple(parent),
        root=roots[0],
        labels=tuple(_alerax_labels(list(names), list(child1), list(child2))),
    )


def species_tree_from_preprocess(species_raw: dict) -> SpeciesTree:
    S = int(species_raw["S"])

    def children(key: str) -> list[int]:
        values = species_raw[key]
        values = values.tolist() if hasattr(values, "tolist") else list(values)
        return [int(v) if 0 <= int(v) < S else -1 for v in values]

    return species_tree_from_arrays(
        list(species_raw["names"]), list(species_raw["lengths"]), children("sp_child1"), children("sp_child2")
    )


def _preorder(tree: SpeciesTree) -> list[int]:
    order, stack = [], [tree.root]
    while stack:
        node = stack.pop()
        order.append(node)
        if not tree.is_leaf(node):
            stack.append(tree.child2[node])
            stack.append(tree.child1[node])
    return order


def alerax_node_order(tree: SpeciesTree) -> list[int]:
    """Leaves in left-to-right order, then internal non-root nodes in pre-order, then the root."""
    preorder = _preorder(tree)
    leaves = [node for node in preorder if tree.is_leaf(node)]
    internals = [node for node in preorder if not tree.is_leaf(node) and node != tree.root]
    return leaves + internals + [tree.root]


def species_tree_newick(tree: SpeciesTree, lengths: list[float]) -> str:
    """AleRax's writer: ``%s:%f`` at leaves, ``(%s,%s)%s:%f`` inside, root length included."""

    def text(node: int) -> str:
        if tree.is_leaf(node):
            return f"{tree.labels[node]}:{lengths[node]:.6f}"
        return f"({text(tree.child1[node])},{text(tree.child2[node])}){tree.labels[node]}:{lengths[node]:.6f}"

    return text(tree.root) + ";\n"
