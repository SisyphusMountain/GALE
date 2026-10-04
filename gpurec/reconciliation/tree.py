"""A sampled reconciliation as a gene tree whose nodes carry AleRax-style events.

gpurec's sampler returns nodes ``(event, species, [left, right], clade)``. AleRax records, per
gene node, zero or more SL/TL events followed by one S, D, T or Leaf event; dying copies
(duplication-loss, transfer whose transferred copy is lost) are not recorded, and gpurec's sampler
does not emit them either. For a transfer the donor-side child comes first, as in AleRax.
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass

EVENT_KINDS = ("S", "SL", "D", "T", "TL", "Leaf")


@dataclass(frozen=True)
class Event:
    kind: str
    species: int
    dest: int
    lost: int
    previous: str


@dataclass(frozen=True)
class GeneNode:
    events: tuple[Event, ...]
    gene: str | None
    left: "GeneNode | None"
    right: "GeneNode | None"


@contextlib.contextmanager
def deep_recursion(depth: int):
    """Raise Python's recursion limit (1000 by default) to walk gene trees deeper than that."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, depth + 1000))
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


@dataclass(frozen=True)
class ReconciledGeneTree:
    root: GeneNode
    n_nodes: int

    def events(self) -> list[Event]:
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            out.extend(node.events)
            if node.left is not None:
                stack.append(node.right)
                stack.append(node.left)
        return out

    def genes(self) -> list[str]:
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.left is None:
                out.append(node.gene)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return out

    @property
    def origin(self) -> int:
        return self.root.events[0].species


def tree_from_sampler_nodes(nodes: list, clade_to_gene: dict[int, str]) -> ReconciledGeneTree:
    def build(index: int, previous: str) -> GeneNode:
        events = []
        while True:
            kind, species, (a, b), clade = nodes[index]
            if kind == "speciation" and a >= 0 and nodes[b][0] == "loss":
                events.append(Event("SL", species, nodes[a][1], nodes[b][1], previous))
                previous, index = "SL", a
                continue
            if kind == "transfer" and a >= 0 and nodes[a][0] == "loss":
                events.append(Event("TL", species, nodes[b][1], -1, previous))
                previous, index = "TL", b
                continue
            if kind == "leaf":
                events.append(Event("Leaf", species, -1, -1, previous))
                return GeneNode(tuple(events), clade_to_gene[clade], None, None)
            if kind == "speciation":
                event, left, right, left_previous, right_previous = Event("S", species, -1, -1, previous), a, b, "S", "S"
            elif kind == "duplication":
                event, left, right, left_previous, right_previous = Event("D", species, -1, -1, previous), a, b, "D", "D"
            elif kind == "transfer":
                donor, recipient = (a, b) if nodes[a][1] == species else (b, a)
                event = Event("T", species, nodes[recipient][1], -1, previous)
                left, right, left_previous, right_previous = donor, recipient, previous, "T"
            else:
                raise ValueError(f"unexpected sampled event {kind!r}")
            events.append(event)
            return GeneNode(tuple(events), None, build(left, left_previous), build(right, right_previous))

    with deep_recursion(4 * len(nodes)):
        root = build(0, "S")
    return ReconciledGeneTree(root, len(nodes))
