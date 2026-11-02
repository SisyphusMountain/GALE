"""Rate classes: AleRax's custom parametrization (``--model-parametrization <file>``).

One duplication, loss and transfer rate per class of species branches instead of one for the whole
tree. The file has one line per group, ``<node label> <types>``, for example ``Asgard DLT`` or
``DPANN D``. Each letter (D, L or T) starts a new class of that rate at the node, which its whole
subtree inherits unless a descendant line starts another; the three rates are independent. The
root's class is the global one, so a line naming the root is skipped, as AleRax does. Node labels
are the species tree's AleRax labels (``SpeciesTree.labels``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpurec.io.species_tree import SpeciesTree, _preorder

KINDS = "DLT"


@dataclass(frozen=True)
class RateClasses:
    """For each rate (D, L, T): the class of every species node, and the label that starts each
    class (the root's label for class 0)."""

    class_of: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    starts: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]

    @property
    def counts(self) -> tuple[int, int, int]:
        return tuple(len(labels) for labels in self.starts)


def parse_rate_classes(text: str, species: SpeciesTree, source: str = "rate classes") -> RateClasses:
    index = {label: node for node, label in enumerate(species.labels)}
    root_label = species.labels[species.root]
    starts: list[list[str]] = [[root_label] for _ in KINDS]
    start_class: list[dict[int, int]] = [{} for _ in KINDS]
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{source}, line {number}: expected '<node label> <types>', got {line.strip()!r}")
        label, types = fields
        if label not in index:
            raise ValueError(f"{source}, line {number}: {label!r} is not a node of the species tree")
        if label in seen:
            raise ValueError(f"{source}, line {number}: {label!r} appears twice")
        seen.add(label)
        unknown = sorted(set(types) - set(KINDS))
        if "O" in unknown:
            raise ValueError(f"{source}, line {number}: origination classes (O) are not supported")
        if unknown:
            raise ValueError(f"{source}, line {number}: unknown rate types {''.join(unknown)!r} (D, L or T)")
        if index[label] == species.root:
            continue
        for k, kind in enumerate(KINDS):
            if kind in types:
                start_class[k][index[label]] = len(starts[k])
                starts[k].append(label)
    class_of = [[0] * len(species.labels) for _ in KINDS]
    for node in _preorder(species):
        parent = species.parent[node]
        for k in range(len(KINDS)):
            if node in start_class[k]:
                class_of[k][node] = start_class[k][node]
            elif parent >= 0:
                class_of[k][node] = class_of[k][parent]
    return RateClasses(
        class_of=tuple(tuple(column) for column in class_of),
        starts=tuple(tuple(labels) for labels in starts),
    )


def read_rate_classes(path: Path, species: SpeciesTree) -> RateClasses:
    return parse_rate_classes(Path(path).read_text(), species, source=str(path))
