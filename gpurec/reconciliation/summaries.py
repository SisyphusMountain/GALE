"""Summary files of AleRax 1.4.0: per-family means, whole-dataset totals, transfer origins."""
from __future__ import annotations

from gpurec.io.species_tree import SpeciesTree
from gpurec.io.text import format_number
from gpurec.reconciliation.files import transfer_pairs
from gpurec.reconciliation.tree import ReconciledGeneTree


def mean_species_event_counts(per_sample: list[list[list[float]]], n_species: int) -> list[list[float]]:
    mean = [[0.0] * 9 for _ in range(n_species)]
    for counts in per_sample:
        for node in range(n_species):
            for column in range(9):
                mean[node][column] += counts[node][column]
    return [[value / len(per_sample) for value in row] for row in mean]


def mean_transfers(trees: list[ReconciledGeneTree]) -> dict[tuple[int, int], float]:
    totals: dict[tuple[int, int], float] = {}
    for tree in trees:
        for pair in transfer_pairs(tree):
            totals[pair] = totals.get(pair, 0.0) + 1.0
    return {pair: count / len(trees) for pair, count in totals.items()}


def _rank(species: SpeciesTree) -> dict[int, int]:
    """AleRax orders species by their label sorted byte-wise."""
    order = sorted(range(len(species.labels)), key=lambda node: species.labels[node].encode())
    return {node: position for position, node in enumerate(order)}


def transfers_table_text(values: dict[tuple[int, int], float], species: SpeciesTree) -> str:
    rank = _rank(species)
    rows = sorted(
        ((count, rank[src], rank[dst], src, dst) for (src, dst), count in values.items() if count > 0),
        reverse=True,
    )
    return "".join(
        f"{species.labels[src]} {species.labels[dst]} {format_number(count)}\n"
        for count, _, _, src, dst in rows
    )


def add_origin_counts(
    trees: list[ReconciledGeneTree], vertical: list[float], by_transfer: list[list[float]], species: SpeciesTree
) -> None:
    """Add one family's arrivals to the running counts: ``vertical[species]`` and
    ``by_transfer[source][destination]`` (both counted over every sample)."""
    for tree in trees:
        for event in tree.events():
            if event.kind in ("T", "TL"):
                by_transfer[event.species][event.dest] += 1
            elif event.kind == "S":
                vertical[species.child1[event.species]] += 1
                vertical[species.child2[event.species]] += 1
            elif event.kind == "SL":
                vertical[event.dest] += 1
                vertical[event.lost] += 1


def origin_texts_from_counts(
    vertical: list[float], by_transfer: list[list[float]], species: SpeciesTree, n_samples: int
) -> dict[int, str]:
    """``origins/<label>.txt`` from counts summed over every family and sample."""
    n = len(species.labels)
    rank = _rank(species)
    texts = {}
    for node in range(n):
        lines = [f"vertical: {format_number(vertical[node] / n_samples)}\n"]
        sources = sorted(
            ((by_transfer[src][node], rank[src], src) for src in range(n) if by_transfer[src][node] > 0),
            reverse=True,
        )
        lines += [f"{species.labels[src]}, {format_number(count / n_samples)}\n" for count, _, src in sources]
        texts[node] = "".join(lines)
    return texts


def origin_texts(samples: dict[str, list[ReconciledGeneTree]], species: SpeciesTree, n_samples: int) -> dict[int, str]:
    """``origins/<label>.txt``: how often a lineage in each species arrived vertically or by transfer."""
    n = len(species.labels)
    vertical = [0.0] * n
    by_transfer = [[0.0] * n for _ in range(n)]  # [source][destination]
    for trees in samples.values():
        add_origin_counts(trees, vertical, by_transfer, species)
    return origin_texts_from_counts(vertical, by_transfer, species, n_samples)
