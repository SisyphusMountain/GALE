"""Per-sample files of AleRax 1.4.0 (GeneRaxCore ``Scenario`` and ``ReconciliationWriter``)."""
from __future__ import annotations

from gpurec.io.species_tree import SpeciesTree
from gpurec.io.text import format_number
from gpurec.reconciliation.tree import GeneNode, ReconciledGeneTree, deep_recursion

SPECIES_COUNT_HEADER = (
    "species_label, speciations, duplications, losses, transfers, presence, origination, "
    "copies, singletons, transfers_to\n"
)
_EVENT_COUNT_NAMES = ("S", "SL", "D", "DL", "T", "TL", "L", "Leaf")
_XML_TAGS = {
    "Leaf": "leaf", "S": "speciation", "SL": "speciation", "D": "duplication",
    "T": "branchingOut", "TL": "branchingOut", "L": "loss",
}
_XML_HEADER = (
    "<recPhylo \n"
    '\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
    '\txsi:schemaLocation="http://www.recg.org ./recGeneTreeXML.xsd"\n'
    '\txmlns="http://www.recg.org">\n'
)


def _species_clades(species: SpeciesTree, node: int, indent: str, out: list[str]) -> None:
    out.append(f"{indent}<clade>\n")
    out.append(f"{indent}\t<name>{species.labels[node]}</name>\n")
    if not species.is_leaf(node):
        _species_clades(species, species.child1[node], indent + "\t", out)
        _species_clades(species, species.child2[node], indent + "\t", out)
    out.append(f"{indent}</clade>\n")


def _events_rec(out: list[str], indent: str, kind: str, location: str, transfer_back: str | None) -> None:
    out.append(f"{indent}<eventsRec>\n")
    if transfer_back is not None:
        out.append(f'{indent}\t<transferBack destinationSpecies="{transfer_back}"/>\n')
    out.append(f'{indent}\t<{_XML_TAGS[kind]} speciesLocation="{location}"/>\n')
    out.append(f"{indent}</eventsRec>\n")


def _transfer_back(previous, on_recipient_side: bool, species: SpeciesTree) -> str | None:
    if previous is not None and (previous.kind == "TL" or (previous.kind == "T" and on_recipient_side)):
        return species.labels[previous.dest]
    return None


def _gene_clades(node: GeneNode, previous, on_recipient_side: bool, indent: str,
                 species: SpeciesTree, out: list[str]) -> None:
    *chain, last = node.events
    for event in chain:
        out.append(f"{indent}<clade>\n")
        indent += "\t"
        out.append(f"{indent}<name>NULL</name>\n")
        _events_rec(out, indent, event.kind, species.labels[event.species],
                    _transfer_back(previous, on_recipient_side, species))
        previous, on_recipient_side = event, False
        loss_species = event.lost if event.kind == "SL" else event.species
        out.append(f"{indent}<clade>\n")
        out.append(f"{indent}\t<name>loss</name>\n")
        _events_rec(out, indent + "\t", "L", species.labels[loss_species], None)
        out.append(f"{indent}</clade>\n")
    out.append(f"{indent}<clade>\n")
    out.append(f"{indent}\t<name>{node.gene if node.gene is not None else 'NULL'}</name>\n")
    _events_rec(out, indent + "\t", last.kind, species.labels[last.species],
                _transfer_back(previous, on_recipient_side, species))
    if node.left is not None:
        _gene_clades(node.left, last, False, indent + "\t", species, out)
        _gene_clades(node.right, last, True, indent + "\t", species, out)
    out.append(f"{indent}</clade>\n")
    for _ in chain:
        indent = indent[:-1]
        out.append(f"{indent}</clade>\n")


def recphyloxml_text(tree: ReconciledGeneTree, species: SpeciesTree) -> str:
    out = [_XML_HEADER, "<spTree>\n<phylogeny>\n"]
    _species_clades(species, species.root, "", out)
    out.append('</phylogeny>\n</spTree>\n<recGeneTree>\n<phylogeny rooted="true">\n')
    with deep_recursion(4 * tree.n_nodes):
        _gene_clades(tree.root, None, False, "", species, out)
    out.append("</phylogeny>\n</recGeneTree>\n</recPhylo>\n")
    return "".join(out)


def samples_newick_line(tree: ReconciledGeneTree) -> str:
    """One line of ``<family>_samples.newick``; gpurec has no branch lengths to write."""

    def text(node: GeneNode) -> str:
        if node.left is None:
            return node.gene
        return f"({text(node.left)},{text(node.right)}){node.events[-1].kind}"

    with deep_recursion(4 * tree.n_nodes):
        return text(tree.root) + ";\n"


def event_counts_text(tree: ReconciledGeneTree) -> str:
    counts = {name: 0 for name in _EVENT_COUNT_NAMES}
    for event in tree.events():
        counts[event.kind] += 1
    return "".join(f"{name}:{counts[name]}\n" for name in _EVENT_COUNT_NAMES)


def species_event_counts(tree: ReconciledGeneTree, species: SpeciesTree) -> list[list[float]]:
    """Columns: speciations, duplications, losses, transfers, presence, origination, copies,
    singletons, transfers_to (AleRax ``Scenario::saveSpeciesEventCounts``)."""
    counts = [[0.0] * 9 for _ in species.labels]
    for event in tree.events():
        row = counts[event.species]
        if event.kind in ("S", "SL", "Leaf"):
            row[0] += 1
            row[4] = 1
            row[6] += 1
            if event.previous in ("S", "SL"):
                row[7] += 1
            if event.kind == "SL":
                counts[event.lost][2] += 1
        elif event.kind == "D":
            row[1] += 1
        elif event.kind == "T":
            row[3] += 1
            counts[event.dest][8] += 1
        elif event.kind == "TL":
            row[2] += 1
            row[3] += 1
            counts[event.dest][8] += 1
    counts[tree.origin][5] = 1
    return counts


def species_event_counts_text(counts: list[list[float]], species: SpeciesTree) -> str:
    rows = [SPECIES_COUNT_HEADER]
    # AleRax sorts species labels byte-wise in its summary files; gpurec uses that order everywhere.
    for label, node in sorted(((label, node) for node, label in enumerate(species.labels)), key=lambda item: item[0].encode()):
        values = counts[node]
        if any(value != 0 for value in values):
            rows.append(label + "".join(f", {format_number(value)}" for value in values) + "\n")
    return "".join(rows)


def transfer_pairs(tree: ReconciledGeneTree) -> list[tuple[int, int]]:
    return [(event.species, event.dest) for event in tree.events() if event.kind in ("T", "TL")]


def transfers_text(tree: ReconciledGeneTree, species: SpeciesTree) -> str:
    return "".join(f"{species.labels[src]} {species.labels[dst]} 1\n" for src, dst in transfer_pairs(tree))
