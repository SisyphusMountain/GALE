"""Write results in AleRax 1.4.0's output layout (see the spec, section 6)."""
from __future__ import annotations

from pathlib import Path

import torch

from gpurec.io.species_tree import SpeciesTree, alerax_node_order, species_tree_newick
from gpurec.io.text import format_number
from gpurec.reconciliation.files import (
    event_counts_text,
    recphyloxml_text,
    samples_newick_line,
    species_event_counts,
    species_event_counts_text,
    transfers_text,
)
from gpurec.interface import FitResult
from gpurec.reconciliation.summaries import (
    add_origin_counts,
    mean_species_event_counts,
    mean_transfers,
    origin_texts_from_counts,
    transfers_table_text,
)
from gpurec.reconciliation.tree import ReconciledGeneTree


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def _coverage_text(data) -> str:
    rows = []
    for label in data.species.leaf_index:
        covering = sum(1 for covered in data.covered_species if label in covered)
        rows.append((covering / data.n_families, label))
    return "".join(f"{label} {format_number(fraction)}\n" for fraction, label in sorted(rows))


class ReconciliationWriter:
    """Writes ``reconciliations/`` one family at a time.

    Each family's files are written by ``add_family``; only running totals are kept for the
    whole-dataset files (``totalSpeciesEventCounts.txt``, ``totalTransfers.txt``, ``origins/``),
    which ``finish`` writes. Memory therefore does not grow with the number of families. The
    totals are sums over families in the order they are added.
    """

    def __init__(self, species: SpeciesTree, folder: Path, n_samples: int):
        if n_samples < 1:
            raise ValueError(f"the number of reconciliation samples per family must be at least 1, got {n_samples}")
        self.species, self.folder, self.n_samples = species, Path(folder), n_samples
        self.all_dir, self.summaries = self.folder / "all", self.folder / "summaries"
        for directory in (self.all_dir, self.summaries, self.folder / "origins"):
            directory.mkdir(parents=True, exist_ok=True)
        n_species = len(species.labels)
        self.total_counts = [[0.0] * 9 for _ in range(n_species)]
        self.total_transfers: dict[tuple[int, int], float] = {}
        self.vertical = [0.0] * n_species
        self.by_transfer = [[0.0] * n_species for _ in range(n_species)]  # [source][destination]

    def add_family(self, name: str, trees: list[ReconciledGeneTree]) -> None:
        if len(trees) != self.n_samples:
            raise ValueError(f"family {name} has {len(trees)} reconciliation samples, expected {self.n_samples}")
        species, all_dir, summaries = self.species, self.all_dir, self.summaries
        n_species = len(species.labels)
        per_sample = []
        for i, tree in enumerate(trees):
            counts = species_event_counts(tree, species)
            per_sample.append(counts)
            _write(all_dir / f"{name}_sample_{i}.xml", recphyloxml_text(tree, species))
            _write(all_dir / f"{name}_eventCounts_{i}.txt", event_counts_text(tree))
            _write(all_dir / f"{name}_speciesEventCounts_{i}.txt", species_event_counts_text(counts, species))
            _write(all_dir / f"{name}_transfers_{i}.txt", transfers_text(tree, species))
        _write(all_dir / f"{name}_samples.newick", "".join(samples_newick_line(tree) for tree in trees))
        mean_counts = mean_species_event_counts(per_sample, n_species)
        _write(summaries / f"{name}_meanSpeciesEventCounts.txt", species_event_counts_text(mean_counts, species))
        family_transfers = mean_transfers(trees)
        _write(summaries / f"{name}_meanTransfers.txt", transfers_table_text(family_transfers, species))
        for node in range(n_species):
            for column in range(9):
                self.total_counts[node][column] += mean_counts[node][column]
        for pair, value in family_transfers.items():
            self.total_transfers[pair] = self.total_transfers.get(pair, 0.0) + value
        add_origin_counts(trees, self.vertical, self.by_transfer, species)

    def finish(self) -> None:
        species, folder = self.species, self.folder
        _write(folder / "totalSpeciesEventCounts.txt", species_event_counts_text(self.total_counts, species))
        _write(folder / "totalTransfers.txt", transfers_table_text(self.total_transfers, species))
        texts = origin_texts_from_counts(self.vertical, self.by_transfer, species, self.n_samples)
        for node, text in texts.items():
            _write(folder / "origins" / f"{species.labels[node]}.txt", text)


def write_alerax(result: FitResult, samples: dict[str, list[ReconciledGeneTree]] | None, prefix: Path) -> None:
    prefix = Path(prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    data, species = result.data, result.data.species
    ordered = sorted((value, name) for name, value in result.log_likelihoods.items())
    _write(prefix / "per_fam_likelihoods.txt", "".join(f"{name} {format_number(value)}\n" for value, name in ordered))

    rates = torch.exp2(result.log2_rates.double()).tolist()
    parameters = prefix / "model_parameters"
    parameters.mkdir(exist_ok=True)
    if result.parametrization == "global":
        d, l, t = (format_number(value) for value in rates[0])
        rows = "".join(f"{species.labels[node]} {d} {l} {t}\n" for node in alerax_node_order(species))
        _write(parameters / "model_parameters.txt", "node D L T\n" + rows)
    else:
        for name, row in zip(data.names, rates):
            _write(parameters / f"{name}_rates.txt", "D L T\n" + " ".join(format_number(v) for v in row) + "\n")

    trees = prefix / "species_trees"
    trees.mkdir(exist_ok=True)
    _write(trees / "starting_species_tree.newick",
           species_tree_newick(species, [length if length > 0 else 1.0 for length in species.lengths]))
    _write(trees / "inferred_species_tree.newick", species_tree_newick(species, [1.0] * len(species.labels)))
    _write(prefix / "perSpeciesCoverage.txt", _coverage_text(data))
    if samples is not None and samples:
        sample_counts = [len(trees) for trees in samples.values()]
        if len(set(sample_counts)) > 1:
            counts_str = ", ".join(f"{name}: {count}" for name, count in zip(samples.keys(), sample_counts))
            raise ValueError(f"every family must have the same number of reconciliation samples, got {counts_str}")
        n_samples = sample_counts[0]
        if n_samples > 0:
            writer = ReconciliationWriter(species, prefix / "reconciliations", n_samples)
            for name, trees in samples.items():
                writer.add_family(name, trees)
            writer.finish()
