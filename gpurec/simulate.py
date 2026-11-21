"""Forward simulation of gene families under GALE's undated duplication-transfer-loss model.

The model is the one GALE's likelihood computes (README, "Simulating gene families"):

- a family starts with one gene copy on a branch given by the origination rule (``ORIGINATIONS``):
  ``"uniform"``, every branch of the species tree equally likely (leaf branches, internal branches
  and the root branch), the start GALE's likelihood assumes; ``"root"``, always the root branch
  (AleRax's ``--origination ROOT``), which GALE's likelihood does not implement;
- a copy on branch e undergoes one event, with probabilities proportional to 1 (speciation),
  D_e (duplication), L_e (loss) and T_e (transfer), the rates of branch e (and of the family):
  a speciation sends one copy to each child branch, or on a leaf branch makes a sampled gene; a
  duplication leaves two copies on e; a transfer leaves one copy on e and sends one to a branch
  h that is neither e nor an ancestor of e, with probability exp(w_h) / (the sum of exp(w) over
  those branches), w the natural-log receiver weights (``GeneReconModel.receiver_weights``); a
  loss ends the copy; every new copy then undergoes its own event;
- GALE's likelihood is the probability of the gene tree given that the family leaves at least one
  gene: a family that leaves none is drawn again, and counted.

The simulation runs in GALE's native module (``crates/gpurec-backtrack/src/simulate.rs``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Real
from pathlib import Path

import numpy as np

from gpurec.core.backtracking.input import _load_native_module as _backtrack_module
from gpurec.core.scheduling.batching import _load_native_module as _preprocess_module
from gpurec.io.species_tree import SpeciesTree, species_tree_from_preprocess
from gpurec.reconciliation.files import recphyloxml_text, species_event_counts_text
from gpurec.reconciliation.tree import ReconciledGeneTree, tree_from_sampler_nodes

# Where a family starts. "uniform": on any branch of the species tree with the same probability,
# the only start GALE's likelihood implements (``--origination UNIFORM``, gpurec/cli/arguments.py);
# "root": always at the top of the root branch, as AleRax's ``--origination ROOT``, for comparisons
# with AleRax: GALE cannot fit families simulated with it under the start they were made with.
ORIGINATIONS = ("uniform", "root")

# Events of a family's whole history, lost copies included (``SimulatedFamily.history``).
HISTORY_COLUMNS = ("speciations", "duplications", "losses", "transfers", "genes")


@dataclass(frozen=True)
class SimulatedFamily:
    """One simulated family. ``newick``: its gene tree, genes named ``<species>_<k>``;
    ``gene_species``: ``{gene: species leaf}``; ``reconciliation``: its true reconciled gene tree,
    as GALE's sampler shows one (``gpurec.reconciliation.files.recphyloxml_text`` writes it);
    ``event_counts``: the visible events of that tree, one row per branch (GALE's numbering), the
    columns of AleRax's ``speciesEventCounts`` files (``gpurec.reconciliation.files``,
    ``SPECIES_COUNT_HEADER``), counted as GALE counts a sampled reconciliation
    (``species_event_counts``); ``history``: the events of the whole history, lost copies included
    (``HISTORY_COLUMNS``)."""

    name: str
    newick: str
    gene_species: dict[str, str]
    reconciliation: ReconciledGeneTree
    event_counts: list[list[float]]
    history: tuple[int, ...]


@dataclass(frozen=True)
class Simulation:
    """The families, the species tree they were simulated on (branches numbered and named as
    GALE numbers and names them), how many attempts were drawn again because they left no gene,
    went over the copy cap or left some genes but fewer than the minimum, and the settings."""

    species: SpeciesTree
    families: tuple[SimulatedFamily, ...]
    attempts_without_gene: int
    attempts_over_cap: int
    attempts_too_few_genes: int
    settings: dict

    def write(self, folder: Path) -> None:
        """Write the families in a new folder, in files GALE reads (README, "Simulating gene
        families")."""
        folder = Path(folder)
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"{folder} is not empty")
        for sub in ("gene_trees", "mappings", "reconciliations", "event_counts"):
            (folder / sub).mkdir(parents=True)
        species = self.species
        (folder / "species_tree.newick").write_text(Path(self.settings["species_tree"]).read_text())
        entries, history = ["[FAMILIES]"], ["family\t" + "\t".join(HISTORY_COLUMNS)]
        for family in self.families:
            tree = (folder / "gene_trees" / f"{family.name}.newick").resolve()
            mapping = (folder / "mappings" / f"{family.name}.link").resolve()
            tree.write_text(family.newick + "\n")
            mapping.write_text("".join(f"{g} {s}\n" for g, s in family.gene_species.items()))
            entries += [f"- {family.name}", f"gene_tree = {tree}", f"mapping = {mapping}"]
            (folder / "reconciliations" / f"{family.name}.xml").write_text(
                recphyloxml_text(family.reconciliation, species))
            (folder / "event_counts" / f"{family.name}_speciesEventCounts.txt").write_text(
                species_event_counts_text(family.event_counts, species))
            history.append("\t".join([family.name, *map(str, family.history)]))
        (folder / "families.txt").write_text("\n".join(entries) + "\n")
        (folder / "history_counts.tsv").write_text("\n".join(history) + "\n")
        record = dict(self.settings, attempts_without_gene=self.attempts_without_gene,
                      attempts_over_cap=self.attempts_over_cap,
                      attempts_too_few_genes=self.attempts_too_few_genes, gale_version=version("gpurec"))
        (folder / "settings.json").write_text(json.dumps(record, indent=1) + "\n")


def _rate_rows(rates, labels) -> list[list[float]]:
    """One (D, L, T) row per branch from three numbers (every branch) or ``{label: (D, L, T)}``."""
    if isinstance(rates, dict):
        missing, unknown = set(labels) - set(rates), set(rates) - set(labels)
        if missing or unknown:
            raise ValueError(f"rates: branches without rates {sorted(missing)[:5]}, names that are not "
                             f"branches {sorted(unknown)[:5]}")
        rows = [list(rates[label]) for label in labels]
    else:
        rows = [list(rates)] * len(labels)
    for row in rows:
        if len(row) != 3 or not all(isinstance(x, Real) and math.isfinite(x) and x >= 0 for x in row):
            raise ValueError(f"rates must be three non-negative numbers (D, L, T), got {row}")
    return rows


# The node kinds of FamilyArrays.event, by code (the Rust code uses the same order).
EVENTS = ("leaf", "speciation", "duplication", "transfer", "loss")


@dataclass(frozen=True)
class FamilyArrays:
    """A batch of simulated families as flat arrays: the families of one batch, one after another
    in each array. Family f has the nodes ``node_offsets[f]:node_offsets[f + 1]`` and the genes
    ``gene_offsets[f]:gene_offsets[f + 1]``.

    Per node (one row of the arrays; the nodes of a family are numbered 0 upwards from its root):
    ``parent``: the number of the parent within the family, -1 at the root; ``children``: [nodes,
    2], the numbers of the two children within the family, -1 for none; ``branch``: the species
    branch, an index into ``species_tree.labels`` (GALE's numbering); ``event``: the kind, an index
    into ``EVENTS`` (0 a sampled gene, 1 a speciation, 2 a duplication, 3 a transfer, 4 a copy
    lost, shown only as a child of a speciation or a transfer); ``gene``: for a sampled gene, its
    number within the family (its name is ``gene_names[gene_offsets[f] + gene]``), -1 otherwise.

    Per family: ``gene_names``: the gene leaf names (``<species>_<k>``), all families' names in one
    list; ``newick``: each family's gene tree as Newick text; ``history``: [families,
    5] the events of the whole history (``HISTORY_COLUMNS``), lost copies included;
    ``attempts_without_gene``, ``attempts_over_cap`` and ``attempts_too_few_genes``: [families],
    the attempts drawn again;
    ``event_counts``: [families, species, 9] the per-species counts (the columns of
    ``gpurec.reconciliation.files.SPECIES_COUNT_HEADER``), or None when they were not asked for.
    """

    species_tree: SpeciesTree
    node_offsets: np.ndarray
    parent: np.ndarray
    children: np.ndarray
    branch: np.ndarray
    event: np.ndarray
    gene: np.ndarray
    gene_offsets: np.ndarray
    gene_names: list[str]
    newick: list[str]
    history: np.ndarray
    attempts_without_gene: np.ndarray
    attempts_over_cap: np.ndarray
    attempts_too_few_genes: np.ndarray
    event_counts: np.ndarray | None


def simulate_family_arrays(species_tree, rates, receiver_weights, origination: str, n_families: int,
                           seed: int, max_copies: int, min_genes: int, threads: int,
                           event_counts: bool) -> FamilyArrays:
    """Simulate ``n_families`` families as flat arrays (``FamilyArrays``). The arguments are those
    of ``simulate_families``, and ``event_counts``: whether to count every family's events per
    species. Needs no GPU: the simulation and the arrays are made in Rust."""
    species = species_tree_from_preprocess(json.loads(_preprocess_module().species_payload(str(species_tree))))
    labels = list(species.labels)
    if origination not in ORIGINATIONS:
        raise ValueError(f"origination must be one of {ORIGINATIONS}, got {origination!r}")
    if not (isinstance(n_families, int) and n_families >= 1):
        raise ValueError(f"n_families must be a positive whole number, got {n_families!r}")
    per_family = isinstance(rates, list) and not all(isinstance(x, Real) for x in rates)
    if per_family and len(rates) != n_families:
        raise ValueError(f"per-family rates: {len(rates)} rate entries for {n_families} families")
    rows = [r for entry in (rates if per_family else [rates]) for r in _rate_rows(entry, labels)]
    if receiver_weights == "uniform":
        weights = [0.0] * len(labels)
    elif isinstance(receiver_weights, dict) and set(receiver_weights) == set(labels):
        weights = [float(receiver_weights[label]) for label in labels]
    else:
        raise ValueError('receiver_weights must be "uniform" or {branch label: weight} for every branch')
    # Probability of starting on each branch (the native call normalises them).
    start = ([1.0] * len(labels) if origination == "uniform"
             else [1.0 if s == species.root else 0.0 for s in range(len(labels))])
    found = _backtrack_module().simulate_arrays(
        list(species.child1), list(species.child2), labels, [x for row in rows for x in row], weights,
        start, n_families, seed, max_copies, min_genes, threads, event_counts)
    return FamilyArrays(
        species_tree=species, node_offsets=found["node_offsets"], parent=found["parent"],
        children=found["children"], branch=found["branch"], event=found["event"], gene=found["gene"],
        gene_offsets=found["gene_offsets"], gene_names=found["gene_names"], newick=found["newick"],
        history=found["history"], attempts_without_gene=found["attempts_without_gene"],
        attempts_over_cap=found["attempts_over_cap"],
        attempts_too_few_genes=found["attempts_too_few_genes"], event_counts=found["event_counts"])


def _family_from_arrays(arrays: FamilyArrays, f: int, name: str) -> SimulatedFamily:
    """The object form of family ``f`` (its reconciled tree, gene species and counts)."""
    labels = arrays.species_tree.labels
    first, last = int(arrays.node_offsets[f]), int(arrays.node_offsets[f + 1])
    names = arrays.gene_names[int(arrays.gene_offsets[f]):int(arrays.gene_offsets[f + 1])]
    nodes = [(EVENTS[int(arrays.event[k])], int(arrays.branch[k]),
              (int(arrays.children[k, 0]), int(arrays.children[k, 1])), int(arrays.gene[k]))
             for k in range(first, last)]
    leaf_branch = [None] * len(names)
    for _, branch, _, gene in nodes:
        if gene >= 0:
            leaf_branch[gene] = branch
    return SimulatedFamily(
        name=name, newick=arrays.newick[f],
        gene_species={names[k]: labels[leaf_branch[k]] for k in range(len(names))},
        reconciliation=tree_from_sampler_nodes(nodes, dict(enumerate(names))),
        event_counts=[[float(x) for x in row] for row in arrays.event_counts[f]],
        history=tuple(int(x) for x in arrays.history[f]))


def simulate_families(species_tree, rates, receiver_weights, origination: str, n_families: int,
                      seed: int, max_copies: int, min_genes: int, threads: int) -> Simulation:
    """Simulate ``n_families`` families that each leave at least ``min_genes`` genes (module
    docstring).

    ``species_tree``: a rooted binary Newick file. ``rates``: duplication, loss and transfer rates
    relative to speciation, in GALE's order (D, L, T): three numbers for every branch (GALE's
    global rates), ``{branch label: (D, L, T)}`` for every branch (per-species rates; labels as
    in GALE's ``model_parameters.txt``), or a list of ``n_families`` of these, one per family
    (per-family rates). ``receiver_weights``: ``"uniform"`` or ``{branch label: natural-log
    weight}`` for every branch (the root's is never used). ``origination``: one of
    ``ORIGINATIONS`` (module docstring). ``seed``: the output depends on it alone, not on
    ``threads``. ``max_copies``: an attempt whose history reaches this many gene copies is
    drawn again and counted (``Simulation.attempts_over_cap``): when that count is not zero, the
    families are those of the model conditioned on staying under the cap, not of the model.
    ``min_genes``: an attempt that leaves some genes but fewer than this is drawn again and counted
    (``Simulation.attempts_too_few_genes``); 1 keeps every family that leaves a gene, which is what
    GALE's likelihood (and AleRax's) assumes. With more than 1, the families are those of the model
    conditioned on leaving at least ``min_genes`` genes, which no likelihood of GALE or AleRax
    accounts for: rates fitted to them are biased (for example for programs that cannot read
    families of one or two genes).

    GALE fits global or per-family rates, with every receiver weight equal and the ``"uniform"``
    start (README, "Simulating gene families"): families simulated with per-species rates, unequal
    receiver weights or the ``"root"`` start are not of a model GALE fits.
    """
    arrays = simulate_family_arrays(species_tree, rates, receiver_weights, origination, n_families,
                                    seed, max_copies, min_genes, threads, True)
    width = len(str(n_families - 1))
    families = tuple(_family_from_arrays(arrays, f, f"family{f:0{width}d}") for f in range(n_families))
    settings = {"species_tree": str(Path(species_tree).resolve()), "rates": rates,
                "receiver_weights": receiver_weights, "origination": origination,
                "n_families": n_families, "seed": seed, "max_copies": max_copies, "min_genes": min_genes,
                "threads": threads}
    return Simulation(arrays.species_tree, families, int(arrays.attempts_without_gene.sum()),
                      int(arrays.attempts_over_cap.sum()), int(arrays.attempts_too_few_genes.sum()), settings)
