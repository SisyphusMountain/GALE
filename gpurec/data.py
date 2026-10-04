"""Gene families parsed once, and GPU models built from them for any subset of families."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from gpurec.api.model import (
    DEFAULT_BATCH_PACKING,
    DEFAULT_FAMILY_CHUNK_SIZE,
    DEFAULT_MAX_WAVE_SIZE,
    GeneReconModel,
)
from gpurec.api.solver_options import SolverOptions
from gpurec.config import GpurecConfig
from gpurec.core.scheduling.batching import preprocess_dataset
from gpurec.io.families_file import (
    FamilyEntry,
    read_families_file,
    read_fraction_missing_file,
    read_mapping_file,
)
from gpurec.io.species_tree import SpeciesTree, species_tree_from_preprocess

logger = logging.getLogger(__name__)

# File extensions taken from a directory of gene trees (AleRax .ale files and Newick files).
_TREE_SUFFIXES = (".ale", ".newick", ".nwk")


@dataclass(frozen=True)
class FamilyData:
    species_tree_path: Path
    species: SpeciesTree
    species_raw: dict
    families: tuple[dict, ...]
    names: tuple[str, ...]
    covered_species: tuple[frozenset[str], ...]
    dropped: tuple[tuple[str, str], ...]
    fraction_missing: dict[str, float] | None

    @property
    def n_families(self) -> int:
        return len(self.names)

    def build_model(
        self,
        indices: list[int],
        mode: str,
        solver_options: SolverOptions,
        config: GpurecConfig,
        clade_budget: int,
    ) -> GeneReconModel:
        species_raw = dict(self.species_raw)
        species_raw["unnorm_row_max"] = self.species_raw["unnorm_row_max"].to(
            config.precision.accumulator_torch_dtype
        )
        return GeneReconModel.from_preprocessed(
            species_raw,
            [self.families[i] for i in indices],
            mode=mode,
            solver_options=solver_options,
            config=config,
            fraction_missing=self.fraction_missing,
            species_name_to_index=self.species.leaf_index,
            per_family_origination=False,
            family_chunk_size=DEFAULT_FAMILY_CHUNK_SIZE,
            clade_budget=clade_budget,
            batch_packing=DEFAULT_BATCH_PACKING,
            max_wave_size=DEFAULT_MAX_WAVE_SIZE,
        )


def _family_entries(families) -> list[FamilyEntry]:
    if isinstance(families, (list, tuple)):
        entries = [FamilyEntry(Path(p).stem, Path(p), None) for p in families]
    else:
        path = Path(families)
        if path.is_dir():
            entries = [
                FamilyEntry(p.stem, p, None) for p in sorted(path.iterdir()) if p.suffix in _TREE_SUFFIXES
            ]
        else:
            entries = read_families_file(path)
    names = [entry.name for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"two families have the same name: {', '.join(duplicates)}")
    return entries


def _unreadable_file_problem(entry: FamilyEntry) -> str | None:
    for kind, path in (("gene tree", entry.gene_trees), ("mapping", entry.mapping)):
        if path is None:
            continue
        if not path.is_file():
            return f"cannot open the {kind} file {path}"
        if path.stat().st_size == 0:
            return f"the {kind} file {path} is empty"
    return None


def _keep(order, keep, reason, entries, dropped) -> list[int]:
    kept = []
    for i in order:
        if keep(i):
            kept.append(i)
        else:
            message = reason(i)
            logger.info("Excluding family %s: %s", entries[i].name, message)
            dropped.append((entries[i].name, message))
    return kept


def _fraction_missing(path: Path, species: SpeciesTree) -> dict[str, float]:
    values = read_fraction_missing_file(path)
    leaves = set(species.leaf_index)
    unknown = sorted(set(values) - leaves)
    missing = sorted(leaves - set(values))
    if unknown:
        raise ValueError(f"{path}: not leaves of the species tree: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{path} does not cover species: {', '.join(missing)}")
    bad = sorted(name for name, value in values.items() if not 0.0 <= value < 1.0)
    if bad:
        raise ValueError(f"{path}: fractions must be in [0, 1): {', '.join(bad)}")
    return values


def load_families(
    families,
    species_tree: Path,
    min_covered_species: int,
    trim_ratio: float,
    max_clade_split_ratio: float | None,
    fraction_missing: Path | None,
    skip_family_filtering: bool,
) -> FamilyData:
    """Read gene families and apply AleRax 1.4.0's family filtering, in AleRax's order."""
    if not 0.0 <= trim_ratio < 1.0:
        raise ValueError(f"trim_ratio must be at least 0 and less than 1, got {trim_ratio}")
    species_path = Path(species_tree)
    entries = _family_entries(families)
    dropped: list[tuple[str, str]] = []
    if not skip_family_filtering:
        entries = [entries[i] for i in _keep(
            range(len(entries)),
            lambda i: _unreadable_file_problem(entries[i]) is None,
            lambda i: _unreadable_file_problem(entries[i]),
            entries,
            dropped,
        )]
    if not entries:
        raise ValueError("no gene family to process")
    mappings = [None if entry.mapping is None else read_mapping_file(entry.mapping) for entry in entries]
    raw = preprocess_dataset(
        str(species_path),
        [str(entry.gene_trees) for entry in entries],
        gene_to_species=mappings,
        family_chunk_size=None,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=DEFAULT_MAX_WAVE_SIZE,
        family_group_assignments=None,
        accumulator_dtype=torch.float64,
    )
    species = species_tree_from_preprocess(raw["species"])
    payloads = raw["families"]
    covered = [
        frozenset(mapping.values()) if mapping is not None
        else frozenset(species.names[column] for column in payload["leaf_col_index"])
        for mapping, payload in zip(mappings, payloads)
    ]
    order = _keep(
        range(len(entries)),
        lambda i: len(covered[i]) >= min_covered_species,
        lambda i: f"covers {len(covered[i])} species, fewer than {min_covered_species}",
        entries,
        dropped,
    )
    if trim_ratio > 0.0:
        by_size = sorted(order, key=lambda i: int(payloads[i]["C"]))
        n_keep = math.floor(len(by_size) * (1.0 - trim_ratio))
        for i in by_size[n_keep:]:
            logger.info("Excluding family %s: trimmed by --trim-ratio %s", entries[i].name, trim_ratio)
            dropped.append((entries[i].name, f"trimmed by --trim-ratio {trim_ratio}"))
        order = by_size[:n_keep]
    if max_clade_split_ratio is not None:
        order = _keep(
            order,
            lambda i: max_clade_split_ratio * (2 * len(payloads[i]["leaf_row_index"]) - 3) >= int(payloads[i]["C"]),
            lambda i: f"has more clades than {max_clade_split_ratio} x (2 x genes - 3)",
            entries,
            dropped,
        )
    if not order:
        raise ValueError("no gene family is left after filtering")
    logger.info("Remaining families: %d", len(order))
    return FamilyData(
        species_tree_path=species_path,
        species=species,
        species_raw=raw["species"],
        families=tuple(payloads[i] for i in order),
        names=tuple(entries[i].name for i in order),
        covered_species=tuple(covered[i] for i in order),
        dropped=tuple(dropped),
        fraction_missing=None if fraction_missing is None else _fraction_missing(Path(fraction_missing), species),
    )
