from pathlib import Path

from gpurec.data import FamilyData, load_families


def family_data(species_tree: Path, gene_trees: list[Path]) -> FamilyData:
    """All given gene-tree files as families, no filtering (family names are the file stems)."""
    return load_families(
        families=[Path(p) for p in gene_trees],
        species_tree=Path(species_tree),
        min_covered_species=1,
        trim_ratio=0.0,
        max_clade_split_ratio=None,
        fraction_missing=None,
        skip_family_filtering=True,
    )
