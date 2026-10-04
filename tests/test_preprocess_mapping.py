import pytest
import torch

from gpurec.core.scheduling.batching import preprocess_dataset


def _run(tmp_path, gene_text, gene_to_species):
    species = tmp_path / "species.nwk"
    species.write_text("((A:1,B:2)AB:1,C:3)R;")
    gene = tmp_path / "gene.nwk"
    gene.write_text(gene_text)
    return preprocess_dataset(
        str(species),
        [str(gene)],
        gene_to_species=[gene_to_species],
        family_chunk_size=None,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8192,
        family_group_assignments=None,
        accumulator_dtype=torch.float64,
    )


def _gene_species(raw):
    names = raw["species"]["names"]
    family = raw["families"][0]
    return {gene: names[col] for gene, col in zip(family["leaf_gene_names"], family["leaf_col_index"])}


def test_default_rule_names_and_lengths(tmp_path):
    raw = _run(tmp_path, "((A_1,B_1),C_1);", None)
    names = raw["species"]["names"]
    assert sorted(names) == ["A", "AB", "B", "C", "R"]
    assert raw["species"]["lengths"][names.index("B")] == 2.0
    assert _gene_species(raw) == {"A_1": "A", "B_1": "B", "C_1": "C"}


def test_mapping_table(tmp_path):
    raw = _run(tmp_path, "((x,y),z);", {"x": "A", "y": "B", "z": "C"})
    assert _gene_species(raw) == {"x": "A", "y": "B", "z": "C"}


def test_gene_mapped_to_internal_node_is_rejected(tmp_path):
    with pytest.raises(Exception, match="not a leaf of the species tree"):
        _run(tmp_path, "((x,y),z);", {"x": "AB", "y": "B", "z": "C"})


def test_gene_missing_from_mapping_is_rejected(tmp_path):
    with pytest.raises(Exception, match="no entry in its mapping file"):
        _run(tmp_path, "((x,y),z);", {"x": "A", "y": "B"})
