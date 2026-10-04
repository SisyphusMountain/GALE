from pathlib import Path

import pytest

from gpurec.io.species_tree import (
    alerax_node_order,
    species_tree_from_arrays,
    species_tree_from_preprocess,
    species_tree_newick,
)

DATA = Path(__file__).parent / "data" / "species_labels"


def _small():
    # ((A,B)AB,C)R in post-order: A=0, B=1, AB=2, C=3, R=4
    return species_tree_from_arrays(
        ["A", "B", "AB", "C", "R"], [1.0, 2.0, 1.0, 3.0, 0.0], [-1, -1, 0, -1, 2], [-1, -1, 1, -1, 3]
    )


def test_structure():
    tree = _small()
    assert tree.root == 4
    assert tree.parent == (2, 2, 4, 4, -1)
    assert tree.labels == ("A", "B", "AB", "C", "R")
    assert tree.leaf_index == {"A": 0, "B": 1, "C": 3}


def test_alerax_node_order_tips_then_preorder_internals_then_root():
    assert alerax_node_order(_small()) == [0, 1, 3, 2, 4]


def test_newick_format():
    tree = _small()
    text = species_tree_newick(tree, [1.0, 2.0, 1.0, 3.0, 0.5])
    assert text == "((A:1.000000,B:2.000000)AB:1.000000,C:3.000000)R:0.500000;\n"


def _preprocessed(tmp_path):
    import torch

    from gpurec.core.scheduling.batching import preprocess_dataset

    raw = preprocess_dataset(
        str(DATA / "species.nwk"),
        [str(DATA / "gene.nwk")],
        gene_to_species=[None],
        family_chunk_size=None,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8192,
        family_group_assignments=None,
        accumulator_dtype=torch.float64,
    )
    return species_tree_from_preprocess(raw["species"])


def test_bootstrap_and_duplicate_internal_labels_are_renamed(tmp_path):
    tree = _preprocessed(tmp_path)
    lengths = [length if length > 0 else 1.0 for length in tree.lengths]
    expected = (DATA / "alerax_starting_species_tree.newick").read_text()
    assert species_tree_newick(tree, lengths) == expected


def test_model_parameter_rows_follow_alerax_order(tmp_path):
    tree = _preprocessed(tmp_path)
    rows = (DATA / "alerax_model_parameters.txt").read_text().splitlines()[1:]
    assert [tree.labels[i] for i in alerax_node_order(tree)] == [row.split()[0] for row in rows]
