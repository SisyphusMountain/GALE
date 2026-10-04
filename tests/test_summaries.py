from gpurec.io.species_tree import species_tree_from_arrays
from gpurec.reconciliation.files import species_event_counts
from gpurec.reconciliation.summaries import (
    mean_species_event_counts,
    mean_transfers,
    origin_texts,
    transfers_table_text,
)
from gpurec.reconciliation.tree import tree_from_sampler_nodes

A, B, AB, C, R = 0, 1, 2, 3, 4
SPECIES = species_tree_from_arrays(["A", "B", "AB", "C", "R"], [1.0] * 5, [-1, -1, 0, -1, 2], [-1, -1, 1, -1, 3])
SPECIATION = tree_from_sampler_nodes(
    [("speciation", R, [1, 2], 10), ("speciation", AB, [3, 4], 5), ("leaf", C, [-1, -1], 7),
     ("leaf", A, [-1, -1], 5), ("loss", B, [-1, -1], -1)],
    {5: "a1", 7: "c1"},
)
TRANSFER = tree_from_sampler_nodes(
    [("transfer", A, [1, 2], 9), ("leaf", C, [-1, -1], 4), ("leaf", A, [-1, -1], 3)], {3: "a1", 4: "c1"}
)


def test_means_over_samples():
    per_sample = [species_event_counts(tree, SPECIES) for tree in (SPECIATION, TRANSFER)]
    mean = mean_species_event_counts(per_sample, 5)
    assert mean[A] == [1.0, 0.0, 0.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.0]
    assert mean_transfers([SPECIATION, TRANSFER]) == {(A, C): 0.5}
    assert transfers_table_text({(A, C): 0.5, (C, A): 2.0}, SPECIES) == "C A 2\nA C 0.5\n"


def test_origins():
    texts = origin_texts({"fam": [SPECIATION, TRANSFER]}, SPECIES, 2)
    assert texts[C] == "vertical: 0.5\nA, 0.5\n"
    assert texts[B] == "vertical: 0.5\n"
    assert texts[R] == "vertical: 0\n"
    assert set(texts) == {A, B, AB, C, R}
