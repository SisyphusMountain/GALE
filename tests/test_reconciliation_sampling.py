from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu
RELEASE = Path(__file__).parent / "data" / "release"


@pytest.fixture
def result(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec import evaluate, load_families, load_fit_config

    monkeypatch.chdir(RELEASE)
    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    return evaluate(data, parametrization="global", rates=(0.1, 0.1, 0.1), config=load_fit_config(None))


def test_every_gene_appears_once_in_its_species(result):
    samples = result.sample_reconciliations(n=5, seed=3)
    assert list(samples) == list(result.data.names)
    species_names = result.data.species.names
    for name, trees in samples.items():
        family = result.data.families[result.data.names.index(name)]
        gene_species = dict(zip(family["leaf_gene_names"], family["leaf_col_index"]))
        assert len(trees) == 5
        for tree in trees:
            assert sorted(tree.genes()) == sorted(family["leaf_gene_names"])
            leaves = [e for e in tree.events() if e.kind == "Leaf"]
            assert len(leaves) == len(gene_species)
            for event in tree.events():
                assert 0 <= event.species < len(species_names)


def test_samples_are_reproducible(result):
    assert result.sample_reconciliations(n=3, seed=7) == result.sample_reconciliations(n=3, seed=7)


def test_seed_depends_on_family_and_index():
    from gpurec.reconciliation.sampling import sample_seed

    seeds = {sample_seed(1, family, i) for family in ("a", "b") for i in range(3)}
    assert len(seeds) == 6


def test_family_by_family_samples_equal_the_whole_dictionary(result):
    import inspect

    from gpurec.reconciliation.sampling import iter_family_reconciliations

    families = iter_family_reconciliations(result.model, result.data, 3, 11)
    assert inspect.isgenerator(families)
    streamed = dict(families)
    assert sorted(streamed) == sorted(result.data.names)
    assert streamed == result.sample_reconciliations(n=3, seed=11)


def test_recphyloxml_events_match_the_event_counts_of_the_same_sample(monkeypatch):
    """At rates high enough for every event kind, each sample's RecPhyloXML elements agree with its
    eventCounts file: speciation = S + SL, duplication = D, branchingOut = T + TL, loss = SL + TL,
    leaf = Leaf; and every gene is exactly one leaf."""
    import xml.etree.ElementTree as ET

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec import evaluate, load_families, load_fit_config
    from gpurec.reconciliation.files import event_counts_text, recphyloxml_text

    monkeypatch.chdir(RELEASE)
    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    result = evaluate(data, parametrization="global", rates=(0.3, 0.3, 0.3), config=load_fit_config(None))
    namespace = "{http://www.recg.org}"
    seen = {"D": 0, "T": 0, "SL": 0, "TL": 0}
    for name, trees in result.sample_reconciliations(n=20, seed=1).items():
        family = data.families[data.names.index(name)]
        assert len(trees) == 20
        for tree in trees:
            counts = {}
            for line in event_counts_text(tree).splitlines():
                kind, value = line.split(":")
                counts[kind] = int(value)
            for kind in seen:
                seen[kind] += counts[kind]
            gene_tree = ET.fromstring(recphyloxml_text(tree, data.species)).find(f"{namespace}recGeneTree")
            elements = {
                tag: len(list(gene_tree.iter(namespace + tag)))
                for tag in ("speciation", "duplication", "branchingOut", "leaf", "loss")
            }
            assert elements == {
                "speciation": counts["S"] + counts["SL"],
                "duplication": counts["D"],
                "branchingOut": counts["T"] + counts["TL"],
                "leaf": counts["Leaf"],
                "loss": counts["SL"] + counts["TL"],
            }
            leaf_names = [
                clade.find(f"{namespace}name").text
                for clade in gene_tree.iter(f"{namespace}clade")
                if clade.find(f"{namespace}eventsRec/{namespace}leaf") is not None
            ]
            assert sorted(leaf_names) == sorted(family["leaf_gene_names"])
    assert all(count > 0 for count in seen.values()), seen
