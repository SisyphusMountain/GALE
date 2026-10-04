from pathlib import Path

import pytest
import torch

from gpurec.data import load_families
from gpurec.interface import FitResult
from gpurec.reconciliation.tree import tree_from_sampler_nodes

RELEASE = Path(__file__).parent / "data" / "release"


def _result(monkeypatch, parametrization):
    monkeypatch.chdir(RELEASE)
    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    log2 = torch.log2(torch.tensor([[0.5, 0.25, 0.125]], dtype=torch.float64)).expand(3, 3).clone()
    return FitResult(data, parametrization, log2, {"fam1": -10.5, "fam2": -20.25, "fam3": -10.5}, None, None, 0.0)


def test_global_files(tmp_path, monkeypatch):
    from gpurec.io.alerax_output import write_alerax

    write_alerax(_result(monkeypatch, "global"), None, tmp_path)
    assert (tmp_path / "per_fam_likelihoods.txt").read_text() == "fam2 -20.25\nfam1 -10.5\nfam3 -10.5\n"
    rows = (tmp_path / "model_parameters" / "model_parameters.txt").read_text().splitlines()
    assert rows[0] == "node D L T" and len(rows) == 1 + 15
    assert all(row.split()[1:] == ["0.5", "0.25", "0.125"] for row in rows[1:])
    assert (tmp_path / "species_trees" / "starting_species_tree.newick").exists()
    inferred = (tmp_path / "species_trees" / "inferred_species_tree.newick").read_text()
    assert ":1.000000" in inferred and inferred.endswith(";\n")
    coverage = (tmp_path / "perSpeciesCoverage.txt").read_text().splitlines()
    assert len(coverage) == 8
    assert not (tmp_path / "reconciliations").exists()


def test_per_family_rate_files(tmp_path, monkeypatch):
    from gpurec.io.alerax_output import write_alerax

    write_alerax(_result(monkeypatch, "per-family"), None, tmp_path)
    assert (tmp_path / "model_parameters" / "fam1_rates.txt").read_text() == "D L T\n0.5 0.25 0.125\n"
    assert not (tmp_path / "model_parameters" / "model_parameters.txt").exists()


def test_reconciliation_files(tmp_path, monkeypatch):
    from gpurec.io.alerax_output import write_alerax

    monkeypatch.chdir(RELEASE)
    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    species = data.species
    r = species.root
    c1 = species.child1[r]
    c2 = species.child2[r]
    tree = tree_from_sampler_nodes(
        [("speciation", r, [1, 2], 0), ("leaf", c1, [-1, -1], 1), ("leaf", c2, [-1, -1], 2)],
        {1: "g1", 2: "g2"}
    )
    samples = {"fam1": [tree, tree], "fam2": [tree, tree], "fam3": [tree, tree]}

    log2 = torch.log2(torch.tensor([[0.5, 0.25, 0.125]], dtype=torch.float64)).expand(3, 3).clone()
    result = FitResult(data, "global", log2, {"fam1": -10.5, "fam2": -20.25, "fam3": -10.5}, None, None, 0.0)

    write_alerax(result, samples, tmp_path)

    # Check that all expected reconciliation files exist
    for f in ["fam1", "fam2", "fam3"]:
        assert (tmp_path / "reconciliations" / "all" / f"{f}_sample_0.xml").exists()
        assert (tmp_path / "reconciliations" / "all" / f"{f}_sample_1.xml").exists()
        assert (tmp_path / "reconciliations" / "all" / f"{f}_eventCounts_1.txt").exists()
        assert (tmp_path / "reconciliations" / "all" / f"{f}_speciesEventCounts_1.txt").exists()
        assert (tmp_path / "reconciliations" / "all" / f"{f}_transfers_1.txt").exists()
        samples_newick = (tmp_path / "reconciliations" / "all" / f"{f}_samples.newick").read_text()
        assert len(samples_newick.splitlines()) == 2
        assert (tmp_path / "reconciliations" / "summaries" / f"{f}_meanSpeciesEventCounts.txt").exists()
        assert (tmp_path / "reconciliations" / "summaries" / f"{f}_meanTransfers.txt").exists()

    assert (tmp_path / "reconciliations" / "totalSpeciesEventCounts.txt").exists()
    assert (tmp_path / "reconciliations" / "totalTransfers.txt").exists()

    # Check origins folder has one file per species node
    origins_dir = tmp_path / "reconciliations" / "origins"
    origin_files = list(origins_dir.glob("*.txt"))
    assert len(origin_files) == len(species.labels)

    # Check specific origin text for c1 (should have vertical arrivals)
    c1_label = species.labels[c1]
    c1_origins = (origins_dir / f"{c1_label}.txt").read_text()
    assert c1_origins.startswith("vertical: 3\n")


def test_zero_samples_writes_no_reconciliations(tmp_path, monkeypatch):
    from gpurec.io.alerax_output import write_alerax

    result = _result(monkeypatch, "global")
    samples = {"fam1": [], "fam2": [], "fam3": []}
    write_alerax(result, samples, tmp_path)

    # Should not crash and should not write reconciliations
    assert not (tmp_path / "reconciliations").exists()
    assert (tmp_path / "per_fam_likelihoods.txt").exists()


def test_unequal_sample_counts_are_rejected(tmp_path, monkeypatch):
    from gpurec.io.alerax_output import write_alerax

    monkeypatch.chdir(RELEASE)
    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    species = data.species
    r = species.root
    c1 = species.child1[r]
    c2 = species.child2[r]
    tree = tree_from_sampler_nodes(
        [("speciation", r, [1, 2], 0), ("leaf", c1, [-1, -1], 1), ("leaf", c2, [-1, -1], 2)],
        {1: "g1", 2: "g2"}
    )

    # Create samples with different counts
    samples = {"fam1": [tree, tree], "fam2": [tree], "fam3": [tree, tree]}

    log2 = torch.log2(torch.tensor([[0.5, 0.25, 0.125]], dtype=torch.float64)).expand(3, 3).clone()
    result = FitResult(data, "global", log2, {"fam1": -10.5, "fam2": -20.25, "fam3": -10.5}, None, None, 0.0)

    with pytest.raises(ValueError, match="every family must have the same number of reconciliation samples"):
        write_alerax(result, samples, tmp_path)


def _mixed_samples(species):
    """Three samples per family mixing speciation, speciation-loss, transfer and duplication."""
    r = species.root
    c1, c2 = species.child1[r], species.child2[r]
    genes = {1: "g1", 2: "g2"}
    s = tree_from_sampler_nodes(
        [("speciation", r, [1, 2], 0), ("leaf", c1, [-1, -1], 1), ("leaf", c2, [-1, -1], 2)], genes)
    t = tree_from_sampler_nodes(
        [("transfer", c1, [1, 2], 0), ("leaf", c2, [-1, -1], 1), ("leaf", c1, [-1, -1], 2)], genes)
    d = tree_from_sampler_nodes(
        [("duplication", c1, [1, 2], 0), ("leaf", c1, [-1, -1], 1), ("leaf", c1, [-1, -1], 2)], genes)
    sl = tree_from_sampler_nodes(
        [("speciation", r, [1, 2], 0), ("leaf", c1, [-1, -1], 1), ("loss", c2, [-1, -1], -1)], genes)
    return {"fam1": [s, t, d], "fam2": [t, t, sl], "fam3": [d, s, sl]}


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def test_family_by_family_writer_gives_the_same_files_as_write_alerax(tmp_path, monkeypatch):
    from gpurec.io.alerax_output import ReconciliationWriter, write_alerax

    result = _result(monkeypatch, "global")
    samples = _mixed_samples(result.data.species)
    write_alerax(result, samples, tmp_path / "whole")
    write_alerax(result, None, tmp_path / "streamed")
    writer = ReconciliationWriter(result.data.species, tmp_path / "streamed" / "reconciliations", 3)
    for name, trees in samples.items():
        writer.add_family(name, trees)
    writer.finish()
    whole, streamed = _files(tmp_path / "whole"), _files(tmp_path / "streamed")
    assert "reconciliations/totalTransfers.txt" in whole and whole["reconciliations/totalTransfers.txt"]
    assert streamed == whole


def test_writer_refuses_a_family_with_another_sample_count(tmp_path, monkeypatch):
    from gpurec.io.alerax_output import ReconciliationWriter

    result = _result(monkeypatch, "global")
    samples = _mixed_samples(result.data.species)
    writer = ReconciliationWriter(result.data.species, tmp_path / "reconciliations", 3)
    with pytest.raises(ValueError, match="fam1 has 2 reconciliation samples, expected 3"):
        writer.add_family("fam1", samples["fam1"][:2])
