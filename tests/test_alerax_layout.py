import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu
RELEASE = Path(__file__).parent / "data" / "release"
REFERENCE = RELEASE / "alerax_reference"
NOT_WRITTEN = re.compile(r"^(ccps/|checkpoint/|alerax\.log|perSpeciesMissing\.txt|.*\.alerec$|.*consensus_50\.newick$)")

# alerax_reference was NOT produced by running AleRax 1.4.0 on the full
# tests/data/release/families.txt (which lists fam1..fam4). AleRax 1.4.0
# aborts ("terminate called after throwing an instance of
# 'std::runtime_error' what(): There was a problem with filling the nodes
# array") while generating ccp files for fam4 (a 2-leaf gene tree,
# "(5_1:1,11_1:1);"), before it ever reaches its own --min-covered-species
# family filter (confirmed: the crash still happens with
# --min-covered-species 4 passed explicitly, and running fam4 alone also
# crashes). Since fam4 covers only 2 species, fewer than the shared default
# of 4, gpurec's own coverage filter also drops it, so the reference here was
# produced from a families.txt listing only fam1, fam2 and fam3 (identical
# gene trees/mapping to the ones in this directory). This means gpurec is
# still exercised on the full 4-family input (fam4.nwk is copied into the
# `ours` fixture below and referenced by families.txt), and its filter is
# expected to drop fam4 the same way, leaving the same 3-family output.


@pytest.fixture(scope="module")
def ours(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.cli.main import main

    work = tmp_path_factory.mktemp("layout")
    for name in ("species.nwk", "families.txt", "fam1.nwk", "fam2.nwk", "fam3.nwk", "fam3.map", "fam4.nwk"):
        shutil.copy(RELEASE / name, work / name)
    import os

    previous = os.getcwd()
    os.chdir(work)
    try:
        assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "2", "--seed", "1"]) == 0
    finally:
        os.chdir(previous)
    return work / "out"


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_same_file_names(ours):
    theirs = {line for line in (REFERENCE / "ALL_FILES.txt").read_text().split() if not NOT_WRITTEN.match(line)}
    assert _files(ours) - {"gpurec.log"} == theirs


def test_species_trees_are_identical(ours):
    for name in ("starting_species_tree.newick", "inferred_species_tree.newick"):
        assert (ours / "species_trees" / name).read_text() == (REFERENCE / "species_trees" / name).read_text()


def test_model_parameters_rows(ours):
    mine = (ours / "model_parameters" / "model_parameters.txt").read_text().splitlines()
    theirs = (REFERENCE / "model_parameters" / "model_parameters.txt").read_text().splitlines()
    assert mine[0] == theirs[0]
    assert [row.split()[0] for row in mine[1:]] == [row.split()[0] for row in theirs[1:]]


def test_likelihood_file_layout(ours):
    mine = [line.split() for line in (ours / "per_fam_likelihoods.txt").read_text().splitlines()]
    theirs = [line.split() for line in (REFERENCE / "per_fam_likelihoods.txt").read_text().splitlines()]
    assert sorted(row[0] for row in mine) == sorted(row[0] for row in theirs)
    assert all(len(row) == 2 for row in mine)


def test_coverage_is_identical(ours):
    def parse(path):
        return [(line.split()[0], float(line.split()[1])) for line in path.read_text().splitlines()]

    assert parse(ours / "perSpeciesCoverage.txt") == pytest.approx(parse(REFERENCE / "perSpeciesCoverage.txt"))


def test_recphyloxml_structure(ours):
    mine = (ours / "reconciliations/all/fam1_sample_0.xml").read_text()
    theirs = (REFERENCE / "reconciliations/all/fam1_sample_0.xml").read_text()
    assert mine.split("<recGeneTree>")[0] == theirs.split("<recGeneTree>")[0]
    vocabulary = {element.tag for element in ET.fromstring(theirs).iter()} | {
        "{http://www.recg.org}" + tag for tag in ("duplication", "branchingOut", "transferBack", "loss", "leaf", "speciation")
    }
    assert {element.tag for element in ET.fromstring(mine).iter()} <= vocabulary


def test_count_file_layouts(ours):
    all_dir, reference_all = ours / "reconciliations/all", REFERENCE / "reconciliations/all"
    assert [l.split(":")[0] for l in (all_dir / "fam1_eventCounts_0.txt").read_text().splitlines()] == [
        l.split(":")[0] for l in (reference_all / "fam1_eventCounts_0.txt").read_text().splitlines()
    ]
    for relative in ("all/fam1_speciesEventCounts_0.txt", "summaries/fam1_meanSpeciesEventCounts.txt",
                     "totalSpeciesEventCounts.txt"):
        mine = (ours / "reconciliations" / relative).read_text().splitlines()
        theirs = (REFERENCE / "reconciliations" / relative).read_text().splitlines()
        assert mine[0] == theirs[0]
        assert all(len(row.split(", ")) == 10 for row in mine[1:])
    for relative in ("all/fam1_transfers_0.txt", "summaries/fam1_meanTransfers.txt", "totalTransfers.txt"):
        assert all(len(row.split()) == 3 for row in (ours / "reconciliations" / relative).read_text().splitlines())
    for path in (ours / "reconciliations/origins").iterdir():
        assert path.read_text().startswith("vertical: ")
