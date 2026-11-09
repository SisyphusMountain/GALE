"""The probability of the observed genes with a fraction-missing file: GALE's reported log-likelihood
includes, for every gene of a family, the factor 1 - m of its leaf species (m the fraction missing),
as AleRax 1.5.0 does (src/ale/UndatedDTLMultiModel.hpp, commit 08ffe5f).

Reference values: AleRax 1.5.0 (release/1.5.0 625a3b5, GeneRaxCore 04ec9ab), 16 fixed-point passes,
on the species tree ((A,B),(C,D)) with m(C) = 0.3 and m(D) = 0.2, rates D = 0.2, L = 0.3, T = 0.1,
PER-FAMILY with fixed rates, no gene-tree samples (run on 9 Nov 2026). AleRax prints six
significant digits, hence the tolerance. Without the file the two programs give the same values; with
it, the gap the factor closes is -ln(0.7) for the gene at C of f2 and -ln(0.7) - ln(0.8) for the genes at
C and D of f3.
"""
from __future__ import annotations

import pytest

from gpurec.cli.main import main

pytestmark = pytest.mark.gpu

SPECIES_TREE = "((A:1,B:1):1,(C:1,D:1):1);\n"
GENE_TREES = {
    "f1": "(A_1:1,B_1:1,A_2:1);\n",
    "f2": "(C_1:1,A_1:1,B_1:1);\n",
    "f3": "(C_1:1,D_1:1,A_1:1);\n",
}
FRACTION_MISSING = "A 0\nB 0\nC 0.3\nD 0.2\n"
# AleRax 1.5.0's per_fam_likelihoods (nats).
ALERAX_WITH_FILE = {"f1": -4.89291, "f2": -5.83995, "f3": -6.28785}
ALERAX_WITHOUT_FILE = {"f1": -5.07795, "f2": -6.03694, "f3": -6.03694}
# Half a unit of AleRax's sixth significant digit, and some room for its 16 passes.
TOLERANCE = 2e-5


def _run(tmp_path, with_file: bool) -> dict[str, float]:
    (tmp_path / "sp.nw").write_text(SPECIES_TREE)
    lines = ["[FAMILIES]"]
    for name, newick in GENE_TREES.items():
        (tmp_path / f"{name}.nw").write_text(newick)
        lines += [f"- {name}", f"gene_tree = {tmp_path / f'{name}.nw'}"]
    (tmp_path / "families.txt").write_text("\n".join(lines) + "\n")
    (tmp_path / "fm.txt").write_text(FRACTION_MISSING)
    prefix = tmp_path / ("with_file" if with_file else "without_file")
    arguments = ["-f", str(tmp_path / "families.txt"), "-s", str(tmp_path / "sp.nw"), "-p", str(prefix),
                 "--model-parametrization", "PER-FAMILY", "--fix-rates", "--d", "0.2", "--l", "0.3",
                 "--t", "0.1", "--gene-tree-samples", "0", "--min-covered-species", "1"]
    if with_file:
        arguments += ["--fraction-missing-file", str(tmp_path / "fm.txt")]
    assert main(arguments) == 0
    values = {}
    for line in (prefix / "per_fam_likelihoods.txt").read_text().splitlines():
        name, value = line.split()
        values[name] = float(value)
    return values


def test_without_fraction_missing_matches_alerax_150(tmp_path):
    values = _run(tmp_path, with_file=False)
    for name, expected in ALERAX_WITHOUT_FILE.items():
        assert values[name] == pytest.approx(expected, abs=TOLERANCE), name


def test_with_fraction_missing_matches_alerax_150(tmp_path):
    values = _run(tmp_path, with_file=True)
    for name, expected in ALERAX_WITH_FILE.items():
        assert values[name] == pytest.approx(expected, abs=TOLERANCE), name
