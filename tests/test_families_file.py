from pathlib import Path

import pytest

from gpurec.io.families_file import (
    FamilyEntry,
    read_families_file,
    read_fraction_missing_file,
    read_mapping_file,
)


def test_both_tree_keys_mapping_and_ignored_keys(tmp_path):
    path = tmp_path / "families.txt"
    path.write_text(
        "ignored header line\n"
        "[FAMILIES]\n"
        "- fam1\n"
        "gene_tree = trees/fam1.nwk\n"
        "- fam2\n"
        "starting_gene_tree = fam2.ale\n"
        "mapping = maps/fam2.map\n"
        "alignment = fam2.fasta\n"
        "subst_model = GTR\n"
        "likelihoods = fam2.lk\n"
    )
    assert read_families_file(path) == [
        FamilyEntry("fam1", Path("trees/fam1.nwk"), None),
        FamilyEntry("fam2", Path("fam2.ale"), Path("maps/fam2.map")),
    ]


def test_crlf_comments_and_spaces(tmp_path):
    path = tmp_path / "families.txt"
    path.write_bytes(b"[FAMILIES]\r\n- fam 1 # first family\r\ngene_tree =  a.nwk # the tree\r\n\r\n")
    assert read_families_file(path) == [FamilyEntry("fam1", Path("a.nwk"), None)]


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / "families.txt"
    path.write_text("[FAMILIES]\n- fam1\ngene_tree=a.nwk\ncolour=blue\n")
    with pytest.raises(ValueError, match="line 4.*colour"):
        read_families_file(path)


def test_family_without_gene_tree_is_an_error(tmp_path):
    path = tmp_path / "families.txt"
    path.write_text("[FAMILIES]\n- fam1\nmapping=m.txt\n")
    with pytest.raises(ValueError, match="fam1.*gene tree"):
        read_families_file(path)


def test_duplicate_family_name_is_an_error(tmp_path):
    path = tmp_path / "families.txt"
    path.write_text("[FAMILIES]\n- fam1\ngene_tree=a.nwk\n- fam1\ngene_tree=b.nwk\n")
    with pytest.raises(ValueError, match="fam1.*twice"):
        read_families_file(path)


def test_mapping_phyldog_and_treerecs(tmp_path):
    phyldog = tmp_path / "p.map"
    phyldog.write_text("A:a1;a2\nB:b1\n")
    treerecs = tmp_path / "t.map"
    treerecs.write_text("a1 A\n\na2 A\nb1 B\n")
    expected = {"a1": "A", "a2": "A", "b1": "B"}
    assert read_mapping_file(phyldog) == expected
    assert read_mapping_file(treerecs) == expected


def test_mapping_last_line_wins(tmp_path):
    path = tmp_path / "t.map"
    path.write_text("a1 A\na1 B\n")
    assert read_mapping_file(path) == {"a1": "B"}


def test_fraction_missing(tmp_path):
    path = tmp_path / "fm.txt"
    path.write_text("A 0.1\nB 0.25\n")
    assert read_fraction_missing_file(path) == {"A": 0.1, "B": 0.25}


def test_empty_value_is_an_error(tmp_path):
    path = tmp_path / "families.txt"
    path.write_text("[FAMILIES]\n- fam1\ngene_tree=\n")
    with pytest.raises(ValueError, match="gene_tree.*has no value"):
        read_families_file(path)

    path2 = tmp_path / "families2.txt"
    path2.write_text("[FAMILIES]\n- fam1\ngene_tree=a.nwk\nmapping=\n")
    with pytest.raises(ValueError, match="mapping.*has no value"):
        read_families_file(path2)


@pytest.mark.parametrize("text, bad_line", [
    ("A:a1;a2\n\nb1 B\n", 3),   # Phyldog format (first line has ':'), a later line without ':'
    ("a1 A\n\nb1\n", 3),        # Treerecs format, a later line with one word
])
def test_malformed_mapping_line_names_the_file_and_line(tmp_path, text, bad_line):
    path = tmp_path / "m.map"
    path.write_text(text)
    with pytest.raises(ValueError, match=f"{path}, line {bad_line}: "):
        read_mapping_file(path)
