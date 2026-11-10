"""Rate-class files (AleRax's --model-parametrization <file>): parsing and the classes it assigns."""
from pathlib import Path

import pytest

from gpurec.cli.arguments import UsageError, parse_arguments
from gpurec.io.rate_classes import parse_rate_classes
from gpurec.io.species_tree import species_tree_from_arrays


def _tree():
    # ((A,B)AB,(C,D)CD)R, nodes 0..6
    names = ["A", "B", "AB", "C", "D", "CD", "R"]
    child1 = [-1, -1, 0, -1, -1, 3, 2]
    child2 = [-1, -1, 1, -1, -1, 4, 5]
    return species_tree_from_arrays(names, [1.0] * 7, child1, child2)


def _by_label(tree, classes, k):
    return {tree.labels[node]: classes.class_of[k][node] for node in range(len(tree.labels))}


def test_a_line_starts_a_class_its_subtree_inherits():
    tree = _tree()
    classes = parse_rate_classes("AB D\n", tree)
    assert classes.counts == (2, 1, 1)
    assert _by_label(tree, classes, 0) == {"A": 1, "B": 1, "AB": 1, "C": 0, "D": 0, "CD": 0, "R": 0}
    assert set(_by_label(tree, classes, 1).values()) == {0} and set(_by_label(tree, classes, 2).values()) == {0}
    assert classes.starts[0] == ("R", "AB")


def test_rates_are_independent_and_a_descendant_starts_its_own_class():
    tree = _tree()
    classes = parse_rate_classes("AB DLT\nA D\nCD T\n", tree)
    assert classes.counts == (3, 2, 3)
    assert _by_label(tree, classes, 0) == {"A": 2, "B": 1, "AB": 1, "C": 0, "D": 0, "CD": 0, "R": 0}
    assert _by_label(tree, classes, 1) == {"A": 1, "B": 1, "AB": 1, "C": 0, "D": 0, "CD": 0, "R": 0}
    assert _by_label(tree, classes, 2) == {"A": 1, "B": 1, "AB": 1, "C": 2, "D": 2, "CD": 2, "R": 0}


def test_root_line_is_skipped_and_blank_lines_ignored():
    classes = parse_rate_classes("\nR DLT\n\n", _tree())
    assert classes.counts == (1, 1, 1)
    assert all(set(column) == {0} for column in classes.class_of)


@pytest.mark.parametrize("text, message", [
    ("X D\n", "not a node"),
    ("AB DQ\n", "unknown rate types"),
    ("AB DO\n", "origination classes"),
    ("AB D\nAB L\n", "appears twice"),
    ("AB\n", "expected"),
    ("AB D L\n", "expected"),
])
def test_errors_name_the_line(text, message):
    with pytest.raises(ValueError, match=message):
        parse_rate_classes(text, _tree(), source="classes.txt")


def test_cli_takes_a_rate_class_file(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("AB DLT\n")
    args = parse_arguments(["-f", "f.txt", "-s", "s.nwk", "--model-parametrization", str(path)])
    assert args.parametrization == "cladewise" and args.rate_classes_file == path


def test_cli_rejects_a_missing_rate_class_file(tmp_path):
    with pytest.raises(UsageError, match="existing rate-class file"):
        parse_arguments(["-f", "f.txt", "-s", "s.nwk", "--model-parametrization", str(tmp_path / "none.txt")])


def test_every_node_of_a_class_gets_its_class_rate(tmp_path, monkeypatch):
    # Three D classes one ulp apart on the release tree, as the class fit left them on saion (job
    # 4806885): torch.exp2 over the [S,3] tensor wrote the root one ulp off the rest of its class.
    import math

    import torch

    from gpurec import load_families, write_alerax
    from gpurec.fit.cladewise_fit import species_log2_rates
    from gpurec.interface import FitResult

    monkeypatch.chdir(Path(__file__).parent / "data" / "release")
    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    classes = parse_rate_classes("Node_5_13_0 D\nNode_5_11_0 D\n", data.species)
    d = math.log2(1.9501728222660242e-05)
    up = math.nextafter(d, 0.0)
    class_log2 = [torch.tensor(v, dtype=torch.float64) for v in ([d, up, up], [math.log2(2.07e-05)], [math.log2(0.0207)])]
    result = FitResult(data, "cladewise", species_log2_rates(classes, class_log2),
                       {name: -1.0 for name in data.names}, None, None, 0.0, classes)
    write_alerax(result, None, tmp_path)
    rows = (tmp_path / "model_parameters" / "model_parameters.txt").read_text().splitlines()[1:]
    written = {row.split()[0]: row.split()[1:] for row in rows}
    for kind, name in enumerate("DLT"):
        per_class_file, per_class_api = {}, {}
        for node, label in enumerate(data.species.labels):
            per_class_file.setdefault(classes.class_of[kind][node], set()).add(written[label][kind])
            per_class_api.setdefault(classes.class_of[kind][node], set()).add(result.rates[label][name])
        assert all(len(values) == 1 for values in per_class_file.values()), name
        assert all(len(values) == 1 for values in per_class_api.values()), name
