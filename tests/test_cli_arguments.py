from pathlib import Path

import pytest

from gpurec.cli.arguments import UsageError, parse_arguments

BASE = ["-f", "families.txt", "-s", "species.nwk"]


def test_alerax_defaults():
    args = parse_arguments(BASE)
    assert args.families == Path("families.txt") and args.species_tree == Path("species.nwk")
    assert args.prefix == Path("alerax_output")
    assert args.parametrization == "global"
    assert args.start_rates == (0.1, 0.1, 0.1)
    assert (args.fix_rates, args.samples, args.seed) == (False, 100, 123)
    assert (args.min_covered_species, args.trim_ratio, args.max_clade_split_ratio) == (4, 0.0, None)
    assert (args.precision, args.config, args.fraction_missing_file) == (None, None, None)


def test_long_names_and_values():
    args = parse_arguments([
        "--families", "f.txt", "--species-tree", "s.nwk", "--prefix", "out",
        "--model-parametrization", "PER-FAMILY", "--d", "0.2", "--l", "0.3", "--t", "0.05",
        "--fix-rates", "--gene-tree-samples", "0", "--seed", "9", "--min-covered-species", "1",
        "--trim-ratio", "0.1", "--max-clade-split-ratio", "2.5", "--skip-family-filtering",
        "--rec-model", "UndatedDTL", "--transfer-constraint", "PARENTS", "--verbose-opt",
        "--precision", "float64", "--config", "c.toml", "--fraction-missing-file", "fm.txt",
    ])
    assert args.parametrization == "per-family" and args.start_rates == (0.2, 0.3, 0.05)
    assert args.fix_rates and args.samples == 0 and args.seed == 9
    assert args.max_clade_split_ratio == 2.5 and args.precision == "float64"


def test_every_problem_is_reported_at_once():
    with pytest.raises(UsageError) as error:
        parse_arguments(BASE + [
            "--rate-optimizer", "LBFGSB", "--highways", "--rec-model", "UndatedDL",
            "--model-parametrization", "PER-SPECIES", "--d", "0",
        ])
    message = str(error.value)
    for text in ("--rate-optimizer", "--highways", "UndatedDL", "PER-SPECIES", "--d"):
        assert text in message


def test_named_species_tree_and_missing_required():
    with pytest.raises(UsageError, match="MiniNJ"):
        parse_arguments(["-f", "f.txt", "-s", "MiniNJ"])
    with pytest.raises(UsageError, match="--families"):
        parse_arguments(["-s", "s.nwk"])


def test_short_rec_model_spelling_as_in_alerax():
    parse_arguments(BASE + ["-r", "UndatedDTL"])
    with pytest.raises(UsageError, match="--rec-model UndatedDL is not supported"):
        parse_arguments(BASE + ["-r", "UndatedDL"])


def test_stray_words_are_reported():
    with pytest.raises(UsageError, match="unexpected argument\\(s\\): stray.txt"):
        parse_arguments(BASE + ["stray.txt"])


def test_value_of_an_unsupported_option_is_not_called_a_stray_word():
    with pytest.raises(UsageError) as error:
        parse_arguments(BASE + ["--rate-optimizer", "LBFGSB"])
    assert "--rate-optimizer" in str(error.value) and "unexpected argument" not in str(error.value)


@pytest.mark.parametrize("option, pointer", [
    ("--per-family-rates", "use --model-parametrization PER-FAMILY"),
    ("--per-species-rates", "per-species rates are not supported"),
    ("--verbose-opt-rates", "use --verbose-opt"),
    ("--gene-tree-sample-frequency", "--ccp-sample-frequency"),
])
def test_deprecated_alerax_spellings_point_to_the_current_one(option, pointer):
    with pytest.raises(UsageError) as error:
        parse_arguments(BASE + [option])
    assert option in str(error.value) and pointer in str(error.value)


@pytest.mark.parametrize("ratio", ["-0.1", "1", "1.5"])
def test_trim_ratio_outside_zero_one_is_refused(ratio):
    with pytest.raises(UsageError, match="--trim-ratio"):
        parse_arguments(BASE + ["--trim-ratio", ratio])


def test_trim_ratio_inside_zero_one_is_accepted():
    assert parse_arguments(BASE + ["--trim-ratio", "0.99"]).trim_ratio == 0.99
