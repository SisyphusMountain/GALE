"""AleRax 1.4.0's command-line arguments, restricted to what gpurec implements."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# AleRax 1.4.0 defaults (src/ale/AleArguments.cpp); they are AleRax's documented values,
# not gpurec settings, which is why they live here and not in the config file.
ALERAX_DEFAULT_PREFIX = "alerax_output"
ALERAX_DEFAULT_RATE = 0.1
ALERAX_DEFAULT_SAMPLES = 100
ALERAX_DEFAULT_SEED = 123
ALERAX_DEFAULT_MIN_COVERED_SPECIES = 4
ALERAX_DEFAULT_TRIM_RATIO = 0.0
ALERAX_DEFAULT_MAX_CLADE_SPLIT_RATIO = -1.0
ALERAX_DEFAULT_PARAMETRIZATION = "GLOBAL"
# AleRax options that describe a model choice; gpurec implements exactly one value of each.
FIXED_VALUE_OPTIONS = {
    "--rec-model": "UndatedDTL",
    "--transfer-constraint": "PARENTS",
    "--gene-tree-rooting": "UNIFORM",
    "--origination": "UNIFORM",
    "--species-tree-search": "SKIP",
}
# AleRax's one-letter spellings of the options above (src/ale/AleArguments.cpp).
SHORT_SPELLINGS = {"--rec-model": "-r"}
# Old AleRax spellings, rejected with a pointer to what replaced them.
DEPRECATED_OPTIONS = {
    "--per-family-rates": "use --model-parametrization PER-FAMILY",
    "--per-species-rates": "per-species rates are not supported by gpurec",
    "--verbose-opt-rates": "use --verbose-opt",
    "--gene-tree-sample-frequency": "now --ccp-sample-frequency, which gpurec does not support",
}
PARAMETRIZATIONS = {"GLOBAL": "global", "PER-FAMILY": "per-family"}
NAMED_SPECIES_TREES = ("Random", "MiniNJ", "NJst", "WMiniNJ", "Ustar", "Cherry", "CherryPro")


class UsageError(Exception):
    """The command line is not valid; the message lists every problem found."""


@dataclass(frozen=True)
class RunArguments:
    families: Path
    species_tree: Path
    prefix: Path
    parametrization: str
    start_rates: tuple[float, float, float]
    fix_rates: bool
    samples: int
    seed: int
    fraction_missing_file: Path | None
    min_covered_species: int
    trim_ratio: float
    max_clade_split_ratio: float | None
    skip_family_filtering: bool
    verbose_opt: bool
    precision: str | None
    config: Path | None


def _dest(option: str) -> str:
    """The attribute name argparse stores a long option under."""
    return option[2:].replace("-", "_")


def _stray_words(argv: list[str], unknown: list[str]) -> list[str]:
    """Words that are neither an option nor an option's value.

    A word right after an unsupported option is taken as that option's value: the option itself
    is already reported, and calling its value a stray word would be misleading.
    """
    unsupported = {token.split("=", 1)[0] for token in unknown if token.startswith("-")}
    stray = [token for token in unknown if not token.startswith("-")]
    for previous, token in zip(argv, argv[1:]):
        if previous in unsupported and token in stray:
            stray.remove(token)
    return stray


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpurec",
        description="Undated DTL reconciliation on a GPU, with AleRax's arguments and output files.",
        allow_abbrev=False,
    )
    parser.add_argument("-f", "--families", help="AleRax families file")
    parser.add_argument("-s", "--species-tree", help="rooted species tree (Newick file)")
    parser.add_argument("-p", "--prefix", default=ALERAX_DEFAULT_PREFIX, help="output folder")
    parser.add_argument("--model-parametrization", default=ALERAX_DEFAULT_PARAMETRIZATION,
                        help="GLOBAL or PER-FAMILY")
    parser.add_argument("--d", type=float, default=ALERAX_DEFAULT_RATE, help="duplication rate (start value)")
    parser.add_argument("--l", type=float, default=ALERAX_DEFAULT_RATE, help="loss rate (start value)")
    parser.add_argument("--t", type=float, default=ALERAX_DEFAULT_RATE, help="transfer rate (start value)")
    parser.add_argument("--fix-rates", action="store_true", help="evaluate at --d --l --t without fitting")
    parser.add_argument("-g", "--gene-tree-samples", type=int, default=ALERAX_DEFAULT_SAMPLES,
                        help="reconciliations sampled per family (0: none)")
    parser.add_argument("--seed", type=int, default=ALERAX_DEFAULT_SEED)
    parser.add_argument("--fraction-missing-file", help="lines 'species fraction'")
    parser.add_argument("--min-covered-species", type=int, default=ALERAX_DEFAULT_MIN_COVERED_SPECIES)
    parser.add_argument("--trim-ratio", type=float, default=ALERAX_DEFAULT_TRIM_RATIO)
    parser.add_argument("--max-clade-split-ratio", type=float, default=ALERAX_DEFAULT_MAX_CLADE_SPLIT_RATIO)
    parser.add_argument("--skip-family-filtering", action="store_true")
    parser.add_argument("--verbose-opt", action="store_true", help="log every optimizer step")
    for option, value in FIXED_VALUE_OPTIONS.items():
        spellings = [SHORT_SPELLINGS[option], option] if option in SHORT_SPELLINGS else [option]
        parser.add_argument(*spellings, dest=_dest(option), default=value, help=f"only {value} is supported")
    parser.add_argument("--precision", choices=("float32", "float64"),
                        help="gpurec only: floating-point precision (default: from the config file)")
    parser.add_argument("--config", help="gpurec only: TOML file with solver, precision and rate settings")
    return parser


def parse_arguments(argv: list[str]) -> RunArguments:
    args, unknown = _parser().parse_known_args(argv)
    problems = []
    unsupported = [token for token in unknown if token.startswith("-")]
    if unsupported:
        described = [
            f"{token} ({DEPRECATED_OPTIONS[token.split('=', 1)[0]]})"
            if token.split("=", 1)[0] in DEPRECATED_OPTIONS else token
            for token in unsupported
        ]
        problems.append("unsupported AleRax option(s): " + ", ".join(described))
    stray = _stray_words(argv, unknown)
    if stray:
        problems.append("unexpected argument(s): " + ", ".join(stray))
    if args.families is None:
        problems.append("missing -f/--families")
    if args.species_tree is None:
        problems.append("missing -s/--species-tree")
    elif args.species_tree in NAMED_SPECIES_TREES:
        problems.append(f"-s {args.species_tree}: gpurec needs a species tree file")
    for option, value in FIXED_VALUE_OPTIONS.items():
        given = getattr(args, _dest(option))
        if given != value:
            problems.append(f"{option} {given} is not supported (only {value})")
    if args.model_parametrization not in PARAMETRIZATIONS:
        problems.append(
            f"--model-parametrization {args.model_parametrization} is not supported (only GLOBAL or PER-FAMILY)"
        )
    for option in ("d", "l", "t"):
        if not getattr(args, option) > 0:
            problems.append(f"--{option} must be positive")
    if not 0.0 <= args.trim_ratio < 1.0:
        problems.append("--trim-ratio must be at least 0 and less than 1")
    if args.gene_tree_samples < 0:
        problems.append("-g/--gene-tree-samples must be 0 or more")
    if problems:
        raise UsageError("; ".join(problems))
    return RunArguments(
        families=Path(args.families),
        species_tree=Path(args.species_tree),
        prefix=Path(args.prefix),
        parametrization=PARAMETRIZATIONS[args.model_parametrization],
        start_rates=(args.d, args.l, args.t),
        fix_rates=args.fix_rates,
        samples=args.gene_tree_samples,
        seed=args.seed,
        fraction_missing_file=None if args.fraction_missing_file is None else Path(args.fraction_missing_file),
        min_covered_species=args.min_covered_species,
        trim_ratio=args.trim_ratio,
        max_clade_split_ratio=args.max_clade_split_ratio if args.max_clade_split_ratio > 0 else None,
        skip_family_filtering=args.skip_family_filtering,
        verbose_opt=args.verbose_opt,
        precision=args.precision,
        config=None if args.config is None else Path(args.config),
    )
