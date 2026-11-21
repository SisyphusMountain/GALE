"""``gale simulate``: simulate gene families under GALE's model and write them for GALE
(``gpurec.simulate_families``; README, "Simulating gene families"). Every option is required."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("gpurec")


def _table(path: Path, header: list[str]) -> dict[str, list[str]]:
    """``{first word: other words}`` of a parameter file whose first line starts with ``header``."""
    lines = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines or lines[0][:len(header)] != header:
        raise ValueError(f"{path}: the first line must start with {' '.join(header)!r}")
    rows = {}
    for row in lines[1:]:
        if len(row) < len(header) or row[0] in rows:
            raise ValueError(f"{path}: bad or repeated row {' '.join(row)!r}")
        rows[row[0]] = row[1:]
    return rows


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gale simulate", allow_abbrev=False, description=__doc__)
    p.add_argument("--species-tree", required=True, type=Path, help="rooted binary Newick species tree")
    rates = p.add_mutually_exclusive_group(required=True)
    rates.add_argument("--rates", nargs=3, type=float, metavar=("D", "L", "T"),
                       help="duplication, loss and transfer rates of every branch, relative to speciation")
    rates.add_argument("--rates-file", type=Path,
                       help="one (D, L, T) per branch, header 'node D L T': GALE's "
                            "model_parameters/model_parameters.txt (GALE fits no per-species rates: "
                            "families with unequal rows are not of a model GALE fits)")
    p.add_argument("--receiver-weights", required=True,
                   help="UNIFORM, or a file with header 'node weight' and one natural-log weight per "
                        "branch (GALE fits no receiver weights: its likelihood takes them all equal, "
                        "as UNIFORM)")
    p.add_argument("--origination", required=True, choices=("UNIFORM", "ROOT"),
                   help="branch of a family's first copy: UNIFORM, every branch equally likely (gale "
                        "--origination UNIFORM, the only start GALE fits); ROOT, always the root branch "
                        "(AleRax's --origination ROOT, for comparisons with AleRax: GALE cannot fit "
                        "these families with the start they were made with)")
    p.add_argument("--families", required=True, type=int, help="families kept (each leaves a gene)")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-copies", required=True, type=int,
                   help="an attempt whose history reaches this many gene copies is drawn again and counted")
    p.add_argument("--min-genes", required=True, type=int,
                   help="an attempt that leaves fewer genes than this is drawn again and counted; 1 is GALE's "
                        "model (and AleRax's): with more, rates fitted to the families are biased")
    p.add_argument("--threads", required=True, type=int, help="threads (the output does not depend on them)")
    p.add_argument("--out", required=True, type=Path, help="new output folder")
    return p


def main(argv: list[str]) -> int:
    from gpurec.simulate import simulate_families

    args = _parser().parse_args(argv)
    try:
        if args.rates is not None:
            rates = tuple(args.rates)
        else:
            rates = {name: tuple(float(x) for x in row[:3]) for name, row in
                     _table(args.rates_file, ["node", "D", "L", "T"]).items()}
        if args.receiver_weights == "UNIFORM":
            weights = "uniform"
        else:
            weights = {name: float(row[0]) for name, row in
                       _table(Path(args.receiver_weights), ["node", "weight"]).items()}
        result = simulate_families(args.species_tree, rates, weights, args.origination.lower(),
                                   args.families, args.seed, args.max_copies, args.min_genes, args.threads)
        result.write(args.out)
    except (ValueError, OSError) as error:
        sys.stderr.write(f"gale simulate: {error}\n")
        return 1
    # gale simulate writes no log file, so the summary goes to stderr only.
    handler = logging.StreamHandler(sys.stderr)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("%d families in %s; attempts drawn again: %d with no gene, %d over the copy cap, "
                    "%d with fewer than %d genes", args.families, args.out, result.attempts_without_gene,
                    result.attempts_over_cap, result.attempts_too_few_genes, args.min_genes)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return 0
