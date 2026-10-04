"""``gpurec``: the AleRax-compatible command line."""
from __future__ import annotations

import logging
import shlex
import sys
import time
from importlib.metadata import version

from gpurec.cli.arguments import UsageError, parse_arguments

logger = logging.getLogger("gpurec")
# Python warnings (for example the float32 tolerance clamp) are routed here by
# logging.captureWarnings, so they reach gpurec.log and the screen in the log format.
warnings_logger = logging.getLogger("py.warnings")
# Files and folders a gpurec run writes under -p/--prefix. An output folder holding any of them is
# refused: gpurec has no checkpoints, so writing into it would mix the files of two runs.
RESULT_NAMES = (
    "per_fam_likelihoods.txt", "model_parameters", "reconciliations", "species_trees", "perSpeciesCoverage.txt",
)


def _configure_logging(log_path, verbose: bool) -> list[logging.Handler]:
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    # The log file is opened first: if that fails, no handler has been attached yet.
    log_file = logging.FileHandler(log_path, mode="a")
    handlers = [logging.StreamHandler(sys.stderr), log_file]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        warnings_logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.captureWarnings(True)
    return handlers


def _log_rates(result) -> None:
    """Record the rates in the log: one line for GLOBAL; for PER-FAMILY a range per rate, and one
    line per family at the debug level (shown with --verbose-opt)."""
    from gpurec.io.text import format_number

    rates = result.rates
    if result.parametrization == "global":
        logger.info("rates: D %s L %s T %s", *(format_number(rates[kind]) for kind in "DLT"))
        return
    ranges = ", ".join(
        f"{kind} from {format_number(min(r[kind] for r in rates.values()))} "
        f"to {format_number(max(r[kind] for r in rates.values()))}"
        for kind in "DLT"
    )
    logger.info("per-family rates of %d families: %s", len(rates), ranges)
    for name, family_rates in rates.items():
        logger.debug("rates of %s: D %s L %s T %s", name, *(format_number(family_rates[kind]) for kind in "DLT"))


def main(argv: list[str]) -> int:
    try:
        args = parse_arguments(argv)
    except UsageError as error:
        sys.stderr.write(f"gpurec: {error}\n")
        return 2
    previous_level = logger.level
    try:
        earlier = [name for name in RESULT_NAMES if (args.prefix / name).exists()]
        if earlier:
            sys.stderr.write(
                f"gpurec: {args.prefix} already holds results of an earlier run ({', '.join(earlier)}); "
                "choose another -p/--prefix or remove them (gpurec does not mix the files of two runs)\n"
            )
            return 2
        args.prefix.mkdir(parents=True, exist_ok=True)
        handlers = _configure_logging(args.prefix / "gpurec.log", args.verbose_opt)
    except OSError as error:
        sys.stderr.write(f"gpurec: cannot use {args.prefix} as the output folder: {error}\n")
        return 1
    started = time.perf_counter()
    try:
        import torch

        from gpurec import cuda_device, evaluate, fit, load_families, load_fit_config, write_alerax
        from gpurec.io.alerax_output import ReconciliationWriter
        from gpurec.reconciliation.sampling import iter_family_reconciliations

        logger.info("gpurec %s", version("gpurec"))
        logger.info("command: %s", shlex.join(["gpurec", *argv]))
        cuda_device()
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        config = load_fit_config(args.config)
        if args.precision is not None:
            config.precision.model_dtype = args.precision
        config.validate()
        data = load_families(
            families=args.families,
            species_tree=args.species_tree,
            min_covered_species=args.min_covered_species,
            trim_ratio=args.trim_ratio,
            max_clade_split_ratio=args.max_clade_split_ratio,
            fraction_missing=args.fraction_missing_file,
            skip_family_filtering=args.skip_family_filtering,
        )
        if args.fix_rates:
            result = evaluate(data, args.parametrization, args.start_rates, config)
        else:
            result = fit(data, args.parametrization, args.start_rates, config)
        _log_rates(result)
        logger.info("total log-likelihood: %s", sum(result.log_likelihoods.values()))
        write_alerax(result, None, args.prefix)
        if args.samples > 0:
            # One batch of families at a time, so memory does not grow with the number of families.
            writer = ReconciliationWriter(result.data.species, args.prefix / "reconciliations", args.samples)
            for name, trees in iter_family_reconciliations(result.model, result.data, args.samples, args.seed):
                writer.add_family(name, trees)
            writer.finish()
        logger.info("done in %.1f s", time.perf_counter() - started)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    finally:
        logging.captureWarnings(False)
        for handler in handlers:
            logger.removeHandler(handler)
            warnings_logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)


def entry() -> None:
    sys.exit(main(sys.argv[1:]))
