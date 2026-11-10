"""``gale``: the AleRax-compatible command line of GALE."""
from __future__ import annotations

import logging
import shlex
import sys
import time
from importlib.metadata import version

from gpurec.cli.arguments import UsageError, parse_arguments

logger = logging.getLogger("gpurec")
# Python warnings (for example the float32 tolerance clamp) are routed here by
# logging.captureWarnings, so they reach gale.log and the screen in the log format.
warnings_logger = logging.getLogger("py.warnings")
# Files and folders a GALE run writes under -p/--prefix. An output folder holding any of them is
# refused: GALE has no checkpoints, so writing into it would mix the files of two runs.
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
    if result.parametrization == "cladewise":
        classes = result.rate_classes
        logger.info("rate classes: %s", ", ".join(f"{count} {kind}" for count, kind in zip(classes.counts, "DLT")))
        for k, kind in enumerate("DLT"):
            for label in classes.starts[k]:
                logger.info("%s from %s: %s", kind, label, format_number(rates[label][kind]))
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
        sys.stderr.write(f"gale: {error}\n")
        return 2
    previous_level = logger.level
    try:
        earlier = [name for name in RESULT_NAMES if (args.prefix / name).exists()]
        if earlier:
            sys.stderr.write(
                f"gale: {args.prefix} already holds results of an earlier run ({', '.join(earlier)}); "
                "choose another -p/--prefix or remove them (GALE does not mix the files of two runs)\n"
            )
            return 2
        args.prefix.mkdir(parents=True, exist_ok=True)
        handlers = _configure_logging(args.prefix / "gale.log", args.verbose_opt)
    except OSError as error:
        sys.stderr.write(f"gale: cannot use {args.prefix} as the output folder: {error}\n")
        return 1
    started = time.perf_counter()
    try:
        import torch

        from gpurec import cuda_device, evaluate, fit, load_families, load_fit_config, write_alerax
        from gpurec.io.rate_classes import read_rate_classes
        from gpurec.io.alerax_output import ReconciliationWriter
        from gpurec.reconciliation.sampling import iter_family_reconciliations

        logger.info("GALE %s", version("gpurec"))
        logger.info("command: %s", shlex.join(["gale", *argv]))
        cuda_device()
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        config = load_fit_config(args.config)
        if args.precision is not None:
            config.precision.model_dtype = args.precision
        if args.clade_budget is not None:
            config.batching.clade_budget = args.clade_budget
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
        extra = {}  # the rate classes, for a rate-class file
        if args.rate_classes_file is not None:
            extra["classes"] = read_rate_classes(args.rate_classes_file, data.species)
        if args.fix_rates:
            result = evaluate(data, args.parametrization, args.start_rates, config, **extra)
        else:
            result = fit(data, args.parametrization, args.start_rates, config, **extra)
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
