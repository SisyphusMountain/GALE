#!/usr/bin/env python
"""Fit gpurec DTL rates on a dataset and write rates (AleRax format) + a JSON sidecar.

Standalone driver over ``gpurec.interface.fit``. There used to be a `gpurec fit` command line this
script shared code with; the command line was removed when fitting moved to
``gpurec.interface.fit``/``evaluate`` over a parsed ``FamilyData``, so this script now calls that
directly and has no ``gpurec.cli`` dependency.
"""
import argparse
import json
import math
import sys
from pathlib import Path

# global -> the shared 3-rate parametrization; genewise -> one rate triple per family.
_MODE_TO_PARAMETRIZATION = {"global": "global", "genewise": "per-family"}


def build_parser():
    p = argparse.ArgumentParser(description="gpurec rate-fit benchmark driver")
    p.add_argument("--species", required=True, help="species tree (Newick)")
    p.add_argument("--gene", required=True, nargs="+", help="gene tree file(s) (Newick), one family each")
    p.add_argument("--mode", choices=["global", "specieswise", "genewise"], default="global")
    p.add_argument("--config", default=None,
                   help="path to a GpurecConfig TOML file merged onto the fitting defaults")
    p.add_argument("--init-rate", type=float, default=0.1, help="initial rate for D, L, T")
    p.add_argument("--out", required=True)
    return p


def _write_outputs(out_path, result, mode):
    """node_rates file: 'node D L T' (gpurec order == AleRax cols) + a JSON sidecar."""
    rates = result.rates
    with open(out_path, "w") as fh:
        fh.write("# node D L T\n")
        if mode == "global":
            fh.write(f"GLOBAL {rates['D']:.10g} {rates['L']:.10g} {rates['T']:.10g}\n")
        else:
            for name, r in rates.items():
                fh.write(f"{name} {r['D']:.10g} {r['L']:.10g} {r['T']:.10g}\n")
    sidecar = {
        "mode": mode, "n_families": result.data.n_families,
        "nll_nats": -sum(result.log_likelihoods.values()),
        "wall_seconds": result.wall_seconds,
        "rates": rates,
    }
    with open(f"{out_path}.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "specieswise":
        # specieswise has no well-posed one-shot fit: the raw MLE over theta[S,3] is
        # non-identifiable and boundary-saturated. Point callers at the MAP/CV entry points instead.
        sys.stderr.write(
            "error: specieswise has no well-posed one-shot fit: fit a single MAP prior with "
            "gpurec.fit.specieswise_fit.fit_specieswise, or cross-validate the prior with "
            "gpurec.fit.map_cv.map_cv\n"
        )
        return 1

    from gpurec import load_families, load_fit_config
    from gpurec.interface import fit

    data = load_families(
        families=list(args.gene), species_tree=Path(args.species), min_covered_species=1,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=True,
    )
    config = load_fit_config(args.config)
    result = fit(
        data, parametrization=_MODE_TO_PARAMETRIZATION[args.mode],
        start_rates=(args.init_rate, args.init_rate, args.init_rate), config=config,
    )

    nll_nats = -sum(result.log_likelihoods.values())
    if not math.isfinite(nll_nats):
        sys.stderr.write(f"error: non-finite NLL for fit({args.mode})\n")
        return 1

    sys.stdout.write(f"fit({args.mode}): final NLL = {nll_nats:.6f} nats "
                     f"({data.n_families} families, {result.wall_seconds:.1f}s)\n")
    _write_outputs(args.out, result, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
