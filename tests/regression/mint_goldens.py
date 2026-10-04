"""Mint committed goldens: fit each mode, measure run-to-run spread, set tolerances = 3x spread.
Run ONCE on the target GPU (RTX 4090). Re-run to re-mint if rustree/gpurec change."""
from __future__ import annotations

import argparse, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gpurec.config import GpurecConfig
from gpurec.data import load_families
from gpurec.fit.genewise_fit import fit_genewise
from gpurec.fit.global_fit import fit_global
from tests.support.simulate import (
    simulate_dataset, SIM_PARAMS, SPECIESWISE_GOLDEN_LAM, SPECIESWISE_GOLDEN_INIT_RATE,
)

GOLDENS = Path(__file__).parent / "goldens"

# The rate global/genewise are fit from (relative to speciation): the plain fit_dtl-era default.
_START_RATE = 0.1


def _git_rev(cwd):
    try:
        return subprocess.check_output(["git", "-C", str(cwd), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def fit_mode(mode, species_path, gene_paths, *, verbose=False):
    """Fit one mode via its production recipe; returns (nll_bits, rates[.,3] D,L,T, wall_s).

    global/genewise are fit over a parsed FamilyData via fit_global/fit_genewise. specieswise has no
    one-shot MLE, so it is fit by fit_specieswise at the committed SPECIESWISE_GOLDEN_LAM prior, with
    init AND prior mean theta_ref = constant log2(SPECIESWISE_GOLDEN_INIT_RATE) (see the design spec)."""
    if mode == "specieswise":
        # NB: no local `import torch` here -- torch is module-level (line 10). A function-local
        # import would make `torch` local to ALL of fit_mode and UnboundLocalError the
        # global/genewise branch's `torch.float32` below.
        import math
        from gpurec.api.model import GeneReconModel
        from gpurec.api.solver_options import SolverOptions
        from gpurec.fit.specieswise_fit import fit_specieswise
        model = GeneReconModel(species_path, gene_paths, mode="specieswise",
                               dtype=torch.float32,
                               solver_options=SolverOptions())
        # init AND ridge center at a sensible rate, NOT the model's degenerate 1e-10 default (which
        # centers the prior on ~zero rates -> fit snaps to the prior in ~2 steps).
        S = model.theta.shape[0]
        theta0 = torch.full((S, 3), math.log2(SPECIESWISE_GOLDEN_INIT_RATE), device="cuda",
                            dtype=torch.float32)
        res = fit_specieswise(model.batch_statics, theta0, model.receiver_weights.detach(),
                              lam=SPECIESWISE_GOLDEN_LAM, theta_ref=theta0, verbose=verbose)
        rates = np.asarray(res["rates"])
        return float(res["nll_bits"]), rates, float(res["wall_s"])

    if mode not in ("global", "genewise"):
        raise ValueError(f"unknown mode {mode!r}; expected one of ('global', 'genewise', 'specieswise')")
    data = load_families(
        families=list(gene_paths), species_tree=Path(species_path), min_covered_species=1,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=True,
    )
    config = GpurecConfig.genewise_reference()
    start = torch.log2(torch.tensor([_START_RATE, _START_RATE, _START_RATE], dtype=torch.float64))
    if mode == "global":
        res = fit_global(data, start_log2_rates=start, config=config, verbose=verbose)
        nll_bits, wall_s = res["nll_bits"], res["wall_s"]
    else:
        t0 = time.perf_counter()
        res = fit_genewise(
            data, start_log2_rates=start.reshape(1, 3).expand(data.n_families, 3).clone(),
            config=config, certify=True, verbose=verbose,
        )
        nll_bits, wall_s = float(res["loss_bits"]), time.perf_counter() - t0
    rates = np.asarray(res["rates"].detach().float().cpu())   # device tensor -> ndarray
    return float(nll_bits), rates, float(wall_s)


def mint(mode, repeats=5, verbose=False):
    p = SIM_PARAMS[mode]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sp, genes = simulate_dataset(mode, Path(d) / mode, n_species=p["n_species"],
                                     n_families=p["n_families"], dtl=p["dtl"], seed=p["seed"])
        nlls, rate_runs, walls = [], [], []
        for r in range(repeats):
            nll, rates, wall = fit_mode(mode, sp, genes, verbose=verbose)
            nlls.append(nll); rate_runs.append(rates); walls.append(wall)
            if verbose:
                print(f"[{mode}] repeat {r + 1}/{repeats}: nll={nll:.6f} wall={wall:.1f}s", flush=True)

    nlls = np.array(nlls); R = np.stack(rate_runs)          # [repeats, ., 3]
    nll_spread = float(nlls.std()) if repeats > 1 else 0.0
    rate_spread_rel = float(np.abs(R - R.mean(0)).max() / (np.abs(R).mean() + 1e-30)) if repeats > 1 else 0.0
    nll_rtol = max(3 * nll_spread / (abs(nlls.mean()) + 1e-30), 1e-3)
    rates_rtol = max(3 * rate_spread_rel, 1e-2)

    golden = {
        "mode": mode,
        "provenance": {
            "rustree_commit": _git_rev("/home/enzo/Documents/git/rustree"),
            "gpurec_commit": _git_rev(Path(__file__).resolve().parents[2]),
            "seed": p["seed"], "n_species": p["n_species"], "n_families": p["n_families"],
            "dtl": p["dtl"], "device": torch.cuda.get_device_name(0), "dtype": "float32",
            "repeats": repeats, "nll_spread": nll_spread, "rate_spread_rel": rate_spread_rel,
            "minted_utc": datetime.now(timezone.utc).isoformat(),
            "specieswise_golden_lam": SPECIESWISE_GOLDEN_LAM,
            "specieswise_golden_init_rate": SPECIESWISE_GOLDEN_INIT_RATE,
        },
        "nll": float(nlls.mean()),
        "rates": R.mean(0).tolist(),
        "recorded_wall_s": float(np.median(walls)),
        "tolerances": {"nll_rtol": nll_rtol, "rates_rtol": rates_rtol, "rates_atol": 1e-4},
    }
    GOLDENS.mkdir(exist_ok=True)
    (GOLDENS / f"{mode}.json").write_text(json.dumps(golden, indent=2))
    print(f"[{mode}] nll={golden['nll']:.6f} rtol={nll_rtol:.2e} rates_rtol={rates_rtol:.2e} "
          f"wall~{golden['recorded_wall_s']:.1f}s")
    return golden


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(SIM_PARAMS), default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--verbose", action="store_true", help="stream per-step optimize progress")
    a = ap.parse_args()
    for m in ([a.mode] if a.mode else list(SIM_PARAMS)):
        mint(m, repeats=a.repeats, verbose=a.verbose)
