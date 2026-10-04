"""The Python interface: fit DTL rates, evaluate likelihoods at fixed rates."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import torch

from gpurec.api.model import DEFAULT_CLADE_BUDGET, GeneReconModel
from gpurec.config import GpurecConfig
from gpurec.data import FamilyData
from gpurec.fit.genewise_fit import fit_genewise
from gpurec.fit.global_fit import fit_global

LN2 = math.log(2.0)
PARAMETRIZATIONS = ("global", "per-family")


@dataclass(frozen=True)
class FitResult:
    data: FamilyData
    parametrization: str
    log2_rates: torch.Tensor
    log_likelihoods: dict[str, float]
    model: GeneReconModel
    history: dict | None
    wall_seconds: float

    @property
    def rates(self) -> dict:
        values = torch.exp2(self.log2_rates.double())
        if self.parametrization == "global":
            return {"D": float(values[0, 0]), "L": float(values[0, 1]), "T": float(values[0, 2])}
        return {
            name: {"D": float(row[0]), "L": float(row[1]), "T": float(row[2])}
            for name, row in zip(self.data.names, values)
        }

    def sample_reconciliations(self, n: int, seed: int) -> dict:
        from gpurec.reconciliation.sampling import sample_family_reconciliations

        return sample_family_reconciliations(self.model, self.data, n, seed)


def _log2_rates(rates) -> torch.Tensor:
    values = [float(value) for value in rates]
    if len(values) != 3 or not all(value > 0.0 and math.isfinite(value) for value in values):
        raise ValueError(f"rates must be three positive numbers (D, L, T), got {tuple(rates)!r}")
    return torch.log2(torch.tensor(values, dtype=torch.float64))


def _check_parametrization(parametrization: str) -> None:
    if parametrization not in PARAMETRIZATIONS:
        raise ValueError(f"parametrization must be one of {PARAMETRIZATIONS}, got {parametrization!r}")


def _result(data, parametrization, log2_rates, config, history, wall_seconds) -> FitResult:
    model = data.build_model(
        list(range(data.n_families)), "genewise", replace(config.solver, warm_adjoint=False), config,
        DEFAULT_CLADE_BUDGET,
    )
    with torch.no_grad():
        model.theta.copy_(log2_rates.to(device=model.theta.device, dtype=model.theta.dtype))
        nll_bits = model.genewise_loss_vector().double().cpu()
    log_likelihoods = {name: -float(bits) * LN2 for name, bits in zip(data.names, nll_bits)}
    return FitResult(data, parametrization, log2_rates.double().cpu(), log_likelihoods, model, history, wall_seconds)


def evaluate(data: FamilyData, parametrization: str, rates, config: GpurecConfig) -> FitResult:
    _check_parametrization(parametrization)
    start = time.perf_counter()
    log2_rates = _log2_rates(rates).reshape(1, 3).expand(data.n_families, 3).clone()
    result = _result(data, parametrization, log2_rates, config, None, 0.0)
    return replace(result, wall_seconds=time.perf_counter() - start)


def fit(data: FamilyData, parametrization: str, start_rates, config: GpurecConfig) -> FitResult:
    _check_parametrization(parametrization)
    start = _log2_rates(start_rates)
    t0 = time.perf_counter()
    if parametrization == "global":
        res = fit_global(data, start_log2_rates=start, config=config, verbose=True)
        log2_rates = res["theta"].double().reshape(1, 3).expand(data.n_families, 3).clone()
        history = {"n_steps": res["n_steps"]}
    else:
        res = fit_genewise(
            data,
            start_log2_rates=start.reshape(1, 3).expand(data.n_families, 3).clone(),
            config=config,
            certify=False,
            verbose=True,
        )
        log2_rates = res["theta"].detach().double().cpu()
        history = res["history"]
    return _result(data, parametrization, log2_rates, config, history, time.perf_counter() - t0)
