"""The Python interface: fit DTL rates, evaluate likelihoods at fixed rates."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import torch

from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig
from gpurec.data import FamilyData
from gpurec.fit.class_fit import class_rates, family_nll, fit_classes, species_log2_rates
from gpurec.fit.genewise_fit import fit_genewise
from gpurec.fit.global_fit import fit_global
from gpurec.io.rate_classes import RateClasses

LN2 = math.log(2.0)
PARAMETRIZATIONS = ("global", "per-family", "classes")


@dataclass(frozen=True)
class FitResult:
    data: FamilyData
    parametrization: str
    log2_rates: torch.Tensor
    log_likelihoods: dict[str, float]
    model: GeneReconModel
    history: dict | None
    wall_seconds: float
    rate_classes: RateClasses | None = None  # the classes of a "classes" fit; log2_rates is then [S,3]

    @property
    def rates(self) -> dict:
        if self.parametrization == "classes":
            labels = self.data.species.labels
            return {labels[node]: dict(zip("DLT", row)) for node, row in enumerate(class_rates(self.log2_rates))}
        values = torch.exp2(self.log2_rates.double())
        if self.parametrization == "global":
            return {"D": float(values[0, 0]), "L": float(values[0, 1]), "T": float(values[0, 2])}
        return {
            name: {"D": float(row[0]), "L": float(row[1]), "T": float(row[2])}
            for name, row in zip(self.data.names, values)
        }


def _log2_rates(rates) -> torch.Tensor:
    values = [float(value) for value in rates]
    if len(values) != 3 or not all(value > 0.0 and math.isfinite(value) for value in values):
        raise ValueError(f"rates must be three positive numbers (D, L, T), got {tuple(rates)!r}")
    return torch.log2(torch.tensor(values, dtype=torch.float64))


def _check_parametrization(parametrization: str, classes: RateClasses | None) -> None:
    if parametrization not in PARAMETRIZATIONS:
        raise ValueError(f"parametrization must be one of {PARAMETRIZATIONS}, got {parametrization!r}")
    if (parametrization == "classes") != (classes is not None):
        raise ValueError("the rate classes are given exactly when the parametrization is 'classes'")


def _class_result(data, classes, theta, config, history, wall_seconds) -> FitResult:
    nll_bits, model = family_nll(data, theta, config, config.solver.pi_iters, config.solver.neumann_terms)
    log_likelihoods = {name: -float(bits) * LN2 for name, bits in zip(data.names, nll_bits.double().cpu())}
    return FitResult(data, "classes", theta.double().cpu(), log_likelihoods, model, history, wall_seconds, classes)


def _result(data, parametrization, log2_rates, config, history, wall_seconds) -> FitResult:
    model = data.build_model(
        list(range(data.n_families)), "genewise", replace(config.solver, warm_adjoint=False), config,
        config.batching.clade_budget,
    )
    with torch.no_grad():
        model.theta.copy_(log2_rates.to(device=model.theta.device, dtype=model.theta.dtype))
        nll_bits = model.genewise_loss_vector().double().cpu()
    log_likelihoods = {name: -float(bits) * LN2 for name, bits in zip(data.names, nll_bits)}
    return FitResult(data, parametrization, log2_rates.double().cpu(), log_likelihoods, model, history, wall_seconds)


def evaluate(data: FamilyData, parametrization: str, rates, config: GpurecConfig,
             classes: RateClasses | None = None) -> FitResult:
    _check_parametrization(parametrization, classes)
    start = time.perf_counter()
    if parametrization == "classes":
        log2 = _log2_rates(rates)
        theta = species_log2_rates(classes, [log2[k].expand(classes.counts[k]).clone() for k in range(3)])
        result = _class_result(data, classes, theta, config, None, 0.0)
        return replace(result, wall_seconds=time.perf_counter() - start)
    log2_rates = _log2_rates(rates).reshape(1, 3).expand(data.n_families, 3).clone()
    result = _result(data, parametrization, log2_rates, config, None, 0.0)
    return replace(result, wall_seconds=time.perf_counter() - start)


def fit(data: FamilyData, parametrization: str, start_rates, config: GpurecConfig,
        classes: RateClasses | None = None) -> FitResult:
    _check_parametrization(parametrization, classes)
    start = _log2_rates(start_rates)
    t0 = time.perf_counter()
    if parametrization == "classes":
        # The class fit refines the global optimum, where every class starts.
        glob = fit_global(data, start_log2_rates=start, config=config, verbose=True)
        res = fit_classes(data, classes, start_log2_rates=glob["theta"], config=config, verbose=True)
        history = {"n_steps_global": glob["n_steps"], "n_steps": res["n_steps"],
                   "n_evaluations": res["n_evaluations"], "converged": res["converged"], "message": res["message"]}
        log_likelihoods = {name: -float(bits) * LN2 for name, bits in zip(data.names, res["nll_vector_bits"])}
        return FitResult(data, "classes", res["theta"].double().cpu(), log_likelihoods, res["model"], history,
                         time.perf_counter() - t0, classes)
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
