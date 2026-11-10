"""Cladewise fit: one duplication, loss and transfer rate per class of species branches.

The classes (``gpurec.io.rate_classes``, AleRax's ``--model-parametrization <file>``) tie the
per-species rates of GALE's specieswise mode: branch ``s`` takes the log2 rates of its classes,
``theta[s] = (x_D[c_D(s)], x_L[c_L(s)], x_T[c_T(s)])``, shared by all families. The objective is the
total NLL over families; the gradient with respect to a class rate is the sum of the per-species
gradients over the class's branches (the chain rule of a shared parameter), so the specieswise
gradient serves unchanged. L-BFGS-B (``[cladewise]`` of the config) minimises it in the rate box
``config.rates``, with the forward solves as ``[solver]`` sets them; a fit that does not converge
raises ``RuntimeError``.

The rate classes are AleRax's custom parametrization (``--model-parametrization <file>``, AleRax
``src/ale/OptimizationClasses.cpp``); Huang et al. (Nat Commun 2026) use it for their clade-wise
models (DTL_br1, DTL_br2). This implementation was ported from RALE (Aleraxv2,
https://github.com/oliver-schick/RALE, ``docs/notes/RATE_CLASSES.md``).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from scipy.optimize import minimize

from gpurec.api._execution import stream_batches, stream_family_nll
from gpurec.api.model import cuda_device
from gpurec.config import GpurecConfig
from gpurec.io.rate_classes import RateClasses
from gpurec.optimization import log2_rate_bounds

logger = logging.getLogger(__name__)


def species_log2_rates(classes: RateClasses, class_log2_rates: list[torch.Tensor]) -> torch.Tensor:
    """``[S,3]`` log2 rates of every species branch from the class rates of D, L and T."""
    columns = []
    for k, rates in enumerate(class_log2_rates):
        index = torch.tensor(classes.class_of[k], dtype=torch.long, device=rates.device)
        columns.append(rates.index_select(0, index))
    return torch.stack(columns, dim=1)


def class_rates(log2_rates: torch.Tensor) -> list[list[float]]:
    """The rates of ``[S,3]`` log2 rates, element by element: ``torch.exp2`` rounds a tensor's
    vectorised body and its scalar tail differently, which wrote two nodes of one class one ulp
    apart."""
    return [[2.0 ** value for value in row] for row in log2_rates.double().tolist()]


def _specieswise_model(data, config: GpurecConfig):
    # the forward solves as [solver] sets them: no tier of the fit's own overrides pi_iters or neumann_terms
    return data.build_model(
        list(range(data.n_families)), "specieswise", config.solver, config, config.batching.clade_budget,
    )


def family_nll(data, theta: torch.Tensor, config: GpurecConfig):
    """Per-family NLL (bits) at per-species rates ``theta`` ``[S,3]``, and the specieswise model that
    computed it (its ``theta`` set to these rates, for sampling reconciliations)."""
    return _model_nll(_specieswise_model(data, config), theta, data.n_families)


def _model_nll(model, theta: torch.Tensor, n_families: int):
    theta = theta.to(device=model.theta.device, dtype=model.theta.dtype)
    with torch.no_grad():
        model.theta.copy_(theta)
    nll = stream_family_nll(
        model.batch_statics, theta, model.receiver_weights.detach(), model.origination_weights.detach(),
        n_families=n_families,
    )
    return nll, model


def make_cladewise_objective(model, classes: RateClasses):
    """``x -> (NLL bits, gradient)`` over the concatenated class log2 rates ``[D..., L..., T...]``
    of a specieswise ``model``: the per-species rates are gathered from the classes, and the
    per-species gradient is summed back over each class's branches."""
    device, dtype = model.theta.device, model.theta.dtype
    counts = list(classes.counts)
    index = [torch.tensor(classes.class_of[k], dtype=torch.long, device=device) for k in range(3)]
    receiver_weights = model.receiver_weights.detach()
    origination_weights = model.origination_weights.detach()

    def objective(x):
        parts = torch.split(torch.as_tensor(x, device=device, dtype=dtype), counts)
        theta = torch.stack([parts[k].index_select(0, index[k]) for k in range(3)], dim=1)
        loss, grad, _, _ = stream_batches(
            model.batch_statics, theta, receiver_weights, origination_weights, genewise=False, need_grad=True
        )
        class_grad = [
            torch.zeros(counts[k], dtype=grad.dtype, device=device).index_add_(0, index[k], grad[:, k])
            for k in range(3)
        ]
        return float(loss), torch.cat(class_grad).double().cpu().numpy()

    return objective


def fit_cladewise(data, classes: RateClasses, *, start_log2_rates, config: GpurecConfig, verbose: bool) -> dict:
    """Fit the class rates from ``start_log2_rates`` (``[3]``: every class of a rate starts there,
    typically the global optimum), with ``[solver]`` for the forward solves and ``[cladewise]`` for
    L-BFGS-B. Raises ``RuntimeError`` when L-BFGS-B does not converge. Returns ``{mode,
    theta[cpu,S,3], class_log2_rates (list of three cpu tensors), nll_vector_bits[cpu,G], nll_bits,
    n_steps, n_evaluations, message, model, wall_s}``."""
    t0 = time.perf_counter()
    lo, hi = log2_rate_bounds(bounds=config.rates)
    opts = config.cladewise
    cuda_device()  # a clear error without a GPU, before the model is built
    model = _specieswise_model(data, config)
    counts = classes.counts
    start = [float(v) for v in torch.as_tensor(start_log2_rates, dtype=torch.float64).reshape(3)]
    x0 = np.concatenate([np.full(counts[k], start[k]) for k in range(3)])
    x0 = np.clip(x0, lo, np.inf if hi is None else hi)
    class_objective = make_cladewise_objective(model, classes)
    n_evaluations = 0

    def objective(x):
        nonlocal n_evaluations
        n_evaluations += 1
        return class_objective(x)

    steps = 0

    def report(intermediate_result):
        nonlocal steps
        steps += 1
        if verbose:
            logger.debug("[fit_cladewise] it=%3d loss=%.6f bits", steps, float(intermediate_result.fun))

    result = minimize(
        objective, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * len(x0), callback=report,
        options={"maxiter": opts.max_iter, "ftol": opts.ftol, "gtol": opts.gtol, "maxcor": opts.maxcor},
    )
    if not result.success:
        raise RuntimeError(
            f"cladewise fit did not converge after {result.nit} L-BFGS-B iterations ({n_evaluations} evaluations): "
            f"{result.message} ([cladewise] max_iter = {opts.max_iter})"
        )
    parts = torch.split(torch.as_tensor(result.x, dtype=torch.float64), list(counts))
    class_log2_rates = [part.clone() for part in parts]
    theta_hat = species_log2_rates(classes, class_log2_rates)
    if verbose:
        logger.info("[fit_cladewise] %s after %d steps, %d evaluations", result.message, result.nit, n_evaluations)

    nll, model = _model_nll(model, theta_hat, data.n_families)
    nll = nll.double().cpu()
    return {
        "mode": "cladewise", "theta": theta_hat, "class_log2_rates": class_log2_rates,
        "nll_vector_bits": nll, "nll_bits": float(nll.sum()), "n_steps": int(result.nit),
        "n_evaluations": n_evaluations, "message": str(result.message),
        "model": model, "wall_s": time.perf_counter() - t0,
    }
