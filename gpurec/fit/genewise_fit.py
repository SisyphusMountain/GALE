"""Default genewise (per-family) DTL-rate fitting recipe.

``fit_genewise`` is the standard way to fit independent per-family Duplication/Loss/Transfer rates to a
converged, box-bounded maximum-likelihood optimum -- the same model AleRax fits with ``--rec-model
UndatedDTL --model-parametrization PER-FAMILY``. The recipe (validated on the archaea / Hogenom genewise
benchmarks; see ``kernel-bench/experiments/alerax_speed/OPTIMIZATION_PLAN.md``) is:

  1. **Adam warm-up** -- a few large clipped steps (lr=1, grad-clip-norm 10) projected against the rate
     box, to enter the basin fast.
  2. **Box-constrained trust-region Newton** on the per-family 3x3 **analytic-HVP** Hessian (3
     broadcast unit-theta-component probes, warm-started across repeated rebuilds; eigenvalue-clamped
     to ``mu`` -> PD), converging on the per-family projected gradient ``|Pg| < tol``.
  3. **Convergence-based rebatching** -- once a fraction of the active batch has converged (verified at
     the high pi/Neumann tier), those families are frozen and dropped and the model is rebuilt over the
     survivors, so the long tail of hard families runs on a small batch.
  4. **pi-tier escalation** -- the bulk runs at a cheap forward pi (16) with the adjoint **warm-start**
     (``SolverOptions.warm_adjoint``, memory-gated automatically -- see ``gpurec.core.memory_policy``);
     only the forward-stiff families escalate to the accurate pi (64).

Returns a dict with the fitted log2-rates ``theta`` ([F,3]), the AleRax-comparable ``rates`` (= 2**theta,
relative to speciation), the rebatch/defer history, and -- when ``certify=True`` -- a final cold PD
certificate (per-family |Pg|, smallest Hessian eigenvalue, interior-PD / bound-active counts, total NLL).
"""
from __future__ import annotations

import logging
import math
import time

import torch

from gpurec.api.model import cuda_device
from gpurec.api.solver_options import SolverOptions
from gpurec.config import GpurecConfig
from gpurec.core.inference.solver import solve_forward_residual
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_
from gpurec.solver.value_and_grad import forward_solve
from gpurec.solver.hvp.exact import make_exact_hvp

logger = logging.getLogger(__name__)

# Proven base solver settings (pi_iters / neumann_terms are overridden per tier below). Single-sourced
# from ``GpurecConfig.genewise_reference().solver`` (task-10 brief) -- edit the values there, not here.
_BASE_SOLVER = {
    k: getattr(GpurecConfig.genewise_reference().solver, k)
    for k in (
        "e_max_iter", "e_tol",
        "e_adjoint_max_iter", "e_adjoint_tol",
        "adjoint_pruning_threshold", "use_adjoint_pruning", "pibar_side_threshold",
    )
}

# Reference recipe tuned for the standard genewise problem. Import and clone-override per dataset:
#   fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4})
# Per-dataset values belong in your experiment script, NOT edited here (see docs/config_convention.md).
GENEWISE_REFERENCE = dict(
    adam_steps=5, adam_lr=1.0, grad_clip=10.0, pi_tiers=(16, 64), neu_opt=16, neu_cert=64,
    tol=1e-3, max_iter=120, check_every=4, drop_frac=0.30, trust=2.0,
    mu=1e-2, fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
    warm_adjoint=True, certify=False,
)


def _analytic_hessian(m, theta, pi_cur):
    """Per-family [G,3,3] curvature via 3 broadcast analytic-HVP probes, warm-started via
    probe_id. Mirrors hvp_exact.py's _make_exact_hvp_streaming genewise gather/scatter for
    single-batch (which the top-level make_exact_hvp does NOT do for you), and lets it handle
    multi-batch itself."""
    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    if len(m.batch_statics) > 1:
        hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u = torch.zeros(G, 3, device=dev, dtype=dtype); u[:, j] = 1.0
            cols.append(hvp(u.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3))
        H = torch.stack(cols, dim=-1)
    else:
        static = m.batch_statics[0]
        fam = static.family_index_tensor.to(dev)
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve(m.batch_statics, theta, rw)
        hvp = make_exact_hvp(m.batch_statics, theta_b, rw, sv, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u_b = torch.zeros(G, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3)
            col = torch.zeros(G, 3, device=dev, dtype=dtype)
            col.index_add_(0, fam, out_b)
            cols.append(col)
        H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))


def fit_genewise(
    data,
    *,
    start_log2_rates,
    config: GpurecConfig,
    # --- the recipe (defaults = the accepted optimized recipe) ---
    adam_steps: int = 5,
    adam_lr: float = 1.0,
    grad_clip: float = 10.0,
    pi_tiers=(16, 64),
    neu_opt: int = 16,
    neu_cert: int = 64,
    tol: float = 1e-3,
    max_iter: int = 120,
    check_every: int = 4,
    drop_frac: float = 0.30,
    trust: float = 2.0,
    mu: float = 1e-2,
    fwd_tol: float = 1e-3,
    improve_frac: float = 0.8,
    verify_drop: bool = True,
    eager_defer: bool = True,
    warm_adjoint: bool = True,
    certify: bool = False,
    verbose: bool = False,
) -> dict:
    """Fit per-family DTL rates to a converged, box-bounded optimum. See module docstring for the recipe.

    ``data`` is a :class:`gpurec.data.FamilyData` (already-parsed families); ``config`` (a top-level
    :class:`GpurecConfig`) supplies the solver settings (the same key subset as ``_BASE_SOLVER``) and
    the rate box (``config.rates``) -- both are taken wholesale, not merged with anything else.

    NOT threaded: ``config.newton`` (this recipe's Newton step is a bespoke box-constrained
    trust-region analytic-HVP 3x3 Hessian solve, not a ``NewtonOptions`` consumer); ``config.regularizer``
    (unused -- this recipe has no regularization term); ``config.memory`` (the adjoint warm-start is
    controlled by this recipe's own ``warm_adjoint`` argument, threaded into each build's
    ``SolverOptions.warm_adjoint``, + the library's own memory gate, not ``config.memory``).
    """
    dev = cuda_device()
    dtype = config.precision.model_torch_dtype
    bounds = config.rates
    if bounds.max_rate is None:
        raise ValueError(
            "fit_genewise needs a finite max_rate, but config.rates.max_rate is None; "
            "start from GpurecConfig.genewise_reference(), whose rate box is [1e-6, 2.0]"
        )
    pis = [int(p) for p in pi_tiers]
    cert_pi = max(pis)
    lo, hi = log2_rate_bounds(bounds=bounds)
    base = {k: getattr(config.solver, k) for k in _BASE_SOLVER}

    def _log(msg):
        if verbose:
            logger.debug(msg)

    def sopts(pi, neu, warm):
        return SolverOptions(**{**base, "pi_iters": pi, "neumann_terms": neu, "warm_adjoint": warm})

    def build(indices, pi, neu, warm):
        m = data.build_model(
            list(indices), "genewise", sopts(pi, neu, warm), config,
            config.batching.clade_budget,
        )
        m.receiver_weights.requires_grad_(False)   # uniform transfer recipients (UndatedDTL default)
        return m

    def lg(m, th):
        lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
        return lv.to(dtype), g.to(dtype)

    def pgmax(th, g):
        return project_rate_gradient_(th, g.clone(), bounds=bounds).abs().amax(dim=1)

    def clamp_(th):
        clamp_log_rate_(th, bounds=bounds)
        return th

    def forward_resid(m, th, pi):
        out = torch.zeros(len(m.families), device=dev, dtype=dtype)
        rw = m.receiver_weights.detach()
        with torch.no_grad():
            for static in m.batch_statics:
                r = solve_forward_residual(static, m._theta_for_static(static, th), rw, pi_iters=pi)
                out[static.family_index_tensor.to(dev)] = r.to(dev)
        return out

    F_all = data.n_families
    theta = clamp_(start_log2_rates.to(device=dev, dtype=dtype).clone())
    active = torch.arange(F_all, device=dev)
    was_dropped = torch.zeros(F_all, dtype=torch.bool, device=dev)
    pg_last = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    rebatch_log, defer_log = [], []
    n_steps = n_builds = 0

    t0 = time.perf_counter()
    carry = None
    for pi_idx, pi_cur in enumerate(pis):
        if carry is not None:
            active = carry
        if active.numel() == 0:
            break
        last_tier = pi_idx == len(pis) - 1
        carry = active[:0].clone()
        m = build(active.tolist(), pi_cur, neu_opt, warm_adjoint); n_builds += 1
        sub = theta.index_select(0, active).clone()
        _log(f"[fit_genewise] tier pi={pi_cur}: {active.numel()} families")

        if pi_idx == 0 and adam_steps > 0:   # Adam warm-up (basin entry), once
            lf = sub.clone().requires_grad_(True)
            ad = torch.optim.Adam([lf], lr=adam_lr)
            for _ in range(adam_steps):
                _, g = lg(m, lf.detach()); lf.grad = g
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(lf, grad_clip)
                project_rate_gradient_(lf.detach(), lf.grad, bounds=bounds)
                ad.step()
                with torch.no_grad():
                    clamp_(lf)
            sub = lf.detach().clone()

        Hd = None
        for it in range(max_iter):
            if active.numel() == 0:
                break
            lv, g = lg(m, sub)
            if it % check_every == 0:
                pgm = pgmax(sub, g)
                plateau = pgm >= improve_frac * pg_last.index_select(0, active)
                pg_last.index_copy_(0, active, pgm)
                conv = pgm < tol
                frac = float(conv.float().mean())
                _log(f"  [pi{pi_cur} it{it}] active={active.numel()} conv={frac*100:.0f}% "
                     f"|Pg|max={float(pgm.max()):.2e}")
                if frac > drop_frac and bool(conv.any()):
                    cert_ok = conv.clone()
                    if verify_drop:   # re-check the converged subset cold at the high tier before freezing
                        m.solver_options = sopts(cert_pi, neu_cert, False)
                        cert_ok = conv & (pgmax(sub, lg(m, sub)[1]) < tol)
                        m.solver_options = sopts(pi_cur, neu_opt, warm_adjoint)
                    drop = cert_ok
                    defer = torch.zeros_like(conv)
                    reject = conv & ~cert_ok
                    resid_max = 0.0
                    if not last_tier and bool(reject.any()):   # stiff: escalate to the next pi tier
                        resid = forward_resid(m, sub, pi_cur)
                        resid_max = float(resid.max())
                        defer = reject & (resid > fwd_tol)
                        if eager_defer:
                            defer = defer | (reject & plateau)
                    if bool(drop.any()) or bool(defer.any()):
                        if bool(drop.any()):
                            theta.index_copy_(0, active[drop], sub[drop]); was_dropped[active[drop]] = True
                            rebatch_log.append(dict(pi=pi_cur, it=it, dropped=int(drop.sum()),
                                                    remain=int((~drop & ~defer).sum())))
                        if bool(defer.any()):
                            theta.index_copy_(0, active[defer], sub[defer]); carry = torch.cat([carry, active[defer]])
                            defer_log.append(dict(pi=pi_cur, it=it, deferred=int(defer.sum()),
                                                  to=pis[pi_idx + 1], resid_max=resid_max))
                        active = active[~(drop | defer)]; sub = sub[~(drop | defer)].clone()
                        if active.numel() == 0:
                            break
                        del m; torch.cuda.empty_cache()
                        m = build(active.tolist(), pi_cur, neu_opt, warm_adjoint); n_builds += 1
                        Hd = None
                        continue
            if it % 5 == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
                H = _analytic_hessian(m, sub, pi_cur)
                e, V = torch.linalg.eigh(H)
                Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
            fixed = ((sub >= hi - bounds.bound_active_eps) & (g < 0)) | \
                ((sub <= lo + bounds.bound_active_eps) & (g > 0))
            free = (~fixed).float()
            Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
            delta = -torch.linalg.solve(Hred, (g * free).unsqueeze(-1)).squeeze(-1)
            dn = delta.norm(dim=1, keepdim=True)
            sub = clamp_(sub + delta * (trust / dn.clamp(min=trust)))   # trust-region cap
            n_steps += 1
        if active.numel() > 0:
            theta.index_copy_(0, active, sub)
            carry = torch.cat([carry, active])
        del m; torch.cuda.empty_cache()

    result = dict(
        theta=theta, rates=torch.exp2(theta), n_families=F_all,
        opt_seconds=time.perf_counter() - t0, n_steps=n_steps, n_builds=n_builds,
        history=dict(rebatch=rebatch_log, defer=defer_log),
    )
    if certify:   # final cold PD certificate over ALL families at the high pi/Neumann tier
        mfull = build(range(F_all), cert_pi, neu_cert, False)
        _, g = lg(mfull, theta); pg = pgmax(theta, g)
        H = _analytic_hessian(mfull, theta, cert_pi)
        lam_min = torch.linalg.eigvalsh(H)[:, 0]
        bound_active = ((theta <= lo + bounds.bound_active_eps) | (theta >= hi - bounds.bound_active_eps)).any(dim=1)
        conv = pg < tol
        nll_bits = float(mfull.genewise_loss_vector(theta=theta).sum())
        result.update(
            converged=int(conv.sum()),
            interior_pd=int((conv & (lam_min > tol) & ~bound_active).sum()),
            bound_active=int(bound_active.sum()),
            unconverged=int((~conv).sum()),
            premature_drops=int((was_dropped & ~conv).sum()),
            pg_max=float(pg.max()),
            loss_bits=nll_bits, loss_nats=nll_bits * math.log(2),
        )
        del mfull; torch.cuda.empty_cache()
    return result
