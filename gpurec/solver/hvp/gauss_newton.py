"""``J^T``, the transpose of ``J = d(Pi_root)/dtheta``: the existing wave adjoint, reused with a
*custom root seed* and, optionally, with the loss's explicit E-norm term dropped (it is not part of
``d(Pi_root)/dtheta``). The exact HVP (``exact.py``) builds its point cache with it.

``vjp_root_to_theta`` is a thin wrapper over the production
``implicit_grad_loglik_vjp_wave`` (``gpurec.api._implicit_grad``): it unpacks ``static``/``sv``
and forwards, supplying a custom ``seed_root`` and ``drop_norm=True`` (the two knobs that turn the
real backward into ``J^T``). The single implementation is regression-gated by the real backward
(``test_optim_golden``) and the exact-HVP FD gate (``test_genewise_hvp``).
"""

from __future__ import annotations

import torch

from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.parameters.extract_parameters import resolve_accumulator_dtype


@torch.no_grad()
def vjp_root_to_theta(static, sv, seed_root, theta, receiver_weights, *, drop_norm,
                      neumann_terms, use_pruning, e_adjoint_tol, cache,
                      origination_log_probs, origination_probs,
                      reserved_scratch_bytes, warm_v, leaf_fm_log):
    """J^T applied to a root-score cotangent ``seed_root`` [n_root, S] -> ``(grad_theta [S,3], grad_col)``.

    Thin wrapper over the production ``implicit_grad_loglik_vjp_wave``: it unpacks ``static``/``sv``
    and forwards. With ``seed_root=None`` the loss seed ``-softmax2(Pi_root)`` is used and ``drop_norm``
    should be False to reproduce the real gradient (regression path); the GGN path passes a custom
    ``seed_root`` and ``drop_norm=True`` (drop the loss's explicit E-norm term, which is not part of
    d(Pi_root)/dtheta). ``neumann_terms``/``use_pruning``/``e_adjoint_tol`` override the solver options
    so the adjoint can be made convergent + unpruned to match the convergent Jvp (M = J^T B J
    symmetric). ``cache`` collects per-wave adjoint state for the exact-HVP tangent sweep.
    ``reserved_scratch_bytes`` mirrors the gradient path's memory-gate reservation (see
    ``_execution.py``): the caller sources it from ``static.warm_scratch_reserved_bytes`` only
    when warm-adjoint is resident for this static, else ``None`` (cold path unchanged) -- this
    wrapper just forwards it through to the gated fast path.
    """
    so = static.solver_options
    pi_state = sv["pi_state"]
    pi_state.validate(
        sv["pi_wave"], sv["pibar_wave"], sv["pibar_row_max"],
        check_values=False,
    )
    return implicit_grad_loglik_vjp_wave(
        static.wave_layout, static.species_helpers,
        Pi_star_wave=sv["pi_wave"], Pibar_star_wave=sv["pibar_wave"],
        E_star=sv["E"], E_s1=sv["E_s1"], E_s2=sv["E_s2"], Ebar=sv["Ebar"],
        log_pS=sv["log_pS"], log_pD=sv["log_pD"], log_pL=sv["log_pL"],
        max_transfer_mat=sv["max_transfer"], receiver_log_probs=sv["receiver_log_probs"],
        use_receiver_weights=not receiver_weights_are_uniform(receiver_weights),
        theta=theta, receiver_weights=receiver_weights,
        uniform_pibar_row_max=sv["pibar_row_max"], family_idx=static.rate_family_idx,
        leaf_fm_log=leaf_fm_log,
        specieswise=static.specieswise, genewise=static.genewise,
        neumann_terms=int(so.neumann_terms if neumann_terms is None else neumann_terms),
        e_adjoint_max_iter=so.e_adjoint_max_iter,
        e_adjoint_tol=(so.e_adjoint_tol if e_adjoint_tol is None else e_adjoint_tol),
        adjoint_pruning_threshold=so.adjoint_pruning_threshold,
        use_adjoint_pruning=bool(so.use_adjoint_pruning if use_pruning is None else use_pruning),
        pibar_side_threshold=so.pibar_side_threshold,
        collect_backward_relres=False,
        seed_root=seed_root, drop_norm=drop_norm, cache=cache,
        origination_log_probs=origination_log_probs, origination_probs=origination_probs,
        accumulator_dtype=resolve_accumulator_dtype(
            static.accumulator_dtype,
            fallback=sv["pi_wave"].dtype,
        ),
        reserved_scratch_bytes=reserved_scratch_bytes, warm_v=warm_v,
        pi_offset=pi_state.pi_offset,
        pibar_offset=pi_state.pibar_offset,
    )

