"""Forward solve over gpurec's batched streaming evaluators (``forward_solve``).

Port of kernel-bench ``newton/vg.py``, re-pointed from kbench's single-``static``
``run_forward`` onto gpurec's multi-batch ``stream_batches``. gpurec carries the E
warm-start on each ``static.warm_E`` and refreshes it internally.

Terminology rename applied at the kbench -> gpurec port boundary: item->family,
col->receiver, state->species, solve_e_pi->solve_resident_e_pi,
col_weights->receiver_weights, and max_coupling->max_transfer. The legacy
receiver-column terminology has been migrated to receiver log weights.
"""

from __future__ import annotations

import torch

from gpurec.api._batch_state import _BatchStatic
from gpurec.api._execution import stream_batches
from gpurec.config.memory import MemoryOptions
from gpurec.core.inference.solver import nll_from_root_rows, solve_resident_e_pi

# Names of the forward-solve intermediates the exact-HVP / tangent path consumes,
# in the order ``solve_resident_e_pi`` returns them. gpurec rename of kbench's
# FORWARD_SAVED_NAMES: max_coupling -> max_transfer and the legacy column-log
# probabilities -> receiver_log_probs.
#
# Row-gauged storage has no extra tuple entry: ``forward_solve`` snapshots the
# exact sidecar produced by this solve under ``saved["pi_state"]``.
# Keeping it out of this tuple preserves the long-standing return contract and
# ensures derivative consumers never read offsets from a later solve through
# the mutable batch-static attribute.
FORWARD_SAVED_NAMES = (
    "E", "E_s1", "E_s2", "Ebar", "root_rows", "pi_wave", "pibar_wave",
    "pibar_row_max", "log_pS", "log_pD", "log_pL", "max_transfer", "receiver_log_probs",
)


def free_cuda_cache_if_tight(min_free_gib: float = MemoryOptions().min_free_gib_driver):
    """Release the caching allocator's pool to the driver when driver-free memory runs low.

    The hand-written backward gates its scratch on ``torch.cuda.mem_get_info`` (driver-free
    bytes), which the caching allocator does not replenish on tensor free -- without this,
    long optimization loops on the big problems trip the gate spuriously.
    """
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        if free_b < min_free_gib * (1024 ** 3):
            torch.cuda.empty_cache()


def _as_static_list(batch_statics):
    """Accept either a single ``_BatchStatic`` or a list of them; return a list."""
    if isinstance(batch_statics, _BatchStatic):
        return [batch_statics]
    return list(batch_statics)


def forward_solve(batch_statics, theta, receiver_weights, *, warm_E=None):
    """Run the forward solve at ``theta``; return ``(loss_tensor, saved)``.

    For a SINGLE batch (the exact-HVP / Newton-polish regime) ``saved`` is the dict of
    forward intermediates the second-order kernels consume (keyed by
    :data:`FORWARD_SAVED_NAMES`). For multiple batches ``saved`` is None -- the HVP path is
    single-batch.
    """
    statics = _as_static_list(batch_statics)
    with torch.no_grad():
        if len(statics) == 1:
            static = statics[0]
            theta_b = (
                theta.index_select(0, static.family_index_tensor) if static.genewise else theta
            )
            out = solve_resident_e_pi(
                static, theta_b, receiver_weights,
                warm_start_E=warm_E if warm_E is not None else static.warm_E,
            )
            saved = dict(zip(FORWARD_SAVED_NAMES, out))
            saved["pi_state"] = getattr(static, "pi_forward_state", None)
            if saved["pi_state"] is None:
                raise RuntimeError("Pi forward did not publish its row-offset state")
            loss = nll_from_root_rows(
                saved["root_rows"],
                saved["E"],
                accumulator_dtype=getattr(static, "accumulator_dtype", None),
            )
            return loss, saved
        loss, _g, _gr, _go = stream_batches(
            statics, theta, receiver_weights, torch.zeros_like(receiver_weights),
            genewise=statics[0].genewise, need_grad=False,
        )
        return loss, None

