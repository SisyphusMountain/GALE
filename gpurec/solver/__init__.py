"""Stable optimization toolkit: forward solve, CG/Lanczos, curvature + exact HVP, tangents."""
from gpurec.solver.penalties import OriginationPenalty
from gpurec.solver.value_and_grad import (
    FORWARD_SAVED_NAMES,
    forward_solve,
    free_cuda_cache_if_tight,
)

__all__ = [
    "FORWARD_SAVED_NAMES", "forward_solve", "free_cuda_cache_if_tight",
    "OriginationPenalty",
]
