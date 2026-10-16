"""Memory-policy / infra knobs for CUDA memory-budget heuristics and cache-empty gates.

``MemoryOptions`` is the single source of truth for these values; the code reads them from it.
These knobs affect memory/performance decisions only, never numerics -- no gradient or
likelihood value depends on them. The two ``min_free_gib_*`` fields are DISTINCT, intentionally
different gates (driver-cache-empty vs HVP-rebuild), not a numeric inconsistency to collapse.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryOptions:
    """Memory/infra knobs for CUDA budget heuristics and cache-empty gates (no numeric effect)."""
    fraction: float = 0.85          # usable frac of total GPU mem
    reserve_gib: float = 1.0        # headroom GiB
    scratch_tensors: int = 10       # [W,S]-tensor multiplier in wave mem estimate
    min_free_gib_driver: float = 4.0  # value_and_grad cache-empty gate
    min_free_gib_hvp: float = 8.0     # curvature HVP-rebuild gate
    free_cache_every: int = 32        # empty CUDA cache every-K-waves
    grad_avg_k: int = 1               # backward passes averaged
