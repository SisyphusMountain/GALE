"""The L-BFGS-B settings of the cladewise fit (``[cladewise]`` of ``defaults.toml``).

The fit's forward solves follow ``[solver]`` (``pi_iters``, ``neumann_terms``) as they are; only the
optimiser's own stopping rules live here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CladewiseOptions:
    """``max_iter``: most L-BFGS-B iterations; ``ftol``, ``gtol``: its relative-reduction and
    projected-gradient stops; ``maxcor``: the corrections it keeps. A fit that stops on
    ``max_iter`` or fails raises ``RuntimeError``."""
    max_iter: int
    ftol: float
    gtol: float
    maxcor: int

    def validate(self) -> None:
        for name in ("max_iter", "maxcor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"[cladewise] {name} must be a positive whole number, got {value!r}")
        for name in ("ftol", "gtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"[cladewise] {name} must be a positive number, got {value!r}")
