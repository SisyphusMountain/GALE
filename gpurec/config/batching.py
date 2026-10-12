"""How the gene families are cut into GPU batches (``[batching]`` of ``defaults.toml``).

``clade_budget`` is the most clades one batch may hold. It is the only batching setting the
release reads: it fixes the batch size of every fit and evaluation. It shapes the batches, never the
likelihood or its gradient, so changing it changes the order of floating-point additions only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatchingOptions:
    """``clade_budget``: the most clades one batch may hold (a whole number of at least 1)."""
    clade_budget: int = 315_000

    def validate(self) -> None:
        value = self.clade_budget
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"[batching] clade_budget must be a positive whole number, got {value!r}")
