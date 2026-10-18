# gpurec/optim/penalties.py
"""Settings of the ``[regularizer]`` section of ``GpurecConfig`` (``PenaltyOptions``).

All quantities are in log2 (bits) units. The penalty functions these settings were written for are
not part of the release; ``load_fit_config`` rejects a ``[regularizer]`` section.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

DEFAULT_TV_EPS = 1e-3


@dataclass
class OriginationPenalty:
    l2: float = 0.0
    depth_lambda: float = 0.0
    depth: "torch.Tensor | None" = None
    root_lambda: float = 0.0
    root_index: "int | None" = None
    dirichlet_c: float = 0.0
    barrier_kind: str = "meanlog"   # meanlog | simpson | renyi2
    dirichlet_pi: "torch.Tensor | None" = None
    floor: float = 0.0


@dataclass
class PenaltyOptions:
    """Facade grouping the existing regularizer config into one object: the origination
    anti-collapse penalty, the TV-prior epsilon, and the ridge-homotopy defaults. Purely additive -- constructing
    this does not change any existing penalty call site; ``OriginationPenalty``/``DEFAULT_TV_EPS``
    remain independently importable and unchanged.
    """
    origination: OriginationPenalty = field(default_factory=OriginationPenalty)
    tv_eps: float = DEFAULT_TV_EPS
    lambdas: "tuple[float, ...]" = (0.0, 1.0, 10.0, 100.0, 1000.0)
    lam_margin: float = 1.3
    lam_floor: float = 1e-3
