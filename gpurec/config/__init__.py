"""Centralized configuration package for gpurec.

Exposes the shared dtype-relative tolerance helpers used by the solver and
implicit-gradient code, the per-area option dataclasses, and the top-level
``GpurecConfig`` that composes them.
"""
from gpurec.config.gpurec_config import (
    DEFAULTS_TOML_PATH,
    GpurecConfig,
    dtype_rel_tol_default,
    dtype_rel_tol_floor,
    load_fit_config,
)
from gpurec.config.memory import MemoryOptions
from gpurec.config.newton import NewtonOptions
from gpurec.config.precision import (
    PrecisionOptions,
    normalize_dtype_name,
    resolve_torch_dtype,
    torch_dtype_name,
)
from gpurec.config.rates import RateBounds
from gpurec.api.solver_options import SolverOptions

__all__ = [
    "dtype_rel_tol_default", "dtype_rel_tol_floor",
    "GpurecConfig", "SolverOptions", "PrecisionOptions", "NewtonOptions", "RateBounds",
    "MemoryOptions", "normalize_dtype_name", "resolve_torch_dtype",
    "torch_dtype_name",
    "DEFAULTS_TOML_PATH", "load_fit_config",
]

