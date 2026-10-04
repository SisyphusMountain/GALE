"""Validation for the Neumann-series E-adjoint solver (``_neumann_e_adjoint``).

The backward E-adjoint solves ``(I - J) wE = q`` where ``J = d(E_from_E)/dE`` is the E-step
self-map Jacobian. The forward E fixed point converges, so ``J`` is a contraction (spectral
radius < 1) and the Neumann series ``(I-J)^{-1} = sum_k J^k`` converges. The E-adjoint is always
solved this way now (the old GMRES/BiCGSTAB solvers have been removed).

This file pins:
  1. solver-level correctness: ``_neumann_e_adjoint`` converges on a contraction operator
     ``A = I - J`` and matches a direct dense solve to tight tolerance,
  2. dtype-relative residual tolerances are respected (fp32 default/floor, fp64 tight residual),
  3. a non-contraction operator (spectral radius >= 1) is rejected rather than silently
     "converging".
"""
import warnings

import pytest
import torch

try:
    from gpurec.api._implicit_grad import _neumann_e_adjoint
except Exception as exc:  # pragma: no cover - import guard for triton-less envs
    pytest.skip(f"gpurec.api._implicit_grad unavailable: {exc}", allow_module_level=True)

from gpurec.config import dtype_rel_tol_default


def _e_adjoint_like(n: int, rho: float, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """``A = I - J`` with ``spectral_radius(J) = rho`` and a NON-symmetric J.

    This is the regime of the real E-adjoint ``I - J_E^T`` (rho(J_E) ~ 0.24) -- eigenvalues
    clustered near 1, small condition number, but non-symmetric.
    """
    g = torch.Generator().manual_seed(seed)
    J = torch.randn(n, n, generator=g, dtype=torch.float64)
    J = J - torch.diag(torch.diagonal(J))          # zero the diagonal for asymmetry
    J = rho * J / float(torch.linalg.matrix_norm(J, ord=2))  # scale so ||J||_2 = rho
    A = torch.eye(n, dtype=torch.float64) - J
    return A.to(dtype)


def _matvec(A):
    return lambda v: A @ v


def _rel_res(A, x, b):
    return float(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))


# ----------------------------------------------------------------------------
# 1. Solver-level: converges on a contraction operator, matches a direct solve.
# ----------------------------------------------------------------------------

def test_neumann_matches_direct_solve():
    for dtype, bar in ((torch.float32, 5e-6), (torch.float64, 1e-10)):
        A = _e_adjoint_like(71, rho=0.24, dtype=dtype, seed=7)
        assert float(torch.linalg.cond(A.double())) < 2.0  # ~1.6, like the real operator
        b = A @ torch.randn(71, dtype=dtype, generator=torch.Generator().manual_seed(8))

        x_neumann = _neumann_e_adjoint(_matvec(A), b)
        x_direct = torch.linalg.solve(A.double(), b.double())

        assert _rel_res(A, x_neumann, b) < bar
        rel_vs_direct = float(
            torch.linalg.vector_norm(x_neumann.double() - x_direct)
            / torch.linalg.vector_norm(x_direct)
        )
        assert rel_vs_direct < 10 * bar, f"dtype={dtype}: neumann vs direct rel={rel_vs_direct:.3e}"


def test_fp32_default_residual():
    A = _e_adjoint_like(64, rho=0.3, dtype=torch.float32, seed=11)
    b = A @ torch.randn(64, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    x = _neumann_e_adjoint(_matvec(A), b)
    assert _rel_res(A, x, b) <= dtype_rel_tol_default(torch.float32)


def test_fp64_reaches_tight_residual():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float64, seed=12)
    xtrue = torch.randn(48, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    b = A @ xtrue
    x = _neumann_e_adjoint(_matvec(A), b, tol=1e-12)
    assert _rel_res(A, x, b) <= 1e-10
    assert float(torch.linalg.vector_norm(x - xtrue) / torch.linalg.vector_norm(xtrue)) < 1e-8


def test_subfloor_tol_warns_and_clamps():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float32, seed=13)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = _neumann_e_adjoint(_matvec(A), b, tol=1e-12)  # unreachable in fp32
    assert any("below the" in str(wi.message) for wi in w)
    assert _rel_res(A, x, b) < 5e-6


def test_none_tol_uses_dtype_default_no_warning():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float32, seed=14)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        x = _neumann_e_adjoint(_matvec(A), b, tol=None)
    assert _rel_res(A, x, b) <= dtype_rel_tol_default(torch.float32)


def test_zero_rhs_returns_zero():
    A = _e_adjoint_like(16, rho=0.1, dtype=torch.float64, seed=15)
    b = torch.zeros(16, dtype=torch.float64)
    x = _neumann_e_adjoint(_matvec(A), b)
    assert float(torch.linalg.vector_norm(x)) == 0.0


def test_non_contraction_raises():
    """rho >= 1 -> not a contraction -> the Neumann series must not silently converge."""
    A = _e_adjoint_like(32, rho=1.5, dtype=torch.float64, seed=16)
    b = A @ torch.randn(32, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    with pytest.raises(RuntimeError):
        _neumann_e_adjoint(_matvec(A), b, max_iter=64)
