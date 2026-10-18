import pytest

from gpurec.config import GpurecConfig
from gpurec.api.solver_options import SolverOptions
from gpurec.config.memory import MemoryOptions
from gpurec.config.newton import NewtonOptions
from gpurec.config.rates import RateBounds
from gpurec.solver.penalties import PenaltyOptions


def test_gpurec_config_builds_with_defaults():
    cfg = GpurecConfig()
    assert isinstance(cfg.solver, SolverOptions)
    assert isinstance(cfg.newton, NewtonOptions)
    assert isinstance(cfg.rates, RateBounds)
    assert isinstance(cfg.regularizer, PenaltyOptions)
    assert isinstance(cfg.memory, MemoryOptions)


def test_gpurec_config_default_equals_composed_defaults():
    cfg = GpurecConfig()
    assert cfg.solver == SolverOptions()
    assert cfg.newton == NewtonOptions()
    assert cfg.rates == RateBounds()
    assert cfg.regularizer == PenaltyOptions()
    assert cfg.memory == MemoryOptions()


def test_gpurec_config_validate_delegates():
    cfg = GpurecConfig()
    cfg.validate()  # must not raise

    bad = GpurecConfig(solver=SolverOptions(e_max_iter=0))
    with pytest.raises(ValueError):
        bad.validate()

    bad_newton = GpurecConfig(newton=NewtonOptions(max_newton=0))
    with pytest.raises(ValueError):
        bad_newton.validate()


def test_gpurec_config_reexports_in_config_package():
    from gpurec.config import GpurecConfig as ReExported
    from gpurec.config import MemoryOptions as MemoryReExported
    from gpurec.config import NewtonOptions as NewtonReExported
    from gpurec.config import RateBounds as RateReExported
    from gpurec.config import SolverOptions as SolverReExported

    assert ReExported is GpurecConfig
    assert SolverReExported is SolverOptions
    assert NewtonReExported is NewtonOptions
    assert RateReExported is RateBounds
    assert MemoryReExported is MemoryOptions
