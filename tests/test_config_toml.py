"""Tests for ``gpurec/config/defaults.toml`` and the ``load_fit_config`` loader.

``test_defaults_toml_matches_dataclass_defaults`` is the PERMANENT guard that
``gpurec/config/defaults.toml`` never drifts from the ``GpurecConfig`` dataclass
defaults (the source of truth). If a dataclass default changes, this test fails
until ``defaults.toml`` is updated to match (or vice versa).
"""
import tomllib

import pytest
import torch

from gpurec.config import (
    DEFAULTS_TOML_PATH,
    GpurecConfig,
    PrecisionOptions,
    load_fit_config,
    resolve_torch_dtype,
)
from gpurec.config.gpurec_config import _merge_into


def _defaults_toml_config():
    """``defaults.toml`` deep-merged onto ``GpurecConfig()`` (the merge ``load_fit_config`` uses)."""
    with open(DEFAULTS_TOML_PATH, "rb") as fh:
        config = _merge_into(GpurecConfig(), tomllib.load(fh), path="")
    config.validate()
    return config


def test_defaults_toml_matches_dataclass_defaults():
    assert _defaults_toml_config() == GpurecConfig()


def test_defaults_toml_path_is_package_relative_and_exists():
    # Package-relative (Path(__file__).parent-derived), not a cwd-relative string.
    assert DEFAULTS_TOML_PATH.is_absolute()
    assert DEFAULTS_TOML_PATH.name == "defaults.toml"
    assert DEFAULTS_TOML_PATH.exists()


def test_precision_defaults_are_toml_safe_and_resolve_to_torch_dtypes():
    precision = PrecisionOptions()
    precision.validate()

    assert precision.model_dtype == "float32"
    assert precision.accumulator_dtype == "float64"
    assert precision.model_torch_dtype == torch.float32
    assert precision.accumulator_torch_dtype == torch.float64
    assert resolve_torch_dtype(" FLOAT64 ") == torch.float64


def test_precision_validation_normalizes_names_and_rejects_narrow_accumulator():
    precision = PrecisionOptions(model_dtype=" FLOAT32 ", accumulator_dtype=" FLOAT64 ")
    precision.validate()
    assert precision == PrecisionOptions()

    with pytest.raises(ValueError, match="at least as wide"):
        PrecisionOptions(model_dtype="float64", accumulator_dtype="float32").validate()
    with pytest.raises(ValueError, match="at least as wide"):
        PrecisionOptions(accumulator_dtype="float32").validate(model_dtype=torch.float64)
    with pytest.raises(ValueError, match="unsupported floating-point dtype"):
        PrecisionOptions(model_dtype="bfloat16").validate()


def test_load_fit_config_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[solver]\nnot_a_real_field = 1\n")
    with pytest.raises(ValueError):
        load_fit_config(p)


def test_precision_partial_toml_override_merges_and_validates(tmp_path):
    p = tmp_path / "precision.toml"
    p.write_text('[precision]\nmodel_dtype = "float64"\n', encoding="utf-8")

    cfg = load_fit_config(p)
    cfg.validate()

    assert cfg.precision.model_dtype == "float64"
    assert cfg.precision.accumulator_dtype == "float64"


def test_none_only_fields_are_omitted_from_defaults_toml_and_stay_none():
    cfg = _defaults_toml_config()
    assert cfg.rates.max_rate is None
    assert cfg.regularizer.origination.depth is None
    assert cfg.regularizer.origination.root_index is None
    assert cfg.regularizer.origination.dirichlet_pi is None


def test_lambdas_tuple_default_coerced_from_toml_array():
    cfg = _defaults_toml_config()
    assert isinstance(cfg.regularizer.lambdas, tuple)
    assert cfg.regularizer.lambdas == (0.0, 1.0, 10.0, 100.0, 1000.0)
    # Equality against the dataclass default only holds if list -> tuple
    # coercion actually happened (TOML arrays decode to Python lists).
    assert cfg.regularizer == GpurecConfig().regularizer


def test_load_fit_config_rejects_sections_that_do_not_affect_fitting(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[memory]\nfraction = 0.5\n")
    with pytest.raises(ValueError, match="have no effect on fitting"):
        load_fit_config(p)


def test_load_fit_config_lists_every_inert_section_given(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[memory]\nfraction = 0.5\n[newton]\nseed = 1\n")
    with pytest.raises(ValueError, match="memory, newton"):
        load_fit_config(p)


def test_load_fit_config_accepts_a_solver_only_file(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text("[solver]\npi_iters = 128\n")
    cfg = load_fit_config(p)
    assert cfg.solver.pi_iters == 128


def test_load_fit_config_rejects_init_rate(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[rates]\ninit_rate = 0.5\n")
    with pytest.raises(ValueError, match="init_rate"):
        load_fit_config(p)


def test_load_fit_config_accepts_the_rate_box(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text("[rates]\nmin_rate = 1e-5\nmax_rate = 1.5\n")
    cfg = load_fit_config(p)
    assert (cfg.rates.min_rate, cfg.rates.max_rate) == (1e-5, 1.5)


def test_load_fit_config_rejects_e_tangent_tol(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[solver]\ne_tangent_tol = 1e-8\n")
    with pytest.raises(ValueError, match="e_tangent_tol"):
        load_fit_config(p)


def test_load_fit_config_accepts_a_solver_table_with_only_e_max_iter(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text("[solver]\ne_max_iter = 200\n")
    cfg = load_fit_config(p)
    assert cfg.solver.e_max_iter == 200


def test_load_fit_config_accepts_use_hvp_warm_start_and_the_per_family_fit_reads_it(tmp_path):
    # The per-family fit builds its SolverOptions from the keys of _BASE_SOLVER
    # (gpurec/fit/genewise_fit.py); use_hvp_warm_start is one of them, so the value of the
    # config file reaches the exact Hessian. (Switching it off keeps the exact Hessian from
    # holding one warm-start array per batch and per probe, which runs out of memory on a
    # 24 GB card when the batches are small.)
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    p = tmp_path / "ok.toml"
    p.write_text("[solver]\nuse_hvp_warm_start = false\n")
    cfg = load_fit_config(p)
    assert cfg.solver.use_hvp_warm_start is False
    assert "use_hvp_warm_start" in _BASE_SOLVER
