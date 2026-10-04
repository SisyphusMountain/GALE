"""Tests for ``gpurec/config/defaults.toml`` and the ``from_toml``/``load_config`` loaders.

``test_defaults_toml_matches_dataclass_defaults`` is the PERMANENT guard that
``gpurec/config/defaults.toml`` never drifts from the ``GpurecConfig`` dataclass
defaults (the source of truth). If a dataclass default changes, this test fails
until ``defaults.toml`` is updated to match (or vice versa).
"""
import dataclasses

import pytest
import torch

from gpurec.config import (
    DEFAULTS_TOML_PATH,
    GpurecConfig,
    PrecisionOptions,
    load_config,
    load_fit_config,
    resolve_torch_dtype,
)


def test_defaults_toml_matches_dataclass_defaults():
    assert GpurecConfig.from_toml(DEFAULTS_TOML_PATH) == GpurecConfig()


def test_defaults_toml_path_is_package_relative_and_exists():
    # Package-relative (Path(__file__).parent-derived), not a cwd-relative string.
    assert DEFAULTS_TOML_PATH.is_absolute()
    assert DEFAULTS_TOML_PATH.name == "defaults.toml"
    assert DEFAULTS_TOML_PATH.exists()


def test_user_toml_overrides_merge():
    cfg = GpurecConfig.from_dict({"solver": {"pi_iters": 128}})
    assert cfg.solver.pi_iters == 128 and cfg.newton.max_cg == 40


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


def test_from_toml_partial_file_merges_onto_defaults(tmp_path):
    p = tmp_path / "user.toml"
    p.write_text("[solver]\npi_iters = 128\n")
    cfg = GpurecConfig.from_toml(p)
    default = GpurecConfig()

    assert cfg.solver.pi_iters == 128
    for f in dataclasses.fields(default.solver):
        if f.name == "pi_iters":
            continue
        assert getattr(cfg.solver, f.name) == getattr(default.solver, f.name)
    assert cfg.newton == default.newton
    assert cfg.precision == default.precision
    assert cfg.rates == default.rates
    assert cfg.regularizer == default.regularizer
    assert cfg.memory == default.memory


def test_from_toml_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[solver]\nnot_a_real_field = 1\n")
    with pytest.raises(ValueError):
        GpurecConfig.from_toml(p)


def test_load_config_none_returns_defaults():
    assert load_config() == GpurecConfig()
    assert load_config(None) == GpurecConfig()


def test_load_config_with_path_loads_and_merges(tmp_path):
    p = tmp_path / "user.toml"
    p.write_text("[rates]\nmin_rate = 1e-8\n")
    cfg = load_config(p)
    default = GpurecConfig()

    assert cfg.rates.min_rate == 1e-8
    assert cfg.rates.max_rate == default.rates.max_rate  # untouched None default
    assert cfg.solver == default.solver
    assert cfg.precision == default.precision
    assert cfg.newton == default.newton
    assert cfg.regularizer == default.regularizer
    assert cfg.memory == default.memory


def test_precision_partial_toml_override_merges_and_validates(tmp_path):
    p = tmp_path / "precision.toml"
    p.write_text('[precision]\nmodel_dtype = "float64"\n', encoding="utf-8")

    cfg = GpurecConfig.from_toml(p)
    cfg.validate()

    assert cfg.precision.model_dtype == "float64"
    assert cfg.precision.accumulator_dtype == "float64"


def test_none_only_fields_are_omitted_from_defaults_toml_and_stay_none():
    cfg = GpurecConfig.from_toml(DEFAULTS_TOML_PATH)
    assert cfg.rates.max_rate is None
    assert cfg.regularizer.origination.depth is None
    assert cfg.regularizer.origination.root_index is None
    assert cfg.regularizer.origination.dirichlet_pi is None


def test_lambdas_tuple_default_coerced_from_toml_array():
    cfg = GpurecConfig.from_toml(DEFAULTS_TOML_PATH)
    assert isinstance(cfg.regularizer.lambdas, tuple)
    assert cfg.regularizer.lambdas == (0.0, 1.0, 10.0, 100.0, 1000.0)
    # Equality against the dataclass default only holds if list -> tuple
    # coercion actually happened (TOML arrays decode to Python lists).
    assert cfg.regularizer == GpurecConfig().regularizer


def test_tuple_field_override_from_dict_list_is_coerced_to_tuple():
    cfg = GpurecConfig.from_dict({"regularizer": {"lambdas": [1.0, 2.0]}})
    assert cfg.regularizer.lambdas == (1.0, 2.0)
    assert isinstance(cfg.regularizer.lambdas, tuple)


def test_tuple_field_override_from_toml_file_is_coerced_to_tuple(tmp_path):
    p = tmp_path / "user.toml"
    p.write_text("[regularizer]\nlambdas = [1.0, 2.0, 3.0]\n")
    cfg = GpurecConfig.from_toml(p)
    assert cfg.regularizer.lambdas == (1.0, 2.0, 3.0)
    assert isinstance(cfg.regularizer.lambdas, tuple)


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


def test_load_fit_config_rejects_use_hvp_warm_start(tmp_path):
    # The per-family fit builds its SolverOptions from a fixed key whitelist that
    # excludes use_hvp_warm_start (gpurec/fit/genewise_fit.py's _BASE_SOLVER), so a
    # config value never reaches it; the global fit never computes an HVP at all.
    # It has no effect on any fit/evaluate/sampling path, so it is rejected too.
    p = tmp_path / "bad.toml"
    p.write_text("[solver]\nuse_hvp_warm_start = false\n")
    with pytest.raises(ValueError, match="use_hvp_warm_start"):
        load_fit_config(p)
