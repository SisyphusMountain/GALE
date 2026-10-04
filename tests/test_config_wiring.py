"""Tests for wiring ``GpurecConfig`` into ``GeneReconModel`` and into the fit entry points.

Covers:
  (a) ``GeneReconModel(..., config=...)`` uses ``config.solver`` as ``solver_options`` when no
      explicit ``solver_options`` is passed, and that an explicit ``solver_options`` still wins.
  (b) threading ``GpurecConfig`` into the fit entry points (``fit_genewise``/``optimize``) -- see
      the section near the bottom of this file.

(There used to be a section here covering the ``gpurec`` command line; the command line was
removed when fitting moved to ``gpurec.interface.fit``/``evaluate`` over a parsed ``FamilyData``,
so those tests are gone too -- see ``tests/test_interface_fit.py``.)
"""
from pathlib import Path
import tempfile

import pytest
import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.config import GpurecConfig, PrecisionOptions
from gpurec.config.rates import RateBounds
from gpurec.solver.penalties import OriginationPenalty, PenaltyOptions


def _build_model(**kwargs):
    from gpurec.api.model import GeneReconModel

    with tempfile.TemporaryDirectory() as tmp:
        species_tree = Path(tmp) / "species.nwk"
        gene_tree = Path(tmp) / "family_0.nwk"
        species_tree.write_text("(A:1,B:1)Root:1;\n", encoding="utf-8")
        gene_tree.write_text("(A_1:1,B_1:1)GeneRoot:1;\n", encoding="utf-8")
        return GeneReconModel(
            species_tree, [gene_tree],
            family_chunk_size=1, clade_budget=None,
            batch_packing="sequential", max_wave_size=4,
            **kwargs,
        )


# --- (a) model constructor -------------------------------------------------

@pytest.mark.gpu
def test_model_uses_config_solver_when_no_explicit_solver_options():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    cfg = GpurecConfig(solver=SolverOptions(pi_iters=32))
    model = _build_model(config=cfg)
    assert model.solver_options.pi_iters == 32


@pytest.mark.gpu
def test_model_explicit_solver_options_wins_over_config():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    cfg = GpurecConfig(solver=SolverOptions(pi_iters=32))
    explicit = SolverOptions(pi_iters=16)
    model = _build_model(config=cfg, solver_options=explicit)
    assert model.solver_options.pi_iters == 16


@pytest.mark.gpu
def test_model_config_none_reproduces_default_solver_options():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    model = _build_model()
    assert model.solver_options == SolverOptions()


@pytest.mark.gpu
def test_model_uses_config_precision_and_copies_it_to_batch_statics():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float64", accumulator_dtype="float64")
    )
    model = _build_model(config=cfg)

    assert model.theta.dtype == torch.float64
    assert model.dtype == torch.float64
    assert model.accumulator_dtype == torch.float64
    assert model.precision_options == cfg.precision
    assert all(static.precision_options == cfg.precision for static in model.batch_statics)
    assert all(static.accumulator_dtype == torch.float64 for static in model.batch_statics)
    assert model.species_helpers["unnorm_row_max"].dtype == torch.float64
    for static in model.batch_statics:
        for meta in static.wave_layout["wave_metas"]:
            if "log_split_probs" in meta:
                assert meta["log_split_probs"].dtype == torch.float64


@pytest.mark.gpu
def test_model_float32_accumulator_reaches_preprocessing_statics():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float32", accumulator_dtype="float32")
    )
    model = _build_model(config=cfg)

    assert model.accumulator_dtype == torch.float32
    assert model.species_helpers["unnorm_row_max"].dtype == torch.float32
    for static in model.batch_statics:
        for meta in static.wave_layout["wave_metas"]:
            if "log_split_probs" in meta:
                assert meta["log_split_probs"].dtype == torch.float32
                assert set(meta["_log_split_probs_by_dtype"]) == {
                    torch.float32,
                    torch.float64,
                }


@pytest.mark.gpu
def test_model_mixed_precision_materializes_each_preprocessing_value_in_its_owner_dtype():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float32", accumulator_dtype="float64")
    )
    model = _build_model(config=cfg)

    # Transfer normalization metadata belongs to accumulator-domain reductions.
    assert model.species_helpers["unnorm_row_max"].dtype == torch.float64
    # Split probabilities are direct dense-kernel inputs and must arrive in the
    # model dtype rather than being narrowed inside every Triton program.
    for static in model.batch_statics:
        for meta in static.wave_layout["wave_metas"]:
            if "log_split_probs" in meta:
                assert meta["log_split_probs"].dtype == torch.float32


@pytest.mark.gpu
def test_model_explicit_dtype_overrides_config_precision():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float64", accumulator_dtype="float64")
    )
    model = _build_model(config=cfg, dtype=torch.float32)

    assert model.theta.dtype == torch.float32
    assert model.precision_options.model_dtype == "float32"
    # Resolving an explicit override must not mutate the caller's config.
    assert cfg.precision.model_dtype == "float64"


def test_model_rejects_accumulator_narrower_than_effective_model_dtype():
    """Precision validation raises inside GeneReconModel.__init__ before the CUDA-only
    ``_setup()`` runs, so this does not need a GPU."""
    cfg = GpurecConfig(
        precision=PrecisionOptions(model_dtype="float64", accumulator_dtype="float32")
    )
    with pytest.raises(ValueError, match="at least as wide"):
        _build_model(config=cfg)


# ================================================================================================
# Thread GpurecConfig into fit_genewise / optimize
#
# Each helper below monkeypatches the ONE heavy call (GeneReconModel construction / first_order /
# the model-building + inner-fit helpers) so the wiring can be asserted WITHOUT a CUDA/Triton
# solve -- these entry points are otherwise expensive end-to-end recipes. The golden/HVP gate
# (test_optim_golden.py / test_genewise_hvp.py) is the real end-to-end value-additive proof; these
# are fast unit-level checks of the resolution precedence itself.
# ================================================================================================

# --- fit_genewise ----------------------------------------------------------

def _run_fit_genewise_capture(monkeypatch, **kwargs):
    """Run ``fit_genewise`` with a stubbed ``FamilyData.build_model`` (never actually touched --
    adam_steps=0, max_iter=0, certify=False mean the tier loop builds a model but never calls into
    it), so the config threading can be asserted without a CUDA/Triton solve. Returns
    ``(solver_options_used, the_full_result_dict)``.
    """
    from gpurec.data import FamilyData
    from gpurec.fit.genewise_fit import fit_genewise

    captured = {}

    class _StubModel:
        def __init__(self):
            self.receiver_weights = torch.zeros(1)

    def _fake_build_model(self, indices, mode, solver_options, config, clade_budget):
        captured["solver_options"] = solver_options
        return _StubModel()

    monkeypatch.setattr(FamilyData, "build_model", _fake_build_model)
    data = FamilyData(
        species_tree_path=Path("sp.nwk"), species=None, species_raw={}, families=(),
        names=("g",), covered_species=(), dropped=(), fraction_missing=None,
    )
    # Deliberately far below any configured floor, so a test that cares about the clamped value
    # can tell the configured min_rate apart from "happened to already be in bounds".
    start = torch.full((1, 3), -100.0, dtype=torch.float64)
    res = fit_genewise(
        data, start_log2_rates=start, adam_steps=0, pi_tiers=(16,),
        max_iter=0, certify=False, verbose=False, **kwargs,
    )
    return captured["solver_options"], res


def test_fit_genewise_config_solver_field_threaded(monkeypatch):
    cfg = GpurecConfig(solver=SolverOptions(e_max_iter=999), rates=RateBounds(max_rate=2.0))
    so, _res = _run_fit_genewise_capture(monkeypatch, config=cfg)
    assert so.e_max_iter == 999
    # pi_iters/neumann_terms are NOT part of _BASE_SOLVER's key subset -- fit_genewise's per-tier
    # `sopts()` always overrides them from `pi_tiers`/`neu_opt`, so config.solver.pi_iters has no
    # effect here.
    assert so.pi_iters == 16


def test_fit_genewise_config_rates_is_used_as_the_bounds_directly(monkeypatch):
    """``bounds = config.rates`` (no separate merge step any more): the far-below-floor start
    theta must be clamped up to exactly ``config.rates.min_rate``."""
    import math

    cfg = GpurecConfig(rates=RateBounds(min_rate=1e-4, max_rate=3.0))
    _so, res = _run_fit_genewise_capture(monkeypatch, config=cfg)
    assert float(res["theta"][0, 0]) == pytest.approx(math.log2(1e-4))


# --- optimize ----------------------------------------------------------

def _run_optimize_capture(monkeypatch, **kwargs):
    """Run ``optimize`` with a stubbed ``first_order`` (returns ``theta0`` unchanged) and
    ``polish_mode="none"`` (skips the Newton polish stage entirely) so no CUDA/Triton solve ever
    runs. Returns the ``origination_penalty`` ``first_order`` was actually called with.
    """
    # NOT `import gpurec.fit.optimize as opt_mod`: gpurec/__init__.py exports a top-level `fit`
    # function (gpurec.interface.fit), which shadows the `gpurec.fit` SUBPACKAGE as an attribute of
    # the `gpurec` module. `import gpurec.fit.optimize` resolves `gpurec.fit` through that shadowed
    # attribute and fails; `from gpurec.fit import optimize` resolves via sys.modules instead and is
    # unaffected.
    from gpurec.fit import optimize as opt_mod

    captured = {}

    def _fake_first_order(batch_statics, theta0, receiver_weights, **fkwargs):
        captured["origination_penalty"] = fkwargs.get("origination_penalty")
        return theta0, [], None

    monkeypatch.setattr(opt_mod, "first_order", _fake_first_order)
    opt_mod.optimize(None, torch.zeros(3), torch.zeros(3), verbose=False, polish_mode="none", **kwargs)
    return captured["origination_penalty"]


def test_optimize_config_sets_origination_penalty(monkeypatch):
    op = OriginationPenalty(l2=5.0)
    cfg = GpurecConfig(regularizer=PenaltyOptions(origination=op))
    got = _run_optimize_capture(monkeypatch, config=cfg)
    assert got is op


def test_optimize_explicit_origination_penalty_wins_over_config(monkeypatch):
    op_cfg = OriginationPenalty(l2=5.0)
    op_explicit = OriginationPenalty(l2=9.0)
    cfg = GpurecConfig(regularizer=PenaltyOptions(origination=op_cfg))
    got = _run_optimize_capture(monkeypatch, config=cfg, origination_penalty=op_explicit)
    assert got is op_explicit


def test_optimize_config_none_reproduces_today_default_origination_penalty(monkeypatch):
    got = _run_optimize_capture(monkeypatch)
    assert got is None


def test_default_origination_penalty_is_noop_equivalent_to_none():
    """VERIFY the brief's claimed equivalence: a default (all-zero, disabled) ``OriginationPenalty``
    -- what ``config.regularizer.origination`` resolves to for a default ``GpurecConfig()`` -- is a
    byte-exact no-op, identical to never applying a penalty at all (``origination_penalty=None``)."""
    from gpurec.solver.penalties import origination_penalty_and_grad

    default_pen = GpurecConfig().regularizer.origination
    assert isinstance(default_pen, OriginationPenalty)
    assert not default_pen.any_active()
    omega = torch.randn(5, dtype=torch.float64)
    pen, grad = origination_penalty_and_grad(omega, default_pen)
    assert float(pen) == 0.0
    assert torch.equal(grad, torch.zeros_like(omega))
