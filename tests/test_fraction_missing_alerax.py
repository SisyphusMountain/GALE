"""Fraction-missing forward likelihood at fixed rates, against an outside reference.

The fixture is a 3-taxon species tree and gene tree, at fixed rates D=L=T=0.1, with per-species
fraction-missing 0.0 and 0.3. Until GALE 1.0 was corrected (CHANGELOG.md) this test ran AleRax
1.4.0's ``alerax`` binary on it and compared the two per-family log-likelihoods. AleRax 1.4.0 counts
once the term of a duplication whose two copies carry the two child clades of a split, and GALE
counts it twice (the two copies can carry the two clades in two ways), so AleRax 1.4.0's value is no
longer the reference here: on this fixture the two differ by 2.2e-3 nats at fraction-missing 0 and
by 2.1e-2 nats at 0.3.

The reference is now GALE 2 (the development version of GALE: its own kernels, written separately)
with ``[model] duplication_split_factor = 2``, at the same rates and fraction-missing values, fully
converged, in float64. With that factor set to 1, GALE 2 reproduces AleRax 1.4.0 on this fixture:
at fraction-missing 0, -3.635615 nats against AleRax's -3.63562 (AleRax prints 5 decimals); at 0.3,
-3.071001 against -3.07125, AleRax being 2.5e-4 nats lower because it stops its extinction
iterations after 4 (its value lies between GALE's 4th and 5th iterations) while GALE iterates to
convergence. When AleRax 1.5.0, which counts the term twice, is released, its values for this
fixture will replace the references below.

Units / sign: GALE's ``stream_batches`` returns a negative log-likelihood in bits; the references
are log-likelihoods in nats, so ``logL_nats == -nll_bits * ln(2)``.

Fraction-missing in GALE is E-only, as in AleRax: it enters only the extinction ``E`` and carries no
term in the Pi/CLV numerator.

Note -- the Pi/CLV constant we deliberately did NOT add
------------------------------------------------------
The exact generative probability P(observe exactly this family) would additionally
multiply the numerator by a per-family CONSTANT ``prod_i (1 - f_{l_i})`` -- the
observation probability ``1 - f`` of each observed gene at its own (matched) leaf.
That factor is rate-INDEPENDENT: it does not change the fitted D/T/L rates, the
gradient, the HVP, or the reconciliation posterior (verified numerically to be an
identical shift across many rate sets). AleRax omits it (it computes a likelihood
conditioned on the observed genes), so we deliberately omit it too and gpurec matches
AleRax's convention. Adding it would only shift the absolute logL down by
``sum_i log(1 - f_{l_i})`` and move gpurec AWAY from AleRax.
"""
from __future__ import annotations

import math

import pytest
from gpurec.config import GpurecConfig

pytestmark = pytest.mark.gpu

_LN2 = math.log(2.0)

_SPECIES_NWK = "((A:1,B:1)AB:1,C:1)Root;"
_GENE_NWK = "((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;"
_D = _L = _T = 0.1
_FM = 0.3
# GALE 2's float64 log-likelihoods (nats) of the fixture, [model] duplication_split_factor = 2, E and
# Pi converged (e_tol 1e-13). Corrected GALE 1.0 gave the same values to 1.3e-15 nats.
_GALE2_NATS = {0.0: -3.6334329815493445, _FM: -3.049772164269457}
# Both programs iterate to convergence in float64: they agree to rounding.
_TOL_NATS = 1e-9


def _run_gpurec_nats(sp_path, g_path, fraction_missing):
    """gpurec forward NLL at theta=log2([D,L,T]) and ``fraction_missing``, in nats.

    Converted from gpurec's native NLL-in-bits via ``logL_nats = -nll_bits*ln(2)``.
    """
    import dataclasses

    import torch

    from gpurec.api._execution import stream_batches
    from gpurec.api.model import GeneReconModel

    m = GeneReconModel(
        str(sp_path),
        [str(g_path)],
        mode="global",
        dtype=torch.float64,
        fraction_missing=fraction_missing, clade_budget=GpurecConfig().batching.clade_budget
    )
    with torch.no_grad():
        m.theta.copy_(
            torch.log2(torch.tensor([_D, _L, _T], dtype=torch.float64, device=m.theta.device))
        )
    # Fully converge the wave solver for a converged-vs-converged comparison.
    m.solver_options = dataclasses.replace(m.solver_options, e_tol=1e-13, e_max_iter=4000, pi_iters=256)
    nll_bits = float(stream_batches(
        m.batch_statics, m.theta, m.receiver_weights, m.origination_weights,
        genewise=m.genewise, need_grad=False,
    )[0].item())
    return -nll_bits * _LN2


@pytest.fixture(scope="module")
def gpurec_numbers(tmp_path_factory):
    """gpurec's log-likelihood of the fixture (nats) at fraction-missing 0 and 0.3."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    workdir = tmp_path_factory.mktemp("fraction_missing_reference")
    (workdir / "sp.nwk").write_text(_SPECIES_NWK + "\n")
    (workdir / "g.nwk").write_text(_GENE_NWK + "\n")
    # fraction_missing=0.0 collapses to the no-missing fast path.
    return {fm: _run_gpurec_nats(workdir / "sp.nwk", workdir / "g.nwk", fm) for fm in _GALE2_NATS}


def test_base_model_matches_reference_fm0(gpurec_numbers):
    """fm=0: the base DTL forward (rate normalization, uniform-receiver transfer, survival and
    origination head) against the reference."""
    g, ref = gpurec_numbers[0.0], _GALE2_NATS[0.0]
    assert abs(g - ref) < _TOL_NATS, f"fm=0 mismatch: gpurec={g:.12f} nats, reference={ref:.12f} nats"


def test_fraction_missing_matches_reference_fm03(gpurec_numbers):
    """fm=0.3: the absolute forward log-likelihood with fraction-missing (E-only model) against the
    reference."""
    g, ref = gpurec_numbers[_FM], _GALE2_NATS[_FM]
    assert abs(g - ref) < _TOL_NATS, f"fm=0.3 mismatch: gpurec={g:.12f} nats, reference={ref:.12f} nats"
