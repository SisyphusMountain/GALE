# tests/test_backtracking_matches_alerax.py
"""Outside check of gpurec's backtracking sampler: the species each sampled history starts at.

test_fraction_missing_alerax.py checks gpurec's forward LIKELIHOOD (a single number per family)
against an outside reference, with and without fraction-missing. This test goes one level deeper:
it checks that gpurec's *sampler* -- the native Rust code in crates/gpurec-backtrack that draws
concrete reconciliations -- picks the species each history starts at (the "root species", the
first node of a sample of the native ``sample_reconciliations_many``, drawn with weights
``Pi(root_clade, .)``) with the right PROBABILITIES, not just the right total likelihood. Two
implementations can agree on a sum while disagreeing on how that sum is split up; this test would
catch that.

Until GALE 1.0 was corrected (CHANGELOG.md), the reference was AleRax 1.4.0's own sampler (``-g N
--seed S``; the ``origination`` column of its ``family_0_speciesEventCounts_<k>.txt``), compared
with a two-sample chi-square test. AleRax 1.4.0 counts once the term of a duplication whose two
copies carry the two child clades of a split, and GALE counts it twice (the two copies can carry
the two clades in two ways), so on this fixture, where duplication and transfer compete, AleRax
1.4.0 draws a different distribution. The reference is now the exact distribution
``2^Pi(root_clade, s) / sum_s 2^Pi(root_clade, s)`` computed by GALE 2 (the development version of
GALE: its own kernels, written separately) with ``[model] duplication_split_factor = 2``, in float64,
and the draws are compared with it by a chi-square goodness-of-fit test. Corrected GALE 1.0's own
Pi row gives the same probabilities to 2.3e-10. When AleRax 1.5.0, which counts the term twice, is
released, a comparison with its sampler can come back here.

Gene tree rooting, for a comparison with AleRax's sampler: do NOT pass ``--gene-tree-rooting ROOTED``
------------------------------------------------------------------------------------------------------
gpurec's CCP preprocessing always marginalizes uniformly over every possible
rooting of the input gene tree (nothing in gpurec/ or crates/ can turn this
off). AleRax's default does the same thing -- its startup log says so
explicitly ("all gene tree root positions are considered with the same
probability") -- so the two are already apples-to-apples with NO rooting flag
passed. An earlier version of this test passed ``--gene-tree-rooting ROOTED``
to force AleRax onto the one given rooting, reasoning that this was needed to
match gpurec; that reasoning was backwards, and it manufactured a large,
reproducible discrepancy in the sampled origination distribution on any
fixture where duplication and transfer both compete for the same clade (e.g.
species "A" drawn ~14% of the time by gpurec vs ~28% by AleRax on the fixture
below) -- a genuinely different quantity being compared, not a model bug. The
other explanations tested along the way (transfer-admissibility rule,
uniform-receiver normalization, fixed-point convergence) all matched AleRax.

Receiver weights have no AleRax equivalent at all (AleRax's transfer receiver
is always uniform), so that piece can only be checked internally -- see
test_backtracking_statistical_consistency.py.
"""
from __future__ import annotations

import collections

import pytest
from scipy import stats
from gpurec.config import GpurecConfig

pytestmark = pytest.mark.gpu

_SPECIES_NWK = "((A:1,B:1)AB:1,C:1)Root;"
_GENE_NWK = "(((A_1:1,B_1:1)x:1,C_1:1)y:1,A_2:1)GeneRoot;"  # A duplicated: forces
# real duplication/transfer competition (unlike a gene tree matching the species
# topology exactly, which has only one possible reconciliation).
_D = _L = _T = 0.3
_FM_HIGH = 0.8  # applied at B and C, the two species with a single observed copy
_N_SAMPLES = 2000
_SIGNIFICANCE = 0.01
# GALE 2's root species probabilities on this fixture (float64, [model] duplication_split_factor = 2),
# without and with fraction-missing.
_GALE2_ROOT_SPECIES = {
    "fm0": {"A": 0.09348122429752294, "B": 0.03555271553413202, "AB": 0.287331986185968,
            "C": 0.07143198721962844, "Root": 0.5122020867627485},
    "fm_high": {"A": 0.04360453443633783, "B": 0.047033109222628114, "AB": 0.230211489874391,
                "C": 0.08670773103461865, "Root": 0.5924431354320244},
}


def _write_fixture(workdir):
    (workdir / "sp.nwk").write_text(_SPECIES_NWK + "\n")
    (workdir / "g.nwk").write_text(_GENE_NWK + "\n")


def _run_gpurec_origination(sp_path, g_path, fraction_missing):
    """Sample _N_SAMPLES reconciliations with gpurec's own backtracking sampler
    (``sample_reconciliations_many``, which the command line uses); return a Counter of the root species."""
    import torch

    from gpurec.api.model import GeneReconModel
    from gpurec.core.backtracking.input import _load_native_module, batch_forward, family_sampler_arguments
    from gpurec.core.scheduling.batching import species_name_to_index

    model = GeneReconModel(
        str(sp_path), [str(g_path)], mode="global", dtype=torch.float64,
        fraction_missing=fraction_missing, clade_budget=GpurecConfig().batching.clade_budget
    )
    with torch.no_grad():
        model.theta.copy_(
            torch.log2(torch.tensor([_D, _L, _T], dtype=torch.float64, device=model.theta.device))
        )
    idx_to_name = {v: k for k, v in species_name_to_index(str(sp_path)).items()}

    static = model.batch_statics[0]
    arguments = family_sampler_arguments(model, static, batch_forward(model, static), 0)
    counts = collections.Counter()
    for nodes in _load_native_module().sample_reconciliations_many(*arguments, list(range(_N_SAMPLES))):
        counts[idx_to_name[nodes[0][1]]] += 1
    return counts


def _pooled_goodness_of_fit(counts, probabilities, min_expected=5.0):
    """chi-square goodness-of-fit test of categorical counts against exact probabilities, pooling
    the categories expected fewer than ``min_expected`` times into a single 'other' bucket first."""
    total = sum(counts.values())
    assert set(counts) <= set(probabilities), f"drawn species outside the reference: {counts}"
    big = [c for c in sorted(probabilities) if probabilities[c] * total >= min_expected]
    small = [c for c in sorted(probabilities) if c not in big]
    columns = big + (["other"] if small else [])
    observed = [counts.get(c, 0) for c in big] + ([sum(counts.get(c, 0) for c in small)] if small else [])
    expected = [probabilities[c] * total for c in big] + (
        [sum(probabilities[c] for c in small) * total] if small else [])
    assert len(columns) >= 2, f"not enough well-populated categories to compare: {columns}"
    statistic, p_value = stats.chisquare(observed, f_exp=expected)
    return statistic, p_value, columns


@pytest.fixture(scope="module")
def origination_counts(tmp_path_factory):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    results = {}
    for tag, gpurec_fm in (("fm0", None), ("fm_high", {"B": _FM_HIGH, "C": _FM_HIGH})):
        workdir = tmp_path_factory.mktemp(f"backtrack_origination_{tag}")
        _write_fixture(workdir)
        results[tag] = _run_gpurec_origination(workdir / "sp.nwk", workdir / "g.nwk", gpurec_fm)

    print(
        "\n[backtracking] origination species over "
        f"{_N_SAMPLES} samples each, D=L=T={_D} (duplication+transfer both active):\n"
        f"  fm=0        gpurec={dict(results['fm0'])}\n"
        f"  fm={_FM_HIGH} (B,C)  gpurec={dict(results['fm_high'])}\n"
    )
    return results


def test_origination_distribution_matches_reference_fm0(origination_counts):
    """No fraction-missing: gpurec's sampled origination species distribution matches the
    reference, with duplication and transfer both active and genuinely competing to explain the
    family."""
    statistic, p_value, columns = _pooled_goodness_of_fit(
        origination_counts["fm0"], _GALE2_ROOT_SPECIES["fm0"]
    )
    assert p_value > _SIGNIFICANCE, (
        f"origination distribution diverges at fm=0: chi2={statistic:.3f}, "
        f"p={p_value:.3g}, categories={columns}\n"
        f"gpurec={dict(origination_counts['fm0'])}\nreference={_GALE2_ROOT_SPECIES['fm0']}"
    )


def test_origination_distribution_matches_reference_fraction_missing(origination_counts):
    """Large fraction-missing at the two species with a single observed copy: gpurec's sampler
    shifts its origination distribution as the reference does, on top of the same
    duplication+transfer competition."""
    statistic, p_value, columns = _pooled_goodness_of_fit(
        origination_counts["fm_high"], _GALE2_ROOT_SPECIES["fm_high"]
    )
    assert p_value > _SIGNIFICANCE, (
        f"origination distribution diverges at fm={_FM_HIGH}: chi2={statistic:.3f}, "
        f"p={p_value:.3g}, categories={columns}\n"
        f"gpurec={dict(origination_counts['fm_high'])}\nreference={_GALE2_ROOT_SPECIES['fm_high']}"
    )
