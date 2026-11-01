"""Per-family log-likelihoods at fixed rates on the AleRax fixtures, against outside references.

Each fixture holds a species tree, a gene tree, and AleRax 1.4.0's fitted rates and log-likelihood
(``ref/``). GALE counts twice the term of a duplication whose two copies carry the two child clades
of a split (the two copies can carry the two clades in two ways); AleRax 1.4.0 counts it once. The
reference of each fixture is therefore:

* test_trees_1, test_trees_2, test_trees_200: AleRax's value. At their rates (duplication rate
  1e-10) no history with a duplication counts, and the two likelihoods agree to rounding.
* test_trees_3: AleRax's value plus ln 2. Its loss and transfer rates are 1e-10, so the only
  reconciliation of its gene tree that counts holds exactly one duplication that separates two
  clades, whose term GALE counts twice.
* test_mixed_200: no closed form. The reference is the value of GALE 2 (the development version
  of GALE: its own kernels, written separately, with ``[model] duplication_split_factor = 2``) in
  float64 at the same rates. GALE 2 with that factor set to 1 gives AleRax's value on all five
  fixtures (largest gap 9.1e-13 nats).

Until GALE 1.0 was corrected (CHANGELOG.md), every fixture was compared with AleRax's value. When
AleRax 1.5.0, which counts the term twice, is released, its fitted rates and log-likelihoods will
go in each fixture's ``ref/`` and replace the references above.
"""
import math

import pytest
from pathlib import Path
from tests.support.alerax_readers import parse_alerax_likelihoods, parse_alerax_parameters, global_rates
from tests.support.fidelity import reconcile_at_alerax_rates, compare

DATA = Path(__file__).parent / "data" / "alerax"
FIXTURES = ["test_trees_1", "test_trees_2", "test_trees_3", "test_trees_200", "test_mixed_200"]
TOL_NATS = 1e-3
# Fixtures whose single reconciliation that counts holds one duplication separating two clades.
ONE_SPLIT_DUPLICATION = {"test_trees_3"}
# GALE 2's float64 log-likelihood (nats) at the fixture's AleRax rates, [model]
# duplication_split_factor = 2. Corrected GALE 1.0 gave the same value to the last digit.
GALE2_NATS = {"test_mixed_200": -5825.071081385615}


def _reference(fixture, alerax_ll):
    """The fixture's reference log-likelihood in nats, keyed like ``alerax_ll``."""
    if fixture in ONE_SPLIT_DUPLICATION:
        return {k: v + math.log(2.0) for k, v in alerax_ll.items()}
    if fixture in GALE2_NATS:
        return {k: GALE2_NATS[fixture] for k in alerax_ll}
    return alerax_ll


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", FIXTURES)
def test_fidelity_matches_alerax_fixed(fixture):
    import torch
    ref = DATA / fixture / "ref"
    rates = global_rates(parse_alerax_parameters(ref / "model_parameters.txt"))
    alerax_ll = parse_alerax_likelihoods(ref / "per_fam_likelihoods.txt")
    gpurec_ll = reconcile_at_alerax_rates(
        str(DATA / fixture / "sp.nwk"), [str(DATA / fixture / "g.nwk")], rates,
        device="cuda", dtype=torch.float64, mode="global")
    # reconcile_at_alerax_rates keys its result by the gene-tree FILE basename
    # (g.nwk -> "g"), while ref/per_fam_likelihoods.txt keys by the AleRax family
    # NAME ("my_family"). These fixtures are single-family, so remap gpurec's
    # single value onto the ref's family key before comparing. (The real
    # archaea60-scale dataset names its gene files after the family, so there
    # the two keys already align naturally and no remap is needed.)
    assert len(gpurec_ll) == 1 and len(alerax_ll) == 1
    gpurec_ll = {k: v for k, v in zip(alerax_ll, gpurec_ll.values())}
    rep = compare(gpurec_ll, _reference(fixture, alerax_ll))
    assert rep.n_matched == 1, f"{fixture}: name mismatch {gpurec_ll} vs {alerax_ll}"
    assert rep.mean_abs_delta <= TOL_NATS, (
        f"{fixture}: |Δ|={rep.mean_abs_delta:.3e} nats > {TOL_NATS} "
        f"(gpurec={gpurec_ll}, reference={_reference(fixture, alerax_ll)})")


TOL_NATS_FP64 = 1e-8


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", FIXTURES)
def test_fidelity_float64_reaches_machine_precision(fixture):
    """float64 reconciliation must be GENUINE, not floored by a float32 batch static.

    The CCP split log-probs (``log_split_probs``) and the transfer row-max
    (``unnorm_row_max``) are computed in float64 by the Rust preprocessor. If either is
    quantized through float32 before an fp64 tensor is created, float64 reconciliation floors
    at ~1e-5 nats instead of reaching f64 round-off. This guards that regression: Rust retains
    f64 values through serialization, then batching materializes each tensor directly in its
    owning model or accumulator dtype. Every fixture lands within 1e-8 nats of its reference
    (observed <=3e-12). The looser 1e-3 gate above
    would not catch a re-truncation; this one will.
    """
    import torch
    ref = DATA / fixture / "ref"
    rates = global_rates(parse_alerax_parameters(ref / "model_parameters.txt"))
    alerax_ll = parse_alerax_likelihoods(ref / "per_fam_likelihoods.txt")
    gpurec_ll = reconcile_at_alerax_rates(
        str(DATA / fixture / "sp.nwk"), [str(DATA / fixture / "g.nwk")], rates,
        device="cuda", dtype=torch.float64, mode="global")
    assert len(gpurec_ll) == 1 and len(alerax_ll) == 1
    gpurec_ll = {k: v for k, v in zip(alerax_ll, gpurec_ll.values())}
    rep = compare(gpurec_ll, _reference(fixture, alerax_ll))
    assert rep.mean_abs_delta <= TOL_NATS_FP64, (
        f"{fixture}: float64 |Δ|={rep.mean_abs_delta:.3e} nats > {TOL_NATS_FP64} "
        f"— a float32 batch static is likely flooring float64 "
        f"(gpurec={gpurec_ll}, reference={_reference(fixture, alerax_ll)})")
