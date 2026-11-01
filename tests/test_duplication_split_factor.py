"""The term of a duplication whose two copies carry the two child clades of a gene split counts twice.

The two ways of giving the two clades to the two copies are two distinct histories, so the term has
a factor 2 (``LOG2_DUPLICATION_SPLIT_FACTOR`` in gpurec/core/kernels/pi_forward.py; ALE's undated
likelihood and AleRax 1.4.0 count it once). Checked here:

* the likelihood: on the species tree ((X,Y),Z), GALE's float64 log-likelihood of small named gene
  trees equals the log of their probability obtained by adding up every history with at most 41 gene
  copies (tests/_history_enumeration.py, no recursion involved), and differs from the value of the
  recursion that counts the term once;
* the slopes and the curvature: on a family with many such duplications, the gradient equals central
  differences of the likelihood (the backward's split kernel) and the Hessian-vector products equal
  central differences of the gradient (the tangent and second-order split kernels);
* the sampler: the native sampler (crates/gpurec-backtrack) draws the duplication of a split as
  often as the tables it reads say, and the terms it weighs add up to those tables.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gpurec.config import GpurecConfig
from tests import _history_enumeration as H
from tests.support.fidelity import reconcile_at_alerax_rates

pytestmark = pytest.mark.gpu

DATA = Path(__file__).parent / "data" / "alerax"

# Histories with at most this many gene copies (about 4 s on one CPU core): the enumerated
# probabilities of the trees below are then within about 2e-8 of their limit, in log.
_MAX_COPIES = 41
_MAX_GENES = 3
# GALE against the enumeration, in nats (observed: at most 1.6e-8, the histories with more copies).
_ENUMERATION_TOLERANCE = 1e-7
# Smallest gap, in nats, between GALE and the recursion that counts the term once, on these trees
# (observed: at least 1.1e-2).
_FACTOR_ONE_GAP = 1e-3
# Representatives of the unrooted named gene trees with two or three genes, up to renaming.
_TREES = [
    frozenset({"X1", "X2"}),
    frozenset({"X1", "Y1"}),
    frozenset({"X1", "Z1"}),
    frozenset({"Z1", "Z2"}),
    frozenset({frozenset({"X1", "X2"}), "X3"}),
    frozenset({frozenset({"X1", "X2"}), "Y1"}),
    frozenset({frozenset({"X1", "Y1"}), "Z1"}),
    frozenset({frozenset({"X1", "Z1"}), "Z2"}),
    frozenset({frozenset({"Z1", "Z2"}), "Z3"}),
]


@pytest.fixture(scope="module")
def enumeration():
    G = H.enumerate_histories(_MAX_COPIES, _MAX_GENES)
    survival = 1.0 - float(np.sum(H.extinction())) / H.S
    return G, survival


def test_recursion_with_the_factor_two_is_the_sum_over_histories(enumeration):
    """Ties the reference: the recursion that counts the term twice gives the enumerated value, the
    one that counts it once does not."""
    G, survival = enumeration
    twice, once = H.Recursion(2), H.Recursion(1)
    for tree in _TREES:
        truth = H.log_likelihood_nats(lambda r: H.enumerated_probability(G, r), tree, survival)
        assert H.log_likelihood_nats(twice.probability, tree, survival) == pytest.approx(
            truth, abs=_ENUMERATION_TOLERANCE), H.show(tree)
        assert abs(H.log_likelihood_nats(once.probability, tree, survival) - truth) > _FACTOR_ONE_GAP


def test_likelihood_equals_the_sum_over_histories(enumeration, tmp_path):
    G, survival = enumeration
    once = H.Recursion(1)
    species = tmp_path / "species.nwk"
    species.write_text(H.SPECIES_NEWICK + "\n")
    for k, tree in enumerate(_TREES):
        gene = tmp_path / f"gene_{k}.nwk"
        gene.write_text(H.newick(tree) + ";\n")
        (gale,) = reconcile_at_alerax_rates(
            str(species), [str(gene)], (H.DUPLICATION, H.LOSS, H.TRANSFER),
            device="cuda", dtype=torch.float64, mode="global").values()
        truth = H.log_likelihood_nats(lambda r: H.enumerated_probability(G, r), tree, survival)
        factor_one = H.log_likelihood_nats(once.probability, tree, survival)
        assert gale == pytest.approx(truth, abs=_ENUMERATION_TOLERANCE), (H.show(tree), gale, truth)
        assert abs(gale - factor_one) > _FACTOR_ONE_GAP, (H.show(tree), gale, factor_one)


# --------------------------------------------------------------------------- slopes and curvature
# test_mixed_200 at these rates: duplication 0.05 gives many duplications that separate two clades.
_RATES_AT = (0.05, 0.1, 0.02)
_GRADIENT_STEP = 1e-5
# With the converged solver settings below the two agree to about 4e-10 (relative), before and after
# the correction; with the default settings (extinction tolerance 1e-8, adjoint pruning at 1e-6) both
# differ by about 7e-6.
_GRADIENT_RTOL = 1e-8
_CONVERGED_SOLVER = dict(e_max_iter=4000, e_tol=1e-14, pi_iters=256, neumann_terms=256,
                         e_adjoint_max_iter=4000, e_adjoint_tol=1e-14, use_adjoint_pruning=False,
                         adjoint_pruning_threshold=0.0)
_HESSIAN_STEP = 1e-4
_HESSIAN_RTOL = 1e-4


def _model(mode):
    from gpurec.api.model import GeneReconModel

    folder = DATA / "test_mixed_200"
    model = GeneReconModel(str(folder / "sp.nwk"), [str(folder / "g.nwk")], mode=mode,
                           dtype=torch.float64, fraction_missing=None,
                           clade_budget=GpurecConfig().batching.clade_budget)
    model.solver_options = dataclasses.replace(model.solver_options, **_CONVERGED_SOLVER)
    with torch.no_grad():
        model.theta.copy_(torch.log2(torch.tensor(_RATES_AT, dtype=torch.float64,
                                                  device=model.theta.device)).expand_as(model.theta))
    return model


def _loss_and_gradient(model, theta, need_grad):
    from gpurec.api._execution import stream_batches

    loss, grad, _, _ = stream_batches(model.batch_statics, theta, model.receiver_weights,
                                      model.origination_weights, genewise=model.genewise,
                                      need_grad=need_grad)
    return float(loss), (None if grad is None else grad.double().reshape(-1))


@pytest.mark.parametrize("mode", ["global", "genewise"])
def test_gradient_matches_central_differences(mode):
    model = _model(mode)
    theta = model.theta.detach().clone()
    _, gradient = _loss_and_gradient(model, theta, True)
    flat = theta.reshape(-1)
    for i in range(flat.numel()):
        step = torch.zeros_like(flat)
        step[i] = _GRADIENT_STEP
        up, _ = _loss_and_gradient(model, (flat + step).reshape(theta.shape), False)
        down, _ = _loss_and_gradient(model, (flat - step).reshape(theta.shape), False)
        assert float(gradient[i]) == pytest.approx((up - down) / (2 * _GRADIENT_STEP),
                                                   rel=_GRADIENT_RTOL), (mode, i)


def test_hessian_vector_products_match_central_differences_of_the_gradient():
    from gpurec.solver.hvp.exact import make_exact_hvp
    from gpurec.solver.value_and_grad import forward_solve

    model = _model("genewise")
    theta = model.theta.detach().clone()
    static = model.batch_statics[0]
    _, state = forward_solve([static], theta, model.receiver_weights.detach())
    receivers = model.receiver_weights.detach()
    hvp = make_exact_hvp([static], theta, receivers, state, tangent_self_iters=128)
    for j in range(3):
        direction = torch.zeros_like(theta)
        direction[:, j] = 1.0
        _, up = _loss_and_gradient(model, theta + _HESSIAN_STEP * direction, True)
        _, down = _loss_and_gradient(model, theta - _HESSIAN_STEP * direction, True)
        central = (up - down) / (2 * _HESSIAN_STEP)
        analytic = hvp(direction.reshape(-1)).double()
        torch.testing.assert_close(analytic, central, rtol=_HESSIAN_RTOL,
                                   atol=_HESSIAN_RTOL * float(central.abs().max()))


# --------------------------------------------------------------------------- the sampler
# One gene split, (A_1, B_1), where a duplication competes with a speciation and with transfers.
_SAMPLER_SPECIES = "((A:1,B:1)AB:1,C:1)Root;"
_SAMPLER_GENE = "(A_1:1,B_1:1);"
_SAMPLER_RATES = (0.3, 0.3, 0.3)
_DRAWS = 20_000
# Largest z-score (difference over its standard error) allowed between the sampled and the exact
# share of draws holding a duplication; the share the sampler would draw with the factor 1 is about
# 9 standard errors away on this family.
_LARGEST_Z = 4.5
_TABLES_TOLERANCE = 1e-9


def _duplication_share(arguments, log2_split_factor):
    """Exact share of draws with a duplication, from the tables the sampler reads.

    The draw walks the root clade from species to species (duplication or transfer followed by a
    loss, speciation with one child lost) until it picks one of the split events; each step picks a
    term with probability proportional to its weight, as ``sample_term`` in
    crates/gpurec-backtrack/src/lib.rs does. Also returns, per species, the base-2 log of the sum of
    the weights, which is the root clade's Pi when the weights are the terms of its equation.
    """
    (root, _leaf_species, parents, leftrights, lsp, pi, pibar, e, ebar, lps, lpd, recv,
     child1, child2, start, end) = arguments
    S = pi.shape[1]
    n = len(parents)
    splits = [i for i in range(n) if parents[i] == root]
    # ancestor_or_self[a, b]: a is b or one of its ancestors (a transfer from b may not go to a).
    ancestor_or_self = np.array([[start[a] <= start[b] < end[a] for b in range(S)] for a in range(S)])
    moves = np.zeros((S, S))
    duplication = np.zeros(S)
    log2_total = np.zeros(S)
    for s in range(S):
        kids = (child1[s], child2[s]) if 0 <= child1[s] < S and 0 <= child2[s] < S else None
        stay = [1.0 + lpd[s] + e[s] + pi[root, s], pi[root, s] + ebar[s]]
        transfer_loss = pibar[root, s] + e[s]
        hidden = ([(kids[0], lps[s] + e[kids[1]] + pi[root, kids[0]]),
                   (kids[1], lps[s] + e[kids[0]] + pi[root, kids[1]])] if kids else [])
        dup, other = [], []
        for i in splits:
            left, right = leftrights[i], leftrights[n + i]
            dup.append(lsp[i] + lpd[s] + log2_split_factor + pi[left, s] + pi[right, s])
            other += [lsp[i] + pi[left, s] + pibar[right, s], lsp[i] + pi[right, s] + pibar[left, s]]
            if kids:
                other += [lsp[i] + lps[s] + pi[left, kids[0]] + pi[right, kids[1]],
                          lsp[i] + lps[s] + pi[right, kids[0]] + pi[left, kids[1]]]
        weights = np.array(stay + [transfer_loss] + [w for _, w in hidden] + dup + other)
        total = float(np.sum(np.exp2(weights[np.isfinite(weights)])))
        log2_total[s] = math.log2(total)
        moves[s, s] += sum(2.0 ** w for w in stay) / total
        for k, w in hidden:
            moves[s, k] += 2.0 ** w / total
        receiver = np.where(ancestor_or_self[:, s], -np.inf, recv + pi[root])
        receiver = np.exp2(receiver - np.max(receiver))
        moves[s] += 2.0 ** transfer_loss / total * receiver / receiver.sum()
        duplication[s] = sum(2.0 ** w for w in dup) / total
    start_species = np.exp2(pi[root] - np.max(pi[root]))
    visits = np.linalg.solve(np.eye(S) - moves.T, start_species / start_species.sum())
    return float(visits @ duplication), log2_total, pi[root]


def test_sampler_draws_the_duplication_of_a_split_as_the_likelihood_counts_it(tmp_path):
    from gpurec.api.model import GeneReconModel
    from gpurec.core.backtracking.input import _load_native_module, batch_forward, family_sampler_arguments
    from gpurec.core.kernels.pi_forward import LOG2_DUPLICATION_SPLIT_FACTOR

    species = tmp_path / "species.nwk"
    species.write_text(_SAMPLER_SPECIES + "\n")
    gene = tmp_path / "gene.nwk"
    gene.write_text(_SAMPLER_GENE + "\n")
    model = GeneReconModel(str(species), [str(gene)], mode="global", dtype=torch.float64,
                           fraction_missing=None, clade_budget=GpurecConfig().batching.clade_budget)
    with torch.no_grad():
        model.theta.copy_(torch.log2(torch.tensor(_SAMPLER_RATES, dtype=torch.float64,
                                                  device=model.theta.device)))
    static = model.batch_statics[0]
    with torch.no_grad():
        arguments = family_sampler_arguments(model, static, batch_forward(model, static), 0)

    factor = float(LOG2_DUPLICATION_SPLIT_FACTOR.value)
    assert factor == 1.0
    share, log2_total, log2_pi = _duplication_share(arguments, factor)
    share_factor_one, log2_total_factor_one, _ = _duplication_share(arguments, 0.0)
    # The forward's tables hold the factor 2: the terms, so weighed, add up to the root clade's Pi.
    np.testing.assert_allclose(log2_total, log2_pi, rtol=0.0, atol=_TABLES_TOLERANCE)
    assert np.max(np.abs(log2_total_factor_one - log2_pi)) > 1e-3

    samples = _load_native_module().sample_reconciliations_many(*arguments, list(range(_DRAWS)))
    drawn = sum(1 for nodes in samples if any(node[0] == "duplication" for node in nodes)) / _DRAWS
    error = math.sqrt(share * (1.0 - share) / _DRAWS)
    assert abs(share - share_factor_one) / error > 2 * _LARGEST_Z, "the test has no power here"
    assert abs(drawn - share) / error < _LARGEST_Z, (drawn, share, share_factor_one)
