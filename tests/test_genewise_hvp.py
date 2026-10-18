"""Genewise analytic HVP gates: analytic forward-over-reverse == fp64 finite-difference (+ symmetry).

The genewise Hessian is block-diagonal per family; broadcasting a unit tangent e_j across all families
recovers column j of every family's block at once. These tests pin the theta block used by the per-family fit.
"""
import dataclasses
import math

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.solver.hvp.exact import make_exact_hvp
from gpurec.api._execution import stream_batches
from gpurec.solver.value_and_grad import forward_solve
from gpurec.config import GpurecConfig


def _value_and_grad(statics, receiver_weights, theta_shape):
    """``vg(theta_vec, warm_E=...) -> (loss, flat gradient)`` over ``statics`` (genewise ``theta``)."""
    from gpurec.solver.value_and_grad import free_cuda_cache_if_tight

    def vg(theta_vec, *, warm_E):
        free_cuda_cache_if_tight()
        loss, grad, _, _ = stream_batches(
            statics, theta_vec.detach().reshape(theta_shape).contiguous(), receiver_weights,
            receiver_weights.new_zeros(receiver_weights.numel()), genewise=True, need_grad=True,
        )
        return float(loss), grad.reshape(-1).contiguous()

    return vg


def _fd_hessian_hvp(vg, theta_vec, warm_E, *, eps):
    """True-Hessian HVP via central finite differences of the gradient ``vg`` (forward + backward), the
    direction normalised so the step has size ``eps`` whatever the scale of ``v``."""
    from gpurec.solver.value_and_grad import free_cuda_cache_if_tight

    base = theta_vec.double()

    def hvp(v):
        v = v.double()
        nv = float(torch.linalg.vector_norm(v))
        if nv == 0.0:
            return torch.zeros_like(v)
        u = v / nv
        free_cuda_cache_if_tight()
        out = vg((base + eps * u).to(theta_vec.dtype), warm_E=warm_E)
        gp = out[1].double()
        del out
        out = vg((base - eps * u).to(theta_vec.dtype), warm_E=warm_E)
        gm = out[1].double()
        del out
        return nv * (gp - gm) / (2 * eps)

    return hvp

_D = "tests/data/alerax/test_trees_200"
_SO = dict(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64,
           adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True, pibar_side_threshold=0.0)


def build_genewise_model(n_fam=2, dtype=torch.float64, device="cuda", per_family_origination=False,
                         family_chunk_size=None):
    so = SolverOptions(**_SO); so.validate()
    kw = {} if family_chunk_size is None else {"family_chunk_size": family_chunk_size}
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * n_fam, mode="genewise", dtype=dtype,
                       solver_options=so, per_family_origination=per_family_origination,
                       clade_budget=GpurecConfig().batching.clade_budget, **kw)
    # single-batch is the default (Tasks 1-7); only force ≥2 batches when family_chunk_size splits them
    if family_chunk_size is None:
        assert len(m.batch_statics) == 1, f"expected 1 batch, got {len(m.batch_statics)}"
    return m


@pytest.mark.gpu
def test_batch_static_warm_v_tangent_defaults_to_none():
    m = build_genewise_model()
    assert m.batch_statics[0].warm_v_tangent is None


@pytest.mark.gpu
def test_build_point_cache_accepts_and_forwards_warm_v():
    from gpurec.solver.hvp.exact import build_point_cache

    m = build_genewise_model()
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)

    # warm_v=None (today's behavior) and warm_v={} (empty -- no cached entries yet) must agree,
    # since an empty dict has nothing to look up (init_v is None either way on a first call).
    g_theta_none, g_col_none, _cache_none = build_point_cache(static, theta, rw, sv, )
    g_theta_empty, g_col_empty, _cache_empty = build_point_cache(static, theta, rw, sv, warm_v={})
    torch.testing.assert_close(g_theta_none, g_theta_empty)
    torch.testing.assert_close(g_col_none, g_col_empty)


@pytest.mark.gpu
def test_build_point_cache_warm_v_poisoned_seed_changes_result():
    """A warm_v dict populated with deliberately wrong values must measurably perturb
    the result at a SMALL, non-contracting neumann_terms budget (proving warm_v is
    actually READ, not silently dropped in forwarding). neumann_terms=1 makes the
    self-loop's fixed-point update v <- rhs + A @ initial_v exactly LINEAR in
    initial_v -- no contraction to wash the poisoned seed's effect away, unlike the
    production neumann_terms=64 default where A^64 would shrink any perturbation
    below detectable tolerance regardless of whether warm_v is truly consumed.
    """
    from gpurec.solver.hvp.exact import build_point_cache

    so = SolverOptions(**{**_SO, "neumann_terms": 1})
    so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * 2, mode="genewise", dtype=torch.float64, solver_options=so, clade_budget=GpurecConfig().batching.clade_budget)
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)

    g_theta_cold, _g_col_cold, _cache_cold = build_point_cache(static, theta, rw, sv, warm_v=None)

    # Let a real call populate a warm_v dict naturally, then corrupt every cached tensor.
    warm_v = {}
    build_point_cache(static, theta, rw, sv, warm_v=warm_v)
    assert len(warm_v) > 0, "expected at least one wave to populate warm_v"
    poisoned = {ws: v * 1000.0 + 50.0 for ws, v in warm_v.items()}

    g_theta_poisoned, _g_col_p, _cache_p = build_point_cache(static, theta, rw, sv, warm_v=poisoned)

    assert not torch.allclose(g_theta_poisoned, g_theta_cold), (
        "poisoned warm_v seed had no effect at neumann_terms=1 (where the self-loop update "
        "is exactly linear in initial_v) -- warm_v is being silently dropped somewhere in "
        "vjp_root_to_theta -> implicit_grad_loglik_vjp_wave forwarding"
    )


@pytest.mark.gpu
def test_genewise_theta_hvp_matches_fd():
    m = build_genewise_model()
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)

    _l, sv = forward_solve([static], theta, rw)
    hvp = make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    fd = _fd_hessian_hvp(_value_and_grad([static], rw, (F, 3)),
                         theta.reshape(-1).contiguous(), None, eps=1e-5)

    probes = []
    for j in range(3):  # broadcast e_j across all families
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64)
        u[:, j] = 1.0
        u = u.reshape(-1)
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"broadcast e_{j}: rel={rel:.2e}"
        probes.append((u, Ha))

    (u, Hu), (w, Hw) = probes[0], probes[1]
    sym = abs(float(torch.dot(u, Hw)) - float(torch.dot(w, Hu)))
    scale = max((abs(float(torch.dot(u, Hu))) * abs(float(torch.dot(w, Hw)))) ** 0.5, 1e-30)
    assert sym / scale < 5e-3, f"non-symmetric: rel_asym={sym / scale:.2e}"


def genewise_hessian_blocks(static, theta, receiver_weights, sv, *, omega=None, active=("theta",)):
    """Theta-only per-family curvature ``[G,3,3]``, built from 3 broadcast HVP probes (one per theta
    component) against the joint analytic HVP (``make_exact_hvp``).

    TEST-ONLY fixture: the production genewise fit
    never assembles blocks -- it runs matrix-free CG directly on the analytic HVP
    (``newton_joint_genewise``). Feeds the fp32-vs-fp64 golden comparison below.
    """
    assert tuple(active) == ("theta",), "P0: theta-only; omega/alpha added in Tasks 4-5"
    G = int(theta.shape[0])
    tsi = int(static.solver_options.pi_iters)
    hvp = make_exact_hvp([static], theta, receiver_weights, sv, tangent_self_iters=tsi)
    cols = []
    for j in range(3):  # broadcast e_j across all families -> column j of every 3x3 block at once
        u = torch.zeros(G, 3, device=theta.device, dtype=theta.dtype); u[:, j] = 1.0
        cols.append(hvp(u.reshape(-1))[: G * 3].reshape(G, 3))
    H = torch.stack(cols, dim=-1)  # [G,3,3], H[:,:,j] = col j
    return {"H_tt": 0.5 * (H + H.transpose(1, 2))}


@pytest.mark.gpu
def test_theta_blocks_fp32_vs_fp64():
    def blocks(dtype):
        m = build_genewise_model(dtype=dtype); st = m.batch_statics[0]
        G = len(m.families); S = int(m.species_helpers["S"])
        th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=dtype)
        rw = torch.zeros(S, device="cuda", dtype=dtype)
        _l, sv = forward_solve([st], th, rw)
        return genewise_hessian_blocks(st, th, rw, sv)["H_tt"]
    H32, H64 = blocks(torch.float32).double(), blocks(torch.float64)
    assert torch.isfinite(H32).all()
    torch.testing.assert_close(H32, H64, rtol=1e-3, atol=1e-2)
    assert float((H64 - H64.transpose(1, 2)).abs().max()) < 1e-6  # each 3x3 block symmetric


@pytest.mark.gpu
def test_genewise_point_cache_warm_start_matches_fd_across_repeated_calls():
    """Two make_exact_hvp calls at nearby theta on the same static: the second call's
    point-cache backward pass reuses static.warm_v (populated by the first call). The
    result must still match FD within the existing correctness tolerance."""
    m = build_genewise_model()
    m.solver_options = {"use_hvp_warm_start": True}  # off by default since d48c48f7; this test is about it
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)

    theta0 = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l0, sv0 = forward_solve([static], theta0, rw)
    make_exact_hvp([static], theta0, rw, sv0, tangent_self_iters=128)  # populates static.warm_v
    assert static.warm_v is not None and len(static.warm_v) > 0
    warm_v_id_after_call1 = id(static.warm_v)

    theta1 = theta0 + 0.01
    _l1, sv1 = forward_solve([static], theta1, rw)
    hvp1 = make_exact_hvp([static], theta1, rw, sv1, tangent_self_iters=128)
    assert id(static.warm_v) == warm_v_id_after_call1, (
        "static.warm_v was reassigned instead of reused across calls -- the warm-start gate "
        "must only create a fresh dict when static.warm_v is None, never unconditionally"
    )
    fd1 = _fd_hessian_hvp(_value_and_grad([static], rw, (F, 3)),
                          theta1.reshape(-1).contiguous(), None, eps=1e-5)
    for j in range(3):
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, j] = 1.0
        u = u.reshape(-1)
        Ha, Hf = hvp1(u).double(), fd1(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"broadcast e_{j}: rel={rel:.2e}"


@pytest.mark.gpu
def test_genewise_point_cache_warm_start_disabled_by_config():
    """use_hvp_warm_start=False -> static.warm_v is never touched."""
    so = SolverOptions(**_SO)
    so.use_hvp_warm_start = False
    so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * 2, mode="genewise", dtype=torch.float64, solver_options=so, clade_budget=GpurecConfig().batching.clade_budget)
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)
    make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    assert static.warm_v is None


@pytest.mark.gpu
def test_stream_batches_multibatch_origination_grad():
    # Multi-batch per-family origination grad: stream_batches must scatter each batch-local [G_b,S]
    # origination grad into the full [G,S] accumulator (index_add_), not shape-mismatch on .add_().
    from gpurec.api._execution import stream_batches
    torch.manual_seed(0)
    m = build_genewise_model(n_fam=4, per_family_origination=True, family_chunk_size=2)
    assert len(m.batch_statics) >= 2, f"need multi-batch, got {len(m.batch_statics)}"
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    om = torch.randn(G, S, device="cuda", dtype=torch.float64) * 0.1
    loss, g_th, g_rw, g_om = stream_batches(
        m.batch_statics, th, rw, om, genewise=True, need_grad=True, need_origination_grad=True)
    assert tuple(g_om.shape) == (G, S)
    assert torch.isfinite(g_om).all()


@pytest.mark.gpu
def test_genewise_tangent_warm_start_matches_fd():
    """probe_id passed on two nearby-theta calls: warm-started tangent-adjoint result
    still matches FD within the existing correctness tolerance, AND static.warm_v_tangent's
    per-probe dicts are genuinely reused (not clobbered) across repeated hvp() calls."""
    m = build_genewise_model()
    m.solver_options = {"use_hvp_warm_start": True}  # off by default since d48c48f7; this test is about it
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)

    theta0 = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l0, sv0 = forward_solve([static], theta0, rw)
    hvp0 = make_exact_hvp([static], theta0, rw, sv0, tangent_self_iters=128)
    for j in range(3):
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, j] = 1.0
        hvp0(u.reshape(-1), probe_id=j)
    assert static.warm_v_tangent is not None
    assert set(static.warm_v_tangent.keys()) == {0, 1, 2}
    probe_cache_ids_after_call1 = {j: id(static.warm_v_tangent[j]) for j in range(3)}

    theta1 = theta0 + 0.01
    _l1, sv1 = forward_solve([static], theta1, rw)
    hvp1 = make_exact_hvp([static], theta1, rw, sv1, tangent_self_iters=128)
    fd1 = _fd_hessian_hvp(_value_and_grad([static], rw, (F, 3)),
                          theta1.reshape(-1).contiguous(), None, eps=1e-5)
    for j in range(3):
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, j] = 1.0
        u = u.reshape(-1)
        Ha, Hf = hvp1(u, probe_id=j).double(), fd1(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"warm probe_id={j}: rel={rel:.2e}"
    for j in range(3):
        assert id(static.warm_v_tangent[j]) == probe_cache_ids_after_call1[j], (
            f"static.warm_v_tangent[{j}] was reassigned instead of reused across calls"
        )


@pytest.mark.gpu
def test_genewise_tangent_warm_start_probes_do_not_cross_contaminate():
    m = build_genewise_model()
    m.solver_options = {"use_hvp_warm_start": True}  # off by default since d48c48f7; this test is about it
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)
    hvp = make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    u0 = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u0[:, 0] = 1.0
    u1 = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u1[:, 1] = 1.0
    hvp(u0.reshape(-1), probe_id=0)
    hvp(u1.reshape(-1), probe_id=1)
    assert static.warm_v_tangent[0].keys() == static.warm_v_tangent[1].keys()
    n_checked = 0
    for ws in static.warm_v_tangent[0]:
        v0 = static.warm_v_tangent[0][ws]
        v1 = static.warm_v_tangent[1][ws]
        # This fixture's adjoint_pruning_threshold=1e-6 (see _SO) marks several of this
        # 2-family tree's internal-node waves entirely inactive (confirmed via each wave's
        # pre-existing cache["waves"][i]["active_mask"], built once by build_point_cache --
        # independent of probe_id); the NaN-safe masking then legitimately zeros v_k for
        # BOTH probes there (nothing to contaminate). Restrict the distinctness check to
        # waves where at least one probe's cached v is non-negligible.
        if max(float(v0.abs().max()), float(v1.abs().max())) < 1e-8:
            continue
        n_checked += 1
        assert not torch.allclose(v0, v1), "different probe directions must cache distinct v_k"
    assert n_checked > 0, "no non-trivial wave found to check cross-contamination"


@pytest.mark.gpu
def test_genewise_tangent_warm_start_disabled_by_config():
    so = SolverOptions(**_SO)
    so.use_hvp_warm_start = False
    so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * 2, mode="genewise", dtype=torch.float64, solver_options=so, clade_budget=GpurecConfig().batching.clade_budget)
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)
    hvp = make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, 0] = 1.0
    hvp(u.reshape(-1), probe_id=0)
    assert static.warm_v_tangent is None or 0 not in static.warm_v_tangent


def _repeated_family_data(tmp_path, n):
    """``n`` families, each a copy of the same gene tree file under its own name (family names are
    file stems, so passing the same path ``n`` times would collide on the name)."""
    import shutil

    from tests.support.families import family_data

    genes = []
    for i in range(n):
        gene = tmp_path / f"g{i}.nwk"
        shutil.copy(f"{_D}/g.nwk", gene)
        genes.append(gene)
    return family_data(f"{_D}/sp.nwk", genes)


@pytest.mark.gpu
def test_fit_genewise_converges_on_small_fixture(tmp_path):
    """Direct behavior test for fit_genewise itself (previously only covered indirectly via
    signature/config-wiring checks). Runs the real recipe end-to-end on a handful of families
    and checks it actually reaches a converged, finite optimum -- this must pass both before and
    after swapping the Hessian source, since the swap must not change the recipe's behavior
    contract, only its internal Hessian construction."""
    from gpurec.config import GpurecConfig
    from gpurec.fit.genewise_fit import fit_genewise

    data = _repeated_family_data(tmp_path, 4)
    start = torch.zeros(data.n_families, 3, dtype=torch.float64)
    res = fit_genewise(
        data, start_log2_rates=start, config=GpurecConfig.genewise_reference(),
        adam_steps=5, pi_tiers=(16,), neu_opt=16, neu_cert=16,
        max_iter=60, certify=True, verbose=False,
    )
    assert res["n_families"] == 4
    assert torch.isfinite(res["theta"]).all()
    assert res["converged"] == 4
    assert res["pg_max"] < 1e-2
    assert math.isfinite(res["loss_bits"])


@pytest.mark.gpu
def test_fit_genewise_converges_with_multibatch_analytic_hvp(tmp_path):
    """Same correctness check as test_fit_genewise_converges_on_small_fixture, but forces
    len(m.batch_statics) > 1 via a small clade_budget so the analytic-HVP multi-batch branch
    (the streaming gather/scatter path -- what actually runs in production at real scale) gets
    automated coverage, not just the single-batch branch. All families use the SAME gene tree,
    so if the multi-batch gather/scatter ever shuffles a family's result to the wrong row, the
    per-family thetas would stop matching each other -- a robust, cheap way to catch attribution
    bugs without needing distinct per-family ground truth."""
    from gpurec.config import GpurecConfig
    from gpurec.fit.genewise_fit import fit_genewise
    from gpurec import GeneReconModel, SolverOptions

    n_fam = 6
    clade_budget = 500  # starting point -- verify this actually forces >1 batch for this fixture,
    # adjust if not (see Step 2 of the dispatch instructions)
    data = _repeated_family_data(tmp_path, n_fam)

    # Confirm the setup genuinely exercises the multi-batch branch before trusting the fit result.
    probe = GeneReconModel(
        f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * n_fam, mode="genewise", dtype=torch.float32,
        solver_options=SolverOptions(), clade_budget=clade_budget,
    )
    assert len(probe.batch_statics) > 1, (
        f"clade_budget={clade_budget} did not force multiple batches for {n_fam} families -- "
        "lower it until this assertion passes, then use the same value in the fit_genewise call below"
    )
    del probe
    torch.cuda.empty_cache()

    start = torch.zeros(data.n_families, 3, dtype=torch.float64)
    config = GpurecConfig.genewise_reference()
    config.batching = dataclasses.replace(config.batching, clade_budget=clade_budget)
    res = fit_genewise(
        data, start_log2_rates=start, config=config,
        adam_steps=5, pi_tiers=(16,), neu_opt=16, neu_cert=16,
        max_iter=60, certify=True, verbose=False,
    )
    assert res["n_families"] == n_fam
    assert torch.isfinite(res["theta"]).all()
    assert res["converged"] == n_fam
    assert res["pg_max"] < 1e-2
    assert math.isfinite(res["loss_bits"])
    # All families are identical gene trees -> their converged thetas must agree with each other.
    # If the multi-batch gather/scatter ever misattributes a family's result to the wrong row,
    # this is what would catch it.
    theta_mean = res["theta"].mean(dim=0, keepdim=True)
    assert torch.allclose(res["theta"], theta_mean.expand_as(res["theta"]), atol=1e-2), (
        f"per-family thetas disagree despite identical gene trees -- possible multi-batch "
        f"gather/scatter attribution bug: {res['theta']}"
    )
