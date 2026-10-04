import pytest

pytest.importorskip("rustree")
torch = pytest.importorskip("torch")


def test_fit_mode_global_path_no_torch_shadow(monkeypatch):
    """Regression: fit_mode's specieswise branch must NOT do a function-local `import torch` -- that
    makes `torch` local to ALL of fit_mode and UnboundLocalErrors the global/genewise branch's
    `torch.float32` below. Stub load_families/fit_global so this runs off-GPU."""
    import types

    import numpy as np
    from tests.regression import mint_goldens
    monkeypatch.setattr(mint_goldens, "load_families", lambda **kwargs: types.SimpleNamespace())
    monkeypatch.setattr(mint_goldens, "fit_global",
                        lambda *a, **k: {"nll_bits": 1.0, "rates": torch.zeros((1, 3)), "wall_s": 0.1})
    nll, rates, wall = mint_goldens.fit_mode("global", "sp.nwk", ["g.nwk"])
    assert nll == 1.0 and np.asarray(rates).shape == (1, 3) and wall == 0.1


def test_fit_specieswise_requires_lam():
    from gpurec.fit.specieswise_fit import fit_specieswise
    with pytest.raises(ValueError, match="requires an explicit prior"):
        # lam is keyword-only with no default -> omitting it is a TypeError; passing None -> ValueError
        fit_specieswise("bs", "th", "rw", lam=None)


@pytest.mark.gpu
def test_fit_specieswise_fits_at_given_lam(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tests.support.simulate import simulate_dataset
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.specieswise_fit import fit_specieswise
    import numpy as np

    sp, genes = simulate_dataset("specieswise", tmp_path, n_species=60, n_families=80,
                                 dtl=0.05, seed=5)
    m = GeneReconModel(sp, genes, mode="specieswise", dtype=torch.float32,
                       solver_options=SolverOptions())
    S = m.theta.shape[0]
    res = fit_specieswise(m.batch_statics, m.theta.detach(), m.receiver_weights.detach(),
                          lam=10.0, verbose=False)
    rates = np.asarray(res["rates"])
    assert res["mode"] == "specieswise" and res["lam"] == 10.0
    assert rates.shape == (S, 3) and np.isfinite(res["nll_bits"]) and np.isfinite(rates).all()
    assert res["gnorm"] < 1.0, f"MAP not converged: |gF|={res['gnorm']}"

    # ridge sanity: a huge lam pins the optimum at theta_ref (a constant reference).
    theta_ref = torch.full((S, 3), float(np.log2(0.1)), device="cuda")
    res_big = fit_specieswise(m.batch_statics, m.theta.detach(), m.receiver_weights.detach(),
                              lam=1e6, theta_ref=theta_ref, verbose=False)
    drift = (res_big["theta"] - theta_ref.cpu()).abs().max().item()
    assert drift < 1e-1, f"large lam should hold theta near theta_ref, drift={drift}"
