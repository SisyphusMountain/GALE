import torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.api._execution import stream_batches
from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig


def _build(tmp_path, mode, fm):
    sp = tmp_path / "s.nwk"; sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    return GeneReconModel(str(sp), [str(g)], mode=mode,
                          dtype=torch.float64, fraction_missing=fm, clade_budget=GpurecConfig().batching.clade_budget)


def _loss_and_grad(m, need_grad):
    loss, grad, _, _ = stream_batches(
        m.batch_statics, m.theta, m.receiver_weights, m.origination_weights,
        genewise=m.genewise, need_grad=need_grad,
    )
    return loss, grad


@pytest.mark.parametrize("mode", ["global", "specieswise"])
def test_grad_matches_central_fd_with_fraction_missing(tmp_path, mode):
    m = _build(tmp_path, mode, 0.3)
    g = _loss_and_grad(m, True)[1].clone()
    eps = 1e-5
    max_rel = 0.0
    with torch.no_grad():
        flat = m.theta.view(-1)
        for i in range(flat.numel()):
            base = flat[i].item()
            flat[i] = base + eps; fp = float(_loss_and_grad(m, False)[0].item())
            flat[i] = base - eps; fmv = float(_loss_and_grad(m, False)[0].item())
            flat[i] = base
            fd = (fp - fmv) / (2 * eps)
            ga = float(g.view(-1)[i])
            rel = abs(fd - ga) / max(1.0, abs(fd))
            max_rel = max(max_rel, rel)
            assert rel < 1e-3, (
                f"mode={mode} i={i}: analytic={ga:.8e} fd={fd:.8e} rel={rel:.3e}"
            )
    print(f"mode={mode} max_rel={max_rel:.3e}")
