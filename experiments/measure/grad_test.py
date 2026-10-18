"""Cost of one loss and gradient against one loss alone (single precision, the default), and check of the gradient
against central finite differences (double precision).
1. All families of the file, two rate points (the same rates for every family): after one warm-up of each, seven timed
   evaluations of the loss alone and seven of the loss and gradient; medians reported.
2. The first 20 families, double precision: the gradient against central finite differences in each of the three
   base-2 logarithms of the rates, for steps of 1e-3, 1e-2 and 5e-2.
usage: grad_test.py FAMILIES_FILE SPECIES_TREE SINGLE_TOML DOUBLE_TOML"""
import statistics
import sys
import time

import torch
import gpurec

FAM, SP, T32, T64 = sys.argv[1:5]


def build(toml, idx):
    cfg = gpurec.load_fit_config(toml)
    data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
    idx = list(range(data.n_families)) if idx is None else idx
    model = data.build_model(idx, "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
    model.receiver_weights.requires_grad_(False)
    return model, len(idx)


def theta_for(model, n, rates):
    t = torch.tensor([float(torch.log2(torch.tensor(r))) for r in rates], dtype=model.theta.dtype, device=model.theta.device)
    return t.reshape(1, 3).expand(n, 3).clone()


def timed(model, theta, need_grad):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = model.genewise_loss_vector_and_grad(theta=theta, need_grad=need_grad)
    torch.cuda.synchronize(); return time.perf_counter() - t, out


# 1. timing, float32, all families of the file
model, n = build(T32, None)
for rates in [(0.2, 0.3, 0.1), (0.05, 0.05, 0.05)]:
    theta = theta_for(model, n, rates)
    timed(model, theta, True); timed(model, theta, False)           # warm-up
    t_loss = [timed(model, theta, False)[0] for _ in range(7)]
    t_grad = [timed(model, theta, True)[0] for _ in range(7)]
    ml, mg = statistics.median(t_loss), statistics.median(t_grad)
    print(f"{n} families, rates {rates}, float32: loss alone {ml:.3f} s, loss+gradient {mg:.3f} s, ratio {mg/ml:.2f} (median of 7)")
del model; torch.cuda.empty_cache()

# 2. gradient against central finite differences, float64, first 20 families
model, n = build(T64, list(range(20)))
for h in (1e-3, 1e-2, 5e-2):                                        # step of the finite differences, in log2 units of the rates
    for rates in [(0.2, 0.3, 0.1), (0.05, 0.05, 0.05)]:
        theta = theta_for(model, n, rates)
        _, out = timed(model, theta, True)
        lv, g = out[0].double().cpu(), out[1].double().cpu()
        fd = torch.zeros_like(g)
        for k in range(3):
            tp, tm = theta.clone(), theta.clone(); tp[:, k] += h; tm[:, k] -= h
            lp = timed(model, tp, False)[1][0].double().cpu(); lm = timed(model, tm, False)[1][0].double().cpu()
            fd[:, k] = (lp - lm) / (2 * h)
        print(f"{n} families, rates {rates}, float64, step {h:g}: relative L2 difference to finite differences {(g-fd).norm()/fd.norm():.3e}, largest entry difference {(g-fd).abs().max():.3e} (largest gradient entry {fd.abs().max():.3e})")
