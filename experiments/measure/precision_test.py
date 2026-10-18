"""Single precision (with double-precision accumulators, the default) against double precision everywhere: loss and
gradient of every family at fixed rates, the same rates for all families, at two rate points. For each precision: one
warm-up evaluation (kernel compilation, caches), then three timed evaluations; the values of the last one are compared
and the median time is reported. The loss of a family is -log2 of its likelihood (bits).
usage: precision_test.py FAMILIES_FILE SPECIES_TREE SINGLE_TOML DOUBLE_TOML
(SINGLE_TOML, DOUBLE_TOML: data/configs/single_precision.toml and data/configs/double_precision.toml)"""
import statistics
import sys
import time

import torch
import gpurec

FAM, SP, T32, T64 = sys.argv[1:5]


def run(toml, rates):
    cfg = gpurec.load_fit_config(toml)
    data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
    model = data.build_model(list(range(data.n_families)), "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
    model.receiver_weights.requires_grad_(False)
    theta = torch.tensor([float(torch.log2(torch.tensor(r))) for r in rates], dtype=model.theta.dtype, device=model.theta.device).reshape(1, 3).expand(data.n_families, 3).clone()

    def once():
        torch.cuda.synchronize(); t = time.perf_counter()
        lv, g, _ = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize(); return lv.double().cpu(), g.double().cpu(), time.perf_counter() - t
    once()                                  # warm-up (kernel compilation, caches)
    runs = [once() for _ in range(3)]
    return runs[-1][0], runs[-1][1], statistics.median(r[2] for r in runs), data.n_families


for rates in [(0.2, 0.3, 0.1), (0.05, 0.05, 0.05)]:
    l32, g32, t32, n = run(T32, rates)
    l64, g64, t64, _ = run(T64, rates)
    dl = (l32 - l64).abs()
    print(f"rates D,L,T = {rates}: {n} families")
    print(f"  loss per family (bits): max |diff| = {dl.max():.3e}   total fp64 = {l64.sum():.6f}  relative diff of total = {abs(l32.sum()-l64.sum())/abs(l64.sum()):.3e}")
    print(f"  gradient: relative L2 difference = {(g32-g64).norm()/g64.norm():.3e}   max |entry diff| = {(g32-g64).abs().max():.3e}   largest fp64 entry = {g64.abs().max():.3e}")
    print(f"  time of one loss+gradient (median of 3): fp32 = {t32:.3f} s   fp64 = {t64:.3f} s   ratio = {t64/t32:.2f}")
