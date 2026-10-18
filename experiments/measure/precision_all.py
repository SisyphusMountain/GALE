"""Loss of every family at its own fitted rates, single precision (double-precision offsets and sums) against double
precision everywhere, one evaluation each (no gradient), timed. The loss of a family is -log2 of its likelihood (bits).
Writes one line per family: name, clades, loss in single precision, loss in double precision.
usage: precision_all.py FAMILIES_FILE SPECIES_TREE RATES_DIR SINGLE_TOML DOUBLE_TOML OUT_TSV
RATES_DIR: the folder model_parameters of a gpurec fit of the same families (FAMILY_rates.txt files)."""
import math
import sys
import time

import torch
import gpurec

FAM, SP, RATES, T32, T64, OUT = sys.argv[1:7]


def run(toml):
    cfg = gpurec.load_fit_config(toml)
    data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
    n = data.n_families
    model = data.build_model(list(range(n)), "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
    model.receiver_weights.requires_grad_(False)
    rows = []
    for name in data.names:
        d, l, t = (float(x) for x in open(f"{RATES}/{name}_rates.txt").read().split("\n")[1].split())
        rows.append([math.log2(d), math.log2(l), math.log2(t)])
    theta = torch.tensor(rows, dtype=model.theta.dtype, device=model.theta.device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    lv = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)[0].double().cpu()
    torch.cuda.synchronize()
    clades = [int(f["C"]) for f in data.families]
    return list(data.names), lv, time.perf_counter() - t0, clades


names, l32, t32, clades = run(T32)
_, l64, t64, _ = run(T64)
d = l32 - l64
with open(OUT, "w") as f:
    f.write("family\tclades\tloss_bits_fp32\tloss_bits_fp64\n")
    for n, c, a, b in zip(names, clades, l32.tolist(), l64.tolist()): f.write(f"{n}\t{c}\t{a!r}\t{b!r}\n")
print(f"{len(names)} families: time of one loss evaluation fp32 {t32:.1f} s, fp64 {t64:.1f} s (ratio {t64/t32:.1f}); max |diff| {d.abs().max():.3e} bits, median {d.abs().median():.3e}, above 1e-4 bits: {int((d.abs() > 1e-4).sum())}, total rel diff {abs(l32.sum()-l64.sum())/abs(l64.sum()):.3e}")
