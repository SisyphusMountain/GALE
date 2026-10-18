"""Likelihood of families at their own fitted rates against the number of fixed-point iterations of the clade values (K)
and of the extinction values. The reference is the largest setting (K=256; 1,024 extinction iterations). Differences
are in nats.
usage: kscan.py FAMILIES_FILE SPECIES_TREE RATES_DIR TOML
RATES_DIR: the folder model_parameters of a gpurec fit covering these families (FAMILY_rates.txt files)."""
import math
import sys
import time

import torch
import gpurec
from gpurec.api.solver_options import SolverOptions
from gpurec.fit.genewise_fit import _BASE_SOLVER

FAM, SP, RATES, TOML = sys.argv[1:5]

cfg = gpurec.load_fit_config(TOML)
data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
n = data.n_families
model = data.build_model(list(range(n)), "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
model.receiver_weights.requires_grad_(False)
rows = []
for name in data.names:
    lines = open(f"{RATES}/{name}_rates.txt").read().split("\n")
    d, l, t = (float(x) for x in lines[1].split())
    rows.append([math.log2(d), math.log2(l), math.log2(t)])
theta = torch.tensor(rows, dtype=model.theta.dtype, device=model.theta.device)
base = {k: getattr(cfg.solver, k) for k in _BASE_SOLVER}


def logl(pi_iters, e_max_iter):
    model.solver_options = SolverOptions(**{**base, "pi_iters": pi_iters, "e_max_iter": e_max_iter, "neumann_terms": 64, "warm_adjoint": False})
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)
    torch.cuda.synchronize()
    return -out[0].double().cpu() * math.log(2.0), time.perf_counter() - t0


def report(label, settings, ref):
    for s in settings:
        v, t = logl(*s)
        d = (v - ref).abs()
        q = torch.quantile(d, 0.95).item()
        print(f"{label} {s}: max |diff| {d.max():.3e}  95th pct {q:.3e}  median {d.median():.3e}  families >0.01 nat: {(d>0.01).sum().item()}  >1 nat: {(d>1).sum().item()}  ({t:.2f} s)")


print(f"{n} families at their own fitted rates; model {gpurec.__file__}")
logl(64, 128)                                                   # warm-up
ref_k, _ = logl(256, 128)
print("-- clade iterations K (extinction values iterated to their tolerance, at most 128), reference K=256")
report("K", [(k, 128) for k in (2, 4, 8, 12, 16, 24, 32, 48, 64, 128)], ref_k)
ref_e, _ = logl(256, 1024)
print("-- extinction iterations (K=256), reference 1024 iterations")
report("E", [(256, e) for e in (1, 2, 4, 8, 16, 32, 64, 128, 256)], ref_e)
