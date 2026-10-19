"""Every family's log-likelihood (nats) against the number K of fixed-point iterations of the clade values, written to a
CSV file with one row per family and one column per K, the last one K = 256 (the reference). The extinction values use
the release's own iteration (at most 128 iterations, stopping when the change is below 1e-8); 64 terms of the adjoint
series, no warm start. TOML: data/configs/single_precision.toml (the release's default precision, written out). analysis/kscan_stats.py turns the
file into the per-K statistics of Figure "fig:kscan" and Table "tab:kscan".
The rates are either the same for every family, or each family's own rates from the folder model_parameters of a fit:
usage: kscan_per_family.py FAMILIES_FILE SPECIES_TREE TOML OUT_CSV fixed D L T
       kscan_per_family.py FAMILIES_FILE SPECIES_TREE TOML OUT_CSV fitted RATES_DIR
RATES_DIR: one FAMILY_rates.txt per family (rates D L T on the second line), as gpurec and AleRax write them."""
import csv
import math
import sys
import time

import torch
import gpurec
from gpurec.api.solver_options import SolverOptions
from gpurec.fit.genewise_fit import _BASE_SOLVER

# The numbers of iterations of the paper's scan; the last one is the reference.
KS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256)

FAM, SP, TOML, OUT, SOURCE = sys.argv[1:6]
cfg = gpurec.load_fit_config(TOML)
data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
n = data.n_families
model = data.build_model(list(range(n)), "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
model.receiver_weights.requires_grad_(False)
if SOURCE == "fixed":
    rates = [[float(x) for x in sys.argv[6:9]]] * n
elif SOURCE == "fitted":
    rates = [[float(x) for x in open(f"{sys.argv[6]}/{name}_rates.txt").read().split("\n")[1].split()]
             for name in data.names]
else:
    raise SystemExit(f"the rates are 'fixed D L T' or 'fitted RATES_DIR', not '{SOURCE}'")
theta = torch.tensor([[math.log2(r) for r in row] for row in rates], dtype=model.theta.dtype, device=model.theta.device)
base = {k: getattr(cfg.solver, k) for k in _BASE_SOLVER}


def logl(k):
    model.solver_options = SolverOptions(**{**base, "pi_iters": k, "neumann_terms": 64, "warm_adjoint": False})
    out = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)
    return (-out[0].double().cpu() * math.log(2.0)).tolist()


logl(64)   # untimed first evaluation: compiles the kernels
columns = {}
for k in KS:
    t = time.perf_counter()
    columns[k] = logl(k)
    print(f"K={k}: {time.perf_counter() - t:.2f} s", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family"] + [f"K{k}" for k in KS])
    for i, name in enumerate(data.names):
        w.writerow([name] + [repr(columns[k][i]) for k in KS])
print(f"{n} families written to {OUT}")
