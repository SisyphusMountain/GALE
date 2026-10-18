"""gpurec log-likelihood (nats) of every family at its own given rates, with converged settings: 256 fixed-point
iterations of the clade values (K), the extinction values iterated to their tolerance (at most 1,024 iterations), 64
terms of the adjoint series, no warm start. Used to score the rates fitted by AleRax and by gpurec with the same
evaluation (in double precision in the paper).
usage: eval_at_rates.py FAMILIES_FILE SPECIES_TREE FRACTION_MISSING TOML RATES OUT_FILE
FRACTION_MISSING: a file of missing fractions per species (archaeal families), or the word none.
RATES: one line per family, "name D L T" (for instance AleRax's fitted rates). OUT_FILE: one line per family,
"name log-likelihood"."""
import math
import sys

import torch
import gpurec
from gpurec.api.solver_options import SolverOptions
from gpurec.fit.genewise_fit import _BASE_SOLVER

FAM, SP, FM, TOML, RATES, OUT = sys.argv[1:7]
cfg = gpurec.load_fit_config(TOML)
data = gpurec.load_families(FAM, SP, 4, 0.0, None, None if FM == "none" else FM, False)
n = data.n_families
model = data.build_model(list(range(n)), "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
if FM != "none":
    assert model.leaf_fm_log is not None, "the fraction-missing file did not reach the model"
model.receiver_weights.requires_grad_(False)
base = {k: getattr(cfg.solver, k) for k in _BASE_SOLVER}
model.solver_options = SolverOptions(**{**base, "pi_iters": 256, "e_max_iter": 1024, "neumann_terms": 64, "warm_adjoint": False})
rates = {}
for line in open(RATES):
    name, d, l, t = line.split()
    rates[name] = [math.log2(float(d)), math.log2(float(l)), math.log2(float(t))]
assert set(rates) == set(data.names), "rates file and family file name different families"
theta = torch.tensor([rates[name] for name in data.names], dtype=model.theta.dtype, device=model.theta.device)
out = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)
ll = -out[0].double().cpu() * math.log(2.0)
with open(OUT, "w") as f:
    for name, v in zip(data.names, ll.tolist()):
        f.write(f"{name} {v!r}\n")
print(f"{n} families at the given per-family rates: total {ll.sum():.4f}")
