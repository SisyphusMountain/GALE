"""gpurec log-likelihood (nats) of every family at one fixed set of rates, the same for all families, with converged
settings: 256 fixed-point iterations of the clade values (K), the extinction values iterated to their tolerance (at most
1,024 iterations), 64 terms of the adjoint series, no warm start.
usage: fixedrate_eval.py FAMILIES_FILE SPECIES_TREE TOML D L T OUT_FILE
TOML: data/configs/single_precision.toml or double_precision.toml. OUT_FILE: one line per family, "name log-likelihood"."""
import math
import sys

import torch
import gpurec
from gpurec.api.solver_options import SolverOptions
from gpurec.fit.genewise_fit import _BASE_SOLVER

FAM, SP, TOML, D, L, T, OUT = sys.argv[1:8]
cfg = gpurec.load_fit_config(TOML)
data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
n = data.n_families
model = data.build_model(list(range(n)), "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
model.receiver_weights.requires_grad_(False)
base = {k: getattr(cfg.solver, k) for k in _BASE_SOLVER}
model.solver_options = SolverOptions(**{**base, "pi_iters": 256, "e_max_iter": 1024, "neumann_terms": 64, "warm_adjoint": False})
t = torch.tensor([math.log2(float(D)), math.log2(float(L)), math.log2(float(T))], dtype=model.theta.dtype, device=model.theta.device)
theta = t.reshape(1, 3).expand(n, 3).clone()
out = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)
ll = -out[0].double().cpu() * math.log(2.0)
with open(OUT, "w") as f:
    for name, v in zip(data.names, ll.tolist()):
        f.write(f"{name} {v!r}\n")
print(f"{n} families at rates D={D} L={L} T={T}: total {ll.sum():.4f}")
