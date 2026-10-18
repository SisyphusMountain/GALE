"""Where the difference of about 1e-4 between gpurec's gradient and central finite differences (measure/grad_test.py)
comes from: the first 20 families of the file, double precision, the same rates for every family, at the two rate
points (0.2, 0.3, 0.1) and (0.05, 0.05, 0.05).
The finite differences use a step of 1e-2 in each of the three base-2 logarithms of the rates, with the forward solves
run far past convergence (512 fixed-point iterations of the rows, 2,048 of the extinction values). The gradient is then
computed with adjoint pruning on (the threshold of the configuration file) and truly off (use_adjoint_pruning = false and
threshold 0, as in data/configs/no_pruning.toml: false alone keeps the threshold and skips the same clades), each with
64 fixed-point iterations of the rows, 64 terms of the adjoint series and 128 iterations for the extinction values and
their adjoint (the default values of gpurec's SolverOptions), and with eight to sixteen times more, no warm start.
Prints, for each, the relative difference (norm of the difference over norm of the finite differences) to the finite
differences.
usage: grad_resid.py FAMILIES_FILE SPECIES_TREE DOUBLE_TOML"""
import sys

import torch
import gpurec
from gpurec.api.solver_options import SolverOptions
from gpurec.fit.genewise_fit import _BASE_SOLVER

FAM, SP, T64 = sys.argv[1:4]
N_FAMILIES = 20                    # as measure/grad_test.py, part 2
STEP = 1e-2                        # step of the finite differences, in base-2 logarithm of the rates
RATE_POINTS = [(0.2, 0.3, 0.1), (0.05, 0.05, 0.05)]
# (fixed-point iterations of the rows, terms of the adjoint series, iterations of the extinction values, iterations of
# their adjoint): the default values of SolverOptions, then eight to sixteen times more
SETTINGS = [(64, 64, 128, 128), (512, 512, 2048, 2048)]

cfg = gpurec.load_fit_config(T64)
data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
model = data.build_model(list(range(N_FAMILIES)), "genewise", cfg.solver, cfg, cfg.batching.clade_budget)
model.receiver_weights.requires_grad_(False)
base = {k: getattr(cfg.solver, k) for k in _BASE_SOLVER}


def set_options(pi, neumann, e_max, e_adjoint, pruning):
    model.solver_options = SolverOptions(**{
        **base, "pi_iters": pi, "neumann_terms": neumann, "e_max_iter": e_max, "e_adjoint_max_iter": e_adjoint,
        "use_adjoint_pruning": pruning,
        "adjoint_pruning_threshold": cfg.solver.adjoint_pruning_threshold if pruning else 0.0, "warm_adjoint": False})


def theta_for(rates):
    t = torch.tensor([float(torch.log2(torch.tensor(x))) for x in rates], dtype=model.theta.dtype, device=model.theta.device)
    return t.reshape(1, 3).expand(N_FAMILIES, 3).clone()


def loss(theta):
    return model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)[0].double().cpu()


def grad(theta):
    return model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)[1].double().cpu()


for rates in RATE_POINTS:
    theta = theta_for(rates)
    set_options(512, 512, 2048, 2048, False)        # converged forward solves for the finite differences
    fd = torch.zeros(N_FAMILIES, 3, dtype=torch.float64)
    for k in range(3):
        tp, tm = theta.clone(), theta.clone()
        tp[:, k] += STEP
        tm[:, k] -= STEP
        fd[:, k] = (loss(tp) - loss(tm)) / (2 * STEP)
    print(f"rates {rates}: finite differences with fully converged forward solves (step {STEP})")
    for pi, neumann, e_max, e_adjoint in SETTINGS:
        for pruning in (True, False):
            set_options(pi, neumann, e_max, e_adjoint, pruning)
            g = grad(theta)
            print(f"  gradient with pi_iters={pi} neumann_terms={neumann} e_max_iter={e_max} e_adjoint_max_iter={e_adjoint} "
                  f"adjoint_pruning={pruning}: relative L2 difference {(g - fd).norm() / fd.norm():.3e}")
