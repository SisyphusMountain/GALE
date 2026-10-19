"""How many terms the adjoint series of the clade rows needs (release 0.1, 1,000-family nested subset, one loss and
gradient). The backward pass runs a fixed number of terms of the series for every wave, with no convergence test. This
script computes the gradient with 2, 4, 8, 16, 32 and 64 terms and compares it with a reference of 256 terms, at the
starting rates 0.1 and at fitted rates. The forward pass uses 64 iterations, pruning of the adjoint is
switched off (use_adjoint_pruning = false and threshold 0) and the warm start is off, so that the only difference between
the runs is the number of terms. Errors are reported relative to the reference gradient and in absolute value, against the
fit's gradient tolerance (relative errors are meaningless near the optimum, where the gradient is near zero). Each
evaluation is timed (CUDA-synchronised, median of 3 after one warm-up).
usage: adjoint_terms.py FAMILIES_FILE SPECIES_TREE RATES_DIR
RATES_DIR: the folder model_parameters of a fit of the same families (the paper: the ablation run with the release
settings)."""
import math, statistics as st, sys, time
FAM, ST, RATES = sys.argv[1:4]
from pathlib import Path
import torch, gpurec
from gpurec.fit import genewise_fit as gf
from gpurec.api.solver_options import SolverOptions
data = gpurec.load_families(families=Path(FAM), species_tree=Path(ST), min_covered_species=4, trim_ratio=0.0,
                            max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False)
config = gpurec.load_fit_config(None)
dtype = config.precision.model_torch_dtype
TERMS = (2, 4, 8, 16, 32, 64)
REFERENCE_TERMS = 256
# The per-family fit stops when the gradient of every family is below 1e-3 (tol of GENEWISE_REFERENCE in
# gpurec/fit/genewise_fit.py), so an error of the gradient matters when it reaches that size.
FIT_TOL = 1e-3
start = torch.full((data.n_families, 3), math.log2(0.1), dtype=torch.float64)
fitted = torch.tensor([[math.log2(float(x)) for x in open(f"{RATES}/{n}_rates.txt").read().split("\n")[1].split()]
                       for n in data.names], dtype=torch.float64)


def evaluate(theta, terms):
    so = SolverOptions(**{**gf._BASE_SOLVER, "pi_iters": 64, "neumann_terms": terms, "warm_adjoint": False,
                          "use_adjoint_pruning": False, "adjoint_pruning_threshold": 0.0})
    m = data.build_model(list(range(data.n_families)), "genewise", so, config, config.batching.clade_budget)
    m.receiver_weights.requires_grad_(False)
    th = theta.to(device="cuda", dtype=dtype)
    times = []
    for _ in range(4):
        torch.cuda.synchronize(); t = time.perf_counter()
        _, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    del m; torch.cuda.empty_cache()
    return g.double().cpu(), st.median(times[1:])


for label, theta in (("starting rates 0.1", start), ("fitted rates", fitted)):
    ref, t_ref = evaluate(theta, REFERENCE_TERMS)
    print(f"== {label}: reference {REFERENCE_TERMS} terms, {t_ref:.3f} s", flush=True)
    for terms in TERMS:
        g, t = evaluate(theta, terms)
        per_family = (g - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
        absolute = (g - ref).abs().max(dim=1).values   # largest error on one rate of the family
        print(f"  {terms:3d} terms: relative gradient difference {((g - ref).norm() / ref.norm()):.3e} (all families), "
              f"largest per family {per_family.max():.3e}, 99th percentile {torch.quantile(per_family, 0.99):.3e}; "
              f"absolute error on one rate: largest {absolute.max():.3e}, 99th percentile {torch.quantile(absolute, 0.99):.3e}, "
              f"families above the fit's gradient tolerance {FIT_TOL:g}: {(absolute > FIT_TOL).sum().item()}; time {t:.3f} s", flush=True)
