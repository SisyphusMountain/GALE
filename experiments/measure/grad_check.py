"""Loss and gradient of the families of a file (the paper: the 100-family nested subset), with the solver settings of
the fit (16 fixed-point iterations, 16 terms of the adjoint series, no warm start), at the starting rates 0.1, computed
three ways: as released, with adjoint pruning off (every clade of the backward pass computed: use_adjoint_pruning
false and threshold 0, as in data/configs/no_pruning.toml), and with the split term
stored by the forward pass instead of recomputed by the backward pass (as wrappers/gpurec_ablate.py, mode storealpha).
Prints the largest differences of the per-family losses and gradients against the released computation, and those of a
second released computation (the run-to-run spread).
usage: grad_check.py FAMILIES_FILE SPECIES_TREE"""
import math
import sys
from pathlib import Path

import torch
import gpurec
from gpurec.api.solver_options import SolverOptions
from gpurec.fit import genewise_fit as gf

FAM, ST = sys.argv[1:3]
data = gpurec.load_families(families=Path(FAM), species_tree=Path(ST), min_covered_species=4, trim_ratio=0.0,
                            max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False)
config = gpurec.load_fit_config(None)


def run(pruning):
    so = SolverOptions(**{**gf._BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16, "warm_adjoint": False, "use_adjoint_pruning": pruning,
                          "adjoint_pruning_threshold": gf._BASE_SOLVER["adjoint_pruning_threshold"] if pruning else 0.0})
    m = data.build_model(list(range(data.n_families)), "genewise", so, config, config.batching.clade_budget)
    m.receiver_weights.requires_grad_(False)
    th = torch.full((data.n_families, 3), math.log2(0.1), device="cuda", dtype=config.precision.model_torch_dtype)
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
    return lv.double().cpu(), g.double().cpu()


base = run(True); base2 = run(True); nopr = run(False)
import gpurec.api._implicit_grad as bm
import gpurec.core.inference.forward as fm
stored, cur = {}, {"b": None}; fdts = fm.compute_dts_forward


def fs(*a, **k):
    out = fdts(*a, **k); b = k["family_idx"].data_ptr()
    if cur["b"] != b: stored.clear(); cur["b"] = b
    stored[a[4].data_ptr()] = out; return out


def rs(*a, **k): return stored[a[4].data_ptr()]


fm.compute_dts_forward = fs; bm.compute_dts_forward = rs
store = run(True)


def diff(name, x):
    dl = (x[0] - base[0]).abs().max().item(); dg = (x[1] - base[1]).abs().max().item(); gmax = base[1].abs().max().item()
    rel = ((x[1] - base[1]).norm() / base[1].norm()).item()
    print(f"{name:34s} max |d loss| {dl:.3e} bits   max |d grad| {dg:.3e} (largest grad entry {gmax:.3e})   relative grad diff (norm) {rel:.3e}")


diff("released, second run (run-to-run)", base2); diff("adjoint pruning off", nopr); diff("split term stored", store)
