"""Checks the 'storealpha' variant of wrappers/gpurec_ablate.py (the split term, the first line of the recursion, stored
by the forward pass instead of recomputed by the backward pass) on the families of a file (the paper: the 1,000-family
nested subset), with one loss and gradient at the fit's solver settings (16 fixed-point iterations, 16 terms of the
adjoint series, no warm start) and the starting rates 0.1: counts the forward calls that store the split term, the
backward calls served from the store, and the largest number of bytes held in the store at once; also the peak GPU
memory of the evaluation with and without the store.
usage: storealpha_check.py FAMILIES_FILE SPECIES_TREE"""
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
so = SolverOptions(**{**gf._BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16, "warm_adjoint": False})


def evaluate():
    m = data.build_model(list(range(data.n_families)), "genewise", so, config, config.batching.clade_budget)
    m.receiver_weights.requires_grad_(False)
    th = torch.full((data.n_families, 3), math.log2(0.1), device="cuda", dtype=config.precision.model_torch_dtype)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    m.genewise_loss_vector_and_grad(theta=th, need_grad=True); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30, len(m.batch_statics), sum(int(f["C"]) for f in data.families)


peak_plain, nb, clades = evaluate()
import gpurec.api._implicit_grad as bm
import gpurec.core.inference.forward as fm
stored, cur, count = {}, {"b": None}, {"store": 0, "read": 0, "max_bytes": 0}
fdts = fm.compute_dts_forward


def fs(*a, **k):
    out = fdts(*a, **k); b = k["family_idx"].data_ptr()
    if cur["b"] != b: stored.clear(); cur["b"] = b
    stored[a[4].data_ptr()] = out; count["store"] += 1
    count["max_bytes"] = max(count["max_bytes"], sum(t.numel() * t.element_size() for pair in stored.values() for t in pair))
    return out


def rs(*a, **k):
    if cur["b"] != k["family_idx"].data_ptr() or a[4].data_ptr() not in stored: raise RuntimeError("missing")
    count["read"] += 1; return stored[a[4].data_ptr()]


fm.compute_dts_forward = fs; bm.compute_dts_forward = rs
peak_store, _, _ = evaluate()
S = int(data.species_raw["S"])
print(f"batches {nb}, clades {clades}, branches {S}")
print(f"forward calls storing the split term: {count['store']}; backward calls served from the store: {count['read']}")
print(f"largest store held at once: {count['max_bytes']/2**20:.1f} MiB")
print(f"peak GPU memory of one loss+gradient: recomputed {peak_plain:.3f} GiB, stored {peak_store:.3f} GiB")
