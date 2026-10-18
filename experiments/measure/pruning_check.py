"""What adjoint pruning does in release 0.1 (the paper: the 1,000-family nested subset), with one loss and gradient at
the fit's solver settings (16 fixed-point iterations, 16 terms of the adjoint series, no warm start), at the starting
rates 0.1 and at fitted rates. Three settings: 'on' (use_adjoint_pruning=True, threshold 1e-6: a clade is kept when the
largest |right-hand side| of its row is >= 1e-6), 'off as released' (use_adjoint_pruning=False, same threshold: kept
when > 1e-6), and 'truly off' (use_adjoint_pruning=False, threshold 0: kept unless its row is exactly zero, which
changes nothing). Counts the clades skipped (wrapping compute_active_adjoint_row_mask), times the evaluation
(CUDA-synchronised, median of 3 after one warm-up), and compares losses and gradients with 'truly off'.
usage: pruning_check.py FAMILIES_FILE SPECIES_TREE RATES_DIR
RATES_DIR: the folder model_parameters of a gpurec fit of the same families (the paper: the ablation run with the
release settings)."""
import math
import statistics as st
import sys
import time
from pathlib import Path

import torch
import gpurec
import gpurec.api._implicit_grad as ig
from gpurec.api.solver_options import SolverOptions
from gpurec.fit import genewise_fit as gf

FAM, ST, RATES = sys.argv[1:4]
data = gpurec.load_families(families=Path(FAM), species_tree=Path(ST), min_covered_species=4, trim_ratio=0.0,
                            max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False)
config = gpurec.load_fit_config(None)
dtype = config.precision.model_torch_dtype
counts = {"rows": 0, "skipped": 0}
mask_fn = ig.compute_active_adjoint_row_mask


def counted(rhs, threshold, *, use_pruning):
    m = mask_fn(rhs, threshold, use_pruning=use_pruning)
    counts["rows"] += int(m.numel()); counts["skipped"] += int((~m).sum()); return m


ig.compute_active_adjoint_row_mask = counted
start = torch.full((data.n_families, 3), math.log2(0.1), dtype=torch.float64)
fitted = torch.tensor([[math.log2(float(x)) for x in open(f"{RATES}/{n}_rates.txt").read().split("\n")[1].split()] for n in data.names], dtype=torch.float64)
settings = {"on": (True, 1e-6), "off as released": (False, 1e-6), "truly off": (False, 0.0)}
for label, th0 in (("starting rates 0.1", start), ("fitted rates", fitted)):
    print(f"== {label}")
    res = {}
    for name, (use, thr) in settings.items():
        so = SolverOptions(**{**gf._BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16, "warm_adjoint": False,
                              "use_adjoint_pruning": use, "adjoint_pruning_threshold": thr})
        m = data.build_model(list(range(data.n_families)), "genewise", so, config, config.batching.clade_budget)
        m.receiver_weights.requires_grad_(False)
        th = th0.to(device="cuda", dtype=dtype)
        times = []
        for rep in range(4):
            counts["rows"] = counts["skipped"] = 0
            torch.cuda.synchronize(); t = time.perf_counter()
            lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        res[name] = (lv.double().cpu(), g.double().cpu())
        print(f"  {name:16s} clades skipped {counts['skipped']:8d} of {counts['rows']:8d} ({counts['skipped']/max(1,counts['rows']):.2%})   time {st.median(times[1:]):.3f} s (median of 3 after one warm-up)")
        del m; torch.cuda.empty_cache()
    ref_l, ref_g = res["truly off"]
    for name in ("on", "off as released"):
        l, g = res[name]
        print(f"  {name:16s} vs truly off: max |d loss| {(l - ref_l).abs().max():.3e} bits, relative grad diff {((g - ref_g).norm() / ref_g.norm()):.3e}, max |d grad| {(g - ref_g).abs().max():.3e}")
