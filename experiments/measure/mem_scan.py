"""Peak GPU memory of one loss-and-gradient evaluation against the clade limit of a batch (10,000 to 160,000 clades),
at fixed rates (log2 rates -2.32, -1.74, -3.32, that is about 0.2, 0.3, 0.1) for every family.
usage: mem_scan.py FAMILIES_FILE SPECIES_TREE TOML"""
import sys

import torch
import gpurec

FAM, SP, TOML = sys.argv[1:4]
cfg = gpurec.load_fit_config(TOML)
data = gpurec.load_families(FAM, SP, 4, 0.0, None, None, False)
n = data.n_families
clades = sum(int(f["C"]) for f in data.families)
print(f"{n} families, {clades:,} clades, {data.species.__class__.__name__}")
for budget in (10000, 20000, 40000, 80000, 160000):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m = data.build_model(list(range(n)), "genewise", cfg.solver, cfg, budget)
        m.receiver_weights.requires_grad_(False)
        theta = torch.tensor([-2.32, -1.74, -3.32], dtype=m.theta.dtype, device=m.theta.device).reshape(1, 3).expand(n, 3).clone()
        base = torch.cuda.memory_allocated() / 2**30
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=True); torch.cuda.synchronize()
        print(f"clade limit {budget:>7,}: batches {len(m.family_batches):>3}, memory after build {base:.2f} GiB, peak of loss+gradient {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        del m
    except torch.OutOfMemoryError as e:
        print(f"clade limit {budget:>7,}: out of memory")
        torch.cuda.empty_cache()
