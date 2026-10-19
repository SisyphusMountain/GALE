"""Data of the appendix "Scheduling the operations on a GPU" (its table and figure hogenom_waves): the waves planned by
release 0.1 for the HOGENOM families of the paper (planning only, on the CPU), in two ways: with the release's limits (at
most 300 families and 80,000 clades per batch, batches filled by depth, waves of at most 8,192 clades) and each family
alone (one family per batch). Families are loaded exactly as in waves_table.py. Writes, in OUT_DIR, one CSV row per wave
(waves_default.csv, waves_alone.csv), one per batch (batches_default.csv, batches_alone.csv) and one per family
(families.csv), and prints the numbers of the appendix; figures/make_fig_waves.py draws the figure from the CSV files.
A wave is a set of clades computed by one launch of the kernels. The lower bound of a batch is
ceil(L/W) + max(max_f d_f, ceil(J/W)) (L single-gene clades, J other clades, d_f the depth of family f, W = 8,192).
usage: waves_figure_data.py FAMILIES_FILE SPECIES_TREE OUT_DIR"""
import csv
import math
import statistics as st
import sys
import time
from pathlib import Path

import gpurec
from gpurec.core.scheduling.batching import plan_batch_wave_layouts

FAM, ST, OUT = sys.argv[1:4]

W = 8192  # largest wave of release 0.1
SMS = 128  # streaming multiprocessors of an RTX 4090; one block of threads computes one clade
FAMILY_LIMIT, CLADE_LIMIT = 300, 80000  # batch limits of release 0.1
OUT = Path(OUT)
OUT.mkdir(parents=True, exist_ok=True)
data = gpurec.load_families(families=Path(FAM), species_tree=Path(ST), min_covered_species=4, trim_ratio=0.0,
                            max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False)
fams = list(data.families)
names = list(data.names)
C = [int(f["C"]) for f in fams]
nleaf = [len(f["leaf_row_index"]) for f in fams]
depth = [int(f["schedule_depth"]) for f in fams]
print(f"families {len(fams)}  clades {sum(C)}  median clades per family {st.median(C)}  largest {max(C)}")
print(f"depth: median {st.median(depth)}  deepest {max(depth)}")

batch_of_family = {}
for label, chunk in (("default", FAMILY_LIMIT), ("alone", 1)):
    t0 = time.perf_counter()
    raw = plan_batch_wave_layouts(fams, family_chunk_size=chunk, clade_budget=CLADE_LIMIT,
                                  batch_packing="depth_first_fit", max_wave_size=W, family_group_assignments=None)
    secs = time.perf_counter() - t0
    waves, batches = [], []
    for b, (idx, lay) in enumerate(zip(raw["batches"], raw["batch_wave_layouts"])):
        if label == "default":
            for i in idx:
                batch_of_family[i] = b
        roots = set(int(r) for r in lay["plan"]["root_clade_ids"])
        metas = lay["plan"]["wave_metas"]
        L = sum(nleaf[i] for i in idx)
        J = sum(C[i] - nleaf[i] for i in idx)
        lb = math.ceil(L / W) + max(max(depth[i] for i in idx), math.ceil(J / W))
        batches.append([b, len(idx), L + J, L, J, max(depth[i] for i in idx), len(metas), lb])
        for k, x in enumerate(metas):
            start, end = int(x["start"]), int(x["end"])
            assert end - start == int(x["W"])
            n_roots = sum(start <= r < end for r in roots)
            waves.append([b, k, int(x["W"]), int(x["phase"]), n_roots])
    with open(OUT / f"waves_{label}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["batch", "wave", "clades", "phase", "root_clades"])
        wr.writerows(waves)
    with open(OUT / f"batches_{label}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["batch", "families", "clades", "leaf_clades", "nonleaf_clades", "max_depth", "waves", "lower_bound"])
        wr.writerows(batches)

    sizes = [r[2] for r in waves]
    n, tot = len(sizes), sum(sizes)
    full = sum(s == W for s in sizes)
    small = sum(s < SMS for s in sizes)
    print(f"[{label}] planned in {secs:.1f} s")
    print(f"  batches {len(batches)}  waves {n}  clades {tot}")
    print(f"  full waves {full} ({100 * full / n:.1f} %), clades in full waves {100 * sum(s for s in sizes if s == W) / tot:.1f} %")
    print(f"  mean filling {100 * tot / (n * W):.2f} %  median clades per wave {st.median(sizes)}  largest wave {max(sizes)}")
    print(f"  waves under {SMS} clades {small} ({100 * small / n:.1f} %)")
    print(f"  waves {sum(b[6] for b in batches)}, lower bound {sum(b[7] for b in batches)}, "
          f"waves above the bound {sum(max(0, b[6] - b[7]) for b in batches)}")
    if label == "alone":
        continue
    # the deepest batch
    d = max(batches, key=lambda b: b[5])
    dw = [r[2] for r in waves if r[0] == d[0]]
    print(f"  deepest batch {d[0] + 1}: families {d[1]}  max depth {d[5]}  clades {d[2]}  leaf clades {d[3]}  "
          f"non-leaf clades {d[4]}  ceil(non-leaf / W) {math.ceil(d[4] / W)}  waves {d[6]}  lower bound {d[7]}  "
          f"mean filling {100 * sum(dw) / (len(dw) * W):.1f} %  full waves {sum(s == W for s in dw)}")
    # what bounds the waves of each batch: the depth or the number of non-leaf clades
    by_depth = sum(b[5] >= math.ceil(b[4] / W) for b in batches)
    print(f"  batches whose deepest family needs more waves than ceil(non-leaf / W): {by_depth} of {len(batches)}")
    # the last wave of each batch
    last = [[r for r in waves if r[0] == b[0]][-1] for b in batches]
    print(f"  last waves holding only root clades: {sum(r[4] == r[2] for r in last)} of {len(last)}; "
          f"clades in the last wave: median {st.median(r[2] for r in last)}, min {min(r[2] for r in last)}, "
          f"max {max(r[2] for r in last)}")
    # waves of single-gene clades (phase 1) and the full waves
    print(f"  waves of single-gene clades (phase 1): {sum(r[3] == 1 for r in waves)}; full waves among them: "
          f"{sum(r[3] == 1 and r[2] == W for r in waves)} of {full}")
    print(f"  batches whose waves = waves of single-gene clades + depth of the deepest family: "
          f"{sum(b[6] == math.ceil(b[3] / W) + b[5] for b in batches)} of {len(batches)}")
    # what fills each batch: batches are opened deepest families first, the last one holds what is left
    by_clades = [b for b in batches[:-1] if b[1] < FAMILY_LIMIT]
    by_fams = [b for b in batches[:-1] if b[1] == FAMILY_LIMIT]
    print(f"  batches with fewer than {FAMILY_LIMIT} families (last batch excepted): {len(by_clades)}, numbers "
          f"{[b[0] + 1 for b in by_clades][:1]} to {[b[0] + 1 for b in by_clades][-1:]}, clades {min(b[2] for b in by_clades)}"
          f" to {max(b[2] for b in by_clades)}, families {min(b[1] for b in by_clades)} to {max(b[1] for b in by_clades)}")
    print(f"  batches of {FAMILY_LIMIT} families (last batch excepted): {len(by_fams)}, numbers "
          f"{[b[0] + 1 for b in by_fams][:1]} to {[b[0] + 1 for b in by_fams][-1:]}, clades {min(b[2] for b in by_fams)}"
          f" to {max(b[2] for b in by_fams)}")
    print(f"  last batch {batches[-1][0] + 1}: families {batches[-1][1]}  clades {batches[-1][2]}")
    print(f"  mean filling per batch: median {st.median(100 * b[2] / (b[6] * W) for b in batches):.1f} %")

with open(OUT / "families.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["family", "clades", "leaf_clades", "depth", "default_batch"])
    for i in range(len(fams)):
        wr.writerow([names[i], C[i], nleaf[i], depth[i], batch_of_family[i]])
