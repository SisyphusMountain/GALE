"""Batching experiment of the appendix "Scheduling the operations on a GPU": the waves of the 1,055 HOGENOM families of
the folder HOGENOM_1055_DIR (species tree hogenom_S.tree, gene-tree samples hogenom_trees/FAMILY.trees, gene-to-species
mappings hogenom_mappings/FAMILY.map), all families kept (no filtering), planned by gpurec's Rust planner in two ways:
(A) each family alone; (B) at most 300 families and 315,000 clades per batch (the limit of an earlier version; the
release uses 80,000), 8,192 clades per wave, batches filled by depth. Writes, in OUT_DIR, one CSV row per wave
(waves_alone.csv, waves_default.csv), one per batch (batches_*.csv) and one per family (families.csv), and prints a
summary; figures/make_fig_waves.py draws the figure from the CSV files.
usage: waves_figure_data.py HOGENOM_1055_DIR OUT_DIR"""
import csv
import importlib
import json
import math
import sys
import time
from pathlib import Path

H, OUT = Path(sys.argv[1]), Path(sys.argv[2])
W = 8192
SMS = 132  # streaming multiprocessors on an H100; one block of threads per clade
m = importlib.import_module("gpurec.gpurec_preprocess")


def readmap(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        sp, genes = line.split(":", 1)
        for g in genes.split(";"):
            if g:
                d[g] = sp
    return d


SPECIES = str(H / "hogenom_S.tree")
names = sorted(p.stem for p in (H / "hogenom_trees").glob("*.trees"))
PATHS = [str(H / "hogenom_trees" / f"{n}.trees") for n in names]
MAPS = [readmap(H / "hogenom_mappings" / f"{n}.map") for n in names]
OUT.mkdir(parents=True, exist_ok=True)
t = time.time()
raw = json.loads(m.preprocess_dataset(
    SPECIES, PATHS, MAPS,
    None, None, "sequential", W, None))
fams = raw["families"]
del raw
print(f"preprocessed {len(fams)} families in {time.time()-t:.0f} s", flush=True)
slim = [{k: f[k] for k in ("C", "N_splits", "split_counts", "split_parents_sorted",
                            "split_leftrights_sorted", "log_split_probs_sorted",
                            "root_clade_id", "schedule_depth", "leaf_row_index",
                            "leaf_col_index")} for f in fams]
fams_json = json.dumps(slim)
C = [int(f["C"]) for f in fams]
nleaf = [len(f["leaf_row_index"]) for f in fams]
depth = [int(f["schedule_depth"]) for f in fams]
with open(OUT / "families.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["family", "clades", "leaf_clades", "depth"])
    for row in zip(names, C, nleaf, depth):
        w.writerow(row)

configs = {
    "alone": dict(chunk=1, budget=None, packing="sequential"),
    "default": dict(chunk=300, budget=315_000, packing="depth_first_fit"),
}
for label, cfg in configs.items():
    t = time.time()
    out = json.loads(m.plan_batch_layouts(fams_json, cfg["chunk"], cfg["budget"], cfg["packing"], W, None))
    print(f"[{label}] planned in {time.time()-t:.0f} s", flush=True)
    waves, batches = [], []
    for b, (idx, lay) in enumerate(zip(out["batches"], out["batch_wave_layouts"])):
        metas = lay["plan"]["wave_metas"]
        sizes = [(int(x["W"]), int(x.get("phase", 2 if x.get("has_splits") else 1))) for x in metas]
        L = sum(nleaf[i] for i in idx); J = sum(C[i] - nleaf[i] for i in idx)
        lb = math.ceil(L / W) + max(max(depth[i] for i in idx), math.ceil(J / W))
        batches.append([b, len(idx), L + J, L, J, max(depth[i] for i in idx), len(sizes), lb])
        for k, (s, ph) in enumerate(sizes):
            waves.append([b, k, s, ph])
    with open(OUT / f"waves_{label}.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["batch", "wave", "clades", "phase"]); wr.writerows(waves)
    with open(OUT / f"batches_{label}.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["batch", "families", "clades", "leaf_clades", "nonleaf_clades", "max_depth", "waves", "lower_bound"]); wr.writerows(batches)
    n = len(waves); tot = sum(r[2] for r in waves)
    full = sum(r[2] == W for r in waves)
    small = sum(r[2] < SMS for r in waves)
    print(f"[{label}] batches {len(batches)}  waves {n}  clades {tot}")
    print(f"  full waves {full} ({100*full/n:.1f} %), clades in full waves {100*sum(r[2] for r in waves if r[2]==W)/tot:.1f} %")
    print(f"  mean fill {100*tot/(n*W):.2f} %, waves under {SMS} clades {small} ({100*small/n:.1f} %)")
    print(f"  waves / lower bound {sum(b[6] for b in batches)} / {sum(b[7] for b in batches)}")
