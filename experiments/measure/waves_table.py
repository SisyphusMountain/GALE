"""Waves planned by release 0.1 for a set of families (planning only, on the CPU), with the release's limits (at most 300
families and 80,000 clades per batch, batches filled by depth, waves of at most 8,192 clades) against each family alone
(one family per batch). A wave is a set of clades computed by one launch of the kernels. For each: batches, waves,
median clades per wave, waves with fewer than 128 clades (the RTX 4090 has 128 multiprocessors), and waves above the
lower bound ceil(L/W) + max(max_f d_f, ceil(J/W)) of each batch (L single-gene clades, J other clades, d_f the depth of
family f, W = 8,192).
usage: waves_table.py FAMILIES_FILE SPECIES_TREE OUT_JSON"""
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import gpurec
from gpurec.core.scheduling.batching import plan_batch_wave_layouts

FAM, ST, OUT = sys.argv[1:4]
W = 8192
data = gpurec.load_families(families=Path(FAM), species_tree=Path(ST), min_covered_species=4, trim_ratio=0.0,
                            max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False)
fams = list(data.families)
result = {"families": len(fams), "clades": sum(int(f["C"]) for f in fams)}
for name, chunk in (("default batches", 300), ("each family alone", 1)):
    t0 = time.perf_counter()
    raw = plan_batch_wave_layouts(fams, family_chunk_size=chunk, clade_budget=80000, batch_packing="depth_first_fit",
                                  max_wave_size=W, family_group_assignments=None)
    secs = time.perf_counter() - t0
    sizes, above = [], 0
    for batch, layout in zip(raw["batches"], raw["batch_wave_layouts"]):
        waves = [int(m["W"]) for m in layout["plan"]["wave_metas"]]
        sizes += waves
        L = sum(len(fams[i]["leaf_row_index"]) for i in batch)
        J = sum(int(fams[i]["C"]) for i in batch) - L
        bound = math.ceil(L / W) + max(max(int(fams[i]["schedule_depth"]) for i in batch), math.ceil(J / W))
        above += max(0, len(waves) - bound)
    result[name] = {"batches": len(raw["batches"]), "waves": len(sizes), "median_clades_per_wave": st.median(sizes),
                    "waves_below_128": sum(s < 128 for s in sizes), "share_waves_below_128": sum(s < 128 for s in sizes) / len(sizes),
                    "waves_above_lower_bound": above, "planning_seconds": round(secs, 1), "clades_in_waves": sum(sizes)}
    print(name, result[name], flush=True)
json.dump(result, open(OUT, "w"), indent=1)
