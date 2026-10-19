"""Figure hogenom_waves of the appendix "Scheduling the operations on a GPU": the sizes of the waves planned for the
10,869 HOGENOM families, each family alone against the release's batches (at most 300 families and 80,000 clades,
filled by depth). Reads the CSV files written by measure/waves_figure_data.py (run_planning.py part waves-figure:
OUTPUT_ROOT/planning/waves_figure).
(a) share of the waves by number of clades; (b) the clades of each wave of every batch, three batches highlighted.
usage: make_fig_waves.py CSV_FOLDER OUT_PDF"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 8192  # largest wave of release 0.1
SMS = 128  # streaming multiprocessors of an RTX 4090
HIGHLIGHT = (1, 30, 38)  # batch numbers (from 1) drawn in colour in panel (b): the deepest, a middle one, the first of 300 families
folder, out = Path(sys.argv[1]), sys.argv[2]


def load(label):
    return [(int(r["batch"]), int(r["clades"])) for r in csv.DictReader(open(folder / f"waves_{label}.csv"))]


alone, default = load("alone"), load("default")
batches = {int(r["batch"]): r for r in csv.DictReader(open(folder / "batches_default.csv"))}

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
bins = np.logspace(0, np.log2(W), 27, base=2)
for data, lab, col in ((alone, f"each family alone ({len(alone):,} waves)", "tab:orange"),
                       (default, f"release batches ({len(default):,} waves)", "tab:blue")):
    sizes = np.array([c for _, c in data])
    a.hist(sizes, bins=bins, weights=np.full(len(sizes), 100 / len(sizes)), alpha=0.6, color=col, label=lab)
a.axvline(W, color="k", ls="--", lw=1)
a.axvline(SMS, color="gray", ls=":", lw=1)
a.text(SMS * 1.1, 0.8, f"{SMS} multiprocessors\n(RTX 4090)", color="gray", fontsize=8, va="top",
       transform=a.get_xaxis_transform())
a.set_xscale("log", base=2)
a.set_xticks([1, 8, 64, 512, 4096], ["1", "8", "64", "512", "4096"])
a.set_xlabel("clades in the wave")
a.set_ylabel("% of waves")
a.legend(fontsize=8, loc="upper right")
a.set_title("(a) size of the waves", fontsize=10)

nb = max(bb for bb, _ in default) + 1
for bb in range(nb):
    if bb + 1 in HIGHLIGHT:
        continue
    sizes = [c for x, c in default if x == bb]
    b.plot(range(1, len(sizes) + 1), sizes, lw=0.6, color="0.8", zorder=1)
b.plot([], [], lw=0.6, color="0.8", label=f"the other {nb - len(HIGHLIGHT)} batches")
for num, col in zip(HIGHLIGHT, ("tab:red", "tab:purple", "tab:green")):
    sizes = [c for x, c in default if x == num - 1]
    r = batches[num - 1]
    b.plot(range(1, len(sizes) + 1), sizes, lw=1.4, color=col, zorder=2,
           label=f"batch {num}: {int(r['families'])} families, depth {r['max_depth']}")
b.axhline(W, color="k", ls="--", lw=1)
b.set_xlabel("wave number in the batch")
b.set_ylabel("clades in the wave")
b.legend(fontsize=8, loc="center right")
b.set_title(f"(b) the waves of the {nb} release batches", fontsize=10)
fig.tight_layout()
fig.savefig(out)
