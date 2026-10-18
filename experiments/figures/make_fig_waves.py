"""Figure "hogenom_waves" of the appendix "Scheduling the operations on a GPU": the sizes of the waves planned for the
1,055 HOGENOM families, each family alone against the batches of at most 300 families and 315,000 clades. Reads the CSV
files written by measure/waves_figure_data.py (run_planning.py part waves-figure: OUTPUT_ROOT/planning/waves_1055).
usage: make_fig_waves.py CSV_FOLDER OUT_PDF"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 8192
folder, out = Path(sys.argv[1]), sys.argv[2]
def load(label):
    return [(int(r["batch"]), int(r["clades"])) for r in csv.DictReader(open(folder / f"waves_{label}.csv"))]
alone, default = load("alone"), load("default")

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
bins = np.logspace(0, np.log2(W), 27, base=2)
for data, lab, col in ((alone, f"each family alone ({len(alone):,} waves)", "tab:red"),
                       (default, f"default batches ({len(default):,} waves)", "tab:blue")):
    sizes = np.array([c for _, c in data])
    a.hist(sizes, bins=bins, weights=np.full(len(sizes), 100 / len(sizes)), alpha=0.6, color=col, label=lab)
a.axvline(W, color="k", ls="--", lw=1); a.axvline(132, color="gray", ls=":", lw=1)
a.text(132 * 1.1, 20, "132 multiprocessors\n(H100)", color="gray", fontsize=8, va="top")
a.set_xscale("log", base=2); a.set_xticks([1, 8, 64, 512, 4096], ["1", "8", "64", "512", "4096"]); a.set_xlabel("clades in the wave"); a.set_ylabel("% of waves")
a.legend(fontsize=8, loc="upper left"); a.set_title("(a) size of the waves", fontsize=10)

nb = max(bb for bb, _ in default) + 1
for bb in range(nb):
    sizes = [c for x, c in default if x == bb]
    b.plot(range(1, len(sizes) + 1), sizes, lw=1.2, label=f"batch {bb + 1}")
b.axhline(W, color="k", ls="--", lw=1)
b.set_xlabel("wave number in the batch"); b.set_ylabel("clades in the wave")
b.legend(fontsize=8); b.set_title("(b) the waves of each default batch", fontsize=10)
fig.tight_layout(); fig.savefig(out)
