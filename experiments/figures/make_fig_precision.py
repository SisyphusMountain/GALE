"""Figure S1 of the supplement: single against double precision over all families: histogram of the difference of -log2
of the likelihood of a family (bits), and the difference against the number of clades of the family. Reads the table
written by measure/precision_all.py (run_precision.py part all-families: OUTPUT_ROOT/precision/precision_all.tsv).
usage: make_fig_precision.py TSV OUT_PDF"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.genfromtxt(sys.argv[1], delimiter="\t", names=True, dtype=None, encoding=None)
clades = d["clades"].astype(float); err = np.abs(d["loss_bits_fp32"] - d["loss_bits_fp64"])
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
a.hist(np.log10(err + 1e-12), bins=60, color="#1f5fa8"); a.set_yscale("log")
a.set_xlabel(r"$\log_{10}$ of the difference between single and double precision (bits)"); a.set_ylabel("families"); a.set_title("(a)", fontsize=10)
b.scatter(clades, err + 1e-12, s=3, alpha=0.3, color="#1f5fa8"); b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel("clades in the family"); b.set_ylabel("difference (bits)"); b.set_title("(b)", fontsize=10)
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(sys.argv[2]); print("written", sys.argv[2])
