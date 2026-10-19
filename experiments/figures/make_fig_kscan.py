"""Figure "fig:kscan" (appendix on the fixed-point iterations): the per-family size of the difference of the
log-likelihood (nats) against the number K of fixed-point iterations, 200 HOGENOM families, for K up to 64. (a) At the
rates fitted to each family, against K = 256; (b) at the rates (0.2, 0.3, 0.1), against AleRax 1.4.0 with the four
changes. Median, 95th and 99th percentiles and the worst family. Differences of exactly 0 are drawn at the bottom edge
of the plot.
Reads the two files of analysis/kscan_stats.py: FITTED_STATS_CSV (mode "fitted") and FIXED_STATS_CSV (mode "fixed").
usage: make_fig_kscan.py FITTED_STATS_CSV FIXED_STATS_CSV OUT_PDF"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FITTED, FIXED, OUT = sys.argv[1:4]
FLOOR = 1e-12      # bottom edge of the plot: differences of exactly 0 are drawn there
K_LARGEST = 64     # the paper's figure stops at K = 64 (the statistics files also hold K = 96, 128 and 256)
SERIES = (("median", "median", "#9ab8dc"), ("p95", "95th percentile", "#5b8fc9"),
          ("p99", "99th percentile", "#1f5fa8"), ("max", "worst family", "#0b2d5c"))


def read(path, ref):
    rows = list(csv.DictReader(open(path)))
    return rows if ref is None else [r for r in rows if r["ref"] == ref]


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), sharey=True)
for ax, rows, title in ((axes[0], read(FITTED, None), "(a) fitted rates, against K = 256"),
                        (axes[1], read(FIXED, "AleRax"), "(b) rates (0.2, 0.3, 0.1), against corrected AleRax")):
    shown = [r for r in rows if int(r["K"]) <= K_LARGEST]
    ks = [int(r["K"]) for r in shown]
    for key, label, colour in SERIES:
        ax.plot(ks, [max(float(r[key]), FLOOR) for r in shown], marker="o", ms=4, lw=1.8, color=colour, label=label)
    ax.axvline(16, color="#888888", lw=0.8, ls=":")
    ax.text(17, 3e2, "K = 16", fontsize=8, color="#555555")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.5, 1e3)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=8)
    ax.set_xlabel("iterations K")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
axes[0].set_ylabel("|difference| of the log-likelihood (nats)")
axes[1].legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT)
print("written", OUT)
