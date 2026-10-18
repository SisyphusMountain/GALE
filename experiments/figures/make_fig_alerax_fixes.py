"""Figure "fig_alerax_fixes": gpurec minus AleRax, per family, at the same fixed rates (0.2, 0.3, 0.1) on 1,042 families,
for four versions of AleRax 1.4.0: (a) as released (4 passes); (b) 16 passes and the addition that keeps the smaller
term; (c) the same and the switch to scaled values whenever the likelihood is below the smallest normal double; (d) the
same and the subtraction that keeps the smaller term. Points are coloured by the range of the family's log-likelihood,
which decides whether AleRax computes it with plain doubles or with scaled values.
Reads OUTPUT_ROOT/fixed_rates (run_fixed_rates.py parts gpurec-1042 and alerax-1042).
usage: make_fig_alerax_fixes.py PATHS_TOML OUT_PDF"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import output_dir, read_paths

paths, out = read_paths(sys.argv[1]), sys.argv[2]
D = output_dir(paths, "fixed_rates")
# Log of the smallest normal double (2.2e-308) and of the smallest positive double (4.9e-324): facts of the number format.
LOG_MIN_NORMAL = math.log(2.2250738585072014e-308)   # -708.40
LOG_MIN_POSITIVE = math.log(4.9406564584124654e-324)  # -744.44


def read(p): return {l.split()[0]: float(l.split()[1]) for l in open(p) if l.strip()}


g = read(D / "gpurec_1042.txt")
versions = [("(a) AleRax 1.4.0 as released", read(D / "out_passes_4" / "per_fam_likelihoods.txt")),
            ("(b) 16 passes, addition corrected", read(D / "out_addition_16" / "per_fam_likelihoods.txt")),
            ("(c) (b) and switch to scaled values", read(D / "out_switch_16" / "per_fam_likelihoods.txt")),
            ("(d) (c) and subtraction corrected", read(D / "out_four_16" / "per_fam_likelihoods.txt"))]
groups = [("above $-708.4$", lambda ll: ll >= LOG_MIN_NORMAL, "#1f5fa8"),
          ("$-744.4$ to $-708.4$", lambda ll: LOG_MIN_POSITIVE <= ll < LOG_MIN_NORMAL, "#e67e22"),
          ("below $-744.4$", lambda ll: ll < LOG_MIN_POSITIVE, "#2e7d32")]

fig, grid = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True, sharey=True)
axes = grid.ravel()
for ax, (title, a) in zip(axes, versions):
    names = sorted(set(g) & set(a)); assert len(names) == len(g) == len(a)
    for label, sel, colour in groups:
        k = [n for n in names if sel(g[n])]
        ax.scatter([-g[n] for n in k], [g[n] - a[n] for n in k], s=6, color=colour, linewidths=0, alpha=0.7, label=label)
    ax.axvspan(-LOG_MIN_NORMAL, -LOG_MIN_POSITIVE, color="#e67e22", alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title(title, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
for ax in grid[:, 0]: ax.set_ylabel("GALE minus AleRax (nats)")
for ax in grid[1, :]: ax.set_xlabel("$-$log-likelihood of the family (nats)")
axes[0].legend(title="log-likelihood", fontsize=7.5, title_fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(out); print("written", out)
