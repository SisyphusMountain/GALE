"""Figure "fig_scaling": wall time of the whole command (reading included) against the number of HOGENOM families
fitted. 1, 3, 10 and 30 families: 20 independent random sets each (run_scaling.py part small-draws; gpurec's timed
second run of each set, with a warm compile cache; AleRax 1.4.0 with one MPI rank per family below 24 families, 24 for
30), median with a bar from the fastest to the slowest set. 100, 300, 1,000 and 3,000 families: nested random subsets
(parts gpurec-nested-large and alerax-nested; gpurec mean of two runs, the runs as small points; AleRax one run).
Full set: the mean of gpurec's two full runs ("done in" of the log) and the sum of AleRax's eight chunks (run_hogenom.py
parts gpurec-full and alerax-full). Also prints the numbers of the paragraph "Scaling".
Colours: gpurec #1f5fa8, AleRax #eb6834.
usage: make_fig_scaling.py PATHS_TOML OUT_PDF"""
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FAMILY_LISTS, output_dir, read_paths, wall_seconds

paths, out = read_paths(sys.argv[1]), sys.argv[2]
S, H = output_dir(paths, "scaling"), output_dir(paths, "hogenom")
DRAWS = output_dir(paths, "scaling/small_draws")
GPUREC, ALERAX = "#1f5fa8", "#eb6834"
INK, MUTED = "#222222", "#666666"
SIZES_DRAWS = [1, 3, 10, 30]
SIZES_NESTED = [100, 300, 1000, 3000]
N_FULL = 10869


def done_in(name):
    return float(re.findall(r"done in ([0-9.]+) s", (H / f"{name}.err").read_text())[-1])


draws_g, draws_a = {}, {}
for n in SIZES_DRAWS:
    sets = sorted(f.stem for f in (FAMILY_LISTS / "small_draws").glob(f"n{n}_d*.txt"))
    draws_g[n] = [wall_seconds(DRAWS / f"g_{s}_timed.time") for s in sets]
    draws_a[n] = [wall_seconds(DRAWS / f"a_{s}.time") for s in sets]
    assert len(draws_g[n]) == len(draws_a[n]) > 0, (n, len(draws_g[n]), len(draws_a[n]))
g_runs = {n: [wall_seconds(S / f"g_nested{n}_{k}.time") for k in (1, 2)] for n in SIZES_NESTED}
a_runs = {n: wall_seconds(S / f"a_nested{n}.time") for n in SIZES_NESTED}
g_full = st.mean([done_in("full_1"), done_in("full_2")])
a_full = sum(float(line.split()[2]) for line in open(H / "alerax_full_timing.txt") if line.strip())

fig, ax = plt.subplots(figsize=(5.6, 3.6))
xs = SIZES_DRAWS + SIZES_NESTED + [N_FULL]
g_line = [st.median(draws_g[n]) for n in SIZES_DRAWS] + [st.mean(g_runs[n]) for n in SIZES_NESTED] + [g_full]
a_line = [st.median(draws_a[n]) for n in SIZES_DRAWS] + [a_runs[n] for n in SIZES_NESTED] + [a_full]
ax.plot(xs, a_line, color=ALERAX, lw=2, marker="o", ms=6, zorder=3)
ax.plot(xs, g_line, color=GPUREC, lw=2, marker="o", ms=6, zorder=3)
for n in SIZES_DRAWS:
    for vals, colour, shift in ((draws_g[n], GPUREC, 0.94), (draws_a[n], ALERAX, 1.06)):
        ax.plot([n * shift, n * shift], [min(vals), max(vals)], color=colour, lw=1.2, alpha=0.6, zorder=2)
for n in SIZES_NESTED:
    ax.scatter([n, n], g_runs[n], s=10, color=GPUREC, zorder=4, linewidths=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("families fitted", color=INK)
ax.set_ylabel("wall time of the command (s)", color=INK)
ax.text(N_FULL * 1.12, a_full, "AleRax 1.4.0\n24 cores", color=INK, va="center", fontsize=8.5)
ax.text(N_FULL * 1.12, g_full, "gpurec\nRTX 4090", color=INK, va="center", fontsize=8.5)
ax.set_xlim(0.7, N_FULL * 4.5)
ax.spines[["top", "right"]].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
fig.tight_layout()
fig.savefig(out)
print("written", out)
for n in SIZES_DRAWS:
    r = [a / g for a, g in zip(draws_a[n], draws_g[n])]
    k = len(r)
    print(f"{n:5d} families, {k} draws: gpurec median {st.median(draws_g[n]):.2f} s [{min(draws_g[n]):.2f}-{max(draws_g[n]):.2f}], "
          f"AleRax median {st.median(draws_a[n]):.2f} s [{min(draws_a[n]):.2f}-{max(draws_a[n]):.2f}], ratio median {st.median(r):.2f} [{min(r):.2f}-{max(r):.2f}], AleRax faster in {sum(x < 1 for x in r)}/{k}")
print("nested, gpurec runs:", {n: [round(x, 1) for x in g_runs[n]] for n in SIZES_NESTED}, "AleRax:", {n: round(a_runs[n], 1) for n in SIZES_NESTED})
print("nested, ratios:", {n: round(a_runs[n] / st.mean(g_runs[n]), 1) for n in SIZES_NESTED}, "full", round(a_full / g_full, 1))
