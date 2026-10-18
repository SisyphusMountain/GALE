"""Figure "fig_times_gpu": (a) wall time of the fit of the full HOGENOM set, AleRax on 24 CPU cores (the sum of its eight
chunks) against gpurec on one GPU (the mean of the two runs full_1 and full_2, as gpurec reports it: "done in");
(b) where the time of a gpurec run goes, as shares of the profiled run of run_profiles.py part gpurec-nsys: the
likelihood (forward pass) and the gradient (backward pass) of the loss-and-gradient evaluations, the second derivatives,
the batch builds and the rest (analysis/split_gradient_ranges.py on profiles/full.sqlite, total "done in" of
profiles/full.err).
Reads OUTPUT_ROOT/hogenom (run_hogenom.py parts alerax-full and gpurec-full) and OUTPUT_ROOT/profiles (run_profiles.py
part gpurec-nsys). About 15 s, most of it reading the profile.
usage: make_fig_times.py PATHS_TOML OUT_PDF"""
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import output_dir, read_paths
from split_gradient_ranges import done_in, split

paths, out = read_paths(sys.argv[1]), sys.argv[2]
H, PROFILES = output_dir(paths, "hogenom"), output_dir(paths, "profiles")

alerax = sum(float(line.split()[2]) for line in open(H / "alerax_full_timing.txt") if line.strip())
configured = statistics.mean([done_in(H / "full_1.err"), done_in(H / "full_2.err")])
shares = split(PROFILES / "full.sqlite")
total = done_in(PROFILES / "full.err")
lik, back, hess, build = shares["forward"], shares["backward"], shares["hessian"], shares["model build"]
other = total - lik - back - hess - build

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
labels = ["AleRax\n24 CPU cores", "GALE\none RTX 4090"]
vals = [alerax, configured]
cols = ["#8a8a8a", "#1f5fa8"]
bars = a.bar(labels, vals, color=cols, width=0.62)
a.set_yscale("log"); a.set_ylabel("wall time (s), log scale"); a.set_ylim(300, 40000)
for rect, v in zip(bars, vals):
    a.text(rect.get_x() + rect.get_width() / 2, v * 1.12, f"{v:,.0f} s", ha="center", va="bottom", fontsize=9)
a.set_title("(a) full HOGENOM, 10,869 families", fontsize=10)
a.spines[["top", "right"]].set_visible(False)

parts = [("likelihood (forward pass)", lik, "#1f5fa8"), ("gradient (backward pass)", back, "#7aa6d6"),
         ("second derivatives", hess, "#d98c3a"), ("batch builds", build, "#6aa36a"), ("other", other, "#bdbdbd")]
left = 0
for name, v, c in parts:
    share = 100 * v / total
    b.barh([0], [share], left=left, color=c, label=f"{name}: {share:.0f} %", height=0.5, edgecolor="white", linewidth=1)
    left += share
b.set_yticks([]); b.set_xlabel("share of the GALE run (%)"); b.set_xlim(0, 100)
b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
b.set_title("(b) where the time of GALE goes", fontsize=10)
b.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig(out)
print("written", out)
print(f"AleRax {alerax:.0f} s, gpurec {configured:.1f} s (mean of full_1 and full_2); profiled run {total:.1f} s:",
      ", ".join(f"{name} {v:.1f} s ({100 * v / total:.1f} %)" for name, v, _ in parts))
