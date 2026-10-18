"""Figure "fig_times_gpu": (a) wall time of the fit of the full HOGENOM set, AleRax on 24 CPU cores (the sum of its eight
chunks) against gpurec on one GPU (the mean of the two runs full_1 and full_2, as gpurec reports it: "done in");
(b) where the time of the timed run full_timers goes (gradient, curvature, batch builds, the rest).
Reads OUTPUT_ROOT/hogenom (run_hogenom.py parts alerax-full, gpurec-full and gpurec-full-timers).
usage: make_fig_times.py PATHS_TOML OUT_PDF"""
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import output_dir, read_paths

paths, out = read_paths(sys.argv[1]), sys.argv[2]
D = output_dir(paths, "hogenom")


def done_in(name):
    return float(re.findall(r"done in ([0-9.]+) s", (D / f"{name}.err").read_text())[-1])


def timer_seconds(label):
    return float(re.search(re.escape(label) + r"\s+calls=\s*\d+\s+seconds=\s*([0-9.]+)", (D / "full_timers.log").read_text()).group(1))


alerax = sum(float(line.split()[2]) for line in open(D / "alerax_full_timing.txt") if line.strip())
configured = statistics.mean([done_in("full_1"), done_in("full_2")])
instr = done_in("full_timers")
grad = timer_seconds("gradient (genewise_loss_vector_and_grad)")
hess = timer_seconds("Hessian (_analytic_hessian)")
build = timer_seconds("model build")
other = instr - grad - hess - build

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
labels = ["AleRax\n24 CPU cores", "gpurec\none RTX 4090"]
vals = [alerax, configured]
cols = ["#8a8a8a", "#1f5fa8"]
bars = a.bar(labels, vals, color=cols, width=0.62)
a.set_yscale("log"); a.set_ylabel("wall time (s), log scale"); a.set_ylim(300, 40000)
for rect, v in zip(bars, vals):
    a.text(rect.get_x() + rect.get_width() / 2, v * 1.12, f"{v:,.0f} s", ha="center", va="bottom", fontsize=9)
a.set_title("(a) full HOGENOM, 10,869 families", fontsize=10)
a.spines[["top", "right"]].set_visible(False)

parts = [("gradient", grad, "#1f5fa8"), ("curvature (Hessian)", hess, "#d98c3a"), ("model builds", build, "#6aa36a"), ("other", other, "#bdbdbd")]
left = 0
for name, v, c in parts:
    b.barh([0], [v], left=left, color=c, label=f"{name}: {v:,.0f} s", height=0.5)
    left += v
b.set_yticks([]); b.set_xlabel("seconds"); b.set_xlim(0, instr * 1.02)
b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
b.set_title(f"(b) where {instr:,.0f} s go (timed run)", fontsize=10)
b.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig(out)
print("written", out, alerax, configured, instr)
