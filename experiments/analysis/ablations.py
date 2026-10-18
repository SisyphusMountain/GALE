"""Summarise the ablation runs of run_ablations.py: for every run, the wall time of the whole command, the number of
kernels Triton compiled during it (compiles.txt), the peak GPU memory and the total log-likelihood it reports; then, for
each configuration, the mean of its two timed runs (made after an untimed run that fills the compile cache) and its
ratio to the release settings ("base").
usage: ablations.py ABLATIONS_FOLDER   (OUTPUT_ROOT/ablations)"""
import glob
import os
import re
import statistics as st
import sys

D = sys.argv[1]
MODES = ("base", "alone", "noevict", "nowarm", "bysize", "inputorder", "base_again", "nopruning", "storealpha")


def wall(path):
    for line in open(path):
        if "Elapsed (wall clock)" in line:
            parts = [float(x) for x in line.split()[-1].split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


compiles = {}
if os.path.exists(f"{D}/compiles.txt"):
    for line in open(f"{D}/compiles.txt"):
        name, value = line.split(); compiles[name] = int(value.split("=")[1])


def total_ll(name):
    log = f"{D}/out_{name}/gpurec.log"
    if not os.path.exists(log): return None
    v = re.findall(r"total log-likelihood: (-?[0-9.]+)", open(log).read())
    return float(v[-1]) if v else None


def peak(name):
    v = re.findall(r"peak GPU memory allocated: ([0-9.]+) GiB", open(f"{D}/{name}.err").read()) if os.path.exists(f"{D}/{name}.err") else []
    return float(v[-1]) if v else None


rows = []
for t in sorted(glob.glob(f"{D}/*.time")):
    name = os.path.basename(t)[:-5]
    rows.append((name, wall(t), compiles.get(name), peak(name), total_ll(name)))
print(f"{'run':28s} {'wall s':>8s} {'compiles':>9s} {'peak GiB':>9s} {'total log-lik':>16s}")
for name, w, c, p, ll in rows:
    print(f"{name:28s} {w:8.1f} {str(c) if c is not None else '-':>9s} {('%.2f' % p) if p else '-':>9s} {('%.3f' % ll) if ll else '-':>16s}")
print("\nmean of the two timed runs (warm compile cache), relative to base:")
base = [w for n, w, *_ in rows if n in ("base_1", "base_2")]
if len(base) == 2:
    b = st.mean(base)
    for m in MODES:
        ws = [w for n, w, *_ in rows if n in (f"{m}_1", f"{m}_2")]
        if ws: print(f"  {m:11s} {st.mean(ws):8.1f} s  x{st.mean(ws)/b:5.2f}  runs {[round(x, 1) for x in ws]}")
