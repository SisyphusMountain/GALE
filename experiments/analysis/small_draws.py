"""Summarise the repeated random draws of run_scaling.py (part small-draws): for each size (1, 3, 10 and 30 families),
over its 20 random sets, the gpurec time (timed run, warm compile cache), the AleRax time and their ratio: median and
range, and on how many sets AleRax was faster; plus the largest number of kernels compiled during a timed run and the
agreement of the two programs' total log-likelihoods.
usage: small_draws.py SMALL_DRAWS_FOLDER   (OUTPUT_ROOT/scaling/small_draws)"""
import glob
import os
import re
import statistics as st
import sys

D = sys.argv[1]


def wall(p):
    for line in open(p):
        if "Elapsed (wall clock)" in line:
            parts = [float(x) for x in line.split()[-1].split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def total(d):
    p = f"{d}/per_fam_likelihoods.txt"
    return sum(float(l.split()[1]) for l in open(p)) if os.path.exists(p) else None


comp = {}
for line in open(f"{D}/compiles.txt"):
    m = re.match(r"(g_\S+) new_triton_cache_entries=(\d+)", line)
    if m: comp[m.group(1)] = int(m.group(2))
print(f"{'size':>5s} {'sets':>4s} {'gpurec s (median, range)':>28s} {'AleRax s (median, range)':>28s} {'AleRax/gpurec (median, range)':>32s} {'AleRax faster':>14s} {'compiles(timed) max':>20s}")
for n in (1, 3, 10, 30):
    g, a, r, c, dll = [], [], [], [], []
    for f in sorted(glob.glob(f"{D}/g_n{n}_d*_timed.time")):
        s = os.path.basename(f)[len("g_"):-len("_timed.time")]
        gt, at = f"{D}/g_{s}_timed.time", f"{D}/a_{s}.time"
        if not os.path.exists(at): continue
        g.append(wall(gt)); a.append(wall(at)); r.append(a[-1] / g[-1]); c.append(comp.get(f"g_{s}_timed", -1))
        tg, ta = total(f"{D}/out_g_{s}_timed"), total(f"{D}/out_a_{s}")
        if tg is not None and ta is not None: dll.append(tg - ta)
    if not r: continue
    print(f"{n:5d} {len(r):4d} {st.median(g):10.2f} [{min(g):6.2f}-{max(g):7.2f}] {st.median(a):10.2f} [{min(a):6.2f}-{max(a):7.2f}] {st.median(r):12.2f} [{min(r):5.2f}-{max(r):6.2f}] {sum(x < 1 for x in r):9d}/{len(r)} {max(c):12d}"
          f"   gpurec-AleRax total logL: median {st.median(dll):+.3f}, min {min(dll):+.3f}, max {max(dll):+.3f}")
