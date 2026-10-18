"""Per-family log-likelihood (nats) of gpurec against AleRax at the same fixed rates: differences gpurec minus AleRax,
their largest and median size, how many exceed each threshold and in which direction, and the total. Then the same by
range of the family's log-likelihood (gpurec's value), which decides how AleRax stores it: above log(2.2e-308) = -708.40
AleRax computes with plain doubles; between log(4.9e-324) = -744.44 and -708.40 a plain double holds the likelihood
with few correct digits (AleRax as released keeps it, the switch to scaled values of patch 04 does not); below -744.44
AleRax computes with scaled values (a double and a count of factors 2^-256).
usage: compare_fixedrate.py GPUREC_FILE ALERAX_FILE  (both: "name log-likelihood" per line)"""
import math
import sys

import numpy as np

# Logarithms of the smallest normal double (2.2e-308) and of the smallest positive double (4.9e-324): facts of the format.
LOG_MIN_NORMAL = math.log(2.2250738585072014e-308)   # -708.40
LOG_MIN_POSITIVE = math.log(4.9406564584124654e-324)  # -744.44


def read(p): return {l.split()[0]: float(l.split()[1]) for l in open(p) if l.strip()}


g, a = read(sys.argv[1]), read(sys.argv[2])
names = sorted(set(g) & set(a)); assert len(names) == len(g) == len(a), (len(g), len(a), len(names))
d = np.array([g[n] - a[n] for n in names]); ll = np.array([a[n] for n in names])
print(f"{len(d)} families: max |diff| {np.abs(d).max():.3e}, median diff {np.median(d):.2e}, median |diff| {np.median(np.abs(d)):.2e}, total diff {d.sum():+.4f}")
for t in (1e-6, 1e-3, 1e-2, 1e-1, 1.0): print(f"  |diff| > {t:g}: {(np.abs(d)>t).sum()}  (gpurec higher {(d>t).sum()}, lower {(d<-t).sum()})")
i = np.abs(d).argmax(); print(f"  largest: {names[i]} AleRax {a[names[i]]:.4f} diff {d[i]:+.4f}")
gl = np.array([g[n] for n in names])
bands = (("above -708.40 (plain doubles)", gl >= LOG_MIN_NORMAL),
         ("-744.44 to -708.40", (gl >= LOG_MIN_POSITIVE) & (gl < LOG_MIN_NORMAL)),
         ("below -708.40 (scaled values once switched)", gl < LOG_MIN_NORMAL))
for label, sel in bands:
    if sel.any():
        x = d[sel]
        print(f"  log-likelihood {label}: {sel.sum()} families, max |diff| {np.abs(x).max():.3e}, largest gpurec higher {x.max():+.2e}, "
              f"gpurec lower by more than 1e-4: {(x < -1e-4).sum()}, total {x.sum():+.4f}")
