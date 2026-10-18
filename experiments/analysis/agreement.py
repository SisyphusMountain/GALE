"""Per-family log-likelihoods of gpurec against AleRax, each at the optimum of its own fit of the full set (each program
scores its own rates): median, mean, total and extreme differences (gpurec minus AleRax, nats), the families where one
is higher by more than 1 nat, and the correlation of the two sets of log-likelihoods (Pearson).
usage: agreement.py GPUREC_FILE ALERAX_FILE
Each file has one line per family: name, log-likelihood in nats (per_fam_likelihoods.txt of each program)."""
import math
import statistics
import sys


def read(p):
    d = {}
    for line in open(p):
        a = line.split()
        if len(a) >= 2: d[a[0]] = float(a[1])
    return d


g, a = read(sys.argv[1]), read(sys.argv[2])
keys = sorted(set(g) & set(a))
print(f"matched {len(keys)} of {len(g)} gpurec / {len(a)} AleRax families")
d = [g[k] - a[k] for k in keys]
n = len(d)
mg = sum(g[k] for k in keys) / n; ma = sum(a[k] for k in keys) / n
cov = sum((g[k]-mg)*(a[k]-ma) for k in keys); vg = sum((g[k]-mg)**2 for k in keys); va = sum((a[k]-ma)**2 for k in keys)
print(f"median {statistics.median(d):+.4f}  mean {sum(d)/n:+.4f}  total {sum(d):+.1f}  max {max(d):+.1f}  min {min(d):+.1f}")
print(f"higher by >1 nat: {sum(x > 1 for x in d)}   lower by >1 nat: {sum(x < -1 for x in d)}   pearson {cov/math.sqrt(vg*va):.7f}")
