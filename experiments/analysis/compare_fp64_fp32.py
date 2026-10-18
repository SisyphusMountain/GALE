"""Double- against single-precision fits of the same families (gpurec, per-family rates). Reads two single-precision
fits of all the families (SINGLE_1, SINGLE_2) and a double-precision fit made in two halves (DOUBLE_HALF_1,
DOUBLE_HALF_2), each an output folder of gpurec with per_fam_likelihoods.txt and model_parameters/FAMILY_rates.txt.
Reports, per family: the log-likelihood each fit reports, double minus single, against single run 1 minus single run 2
(the run-to-run spread of single precision), and the fitted rates compared as base-2 logarithms of their ratio.
usage: compare_fp64_fp32.py SINGLE_1 SINGLE_2 DOUBLE_HALF_1 DOUBLE_HALF_2"""
import math
import os
import statistics as st
import sys

S1, S2, H1, H2 = sys.argv[1:5]


def lls(run):
    return {l.split()[0]: float(l.split()[1]) for l in open(os.path.join(run, "per_fam_likelihoods.txt"))}


def rates(run):
    out = {}
    folder = os.path.join(run, "model_parameters")
    for f in os.listdir(folder):
        if f.endswith("_rates.txt"):
            out[f[:-len("_rates.txt")]] = [float(x) for x in open(os.path.join(folder, f)).read().split("\n")[1].split()]
    return out


s1, s2 = lls(S1), lls(S2)
d = {**lls(H1), **lls(H2)}
assert set(s1) == set(s2), (len(s1), len(s2))
assert set(d) == set(s1), f"double-precision fits cover {len(d)} families, single {len(s1)}"
r1, rd = rates(S1), {**rates(H1), **rates(H2)}


def summary(name, a, b):
    x = sorted(a[k] - b[k] for k in a)
    ax = sorted(abs(v) for v in x)
    print(f"{name}: total {sum(a.values()):.2f} vs {sum(b.values()):.2f}, difference {sum(x):+.3f}; per family median |diff| {st.median(ax):.2e}, "
          f"max {ax[-1]:.3g}; above 1e-3: {sum(v > 1e-3 for v in ax)}, above 0.01: {sum(v > 0.01 for v in ax)}, above 1: {sum(v > 1 for v in ax)}; "
          f"min {x[0]:+.4f}, max {x[-1]:+.4f}")


summary("single run 1 minus single run 2", s1, s2)
summary("double minus single run 1", d, s1)
summary("double minus single run 2", d, s2)

apart = sorted(k for k in s1 if max(s1[k], s2[k], d[k]) - min(s1[k], s2[k], d[k]) > 0.01)
print(f"families where two of the three fits differ by more than 0.01 nat: {len(apart)}")
for k in apart:
    print(f"   {k}: single run 1 {s1[k]:.4f}, single run 2 {s2[k]:.4f}, double {d[k]:.4f}")

for i, name in enumerate("DLT"):
    lr = sorted(abs(math.log2(rd[k][i] / r1[k][i])) for k in rd)
    print(f"rate {name}, double vs single run 1: median |log2 ratio| {st.median(lr):.1e}, families with a ratio above 2: {sum(v > 1 for v in lr)}")
