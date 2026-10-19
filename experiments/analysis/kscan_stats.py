"""Statistics of a scan of measure/kscan_per_family.py: for each number K of fixed-point iterations, the median, the 95th
and 99th percentiles and the largest size of the per-family difference of the log-likelihood (nats), against K = 256,
and, at fixed rates, also against AleRax 1.4.0 with the four changes (16 passes) at the same rates. Writes them to a
CSV file (read by figures/make_fig_kscan.py; the values of Figure "fig:kscan" and Table "tab:kscan") and prints them.
usage: kscan_stats.py fitted KSCAN_CSV OUT_CSV
       kscan_stats.py fixed KSCAN_CSV ALERAX_FILE OUT_CSV
ALERAX_FILE: "name log-likelihood" per line, for instance OUTPUT_ROOT/fixed_rates/out_four_16/per_fam_likelihoods.txt
(run_fixed_rates.py alerax-1042, the rates (0.2, 0.3, 0.1)). OUT_CSV: columns K, median, p95, p99, max for "fitted";
K, ref (K256 or AleRax), median, p95, p99, max for "fixed"."""
import csv
import sys

import numpy as np

MODE, SCAN = sys.argv[1:3]
if MODE == "fitted":
    OUT = sys.argv[3]
elif MODE == "fixed":
    ALERAX, OUT = sys.argv[3:5]
else:
    raise SystemExit(f"the first argument is 'fitted' or 'fixed', not '{MODE}'")

rows = list(csv.reader(open(SCAN)))
ks = [int(h[1:]) for h in rows[0][1:]]
names = [r[0] for r in rows[1:]]
values = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
references = [("K256", values[:, ks.index(256)])]
if MODE == "fixed":
    alerax = {line.split()[0]: float(line.split()[1]) for line in open(ALERAX) if line.strip()}
    references.append(("AleRax", np.array([alerax[n] for n in names])))


def summary(d):
    return [f"{x:.3e}" for x in (np.median(d), np.percentile(d, 95), np.percentile(d, 99), d.max())]


if MODE == "fitted":
    out = [["K", "median", "p95", "p99", "max"]]
    out += [[k] + summary(np.abs(values[:, j] - references[0][1])) for j, k in enumerate(ks)]
else:
    out = [["K", "ref", "median", "p95", "p99", "max"]]
    out += [[k, label] + summary(np.abs(values[:, j] - ref)) for j, k in enumerate(ks) for label, ref in references]
with open(OUT, "w", newline="") as f:
    csv.writer(f).writerows(out)
print(f"{len(names)} families, |difference| of the log-likelihood in nats, written to {OUT}")
for r in out:
    print(*r)
