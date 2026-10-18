"""Compare the optima that gpurec and AleRax reach when each fits its own rates, both scored by the same evaluation
(measure/eval_at_rates.py, double precision, converged fixed points), so that the comparison is between the rates each
program finds. Differences are gpurec minus AleRax, in nats (natural-logarithm units of likelihood).
Prints the total difference; the families where gpurec is higher, lower, or within 0.001 nats; the largest
differences in each direction; the families where AleRax's rates are still its starting values (0.1, 0.1, 0.1), and
their share of the total; and, for the families where gpurec is lower by more than 0.001 nats, how many have an AleRax
rate at AleRax's lower bound (1e-10).
usage: compare_optima.py GPUREC_SCORES ALERAX_SCORES ALERAX_RATES
GPUREC_SCORES, ALERAX_SCORES: "family log-likelihood" per line (eval_at_rates.py at each program's rates).
ALERAX_RATES: "family D L T" per line, AleRax's fitted rates."""
import sys

# AleRax 1.4.0's starting rates and lower bound on a rate: facts of AleRax, not settings.
ALERAX_START = (0.1, 0.1, 0.1)
ALERAX_LOWER_BOUND = 1e-10


def read(path):
    return {line.split()[0]: float(line.split()[1]) for line in open(path) if line.strip()}


g, a = read(sys.argv[1]), read(sys.argv[2])
rates = {line.split()[0]: [float(x) for x in line.split()[1:4]] for line in open(sys.argv[3]) if line.strip()}
names = sorted(g)
assert set(names) == set(a) == set(rates), "the three files name different families"
d = {k: g[k] - a[k] for k in names}
print(f"{len(names)} families: total gpurec minus AleRax {sum(d.values()):+.4f} nats")
for t in (1e-3, 1e-2, 1e-1, 1.0):
    print(f"  gpurec higher by more than {t:g}: {sum(v > t for v in d.values())};  lower by more than {t:g}: {sum(v < -t for v in d.values())}")
print(f"  within 0.001 nats: {sum(abs(v) <= 1e-3 for v in d.values())}")
hi, lo = max(names, key=d.get), min(names, key=d.get)
print(f"  largest gpurec higher: {hi} {d[hi]:+.4f};  largest AleRax higher: {lo} {d[lo]:+.4f}")
stuck = [k for k in names if tuple(rates[k]) == ALERAX_START]
print(f"  AleRax rates still at its start {ALERAX_START}: {len(stuck)} families, their total {sum(d[k] for k in stuck):+.4f} nats")
low = [k for k in names if d[k] < -1e-3]
at_bound = [k for k in low if min(rates[k]) <= ALERAX_LOWER_BOUND * (1 + 1e-6)]
print(f"  gpurec lower by more than 0.001: {len(low)} families, {len(at_bound)} of them with an AleRax rate at {ALERAX_LOWER_BOUND:g}")
for k in sorted(low, key=d.get)[:5]:
    print(f"    {k}: {d[k]:+.4f} nats; gpurec {g[k]:.4f}, AleRax rates {rates[k]}")
