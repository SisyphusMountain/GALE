"""Figure "fig_rebatch". (a) Families still in the fit against time, with and without removal of converged families; (b)
cumulative number of batch evaluations (loss and gradient calls) against time. Reads two traces written by
wrappers/gpurec_trace.py (run_warm_rebatch.py part rebatch-1042: OUTPUT_ROOT/warm_rebatch/trace_removal_on_N.json and
trace_removal_off.json; the paper drew a run with removal that took 231 s).
usage: make_fig_rebatch.py TRACE_ON TRACE_OFF OUT_PDF"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
on, off, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
def active(trace):
    ts, ns = [0.0], [None]
    for e in trace["events"]:
        if e["event"] == "build": ts.append(e["t"]); ns.append(e["families"])
    ns[0] = ns[1]
    ts.append(trace["total_s"]); ns.append(ns[-1])
    return ts, ns
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
for tr, lab, col in ((on, "converged families removed (release)", "#1f5fa8"), (off, "no removal", "#c0392b")):
    ts, ns = active(tr); a.step(ts, ns, where="post", label=f"{lab}: {tr['total_s']:.0f} s", color=col)
    g = [e["t"] for e in tr["events"] if e["event"] == "grad"]; b.plot(g, range(1, len(g) + 1), label=lab, color=col)
a.set_xlabel("time (s)"); a.set_ylabel("families still in the fit"); a.set_title("(a)", fontsize=10); a.legend(fontsize=8, frameon=False)
b.set_xlabel("time (s)"); b.set_ylabel("loss-and-gradient evaluations so far"); b.set_title("(b)", fontsize=10)
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(out); print("written", out)
