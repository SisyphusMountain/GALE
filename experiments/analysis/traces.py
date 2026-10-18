"""Summary of the traces written by wrappers/gpurec_trace.py: for each trace, the settings (warm start, removal of
converged families), the total time, the peak GPU memory allocated, the number of loss-and-gradient evaluations, and the
model builds (time of each build and number of families in it: the first build covers all families, each later one
the families still in the fit).
usage: traces.py TRACE_JSON..."""
import json
import os
import sys

for path in sys.argv[1:]:
    t = json.load(open(path))
    builds = [e for e in t["events"] if e["event"] == "build"]
    grads = [e for e in t["events"] if e["event"] == "grad"]
    print(f"{os.path.basename(path)}: warm {t['warm']}, removal {t['rebatch']}: {t['total_s']:.1f} s, peak allocated "
          f"{t['peak_alloc_gib']:.2f} GiB, {len(grads)} loss-and-gradient evaluations")
    print("   builds (time s: families): " + ", ".join(f"{e['t']:.0f}: {e['families']}" for e in builds))
