"""The gpurec command line, unchanged, with timers around the three stages of a per-family fit: the gradient
(genewise_loss_vector_and_grad), the curvature (_analytic_hessian, the 3x3 Hessian of each family) and the batch builds
(FamilyData.build_model); it also counts the calls of torch.linalg.eigh (one per refresh of the curvature). Each timer
waits for the GPU (torch.cuda.synchronize) before and after the call, so its seconds are real GPU time; these waits
lengthen the run. No gpurec code is changed: the functions are wrapped at run time. At exit, a table of calls and seconds
is printed to standard output, then one line per call.
usage: gpurec_timers.py <gpurec arguments...>"""
import atexit
import collections
import importlib
import sys
import time

sys.argv = ["gale"] + sys.argv[1:]
import torch
from gpurec import GeneReconModel

_calls = collections.Counter()
_secs = collections.defaultdict(float)
_t_start = time.perf_counter()
_events = []


def timed(label, fn):
    def wrapper(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize(); dt = time.perf_counter() - t; _secs[label] += dt; _calls[label] += 1
            _events.append((label, dt, time.perf_counter() - _t_start))
    return wrapper


def patch(obj, name, label):
    setattr(obj, name, timed(label, getattr(obj, name)))


def count(label, fn):
    def wrapper(*a, **k):
        _calls[label] += 1
        return fn(*a, **k)
    return wrapper


@atexit.register
def report():
    wall = time.perf_counter() - _t_start
    print("\n=== INSTRUMENTATION (CUDA-synchronised) ===", flush=True)
    for label in sorted(_calls):
        s = _secs.get(label)
        print(f"  {label:<34} calls={_calls[label]:>6}" + (f"  seconds={s:>8.1f}  ({100*s/wall:4.1f}% of wall)" if s is not None else ""), flush=True)
    print(f"  wall since start of process: {wall:.1f} s", flush=True)
    print("  per-call log (label, seconds, wall time at end):", flush=True)
    for label, dt, tw in _events:
        if label != "model build":
            print(f"    CALL {label.split(' ')[0]:<9} {dt:7.2f} s  at {tw:7.1f} s", flush=True)


gf = importlib.import_module("gpurec.fit.genewise_fit")
gd = importlib.import_module("gpurec.data")
patch(GeneReconModel, "genewise_loss_vector_and_grad", "gradient (genewise_loss_vector_and_grad)")
patch(gf, "_analytic_hessian", "Hessian (_analytic_hessian)")
patch(gd.FamilyData, "build_model", "model build")
torch.linalg.eigh = count("Hessian refreshes (eigh calls)", torch.linalg.eigh)
from gpurec.cli.main import entry

entry()
