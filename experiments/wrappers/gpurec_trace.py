"""The gpurec command line with two settings of the per-family fit switched on or off, a trace of the fit and the peak
GPU memory. Used for the warm start of the adjoint and for the adaptive rebatching (removal of converged families).
usage: gpurec_trace.py WARM REBATCH TRACE_JSON <gpurec arguments...>
  WARM     on | off   adjoint warm start of the fit (on = untouched: used whenever the memory check allows it;
                      off = the fit receives warm_adjoint=False)
  REBATCH  on | off   removal of converged families from the batches (off: the fit receives drop_frac=1.1, a fraction
                      above 1, so that no family is ever removed)
  TRACE_JSON          file receiving the events: each model build (time, families in the model, batches) and each
                      loss-and-gradient call (time, families), with the total time and the peak GPU memory
"""
import atexit
import json
import sys
import time

WARM, REBATCH, TRACE = sys.argv[1:4]
assert WARM in ("on", "off") and REBATCH in ("on", "off")
sys.argv = ["gpurec"] + sys.argv[4:]
import torch
import gpurec.data as gdata
import gpurec.interface as interface
from gpurec.api.model import GeneReconModel

t0 = time.perf_counter(); events = []
orig_fit = interface.fit_genewise


def fit(*args, **kwargs):
    if WARM == "off": kwargs["warm_adjoint"] = False
    if REBATCH == "off": kwargs["drop_frac"] = 1.1
    return orig_fit(*args, **kwargs)


interface.fit_genewise = fit
orig_build = gdata.FamilyData.build_model


def build(self, indices, *a, **k):
    m = orig_build(self, indices, *a, **k)
    events.append({"t": time.perf_counter() - t0, "event": "build", "families": len(indices), "batches": len(m.family_batches)})
    return m


gdata.FamilyData.build_model = build
orig_grad = GeneReconModel.genewise_loss_vector_and_grad


def grad(self, *a, **k):
    out = orig_grad(self, *a, **k)
    if k.get("need_grad", True): events.append({"t": time.perf_counter() - t0, "event": "grad", "families": int(self.theta.shape[0])})
    return out


GeneReconModel.genewise_loss_vector_and_grad = grad


def finish():
    json.dump({"warm": WARM, "rebatch": REBATCH, "total_s": time.perf_counter() - t0,
               "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "events": events}, open(TRACE, "w"))
    sys.stderr.write(f"[trace warm={WARM} rebatch={REBATCH}] peak GPU memory allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, {len(events)} events\n")


atexit.register(finish)
from gpurec.cli.main import entry

entry()
