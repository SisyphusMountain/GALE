"""The gpurec command line with one design choice switched off, for the ablations. The plan of batches that the model
receives and the peak GPU memory are printed to standard error.
usage: gpurec_ablate.py MODE <gpurec arguments...>
MODE:
  base        nothing changed
  nowarm      the fit receives warm_adjoint=False: every adjoint solve starts from zero instead of from the adjoint of
              the previous step
  noevict     the fit receives drop_frac=1.1: converged families are never removed from the batches
  bysize      batches filled by decreasing clade count ('clade_first_fit') instead of by depth ('depth_first_fit')
  inputorder  batches filled in input order ('sequential') instead of by depth
  alone       one family per batch, so that the waves never mix families
  storealpha  the forward pass keeps the split term (the first line of the recursion, written alpha in the code) of
              every wave of the current batch, and the backward pass reads it instead of recomputing it from the
              stored rows
Adjoint pruning is switched off with a configuration file (data/configs/no_pruning.toml, option --config), not here."""
import atexit
import sys

MODE = sys.argv[1]
assert MODE in ("base", "nowarm", "noevict", "bysize", "inputorder", "alone", "storealpha"), MODE
sys.argv = ["gpurec"] + sys.argv[2:]
import torch
import gpurec.api.model as model
import gpurec.data as data
import gpurec.interface as interface

if MODE in ("nowarm", "noevict"):
    original = interface.fit_genewise
    override = {"nowarm": {"warm_adjoint": False}, "noevict": {"drop_frac": 1.1}}[MODE]
    interface.fit_genewise = lambda *a, **k: original(*a, **{**k, **override})
if MODE == "bysize": data.DEFAULT_BATCH_PACKING = "clade_first_fit"
if MODE == "inputorder": data.DEFAULT_BATCH_PACKING = "sequential"
if MODE == "alone": data.DEFAULT_FAMILY_CHUNK_SIZE = 1
if MODE == "storealpha":
    import gpurec.api._implicit_grad as backward_module
    import gpurec.core.inference.forward as forward_module
    stored, current = {}, {"batch": None}
    forward_dts = forward_module.compute_dts_forward

    def forward_and_store(*a, **k):
        out = forward_dts(*a, **k)
        batch = k["family_idx"].data_ptr()
        if current["batch"] != batch:          # one batch at a time, as an implementation that stores alpha would
            stored.clear(); current["batch"] = batch
        stored[a[4].data_ptr()] = out          # key: this wave's split-left index tensor (meta["sl"])
        return out

    def read_stored(*a, **k):
        if current["batch"] != k["family_idx"].data_ptr() or a[4].data_ptr() not in stored:
            raise RuntimeError("storealpha: no stored split term for this wave")
        return stored[a[4].data_ptr()]

    forward_module.compute_dts_forward = forward_and_store
    backward_module.compute_dts_forward = read_stored
plan_original = model.plan_batch_wave_layouts


def plan_logged(families, **k):
    raw = plan_original(families, **k)
    sys.stderr.write(f"[ablate {MODE}] plan: {len(families)} families -> {len(raw['batches'])} batches "
                     f"(family_chunk_size={k['family_chunk_size']}, batch_packing={k['batch_packing']}, clade_budget={k['clade_budget']})\n")
    return raw


model.plan_batch_wave_layouts = plan_logged
atexit.register(lambda: sys.stderr.write(f"[ablate {MODE}] peak GPU memory allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB\n"))
from gpurec.cli.main import entry

entry()
