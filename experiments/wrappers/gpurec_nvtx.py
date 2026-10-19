"""The gpurec command line, unchanged, with NVTX ranges (named intervals that NVIDIA's profiler Nsight Systems records)
around the three stages that take most of the time of a per-family fit: the gradient (genewise_loss_vector_and_grad),
the curvature (_analytic_hessian, the 3x3 Hessian of each family) and the batch builds (FamilyData.build_model). No CUDA
synchronisation is added: Nsight Systems projects each range onto the time the GPU spent on the work launched inside it.
usage: gpurec_nvtx.py <gpurec arguments...>   (run under nsys profile --trace=cuda,nvtx)"""
import importlib
import sys

sys.argv = ["gale"] + sys.argv[1:]
import torch
from gpurec import GeneReconModel


def ranged(label, fn):
    def wrapper(*a, **k):
        torch.cuda.nvtx.range_push(label)
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.nvtx.range_pop()
    return wrapper


def patch(obj, name, label):
    setattr(obj, name, ranged(label, getattr(obj, name)))


gf = importlib.import_module("gpurec.fit.genewise_fit")
gd = importlib.import_module("gpurec.data")
patch(GeneReconModel, "genewise_loss_vector_and_grad", "gradient")
patch(gf, "_analytic_hessian", "hessian")
patch(gd.FamilyData, "build_model", "model build")
from gpurec.cli.main import entry

entry()
