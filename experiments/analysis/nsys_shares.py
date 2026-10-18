"""Where the time of a profiled gpurec fit goes, from two reports of NVIDIA Nsight Systems (nsys stats, CSV format) and
gpurec's log: for each NVTX range of wrappers/gpurec_nvtx.py (curvature "hessian", "gradient", "model build"), its time
as a share of the run ("done in" of the log): the GPU time of the work launched inside the range (projected time) for
the curvature and the gradient, the time inside the range for the model builds (mostly CPU work); then the share of the
run during which the GPU executes kernels, and the share of kernel time of the largest kernels.
usage: nsys_shares.py NVTX_GPU_PROJ_SUM_CSV CUDA_GPU_KERN_SUM_CSV GPUREC_ERR"""
import csv
import re
import sys

PROJ, KERN, ERR = sys.argv[1:4]
run = float(re.findall(r"done in ([0-9.]+) s", open(ERR).read())[-1])
ranges = {row["Range"].lstrip(":"): row for row in csv.DictReader(open(PROJ))}
print(f"run: {run:.1f} s")
for name, column in (("hessian", "Total Proj Time (ns)"), ("gradient", "Total Proj Time (ns)"), ("model build", "Total Range Time (ns)")):
    r = ranges[name]
    seconds = float(r[column]) / 1e9
    print(f"  {name:12s} {seconds:8.1f} s ({column}), {int(r['Range Instances'])} calls, {100 * seconds / run:.1f} % of the run")
kernels = list(csv.DictReader(open(KERN)))
busy = sum(float(k["Total Time (ns)"]) for k in kernels) / 1e9
print(f"  GPU kernels  {busy:8.1f} s, {100 * busy / run:.1f} % of the run")
for k in kernels[:5]:
    print(f"    {float(k['Time (%)']):5.1f} % of kernel time  {k['Name']}")
