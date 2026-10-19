"""The kernels of a profiled gpurec fit, from the SQLite database that run_profiles.py gpurec-nsys exports from NVIDIA
Nsight Systems (full.sqlite) and gpurec's log (full.err): the time during which the GPU runs kernels (the sum of the
durations of all kernel launches) and its share of the run ("done in" of the log), the number of kernel launches, then
for the kernels that take the most time: their share of the kernel time, their number of launches and the median
duration of one launch (Table "tab:kernels" of the paper, and the macros \\prBusy and \\prLaunches).
usage: nsys_kernels.py PROFILE_SQLITE GPUREC_ERR"""
import re
import sqlite3
import sys

import numpy as np

# The paper's table lists the five kernels that take the most time; not a setting of the measurement.
ROWS = 5

SQLITE, ERR = sys.argv[1:3]
run = float(re.findall(r"done in ([0-9.]+) s", open(ERR).read())[-1])
db = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
# One row per kernel, grouped by its full name as Nsight Systems' own summary (cuda_gpu_kern_sum) groups them.
kernels = db.execute("SELECT s.value, k.demangledName, COUNT(*), SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                     "JOIN StringIds s ON s.id = k.demangledName GROUP BY k.demangledName "
                     "ORDER BY SUM(k.end - k.start) DESC").fetchall()
busy_ns = sum(row[3] for row in kernels)
launches = sum(row[2] for row in kernels)
print(f"run: {run:.1f} s (done in)")
print(f"GPU running kernels: {busy_ns / 1e9:.1f} s, {100 * busy_ns / run / 1e9:.1f} % of the run")
print(f"kernel launches: {launches:,}")
print(f"the {ROWS} kernels that take the most time:")
print(f"  {'share of kernel time':>20s} {'launches':>10s} {'median duration':>16s}  kernel")
for name, name_id, count, total_ns in kernels[:ROWS]:
    durations = np.array([d for (d,) in db.execute(
        "SELECT end - start FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName = ?", (name_id,))], dtype=np.int64)
    median_us = float(np.median(durations)) / 1e3
    median = f"{median_us / 1e3:.3g} ms" if median_us >= 1e3 else f"{median_us:.3g} us"
    print(f"  {100 * total_ns / busy_ns:19.1f} % {count:10,d} {median:>16s}  {name}")
