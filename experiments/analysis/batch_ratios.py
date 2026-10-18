"""Ratio of the AleRax time to the gpurec time (wall time of the whole command) on each of the 22 random batches of 500
families that partition the full set (run_scaling.py parts gpurec-batches and alerax-batches): per batch, and their
median, smallest and largest.
usage: batch_ratios.py SCALING_FOLDER   (OUTPUT_ROOT/scaling)"""
import statistics as st
import sys


def wall(path):
    for line in open(path):
        if "Elapsed (wall clock)" in line:
            parts = [float(x) for x in line.split()[-1].split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    raise ValueError(f"no wall time in {path}")


D = sys.argv[1]
ratios = []
for i in range(1, 23):
    g, a = wall(f"{D}/g_b{i:02d}.time"), wall(f"{D}/a_b{i:02d}.time")
    ratios.append(a / g)
    print(f"batch {i:02d}: gpurec {g:7.1f} s, AleRax {a:8.1f} s, ratio {a / g:5.1f}")
print(f"median ratio {st.median(ratios):.1f}, smallest {min(ratios):.1f}, largest {max(ratios):.1f}")
