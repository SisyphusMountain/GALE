"""One line per run of the drivers: the wall time of the whole command (from NAME.time, written by /usr/bin/time -v),
and, from gpurec's log (NAME.err), the time gpurec reports ("done in"), the total log-likelihood, the number of families
kept and the peak GPU memory printed by the wrappers. For AleRax runs, only the wall time.
usage: run_summary.py RUN_FOLDER NAME...   (NAME: the name of a run, as in NAME.time; a shell pattern such as
'full_*' is expanded inside RUN_FOLDER)"""
import glob
import os
import re
import sys

FOLDER = sys.argv[1]


def wall(path):
    for line in open(path):
        if "Elapsed (wall clock)" in line:
            parts = [float(x) for x in line.split()[-1].split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return None


def last(pattern, text):
    found = re.findall(pattern, text)
    return found[-1] if found else "-"


names = []
for pattern in sys.argv[2:]:
    names += sorted(os.path.basename(p)[:-len(".time")] for p in glob.glob(os.path.join(FOLDER, pattern + ".time")))
print(f"{'run':32s} {'wall s':>9s} {'done in s':>10s} {'families':>9s} {'peak GiB':>9s} {'total log-likelihood':>22s}")
for name in names:
    err = os.path.join(FOLDER, name + ".err")
    text = open(err).read() if os.path.exists(err) else ""
    w = wall(os.path.join(FOLDER, name + ".time"))
    print(f"{name:32s} {w if w is None else round(w, 1)!s:>9s} {last(r'done in ([0-9.]+) s', text):>10s} "
          f"{last(r'Remaining families: ([0-9]+)', text):>9s} {last(r'peak GPU memory allocated:? ([0-9.]+) GiB', text):>9s} "
          f"{last(r'total log-likelihood: (-?[0-9.]+)', text):>22s}")
