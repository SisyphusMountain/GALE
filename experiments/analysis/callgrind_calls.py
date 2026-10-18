"""Sum, for each function named in a callgrind output file (valgrind --tool=callgrind), the number of times it was
called and its inclusive instruction count (instructions executed inside it and inside everything it calls).
usage: callgrind_calls.py CALLGRIND_OUT PATTERN...   (prints the functions whose name contains a PATTERN)"""
import re
import sys
from collections import defaultdict

names, calls, incl = {}, defaultdict(int), defaultdict(int)


def name_of(tok):
    m = re.match(r"\((\d+)\)(?: (.*))?", tok)
    if m.group(2): names[m.group(1)] = m.group(2)
    return names[m.group(1)]


pending = None
for line in open(sys.argv[1], errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("cfn="): callee = name_of(line[4:])
    elif line.startswith("fn="): name_of(line[3:])
    elif line.startswith("calls="): pending = (callee, int(line[6:].split()[0]))
    elif pending and line and (line[0].isdigit() or line[0] in "+-*"):
        c, n = pending; calls[c] += n; incl[c] += int(line.split()[1]); pending = None
for pat in sys.argv[2:]:
    for k in sorted(k for k in calls if pat in k):
        print(f"{calls[k]:>14,} calls  {incl[k]:>18,} instr incl  {k[:110]}")
