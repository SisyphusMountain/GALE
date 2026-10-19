"""Summarise the ablation runs of run_ablations.py. Two ways the runs can have been made:

  separate FOLDER   the parts run one at a time (OUTPUT_ROOT/ablations): every configuration ran once untimed and then
                    twice timed (NAME_1, NAME_2). For every run: the wall time of the whole command, the number of
                    kernels Triton compiled during it (compiles.txt), the peak GPU memory and the total log-likelihood
                    it reports; then, for each configuration, the mean of its two timed runs and its ratio to the release
                    settings ("base").
  session FOLDER    the part "session" (OUTPUT_ROOT/ablations/session), which gives the paper's Table "tab:ablations":
                    three rounds over the eight configurations, round 0 untimed, rounds 1 and 2 timed (runs NAME_0,
                    NAME_1, NAME_2; round 2 only for the configurations near the release settings, see TIMED_RUNS; a
                    run NAME_3 is a rerun made alone after the session, used in place of a disturbed one). For each
                    configuration: the wall times of its timed runs not flagged below, their mean and its ratio
                    to the mean of base, the peak GPU memory and the total log-likelihood of each timed run, the kernels
                    compiled during the timed runs; and every run during which progress.txt shows another process on the
                    GPU is flagged (such a run is not to be used). It also reads the folder of the paper's own session
                    script (results_gpu/scaling_ablations/queue5.sh, 19 October 2026), whose progress lines end with
                    "gpu_others: [...]" and whose pruning-off runs are named noprune_ROUND.

usage: ablations.py separate ABLATIONS_FOLDER
       ablations.py session SESSION_FOLDER"""
import glob
import os
import re
import statistics as st
import sys

SEPARATE_MODES = ("base", "alone", "noevict", "nowarm", "bysize", "inputorder", "base_again", "nopruning", "storealpha")
SESSION_MODES = ("base", "alone", "noevict", "nowarm", "bysize", "inputorder", "storealpha", "nopruning")
# Timed runs per configuration, as run_ablations.py SESSION: one for the configurations far from the release settings.
TIMED_RUNS = {"base": 2, "alone": 1, "noevict": 1, "nowarm": 1, "bysize": 2, "inputorder": 2, "storealpha": 2,
              "nopruning": 2}


def wall(path):
    for line in open(path):
        if "Elapsed (wall clock)" in line:
            parts = [float(x) for x in line.split()[-1].split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return None                                      # the run did not finish (time writes the line at the end)


def compiles(folder):
    out = {}
    if os.path.exists(f"{folder}/compiles.txt"):
        for line in open(f"{folder}/compiles.txt"):
            name, value = line.split()
            out[name] = int(value.split("=")[1])
    return out


# The log of a run is gale.log; the paper's own runs were made before the command was renamed from gpurec to gale
# and wrote gpurec.log, which is read when gale.log is absent.
LOG_NAMES = ("gale.log", "gpurec.log")


def total_ll(folder, name):
    logs = [f"{folder}/out_{name}/{n}" for n in LOG_NAMES if os.path.exists(f"{folder}/out_{name}/{n}")]
    if not logs:
        return None
    log = logs[0]
    v = re.findall(r"total log-likelihood: (-?[0-9.]+)", open(log).read())
    return float(v[-1]) if v else None


def peak(folder, name):
    err = f"{folder}/{name}.err"
    v = re.findall(r"peak GPU memory allocated: ([0-9.]+) GiB", open(err).read()) if os.path.exists(err) else []
    return float(v[-1]) if v else None


def text(value, form):
    return form % value if value is not None else "-"


def separate(folder):
    counts = compiles(folder)
    rows = []
    for t in sorted(glob.glob(f"{folder}/*.time")):
        name = os.path.basename(t)[:-5]
        rows.append((name, wall(t), counts.get(name), peak(folder, name), total_ll(folder, name)))
    print(f"{'run':28s} {'wall s':>8s} {'compiles':>9s} {'peak GiB':>9s} {'total log-lik':>16s}")
    for name, w, c, p, ll in rows:
        print(f"{name:28s} {text(w, '%8.1f'):>8s} {text(c, '%d'):>9s} {text(p, '%.2f'):>9s} {text(ll, '%.3f'):>16s}")
    print("\nmean of the two timed runs (warm compile cache), relative to base:")
    base = [w for n, w, *_ in rows if n in ("base_1", "base_2")]
    if len(base) == 2:
        b = st.mean(base)
        for m in SEPARATE_MODES:
            ws = [w for n, w, *_ in rows if n in (f"{m}_1", f"{m}_2")]
            if ws:
                print(f"  {m:11s} {st.mean(ws):8.1f} s  x{st.mean(ws)/b:5.2f}  runs {[round(x, 1) for x in ws]}")


def other_gpu_processes(folder):
    """For every run, the other GPU processes that progress.txt records before and after it (empty if none)."""
    seen = {}
    if os.path.exists(f"{folder}/progress.txt"):
        for line in open(f"{folder}/progress.txt"):
            m = (re.search(r": (\S+): (?:before|after), other GPU processes: \[(.*)\]$", line.rstrip("\n"))
                 or re.search(r"^(\S+): (?:start|exit=\S+) .* gpu_others: \[(.*)\]$", line.rstrip("\n")))
            if m:
                seen.setdefault(m.group(1), [])
                if m.group(2):
                    seen[m.group(1)].append(m.group(2))
    return seen


def session(folder):
    pruning_off = "noprune" if os.path.exists(f"{folder}/noprune_0.time") else "nopruning"
    modes = [pruning_off if m == "nopruning" else m for m in SESSION_MODES]
    counts = compiles(folder)
    others = other_gpu_processes(folder)
    print(f"{'run':16s} {'wall s':>8s} {'compiles':>9s} {'peak GiB':>9s} {'total log-lik':>16s}  other GPU processes")
    runs = {}
    for m in modes:
        for k in (0, 1, 2, 3):
            name = f"{m}_{k}"
            if not os.path.exists(f"{folder}/{name}.time"):
                continue
            runs[name] = (wall(f"{folder}/{name}.time"), counts.get(name), peak(folder, name), total_ll(folder, name))
            w, c, p, ll = runs[name]
            seen = others.get(name)
            flag = "not recorded" if seen is None else ("; ".join(seen) if seen else "none")
            print(f"{name:16s} {text(w, '%8.1f'):>8s} {text(c, '%d'):>9s} {text(p, '%.2f'):>9s} {text(ll, '%.3f'):>16s}  {flag}")

    def used(m):
        """The timed runs of configuration m that are used: rounds 1, 2 and the rerun 3, in that order, leaving out
        a run that failed or during which another process was on the GPU, up to the number of TIMED_RUNS."""
        key = "nopruning" if m == pruning_off else m
        good = [k for k in (1, 2, 3) if f"{m}_{k}" in runs and runs[f"{m}_{k}"][0] is not None
                and runs[f"{m}_{k}"][3] is not None and not others.get(f"{m}_{k}")]
        return good[:TIMED_RUNS[key]], TIMED_RUNS[key]

    print("\nper configuration, mean of its timed runs (warm compile cache), relative to base:")
    base_runs, base_needed = used("base")
    if len(base_runs) != base_needed:
        print("  base has not finished its timed runs")
        return
    b = st.mean(runs[f"base_{k}"][0] for k in base_runs)
    for m in modes:
        ks, needed = used(m)
        if len(ks) != needed:
            print(f"  {m:11s} usable timed runs: {len(ks)} of {needed}")
            continue
        ws = [runs[f"{m}_{k}"][0] for k in ks]
        print(f"  {m:11s} {st.mean(ws):8.1f} s  x{st.mean(ws)/b:5.2f}  runs {ks}: {[round(x, 1) for x in ws]}  "
              f"peak GiB {[text(runs[f'{m}_{k}'][2], '%.2f') for k in ks]}  "
              f"total log-lik {[text(runs[f'{m}_{k}'][3], '%.3f') for k in ks]}  compiles {[runs[f'{m}_{k}'][1] for k in ks]}")


if __name__ == "__main__":
    MODES = {"separate": separate, "session": session}
    if len(sys.argv) != 3 or sys.argv[1] not in MODES:
        raise SystemExit(__doc__)
    MODES[sys.argv[1]](sys.argv[2])
