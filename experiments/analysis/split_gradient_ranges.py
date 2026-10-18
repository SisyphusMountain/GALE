"""Split the wall time of the profiled full fit (run_profiles.py part gpurec-nsys: the database full.sqlite that Nsight
Systems exports from full.nsys-rep, and gpurec's screen output full.err) into the likelihood (forward pass), the
gradient (backward pass), the second derivatives, the batch builds and the rest (paper: Figure 2b, \\prTLik, \\prTBack,
\\prTHess, \\prTBuild, \\prTTotal, \\prLik, \\prBack, \\prOther).
The NVTX ranges of wrappers/gpurec_nvtx.py give the second derivatives ('hessian'), the batch builds ('model build') and
the loss-and-gradient evaluations ('gradient'). An evaluation runs, batch after batch, a forward pass then a backward
pass, so its time is split along the GPU timeline: the kernels of the range are taken in the order they ran, and the
time from the start of one kernel to the start of the next goes to the current phase. The phase becomes 'backward' at a
backward kernel (name containing 'vjp', 'transpose' or 'adjoint') and 'forward' at a forward kernel (the extinction
update, the split sums, the leaf initialisation, the row update); the other kernels (array operations of PyTorch) keep
the current phase. The time before the first kernel of the range counts as forward, the time after its last kernel goes
to the phase of that kernel. A kernel belongs to the range in which the CPU launched it. The rest ('other': reading,
planning, the optimiser on the CPU) is the time gpurec reports ("done in" of full.err) minus the four parts.
Reads the database read-only; about 15 s and 2 GB of memory for the paper's profile.
usage: split_gradient_ranges.py FULL_SQLITE FULL_ERR"""
import bisect
import re
import sqlite3
import sys

# Names of gpurec's forward kernels (facts of the code of release 0.1, not settings).
FORWARD = ("_update_extinction_log_probabilities_kernel", "_reduce_single_gene_split_events_kernel",
           "_stage_multiple_gene_split_event_reduction_kernel", "_finalize_multiple_gene_split_event_reduction_kernel",
           "_initialize_leaf_reconciliation_likelihood_kernel", "_update_reconciliation_likelihood_kernel")


def is_backward(name):
    return "vjp" in name or "transpose" in name or "adjoint" in name


def split(sqlite_path):
    """Seconds of the forward pass, the backward pass, the 'hessian' ranges and the 'model build' ranges."""
    c = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    # (CPU launch time, GPU start time, name) of every kernel
    kernels = sorted((r, g, names[k]) for k, r, g in c.execute(
        "select k.shortName, r.start, k.start from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId = r.correlationId"))
    launches = [k[0] for k in kernels]
    t = {"forward": 0, "backward": 0, "hessian": 0, "model build": 0}
    ranges = c.execute("select text, start, end from NVTX_EVENTS where text in ('gradient', 'hessian', 'model build')")
    for text, s, e in ranges.fetchall():
        if text != "gradient":
            t[text] += e - s
            continue
        i, j = bisect.bisect_left(launches, s), bisect.bisect_right(launches, e)
        inside = sorted((g, n) for _, g, n in kernels[i:j])
        if not inside:
            t["forward"] += e - s
            continue
        phase, prev = "forward", s
        for g, n in inside:
            g = min(max(g, prev), e)
            t[phase] += g - prev
            prev = g
            if is_backward(n):
                phase = "backward"
            elif n in FORWARD:
                phase = "forward"
        t[phase] += e - prev
    return {k: v / 1e9 for k, v in t.items()}


def done_in(err_file):
    """The time gpurec reports for the run ("done in X s", the last one of its screen output)."""
    return float(re.findall(r"done in ([0-9.]+) s", open(err_file).read())[-1])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    parts, total = split(sys.argv[1]), done_in(sys.argv[2])
    parts["other"] = total - sum(parts.values())
    for k, v in parts.items():
        print(f"{k:12s} {v:7.1f} s  {100 * v / total:5.1f} %")
    print(f"total {total:.1f} s (done in)")
