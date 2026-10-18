"""Cost of one GPU kernel launch: an empty Triton kernel (one block, then 128 and 4,096 blocks), launched 20,000 times
in a row and timed to the end of the last one (the launches are queued asynchronously), and the time of one launch
followed by a wait for its end (median of 200). The paper's measurement was made while a CPU job was running.
usage: launch_cost.py"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def empty_kernel(x_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(x_ptr + pid, pid)


x = torch.zeros(8192, device="cuda", dtype=torch.int32)
for blocks in (1, 128, 4096):
    for _ in range(200): empty_kernel[(blocks,)](x, N=1)
    torch.cuda.synchronize()
    n = 20000
    t = time.perf_counter()
    for _ in range(n): empty_kernel[(blocks,)](x, N=1)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t) / n * 1e6
    ts = []
    for _ in range(200):
        t = time.perf_counter(); empty_kernel[(blocks,)](x, N=1); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"{blocks:5d} blocks: {per:.1f} us per launch when queued back to back; {ts[100]*1e6:.1f} us for one launch plus a wait (median of 200)")
