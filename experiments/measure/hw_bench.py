"""Measured throughput of the CPU and the GPU of a machine with PyTorch (CPU: Intel MKL; GPU: cuBLAS; the paper used
PyTorch 2.13, 18 Oct 2026):
dense matrix products C = A @ B in single and double precision (2 n^3 floating-point operations each), and a memory
copy b.copy_(a) of a large array (bytes read plus bytes written, the STREAM 'copy' convention). Median of the timed
repetitions after warm-up runs; on the CPU, the best of 8, 16, 24 and 32 threads. On the GPU, TF32 is disabled so that
the single-precision product uses FP32 arithmetic.
usage: hw_bench.py OUT_JSON"""
import json, statistics as st, sys, time
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
out = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
def timed(fn, sync, warm, reps):
    for _ in range(warm): fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return st.median(ts)
nosync = lambda: None
cuda_sync = torch.cuda.synchronize
# CPU matrix products
for dtype, name in ((torch.float32, "cpu_matmul_fp32"), (torch.float64, "cpu_matmul_fp64")):
    n = 4096
    a = torch.randn(n, n, dtype=dtype); b = torch.randn(n, n, dtype=dtype); c = torch.empty(n, n, dtype=dtype)
    best = None
    for th in (8, 16, 24, 32):
        torch.set_num_threads(th)
        t = timed(lambda: torch.matmul(a, b, out=c), nosync, 2, 7)
        g = 2 * n ** 3 / t / 1e9
        out[f"{name}_threads{th}_gflops"] = round(g, 1)
        if best is None or g > best[0]: best = (g, th)
    out[name + "_gflops"], out[name + "_best_threads"] = round(best[0], 1), best[1]
    print(name, out[name + "_gflops"], "GFLOP/s with", best[1], "threads", flush=True)
# CPU memory copy
for th in (8, 16, 24, 32):
    torch.set_num_threads(th)
    a = torch.ones(512 * 2 ** 20, dtype=torch.float32); b = torch.empty_like(a)   # 2 GiB each
    t = timed(lambda: b.copy_(a), nosync, 2, 7)
    out[f"cpu_copy_threads{th}_gbs"] = round(2 * a.numel() * 4 / t / 1e9, 1)
out["cpu_copy_gbs"] = max(v for k, v in out.items() if k.startswith("cpu_copy_threads"))
print("cpu copy", out["cpu_copy_gbs"], "GB/s", flush=True)
del a, b
# GPU matrix products
for dtype, name, n in ((torch.float32, "gpu_matmul_fp32", 8192), (torch.float64, "gpu_matmul_fp64", 4096)):
    a = torch.randn(n, n, dtype=dtype, device="cuda"); b = torch.randn(n, n, dtype=dtype, device="cuda"); c = torch.empty(n, n, dtype=dtype, device="cuda")
    t = timed(lambda: torch.matmul(a, b, out=c), cuda_sync, 3, 20)
    out[name + "_gflops"] = round(2 * n ** 3 / t / 1e9, 1)
    print(name, out[name + "_gflops"], "GFLOP/s", flush=True)
    del a, b, c
# GPU memory copy
a = torch.ones(1024 * 2 ** 20, dtype=torch.float32, device="cuda"); b = torch.empty_like(a)   # 4 GiB each
t = timed(lambda: b.copy_(a), cuda_sync, 3, 20)
out["gpu_copy_gbs"] = round(2 * a.numel() * 4 / t / 1e9, 1)
print("gpu copy", out["gpu_copy_gbs"], "GB/s", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
