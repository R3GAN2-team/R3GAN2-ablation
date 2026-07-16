"""Wgrad go/no-go: cuDNN grouped 3x3 weight-grad vs its analytic floors.

For each training (res, C) cell at N=1024: times torch.nn.grad.conv2d_weight
plain (k=1) and block-diagonal widened (production maps), and prints achieved
useful TF plus two floors:
  traffic floor = (read x + read dz) / peak_bw        (dw output is tiny)
  mma floor     = useful FLOPs / dense bf16 peak      (widened pays k x this)
The gap between best-measured and max(floors) is the budget a custom wgrad
kernel could recover. Usage: python bench_wgrad.py [N]   (default 1024)
"""
import sys
import time
import torch
from torch.nn.grad import conv2d_weight

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
WIDEN_WG = {8: 4, 16: 4}          # production map; 32 excluded (fast lane already)
PEAK_BW_TBS = 7.7                  # B200 HBM3e, adjust if you have a measured number
PEAK_TF = 2250.0                   # dense bf16 TC peak, for the mma floor only


def clock(fn, reps=30):
    for _ in range(5):
        fn()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / reps   # us


def expand(w, groups, k):
    cout, cpg, kh, kw = w.shape
    gm = groups // k
    we = w.new_zeros(gm, k, cout // groups, k, cpg, kh, kw)
    src = w.view(gm, k, cout // groups, cpg, kh, kw).transpose(0, 1)
    idx = torch.arange(k, device=w.device)
    we[:, idx, :, idx] = src
    return we.reshape(cout, k * cpg, kh, kw), gm


print(f"N={N}  bf16 channels_last  reps=30 median-free mean  (us | useful TF)")
hdr = f"{'shape':>14} {'plain us':>9} {'plain TF':>9} {'wide us':>9} {'wide TF':>9} {'traffic floor':>14} {'mma floor':>10} {'headroom':>9}"
print(hdr); print("-" * len(hdr))

for res in (32, 16, 8):
    for C in (512, 1024):
        G = C // 32
        x = torch.randn(N, C, res, res, device="cuda").to(torch.bfloat16).to(memory_format=torch.channels_last)
        dz = torch.randn_like(x)
        flops = 2.0 * N * res * res * C * 32 * 9
        bytes_rd = 2.0 * N * res * res * C * 2          # x + dz, bf16
        t_floor = bytes_rd / (PEAK_BW_TBS * 1e12) * 1e6  # us
        m_floor = flops / (PEAK_TF * 1e12) * 1e6         # us

        t_plain = clock(lambda: conv2d_weight(x, (C, 32, 3, 3), dz, padding=1, groups=G))
        k = WIDEN_WG.get(res, 1)
        if k > 1 and G % k == 0:
            t_wide = clock(lambda: conv2d_weight(x, (C, k * 32, 3, 3), dz, padding=1, groups=G // k))
        else:
            t_wide = t_plain
        tf = lambda us: flops / 1e12 / (us * 1e-6)
        best = min(t_plain, t_wide)
        floor = max(t_floor, m_floor)
        print(f"{res:>3}x{res:<3} C{C:<5} {t_plain:>9.1f} {tf(t_plain):>9.0f} {t_wide:>9.1f} {tf(t_wide):>9.0f} "
              f"{t_floor:>11.1f} us {m_floor:>7.1f} us {best/floor:>8.2f}x")
        del x, dz
        torch.cuda.empty_cache()

print("\nheadroom = best measured / max(floor): 1.0x means cuDNN is already at the")
print("physics; >2x means a custom wgrad kernel has real room. The go/no-go: sum")
print("(best - floor) across your stage mix and compare against a kernel campaign.")
