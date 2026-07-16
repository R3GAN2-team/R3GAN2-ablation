# R3GAN/kernels/sm100/gemm_forward.py
"""SM100 fused FFN forward GEMMs (R3GAN2 L1 / L1N / L3).

    from R3GAN.kernels.sm100.gemm_forward import FFNForward
    ffn = FFNForward()                            # JIT-compiles once
    y = ffn.l1(x, w1, bias, slope=0.2)            # LeakyReLU(conv1x1(x) + bias)
    y = ffn.l1n(x, w1, bias, noise, scale, 0.2)   # + noise*scale (per-pixel x per-channel)
    y = ffn.l3(x, w3, residual)                   # conv1x1(x) + residual

x is NHWC bf16 [N,H,W,Cin]; weight is [Cout,Cin] bf16; bias/noise/scale fp32.
Output is NHWC bf16 [N,H,W,Cout]. Pass out=<preallocated> to avoid an alloc.

Per (variant, M, N, K) the first call autotunes across the production config
set {c1,c2,c4} on the real tensors and caches the winner (see
gemm_forward_configs.cu). Call .warmup([...]) at init to pre-tune known shapes.

Run directly for a speed + accuracy table vs cuDNN:
    python gemm_forward.py
"""

import math
import os
import re
import sys

from .._build import (KernelSpec, cutlass_fingerprint, cutlass_include_dirs,
                     ensure_extension, require_dirs, resolve_cutlass)

CONFIGS_CU = "gemm_forward_configs.cu"
_BINDING_NAME = "bind_gemm_forward.cu"


def _arch_gencode(arch_list: str) -> str:
    """'10.0a' -> '-gencode=arch=compute_100a,code=sm_100a'. Passing the arch
    explicitly (rather than via TORCH_CUDA_ARCH_LIST) keeps the nvcc command
    line deterministic and avoids mutating process-global environment, which
    previously changed the build fingerprint of *other* extensions depending
    on construction order."""
    m = re.fullmatch(r"(\d+)\.(\d+)([a-z]?)", arch_list.strip())
    if not m:
        raise ValueError(f"unsupported arch_list {arch_list!r}; expected e.g. '10.0a'")
    tag = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return f"-gencode=arch=compute_{tag},code=sm_{tag}"


def _binding_source(pairs) -> str:
    decl = ("torch::Tensor {s}(torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, "
            "c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<double>, "
            "c10::optional<torch::Tensor>, c10::optional<torch::Tensor>);")
    df = ('  m.def("{sh}", &{s}, py::arg("input"), py::arg("weight"), py::arg("bias")=py::none(), '
          'py::arg("noise")=py::none(), py::arg("scale")=py::none(), py::arg("leaky_slope")=py::none(), '
          'py::arg("residual")=py::none(), py::arg("out")=py::none());')
    decls = "\n".join(decl.format(s=s) for s, _ in pairs)
    defs = "\n".join(df.format(s=s, sh=sh) for s, sh in pairs)
    return (f"#include <torch/extension.h>\nnamespace py = pybind11;\n{decls}\n"
            f"PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{\n{defs}\n}}\n")


def build_extension(cutlass=None, src_dir=None, cache_root=None,
                    arch_list="10.0a", verbose=False):
    """Build or fast-load the forward-GEMM extension; returns (module,
    config short names). Safe under concurrent launches; the generated
    binding is materialized inside the immutable per-key cache dir, never in
    a shared location (see _build.py)."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = src_dir or here
    cutlass = cutlass or resolve_cutlass("gemm_forward/sm100")

    cu = os.path.join(src_dir, CONFIGS_CU)
    if not os.path.isfile(cu):
        raise FileNotFoundError(f"missing {CONFIGS_CU} in {src_dir}")
    syms = re.findall(r"GEMM_FFN_SIG\(\s*(\w+)\s*\)\s*\{", open(cu).read())
    if not syms:
        raise RuntimeError(f"no GEMM_FFN_SIG(...) definitions in {cu}")
    pairs = [(s, s.replace("gemm_ffn_", "")) for s in syms]
    inc = cutlass_include_dirs(cutlass)
    require_dirs(inc, "set CUTLASS_DIR")

    spec = KernelSpec(
        name="gemm_forward_sm100",
        cc_major=10,
        sources=(cu,),
        hash_files=(os.path.join(src_dir, "gemm_forward_kernel.cuh"),),
        generated=((_BINDING_NAME, _binding_source(pairs)),),
        cflags=("-O3", "-std=c++17"),
        cuda_cflags=("-std=c++17", "-O3", "-DGEMM_BF16",
                     "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                     "--expt-relaxed-constexpr",
                     "--expt-extended-lambda",
                     _arch_gencode(arch_list)),
        include_paths=tuple(inc),
        src_dir=src_dir,
        extra_key=(f"cutlass:{cutlass_fingerprint(cutlass)}",),
    )
    mod = ensure_extension(spec, verbose=verbose, cache_root_override=cache_root)
    return mod, [sh for _, sh in pairs]


class FFNForward:
    def __init__(self, cutlass=None, src_dir=None, build_dir=None, autotune=True,
                 tune_iters=20, tune_warmup=5, flush_mb=256, arch_list="10.0a",
                 verbose=False):
        import torch
        self.torch = torch
        self.verbose = verbose
        self.autotune = autotune
        self.tune_iters = tune_iters
        self.tune_warmup = tune_warmup
        # build_dir, when given, overrides the cache ROOT for this extension
        # (each configuration still gets its own immutable subdirectory).
        self.mod, self.cfgs = build_extension(
            cutlass=cutlass, src_dir=src_dir, cache_root=build_dir,
            arch_list=arch_list, verbose=verbose)
        self.cache = {}   # (variant, M, N, K) -> cfg short name
        self._flush = (torch.empty(flush_mb * 1024 * 1024 // 4, dtype=torch.int32,
                                   device="cuda") if flush_mb > 0 else None)

    # -- dispatch ------------------------------------------------------------
    def _call(self, cfg, variant, x, w, bias, noise, scale, resid, slope, out=None):
        fn = getattr(self.mod, cfg)
        if variant == "L1":
            return fn(x, w, bias, None, None, slope, None, out)
        if variant == "L1N":
            return fn(x, w, bias, noise, scale, slope, None, out)
        return fn(x, w, None, None, None, None, resid, out)   # L3

    def _time(self, fn):
        t = self.torch
        for _ in range(self.tune_warmup):
            fn()
        t.cuda.synchronize()
        st = [t.cuda.Event(enable_timing=True) for _ in range(self.tune_iters)]
        en = [t.cuda.Event(enable_timing=True) for _ in range(self.tune_iters)]
        for i in range(self.tune_iters):
            if self._flush is not None:
                self._flush.zero_()
            st[i].record(); fn(); en[i].record()
        t.cuda.synchronize()
        ms = sorted(st[i].elapsed_time(en[i]) for i in range(self.tune_iters))
        return ms[len(ms) // 2]

    def _best(self, variant, x, w, bias, noise, scale, resid, slope):
        M = x.shape[0] * x.shape[1] * x.shape[2]
        N, K = w.shape[0], w.shape[1]
        key = (variant, M, N, K)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if not self.autotune:
            pick = "c4" if (M >= 32768 and "c4" in self.cfgs) else (
                   "c1" if (M <= 2048 and K >= 2048 and "c1" in self.cfgs) else
                   ("c2" if "c2" in self.cfgs else self.cfgs[0]))
            self.cache[key] = pick
            return pick
        best, best_ms = self.cfgs[0], float("inf")
        for cfg in self.cfgs:
            try:
                ms = self._time(lambda c=cfg: self._call(c, variant, x, w, bias,
                                                         noise, scale, resid, slope))
            except Exception as ex:
                if self.verbose:
                    print(f"[gemm_forward] config {cfg} failed on {variant} "
                          f"M={M} N={N} K={K}: {ex}")
                ms = float("inf")
            if ms < best_ms:
                best_ms, best = ms, cfg
        self.cache[key] = best
        if self.verbose:
            print(f"[gemm_forward autotune] {variant} M={M} N={N} K={K} -> {best} "
                  f"({best_ms*1e3:.1f} us)")
        return best

    # -- public layers -------------------------------------------------------
    def l1(self, x, w, bias, slope=0.2, out=None):
        bias = bias.contiguous()   # EffectiveWeightBias() returns a strided [Cout] slice view
        cfg = self._best("L1", x, w, bias, None, None, None, slope)
        return self._call(cfg, "L1", x, w, bias, None, None, None, slope, out)

    def l1n(self, x, w, bias, noise, scale, slope=0.2, out=None):
        bias, noise, scale = bias.contiguous(), noise.contiguous(), scale.contiguous()
        cfg = self._best("L1N", x, w, bias, noise, scale, None, slope)
        return self._call(cfg, "L1N", x, w, bias, noise, scale, None, slope, out)

    def l3(self, x, w, residual, out=None):
        cfg = self._best("L3", x, w, None, None, None, residual, 0.0)
        return self._call(cfg, "L3", x, w, None, None, None, residual, 0.0, out)

    def warmup(self, shapes, slope=0.2):
        """Pre-tune known shapes at init. Each item: (variant, N, H, W, Cin, Cout)."""
        t = self.torch
        dev, bf16, f32 = "cuda", t.bfloat16, t.float32
        for variant, N, H, W, Cin, Cout in shapes:
            M = N * H * W
            x = t.empty(N, H, W, Cin, device=dev, dtype=bf16)
            w = t.empty(Cout, Cin, device=dev, dtype=bf16)
            if variant == "L1":
                self.l1(x, w, t.empty(Cout, device=dev, dtype=f32), slope)
            elif variant == "L1N":
                self.l1n(x, w, t.empty(Cout, device=dev, dtype=f32),
                         t.empty(M, device=dev, dtype=f32),
                         t.empty(Cout, device=dev, dtype=f32), slope)
            else:
                self.l3(x, w, t.empty(N, H, W, Cout, device=dev, dtype=bf16))


# ============================================================================
# Validation table: fused vs cuDNN, speed + accuracy.
# ============================================================================
def _table(args):
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    ffn = FFNForward(cutlass=args.cutlass, verbose=args.verbose)
    bf16, f32 = torch.bfloat16, torch.float32
    a = args.alpha
    C = args.channels

    res_list = [int(r) for r in args.resolutions.split(",")]
    bspec = [int(b) for b in args.batches.split(",")]
    if len(bspec) == 1:
        bspec = bspec * len(res_list)
    if len(bspec) != len(res_list):
        sys.exit("--batches length must be 1 or match --resolutions")

    def leaky(x):
        return F.leaky_relu(x, a)

    def t_ms(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        for i in range(args.iters):
            ffn._flush.zero_() if ffn._flush is not None else None
            st[i].record(); fn(); en[i].record()
        torch.cuda.synchronize()
        ms = sorted(st[i].elapsed_time(en[i]) for i in range(args.iters))
        return ms[len(ms) // 2]

    print(f"\nFFN pointwise (1x1) layer   L1 = LeakyReLU(conv1x1(x) + bias)   "
          f"C={C}  bf16  (alpha={a})")
    print("fused = SM100 EVT kernel (autotuned cfg) | cuDNN = F.conv2d 1x1 "
          "channels_last + bias + leaky")
    print("err = max|impl - fp32 oracle|\n")
    hdr = (f"{'res':>7} {'batch':>6} {'M':>8} {'cfg':>4} {'fused_ms':>9} {'cudnn_ms':>9} "
           f"{'speedup':>8} {'fused_err':>10} {'cudnn_err':>10}  {'accuracy':<16}")
    print(hdr)
    print("-" * len(hdr))

    for res, N in zip(res_list, bspec):
        M = N * res * res
        gen = torch.Generator(device=dev).manual_seed(0)
        x = (torch.randn(N, res, res, C, generator=gen, device=dev) * 0.5).to(bf16)
        w = (torch.randn(C, C, generator=gen, device=dev) / math.sqrt(C)).to(bf16)
        bias_f = (torch.randn(C, generator=gen, device=dev) * 0.1).to(f32)
        bias_b = bias_f.to(bf16)
        x_cl = x.permute(0, 3, 1, 2)
        w1x1 = w.reshape(C, C, 1, 1)

        fused_fn = lambda: ffn.l1(x, w, bias_f, a)
        cudnn_fn = lambda: leaky(F.conv2d(x_cl, w1x1, bias_b))

        A = x.reshape(M, C)
        oracle = leaky(A.float() @ w.float().t() + bias_f[None, :]).reshape(N, res, res, C)
        fused = fused_fn().float()
        cudnn = cudnn_fn().permute(0, 2, 3, 1).float()
        fused_err = (fused - oracle).abs().max().item()
        cudnn_err = (cudnn - oracle).abs().max().item()

        f_ms = t_ms(fused_fn)
        c_ms = t_ms(cudnn_fn)
        cfg = ffn.cache[("L1", M, C, C)]
        verdict = "fused <= cudnn" if fused_err <= cudnn_err + 1e-6 else "fused > cudnn"
        print(f"{res:>5}x{res:<1} {N:>6} {M:>8} {cfg:>4} {f_ms:>9.4f} {c_ms:>9.4f} "
              f"{c_ms/f_ms:>7.2f}x {fused_err:>10.4f} {cudnn_err:>10.4f}  {verdict:<16}")

        del x, w, bias_f, bias_b, x_cl, w1x1, A, oracle, fused, cudnn
        torch.cuda.empty_cache()

    print("\nspeedup = cudnn_ms / fused_ms (>1 means fused is faster)")


def main():
    import argparse
    p = argparse.ArgumentParser(description="SM100 fused FFN forward vs cuDNN table.")
    p.add_argument("--cutlass", default=None)
    p.add_argument("--channels", type=int, default=1024)
    p.add_argument("--resolutions", default="8,16,32,64,128")
    p.add_argument("--batches", default="32,32,16,8,4",
                   help="Per-resolution batch; length 1 or matching --resolutions.")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    _table(args)


if __name__ == "__main__":
    main()
