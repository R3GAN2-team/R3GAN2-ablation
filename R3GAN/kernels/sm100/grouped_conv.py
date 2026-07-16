"""R3GAN/kernels/sm100/grouped_conv.py -- SM100 grouped 3x3 conv (L2).

Warp-specialized TMA/TMEM kernel, group width 32, stride 1, pad 1, NHWC bf16,
fp32 accumulation. One binary serves fprop and dgrad (host-side weight
transform + pack). Fused epilogues: fprop lrelu + sign bitmap (FACT), dgrad
activation-slope multiply (FSLP).

Public surface:
    gconv32(x, w)                      plain conv, autograd (cuDNN backward)
    gconv32_auto(x, w, alt=...)        per-shape measured routing
    gconv32_act(x, w, alpha)           -> (a2, bits); autograd-safe
    gconv32_dgrad_slope(dz2, w, a1)    manual-pipeline fused dgrad
Verification: GCONV32_VERIFY=<n> checks the first n calls against cuDNN.
Compilation is lazy (first kernel call) and cached across launches and
concurrent runs; see _build.py. GCONV32_VERBOSE=1 for ninja output.
Design record: sm100fable docs/verification_protocol.md (F1-F4a closeout).
"""
import os
import torch
import torch.nn.functional as F

from .._build import (KernelSpec, cutlass_fingerprint, cutlass_include_dirs,
                     ensure_extension, require_dirs, resolve_cutlass, ccbin)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _spec() -> KernelSpec:
    cutlass = resolve_cutlass("grouped_conv/sm100")
    require_dirs(cutlass_include_dirs(cutlass), "set CUTLASS_DIR")
    return KernelSpec(
        name="grouped_conv_sm100",
        cc_major=10,
        # two-TU split: kernel TU has ZERO torch headers (torch/extension.h in
        # the kernel TU made ptxas spill 4/8 instantiations at 254 regs,
        # ~1.8x slower)
        sources=(os.path.join(_HERE, "grouped_conv_kernel.cu"),
                 os.path.join(_HERE, "grouped_conv_binding.cpp")),
        cuda_cflags=(
            "-O3", "-std=c++17", "-arch=sm_100a",
            "--expt-relaxed-constexpr", f"-ccbin={ccbin()}",
            # torch's COMMON_NVCC_FLAGS -D__CUDA_NO_{HALF,BFLOAT16}_* disable
            # the native bf16/half conversion intrinsics; CUTLASS converts then
            # falls back to software paths, ptxas hits the 254-register cap and
            # spills 700+ B of stack per thread (~1.8x kernel slowdown,
            # measured). -U after torch's -D restores the intrinsics.
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ),
        cflags=("-O2", "-std=c++17"),
        include_paths=tuple(cutlass_include_dirs(cutlass)),
        src_dir=_HERE,
        extra_key=(f"cutlass:{cutlass_fingerprint(cutlass)}",),
    )


def build_extension(verbose: bool | None = None):
    """Build (first launch of a configuration) or fast-load (every launch
    after) the compiled extension. Safe under concurrent launches; see
    _build.py. Called lazily on first kernel use, or explicitly by
    `python -m R3GAN.kernels.prebuild`."""
    if verbose is None:
        verbose = os.environ.get("GCONV32_VERBOSE", "0") == "1"
    return ensure_extension(_spec(), verbose=verbose)


class _LazyExt:
    """Defers the JIT build to first kernel use so that importing this module
    (tests, tooling, non-SM100 machines) stays free of side effects."""
    __slots__ = ()

    def __getattr__(self, attr):
        return getattr(build_extension(), attr)


_ext = _LazyExt()

VERIFY_CALLS = int(os.environ.get("GCONV32_VERIFY", "3"))
_verified = 0


_pack_idx = None
try:                                   # weak keys: id()-reuse of collected
    from torch.utils.weak import WeakTensorKeyDictionary   # tensors aliased
    _pack_cache = WeakTensorKeyDictionary()                 # a plain id-dict
except ImportError:
    _pack_cache = None


def _pack(w: torch.Tensor) -> torch.Tensor:
    # [C,32,3,3] -> [C,288] (k = c + 32*dx + 96*dy) -> SMEM byte order (v1.72)
    global _pack_idx
    hit = _pack_cache.get(w) if _pack_cache is not None else None
    if hit is not None and hit[0] == w._version:
        return hit[1]
    if _pack_idx is None or _pack_idx.device != w.device:
        _pack_idx = _ext.pack_order().to(w.device)
    C = w.size(0)
    w288 = w.permute(0, 2, 3, 1).reshape(C // 32, 32 * 288)
    packed = w288.index_select(1, _pack_idx).reshape(C, 288).contiguous()
    if _pack_cache is not None:
        _pack_cache[w] = (w._version, packed)
    return packed


_dgrad_cache = (WeakTensorKeyDictionary() if _pack_cache is not None else None)


def _pack_dgrad(w: torch.Tensor) -> torch.Tensor:
    """Pack for the transposed conv: convT(., w) == conv(., w').

    Per group g: w'[a, b, ky, kx] = w[b, a, 2-ky, 2-kx] (co<->ci transpose +
    spatial flip). Same kernel binary, different pack. Cached on (w, version).
    """
    hit = _dgrad_cache.get(w) if _dgrad_cache is not None else None
    if hit is not None and hit[0] == w._version:
        return hit[1]
    C = w.size(0)
    wd = (w.view(C // 32, 32, 32, 3, 3).transpose(1, 2)
           .flip(3, 4).reshape(C, 32, 3, 3))
    packed = _pack(wd)
    if _dgrad_cache is not None:
        _dgrad_cache[w] = (w._version, packed)
    return packed


def _in_scope(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 4
            and x.is_contiguous(memory_format=torch.channels_last)
            and w.size(1) == 32 and w.size(2) == 3 and w.size(3) == 3
            and x.size(1) % 128 == 0 and w.size(0) == x.size(1)
            and ((x.size(2) == 8 and x.size(3) == 8 and x.size(0) % 2 == 0)
                 or (x.size(2) % 16 == 0 and x.size(3) % 8 == 0)))


class _Gconv32Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, gs=4):
        global _verified
        y = _ext.fprop(x, _pack(w.to(torch.bfloat16)), gs)
        if _verified < VERIFY_CALLS:
            ref = F.conv2d(x, w.to(torch.bfloat16), stride=1, padding=1,
                           groups=x.size(1) // 32)
            err = (y.float() - ref.float()).abs().max().item()
            scale = ref.float().abs().max().item() + 1e-6
            assert err / scale < 2e-2, f"gconv32 verify FAILED: rel {err/scale:.3e}"
            print(f"[gconv32] verify {_verified + 1}/{VERIFY_CALLS} "
                  f"shape {tuple(x.shape)} max_abs {err:.3e} (rel {err/scale:.1e}) OK")
            _verified += 1
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        # (third forward arg gs needs no grad)
        gy = gy.contiguous(memory_format=torch.channels_last)
        gx, gw, _ = torch.ops.aten.convolution_backward(
            gy, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0],
            x.size(1) // 32, [ctx.needs_input_grad[0], ctx.needs_input_grad[1], False])
        return gx, gw, None


def gconv32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Grouped 3x3 conv, stride 1, pad 1, groups = C//32."""
    if _in_scope(x, w):
        return _Gconv32Fn.apply(x, w)
    return F.conv2d(x, w, stride=1, padding=1, groups=x.size(1) // 32)


class _Gconv32ActFn(torch.autograd.Function):
    """Fused y2 = conv(x, w); a2 = leaky_relu(y2, alpha); bits = (y2 >= 0).

    bits is the sign bitmap of the PRE-activation, packed 32 channels/word,
    NHWC-linear [N, H, W, C/32] int32, pair-interleaved bit order:
    channel c of block cb lives at bit (c>>1) | ((c&1)<<4) of bits[n,h,w,cb]
    (sign extraction from converted bf16 pairs; constant map, apply once
    in any consumer). It is the slope source for the downstream L3-backward GEMM (their
    UnscaledLeakyReLU.Slope convention, x >= 0 -> slope 1 else alpha).
    Backward here uses sign(a2) == sign(y2) (alpha > 0), so no bit unpack.
    """
    @staticmethod
    def forward(ctx, x, w, alpha=0.2, gs=4):
        global _verified
        y, bits = _ext.fprop_act(x, _pack(w.to(torch.bfloat16)), gs, alpha)
        if _verified < VERIFY_CALLS:
            y2 = F.conv2d(x, w.to(torch.bfloat16), stride=1, padding=1,
                          groups=x.size(1) // 32)
            ref = F.leaky_relu(y2, alpha)
            err = (y.float() - ref.float()).abs().max().item()
            scale = ref.float().abs().max().item() + 1e-6
            bref = (y2 >= 0)
            _ar = torch.arange(32, device=x.device)
            _bp = (_ar >> 1) | ((_ar & 1) << 4)          # channel -> bit position
            bgot = ((bits.unsqueeze(-1) >> _bp) & 1).bool()
            bgot = bgot.reshape(x.size(0), x.size(2), x.size(3), x.size(1)).permute(0, 3, 1, 2)
            bad = (bgot != bref).sum().item()
            assert err / scale < 2e-2 and bad == 0, \
                f"gconv32_act verify FAILED: rel {err/scale:.3e} bad_bits {bad}"
            print(f"[gconv32] verify-act {_verified + 1}/{VERIFY_CALLS} "
                  f"shape {tuple(x.shape)} max_abs {err:.3e} bad_bits {bad} OK")
            _verified += 1
        ctx.save_for_backward(x, w, y)
        ctx.alpha = alpha
        ctx.gs = gs
        return y, bits

    @staticmethod
    def backward(ctx, ga, _gbits):
        x, w, a2 = ctx.saved_tensors
        alpha = ctx.alpha
        ga = ga.contiguous(memory_format=torch.channels_last)
        # slope(y2) from a2 in ONE fused pass (torch's own lrelu-backward;
        # self_is_result=True reads the activation output). Matches torch
        # autograd bit-for-bit incl. slope alpha at exact zeros; the >=0
        # convention lives in the bitmap for the explicit-VJP/R1 consumers.
        dz2 = torch.ops.aten.leaky_relu_backward(ga, a2, alpha, True)
        gx = gw = None
        if ctx.needs_input_grad[0]:
            wb = w.to(torch.bfloat16)
            dz2 = dz2.contiguous(memory_format=torch.channels_last)
            if _in_scope(dz2, wb):
                # F3: dx = convT(dz2, w) == our kernel on the dgrad pack.
                # NOTE: plain convT on purpose -- in the autograd graph the
                # upstream in-place leaky_relu node applies slope(y1) itself;
                # the fused-slope kernel (gconv32_dgrad_slope) is for manual
                # backward pipelines with no lrelu node.
                gx = _ext.fprop(dz2, _pack_dgrad(wb), ctx.gs)
            else:
                gx, _, _ = torch.ops.aten.convolution_backward(
                    dz2, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0],
                    x.size(1) // 32, [True, False, False])
        if ctx.needs_input_grad[1]:
            _, gw, _ = torch.ops.aten.convolution_backward(
                dz2, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0],
                x.size(1) // 32, [False, True, False])
        return gx, gw, None, None


def gconv32_conv_slope(x: torch.Tensor, w: torch.Tensor, src: torch.Tensor,
                       alpha: float = 0.2):
    """y = conv(x, w) * slope(src), one kernel: FSLP with the FORWARD pack.

    The vjp backward's q step (q = conv(p, w2) * slope(y2)). slope = 1 where
    src's sign bit is clear, else alpha; differences vs leaky_relu_backward
    exist only at exact bf16 zeros. No autograd; raw kernel call.
    """
    wb = w.to(torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last)
    if _in_scope(x, wb):
        return _ext.fprop_slope(x, _pack(wb), 4, alpha,
                                src.contiguous(memory_format=torch.channels_last))
    y = F.conv2d(x, wb, stride=1, padding=1, groups=wb.size(0) // 32)
    return torch.ops.aten.leaky_relu_backward(y, src, alpha, True)


def gconv32_dgrad_slope(dz2: torch.Tensor, w: torch.Tensor, a1: torch.Tensor,
                        alpha: float = 0.2):
    """dy1 = convT(dz2, w) * slope(y1), one kernel (F4a). Manual-pipeline op.

    Drop-in for fused_ffn.l2_dgrad_slope1: dz2 = pre-activation grad wrt y2,
    w = RAW forward L2 weight [C,32,3,3] (dgrad transform + pack done here,
    cached on (w, version)), a1 = post-activation-1 tensor (slope source:
    sign(a1) == sign(y1) for alpha > 0; slope 1 when the sign bit is clear,
    matching UnscaledLeakyReLU.Slope's >= 0 convention -- including at -0,
    where a >= test on a1 would be wrong). No autograd; raw kernel call.
    """
    wb = w.to(torch.bfloat16)
    dz2 = dz2.contiguous(memory_format=torch.channels_last)
    if _in_scope(dz2, wb):
        return _ext.fprop_slope(dz2, _pack_dgrad(wb), 4, alpha,
                                a1.contiguous(memory_format=torch.channels_last))
    y = F.conv2d(dz2, (wb.view(wb.size(0) // 32, 32, 32, 3, 3).transpose(1, 2)
                       .flip(3, 4).reshape(wb.size(0), 32, 3, 3)),
                 stride=1, padding=1, groups=wb.size(0) // 32)
    return torch.ops.aten.leaky_relu_backward(y, a1, alpha, True)


def gconv32_act(x: torch.Tensor, w: torch.Tensor, alpha: float = 0.2):
    """Fused grouped conv + UnscaledLeakyReLU + sign bitmap.

    Returns (a2, bits). Drop-in for `NonLinearity(LinearLayer2(a1, Gain=...))`
    when the Gain is already folded into w (pass EffectiveWeight output).
    bits: int32 [N,H,W,C/32] pair-interleaved sign bitmap (see _Gconv32ActFn), for the L3
    backward slope; non-differentiable.
    """
    if _in_scope(x, w):
        return _Gconv32ActFn.apply(x, w, alpha)
    y2 = F.conv2d(x, w, stride=1, padding=1, groups=x.size(1) // 32)
    n, c, h, wd = y2.shape
    m = (y2 >= 0).permute(0, 2, 3, 1).reshape(n, h, wd, c // 32, 32)
    _ar = torch.arange(32, device=y2.device)
    _bp = (_ar >> 1) | ((_ar & 1) << 4)                  # channel -> bit position
    bits = (m.int() << _bp).sum(-1, dtype=torch.int32)
    return F.leaky_relu(y2, alpha), bits


# ---- per-shape autotuned routing (footgun-15 style: measure, never assume) --
_route = {}


def gconv32_auto(x, w, alt=None):
    """Route per shape between our kernel and `alt` (your dispatch hook);
    winner decided by a one-time timing at first encounter of each shape."""
    if alt is None:
        def alt(x_, w_):
            return F.conv2d(x_, w_, stride=1, padding=1, groups=x_.size(1) // 32)
    if not _in_scope(x, w):
        return alt(x, w)
    key = tuple(x.shape)
    use_ours = _route.get(key)
    if use_ours is None:
        def _t(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(10):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1)
        with torch.no_grad():
            wp = _pack(w)
            t4 = _t(lambda: _ext.fprop(x, wp, 4))
            t2 = _t(lambda: _ext.fprop(x, wp, 2))
            ta = _t(lambda: alt(x, w))
        use_ours = min(t4, t2) < ta
        _route[key] = (4 if t4 <= t2 else 2) if use_ours else 0
        use_ours = _route[key]
        print(f"[gconv32] route {key}: "
              f"{('gs%d' % use_ours) if use_ours else 'alt'} "
              f"(gs4 {t4/10:.3f} / gs2 {t2/10:.3f} / alt {ta/10:.3f} ms)")
    return _Gconv32Fn.apply(x, w, use_ours) if use_ours else alt(x, w)
