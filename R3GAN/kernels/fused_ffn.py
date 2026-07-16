# R3GAN/kernels/fused_ffn.py
"""SM100 fused-FFN installer for R3GAN.

Architecture-agnostic dispatch layer: this file owns the FeedForwardNetwork
monkey-patch, the autograd wiring, and the layer routing; everything
SM100-specific lives in R3GAN.kernels.sm100 behind small wrapper classes, so a
future SM generation slots in by providing the same surface:

  R3GAN.kernels.sm100.gemm_forward.FFNForward.{l1, l1n, l3}
  R3GAN.kernels.sm100.gemm_backward.FFNBackward.{dx, dw}
  R3GAN.kernels.sm100.grouped_conv.{gconv32, gconv32_act, gconv32_dgrad_slope}

Boundary convention:
  R3GAN network tensors are NCHW shape with channels_last storage.
  The GEMM kernels consume logical NHWC contiguous tensors (zero-copy
  as_strided views); grouped_conv consumes NCHW channels_last directly.

L2 (grouped 3x3) routing, measured July 2026 (N512, B200, vs the wide-group
block-diagonal cuDNN path this file previously used):
  res 32: 1.40-1.44x block, res 16: 1.50x, res 8: 1.02-1.03x -- every cell,
  C1024 and C2048, in favor of the gconv32 kernels (fused lrelu fprop; dgrad
  with the activation-1 slope fused in the epilogue). gconv32 is therefore the
  default L2 path at every resolution; R3GAN_L2_GCONV=0 restores wide-group
  cuDNN for A/B. Wide-group WGRAD widening is unrelated to gconv32 (no wgrad
  kernel exists) and keeps its measured maps.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight as _conv2d_weight

# Resolution gate (0 = always fused). With the gconv32 L2 path the fused block
# wins at every training resolution (see module docstring), so the gate is off
# by default; it remains available for bring-up on new shapes.
_FUSED_MIN_RES = int(os.environ.get("R3GAN_FUSED_MIN_RES", "0"))

# L2 kernel selection. gconv32 (SM100 halo kernel, fused epilogues) is the
# measured winner at every (res, C) cell; set R3GAN_L2_GCONV=0 to route L2
# through the wide-group block-diagonal cuDNN path instead (kept for A/B and
# for out-of-scope shapes).
_L2_GCONV = os.environ.get("R3GAN_L2_GCONV", "1") == "1"

# Fused explicit-vjp (R1 path). Routes FeedForwardNetwork.explicit_vjp through
# the SM100 dx/dw kernels via a custom autograd.Function with a hand-derived
# backward (R1 weight grads flow THROUGH the vjp, so raw kernel calls without
# a backward would silently sever R1's dW contribution). DEFAULT OFF until
# test_fused_vjp.py passes on your setup: enable with install_fused_ffn(...,
# vjp=True) or R3GAN_FUSED_VJP=1.
_VJP_FUSED = os.environ.get("R3GAN_FUSED_VJP", "0") == "1"

# Weight-grad kernel for the 1x1 layers. bench_kernels_isolated: the stream-K
# dw kernel runs 0.72-0.81x vs cuBLAS at every resolution, so dW1/dW3 default
# to cuBLAS matmul on zero-copy NHWC views. Trade: bf16-out final-store
# rounding (relL2 ~1.3e-4 vs fp32 oracle; accumulation is fp32 inside cuBLAS
# either way). R3GAN_FUSED_DW_STREAMK=1 restores the fp32-out stream-K kernel.
_DW_STREAMK = os.environ.get("R3GAN_FUSED_DW_STREAMK", "0") == "1"


_ONES_ROW: Dict[Tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def _ones_row(m: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Cached [m] ones row for the db/ds stacked GEMM (the row never changes;
    re-filling it every backward showed up as an M-sized aten::fill_ per
    noise block in the step profile)."""
    key = (m, dtype, device)
    hit = _ONES_ROW.get(key)
    if hit is None:
        hit = torch.ones(m, dtype=dtype, device=device)
        _ONES_ROW[key] = hit
    return hit


def _mm_wgrad(a_nchw: torch.Tensor, b_nchw: torch.Tensor) -> torch.Tensor:
    """a^T @ b over pixels via cuBLAS. permute on channels_last tensors is a
    zero-copy NHWC view; the .t() is handled as a cuBLAS transpose op."""
    a2 = a_nchw.permute(0, 2, 3, 1).reshape(-1, a_nchw.shape[1])
    b2 = b_nchw.permute(0, 2, 3, 1).reshape(-1, b_nchw.shape[1])
    return a2.t() @ b2


# -----------------------------------------------------------------------------
# Wide-group block-diagonal cuDNN machinery.
#
# Kept for (a) L2 WGRAD, where widening measured 6.5x/10.3x at res 8/16 and is
# the production path, and (b) the R3GAN_L2_GCONV=0 fallback for fwd/dgrad.
# Maps are per-resolution "res:k" comma lists; re-run bench_l2_widegroup.py
# and override via env if batch size, channel widths, or cuDNN version change.
# -----------------------------------------------------------------------------

def _parse_widen_map(s: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        r, k = part.split(":")
        out[int(r)] = int(k)
    return out


_L2_WIDEN = _parse_widen_map(os.environ.get("R3GAN_L2_WIDEN", "8:2,16:4,32:2"))
_L2_WIDEN_DGRAD = _parse_widen_map(os.environ.get("R3GAN_L2_WIDEN_DGRAD", "")) or _L2_WIDEN
_L2_WIDEN_WGRAD = _parse_widen_map(os.environ.get("R3GAN_L2_WIDEN_WGRAD", "8:4,16:4"))


def _expand_grouped_w(w: torch.Tensor, groups: int, k: int) -> Tuple[torch.Tensor, int]:
    """[Cout, cpg, kh, kw] @ groups -> block-diag [Cout, k*cpg, kh, kw] @ groups//k."""
    cout, cpg, kh, kw = w.shape
    gm = int(groups) // k
    we = w.new_zeros(gm, k, cout // int(groups), k, cpg, kh, kw)
    src = w.view(gm, k, cout // int(groups), cpg, kh, kw).transpose(0, 1)
    idx = torch.arange(k, device=w.device)
    we[:, idx, :, idx] = src
    return we.reshape(cout, k * cpg, kh, kw), gm


def _extract_blockdiag_wgrad(dwe: torch.Tensor, groups: int, k: int, cpg: int) -> torch.Tensor:
    cout, _, kh, kw = dwe.shape
    gm = int(groups) // k
    idx = torch.arange(k, device=dwe.device)
    v = dwe.view(gm, k, cout // int(groups), k, cpg, kh, kw)
    return v[:, idx, :, idx].transpose(0, 1).reshape(cout, cpg, kh, kw)


def _grouped_conv_fwd(x_nchw: torch.Tensor, w: torch.Tensor, groups: int,
                      widen_map: Dict[int, int]) -> torch.Tensor:
    k = widen_map.get(int(x_nchw.shape[-1]), 1)
    if k > 1 and int(groups) % k == 0:
        we, gm = _expand_grouped_w(w, groups, k)
        return F.conv2d(x_nchw, we, padding=1, groups=gm)
    return F.conv2d(x_nchw, w, padding=1, groups=int(groups))


def _grouped_wgrad(inp_nchw: torch.Tensor, w_shape, gout_nchw: torch.Tensor,
                   groups: int) -> torch.Tensor:
    k = _L2_WIDEN_WGRAD.get(int(inp_nchw.shape[-1]), 1)
    if k > 1 and int(groups) % k == 0:
        cout, cpg, kh, kw = w_shape
        dwe = _conv2d_weight(inp_nchw, (cout, k * cpg, kh, kw), gout_nchw,
                             padding=1, groups=int(groups) // k)
        return _extract_blockdiag_wgrad(dwe, groups, k, cpg)
    return _conv2d_weight(inp_nchw, w_shape, gout_nchw, padding=1, groups=int(groups))


def _flip_transpose_grouped_w2(w: torch.Tensor, groups: int) -> torch.Tensor:
    """Dgrad forward-conv weight (fallback path only; grouped_conv owns its
    own transform + pack): W''[ci,co,a,b] = W[co,ci,2-a,2-b] within a group."""
    cout, cin_g, kh, kw = w.shape
    cout_g = cout // int(groups)
    return (w.view(groups, cout_g, cin_g, kh, kw)
             .transpose(1, 2)
             .flip(-1, -2)
             .contiguous()
             .view(groups * cin_g, cout_g, kh, kw))


# -----------------------------------------------------------------------------
# Layout helpers
# -----------------------------------------------------------------------------

def _require_nchw_cl(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 4:
        raise RuntimeError(f"{name}: expected NCHW 4D tensor, got shape={tuple(x.shape)}")
    if not x.is_contiguous(memory_format=torch.channels_last):
        raise RuntimeError(
            f"{name}: expected channels_last NCHW tensor, got shape={tuple(x.shape)} "
            f"stride={tuple(x.stride())}. Do not insert permute/contiguous here; "
            "fix the caller to preserve channels_last."
        )
    return x


def _nchw_cl_to_nhwc(x: torch.Tensor, name: str) -> torch.Tensor:
    x = _require_nchw_cl(x, name)
    n, c, h, w = x.shape
    sn, sc, sh, sw = x.stride()
    y = x.as_strided((n, h, w, c), (sn, sh, sw, sc))
    if not y.is_contiguous():
        raise RuntimeError(f"{name}: NHWC view is not contiguous: "
                           f"shape={tuple(y.shape)} stride={tuple(y.stride())}")
    return y


def _nhwc_to_nchw_cl(y: torch.Tensor, expected: Tuple[int, int, int, int], name: str) -> torch.Tensor:
    if y.ndim != 4 or not y.is_contiguous():
        raise RuntimeError(f"{name}: expected contiguous NHWC kernel output, got "
                           f"shape={tuple(y.shape)} stride={tuple(y.stride())}")
    n, c, h, w = expected
    if tuple(y.shape) != (n, h, w, c):
        raise RuntimeError(f"{name}: expected NHWC {(n, h, w, c)}, got {tuple(y.shape)}")
    sn, sh, sw, sc = y.stride()
    return y.as_strided((n, c, h, w), (sn, sc, sh, sw))


def _w2d_1x1(w: torch.Tensor) -> torch.Tensor:
    if w.ndim == 2:
        return w.contiguous()
    if w.ndim == 4 and w.shape[-2:] == (1, 1):
        return w[:, :, 0, 0].contiguous()
    raise RuntimeError(f"expected 1x1 weight as [Cout,Cin] or [Cout,Cin,1,1], got {tuple(w.shape)}")


def _sum_channels_nchw(x: torch.Tensor) -> torch.Tensor:
    # dtype=float32 gives fp32 accumulation inside the reduce kernel WITHOUT
    # materializing a full-size fp32 copy of x.
    return x.sum(dim=(0, 2, 3), dtype=torch.float32)


# -----------------------------------------------------------------------------
# Concrete kernel bundle
# -----------------------------------------------------------------------------

class _SM100Kernels:
    def __init__(self, l2_gconv: bool):
        from R3GAN.kernels.sm100.gemm_forward import FFNForward
        from R3GAN.kernels.sm100.gemm_backward import FFNBackward
        from R3GAN.kernels.sm100 import grouped_conv
        self.ffn = FFNForward()
        self.bwd = FFNBackward()
        self.gc = grouped_conv
        self.l2_gconv = bool(l2_gconv)

    def symbols(self) -> Dict[str, Any]:
        return {
            "gemm_forward": [x for x in dir(self.ffn) if not x.startswith("__")],
            "grouped_conv": ("gconv32/gconv32_act/gconv32_dgrad_slope"
                             if self.l2_gconv else "disabled -> wide-group cuDNN"),
            "gemm_backward": (self.bwd.symbols() if hasattr(self.bwd, "symbols")
                              else [x for x in dir(self.bwd) if not x.startswith("__")]),
        }

    # Forward 1x1s.
    def l1(self, x_nchw: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
           noise_col: Optional[torch.Tensor], scale: Optional[torch.Tensor]) -> torch.Tensor:
        x = _nchw_cl_to_nhwc(x_nchw, "l1.x")
        # L1 maps InputChannels -> HiddenChannels; the output channel count comes
        # from w1, NOT from x (they differ whenever FFNWidthRatio != 1).
        n, _, h, wpx = x_nchw.shape
        expected = (n, int(w1.shape[0]), h, wpx)
        if noise_col is None:
            y = self.ffn.l1(x, w1, b1)
        else:
            y = self.ffn.l1n(x, w1, b1, noise_col, scale)
        return _nhwc_to_nchw_cl(y, expected, "l1.y")

    def l2_forward(self, a1_nchw: torch.Tensor, w2: torch.Tensor, groups: int,
                   alpha: float) -> torch.Tensor:
        if self.l2_gconv:
            # gconv32 fused fprop: conv + UnscaledLeakyReLU in one kernel; the
            # sign bitmap it also emits is not consumed here yet (candidate
            # slope source for bwd.dx(act=a2), 16x cheaper than reading a2).
            a2, _bits = self.gc.gconv32_act(a1_nchw, w2.to(a1_nchw.dtype), float(alpha))
            return a2
        y = _grouped_conv_fwd(a1_nchw, w2, groups, _L2_WIDEN)
        return F.leaky_relu(y, negative_slope=float(alpha), inplace=True)

    def l3(self, a2_nchw: torch.Tensor, w3: torch.Tensor, residual_nchw: torch.Tensor) -> torch.Tensor:
        a2 = _nchw_cl_to_nhwc(a2_nchw, "l3.a2")
        residual = _nchw_cl_to_nhwc(residual_nchw, "l3.residual")
        y = self.ffn.l3(a2, w3, residual)
        return _nhwc_to_nchw_cl(y, tuple(residual_nchw.shape), "l3.y")

    # Backward 1x1s.
    def dx(self, dy_nchw: torch.Tensor, w: torch.Tensor,
           residual_nchw: Optional[torch.Tensor] = None,
           act_nchw: Optional[torch.Tensor] = None) -> torch.Tensor:
        dy = _nchw_cl_to_nhwc(dy_nchw, "dx.dy")
        residual = _nchw_cl_to_nhwc(residual_nchw, "dx.residual") if residual_nchw is not None else None
        act = _nchw_cl_to_nhwc(act_nchw, "dx.act") if act_nchw is not None else None
        y = self.bwd.dx(dy, w, residual, act)
        n, _, h, ww = dy_nchw.shape
        expected = (n, int(w.shape[1]), h, ww)
        return _nhwc_to_nchw_cl(y, expected, "dx.y")

    def dw(self, dy_nchw: torch.Tensor, x_nchw: torch.Tensor) -> torch.Tensor:
        dy = _nchw_cl_to_nhwc(dy_nchw, "dw.dy")
        x = _nchw_cl_to_nhwc(x_nchw, "dw.x")
        return self.bwd.dw(dy, x).contiguous()

    def l2_dgrad_slope1(self, dz2_nchw: torch.Tensor, w2: torch.Tensor, a1_nchw: torch.Tensor,
                        groups: int, alpha: float) -> torch.Tensor:
        if self.l2_gconv:
            # gconv32 dgrad with slope(y1) fused in the epilogue (slope source:
            # sign(a1); matches Slope()'s >= 0 convention including at -0).
            # One kernel replaces flip-weight conv + leaky_relu_backward.
            return self.gc.gconv32_dgrad_slope(dz2_nchw, w2, a1_nchw, float(alpha))
        wd = _flip_transpose_grouped_w2(w2, int(groups))
        y = _grouped_conv_fwd(dz2_nchw, wd, groups, _L2_WIDEN_DGRAD)
        return torch.ops.aten.leaky_relu_backward(y, a1_nchw, float(alpha), True)

    def l2_conv_slope(self, x_nchw: torch.Tensor, w2: torch.Tensor, y2_nchw: torch.Tensor,
                      groups: int, alpha: float) -> torch.Tensor:
        """q = conv(x, w2) * slope(y2), one kernel (vjp backward's q step)."""
        if self.l2_gconv:
            return self.gc.gconv32_conv_slope(x_nchw, w2, y2_nchw, float(alpha))
        q = _grouped_conv_fwd(x_nchw, w2, groups, _L2_WIDEN)
        return torch.ops.aten.leaky_relu_backward(q, y2_nchw, float(alpha), True)


_KERNELS: Optional[_SM100Kernels] = None
_KERNELS_MU = threading.Lock()


def _kernels() -> _SM100Kernels:
    # TRUE singleton, thread-safe: build the extensions exactly once per
    # process. Cross-process build coordination (concurrent launches, DDP
    # ranks) is handled by R3GAN.kernels.sm100._build via an fcntl lock and a
    # content-addressed cache. L2 routing is a runtime branch, never a rebuild.
    global _KERNELS
    if _KERNELS is None:
        with _KERNELS_MU:
            if _KERNELS is None:
                _KERNELS = _SM100Kernels(_L2_GCONV)
                return _KERNELS
    _KERNELS.l2_gconv = _L2_GCONV
    return _KERNELS


def kernel_status() -> Dict[str, Any]:
    return _kernels().symbols()


discover_kernel_symbols = kernel_status


# -----------------------------------------------------------------------------
# Autograd function
# -----------------------------------------------------------------------------

class _FusedFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                x: torch.Tensor,
                w1: torch.Tensor,
                b1: torch.Tensor,
                s1: torch.Tensor,
                w2: torch.Tensor,
                w3: torch.Tensor,
                alpha: float,
                groups: int,
                has_noise: bool):
        x = _require_nchw_cl(x, "ffn.x")
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        w3 = w3.contiguous()
        b1 = b1.float().contiguous()
        s1 = s1.float().contiguous()
        alpha = float(alpha)
        groups = int(groups)
        has_noise = bool(has_noise)

        if has_noise:
            n, _, h, w = x.shape
            noise_col = torch.randn((n * h * w,), device=x.device, dtype=torch.float32).contiguous()
            if s1.numel() == 0:
                raise RuntimeError("has_noise=True but s1 is empty")
        else:
            noise_col = torch.empty((0,), device=x.device, dtype=torch.float32)

        k = _kernels()
        a1 = k.l1(x, w1, b1, noise_col if has_noise else None, s1 if has_noise else None)
        a2 = k.l2_forward(a1, w2, groups, alpha)
        out = k.l3(a2, w3, x)

        ctx.alpha = alpha
        ctx.groups = groups
        ctx.has_noise = has_noise
        ctx.save_for_backward(x, w1, b1, s1, noise_col, a1, w2, a2, w3)
        # backward() ignores grads flowing into a1/a2 (only their SIGN is consumed
        # downstream, by explicit_vjp's Slope). Mark them so any other use errors
        # loudly instead of silently dropping gradients -- and disable grad
        # materialization: the engine otherwise ALLOCATES ZERO TENSORS of a1/a2's
        # shapes before every backward call just to fill the unused _ga1/_ga2
        # arguments (~21 ms/step of batch-sized zeros, trace-attributed).
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(a1, a2)
        return out, a1, a2

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor,
                 _ga1: Optional[torch.Tensor] = None, _ga2: Optional[torch.Tensor] = None):
        x, w1, b1, s1, noise_col, a1, w2, a2, w3 = ctx.saved_tensors
        alpha = float(ctx.alpha)
        groups = int(ctx.groups)
        has_noise = bool(ctx.has_noise)
        grad_out = _require_nchw_cl(grad_out, "ffn.grad_out")
        k = _kernels()

        # L3 backward. dx kernel fuses activation-2 slope when act=a2 is supplied.
        dz2 = k.dx(grad_out, w3, residual_nchw=None, act_nchw=a2)
        dw3 = (k.dw(grad_out, a2) if _DW_STREAMK else _mm_wgrad(grad_out, a2)).to(w3.dtype)

        # L2 weight grad via cuDNN/ATen (wide-group widened per _L2_WIDEN_WGRAD).
        dw2 = _grouped_wgrad(a1, w2.shape, dz2, groups).to(w2.dtype).contiguous()

        # L2 input grad with activation-1 slope fused (gconv32 epilogue).
        dz1 = k.l2_dgrad_slope1(dz2, w2, a1, groups, alpha)

        # L1 backward. dx kernel adds residual skip grad_out.
        dx = k.dx(dz1, w1, residual_nchw=grad_out, act_nchw=None)
        dw1 = (k.dw(dz1, x) if _DW_STREAMK else _mm_wgrad(dz1, x)).to(w1.dtype)

        if has_noise:
            n, _, h, w = x.shape
            m = n * h * w
            dz1_2d = _nchw_cl_to_nhwc(dz1, "dz1.for_db_ds").reshape(m, -1)
            # db1 and ds1 both reduce over the same dz1: db1[c] = sum_m dz1[m,c],
            # ds1[c] = sum_m noise[m]*dz1[m,c]. Stacking [ones; noise] into one
            # [2,M]@[M,C] GEMM computes both in a single pass over dz1. fp32
            # accumulation in cuBLAS; db1 is bf16-rounded at the store.
            # (Candidate F4b: fold these sums into the gconv32 dgrad epilogue
            # and delete this GEMM's dz1 read entirely.)
            lhs = torch.stack((_ones_row(m, dz1_2d.dtype, dz1_2d.device),
                               noise_col.to(dz1_2d.dtype)))
            both = lhs @ dz1_2d
            db1 = both[0].to(b1.dtype)
            ds1 = both[1].to(s1.dtype)
        else:
            db1 = _sum_channels_nchw(dz1).to(b1.dtype)
            ds1 = torch.zeros_like(s1) if s1.numel() else None

        return dx, dw1, db1, ds1, dw2, dw3, None, None, None


# -----------------------------------------------------------------------------
# Fused explicit vjp (R1 path)
# -----------------------------------------------------------------------------

class _FusedFFNVjp(torch.autograd.Function):
    """v_in = J_ffn(x)^T v_out using the SM100 kernels, differentiable wrt the
    effective weights (required: R1's dW flows through this op).

    Forward chain (slope sources are the cached post-activation tensors):
        u2  = (v @ W3) * s2                          dx kernel, act=y2
        t1  = grouped_dgrad(u2, W2) * s1             gconv32 dgrad, slope=y1
        out = (t1 @ W1) + v                          dx kernel, residual=v

    Backward, given g = dL/d(out) -- the mirror chain with transposed weights:
        p   = (g @ W1^T) * s1                        dx kernel, act=y1
        dW1 = t1^T @ g                               dw kernel / cuBLAS
        dW2 = conv2d_weight(input=p, grad_out=u2)    adjoint identity for convT
        q   = s2 * conv2d(p, W2, groups)             gconv32, slope(y2) fused
        dW3 = v^T @ q                                dw kernel / cuBLAS
        dv  = (q @ W3^T) + g                         dx kernel, residual=g

    Slope-at-zero conventions: the dx kernel and gconv32 use >= 0 -> 1 (matches
    the original Slope(), including at -0); the q step uses leaky_relu_backward
    (> 0). Differences exist only at exact bf16 zeros.
    """

    @staticmethod
    def forward(ctx: Any, v: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                w3: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor,
                alpha: float, groups: int):
        if abs(float(alpha) - 0.2) > 1e-6:
            raise RuntimeError(f"fused vjp requires alpha=0.2 (DX kernel compile-time slope), got {alpha}")
        v = _require_nchw_cl(v, "vjp.v")
        k = _kernels()
        u2 = k.dx(v, w3, residual_nchw=None, act_nchw=y2)
        t1 = k.l2_dgrad_slope1(u2, w2, y1, int(groups), float(alpha))
        out = k.dx(t1, w1, residual_nchw=v, act_nchw=None)
        ctx.alpha, ctx.groups = float(alpha), int(groups)
        ctx.save_for_backward(v, u2, t1, y1, y2, w1, w2, w3)
        return out

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor):
        v, u2, t1, y1, y2, w1, w2, w3 = ctx.saved_tensors
        alpha, groups = ctx.alpha, ctx.groups
        if not g.is_contiguous(memory_format=torch.channels_last):
            g = g.contiguous(memory_format=torch.channels_last)
        k = _kernels()

        p = k.dx(g, w1.t().contiguous(), residual_nchw=None, act_nchw=y1)

        _wg = k.dw if _DW_STREAMK else _mm_wgrad
        dw1 = _wg(t1, g).to(w1.dtype) if ctx.needs_input_grad[1] else None
        dw2 = (_grouped_wgrad(p, w2.shape, u2, groups)
               .to(w2.dtype).contiguous()) if ctx.needs_input_grad[2] else None

        q = k.l2_conv_slope(p, w2, y2, groups, alpha)

        dw3 = _wg(v, q).to(w3.dtype) if ctx.needs_input_grad[3] else None
        dv = k.dx(q, w3.t().contiguous(), residual_nchw=g, act_nchw=None) \
            if ctx.needs_input_grad[0] else None
        return dv, dw1, dw2, dw3, None, None, None, None


def _fused_explicit_vjp(self: Any, v_out: torch.Tensor, cache: Dict[str, Any],
                        KeepInputDependency: bool = False) -> torch.Tensor:
    """Drop-in for FeedForwardNetwork.explicit_vjp. Effective weights are
    computed exactly as the original (differentiable; R1 dW flows through
    them), then the chain runs on the fused kernels."""
    dtype = v_out.dtype
    v = v_out if v_out.is_contiguous(memory_format=torch.channels_last) \
        else v_out.contiguous(memory_format=torch.channels_last)

    w3 = _w2d_1x1(self.LinearLayer3.EffectiveWeight(
        dtype, Gain=self.NonLinearity.Gain * cache['residual_gain']))
    w2 = self.LinearLayer2.EffectiveWeight(dtype, Gain=self.NonLinearity.Gain).contiguous()
    w1_full, _ = self._pointwise_effective_weight_bias(self.LinearLayer1, dtype, cache['input_gain'])
    w1 = _w2d_1x1(w1_full)

    v_x = _FusedFFNVjp.apply(v, w1, w2, w3, cache['y1'], cache['y2'],
                             float(self.NonLinearity.α), int(self.LinearLayer2.Groups))
    if KeepInputDependency and 'x' in cache:
        v_x = v_x + cache['x'] * 0
    return v_x


# -----------------------------------------------------------------------------
# FeedForwardNetwork monkey patch
# -----------------------------------------------------------------------------

def _effective_tensors(layer: Any, x: torch.Tensor, InputGain: torch.Tensor, ResidualGain: torch.Tensor):
    input_gain = InputGain.view(1, -1, 1, 1)
    residual_gain = ResidualGain.view(-1, 1, 1, 1)
    alpha = float(layer.NonLinearity.α)

    has_noise = hasattr(layer.LinearLayer1, "EffectiveWeightBiasNoiseScale")
    if has_noise:
        w1, b1, s1 = layer.LinearLayer1.EffectiveWeightBiasNoiseScale(x.dtype, Gain=input_gain)
        s1 = s1.float().contiguous()
    else:
        w1, b1 = layer._pointwise_effective_weight_bias(layer.LinearLayer1, x.dtype, input_gain)
        s1 = torch.empty((0,), device=x.device, dtype=torch.float32)

    w2 = layer.LinearLayer2.EffectiveWeight(x.dtype, Gain=layer.NonLinearity.Gain)
    w3 = layer.LinearLayer3.EffectiveWeight(x.dtype, Gain=layer.NonLinearity.Gain * residual_gain)
    return (_w2d_1x1(w1), b1.float().contiguous(), s1, w2.contiguous(), _w2d_1x1(w3),
            alpha, int(layer.LinearLayer2.Groups), bool(has_noise))


def _fused_forward(self: Any, x: torch.Tensor, InputGain: torch.Tensor,
                   ResidualGain: torch.Tensor) -> torch.Tensor:
    if _FUSED_MIN_RES and min(x.shape[-2:]) < _FUSED_MIN_RES:
        return type(self)._r3gan_fused_ffn_originals["forward"](self, x, InputGain, ResidualGain)
    x = _require_nchw_cl(x, "FeedForwardNetwork.forward.x")
    w1, b1, s1, w2, w3, alpha, groups, has_noise = _effective_tensors(self, x, InputGain, ResidualGain)
    out, _, _ = _FusedFFN.apply(x, w1, b1, s1, w2, w3, alpha, groups, has_noise)
    return out


def _fused_forward_with_cache(self: Any, x: torch.Tensor, InputGain: torch.Tensor,
                              ResidualGain: torch.Tensor):
    if _FUSED_MIN_RES and min(x.shape[-2:]) < _FUSED_MIN_RES:
        return type(self)._r3gan_fused_ffn_originals["forward_with_cache"](self, x, InputGain, ResidualGain)
    x = _require_nchw_cl(x, "FeedForwardNetwork.forward_with_cache.x")
    w1, b1, s1, w2, w3, alpha, groups, has_noise = _effective_tensors(self, x, InputGain, ResidualGain)
    out, a1, a2 = _FusedFFN.apply(x, w1, b1, s1, w2, w3, alpha, groups, has_noise)
    cache = dict(
        Layer=self,
        y1=a1,
        y2=a2,
        input_gain=InputGain.view(1, -1, 1, 1),
        residual_gain=ResidualGain.view(-1, 1, 1, 1),
    )
    return out, cache


def install_fused_ffn(FeedForwardNetwork: type, *, enable: bool = True,
                      l2_gconv: Optional[bool] = None, vjp: Optional[bool] = None) -> type:
    """l2_gconv: True (default via R3GAN_L2_GCONV) -> L2 via the gconv32
    kernels; False -> wide-group block-diagonal cuDNN.
    vjp: True -> FeedForwardNetwork.explicit_vjp (R1 path) runs on the fused
    kernels via _FusedFFNVjp; None (default) -> R3GAN_FUSED_VJP env, default
    OFF. Run test_fused_vjp.py before enabling."""
    global _L2_GCONV, _VJP_FUSED
    if l2_gconv is not None:
        _L2_GCONV = bool(l2_gconv)
    if vjp is not None:
        _VJP_FUSED = bool(vjp)
    saved = getattr(FeedForwardNetwork, "_r3gan_fused_ffn_originals", None)
    if enable:
        # Force import/build now. If backward symbols are not bound, fail before
        # training starts.
        _kernels()
        if saved is None:
            FeedForwardNetwork._r3gan_fused_ffn_originals = {
                "forward": FeedForwardNetwork.forward,
                "forward_with_cache": FeedForwardNetwork.forward_with_cache,
                "explicit_vjp": FeedForwardNetwork.explicit_vjp,
            }
        FeedForwardNetwork.forward = _fused_forward
        FeedForwardNetwork.forward_with_cache = _fused_forward_with_cache
        FeedForwardNetwork.explicit_vjp = (
            _fused_explicit_vjp if _VJP_FUSED
            else FeedForwardNetwork._r3gan_fused_ffn_originals["explicit_vjp"])
        FeedForwardNetwork._r3gan_fused_ffn_enabled = True
    else:
        if saved is not None:
            FeedForwardNetwork.forward = saved["forward"]
            FeedForwardNetwork.forward_with_cache = saved["forward_with_cache"]
            FeedForwardNetwork.explicit_vjp = saved["explicit_vjp"]
        FeedForwardNetwork._r3gan_fused_ffn_enabled = False
    return FeedForwardNetwork


def fused_ffn_is_installed(FeedForwardNetwork: type) -> bool:
    return bool(getattr(FeedForwardNetwork, "_r3gan_fused_ffn_enabled", False))
