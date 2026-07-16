# Strict modern R3GAN resampling plugin.
# Optimized NHWC kernels, strict FP32 gain boundaries, no outdiv/output reconstruction.
# FP32-base-reuse rung: fused output-gain backward computes dx/ddy and dGain from one FP32 base.

import os
import numpy as np
import torch
from .. import custom_ops
from .. import misc
from . import upfirdn2d_gain_fp32_strict as _stylegan_strict

_plugin = None

# Clean production mode table.  Debug/search modes from the exploratory plugin are intentionally not exposed.
# All gain modes apply channel_gain in FP32 inside CUDA.  Int64 variants use the same tiling
# choices as the int32 variants, but switch tensor addressing to int64.
MODES = {
    # Fast upsample path: VecC=4, float4 shared tile, cached FP32 gain.
    'up_vec4s_32x16_i32_gain': 100,       # best for <=4x4 filters
    'up_vec4s_16x32_b512_i32_gain': 101,  # best for >=5x5 filters
    'up_vec4s_32x16_i64_gain': 102,
    'up_vec4s_16x32_b512_i64_gain': 103,

    # Fast downsample path: scalar shared tile, cached FP32 gain.
    'down_tile8x8_i32_gain': 104,
    'down_tile8x8_i64_gain': 105,

    # Fast explicit-VJP/output-gain paths.
    'up_vec4s_32x16_i32_outgain': 106,
    'up_vec4s_16x32_b512_i32_outgain': 107,
    'up_vec4s_32x16_i64_outgain': 108,
    'up_vec4s_16x32_b512_i64_outgain': 109,
    'down_tile8x8_i32_outgain': 110,
    'down_tile8x8_i64_outgain': 111,

    # No-gain paths, used when channel_gain is absent.
    'up_shared_i32_nogain': 112,
    'up_shared_i64_nogain': 113,
    'down_shared_i32_nogain': 114,
    'down_shared_i64_nogain': 115,

    # Generic shared-memory fallback for unsupported GPU/layout/C-tail cases.
    'up_shared_i32_gain': 116,
    'up_shared_i64_gain': 117,
    'down_shared_i32_gain': 118,
    'down_shared_i64_gain': 119,
    'up_shared_i32_outgain': 120,
    'up_shared_i64_outgain': 121,
    'down_shared_i32_outgain': 122,
    'down_shared_i64_outgain': 123,
}

MODE_DESCRIPTIONS = {k: k for k in MODES}

def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='resample_r3gan_modern_strict_plugin',
            sources=['resample_r3gan_modern_strict.cpp', 'resample_r3gan_modern_strict.cu'],
            headers=['resample_r3gan_modern_strict.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--allow-unsupported-compiler'],
        )
    return True

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    sx, sy = scaling
    assert isinstance(sx, int) and isinstance(sy, int) and sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    assert len(padding) == 4
    return tuple(int(x) for x in padding)

def _get_filter_size(f):
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw, fh = f.shape[-1], f.shape[0]
    with misc.suppress_tracer_warnings():
        fw, fh = int(fw), int(fh)
    assert fw >= 1 and fh >= 1
    return fw, fh

def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2] and f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    return f.to(device=device)

def _mode_id(mode):
    if isinstance(mode, str):
        if mode not in MODES:
            raise ValueError(f'unknown mode {mode!r}; choices={list(MODES)}')
        return MODES[mode]
    return int(mode)

def _prepare_gain(channel_gain, device):
    if channel_gain is None:
        return torch.empty([0], dtype=torch.float32, device=device)
    assert isinstance(channel_gain, torch.Tensor)
    # Differentiable: gradients returned to this prepared tensor flow through
    # the cast/reshape/contiguous copy to the parameters that produced gain.
    return channel_gain.to(torch.float32).reshape(-1).contiguous()

def _ensure_filter2d(f, device):
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=device)
    if f.ndim == 1:
        f = f.ger(f).to(device=device, dtype=torch.float32)
    else:
        f = f.to(device=device, dtype=torch.float32)
    assert f.ndim == 2 and f.dtype == torch.float32
    return f

def _mode_is_no_gain(mode):
    mid = _mode_id(mode)
    return mid in (112, 113, 114, 115)

def _is_up2(up, down):
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    return upx == 2 and upy == 2 and downx == 1 and downy == 1

def _is_down2(up, down):
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    return upx == 1 and upy == 1 and downx == 2 and downy == 2

def _storage_footprint_fits_int32(t):
    # Exact same criterion as the C++ launcher: all storage offsets touched by the tensor
    # must fit signed int32.  Individual dimensions may be small while the footprint is large.
    max_off = 0
    for size, stride in zip(t.shape, t.stride()):
        if int(size) > 0:
            max_off += (int(size) - 1) * int(stride)
    return max_off <= (2 ** 31 - 1)

def _use_int64_addressing(x):
    return not _storage_footprint_fits_int32(x)

def _supports_vec4s_fast(x):
    # Vec4s uses only ordinary CUDA/BF16 bit packing, not arch-specific inline asm, so it is
    # usable on Ada/Hopper/Blackwell/future Blackwell (sm89/sm90/sm100/sm120) when compiled
    # for that device.  Keep a conservative fallback for older devices or odd channel counts.
    if x.device.type != 'cuda' or x.stride(1) != 1 or x.shape[1] % 32 != 0:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    major, minor = torch.cuda.get_device_capability(x.device)
    sm = major * 10 + minor
    return sm >= 89 or os.environ.get('MODERN_RESAMPLE_FORCE_VEC4S', '0') == '1'

def _auto_mode(x, f, up=2, down=1, channel_gain=None, no_gain=False):
    use_i64 = _use_int64_addressing(x)
    i = 'i64' if use_i64 else 'i32'
    if _is_up2(up, down):
        if no_gain or channel_gain is None:
            return f'up_shared_{i}_nogain'
        if _supports_vec4s_fast(x):
            fw, fh = _get_filter_size(f)
            if fw <= 4 and fh <= 4:
                return f'up_vec4s_32x16_{i}_gain'
            return f'up_vec4s_16x32_b512_{i}_gain'
        return f'up_shared_{i}_gain'
    if _is_down2(up, down):
        if no_gain or channel_gain is None:
            return f'down_shared_{i}_nogain'
        return f'down_tile8x8_{i}_gain' if _supports_vec4s_fast(x) else f'down_shared_{i}_gain'
    raise NotImplementedError('modern resample handles 2x upsample and 2x downsample')


def _auto_output_gain_mode(x, f, up=1, down=1):
    use_i64 = _use_int64_addressing(x)
    i = 'i64' if use_i64 else 'i32'
    if _is_up2(up, down):
        if _supports_vec4s_fast(x):
            fw, fh = _get_filter_size(f)
            if fw <= 4 and fh <= 4:
                return f'up_vec4s_32x16_{i}_outgain'
            return f'up_vec4s_16x32_b512_{i}_outgain'
        return f'up_shared_{i}_outgain'
    if _is_down2(up, down):
        return f'down_tile8x8_{i}_outgain' if _supports_vec4s_fast(x) else f'down_shared_{i}_outgain'
    raise NotImplementedError('output-gain reverse mode only handles 2x upsample/downsample')

def _call_plugin(x, f, up=2, down=1, padding=0, flip_filter=False, gain=4, channel_gain=None, mode='auto'):
    assert isinstance(x, torch.Tensor) and x.ndim == 4 and x.device.type == 'cuda'
    assert x.stride(1) == 1, 'modern resample plugin requires channels-last/NHWC input'
    f = _ensure_filter2d(f, x.device)
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    if mode == 'auto':
        mode = _auto_mode(x, f, up=up, down=down, channel_gain=channel_gain, no_gain=(channel_gain is None))
    _init()
    cg = _prepare_gain(channel_gain, x.device)
    return _plugin.resample_r3gan_modern_debug(
        x, f, cg, upx, upy, downx, downy, padx0, padx1, pady0, pady1,
        bool(flip_filter), float(gain), _mode_id(mode))

# Backward-compatible raw debug entry point.
def upfirdn2d_debug(x, f, up=2, down=1, padding=0, flip_filter=False, gain=4, channel_gain=None, mode='auto'):
    return _call_plugin(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain, channel_gain=channel_gain, mode=mode)

_resample_modern_cuda_cache = {}

def _resample_modern_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1, mode='auto', reverse_mode='auto'):
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, bool(flip_filter), float(gain), str(mode), str(reverse_mode))
    if key in _resample_modern_cuda_cache:
        return _resample_modern_cuda_cache[key]

    class ResampleModernCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f, channel_gain):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            f = _ensure_filter2d(f, x.device)
            empty_gain = torch.empty([0], dtype=torch.float32, device=x.device)
            has_channel_gain = channel_gain is not None and channel_gain.numel() > 0
            if has_channel_gain:
                assert channel_gain.dtype == torch.float32
                assert channel_gain.device == x.device
                assert channel_gain.numel() in [1, x.shape[1]]
            cg = channel_gain if has_channel_gain else None
            y = _call_plugin(x, f, up=[upx, upy], down=[downx, downy], padding=[padx0, padx1, pady0, pady1], flip_filter=flip_filter, gain=gain, channel_gain=cg, mode=mode)
            ctx.save_for_backward(f, channel_gain if has_channel_gain else empty_gain, x)
            ctx.has_channel_gain = has_channel_gain
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            return y

        @staticmethod
        def backward(ctx, dy):
            f, channel_gain, x = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None
            dchannel_gain = None
            need_dx = ctx.needs_input_grad[0]
            need_dgain = ctx.has_channel_gain and ctx.needs_input_grad[2]
            if need_dx or need_dgain:
                dy_cl = dy.contiguous(memory_format=torch.channels_last)
                if ctx.has_channel_gain:
                    # Strict FP32 boundary:
                    #   dx = cast( R^T(dy)_fp32 * gain_fp32 )
                    # must not be implemented as cast(R^T(dy)) followed by another
                    # gain multiply, because that inserts an extra BF16/FP16 boundary.
                    rev_mode = _auto_output_gain_mode(dy_cl, f, up=[downx, downy], down=[upx, upy]) if reverse_mode == 'auto' else reverse_mode
                    if need_dx and need_dgain and _can_fuse_outgain_both(dy_cl, x, channel_gain):
                        # Fused strict FP32-base reuse:
                        #   base = R^T(dy)_fp32 * scalar_gain
                        #   dx   = cast(base * channel_gain)
                        #   dg   = sum_fp32(x * base)
                        # The base is never stored as BF16/FP16 or as an activation-sized FP32 tensor.
                        dx, dchannel_gain = _outgain_both_cuda(
                            dy_cl, x, f,
                            up=[downx, downy], down=[upx, upy], padding=p,
                            flip_filter=(not flip_filter), gain=gain,
                            channel_gain=channel_gain, mode=rev_mode)
                        dchannel_gain = dchannel_gain.reshape_as(channel_gain)
                    else:
                        if need_dx:
                            dx = _call_plugin(dy_cl, f,
                                up=[downx, downy], down=[upx, upy], padding=p,
                                flip_filter=(not flip_filter), gain=gain, channel_gain=channel_gain, mode=rev_mode)
                        if need_dgain:
                            # Strict fallback: recompute R^T(dy) inside the reduction in FP32.
                            dchannel_gain = _strict_channel_gain_grad(
                                dy_cl, x, f,
                                up=[downx, downy], down=[upx, upy], padding=p,
                                flip_filter=(not flip_filter), gain=gain,
                                gain_numel=int(channel_gain.numel()),
                            ).reshape_as(channel_gain)
                else:
                    if need_dx:
                        dx = _call_plugin(dy_cl, f,
                            up=[downx, downy], down=[upx, upy], padding=p,
                            flip_filter=(not flip_filter), gain=gain, channel_gain=None, mode=reverse_mode)
            assert not ctx.needs_input_grad[1]
            return dx, df, dchannel_gain

    _resample_modern_cuda_cache[key] = ResampleModernCuda
    return ResampleModernCuda

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda', channel_gain=None, mode='auto', reverse_mode='auto'):
    assert isinstance(x, torch.Tensor)
    assert impl in ['cuda']
    f = _ensure_filter2d(f, x.device)
    cg = None if channel_gain is None else _prepare_gain(channel_gain, x.device)
    return _resample_modern_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain, mode=mode, reverse_mode=reverse_mode).apply(x, f, cg)

def filter2d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda', channel_gain=None, mode='auto', reverse_mode='auto'):
    f = _ensure_filter2d(f, x.device)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn2d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl, channel_gain=channel_gain, mode=mode, reverse_mode=reverse_mode)

def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda', channel_gain=None, mode='auto', reverse_mode='auto'):
    f = _ensure_filter2d(f, x.device)
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain * upx * upy, impl=impl, channel_gain=channel_gain, mode=mode, reverse_mode=reverse_mode)

def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda', channel_gain=None, mode='auto', reverse_mode='auto'):
    f = _ensure_filter2d(f, x.device)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl, channel_gain=channel_gain, mode=mode, reverse_mode=reverse_mode)



_outgain_vjp_cuda_cache = {}

def _gain_grad_cuda(base, other, gain_numel):
    _init()
    # Production default: coalesced32 for strict NHWC vector gain, generic fallback otherwise.
    # The C++ wrapper chooses the fallback when scalar gain/layout does not match.
    return _plugin.resample_r3gan_modern_gain_grad(base, other, int(gain_numel))

def _apply_gain_cuda(x, channel_gain):
    # FP32 gain multiply with exactly one output quantization boundary.  Kept for
    # debug use only; strict production backward uses output-gain kernels to avoid
    # materializing/casting the no-gain base before the gain multiply.
    _init()
    cg = _prepare_gain(channel_gain, x.device)
    return _plugin.resample_r3gan_modern_apply_gain(x.contiguous(memory_format=torch.channels_last), cg)


def _strict_channel_gain_grad(dy, x, f, up, down, padding, flip_filter, gain, gain_numel):
    """Compute dGain = sum x * R^T(dy) with R^T accumulated in FP32.

    This deliberately reuses the strict StyleGAN-derived fused reduction kernel.
    It avoids the invalid shortcut

        base_bf16 = cast(R^T(dy))
        dGain = sum x * base_bf16

    which inserts an extra BF16/FP16 quantization boundary and produced ~1e-3
    relative errors in the optimization rung.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    f = _ensure_filter2d(f, dy.device)
    _stylegan_strict._init()
    return _stylegan_strict._plugin.channel_gain_grad(
        dy, x, f,
        upx, upy, downx, downy,
        padx0, padx1, pady0, pady1,
        bool(flip_filter), float(gain), int(gain_numel),
    )


def _can_fuse_outgain_both(inp, other, channel_gain):
    return (
        isinstance(inp, torch.Tensor) and isinstance(other, torch.Tensor)
        and inp.device.type == 'cuda' and other.device == inp.device
        and inp.dtype == other.dtype
        and inp.ndim == 4 and other.ndim == 4
        and inp.stride(1) == 1 and other.stride(1) == 1
        and channel_gain is not None and channel_gain.dtype == torch.float32
        and channel_gain.device == inp.device
        and channel_gain.numel() == other.shape[1]
    )


def _outgain_both_cuda(inp, other, f, up, down, padding, flip_filter, gain, channel_gain, mode):
    """Compute output-gain resample and dGain from one FP32 base.

    Returns (out, dchannel_gain).  Inside CUDA, each output element computes
        base_fp32 = R(inp)_fp32 * scalar_gain
    once, stores
        out = cast(base_fp32 * channel_gain)
    and accumulates
        dchannel_gain += other * base_fp32
    without materializing base_fp32 as a tensor.  This is valid only for vector
    gain and strict NHWC tensors; callers fall back to the recompute path for
    all other cases.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    f = _ensure_filter2d(f, inp.device)
    _init()
    out, dgain = _plugin.resample_r3gan_modern_outgain_both(
        inp, other, f, channel_gain,
        upx, upy, downx, downy,
        padx0, padx1, pady0, pady1,
        bool(flip_filter), float(gain), _mode_id(mode))
    return out, dgain


def _explicit_vjp_outgain_cuda(input_shape, output_shape, up=1, down=1, padding=0, flip_filter=False, gain=1, mode='auto', forward_mode='auto'):
    """Differentiable fast explicit VJP for y = R(x * gain_vec).

    Forward computes v_in = cast(R^T(v_out)_fp32 * gain_vec) using output-gain
    CUDA modes.  Backward wrt v_out must also use output-gain semantics,
    ddy = cast(R(grad_vin)_fp32 * gain_vec), not input-side gain semantics.
    When dy and gain gradients are both needed, the gain derivative reuses the same FP32 base without materializing it; otherwise it falls back to strict recompute. No outdiv.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    input_shape = tuple(int(v) for v in input_shape)
    output_shape = tuple(int(v) for v in output_shape)
    key = (input_shape, output_shape, upx, upy, downx, downy, padx0, padx1, pady0, pady1, bool(flip_filter), float(gain), str(mode), str(forward_mode))
    if key in _outgain_vjp_cuda_cache:
        return _outgain_vjp_cuda_cache[key]

    _, _, ih, iw = input_shape
    _, _, oh, ow = output_shape

    class ExplicitVJPOutgainCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dy, f, channel_gain):
            assert isinstance(dy, torch.Tensor) and dy.ndim == 4
            f = _ensure_filter2d(f, dy.device)
            assert channel_gain is not None and channel_gain.numel() > 0
            assert channel_gain.dtype == torch.float32 and channel_gain.device == dy.device
            assert channel_gain.numel() in [1, input_shape[1]]
            fw, fh = _get_filter_size(f)
            reverse_padding = [
                fw - padx0 - 1,
                iw * upx - oh * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dy_cl = dy.contiguous(memory_format=torch.channels_last)
            rev_mode = _auto_output_gain_mode(dy_cl, f, up=[downx, downy], down=[upx, upy]) if mode == 'auto' else mode
            out = _call_plugin(dy_cl, f,
                up=[downx, downy], down=[upx, upy], padding=reverse_padding,
                flip_filter=(not flip_filter), gain=gain, channel_gain=channel_gain, mode=rev_mode)
            # Strict rung: do not use out/base reconstruction.  The backward
            # wrt channel_gain recomputes the no-gain base = R^T(dy), so the
            # gain derivative is arithmetically the exact FP32-gain VJP gradient
            # modulo ordinary FP32 reduction/evaluation order.
            ctx.save_for_backward(dy_cl, f, channel_gain)
            ctx.reverse_padding = reverse_padding
            return out

        @staticmethod
        def backward(ctx, grad_vin):
            dy_cl, f, channel_gain = ctx.saved_tensors
            reverse_padding = ctx.reverse_padding
            grad_vin_cl = grad_vin.contiguous(memory_format=torch.channels_last)
            ddy = None
            df = None
            dchannel_gain = None
            # Important: ctx.needs_input_grad reflects which forward inputs had
            # requires_grad=True.  To get a true gain-only/dy-only benchmark, the
            # other input must be detached in the benchmark graph.  In real R1,
            # both will usually be true.
            need_dy = ctx.needs_input_grad[0]
            need_dgain = ctx.needs_input_grad[2]

            fwd_outgain_mode = (_auto_output_gain_mode(
                grad_vin_cl, f, up=[upx, upy], down=[downx, downy]
            ) if forward_mode == 'auto' else forward_mode)

            if need_dy and need_dgain and _can_fuse_outgain_both(grad_vin_cl, dy_cl, channel_gain):
                # Fused strict FP32-base reuse for the higher-order VJP path:
                #   base = R(grad_vin)_fp32 * scalar_gain
                #   ddy  = cast(base * channel_gain)
                #   dg   = sum_fp32(dy * base)
                ddy, dchannel_gain = _outgain_both_cuda(
                    grad_vin_cl, dy_cl, f,
                    up=[upx, upy], down=[downx, downy], padding=[padx0, padx1, pady0, pady1],
                    flip_filter=flip_filter, gain=gain,
                    channel_gain=channel_gain, mode=fwd_outgain_mode)
                dchannel_gain = dchannel_gain.reshape_as(channel_gain)
            else:
                if need_dy:
                    # Backward wrt VJP input dy must preserve the strict FP32 island:
                    #   explicit VJP forward: out = cast(R^T(dy)_fp32 * gain_fp32)
                    #   adjoint wrt dy:       ddy = cast(R(grad_vin)_fp32 * gain_fp32)
                    ddy = _call_plugin(grad_vin_cl, f,
                        up=[upx, upy], down=[downx, downy], padding=[padx0, padx1, pady0, pady1],
                        flip_filter=flip_filter, gain=gain, channel_gain=channel_gain, mode=fwd_outgain_mode)

                if need_dgain:
                    # Strict fallback: recompute the no-gain base in FP32.  No outdiv.
                    dchannel_gain = _strict_channel_gain_grad(
                        dy_cl, grad_vin_cl, f,
                        up=[downx, downy], down=[upx, upy], padding=reverse_padding,
                        flip_filter=(not flip_filter), gain=gain,
                        gain_numel=int(channel_gain.numel()),
                    ).reshape_as(channel_gain)

            assert not ctx.needs_input_grad[1]
            return ddy, df, dchannel_gain

    _outgain_vjp_cuda_cache[key] = ExplicitVJPOutgainCuda
    return ExplicitVJPOutgainCuda
def explicit_vjp(dy, f, input_shape, output_shape=None, up=1, down=1, padding=0, flip_filter=False, gain=1, channel_gain=None, mode='auto'):
    # Input-side VJP for y = R(x * channel_gain).  This mirrors the original
    # StyleGAN backward padding formula, but routes the spatial adjoint through
    # this plugin instead of the old upfirdn2d plugin.
    f = _ensure_filter2d(f, dy.device)
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    _, _, ih, iw = input_shape
    if output_shape is None:
        output_shape = dy.shape
    _, _, oh, ow = output_shape
    fw, fh = _get_filter_size(f)
    p = [
        fw - padx0 - 1,
        iw * upx - ow * downx + padx0 - upx + 1,
        fh - pady0 - 1,
        ih * upy - oh * downy + pady0 - upy + 1,
    ]
    dy_cl = dy.contiguous(memory_format=torch.channels_last)
    if channel_gain is not None:
        cg = _prepare_gain(channel_gain, dy.device)
        return _explicit_vjp_outgain_cuda(
            input_shape=input_shape, output_shape=output_shape,
            up=[upx, upy], down=[downx, downy], padding=[padx0, padx1, pady0, pady1],
            flip_filter=flip_filter, gain=gain, mode=mode, forward_mode='auto',
        ).apply(dy_cl, f, cg)
    return _call_plugin(dy_cl, f,
        up=[downx, downy], down=[upx, upy], padding=p,
        flip_filter=(not flip_filter), gain=gain, channel_gain=None, mode=mode)

# Raw debug convenience wrappers.
def upsample2d_debug(x, f, channel_gain=None, mode='auto'):
    f = _ensure_filter2d(f, x.device)
    fw, fh = _get_filter_size(f)
    p = [(fw + 2 - 1) // 2, (fw - 2) // 2, (fh + 2 - 1) // 2, (fh - 2) // 2]
    return upfirdn2d_debug(x, f, up=2, down=1, padding=p, flip_filter=False, gain=4, channel_gain=channel_gain, mode=mode)

def downsample2d_debug(x, f, channel_gain=None, mode='auto'):
    f = _ensure_filter2d(f, x.device)
    fw, fh = _get_filter_size(f)
    p = [(fw - 2 + 1) // 2, (fw - 2) // 2, (fh - 2 + 1) // 2, (fh - 2) // 2]
    return upfirdn2d_debug(x, f, up=1, down=2, padding=p, flip_filter=False, gain=1, channel_gain=channel_gain, mode=mode)


def gain_grad_debug(base_dx, x, gain_numel, impl='auto'):
    return _gain_grad_cuda(base_dx, x, int(gain_numel))
