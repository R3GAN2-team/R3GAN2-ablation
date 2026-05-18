import torch
import torch.nn as nn
import numpy
from torch_utils.ops import upfirdn2d


def CreateLowpassKernel(Weights, Inplace):
    Kernel = numpy.array([Weights]) if Inplace else numpy.convolve(Weights, [1, 1]).reshape(1, -1)
    Kernel = torch.Tensor(Kernel.T @ Kernel)
    return Kernel / torch.sum(Kernel)


# ---------------------------------------------------------------------------
# Small helpers mirroring torch_utils.ops.upfirdn2d internals.
# These are used only for explicit input-side VJPs. Forward behavior below is
# unchanged.
# ---------------------------------------------------------------------------

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    assert len(padding) == 4
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _get_filter_size(f):
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = int(f.shape[-1])
    fh = int(f.shape[0])
    assert fw >= 1 and fh >= 1
    return fw, fh


def _upsample2d_internal_padding(f, up=2, padding=0):
    # Matches torch_utils.ops.upfirdn2d.upsample2d().
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    return [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]


def _downsample2d_internal_padding(f, down=2, padding=0):
    # Matches torch_utils.ops.upfirdn2d.downsample2d().
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    return [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]


def _upfirdn2d_explicit_vjp(dy, f, x_shape, y_shape, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Explicit input-side VJP for StyleGAN upfirdn2d.

    This mirrors the exact padding formula used in torch_utils.ops.upfirdn2d's
    custom autograd backward implementation.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    _, _, ih, iw = x_shape
    _, _, oh, ow = y_shape
    fw, fh = _get_filter_size(f)

    p = [
        fw - padx0 - 1,
        iw * upx - ow * downx + padx0 - upx + 1,
        fh - pady0 - 1,
        ih * upy - oh * downy + pady0 - upy + 1,
    ]

    return upfirdn2d.upfirdn2d(
        dy,
        f,
        up=down,
        down=up,
        padding=p,
        flip_filter=(not flip_filter),
        gain=gain,
    )


class InterpolativeUpsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeUpsamplerReference, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
        self.FilterRadius = len(Filter) // 2

    def forward(self, x):
        Kernel = 4 * self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        y = nn.functional.conv_transpose2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=2, padding=self.FilterRadius)
        return y.view(x.shape[0], x.shape[1], y.shape[2], y.shape[3])

    def explicit_vjp(self, v, input_shape):
        b, c, h, w = input_shape
        Kernel = 4 * self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(v.dtype)
        y = nn.functional.conv2d(v.view(v.shape[0] * v.shape[1], 1, v.shape[2], v.shape[3]), Kernel, stride=2, padding=self.FilterRadius)
        return y.view(b, c, h, w)


class InterpolativeDownsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeDownsamplerReference, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
        self.FilterRadius = len(Filter) // 2

    def forward(self, x):
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        y = nn.functional.conv2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=2, padding=self.FilterRadius)
        return y.view(x.shape[0], x.shape[1], y.shape[2], y.shape[3])

    def explicit_vjp(self, v, input_shape):
        b, c, h, w = input_shape
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(v.dtype)
        y = nn.functional.conv_transpose2d(v.view(v.shape[0] * v.shape[1], 1, v.shape[2], v.shape[3]), Kernel, stride=2, padding=self.FilterRadius)
        return y.view(b, c, h, w)


class InplaceUpsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InplaceUpsamplerReference, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2

    def forward(self, x):
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        x = nn.functional.pixel_shuffle(x, 2)
        return nn.functional.conv2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=1, padding=self.FilterRadius).view(*x.shape)

    def explicit_vjp(self, v, input_shape):
        b, c, h, w = input_shape
        shuffled_shape = (b, c // 4, h * 2, w * 2)
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(v.dtype)
        z = nn.functional.conv_transpose2d(v.view(v.shape[0] * v.shape[1], 1, v.shape[2], v.shape[3]), Kernel, stride=1, padding=self.FilterRadius).view(*shuffled_shape)
        return nn.functional.pixel_unshuffle(z, 2)


class InplaceDownsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InplaceDownsamplerReference, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2

    def forward(self, x):
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        y = nn.functional.conv2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=1, padding=self.FilterRadius).view(*x.shape)
        return nn.functional.pixel_unshuffle(y, 2)

    def explicit_vjp(self, v, input_shape):
        z = nn.functional.pixel_shuffle(v, 2)
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(v.dtype)
        y = nn.functional.conv_transpose2d(z.view(z.shape[0] * z.shape[1], 1, z.shape[2], z.shape[3]), Kernel, stride=1, padding=self.FilterRadius).view(*input_shape)
        return y


class InterpolativeUpsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeUpsamplerCUDA, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))

    def forward(self, x):
        return upfirdn2d.upsample2d(x, self.Kernel)

    def explicit_vjp(self, v, input_shape):
        padding = _upsample2d_internal_padding(self.Kernel, up=2, padding=0)
        return _upfirdn2d_explicit_vjp(v, self.Kernel, x_shape=input_shape, y_shape=v.shape, up=2, down=1, padding=padding, flip_filter=False, gain=4)


class InterpolativeDownsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeDownsamplerCUDA, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))

    def forward(self, x):
        return upfirdn2d.downsample2d(x, self.Kernel)

    def explicit_vjp(self, v, input_shape):
        padding = _downsample2d_internal_padding(self.Kernel, down=2, padding=0)
        return _upfirdn2d_explicit_vjp(v, self.Kernel, x_shape=input_shape, y_shape=v.shape, up=1, down=2, padding=padding, flip_filter=False, gain=1)


class InplaceUpsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InplaceUpsamplerCUDA, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2

    def forward(self, x):
        return upfirdn2d.upfirdn2d(nn.functional.pixel_shuffle(x, 2), self.Kernel, padding=self.FilterRadius)

    def explicit_vjp(self, v, input_shape):
        b, c, h, w = input_shape
        shuffled_shape = (b, c // 4, h * 2, w * 2)
        z = _upfirdn2d_explicit_vjp(v, self.Kernel, x_shape=shuffled_shape, y_shape=v.shape, up=1, down=1, padding=self.FilterRadius, flip_filter=False, gain=1)
        return nn.functional.pixel_unshuffle(z, 2)


class InplaceDownsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InplaceDownsamplerCUDA, self).__init__()
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2

    def forward(self, x):
        return nn.functional.pixel_unshuffle(upfirdn2d.upfirdn2d(x, self.Kernel, padding=self.FilterRadius), 2)

    def explicit_vjp(self, v, input_shape):
        z = nn.functional.pixel_shuffle(v, 2)
        return _upfirdn2d_explicit_vjp(z, self.Kernel, x_shape=input_shape, y_shape=z.shape, up=1, down=1, padding=self.FilterRadius, flip_filter=False, gain=1)


InterpolativeUpsampler = InterpolativeUpsamplerCUDA
InterpolativeDownsampler = InterpolativeDownsamplerCUDA
InplaceUpsampler = InplaceUpsamplerCUDA
InplaceDownsampler = InplaceDownsamplerCUDA

