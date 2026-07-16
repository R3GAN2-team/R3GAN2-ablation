# R3GAN/kernels: architecture dispatch for the fused FFN.
#
# `import R3GAN.kernels` is torch-free; the fused_ffn re-exports below are
# resolved lazily (PEP 562) so cache maintenance (clear/prune, prebuild CLI,
# tests) works on nodes without torch installed.
from ._build import clear_kernel_cache, prune_kernel_cache  # noqa: F401

_FUSED_FFN_EXPORTS = ("install_fused_ffn", "fused_ffn_is_installed", "kernel_status")

__all__ = ["clear_kernel_cache", "prune_kernel_cache", *_FUSED_FFN_EXPORTS]


def __getattr__(name):
    if name in _FUSED_FFN_EXPORTS:
        from . import fused_ffn
        return getattr(fused_ffn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
