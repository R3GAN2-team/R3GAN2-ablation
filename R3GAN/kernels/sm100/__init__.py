# R3GAN/kernels/sm100: B200 kernel implementations.
#   gemm_forward.FFNForward      L1 / L1N / L3 fused 1x1 GEMMs (EVT epilogues)
#   gemm_backward.FFNBackward    dx (residual/slope epilogues) and dw GEMMs
#   grouped_conv                 L2 grouped 3x3 (fused lrelu fprop + bitmap,
#                                dgrad with fused activation-1 slope)
# Imports are intentionally lazy (each module JIT-compiles on first *use*,
# via the shared cached builder in ../_build.py; `python -m
# R3GAN.kernels.prebuild` compiles everything ahead of time).


def prebuild_targets():
    """(label, build_fn(verbose)) pairs consumed by R3GAN.kernels.prebuild."""
    from . import gemm_backward, gemm_forward, grouped_conv  # torch import deferred
    return (
        ("grouped_conv_sm100", lambda v: grouped_conv.build_extension(verbose=v)),
        ("gemm_forward_sm100", lambda v: gemm_forward.build_extension(verbose=v)[0]),
        ("gemm_backward_sm100", lambda v: gemm_backward.build_extension(verbose=v)),
    )
