# R3GAN/kernels/sm100/gemm_backward.py
"""SM100 FFN 1x1 backward GEMMs (dx with fused residual/slope epilogue, dw).

The CUDA config file (gemm_backward_configs.cu) exports
    dx_256x128_2x1, dx_256x256_2x1, dx_256x256_2x2
    dw_256x256_2sm, dw_256x128_2sm, dw_256x64_2sm
via gemm_backward_binding.cu; this wrapper makes them Python-callable and
picks a tile per shape.

Inputs are logical NHWC contiguous tensors, matching gemm_forward.py and
grouped_conv.py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .._build import (KernelSpec, cutlass_fingerprint, ensure_extension,
                     resolve_cutlass)

_THIS_DIR = Path(__file__).resolve().parent
_EXT = None


def _cutlass_include_dirs(root: str) -> list[str]:
    dirs: list[str] = [str(_THIS_DIR)]
    r = Path(root)
    for p in (r / "include", r / "tools" / "util" / "include", r):
        if p.exists():
            dirs.append(str(p))
    return dirs


def build_extension(verbose: bool | None = None):
    """Build or fast-load the backward-GEMM extension. Cached across launches
    and safe under concurrent runs (see _build.py)."""
    header = _THIS_DIR / "gemm_backward_kernel.cuh"
    bind = _THIS_DIR / "gemm_backward_binding.cu"
    inst = _THIS_DIR / "gemm_backward_configs.cu"
    for f in (header, bind, inst):
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}. Copy it into R3GAN/kernels/sm100/.")

    if verbose is None:
        verbose = os.environ.get("R3GAN_GEMM_BACKWARD_BUILD_VERBOSE", "0") == "1"
    cutlass = resolve_cutlass("gemm_backward/sm100")
    spec = KernelSpec(
        name="gemm_backward_sm100",
        cc_major=10,
        sources=(str(bind), str(inst)),
        hash_files=(str(header),),
        cuda_cflags=(
            "-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "-DGEMM_BF16",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_100a,code=sm_100a",
        ),
        ldflags=("-Xlinker", "--allow-shlib-undefined"),
        include_paths=tuple(_cutlass_include_dirs(cutlass)),
        src_dir=str(_THIS_DIR),
        extra_key=(f"cutlass:{cutlass_fingerprint(cutlass)}",),
    )
    return ensure_extension(spec, verbose=verbose)


def _load_ext():
    global _EXT
    if _EXT is None:
        _EXT = build_extension()
    return _EXT


def _check_nhwc(x: torch.Tensor, name: str) -> None:
    if not (x.is_cuda and x.dtype is torch.bfloat16 and x.ndim == 4 and x.is_contiguous()):
        raise ValueError(f"{name} must be CUDA bf16 contiguous NHWC, got "
                         f"shape={tuple(x.shape)} dtype={x.dtype} stride={tuple(x.stride())}")


def _check_w2d(w: torch.Tensor, name: str) -> None:
    if not (w.is_cuda and w.dtype is torch.bfloat16 and w.ndim == 2 and w.is_contiguous()):
        raise ValueError(f"{name} must be CUDA bf16 contiguous 2D [Cout,Cin], got "
                         f"shape={tuple(w.shape)} dtype={w.dtype} stride={tuple(w.stride())}")


class FFNBackward:
    def __init__(self):
        self._binding = _load_ext()
        self._best_dx: Dict[Tuple[int, int, int], Any] = {}
        self._best_dw: Dict[Tuple[int, int, int], Any] = {}

    def _pick_dx(self, M: int, Cout: int, Cin: int):
        # Config-file guidance: 128 tile for small M, 256x2x1 mid, 256x2x2 huge.
        if M < 16_384:
            return self._binding.dx_256x128_2x1
        if M > 1_000_000:
            return self._binding.dx_256x256_2x2
        return self._binding.dx_256x256_2x1

    def _pick_dw(self, M: int, Cout: int, Cin: int):
        # Small reduction dimension benefits from more output tiles; large M uses wider tiles.
        if M < 32_768:
            return self._binding.dw_256x64_2sm
        if M < 262_144:
            return self._binding.dw_256x128_2sm
        return self._binding.dw_256x256_2sm

    def dx(self, dY: torch.Tensor, W: torch.Tensor,
           residual: Optional[torch.Tensor] = None,
           act: Optional[torch.Tensor] = None,
           raster: int = 0,
           swizzle: int = 0) -> torch.Tensor:
        _check_nhwc(dY, "dY")
        _check_w2d(W, "W")
        if residual is not None:
            _check_nhwc(residual, "residual")
        if act is not None:
            _check_nhwc(act, "act")
        M = int(dY.shape[0] * dY.shape[1] * dY.shape[2])
        fn = self._pick_dx(M, int(W.shape[0]), int(W.shape[1]))
        return fn(dY, W, residual, act, int(raster), int(swizzle))

    def dw(self, dY: torch.Tensor, X: torch.Tensor,
           decomp: int = 0,
           splits: int = 1,
           reduction: Optional[torch.Tensor] = None) -> torch.Tensor:
        _check_nhwc(dY, "dY")
        _check_nhwc(X, "X")
        M = int(dY.shape[0] * dY.shape[1] * dY.shape[2])
        fn = self._pick_dw(M, int(dY.shape[3]), int(X.shape[3]))
        # Kernel signature: run_dw(dY, X, decomp, splits, reduction).
        # splits=1 avoids requiring an external reduction workspace.
        return fn(dY, X, int(decomp), int(splits), reduction)

    def symbols(self) -> list[str]:
        return [x for x in dir(self._binding) if x.startswith(("dx_", "dw_"))]


def main() -> None:
    print(FFNBackward().symbols())


if __name__ == "__main__":
    main()
