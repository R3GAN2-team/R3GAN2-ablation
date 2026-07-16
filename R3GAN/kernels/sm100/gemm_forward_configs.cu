// =============================================================================
//  ffn_prod_configs.cu
//  Production config set for the SM100 fused FFN GEMM forward (L1 / L1N / L3).
//
//  Empirically selected from the full res 8-128 x C 1024-4096 sweep:
//    - c5/c6 (2x2 cluster) and c7 (deep K=128) NEVER won -> dropped.
//    - c3 (1SM wide-N) was always dominated by c4 -> dropped.
//  The surviving three span the M range:
//    c1  128x128x64 / cluster 1x1  (1SM)            -- small M  (low res / small batch)
//    c2  256x128x64 / cluster 2x1  (2SM)            -- mid M
//    c4  256x256x64 / cluster 2x1  (2SM, wide N)    -- large M  (high res, deep Cout)
//
//  ffn_forward.py compiles this file and autotunes among these per
//  (variant, M, N, K), caching the winner. Filename intentionally does NOT match
//  the sweep harness glob (sm100_gemm_ffn_*.cu), so the two can coexist.
//  nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O3 -DGEMM_BF16
// =============================================================================
#include "gemm_forward_kernel.cuh"

GEMM_FFN_SIG(gemm_ffn_c1) { return run_gemm<Shape<_128,_128,_64>, Shape<_1,_1,_1>>(input, weight, bias, noise, scale, leaky_slope, residual, out); }
GEMM_FFN_SIG(gemm_ffn_c2) { return run_gemm<Shape<_256,_128,_64>, Shape<_2,_1,_1>>(input, weight, bias, noise, scale, leaky_slope, residual, out); }
GEMM_FFN_SIG(gemm_ffn_c4) { return run_gemm<Shape<_256,_256,_64>, Shape<_2,_1,_1>>(input, weight, bias, noise, scale, leaky_slope, residual, out); }
