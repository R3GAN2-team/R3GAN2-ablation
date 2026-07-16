// =============================================================================
//  sm100_ffn_backward_instances.cu
//  Production tile-shape instantiations of the SM100 backward FFN GEMMs.
//
//  ffn_backward.py discovers the dx_* / dw_* symbols, binds them, and autotunes
//  per (op, shape). Naming: dx_<MmaM>x<N>_<clusterM>x<clusterN> ; MmaM=256 (2SM).
//  dx also takes runtime (raster, swizzle) tile-scheduler knobs (no recompile).
//
//  dx set pruned from an 8-config sweep at fixed batch (M = N*res^2), C=1024:
//    256x128_2x1  wins small M (<~16k)         -- narrow N-tile, least waste
//    256x256_2x1  workhorse, wins mid M (~65k-260k); beats cuBLAS there
//    256x256_2x2  wins the huge-M tail (>~1M)  -- cluster crossover: bigger
//                 B-multicast finally amortizes once M is enormous
//  Dropped: 1SM tiles (never won, even at M=2k); deep-K (M-independent, dead);
//  256x128_4x1 / 256x256_4x1 / 256x256_4x2 (dominated); raster/swizzle tuning
//  (no-op -- W is L2-resident, so tile placement doesn't move the needle).
//
//  Build (driven by the frontend via torch cpp_extension):
//    nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O3 -DGEMM_BF16
//    +  -DDX_RESIDUAL_ONLY  for a fast dx-profiling build (residual epilogue only)
// =============================================================================
#include "gemm_backward_kernel.cuh"

// ---- dx: dX = dY @ W (+ residual | . slope(act)); persistent scheduler --------
// RowMajor B (weight as-stored) -- a K-major (ColumnMajor B, transposed weight)
// variant was tested and is ~9% SLOWER at large M, so there is no layout penalty
// to recover; the UMMA feeds its B operand faster from an N-contiguous layout.
DX_SIG(dx_256x128_2x1) { return run_dx<Shape<_256,_128,_64>, Shape<_2,_1,_1>>(dY, W, residual, act, raster, swizzle); }
DX_SIG(dx_256x256_2x1) { return run_dx<Shape<_256,_256,_64>, Shape<_2,_1,_1>>(dY, W, residual, act, raster, swizzle); }
DX_SIG(dx_256x256_2x2) { return run_dx<Shape<_256,_256,_64>, Shape<_2,_2,_1>>(dY, W, residual, act, raster, swizzle); }

// ---- dw: dW = dY^T @ X, reduce over pixels via stream-K; fp32 out -------------
// Tile shape sets the output-TILE-COUNT for the fixed [Cout,Cin] output. A tile-
// count sweep (16->128) showed MORE tiles does NOT break the ~52% ceiling: the
// cap is the stream-K reduction epilogue, not occupancy, so narrow-N/1SM tiles
// just trade MMA efficiency for reduction depth (net zero) and lose at large K.
// Kept: 256x256/256x128 (large-K workhorses) + 256x64 (small-K: fills 64 SMs
// data-parallel when K is too small for stream-K). Dropped 128x128/128x64 1SM.
DW_SIG(dw_256x256_2sm) { return run_dw<Shape<_256,_256,_64>, Shape<_2,_1,_1>>(dY, X, decomp, splits, reduction); }  // 16 tiles, large K
DW_SIG(dw_256x128_2sm) { return run_dw<Shape<_256,_128,_64>, Shape<_2,_1,_1>>(dY, X, decomp, splits, reduction); }  // 32 tiles, large K
DW_SIG(dw_256x64_2sm)  { return run_dw<Shape<_256, _64,_64>, Shape<_2,_1,_1>>(dY, X, decomp, splits, reduction); }  // 64 tiles, small K
