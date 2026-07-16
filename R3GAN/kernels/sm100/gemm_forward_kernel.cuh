// =============================================================================
//  sm100_gemm_ffn_tma_warpspecialized_evt.cuh
//  Forward 1x1-conv FFN GEMM (L1, L3) -- SM100 / Blackwell (B200).
//  CUTLASS 4.6 collective GEMM (CollectiveBuilder mainloop + epilogue, tcgen05
//  UMMA, Auto schedules, persistent scheduler) + EVT epilogue (bias, noise-
//  scale, LeakyReLU, residual fused). bf16 I/O, fp32 accumulate.
//  Build: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O3 -DGEMM_BF16
// =============================================================================
//
// Port of sm90_gemm_ffn_tma_warpspecialized_cooperative_evt.cuh to Blackwell.
// Same problem mapping, same EVT trees, same launch-path caching and entry
// signature -- only the arch-specific collective wiring changes:
//
//   * ArchTag = Sm100; the mainloop/epilogue CollectiveBuilders are driven by
//     a tcgen05 MmaTile (NOT a CTA tile) plus a cluster shape, with
//     KernelScheduleAuto / EpilogueScheduleAuto picking the 1SM vs 2SM UMMA.
//   * 2SM is selected when ClusterShape M is even (the Auto-schedule rule); the
//     UMMA M is then 256 and the two CTAs in the cluster each own M/2 = 128 of
//     the output. The epilogue's per-CTA tile is therefore (MmaM/2, N, K) in
//     2SM and MmaTile in 1SM -- this is what the Sm90 row/col-broadcast EVT
//     leaves must be templated on, since the epilogue builder passes a raw EVT
//     through verbatim (only tagged fusion::Operation types are rebuilt).
//   * Blackwell reuses the Sm90* EVT nodes (they alias to their Hopper
//     counterparts), so the Trees<> below are unchanged from SM90 apart from
//     receiving the correct per-CTA tile.
//
// A 1x1 convolution over [N, H, W, Cin] with weight [Cout, Cin] is a single
// GEMM with M = N*H*W, N = Cout, K = Cin: A = flattened input (row-major MxK),
// B = weight (column-major NxK, i.e. K-contiguous). Three fused variants:
//
//   L1   D = LeakyReLU(A@B + bias)
//   L1N  D = LeakyReLU(A@B + bias + noise * scale)
//   L3   D = A@B + residual                  (residual enters as source C)
//
// bias and scale broadcast per output channel (row vectors of length Cout);
// noise broadcasts per pixel (a column vector of length M). All three
// auxiliary tensors are fp32 regardless of the I/O element type.
//
// Profiling sweeps: run_gemm is templated on <MmaTile, ClusterShape>; each
// instantiation TU pins one (tile, cluster) and exports a sweep symbol
// (gemm_ffn_cN), exactly as on SM90. See the recommended SM100 grid at the
// bottom of this file. Within a config, repeat same-shape calls take the
// lightweight update() path under the persistent scheduler.
//
// Build: -DGEMM_BF16 for bf16 I/O (fp32 accumulate); default is TF32.
//        -DGEMM_FFN_LIGHT_UPDATE=0 to always initialize() (fallback).

#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <memory>
#include <tuple>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/functional.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

#ifndef GEMM_FFN_LIGHT_UPDATE
#define GEMM_FFN_LIGHT_UPDATE 1
#endif

#define CUTLASS_CHECK(status)                                                  \
  do {                                                                         \
    cutlass::Status s_ = (status);                                             \
    TORCH_CHECK(s_ == cutlass::Status::kSuccess,                               \
                "CUTLASS error: ", cutlassGetStatusString(s_));                \
  } while (0)

// Uniform entry signature shared by every configuration TU and the binding
// file. The trailing optional `out` is a caller-preallocated output tensor.
#define GEMM_FFN_SIG(NAME)                                                     \
  torch::Tensor NAME(torch::Tensor input, torch::Tensor weight,               \
                     c10::optional<torch::Tensor> bias,                       \
                     c10::optional<torch::Tensor> noise,                      \
                     c10::optional<torch::Tensor> scale,                      \
                     c10::optional<double> leaky_slope,                       \
                     c10::optional<torch::Tensor> residual,                   \
                     c10::optional<torch::Tensor> out)

#ifdef GEMM_BF16
using ElementIO = cutlass::bfloat16_t;
static constexpr int AlignIO = 8;
#else
using ElementIO = float;
static constexpr int AlignIO = 4;
#endif
using ElementA   = ElementIO;
using ElementB   = ElementIO;
using ElementOut = ElementIO;
using ElementAcc = float;
using ElementComp= float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
constexpr auto RN = cutlass::FloatRoundStyle::round_to_nearest;

using ArchTag = cutlass::arch::Sm100;

namespace fusion = cutlass::epilogue::fusion;

// ---------------------------------------------------------------------------
// Epilogue visitor trees (identical to SM90 -- the Sm90* nodes alias on
// Blackwell). CtaTile here is the SM100 PER-CTA epilogue tile, not the MMA
// tile: the row/col broadcasts partition the N/M extent a single CTA owns.
//   Acc fetches the accumulator; Src fetches source C (the residual).
//   BiasRow/ScaleRow broadcast per-N row vectors; NoiseCol broadcasts a
//   per-M column vector.
// ---------------------------------------------------------------------------
template <class CtaTile>
struct Trees {
  using Acc      = fusion::Sm90AccFetch;
  using Src      = fusion::Sm90SrcFetch<ElementOut>;
  using BiasRow  = fusion::Sm90RowBroadcast<0, CtaTile, float>;
  using ScaleRow = fusion::Sm90RowBroadcast<0, CtaTile, float>;
  using NoiseCol = fusion::Sm90ColBroadcast<0, CtaTile, float>;
  using PlusF    = fusion::Sm90Compute<cutlass::plus,       ElementComp, ElementComp, RN>;
  using MulF     = fusion::Sm90Compute<cutlass::multiplies, ElementComp, ElementComp, RN>;
  using PlusO    = fusion::Sm90Compute<cutlass::plus,       ElementOut,  ElementComp, RN>;
  using LRelu    = fusion::Sm90Compute<cutlass::epilogue::thread::LeakyReLU, ElementOut, ElementComp, RN>;

  using AccBias  = fusion::Sm90EVT<PlusF, Acc, BiasRow>;
  using NScale   = fusion::Sm90EVT<MulF,  NoiseCol, ScaleRow>;
  using PreActN  = fusion::Sm90EVT<PlusF, AccBias, NScale>;

  using L1  = fusion::Sm90EVT<LRelu, AccBias>;
  using L1N = fusion::Sm90EVT<LRelu, PreActN>;
  using L3  = fusion::Sm90EVT<PlusO, Acc, Src>;
};

// The SM count is immutable per device; query it once and cache it. The map
// is leaked deliberately so no static destructor runs after CUDA teardown.
static int cached_sm_count(int device_id) {
  static std::mutex m;
  static auto* cache = new std::map<int, int>();
  std::lock_guard<std::mutex> lk(m);
  auto it = cache->find(device_id);
  if (it != cache->end()) return it->second;
  int n = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  (*cache)[device_id] = n;
  return n;
}

// ---------------------------------------------------------------------------
// Builds the collective mainloop/epilogue for one (MmaTile, cluster, EVT)
// combination and launches it through the cached plan. The SM100 builders take
// the MMA tile + cluster and select 1SM/2SM via the (Auto) schedule; the
// default GemmUniversal tile scheduler is persistent.
// ---------------------------------------------------------------------------
template <class MmaTile, class ClusterShape, class MainloopSched, class EpilogueSched,
          class FusionEVT, class FArgs>
static torch::Tensor run_one(torch::Tensor input, torch::Tensor weight,
                             int M, int N, int K, torch::Tensor output,
                             ElementOut* C_ptr, FArgs fargs) {
  // footgun 31: descriptor encode + launches must run under the input
  // tensor's device context (multi-GPU: rank-local tensors, spawn procs).
  const c10::cuda::CUDAGuard _dev_guard(input.device());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      MmaTile, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementComp,
      ElementOut, LayoutC, AlignIO,
      ElementOut, LayoutC, AlignIO,
      EpilogueSched, FusionEVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignIO,
      ElementB, LayoutB, AlignIO,
      ElementAcc, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSched>::CollectiveOp;
  // Default (persistent) tile scheduler -- omitted, as in the Blackwell examples.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  auto sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  const int dev = input.device().index();
  cutlass::KernelHardwareInfo hw;
  hw.device_id = dev;
  hw.sm_count  = cached_sm_count(dev);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA*>(input.data_ptr()), sA,
       reinterpret_cast<ElementB*>(weight.data_ptr()), sB},
      {{}, C_ptr, sC, reinterpret_cast<ElementOut*>(output.data_ptr()), sD},
      hw};
  args.epilogue.thread = fargs;

  auto stream = at::cuda::getCurrentCUDAStream();

  // Cached launch plan keyed by (device, M, N, K): the operator, workspace, and
  // can_implement check are paid once. Later same-shape calls take update()
  // (lightweight re-arg for the persistent scheduler); the workspace handed to
  // initialize() is retained, so update() in CUTLASS 4.6 takes only `args`.
  struct Plan { std::unique_ptr<Gemm> op; torch::Tensor ws; bool ready = false; };
  using Key = std::tuple<int, int, int, int>;
  static std::mutex* mtx   = new std::mutex();
  static auto*       cache = new std::map<Key, Plan>();

  Key key{dev, M, N, K};
  std::lock_guard<std::mutex> lock(*mtx);
  Plan& plan = (*cache)[key];

  if (!plan.ready) {
    plan.op = std::make_unique<Gemm>();
    size_t ws = Gemm::get_workspace_size(args);
    plan.ws = ws ? torch::empty({(int64_t)ws}, input.options().dtype(torch::kUInt8))
                 : torch::Tensor();
    void* wptr = ws ? plan.ws.data_ptr() : nullptr;
    CUTLASS_CHECK(plan.op->can_implement(args));
    CUTLASS_CHECK(plan.op->initialize(args, wptr, stream));
    plan.ready = true;
  } else {
#if GEMM_FFN_LIGHT_UPDATE
    CUTLASS_CHECK(plan.op->update(args));
#else
    void* wptr = plan.ws.defined() ? plan.ws.data_ptr() : nullptr;
    CUTLASS_CHECK(plan.op->initialize(args, wptr, stream));
#endif
  }
  CUTLASS_CHECK(plan.op->run(stream));
  return output;
}

// ---------------------------------------------------------------------------
// Entry point: shape/dtype checks, output allocation (or out= reuse), and
// epilogue selection from the optional arguments. Templated on the SM100 MMA
// tile + cluster; derives 2SM-ness and the per-CTA epilogue tile that the EVT
// broadcasts are built against.
// ---------------------------------------------------------------------------
template <class MmaTile, class ClusterShape,
          class MainloopSched = cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSched = cutlass::epilogue::collective::EpilogueScheduleAuto>
static torch::Tensor run_gemm(torch::Tensor input, torch::Tensor weight,
                              c10::optional<torch::Tensor> bias,
                              c10::optional<torch::Tensor> noise,
                              c10::optional<torch::Tensor> scale,
                              c10::optional<double> leaky_slope,
                              c10::optional<torch::Tensor> residual,
                              c10::optional<torch::Tensor> out = c10::nullopt) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(input.dim() == 4, "input must be NHWC (N,H,W,Cin)");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D (Cout,Cin)");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "contiguous required");
#ifdef GEMM_BF16
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "bf16 build expects bf16 input/weight");
#else
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "tf32 build expects float32 input/weight");
#endif
  int Nn = input.size(0), H = input.size(1), W = input.size(2), K = input.size(3);
  int Ncout = weight.size(0);
  TORCH_CHECK(weight.size(1) == K, "weight (Cout,Cin): Cin must match input channels");
  int M = Nn * H * W;

  torch::Tensor output;
  if (out) {
    TORCH_CHECK(out->is_cuda() && out->is_contiguous()
                && out->scalar_type() == input.scalar_type()
                && out->dim() == 4 && out->size(0) == Nn && out->size(1) == H
                && out->size(2) == W && out->size(3) == Ncout,
                "out must be a contiguous CUDA NHWC tensor (N,H,W,Cout) matching input dtype");
    output = *out;
  } else {
    output = torch::empty({Nn, H, W, Ncout}, input.options());
  }

  // ---- SM100 tile derivation -------------------------------------------------
  // 2SM UMMA when cluster M is even (the Auto-schedule rule); UMMA M is then 256
  // and each of the 2 CTAs owns M/2 of the output, so the per-CTA epilogue tile
  // is (MmaM/2, N, K). 1SM otherwise, per-CTA tile == MmaTile. This must match
  // the epilogue builder's internal cta_tile_shape() for the broadcast leaves.
  static constexpr int  kClusterM = cute::size<0>(ClusterShape{});
  static constexpr int  kMmaM     = cute::size<0>(MmaTile{});
  static constexpr bool kUse2Sm   = (kClusterM % 2 == 0);
  static_assert(kMmaM == (kUse2Sm ? 256 : 128),
      "SM100 FFN sweep convention: MmaTile M must be 256 when ClusterShape M is even (2SM) else 128 (1SM).");
  using CtaTile = cute::conditional_t< kUse2Sm,
      cute::Shape<cute::Int<kMmaM / 2>, decltype(cute::get<1>(MmaTile{})), decltype(cute::get<2>(MmaTile{}))>,
      MmaTile >;

  using T = Trees<CtaTile>;
  const float slope = leaky_slope ? static_cast<float>(*leaky_slope) : 0.f;
  auto Pf = [](c10::optional<torch::Tensor>& t) { return t->data_ptr<float>(); };

  if (residual) {
    TORCH_CHECK(residual->is_contiguous() && residual->scalar_type() == input.scalar_type(),
                "residual must be contiguous and match input dtype");
    typename T::L3::Arguments a{ {}, {}, {} };
    return run_one<MmaTile, ClusterShape, MainloopSched, EpilogueSched, typename T::L3>(
        input, weight, M, Ncout, K, output,
        reinterpret_cast<ElementOut*>(residual->data_ptr()), a);
  }
  if (bias && noise && scale && leaky_slope) {
    typename T::L1N::Arguments a{
        { { {}, {Pf(bias)}, {} }, { {Pf(noise)}, {Pf(scale)}, {} }, {} }, { slope } };
    return run_one<MmaTile, ClusterShape, MainloopSched, EpilogueSched, typename T::L1N>(
        input, weight, M, Ncout, K, output, nullptr, a);
  }
  if (bias && leaky_slope) {
    typename T::L1::Arguments a{ { {}, {Pf(bias)}, {} }, { slope } };
    return run_one<MmaTile, ClusterShape, MainloopSched, EpilogueSched, typename T::L1>(
        input, weight, M, Ncout, K, output, nullptr, a);
  }
  TORCH_CHECK(false, "unsupported epilogue: pass residual, or bias+leaky_slope (+noise+scale)");
}

// =============================================================================
// Recommended SM100 sweep grid (one explicit-instantiation TU each, mirroring
// the SM90 gemm_ffn_cN naming). MmaTile M is fixed by the 1SM/2SM choice:
//
//   1SM (odd cluster M):   MmaTile Shape<_128, TILE_N, _64>
//   2SM (even cluster M):  MmaTile Shape<_256, TILE_N, _64>   (per-CTA N == TILE_N)
//
//   TILE_N   in {128, 256}             (Cout-tile; 256 needs Cout % 256 padding-friendly)
//   cluster  in {1x1x1, 2x1x1, 1x2x1, 2x2x1, 4x4x1, 2x4x1, ...}
//
// Examples:
//   run_gemm<Shape<_128,_128,_64>, Shape<_1,_1,_1>>   // 1SM, no cluster
//   run_gemm<Shape<_128,_256,_64>, Shape<_1,_2,_1>>   // 1SM, wide N, N-cluster
//   run_gemm<Shape<_256,_128,_64>, Shape<_2,_1,_1>>   // 2SM
//   run_gemm<Shape<_256,_256,_64>, Shape<_2,_2,_1>>   // 2SM, wide N, 2x2 cluster
//
// For the FFN shapes (M = N*H*W large, K = Cin, N = Cout), large clusters with
// 2SM and TILE_N=256 usually win at high resolution; small M (low res) prefers
// 1SM / small clusters. Sweep, then pin per (resolution, Cin, Cout).
// =============================================================================
