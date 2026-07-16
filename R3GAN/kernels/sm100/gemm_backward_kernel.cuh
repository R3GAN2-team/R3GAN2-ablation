// =============================================================================
//  sm100_gemm_ffn_backward_evt.cuh
//  Backward 1x1-conv FFN GEMMs (dx, dw) -- SM100 / Blackwell (B200).
//  CUTLASS 4.6 collective GEMM (CollectiveBuilder mainloop + epilogue, tcgen05
//  UMMA, Auto schedules) + EVT epilogue. bf16 I/O, fp32 accumulate.
//  Build: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O3 -DGEMM_BF16
//         -DDX_RESIDUAL_ONLY   compiles only the residual dx epilogue (fast
//                              profiling build; the slope branch TORCH_CHECKs).
//
//  Companion to sm100_gemm_ffn_tma_warpspecialized_evt.cuh (forward).  Included
//  by sm100_ffn_backward_instances.cu, which pins (MmaTile, ClusterShape) per
//  exported dx_* / dw_* symbol.  Same problem mapping conventions:
//
//  dx:  dX = dY @ W  (+ residual  |  . slope(act))
//       dY [N,H,W,Cout] NHWC contiguous, W [Cout,Cin] as-stored (row-major).
//       GEMM: M = N*H*W, N = Cin, K = Cout.
//         A = dY   RowMajor    (K-contiguous)
//         B = W    RowMajor    (N-contiguous -- weight as-stored; the K-major
//                               transposed variant measured ~9% slower)
//       Epilogue (exactly one of):
//         residual:  D = acc + residual          (L1 backward: skip-path grad)
//         act:       D = acc * slope(act)        (L3 backward: fuse act-2 slope;
//                    slope taken from POST-activation sign, valid for lrelu
//                    with alpha > 0.  alpha is compile-time: -DDX_LEAKY_SLOPE,
//                    default 0.2f -- the DX_SIG surface carries no alpha, by
//                    design; it must match Networks.UnscaledLeakyReLU.)
//       Default (persistent) tile scheduler; raster/swizzle are RUNTIME knobs
//       (no recompile): raster 0=Heuristic 1=AlongM 2=AlongN, swizzle >= 1.
//
//  dw:  dW = dY^T @ X, fp32 out [Cout, Cin], reduction over pixels via stream-K.
//       GEMM: M = Cout, N = Cin, K = N*H*W.
//         A = dY^T  ColumnMajor (dY as-stored: offset(co, m) = m*Cout + co)
//         B = X     RowMajor    (X  as-stored: offset(ci, m) = m*Cin  + ci)
//       Runtime knobs: decomp 0=Heuristic 1=SplitK 2=StreamK 3=DataParallel,
//       splits >= 1, reduction 0=Deterministic 1=Nondeterministic.
//       initialize() runs on EVERY call (verified against CUTLASS source):
//       stream-K fixup locks are never reset by the kernel, so the barrier
//       workspace must be re-zeroed per launch; initialize() does this with a
//       KB-scale async memset on the caller's stream.
//
//  Plan caching as in the forward header: per (device, M, N, K, knobs), the
//  operator + workspace + can_implement are paid once; dx same-shape calls take
//  the lightweight update() path.
//
//  All internals live in namespace sm100_ffn_bwd (hoisted via using-decls at
//  the bottom) so this header can coexist with the forward header in one TU.
// =============================================================================

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
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                                  \
  do {                                                                         \
    cutlass::Status s_ = (status);                                             \
    TORCH_CHECK(s_ == cutlass::Status::kSuccess,                               \
                "CUTLASS error: ", cutlassGetStatusString(s_));                \
  } while (0)
#endif

// LeakyReLU negative slope baked into the dx slope epilogue.  The DX_SIG
// surface deliberately has no alpha argument; must match the network's
// UnscaledLeakyReLU alpha (0.2).
#ifndef DX_LEAKY_SLOPE
#define DX_LEAKY_SLOPE 0.2f
#endif

// Uniform entry signatures shared by every instantiation TU and the binding
// file.  These are the ONLY names fused-FFN frontends link against.
#define DX_SIG(NAME)                                                           \
  torch::Tensor NAME(torch::Tensor dY, torch::Tensor W,                        \
                     c10::optional<torch::Tensor> residual,                    \
                     c10::optional<torch::Tensor> act,                         \
                     c10::optional<int64_t> raster,                            \
                     c10::optional<int64_t> swizzle)

#define DW_SIG(NAME)                                                           \
  torch::Tensor NAME(torch::Tensor dY, torch::Tensor X,                        \
                     c10::optional<int64_t> decomp,                            \
                     c10::optional<int64_t> splits,                            \
                     c10::optional<int64_t> reduction)

namespace sm100_ffn_bwd {

#ifdef GEMM_BF16
using ElementIO = cutlass::bfloat16_t;
static constexpr int AlignIO = 8;
#else
using ElementIO = float;
static constexpr int AlignIO = 4;
#endif
using ElementA    = ElementIO;
using ElementB    = ElementIO;
using ElementOut  = ElementIO;   // dx output element
using ElementAcc  = float;
using ElementComp = float;
using ElementDw   = float;       // dw output element (fp32, reduced over pixels)
constexpr auto RN = cutlass::FloatRoundStyle::round_to_nearest;

using ArchTag = cutlass::arch::Sm100;

namespace fusion = cutlass::epilogue::fusion;

// ---------------------------------------------------------------------------
// dx epilogue visitor trees.  No row/col broadcasts here, so no per-CTA tile
// templating is needed (only Acc + source-C fetch + a compute node).
//   DxResidual:  D = acc + C           (C = residual, bf16)
//   DxSlope:     D = acc * slope(C)    (C = post-activation a, bf16;
//                                       slope = a < 0 ? alpha : 1)
// ---------------------------------------------------------------------------
template <class T>
struct LReluBwd {
  // f(acc, a) = acc * d/dz lrelu(z) with the sign read off the POST-activation
  // value a (lrelu with alpha > 0 preserves sign; x == 0 uses the positive-side
  // convention, matching Networks.UnscaledLeakyReLU.Slope).
  CUTLASS_HOST_DEVICE
  T operator()(T const& acc, T const& a) const {
    return (a < T(0)) ? T(acc * T(DX_LEAKY_SLOPE)) : acc;
  }
};

// Sm90Compute applies the op to whole register fragments (cutlass::Array<T,N>),
// not scalars -- same pattern as the Array specializations in
// cutlass/epilogue/thread/activation.h.
template <class T, int N>
struct LReluBwd<cutlass::Array<T, N>> {
  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& acc,
                                  cutlass::Array<T, N> const& a) const {
    cutlass::Array<T, N> out;
    LReluBwd<T> op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      out[i] = op(acc[i], a[i]);
    }
    return out;
  }
};

struct DxTrees {
  using Acc    = fusion::Sm90AccFetch;
  using Src    = fusion::Sm90SrcFetch<ElementOut>;
  using PlusO  = fusion::Sm90Compute<cutlass::plus, ElementOut, ElementComp, RN>;
  using SlopeO = fusion::Sm90Compute<LReluBwd,      ElementOut, ElementComp, RN>;

  using DxResidual = fusion::Sm90EVT<PlusO,  Acc, Src>;
  using DxSlope    = fusion::Sm90EVT<SlopeO, Acc, Src>;
};

// dw epilogue: plain fp32 store of the accumulator.
struct DwTrees {
  using Acc   = fusion::Sm90AccFetch;
  using Ident = fusion::Sm90Compute<cutlass::epilogue::thread::Identity,
                                    ElementDw, ElementComp, RN>;
  using DwOut = fusion::Sm90EVT<Ident, Acc>;
};

// SM count is immutable per device; query once (map leaked deliberately, as in
// the forward header, so no static destructor runs after CUDA teardown).
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
// Runtime tile-scheduler knobs, set only if the scheduler Arguments struct has
// the field (member-detection keeps this robust across CUTLASS minor versions
// and scheduler selections; absent fields make the knob a documented no-op).
// ---------------------------------------------------------------------------
template <class T, class = void> struct has_max_swizzle : std::false_type {};
template <class T> struct has_max_swizzle<T,
  std::void_t<decltype(std::declval<T&>().max_swizzle_size)>> : std::true_type {};

template <class T, class = void> struct has_raster : std::false_type {};
template <class T> struct has_raster<T,
  std::void_t<decltype(std::declval<T&>().raster_order)>> : std::true_type {};

template <class T, class = void> struct has_splits : std::false_type {};
template <class T> struct has_splits<T,
  std::void_t<decltype(std::declval<T&>().splits)>> : std::true_type {};

template <class T, class = void> struct has_decomp : std::false_type {};
template <class T> struct has_decomp<T,
  std::void_t<decltype(std::declval<T&>().decomposition_mode)>> : std::true_type {};

template <class T, class = void> struct has_reduction : std::false_type {};
template <class T> struct has_reduction<T,
  std::void_t<decltype(std::declval<T&>().reduction_mode)>> : std::true_type {};

template <class Sched>
static void set_persistent_knobs(Sched& sch, int raster, int swizzle) {
  if constexpr (has_max_swizzle<Sched>::value)
    sch.max_swizzle_size = swizzle > 0 ? swizzle : 1;
  if constexpr (has_raster<Sched>::value) {
    using R = std::decay_t<decltype(sch.raster_order)>;
    sch.raster_order = static_cast<R>(raster);   // 0=Heuristic 1=AlongM 2=AlongN
  }
}

template <class Sched>
static void set_streamk_knobs(Sched& sch, int64_t decomp, int64_t splits, int64_t reduction) {
  if constexpr (has_splits<Sched>::value)
    if (splits > 0) sch.splits = static_cast<int>(splits);
  if constexpr (has_decomp<Sched>::value) {
    using D = std::decay_t<decltype(sch.decomposition_mode)>;
    sch.decomposition_mode = static_cast<D>(decomp);   // 0=Heuristic 1=SplitK 2=StreamK 3=DataParallel
  }
  if constexpr (has_reduction<Sched>::value) {
    using R = std::decay_t<decltype(sch.reduction_mode)>;
    sch.reduction_mode = static_cast<R>(reduction);    // 0=Deterministic 1=Nondeterministic
  }
}

// Shared 2SM convention check (identical to the forward header's rule).
template <class MmaTile, class ClusterShape>
static constexpr void check_2sm_convention() {
  constexpr int  kClusterM = cute::size<0>(ClusterShape{});
  constexpr int  kMmaM     = cute::size<0>(MmaTile{});
  constexpr bool kUse2Sm   = (kClusterM % 2 == 0);
  static_assert(kMmaM == (kUse2Sm ? 256 : 128),
      "SM100 FFN sweep convention: MmaTile M must be 256 when ClusterShape M is even (2SM) else 128 (1SM).");
}

// ---------------------------------------------------------------------------
// dx launcher: builds one (MmaTile, cluster, EVT) combination on the default
// (persistent) tile scheduler and launches through the cached plan.
//   A = dY [M,K]  RowMajor;  B = W [K,N]  RowMajor (weight as-stored);
//   C = residual|act [M,N] bf16 (source fetch);  D = dX [M,N] bf16.
// ---------------------------------------------------------------------------
template <class MmaTile, class ClusterShape, class MainloopSched, class EpilogueSched,
          class FusionEVT, class FArgs>
static torch::Tensor run_dx_one(torch::Tensor dY, torch::Tensor W,
                                int M, int N, int K, torch::Tensor output,
                                ElementOut* C_ptr, FArgs fargs,
                                int raster, int swizzle) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

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
  // Default (persistent) tile scheduler -- omitted, as in the forward header.
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

  const int dev = dY.device().index();
  cutlass::KernelHardwareInfo hw;
  hw.device_id = dev;
  hw.sm_count  = cached_sm_count(dev);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA*>(dY.data_ptr()), sA,
       reinterpret_cast<ElementB*>(W.data_ptr()), sB},
      {{}, C_ptr, sC, reinterpret_cast<ElementOut*>(output.data_ptr()), sD},
      hw};
  args.epilogue.thread = fargs;
  set_persistent_knobs(args.scheduler, raster, swizzle);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Cached launch plan keyed by (device, M, N, K, raster, swizzle): operator,
  // workspace, and can_implement paid once; later same-key calls take update().
  struct Plan { std::unique_ptr<Gemm> op; torch::Tensor ws; bool ready = false; };
  using Key = std::tuple<int, int, int, int, int, int>;
  static std::mutex* mtx   = new std::mutex();
  static auto*       cache = new std::map<Key, Plan>();

  Key key{dev, M, N, K, raster, swizzle};
  std::lock_guard<std::mutex> lock(*mtx);
  Plan& plan = (*cache)[key];

  if (!plan.ready) {
    plan.op = std::make_unique<Gemm>();
    size_t ws = Gemm::get_workspace_size(args);
    plan.ws = ws ? torch::empty({(int64_t)ws}, dY.options().dtype(torch::kUInt8))
                 : torch::Tensor();
    void* wptr = ws ? plan.ws.data_ptr() : nullptr;
    CUTLASS_CHECK(plan.op->can_implement(args));
    CUTLASS_CHECK(plan.op->initialize(args, wptr, stream));
    plan.ready = true;
  } else {
    CUTLASS_CHECK(plan.op->update(args));
  }
  CUTLASS_CHECK(plan.op->run(stream));
  return output;
}

// ---------------------------------------------------------------------------
// dx entry: shape/dtype checks, output allocation, epilogue selection.
//   dX = dY @ W (+ residual | . slope(act));  exactly one of residual/act.
// ---------------------------------------------------------------------------
template <class MmaTile, class ClusterShape,
          class MainloopSched = cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSched = cutlass::epilogue::collective::EpilogueScheduleAuto>
static torch::Tensor run_dx(torch::Tensor dY, torch::Tensor W,
                            c10::optional<torch::Tensor> residual,
                            c10::optional<torch::Tensor> act,
                            c10::optional<int64_t> raster,
                            c10::optional<int64_t> swizzle) {
  // footgun 31: descriptor encode + launches must run under the input
  // tensor's device context (multi-GPU: rank-local tensors, spawn procs).
  const c10::cuda::CUDAGuard _dev_guard(dY.device());

  check_2sm_convention<MmaTile, ClusterShape>();

  TORCH_CHECK(dY.is_cuda() && W.is_cuda(), "dx: CUDA tensors required");
  TORCH_CHECK(dY.dim() == 4, "dx: dY must be NHWC (N,H,W,Cout)");
  TORCH_CHECK(W.dim() == 2, "dx: W must be 2D (Cout,Cin), as-stored (RowMajor B)");
  TORCH_CHECK(dY.is_contiguous() && W.is_contiguous(), "dx: contiguous required");
#ifdef GEMM_BF16
  TORCH_CHECK(dY.scalar_type() == torch::kBFloat16 && W.scalar_type() == torch::kBFloat16,
              "dx: bf16 build expects bf16 dY/W");
#else
  TORCH_CHECK(dY.scalar_type() == torch::kFloat32 && W.scalar_type() == torch::kFloat32,
              "dx: tf32 build expects float32 dY/W");
#endif
  const int Nn = dY.size(0), H = dY.size(1), Wp = dY.size(2);
  const int K  = dY.size(3);                 // Cout
  TORCH_CHECK(W.size(0) == K, "dx: W (Cout,Cin): Cout must match dY channels");
  const int N  = W.size(1);                  // Cin
  const int M  = Nn * H * Wp;
  TORCH_CHECK(K % AlignIO == 0 && N % AlignIO == 0,
              "dx: Cout and Cin must be multiples of ", AlignIO, " (TMA alignment)");

  torch::Tensor output = torch::empty({Nn, H, Wp, N}, dY.options());

  const int r = raster  ? static_cast<int>(*raster)  : 0;
  const int s = swizzle ? static_cast<int>(*swizzle) : 1;

  auto check_aux = [&](torch::Tensor const& t, char const* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dY.scalar_type()
                && t.dim() == 4 && t.size(0) == Nn && t.size(1) == H
                && t.size(2) == Wp && t.size(3) == N,
                "dx: ", name, " must be a contiguous NHWC (N,H,W,Cin) tensor matching dY dtype");
  };

  if (residual) {
    TORCH_CHECK(!act, "dx: pass residual OR act, not both");
    check_aux(*residual, "residual");
    typename DxTrees::DxResidual::Arguments a{ {}, {}, {} };
    return run_dx_one<MmaTile, ClusterShape, MainloopSched, EpilogueSched,
                      typename DxTrees::DxResidual>(
        dY, W, M, N, K, output,
        reinterpret_cast<ElementOut*>(residual->data_ptr()), a, r, s);
  }
#ifndef DX_RESIDUAL_ONLY
  if (act) {
    check_aux(*act, "act");
    typename DxTrees::DxSlope::Arguments a{ {}, {}, {} };
    return run_dx_one<MmaTile, ClusterShape, MainloopSched, EpilogueSched,
                      typename DxTrees::DxSlope>(
        dY, W, M, N, K, output,
        reinterpret_cast<ElementOut*>(act->data_ptr()), a, r, s);
  }
  TORCH_CHECK(false, "dx: unsupported epilogue -- pass residual (D = acc + residual) "
                     "or act (D = acc * slope(act))");
#else
  TORCH_CHECK(false, "dx: DX_RESIDUAL_ONLY build -- only the residual epilogue is compiled");
#endif
}

// ---------------------------------------------------------------------------
// dw entry: dW = dY^T @ X, fp32 out [Cout, Cin], stream-K over K = N*H*W.
//   A = dY^T  ColumnMajor  (dY as-stored: offset(co, m) = m*Cout + co)
//   B = X     RowMajor     (X  as-stored: offset(ci, m) = m*Cin  + ci)
// initialize() runs EVERY call: the stream-K workspace (fixup barriers +
// reduction buffer) must be re-zeroed per launch; update() does not do that.
// ---------------------------------------------------------------------------
template <class MmaTile, class ClusterShape,
          class MainloopSched = cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSched = cutlass::epilogue::collective::EpilogueScheduleAuto>
static torch::Tensor run_dw(torch::Tensor dY, torch::Tensor X,
                            c10::optional<int64_t> decomp,
                            c10::optional<int64_t> splits,
                            c10::optional<int64_t> reduction) {
  // footgun 31: descriptor encode + launches must run under the input
  // tensor's device context (multi-GPU: rank-local tensors, spawn procs).
  const c10::cuda::CUDAGuard _dev_guard(dY.device());

  check_2sm_convention<MmaTile, ClusterShape>();

  TORCH_CHECK(dY.is_cuda() && X.is_cuda(), "dw: CUDA tensors required");
  TORCH_CHECK(dY.dim() == 4 && X.dim() == 4, "dw: dY/X must be NHWC");
  TORCH_CHECK(dY.is_contiguous() && X.is_contiguous(), "dw: contiguous required");
#ifdef GEMM_BF16
  TORCH_CHECK(dY.scalar_type() == torch::kBFloat16 && X.scalar_type() == torch::kBFloat16,
              "dw: bf16 build expects bf16 dY/X");
#else
  TORCH_CHECK(dY.scalar_type() == torch::kFloat32 && X.scalar_type() == torch::kFloat32,
              "dw: tf32 build expects float32 dY/X");
#endif
  TORCH_CHECK(dY.size(0) == X.size(0) && dY.size(1) == X.size(1) && dY.size(2) == X.size(2),
              "dw: dY and X must share (N,H,W)");
  const int M = dY.size(3);                                  // Cout
  const int N = X.size(3);                                   // Cin
  const int K = dY.size(0) * dY.size(1) * dY.size(2);        // pixels
  TORCH_CHECK(M % AlignIO == 0 && N % AlignIO == 0,
              "dw: Cout and Cin must be multiples of ", AlignIO, " (TMA alignment)");

  torch::Tensor output = torch::empty({M, N}, dY.options().dtype(torch::kFloat32));

  using LayoutA = cutlass::layout::ColumnMajor;   // A = dY^T, M(Cout)-contiguous
  using LayoutB = cutlass::layout::RowMajor;      // B = X,    N(Cin)-contiguous
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignDw = 4;                      // fp32 D: 4 elems = 16B

  using FusionEVT = typename DwTrees::DwOut;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      MmaTile, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementComp,
      ElementDw, LayoutC, AlignDw,
      ElementDw, LayoutC, AlignDw,
      EpilogueSched, FusionEVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignIO,
      ElementB, LayoutB, AlignIO,
      ElementAcc, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSched>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  auto sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  const int dev = dY.device().index();
  cutlass::KernelHardwareInfo hw;
  hw.device_id = dev;
  hw.sm_count  = cached_sm_count(dev);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA*>(dY.data_ptr()), sA,
       reinterpret_cast<ElementB*>(X.data_ptr()), sB},
      {{}, nullptr, sC, reinterpret_cast<ElementDw*>(output.data_ptr()), sD},
      hw};
  typename FusionEVT::Arguments fargs{ {}, {} };
  args.epilogue.thread = fargs;

  const int64_t kd = decomp    ? *decomp    : 0;   // Heuristic
  const int64_t ks = splits    ? *splits    : 0;   // scheduler default
  const int64_t kr = reduction ? *reduction : 0;   // Deterministic
  set_streamk_knobs(args.scheduler, kd, ks, kr);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Plan keyed by (device, M, N, K, knobs): knobs change the workspace size, so
  // they are part of the key.  Unlike dx there is no update() fast path -- see
  // the header comment on stream-K workspace re-zeroing.
  struct Plan { std::unique_ptr<Gemm> op; torch::Tensor ws; size_t ws_bytes = 0; bool ready = false; };
  using Key = std::tuple<int, int, int, int, int64_t, int64_t, int64_t>;
  static std::mutex* mtx   = new std::mutex();
  static auto*       cache = new std::map<Key, Plan>();

  Key key{dev, M, N, K, kd, ks, kr};
  std::lock_guard<std::mutex> lock(*mtx);
  Plan& plan = (*cache)[key];

  if (!plan.ready) {
    plan.op = std::make_unique<Gemm>();
    plan.ws_bytes = Gemm::get_workspace_size(args);
    plan.ws = plan.ws_bytes
        ? torch::empty({(int64_t)plan.ws_bytes}, dY.options().dtype(torch::kUInt8))
        : torch::Tensor();
    CUTLASS_CHECK(plan.op->can_implement(args));
    plan.ready = true;
  }
  // initialize() on EVERY call, deliberately.  Verified against the CUTLASS
  // source: the stream-K fixup locks are incremented by arrive_inc and the
  // final split of each tile does wait_eq (STRICT equality) + load_add without
  // ever resetting the lock, so a relaunch on a dirty workspace spins forever
  // on intermediate counts the stale value has already passed.  update() also
  // rejects a null workspace (kErrorWorkspaceNull) and would not re-zero it
  // anyway.  initialize() re-zeros only the barrier segment via an async
  // memset on our stream (KB-scale) -- negligible against the dw kernel.
  void* wptr = plan.ws_bytes ? plan.ws.data_ptr() : nullptr;
  CUTLASS_CHECK(plan.op->initialize(args, wptr, stream));
  CUTLASS_CHECK(plan.op->run(stream));
  return output;
}

}  // namespace sm100_ffn_bwd

using sm100_ffn_bwd::run_dx;
using sm100_ffn_bwd::run_dw;
