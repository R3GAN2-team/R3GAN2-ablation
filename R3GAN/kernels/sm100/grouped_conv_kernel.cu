// gconv32_kernel.cu -- kernel TU, ZERO torch headers by design.
// torch/extension.h in the same TU as the kernel templates caused ptxas to
// spill 4 of 8 instantiations at 254 regs / 728-752 B stack (~1.8x slowdown,
// measured; harness-identical flags+nvcc). The binding lives in
// gconv32_binding.cpp and calls these extern "C" entry points.
// Kernel body generated from kernel_harness.cu v2.4-F1e.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;

static constexpr int GSLAB   = 4;               // groups per slab (template knob later)
static constexpr int TH      = 16, TW = 8;      // output tile 16x8 = M 128
static constexpr int HALO_H  = TH + 2, HALO_W = TW + 2;   // 18 x 10
static constexpr int PITCH   = HALO_W * 32;     // 320 elems / halo row / group
static constexpr int IMG_EL  = HALO_H * PITCH;  // 5760 elems per group image
static constexpr int WBLK_EL = 32 * 288;        // per-group packed weight block

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// ---- layouts (G2-proven family) --------------------------------------------
// A window: canonical B64 K-major, halo row pitch in the SBO slot, taps folded
// into NumK = (6,3) : (16, 320)  [(k16,dx) fused contiguous, dy at row pitch]
template <int DY>   // dy tap stride: 320 (16x8 tiles) or 640 (8x8 pair, cfg4)
CUTE_HOST_DEVICE constexpr auto make_win_layout() {
  [[maybe_unused]] auto plain = make_layout(
    make_shape (make_shape(make_shape(Int<8>{}, Int<16>{}), make_shape(Int<8>{}, Int<2>{})),
                Int<1>{}, make_shape(Int<6>{}, Int<3>{})),
    make_stride(make_stride(make_stride(Int<32>{}, Int<PITCH>{}), make_stride(Int<1>{}, Int<8>{})),
                Int<0>{}, make_stride(Int<16>{}, Int<DY>{})));
  return ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag_bits<16>, decltype(plain)>{};
}
static constexpr int IMG_EL8 = 2 * 10 * 10 * 32;   // cfg4: two interleaved 8x8 images

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto make_wblk_layout(TiledMma const& mma) {
  return UMMA::tile_to_mma_shape(UMMA::Layout_K_SW64_Atom<cutlass::bfloat16_t>{},
           partition_shape_B(mma, make_shape(Int<32>{}, Int<288>{})));
}

// TMA view of one group halo image: modes (32ch, (10px, 18row)) so
// make_tma_copy maps them onto the gmem (C, W, H, N) tensor's leading modes.
CUTE_HOST_DEVICE constexpr auto make_img_tma_layout() {
  // v1.41: FLAT rank-3, one smem mode per gmem tile mode (C, W, H) -- the
  // nested (10,18) form risked mispairing against the rank-4 gmem tensor.
  [[maybe_unused]] auto plain = make_layout(
    make_shape (Int<32>{}, Int<HALO_W>{}, Int<HALO_H>{}),
    make_stride(Int<1>{},  Int<32>{},     Int<PITCH>{}));
  return ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag_bits<16>, decltype(plain)>{};
}

CUTE_HOST_DEVICE constexpr auto make_img8_tma_layout() {
  [[maybe_unused]] auto plain = make_layout(
    make_shape (Int<32>{}, Int<10>{}, Int<10>{}),
    make_stride(Int<1>{},  Int<32>{}, Int<640>{}));   // rows interleave 2 images
  return ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag_bits<16>, decltype(plain)>{};
}

template <class TypeAB>
CUTE_HOST auto make_conv_tma8(TypeAB const* dIn, int N, int W, int C) {
  Tensor gA = make_tensor(make_gmem_ptr(dIn),
      make_shape (C, W, 8, N),
      make_stride(Int<1>{}, int64_t(C), int64_t(W) * C, int64_t(8) * W * C));
  return make_tma_copy(SM90_TMA_LOAD{}, gA, make_img8_tma_layout());
}

template <class TypeAB, int GS>
CUTE_HOST auto make_wpk_tma(TypeAB const* dWpk, int C) {
  // wpk = [C/32 blocks][9216 elems], blocks contiguous; one (256, GS*36)
  // box covers a slab's weights. Flat layouts both sides (host pre-permuted).
  Tensor gW = make_tensor(make_gmem_ptr(dWpk),
      make_shape (Int<256>{}, (C / 32) * 36),
      make_stride(Int<1>{},   Int<256>{}));
  auto sW = make_layout(make_shape(Int<256>{}, Int<GS * 36>{}),
                        make_stride(Int<1>{}, Int<256>{}));
  return make_tma_copy(SM90_TMA_LOAD{}, gW, sW);
}

template <class TypeAB>
CUTE_HOST auto make_conv_tma(TypeAB const* dIn, int N, int H, int W, int C) {
  Tensor gA = make_tensor(make_gmem_ptr(dIn),
      make_shape (C, W, H, N),
      make_stride(Int<1>{}, int64_t(C), int64_t(W) * C, int64_t(H) * W * C));
  return make_tma_copy(SM90_TMA_LOAD{}, gA, make_img_tma_layout());
}

template <class TypeAB, class BBlkLayout, int GS>
struct SharedStorageGC
{
  alignas(1024) cute::ArrayEngine<TypeAB, 2 * GS * IMG_EL8> A;           // 2 stages x GS images (8x8-pair size >= 16x8)
  alignas(1024) cute::ArrayEngine<TypeAB, GS * cute::cosize_v<BBlkLayout>> W;
  alignas(16)   cute::uint64_t w_full;          // weights TMA   (expect_tx, count 1)
  alignas(16)   cute::uint64_t stage_full[2];   // loaders -> mma   (count 96)
  alignas(16)   cute::uint64_t mma_done[2];     // umma retirement  (count 1)
  alignas(16)   cute::uint64_t acc_free[2];     // epi tld done     (count 128)
  alignas(16)   cute::uint32_t tmem_base_ptr;
};

// cp.async 16B with zfill: src_size 0 => destination zero-filled. This IS the
// zero padding. cute's Swizzle<2,4,3> on the byte offset so stores land where
// the B64 descriptors read:  off ^ ((off >> 3) & 0x30).
CUTE_DEVICE void cp_async_zfill16(uint32_t smem_byte, void const* gmem, bool valid)
{
  int src_size = valid ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_byte), "l"(gmem), "r"(src_size));
}
CUTE_DEVICE void mbar_arrive(cute::uint64_t& bar) {
  uint32_t a = cast_smem_ptr_to_uint(&bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
CUTE_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// waits use cute::cp_async_wait<N>() from cute/arch/copy_sm80.hpp

template <class SharedStorage, class TypeAB, class BBlkLayout, class TiledMma, class TmaT, class TmaW, int GS, bool IS8, bool FACT = false, bool FSLP = false>
__global__ static void __launch_bounds__(256, GS == 2 ? 2 : 1)
gconv32_device(CUTE_GRID_CONSTANT TmaT const tma,
                CUTE_GRID_CONSTANT TmaW const tmaw,
                TypeAB const* __restrict__ in,      // NHWC
                TypeAB const* __restrict__ wpk,     // [S][GSLAB][32n][288k] packed
                TypeAB* __restrict__ out,           // NHWC
                unsigned long long* __restrict__ prof,   // [10] role timers (block 0)
                TiledMma tiled_mma,
                int N, int H, int W, int C,
                int S, int peers, long items_per_slab,
                float alpha, uint32_t* __restrict__ bits,
                TypeAB const* __restrict__ slope_src)
{
  // v1.0 warp specialization (256 threads):
  //   warp 0        MMA issue only
  //   warps 4-7     epilogue: tmem 32x load (warp%4 == lane group),
  //                 convert+stage to E, coalesced flush
  //   warps 1-3     halo loads (cp.async zfill), incremental addressing
  // TMEM accumulators double-buffered: set p = tile parity, 2x128 of 512 cols.
  // Pipeline barriers (tile parity p couples stage p <-> acc set p):
  //   stage_full[p] : loaders done          -> mma may read stage p
  //   mma_done[p]   : tile-p MMAs retired   -> loaders may refill stage p (j+2)
  //                                            AND epi may drain acc set p
  //   acc_free[p]   : epi tld complete      -> mma may Zero acc set p (j+2)
  long long pk0 = clock64();
  extern __shared__ char smem_raw[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

  int slab = blockIdx.x % S;
  int peer = blockIdx.x / S;
  int c0   = slab * (GS * 32);
  constexpr int IMGE = IS8 ? IMG_EL8 : IMG_EL;
  int tpr  = IS8 ? 1 : W / TW;
  int tpi  = IS8 ? 1 : (H / TH) * tpr;

  int warp = threadIdx.x >> 5;
  bool is_mma  = (warp == 0);
  bool is_epi  = (warp >= 4);                  // threads 128..255: warp%4 == TMEM
                                               // lane group (hardware constraint)

  ThrMMA cta_mma = tiled_mma.get_slice(0);
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(2 * GS * 32, &ss.tmem_base_ptr);   // 2 acc sets x GS groups
  }
  if (threadIdx.x == 0) {
    cute::initialize_barrier(ss.w_full, 1);
    cute::initialize_barrier(ss.stage_full[0], 1);   // expect_tx arrive
    cute::initialize_barrier(ss.stage_full[1], 1);
    cute::initialize_barrier(ss.mma_done[0], 1);
    cute::initialize_barrier(ss.mma_done[1], 1);
    cute::initialize_barrier(ss.acc_free[0], 128);
    cute::initialize_barrier(ss.acc_free[1], 128);
  }
  __syncthreads();

  // ---- weights: one TMA box per CTA (K2; flat both sides since v1.72) ----
  static_assert(cute::cosize_v<BBlkLayout> == WBLK_EL,
                "flat weight load assumes a compact canonical block layout");
  if (threadIdx.x == 0) {
    cute::set_barrier_transaction_bytes(ss.w_full, GS * WBLK_EL * 2);
    Tensor gWc = tmaw.get_tma_tensor(make_shape(Int<256>{}, (C / 32) * 36));
    Tensor gBoxW = domain_offset(make_coord(0, slab * GS * 36),
        make_tensor(gWc.data(), make_shape(Int<256>{}, Int<GS * 36>{}), gWc.stride()));
    Tensor sW = make_tensor(make_smem_ptr(ss.W.begin()),
        make_layout(make_shape(Int<256>{}, Int<GS * 36>{}),
                    make_stride(Int<1>{}, Int<256>{})));
    auto [tG, tS] = tma_partition(tmaw, Int<0>{}, Layout<_1>{},
                                  group_modes<0,2>(sW), group_modes<0,2>(gBoxW));
    copy(tmaw.with(ss.w_full), tG, tS);
  }
  // K3: only the MMA warp reads W -- it alone waits (below), so the loader's
  // first halo TMA and all construction proceed under the weights transfer.
  __syncthreads();
  long long pk1 = clock64();   // barriers + weights resident

  // ---- hoisted MMA fragments (warp 0 uses these) ----
  auto mkA = [&](int st, int g) {
    return cta_mma.make_fragment_A(
        make_tensor(make_smem_ptr(ss.A.begin() + st * (GS * IMGE) + g * IMGE), make_win_layout<IS8 ? 640 : 320>()));
  };
  auto mkB = [&](int g) {
    return cta_mma.make_fragment_B(
        make_tensor(make_smem_ptr(ss.W.begin() + g * cute::cosize_v<BBlkLayout>), BBlkLayout{}));
  };
  using FA = decltype(mkA(0, 0));
  using FB = decltype(mkB(0));
  FA fA[2][GSLAB] = {{mkA(0,0), mkA(0,1), mkA(0,2), mkA(0,3)},
                     {mkA(1,0), mkA(1,1), mkA(1,2), mkA(1,3)}};
  FB fB[GSLAB]    =  {mkB(0),   mkB(1),   mkB(2),   mkB(3)};

  // ---- accumulators: 2 sets x GSLAB, disjoint TMEM columns ----
  auto gD_proto = make_tensor(make_gmem_ptr(out),
      make_layout(make_shape(make_shape(Int<TW>{}, Int<TH>{}), Int<32>{}),
                  make_stride(make_stride(C, W * C), _1{})));
  auto tCgD_proto = cta_mma.partition_C(gD_proto);
  using AccT = decltype(cta_mma.make_fragment_C(tCgD_proto));
  AccT acc[2][GSLAB] = {
    {cta_mma.make_fragment_C(tCgD_proto), cta_mma.make_fragment_C(tCgD_proto),
     cta_mma.make_fragment_C(tCgD_proto), cta_mma.make_fragment_C(tCgD_proto)},
    {cta_mma.make_fragment_C(tCgD_proto), cta_mma.make_fragment_C(tCgD_proto),
     cta_mma.make_fragment_C(tCgD_proto), cta_mma.make_fragment_C(tCgD_proto)}};
  for (int p = 0; p < 2; ++p)
    for (int g = 0; g < GS; ++g)
      acc[p][g].data() = ss.tmem_base_ptr + uint32_t(p * (GS * 32) + g * 32);

  // ---- hoisted epilogue machinery (warps 1-4; lane map at -32 offset) ----
  int tid_e = threadIdx.x - 128;  // epi copy-thread; warp%4 = lane group
  TiledCopy epi_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, acc[0][0]);
  ThrCopy   epi_thr = epi_t2r.get_slice(is_epi ? tid_e : 0);
  auto mk_tdt = [&](int p, int g) { return epi_thr.partition_S(acc[p][g]); };
  auto mk_tde = [&](int g) {   // gmem destination proto (base offset added per tile)
    if constexpr (IS8) {
      // pair path: tmem row m = mx + 8*img + 16*my; gmem (img,my) do NOT
      // flatten (strides 8WC, WC) -> honest nested mode
      Tensor gDg = make_tensor(make_gmem_ptr(out + g * 32),
          make_layout(make_shape (make_shape(Int<8>{}, make_shape(Int<2>{}, Int<8>{})), Int<32>{}),
                      make_stride(make_stride(C, make_stride(8 * W * C, W * C)), _1{})));
      return epi_thr.partition_D(cta_mma.partition_C(gDg));
    } else {
      Tensor gDg = make_tensor(make_gmem_ptr(out + g * 32),
          make_layout(make_shape(make_shape(Int<TW>{}, Int<TH>{}), Int<32>{}),
                      make_stride(make_stride(C, W * C), _1{})));
      return epi_thr.partition_D(cta_mma.partition_C(gDg));
    }
  };
  using TDt = decltype(mk_tdt(0, 0));
  using TDe = decltype(mk_tde(0));
  TDt epi_src[2][GSLAB] = {{mk_tdt(0,0), mk_tdt(0,1), mk_tdt(0,2), mk_tdt(0,3)},
                           {mk_tdt(1,0), mk_tdt(1,1), mk_tdt(1,2), mk_tdt(1,3)}};
  TDe epi_dst[GSLAB]    =  {mk_tde(0),   mk_tde(1),   mk_tde(2),   mk_tde(3)};

  uint32_t stage_byte[2] = {
      cast_smem_ptr_to_uint(ss.A.begin()),
      cast_smem_ptr_to_uint(ss.A.begin() + GS * IMGE)};

  auto item_coords = [&](long it, int& n, int& row0, int& col0) {
    if constexpr (IS8) { n = int(it) * 2; row0 = 0; col0 = 0; return; }  // n = first of pair
    long t = it % tpi;
    n    = int(it / tpi);
    row0 = int(t / tpr) * TH;
    col0 = int(t % tpr) * TW;
  };

  long long pk2 = clock64();   // all hoisted construction done
  if (blockIdx.x == 0 && threadIdx.x == 0 && prof) {
    prof[8] = (unsigned long long)(pk1 - pk0);   // init + weights
    prof[9] = (unsigned long long)(pk2 - pk1);   // cute construction
  }
  long it0 = peer;
  long jmax = (items_per_slab - it0 + peers - 1) / peers;   // tiles this CTA runs
  if (it0 >= items_per_slab) jmax = 0;
  if (jmax == 0) {
    __syncthreads();
    if (elect_one_warp) {
      tmem_allocator.release_allocation_lock();
      tmem_allocator.free(ss.tmem_base_ptr, 2 * GS * 32);
    }
    return;
  }

  unsigned long long p0 = 0, p1 = 0, p2 = 0, p4 = 0, p6 = 0;

  // =========================== LOADERS (warps 5-7) ===========================
  if (threadIdx.x == 32) {
    // v1.4: TMA loader. One thread issues 4 descriptor copies per tile;
    // completion arrives transaction bytes on stage_full (expect_tx).
    // Negative halo origins via domain_offset; TMA zero-fills OOB.
    int ph_md[2] = {0, 0};
    Tensor gFull = tma.get_tma_tensor(make_shape(C, W, H, N));
    for (long j = 0; j < jmax; ++j) {
      int s = int(j & 1);
      long long l0 = clock64();
      if (j >= 2) { cute::wait_barrier(ss.mma_done[s], ph_md[s]); ph_md[s] ^= 1; }
      long long l1 = clock64();
      cute::set_barrier_transaction_bytes(ss.stage_full[s], GS * IMGE * 2);
      int n, r0, cl0; item_coords(it0 + j * peers, n, r0, cl0);
      CUTE_UNROLL
      for (int g = 0; g < GS; ++g) {
        if constexpr (IS8) {   // two per-image boxes; padding never crosses the pair
          CUTE_UNROLL
          for (int im = 0; im < 2; ++im) {
            Tensor gBox = domain_offset(make_coord(c0 + g * 32, -1, -1, n + im),
                make_tensor(gFull.data(),
                            make_shape(Int<32>{}, Int<10>{}, Int<10>{}, Int<1>{}),
                            gFull.stride()));
            Tensor sImg = make_tensor(
                make_smem_ptr(ss.A.begin() + s * (GS * IMGE) + g * IMGE + im * 320),
                make_img8_tma_layout());
            auto [tG, tS] = tma_partition(tma, Int<0>{}, Layout<_1>{},
                                          group_modes<0,3>(sImg), group_modes<0,4>(gBox));
            copy(tma.with(ss.stage_full[s]), tG, tS);
          }
        } else {
          Tensor gBox = domain_offset(make_coord(c0 + g * 32, cl0 - 1, r0 - 1, n),
              make_tensor(gFull.data(),
                          make_shape(Int<32>{}, Int<HALO_W>{}, Int<HALO_H>{}, Int<1>{}),
                          gFull.stride()));
          Tensor sImg = make_tensor(make_smem_ptr(ss.A.begin() + s * (GS * IMGE) + g * IMGE),
                                    make_img_tma_layout());
          auto [tG, tS] = tma_partition(tma, Int<0>{}, Layout<_1>{},
                                        group_modes<0,3>(sImg), group_modes<0,4>(gBox));
          copy(tma.with(ss.stage_full[s]), tG, tS);
        }
      }
      long long l2 = clock64();
      p1 += l1 - l0; p2 += l2 - l1;
    }
  }

  // ============================= MMA (warp 0) ================================
  if (is_mma) {
    cute::wait_barrier(ss.w_full, 0);            // weights resident (K3)
    int ph_sf[2] = {0, 0}, ph_af[2] = {0, 0};
    for (long j = 0; j < jmax; ++j) {
      int s = int(j & 1);
      long long t0 = clock64();
      cute::wait_barrier(ss.stage_full[s], ph_sf[s]); ph_sf[s] ^= 1;
      long long t1 = clock64();
      if (j >= 2) { cute::wait_barrier(ss.acc_free[s], ph_af[s]); ph_af[s] ^= 1; }
      long long t2 = clock64();
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      if (s == 0) {
        for (int k = 0; k < 18; ++k) {
          CUTE_UNROLL
          for (int g = 0; g < GS; ++g)
            cute::gemm(tiled_mma, fA[0][g](_,_,k), fB[g](_,_,k), acc[0][g]);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
      } else {
        for (int k = 0; k < 18; ++k) {
          CUTE_UNROLL
          for (int g = 0; g < GS; ++g)
            cute::gemm(tiled_mma, fA[1][g](_,_,k), fB[g](_,_,k), acc[1][g]);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
      }
      cutlass::arch::umma_arrive(&ss.mma_done[s]);
      long long t3 = clock64();
      if (threadIdx.x == 0) { p0 += t1 - t0; p1 += t2 - t1; p2 += t3 - t2; }
    }
  }

  // =========================== EPILOGUE (warps 1-4) ==========================
  if (is_epi) {
    int ph_md[2] = {0, 0};
    for (long j = 0; j < jmax; ++j) {
      int s = int(j & 1);
      int n, r0, cl0; item_coords(it0 + j * peers, n, r0, cl0);
      long base_el = (((long)n * H + r0) * W + cl0) * C + c0;
      [[maybe_unused]] uint32_t slw[GS][16];
      if constexpr (FSLP) {   // pre-wait register prefetch: hides slope-read latency
        CUTE_UNROLL
        for (int g = 0; g < GS; ++g) {
          const uint4* sp = reinterpret_cast<const uint4*>(
              slope_src + (size_t)(raw_pointer_cast(epi_dst[g].data()) - out) + base_el);
          CUTE_UNROLL
          for (int k = 0; k < 4; ++k) {
            uint4 q = sp[k];
            slw[g][4*k] = q.x; slw[g][4*k+1] = q.y; slw[g][4*k+2] = q.z; slw[g][4*k+3] = q.w;
          }
        }
      }
      long long e0 = clock64();
      cute::wait_barrier(ss.mma_done[s], ph_md[s]); ph_md[s] ^= 1;
      long long e1 = clock64();
      Tensor tDr = make_tensor<float>(shape(epi_dst[0]));
      Tensor tDo = make_tensor<TypeAB>(shape(epi_dst[0]));
      [[maybe_unused]] uint32_t wbv[GS]; [[maybe_unused]] size_t w0off = 0;
      if (s == 0) {
        CUTE_UNROLL
        for (int g = 0; g < GS; ++g) {
          copy(epi_t2r, epi_src[0][g], tDr);
          if (g == GS - 1) mbar_arrive(ss.acc_free[0]);
          Tensor dj = make_tensor(epi_dst[g].data() + base_el, epi_dst[g].layout());
          if constexpr (FSLP) {
            CUTE_UNROLL
            for (int k = 0; k < 16; ++k) {         // prefetched pairs; signs 15, 31
              uint32_t sb = slw[g][k];
              tDr(2 * k)     *= (sb & 0x00008000u) ? alpha : 1.0f;
              tDr(2 * k + 1) *= (sb & 0x80000000u) ? alpha : 1.0f;
            }
          }
          if constexpr (FACT) {
            CUTE_UNROLL
            for (int i = 0; i < size(tDr); ++i) {
              float v = tDr(i);
              tDo(i) = TypeAB(fmaxf(v, alpha * v));    // lrelu, predicate-free
            }
            // sign word from converted bf16 pairs (tDr already dead):
            // u32 k holds tDo(2k) | tDo(2k+1)<<16; ~sign bits at 15, 31
            // channel c <-> bit (c>>1) | ((c&1)<<4)   [documented interleave]
            Tensor tDu = recast<uint32_t>(tDo);
            uint32_t wb = 0;
            CUTE_UNROLL
            for (int k = 0; k < size(tDu); ++k)
              wb |= ((~tDu(k)) & 0x80008000u) >> (15 - k);
            wbv[g] = wb;
            if (g == 0) w0off = (size_t)(raw_pointer_cast(dj.data()) - out) >> 5;
          } else {
            CUTE_UNROLL
            for (int i = 0; i < size(tDr); ++i) tDo(i) = TypeAB(tDr(i));
          }
          copy(tDo, dj);
        }
        if constexpr (FACT) {   // groups g are consecutive bitmap words; 16B/8B aligned (C%128)
          if constexpr (GS == 4)
            *reinterpret_cast<uint4*>(bits + w0off) = make_uint4(wbv[0], wbv[1], wbv[2], wbv[3]);
          else
            *reinterpret_cast<uint2*>(bits + w0off) = make_uint2(wbv[0], wbv[1]);
        }
      } else {
        CUTE_UNROLL
        for (int g = 0; g < GS; ++g) {
          copy(epi_t2r, epi_src[1][g], tDr);
          if (g == GS - 1) mbar_arrive(ss.acc_free[1]);
          Tensor dj = make_tensor(epi_dst[g].data() + base_el, epi_dst[g].layout());
          if constexpr (FSLP) {
            CUTE_UNROLL
            for (int k = 0; k < 16; ++k) {         // prefetched pairs; signs 15, 31
              uint32_t sb = slw[g][k];
              tDr(2 * k)     *= (sb & 0x00008000u) ? alpha : 1.0f;
              tDr(2 * k + 1) *= (sb & 0x80000000u) ? alpha : 1.0f;
            }
          }
          if constexpr (FACT) {
            CUTE_UNROLL
            for (int i = 0; i < size(tDr); ++i) {
              float v = tDr(i);
              tDo(i) = TypeAB(fmaxf(v, alpha * v));    // lrelu, predicate-free
            }
            // sign word from converted bf16 pairs (tDr already dead):
            // u32 k holds tDo(2k) | tDo(2k+1)<<16; ~sign bits at 15, 31
            // channel c <-> bit (c>>1) | ((c&1)<<4)   [documented interleave]
            Tensor tDu = recast<uint32_t>(tDo);
            uint32_t wb = 0;
            CUTE_UNROLL
            for (int k = 0; k < size(tDu); ++k)
              wb |= ((~tDu(k)) & 0x80008000u) >> (15 - k);
            wbv[g] = wb;
            if (g == 0) w0off = (size_t)(raw_pointer_cast(dj.data()) - out) >> 5;
          } else {
            CUTE_UNROLL
            for (int i = 0; i < size(tDr); ++i) tDo(i) = TypeAB(tDr(i));
          }
          copy(tDo, dj);
        }
        if constexpr (FACT) {   // groups g are consecutive bitmap words; 16B/8B aligned (C%128)
          if constexpr (GS == 4)
            *reinterpret_cast<uint4*>(bits + w0off) = make_uint4(wbv[0], wbv[1], wbv[2], wbv[3]);
          else
            *reinterpret_cast<uint2*>(bits + w0off) = make_uint2(wbv[0], wbv[1]);
        }
      }
      long long e2 = clock64();
      if (threadIdx.x == 128) { p4 += e1 - e0; p6 += e2 - e1; }
    }
  }

  __syncthreads();
  if (blockIdx.x == 0 && prof) {
    if (threadIdx.x == 0)  { prof[0] = p0; prof[1] = p1; prof[2] = p2; prof[5] = (unsigned long long)jmax; }
    if (threadIdx.x == 128) { prof[3] = p4; prof[6] = p6; }
    if (threadIdx.x == 32)  { prof[4] = p1; prof[7] = p2; }

  }
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(ss.tmem_base_ptr, 2 * GS * 32);
  }
}


// ============================ torch extension glue ============================
using TypeAB = cutlass::bfloat16_t;

// map caches: activation/weight pointers recycle (allocator / per-step pack)
template <class Maker>
static auto const& cached_tma(Maker make, void const* p, int n, int h, int w, int c)
{
  using T = decltype(make());
  struct K { void const* p; int n, h, w, c;
    bool operator==(K const& o) const { return p==o.p&&n==o.n&&h==o.h&&w==o.w&&c==o.c; } };
  static K  ck[8] = {};
  static T* cv[8] = {};
  static int ci = 0;
  K k{p, n, h, w, c};
  for (int i = 0; i < 8; ++i) if (cv[i] && ck[i] == k) return *cv[i];
  delete cv[ci];
  cv[ci] = new T(make());
  ck[ci] = k;
  T const& r = *cv[ci];
  ci = (ci + 1) & 7;
  return r;
}

template <int GS, bool IS8, bool FACT, bool FSLP, class TmaT, class TmaW>
static void gconv32_launch_i(TmaT const& tma, TmaW const& tmaw,
                             void const* x, void const* wpk, void* y,
                             int N, int H, int W, int C, cudaStream_t stream,
                             float alpha = 0.2f, uint32_t* bits = nullptr,
                             void const* slope_src = nullptr)
{
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeAB, TypeAB, float, 128, 32,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto wblk = make_wblk_layout(tiled_mma);
  using SS = SharedStorageGC<TypeAB, decltype(wblk), GS>;
  auto* kptr = &gconv32_device<SS, TypeAB, decltype(wblk), decltype(tiled_mma),
                                TmaT, TmaW, GS, IS8, FACT, FSLP>;
  static int sms = 0;
  static bool attr_set = false;
  if (!attr_set) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(SS));
    attr_set = true;
  }
  int S = C / (GS * 32);
  long items = IS8 ? (long)N / 2 : (long)N * (H / TH) * (W / TW);
  int  maxcta = sms * (GS == 2 ? 2 : 1);
  int  peers = (int)std::max(1L, std::min((long)std::max(1, maxcta / S), items));
  int  grid  = S * peers;
  // launch path matched to the harness (cudaLaunchKernelEx via cutlass cluster
  // launch, cluster 1x1x1). Plain <<<>>> measured ~1.8x slower for this kernel
  // from the torch process while cuDNN kernels ran at full speed -- mechanism
  // under investigation; this launch is semantically identical.
  cutlass::ClusterLaunchParams params = {
      dim3(grid, 1, 1), dim3(256, 1, 1), dim3(1, 1, 1), (int)sizeof(SS), stream};
  cutlass::launch_kernel_on_cluster(params, (void const*)kptr,
      tma, tmaw, reinterpret_cast<TypeAB const*>(x), reinterpret_cast<TypeAB const*>(wpk),
      reinterpret_cast<TypeAB*>(y), (unsigned long long*)nullptr, tiled_mma,
      N, H, W, C, S, peers, items, alpha, bits,
      reinterpret_cast<TypeAB const*>(slope_src));
}



// ---- C entry points for the binding TU ------------------------------------
extern "C" void gconv32_run(void const* x, void const* wpk, void* y,
                            int N, int H, int W, int C, int gslab,
                            float alpha, uint32_t* bits, void const* slope_src,
                            cudaStream_t stream)
{
  auto xp = reinterpret_cast<TypeAB const*>(x);
  auto wp = reinterpret_cast<TypeAB const*>(wpk);
  bool f  = (bits != nullptr);
  bool sl = (slope_src != nullptr);
  if (H == 8) {
    auto const& tma = cached_tma([&]{ return make_conv_tma8(xp, N, W, C); },
                                 x, N, 8, W, C);
    if (gslab == 2) {
      auto const& tw = cached_tma([&]{ return make_wpk_tma<TypeAB, 2>(wp, C); },
                                  wpk, 2, 0, 0, C);
      if (f)       gconv32_launch_i<2, true , true , false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else if (sl) gconv32_launch_i<2, true , false, true >(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else         gconv32_launch_i<2, true , false, false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
    } else {
      auto const& tw = cached_tma([&]{ return make_wpk_tma<TypeAB, 4>(wp, C); },
                                  wpk, 4, 0, 0, C);
      if (f)       gconv32_launch_i<4, true , true , false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else if (sl) gconv32_launch_i<4, true , false, true >(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else         gconv32_launch_i<4, true , false, false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
    }
  } else {
    auto const& tma = cached_tma([&]{ return make_conv_tma(xp, N, H, W, C); },
                                 x, N, H, W, C);
    if (gslab == 2) {
      auto const& tw = cached_tma([&]{ return make_wpk_tma<TypeAB, 2>(wp, C); },
                                  wpk, 2, 0, 0, C);
      if (f)       gconv32_launch_i<2, false, true , false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else if (sl) gconv32_launch_i<2, false, false, true >(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else         gconv32_launch_i<2, false, false, false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
    } else {
      auto const& tw = cached_tma([&]{ return make_wpk_tma<TypeAB, 4>(wp, C); },
                                  wpk, 4, 0, 0, C);
      if (f)       gconv32_launch_i<4, false, true , false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else if (sl) gconv32_launch_i<4, false, false, true >(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
      else         gconv32_launch_i<4, false, false, false>(tma, tw, x, wpk, y, N, H, W, C, stream, alpha, bits, slope_src);
    }
  }
}

extern "C" void gconv32_pack_order_fill(long long* p)
{
  TiledMMA mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeAB, TypeAB, float, 128, 32,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto l = make_wblk_layout(mma);
  for (int i = 0; i < 32 * 288; ++i) {
    int n = i % 32;
    int k = (i / 32) % 16 + 16 * (i / 512);
    uint32_t byte = uint32_t(l.layout_b()(i)) * 2u;   // fg 24: byte-domain swizzle
    byte ^= ((byte >> 3) & 0x30u);
    p[byte / 2] = (long long)n * 288 + k;
  }
}

#else   // !CUTLASS_ARCH_MMA_SM100_SUPPORTED
extern "C" void gconv32_run(void const*, void const*, void*, int, int, int, int,
                            int, float, uint32_t*, void const*, cudaStream_t)
{ fprintf(stderr, "gconv32: built without SM100 support\n"); abort(); }
extern "C" void gconv32_pack_order_fill(long long*)
{ fprintf(stderr, "gconv32: built without SM100 support\n"); abort(); }
#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
