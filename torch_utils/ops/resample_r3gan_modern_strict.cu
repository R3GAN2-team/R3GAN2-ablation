// Strict modern R3GAN resampling plugin: optimized kernels with strict FP32 gain boundaries.
// Restricted to NHWC/channels-last 2x upsampling with filters <= 8x8.

#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "resample_r3gan_modern_strict.h"

template <class T> struct InternalType;
template <> struct InternalType<double>        { typedef double scalar_t; };
template <> struct InternalType<float>         { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>     { typedef float  scalar_t; };
template <> struct InternalType<c10::BFloat16> { typedef float  scalar_t; };

template <class T, bool Volatile> struct SharedType { typedef T type; };
template <class T> struct SharedType<T, true> { typedef volatile T type; };

static __device__ __forceinline__ int floor_div_modern(int a, int b)
{
    int t = 1 - a / b;
    return (a + t * b) / b - t;
}

// Exact floor(a / 2) with a non-negative remainder.  This is used by the
// modern 2x-only kernels to avoid the generic division helper in the inner
// output loop while preserving the negative-padding edge cases.
static __device__ __forceinline__ int floor_div2_modern(int a)
{
    return (a >= 0) ? (a >> 1) : -((1 - a) >> 1);
}

static __device__ __forceinline__ int64_t offset64_in(const resample_r3gan_modern_debug_kernel_params& p, int x, int y, int c, int n)
{
    return (int64_t)x * p.inStride.x + (int64_t)y * p.inStride.y + (int64_t)c * p.inStride.z + (int64_t)n * p.inStride.w;
}

static __device__ __forceinline__ int64_t offset64_out(const resample_r3gan_modern_debug_kernel_params& p, int x, int y, int c, int n)
{
    return (int64_t)x * p.outStride.x + (int64_t)y * p.outStride.y + (int64_t)c * p.outStride.z + (int64_t)n * p.outStride.w;
}

static __device__ __forceinline__ int offset32_in(const resample_r3gan_modern_debug_kernel_params& p, int x, int y, int c, int n)
{
    return x * p.inStride32.x + y * p.inStride32.y + c * p.inStride32.z + n * p.inStride32.w;
}

static __device__ __forceinline__ int offset32_out(const resample_r3gan_modern_debug_kernel_params& p, int x, int y, int c, int n)
{
    return x * p.outStride32.x + y * p.outStride32.y + c * p.outStride32.z + n * p.outStride32.w;
}

static __device__ __forceinline__ float load_gain(const resample_r3gan_modern_debug_kernel_params& p, int c)
{
    return p.channelGain[(p.channelGainStride == 0) ? 0 : (int64_t)c * p.channelGainStride];
}


template <class T> struct IsBF16ModernDebug { static constexpr bool value = false; };
template <> struct IsBF16ModernDebug<c10::BFloat16> { static constexpr bool value = true; };

static __device__ __forceinline__ float bf16_bits_to_float_modern_debug(uint16_t h)
{
    return __uint_as_float(((uint32_t)h) << 16);
}

static __device__ __forceinline__ uint16_t float_to_bf16_bits_rne_modern_debug(float f)
{
    uint32_t x = __float_as_uint(f);
    uint32_t lsb = (x >> 16) & 1;
    uint32_t bias = 0x7fff + lsb;
    return (uint16_t)((x + bias) >> 16);
}

template <int VecC>
static __device__ __forceinline__ void load_bf16_vec_modern_debug(const void* ptr, int64_t elem_offset, float (&vals)[VecC])
{
    static_assert(VecC % 4 == 0, "VecC must be a multiple of 4 for BF16 vector load");
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(ptr) + elem_offset;
    #pragma unroll
    for (int i = 0; i < VecC; i += 4)
    {
        uint64_t raw = *reinterpret_cast<const uint64_t*>(p16 + i);
        vals[i + 0] = bf16_bits_to_float_modern_debug((uint16_t)( raw        & 0xffffu));
        vals[i + 1] = bf16_bits_to_float_modern_debug((uint16_t)((raw >> 16) & 0xffffu));
        vals[i + 2] = bf16_bits_to_float_modern_debug((uint16_t)((raw >> 32) & 0xffffu));
        vals[i + 3] = bf16_bits_to_float_modern_debug((uint16_t)((raw >> 48) & 0xffffu));
    }
}

template <int VecC>
static __device__ __forceinline__ void store_bf16_vec_modern_debug(void* ptr, int64_t elem_offset, const float (&vals)[VecC])
{
    static_assert(VecC % 4 == 0, "VecC must be a multiple of 4 for BF16 vector store");
    uint16_t* p16 = reinterpret_cast<uint16_t*>(ptr) + elem_offset;
    #pragma unroll
    for (int i = 0; i < VecC; i += 4)
    {
        uint64_t raw = 0;
        raw |= (uint64_t)float_to_bf16_bits_rne_modern_debug(vals[i + 0]);
        raw |= (uint64_t)float_to_bf16_bits_rne_modern_debug(vals[i + 1]) << 16;
        raw |= (uint64_t)float_to_bf16_bits_rne_modern_debug(vals[i + 2]) << 32;
        raw |= (uint64_t)float_to_bf16_bits_rne_modern_debug(vals[i + 3]) << 48;
        *reinterpret_cast<uint64_t*>(p16 + i) = raw;
    }
}

// Shared-memory StyleGAN-like small-filter kernel.
// HasGain=false gives a compiled no-gain baseline.
// GainAtOutput=true tests commuting gain to output side.
// Volatile=true reproduces old-style volatile shared arrays.
template <class T, int upx, int upy, int downx, int downy, int filterW, int filterH,
          int tileOutW, int tileOutH, int loopMinor, bool UseInt32,
          bool Volatile, bool HasGain, bool GainAtOutput>
static __global__ void kernel_small_shared(resample_r3gan_modern_debug_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    typedef typename SharedType<scalar_t, Volatile>::type shared_scalar_t;
    const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;

    __shared__ shared_scalar_t sf[filterH][filterW];
    __shared__ shared_scalar_t sx[tileInH][tileInW][loopMinor];
    __shared__ float s_dgain[loopMinor];

    int minorBase = blockIdx.x;
    int tileOutY = minorBase / p.launchMinor;
    minorBase -= tileOutY * p.launchMinor;
    minorBase *= loopMinor;
    tileOutY *= tileOutH;
    int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
    int64_t majorBase = (int64_t)blockIdx.z * p.loopMajor;
    if (tileOutXBase >= p.outSize.x | tileOutY >= p.outSize.y | majorBase >= p.sizeMajor)
        return;

    for (int tapIdx = threadIdx.x; tapIdx < filterH * filterW; tapIdx += blockDim.x)
    {
        int fy = tapIdx / filterW;
        int fx = tapIdx - fy * filterW;
        scalar_t v = 0;
        if (fx < p.filterSize.x & fy < p.filterSize.y)
        {
            int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
            int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
            v = (scalar_t)p.f[ffx * p.filterStride.x + ffy * p.filterStride.y];
        }
        sf[fy][fx] = v;
    }

    for (int64_t majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    {
        int64_t baseNC = major * (int64_t)p.sizeMinor + minorBase;
        int n = (int)(baseNC / p.inSize.z);
        int baseC = (int)(baseNC - (int64_t)n * p.inSize.z);

        if (p.dgainPartial != nullptr)
        {
            for (int relC = threadIdx.x; relC < loopMinor; relC += blockDim.x)
                s_dgain[relC] = 0.0f;
            __syncthreads();
        }

        for (int loopX = 0, tileOutX = tileOutXBase; loopX < p.loopX & tileOutX < p.outSize.x; loopX++, tileOutX += tileOutW)
        {
            int tileMidX = tileOutX * downx + upx - 1 - p.pad0.x;
            int tileMidY = tileOutY * downy + upy - 1 - p.pad0.y;
            int tileInX = floor_div_modern(tileMidX, upx);
            int tileInY = floor_div_modern(tileMidY, upy);
            __syncthreads();

            for (int inIdx = threadIdx.x; inIdx < tileInH * tileInW * loopMinor; inIdx += blockDim.x)
            {
                int relC = inIdx;
                int relInX = relC / loopMinor;
                int relInY = relInX / tileInW;
                relC -= relInX * loopMinor;
                relInX -= relInY * tileInW;
                int c = baseC + relC;
                int inX = tileInX + relInX;
                int inY = tileInY + relInY;
                scalar_t v = 0;
                if (inX >= 0 & inY >= 0 & inX < p.inSize.x & inY < p.inSize.y & c < p.inSize.z)
                {
                    scalar_t xval;
                    if constexpr (UseInt32)
                        xval = (scalar_t)((const T*)p.x)[(int64_t)offset32_in(p, inX, inY, c, n)];
                    else
                        xval = (scalar_t)((const T*)p.x)[offset64_in(p, inX, inY, c, n)];
                    if constexpr (HasGain && !GainAtOutput)
                        xval *= (scalar_t)load_gain(p, c);
                    v = xval;
                }
                sx[relInY][relInX][relC] = v;
            }

            __syncthreads();
            for (int outIdx = threadIdx.x; outIdx < tileOutH * tileOutW * loopMinor; outIdx += blockDim.x)
            {
                int relC = outIdx;
                int relOutX = relC / loopMinor;
                int relOutY = relOutX / tileOutW;
                relC -= relOutX * loopMinor;
                relOutX -= relOutY * tileOutW;
                int c = baseC + relC;
                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY;

                int midX = tileMidX + relOutX * downx;
                int midY = tileMidY + relOutY * downy;
                int inX = floor_div_modern(midX, upx);
                int inY = floor_div_modern(midY, upy);
                int relInX = inX - tileInX;
                int relInY = inY - tileInY;
                int filterX = (inX + 1) * upx - midX - 1;
                int filterY = (inY + 1) * upy - midY - 1;

                if (outX < p.outSize.x & outY < p.outSize.y & c < p.outSize.z)
                {
                    scalar_t v = 0;
                    #pragma unroll
                    for (int yy = 0; yy < filterH / upy; yy++)
                        #pragma unroll
                        for (int xx = 0; xx < filterW / upx; xx++)
                            v += (scalar_t)sx[relInY + yy][relInX + xx][relC] * (scalar_t)sf[filterY + yy * upy][filterX + xx * upx];
                    if (p.dgainPartial != nullptr)
                    {
                        float base_scaled = (float)v * p.gain;
                        int64_t oo = UseInt32 ? (int64_t)offset32_out(p, outX, outY, c, n) : offset64_out(p, outX, outY, c, n);
                        float other_v = (float)((const T*)p.other)[oo];
                        atomicAdd(&s_dgain[relC], other_v * base_scaled);
                    }
                    if constexpr (HasGain && GainAtOutput)
                        v *= (scalar_t)load_gain(p, c);
                    v *= p.gain;
                    if constexpr (UseInt32)
                        ((T*)p.y)[(int64_t)offset32_out(p, outX, outY, c, n)] = (T)v;
                    else
                        ((T*)p.y)[offset64_out(p, outX, outY, c, n)] = (T)v;
                }
            }
        }
        if (p.dgainPartial != nullptr)
        {
            __syncthreads();
            int minorTile = (int)(minorBase / loopMinor);
            int tileYBlock = blockIdx.x / p.launchMinor;
            int spatialBlock = tileYBlock * gridDim.y + blockIdx.y;
            int64_t partialBase = ((((int64_t)major * p.launchMinor + minorTile) * p.partialNumSpatialBlocks + spatialBlock) * loopMinor);
            for (int relC = threadIdx.x; relC < loopMinor; relC += blockDim.x)
                p.dgainPartial[partialBase + relC] = s_dgain[relC];
        }
    }
}


// Vectorized-C shared-memory kernel.  Each thread handles VecC adjacent NHWC
// channels for a given input/output spatial position.  For BF16 and VecC a
// multiple of 4, the input tile load and final output store use packed 64-bit
// transactions along contiguous C.  Arithmetic and gain remain FP32.
template <class T, int upx, int upy, int downx, int downy, int filterW, int filterH,
          int tileOutW, int tileOutH, int loopMinor, int VecC>
static __global__ void kernel_small_shared_vecC(resample_r3gan_modern_debug_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;
    static_assert(loopMinor % VecC == 0, "loopMinor must be divisible by VecC");
    constexpr int channelGroups = loopMinor / VecC;

    __shared__ scalar_t sf[filterH][filterW];
    __shared__ scalar_t sx[tileInH][tileInW][loopMinor];

    int minorBase = blockIdx.x;
    int tileOutY = minorBase / p.launchMinor;
    minorBase -= tileOutY * p.launchMinor;
    minorBase *= loopMinor;
    tileOutY *= tileOutH;
    int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
    int64_t majorBase = (int64_t)blockIdx.z * p.loopMajor;
    if (tileOutXBase >= p.outSize.x | tileOutY >= p.outSize.y | majorBase >= p.sizeMajor)
        return;

    for (int tapIdx = threadIdx.x; tapIdx < filterH * filterW; tapIdx += blockDim.x)
    {
        int fy = tapIdx / filterW;
        int fx = tapIdx - fy * filterW;
        scalar_t v = 0;
        if (fx < p.filterSize.x & fy < p.filterSize.y)
        {
            int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
            int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
            v = (scalar_t)p.f[ffx * p.filterStride.x + ffy * p.filterStride.y];
        }
        sf[fy][fx] = v;
    }

    for (int64_t majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    {
        int64_t baseNC = major * (int64_t)p.sizeMinor + minorBase;
        int n = (int)(baseNC / p.inSize.z);
        int baseC = (int)(baseNC - (int64_t)n * p.inSize.z);

        for (int loopX = 0, tileOutX = tileOutXBase; loopX < p.loopX & tileOutX < p.outSize.x; loopX++, tileOutX += tileOutW)
        {
            int tileMidX = tileOutX * downx + upx - 1 - p.pad0.x;
            int tileMidY = tileOutY * downy + upy - 1 - p.pad0.y;
            int tileInX = floor_div_modern(tileMidX, upx);
            int tileInY = floor_div_modern(tileMidY, upy);
            __syncthreads();

            for (int inPackIdx = threadIdx.x; inPackIdx < tileInH * tileInW * channelGroups; inPackIdx += blockDim.x)
            {
                int relGroup = inPackIdx;
                int relInX = relGroup / channelGroups;
                int relInY = relInX / tileInW;
                relGroup -= relInX * channelGroups;
                relInX -= relInY * tileInW;
                int relC = relGroup * VecC;
                int c = baseC + relC;
                int inX = tileInX + relInX;
                int inY = tileInY + relInY;

                float vals[VecC];
                #pragma unroll
                for (int i = 0; i < VecC; i++) vals[i] = 0.0f;

                if (inX >= 0 & inY >= 0 & inX < p.inSize.x & inY < p.inSize.y & c < p.inSize.z)
                {
                    constexpr bool can_vec_bf16 = IsBF16ModernDebug<T>::value && (VecC % 4 == 0);
                    bool full_vec = (c + VecC <= p.inSize.z) & (p.inStride32.z == 1) & ((c & 3) == 0);
                    if constexpr (can_vec_bf16)
                    {
                        if (full_vec)
                        {
                            int64_t off = (int64_t)offset32_in(p, inX, inY, c, n);
                            load_bf16_vec_modern_debug<VecC>(p.x, off, vals);
                        }
                        else
                        {
                            #pragma unroll
                            for (int i = 0; i < VecC; i++)
                                if (c + i < p.inSize.z)
                                    vals[i] = (float)((const T*)p.x)[(int64_t)offset32_in(p, inX, inY, c + i, n)];
                        }
                    }
                    else
                    {
                        #pragma unroll
                        for (int i = 0; i < VecC; i++)
                            if (c + i < p.inSize.z)
                                vals[i] = (float)((const T*)p.x)[(int64_t)offset32_in(p, inX, inY, c + i, n)];
                    }

                    #pragma unroll
                    for (int i = 0; i < VecC; i++)
                        if (c + i < p.inSize.z)
                            vals[i] *= load_gain(p, c + i);
                }

                #pragma unroll
                for (int i = 0; i < VecC; i++)
                    sx[relInY][relInX][relC + i] = (scalar_t)vals[i];
            }

            __syncthreads();
            for (int outPackIdx = threadIdx.x; outPackIdx < tileOutH * tileOutW * channelGroups; outPackIdx += blockDim.x)
            {
                int relGroup = outPackIdx;
                int relOutX = relGroup / channelGroups;
                int relOutY = relOutX / tileOutW;
                relGroup -= relOutX * channelGroups;
                relOutX -= relOutY * tileOutW;
                int relC = relGroup * VecC;
                int c = baseC + relC;
                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY;

                int midX = tileMidX + relOutX * downx;
                int midY = tileMidY + relOutY * downy;
                int inX = floor_div_modern(midX, upx);
                int inY = floor_div_modern(midY, upy);
                int relInX = inX - tileInX;
                int relInY = inY - tileInY;
                int filterX = (inX + 1) * upx - midX - 1;
                int filterY = (inY + 1) * upy - midY - 1;

                if (outX < p.outSize.x & outY < p.outSize.y & c < p.outSize.z)
                {
                    float vals[VecC];
                    #pragma unroll
                    for (int i = 0; i < VecC; i++) vals[i] = 0.0f;

                    #pragma unroll
                    for (int yy = 0; yy < filterH / upy; yy++)
                    {
                        #pragma unroll
                        for (int xx = 0; xx < filterW / upx; xx++)
                        {
                            float fv = (float)sf[filterY + yy * upy][filterX + xx * upx];
                            #pragma unroll
                            for (int i = 0; i < VecC; i++)
                                vals[i] += (float)sx[relInY + yy][relInX + xx][relC + i] * fv;
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < VecC; i++) vals[i] *= p.gain;

                    constexpr bool can_vec_bf16 = IsBF16ModernDebug<T>::value && (VecC % 4 == 0);
                    bool full_vec = (c + VecC <= p.outSize.z) & (p.outStride32.z == 1) & ((c & 3) == 0);
                    if constexpr (can_vec_bf16)
                    {
                        if (full_vec)
                        {
                            int64_t off = (int64_t)offset32_out(p, outX, outY, c, n);
                            store_bf16_vec_modern_debug<VecC>(p.y, off, vals);
                        }
                        else
                        {
                            #pragma unroll
                            for (int i = 0; i < VecC; i++)
                                if (c + i < p.outSize.z)
                                    ((T*)p.y)[(int64_t)offset32_out(p, outX, outY, c + i, n)] = (T)vals[i];
                        }
                    }
                    else
                    {
                        #pragma unroll
                        for (int i = 0; i < VecC; i++)
                            if (c + i < p.outSize.z)
                                ((T*)p.y)[(int64_t)offset32_out(p, outX, outY, c + i, n)] = (T)vals[i];
                    }
                }
            }
        }
    }

}

// B200/H100/Ada/Blackwell experimental VecC=4 kernel.
// Compared with kernel_small_shared_vecC, this version folds several production
// assumptions into the generated code:
//   * fixed 2x upsample / down=1, NHWC offsets; UseInt32 selects int32 vs int64 addressing;
//   * VecC=4 only, with one thread producing four adjacent channels;
//   * C is a multiple of loopMinor, so there is no channel-tail path;
//   * per-channel FP32 gain is cached once per block in shared memory;
//   * the shared input tile is stored as float4 instead of scalar float lanes;
//   * the inner output loop uses exact 2x parity math instead of generic floor_div.
// The C++ launcher guards the C % loopMinor precondition for modes that call this
// kernel.  Keep the older vec4 modes as correctness/performance fallbacks.
template <class T, int filterW, int filterH, int tileOutW, int tileOutH, int loopMinor, bool UseInt32=true, bool GainAtOutput=false>
static __global__ void kernel_small_shared_vec4s(resample_r3gan_modern_debug_kernel_params p)
{
    constexpr int VecC = 4;
    constexpr int channelGroups = loopMinor / VecC;
    static_assert(loopMinor % VecC == 0, "loopMinor must be divisible by 4");
    const int tileInW = ((tileOutW - 1) + filterW - 1) / 2 + 1;
    const int tileInH = ((tileOutH - 1) + filterH - 1) / 2 + 1;

    __shared__ float sf[filterH][filterW];
    __shared__ float sgain[loopMinor];
    __shared__ float s_dgain[loopMinor];
    __shared__ float4 sx4[tileInH][tileInW][channelGroups];

    const T* __restrict__ xptr = reinterpret_cast<const T*>(p.x);
    T* __restrict__ yptr = reinterpret_cast<T*>(p.y);
    const float* __restrict__ fptr = p.f;
    const float* __restrict__ gptr = p.channelGain;
    (void)xptr; (void)yptr; (void)fptr; (void)gptr;

    int minorBase = blockIdx.x;
    int tileOutY = minorBase / p.launchMinor;
    minorBase -= tileOutY * p.launchMinor;
    minorBase *= loopMinor;
    tileOutY *= tileOutH;
    int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
    int64_t majorBase = (int64_t)blockIdx.z * p.loopMajor;
    if (tileOutXBase >= p.outSize.x | tileOutY >= p.outSize.y | majorBase >= p.sizeMajor)
        return;

    for (int tapIdx = threadIdx.x; tapIdx < filterH * filterW; tapIdx += blockDim.x)
    {
        int fy = tapIdx / filterW;
        int fx = tapIdx - fy * filterW;
        float v = 0.0f;
        if (fx < p.filterSize.x & fy < p.filterSize.y)
        {
            int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
            int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
            v = fptr[ffx * p.filterStride.x + ffy * p.filterStride.y];
        }
        sf[fy][fx] = v;
    }

    for (int64_t majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    {
        int64_t baseNC = major * (int64_t)p.sizeMinor + minorBase;
        int n = (int)(baseNC / p.inSize.z);
        int baseC = (int)(baseNC - (int64_t)n * p.inSize.z);

        if (p.dgainPartial != nullptr)
        {
            for (int relC = threadIdx.x; relC < loopMinor; relC += blockDim.x)
                s_dgain[relC] = 0.0f;
            __syncthreads();
        }

        // Cache the gain vector once per channel tile.  This removes repeated
        // global/L1 gain loads from every spatial input tile element.
        for (int relC = threadIdx.x; relC < loopMinor; relC += blockDim.x)
        {
            int c = baseC + relC;
            sgain[relC] = gptr[(p.channelGainStride == 0) ? 0 : (int64_t)c * p.channelGainStride];
        }

        for (int loopX = 0, tileOutX = tileOutXBase; loopX < p.loopX & tileOutX < p.outSize.x; loopX++, tileOutX += tileOutW)
        {
            int tileMidX = tileOutX + 1 - p.pad0.x;
            int tileMidY = tileOutY + 1 - p.pad0.y;
            int tileInX = floor_div2_modern(tileMidX);
            int tileInY = floor_div2_modern(tileMidY);
            __syncthreads();

            for (int inPackIdx = threadIdx.x; inPackIdx < tileInH * tileInW * channelGroups; inPackIdx += blockDim.x)
            {
                int relGroup = inPackIdx;
                int relInX = relGroup / channelGroups;
                int relInY = relInX / tileInW;
                relGroup -= relInX * channelGroups;
                relInX -= relInY * tileInW;
                int relC = relGroup * VecC;
                int c = baseC + relC;
                int inX = tileInX + relInX;
                int inY = tileInY + relInY;

                float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (inX >= 0 & inY >= 0 & inX < p.inSize.x & inY < p.inSize.y)
                {
                    if constexpr (IsBF16ModernDebug<T>::value)
                    {
                        float vals[VecC];
                        int64_t off = UseInt32 ? (int64_t)offset32_in(p, inX, inY, c, n) : offset64_in(p, inX, inY, c, n);
                        load_bf16_vec_modern_debug<VecC>(p.x, off, vals);
                        xv.x = vals[0]; xv.y = vals[1]; xv.z = vals[2]; xv.w = vals[3];
                    }
                    else
                    {
                        xv.x = (float)xptr[UseInt32 ? (int64_t)offset32_in(p, inX, inY, c + 0, n) : offset64_in(p, inX, inY, c + 0, n)];
                        xv.y = (float)xptr[UseInt32 ? (int64_t)offset32_in(p, inX, inY, c + 1, n) : offset64_in(p, inX, inY, c + 1, n)];
                        xv.z = (float)xptr[UseInt32 ? (int64_t)offset32_in(p, inX, inY, c + 2, n) : offset64_in(p, inX, inY, c + 2, n)];
                        xv.w = (float)xptr[UseInt32 ? (int64_t)offset32_in(p, inX, inY, c + 3, n) : offset64_in(p, inX, inY, c + 3, n)];
                    }
                    if constexpr (!GainAtOutput)
                    {
                        xv.x *= sgain[relC + 0];
                        xv.y *= sgain[relC + 1];
                        xv.z *= sgain[relC + 2];
                        xv.w *= sgain[relC + 3];
                    }
                }
                sx4[relInY][relInX][relGroup] = xv;
            }

            __syncthreads();
            for (int outPackIdx = threadIdx.x; outPackIdx < tileOutH * tileOutW * channelGroups; outPackIdx += blockDim.x)
            {
                int relGroup = outPackIdx;
                int relOutX = relGroup / channelGroups;
                int relOutY = relOutX / tileOutW;
                relGroup -= relOutX * channelGroups;
                relOutX -= relOutY * tileOutW;
                int relC = relGroup * VecC;
                int c = baseC + relC;
                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY;

                if (outX < p.outSize.x & outY < p.outSize.y)
                {
                    int midX = tileMidX + relOutX;
                    int midY = tileMidY + relOutY;
                    int inX = floor_div2_modern(midX);
                    int inY = floor_div2_modern(midY);
                    int relInX = inX - tileInX;
                    int relInY = inY - tileInY;
                    int filterX = 1 - (midX - (inX << 1));
                    int filterY = 1 - (midY - (inY << 1));

                    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    #pragma unroll
                    for (int yy = 0; yy < filterH / 2; yy++)
                    {
                        #pragma unroll
                        for (int xx = 0; xx < filterW / 2; xx++)
                        {
                            float fv = sf[filterY + yy * 2][filterX + xx * 2];
                            float4 sv = sx4[relInY + yy][relInX + xx][relGroup];
                            acc.x += sv.x * fv;
                            acc.y += sv.y * fv;
                            acc.z += sv.z * fv;
                            acc.w += sv.w * fv;
                        }
                    }
                    if (p.dgainPartial != nullptr)
                    {
                        int64_t oo = UseInt32 ? (int64_t)offset32_out(p, outX, outY, c, n) : offset64_out(p, outX, outY, c, n);
                        const T* other = (const T*)p.other;
                        atomicAdd(&s_dgain[relC + 0], (float)other[oo + 0] * (acc.x * p.gain));
                        atomicAdd(&s_dgain[relC + 1], (float)other[oo + 1] * (acc.y * p.gain));
                        atomicAdd(&s_dgain[relC + 2], (float)other[oo + 2] * (acc.z * p.gain));
                        atomicAdd(&s_dgain[relC + 3], (float)other[oo + 3] * (acc.w * p.gain));
                    }
                    if constexpr (GainAtOutput)
                    {
                        acc.x *= sgain[relC + 0];
                        acc.y *= sgain[relC + 1];
                        acc.z *= sgain[relC + 2];
                        acc.w *= sgain[relC + 3];
                    }
                    acc.x *= p.gain;
                    acc.y *= p.gain;
                    acc.z *= p.gain;
                    acc.w *= p.gain;

                    if constexpr (IsBF16ModernDebug<T>::value)
                    {
                        float vals[VecC] = {acc.x, acc.y, acc.z, acc.w};
                        int64_t off = UseInt32 ? (int64_t)offset32_out(p, outX, outY, c, n) : offset64_out(p, outX, outY, c, n);
                        store_bf16_vec_modern_debug<VecC>(p.y, off, vals);
                    }
                    else
                    {
                        yptr[UseInt32 ? (int64_t)offset32_out(p, outX, outY, c + 0, n) : offset64_out(p, outX, outY, c + 0, n)] = (T)acc.x;
                        yptr[UseInt32 ? (int64_t)offset32_out(p, outX, outY, c + 1, n) : offset64_out(p, outX, outY, c + 1, n)] = (T)acc.y;
                        yptr[UseInt32 ? (int64_t)offset32_out(p, outX, outY, c + 2, n) : offset64_out(p, outX, outY, c + 2, n)] = (T)acc.z;
                        yptr[UseInt32 ? (int64_t)offset32_out(p, outX, outY, c + 3, n) : offset64_out(p, outX, outY, c + 3, n)] = (T)acc.w;
                    }
                }
            }
        }
        if (p.dgainPartial != nullptr)
        {
            __syncthreads();
            int minorTile = (int)(minorBase / loopMinor);
            int tileYBlock = blockIdx.x / p.launchMinor;
            int spatialBlock = tileYBlock * gridDim.y + blockIdx.y;
            int64_t partialBase = ((((int64_t)major * p.launchMinor + minorTile) * p.partialNumSpatialBlocks + spatialBlock) * loopMinor);
            for (int relC = threadIdx.x; relC < loopMinor; relC += blockDim.x)
                p.dgainPartial[partialBase + relC] = s_dgain[relC];
        }
    }
}

// Direct global-memory kernel: tests whether shared-memory staging/syncs are worthwhile.
template <class T, int filterW, int filterH, bool UseInt32, bool HasGain, bool InteriorFast>
static __global__ void kernel_global(resample_r3gan_modern_debug_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    int64_t total = (int64_t)p.outSize.w * p.outSize.z * p.outSize.y * p.outSize.x;
    int64_t block_linear = ((int64_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    int64_t grid_linear = (int64_t)gridDim.x * gridDim.y * gridDim.z;
    for (int64_t linear = block_linear * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * grid_linear)
    {
        int outX = (int)(linear % p.outSize.x);
        int64_t t = linear / p.outSize.x;
        int outY = (int)(t % p.outSize.y); t /= p.outSize.y;
        int c = (int)(t % p.outSize.z);
        int n = (int)(t / p.outSize.z);

        int midX = outX + 1 - p.pad0.x; // up=2, down=1
        int midY = outY + 1 - p.pad0.y;
        int inX = floor_div_modern(midX, 2);
        int inY = floor_div_modern(midY, 2);
        int filterX = (inX + 1) * 2 - midX - 1;
        int filterY = (inY + 1) * 2 - midY - 1;

        scalar_t v = 0;
        constexpr int tapsX = filterW / 2;
        constexpr int tapsY = filterH / 2;
        bool interior = false;
        if constexpr (InteriorFast)
            interior = (inX >= 0 && inY >= 0 && inX + tapsX - 1 < p.inSize.x && inY + tapsY - 1 < p.inSize.y && c < p.inSize.z);

        if constexpr (InteriorFast)
        {
            if (interior)
            {
                #pragma unroll
                for (int yy = 0; yy < tapsY; yy++)
                    #pragma unroll
                    for (int xx = 0; xx < tapsX; xx++)
                    {
                        int ix = inX + xx;
                        int iy = inY + yy;
                        scalar_t xval;
                        if constexpr (UseInt32)
                            xval = (scalar_t)((const T*)p.x)[(int64_t)offset32_in(p, ix, iy, c, n)];
                        else
                            xval = (scalar_t)((const T*)p.x)[offset64_in(p, ix, iy, c, n)];
                        if constexpr (HasGain)
                            xval *= (scalar_t)load_gain(p, c);
                        int fx = filterX + xx * 2;
                        int fy = filterY + yy * 2;
                        int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
                        int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
                        scalar_t fv = 0;
                        if (fx < p.filterSize.x & fy < p.filterSize.y)
                            fv = (scalar_t)p.f[ffx * p.filterStride.x + ffy * p.filterStride.y];
                        v += xval * fv;
                    }
            }
            else
            {
                #pragma unroll
                for (int yy = 0; yy < tapsY; yy++)
                    #pragma unroll
                    for (int xx = 0; xx < tapsX; xx++)
                    {
                        int ix = inX + xx;
                        int iy = inY + yy;
                        int fx = filterX + xx * 2;
                        int fy = filterY + yy * 2;
                        if (ix >= 0 & iy >= 0 & ix < p.inSize.x & iy < p.inSize.y & fx < p.filterSize.x & fy < p.filterSize.y)
                        {
                            scalar_t xval;
                            if constexpr (UseInt32)
                                xval = (scalar_t)((const T*)p.x)[(int64_t)offset32_in(p, ix, iy, c, n)];
                            else
                                xval = (scalar_t)((const T*)p.x)[offset64_in(p, ix, iy, c, n)];
                            if constexpr (HasGain)
                                xval *= (scalar_t)load_gain(p, c);
                            int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
                            int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
                            v += xval * (scalar_t)p.f[ffx * p.filterStride.x + ffy * p.filterStride.y];
                        }
                    }
            }
        }
        else
        {
            #pragma unroll
            for (int yy = 0; yy < tapsY; yy++)
                #pragma unroll
                for (int xx = 0; xx < tapsX; xx++)
                {
                    int ix = inX + xx;
                    int iy = inY + yy;
                    int fx = filterX + xx * 2;
                    int fy = filterY + yy * 2;
                    if (ix >= 0 & iy >= 0 & ix < p.inSize.x & iy < p.inSize.y & fx < p.filterSize.x & fy < p.filterSize.y)
                    {
                        scalar_t xval;
                        if constexpr (UseInt32)
                            xval = (scalar_t)((const T*)p.x)[(int64_t)offset32_in(p, ix, iy, c, n)];
                        else
                            xval = (scalar_t)((const T*)p.x)[offset64_in(p, ix, iy, c, n)];
                        if constexpr (HasGain)
                            xval *= (scalar_t)load_gain(p, c);
                        int ffx = (p.flip) ? fx : p.filterSize.x - 1 - fx;
                        int ffy = (p.flip) ? fy : p.filterSize.y - 1 - fy;
                        v += xval * (scalar_t)p.f[ffx * p.filterStride.x + ffy * p.filterStride.y];
                    }
                }
        }

        v *= p.gain;
        if constexpr (UseInt32)
            ((T*)p.y)[(int64_t)offset32_out(p, outX, outY, c, n)] = (T)v;
        else
            ((T*)p.y)[offset64_out(p, outX, outY, c, n)] = (T)v;
    }
}

template <class T, int filterW, int filterH, int Mode>
static __host__ resample_r3gan_modern_debug_kernel_spec pick_mode_filter(const resample_r3gan_modern_debug_kernel_params& p)
{
    (void)p;
    // modes:
    // Clean production modes.  These are the only modes selected by the Python wrapper.
    if constexpr (Mode == 100)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,32,16,32,true,false>,32,16,32,1,256};
    if constexpr (Mode == 101)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,16,32,32,true,false>,16,32,32,1,512};
    if constexpr (Mode == 102)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,32,16,32,false,false>,32,16,32,1,256};
    if constexpr (Mode == 103)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,16,32,32,false,false>,16,32,32,1,512};
    if constexpr (Mode == 104)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,true,false,true,false>,8,8,16,1,256};
    if constexpr (Mode == 105)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,false,false,true,false>,8,8,16,1,256};
    if constexpr (Mode == 106)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,32,16,32,true,true>,32,16,32,1,256};
    if constexpr (Mode == 107)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,16,32,32,true,true>,16,32,32,1,512};
    if constexpr (Mode == 108)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,32,16,32,false,true>,32,16,32,1,256};
    if constexpr (Mode == 109)
        return {(void*)kernel_small_shared_vec4s<T,filterW,filterH,16,32,32,false,true>,16,32,32,1,512};
    if constexpr (Mode == 110)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,true,false,true,true>,8,8,16,1,256};
    if constexpr (Mode == 111)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,false,false,true,true>,8,8,16,1,256};
    if constexpr (Mode == 112)
        return {(void*)kernel_small_shared<T,2,2,1,1,filterW,filterH,16,16,16,true,false,false,false>,16,16,16,1,256};
    if constexpr (Mode == 113)
        return {(void*)kernel_small_shared<T,2,2,1,1,filterW,filterH,16,16,16,false,false,false,false>,16,16,16,1,256};
    if constexpr (Mode == 114)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,true,false,false,false>,8,8,16,1,256};
    if constexpr (Mode == 115)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,false,false,false,false>,8,8,16,1,256};
    if constexpr (Mode == 116)
        return {(void*)kernel_small_shared<T,2,2,1,1,filterW,filterH,16,16,16,true,false,true,false>,16,16,16,1,256};
    if constexpr (Mode == 117)
        return {(void*)kernel_small_shared<T,2,2,1,1,filterW,filterH,16,16,16,false,false,true,false>,16,16,16,1,256};
    if constexpr (Mode == 118)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,true,false,true,false>,8,8,16,1,256};
    if constexpr (Mode == 119)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,false,false,true,false>,8,8,16,1,256};
    if constexpr (Mode == 120)
        return {(void*)kernel_small_shared<T,2,2,1,1,filterW,filterH,16,16,16,true,false,true,true>,16,16,16,1,256};
    if constexpr (Mode == 121)
        return {(void*)kernel_small_shared<T,2,2,1,1,filterW,filterH,16,16,16,false,false,true,true>,16,16,16,1,256};
    if constexpr (Mode == 122)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,true,false,true,true>,8,8,16,1,256};
    if constexpr (Mode == 123)
        return {(void*)kernel_small_shared<T,1,1,2,2,filterW,filterH,8,8,16,false,false,true,true>,8,8,16,1,256};

    return {nullptr,-1,-1,1,1,256};
}

template <class T, int Mode>
static __host__ resample_r3gan_modern_debug_kernel_spec pick_filter(const resample_r3gan_modern_debug_kernel_params& p)
{
    if (p.filterSize.x <= 4 && p.filterSize.y <= 4)
        return pick_mode_filter<T,4,4,Mode>(p);
    if (p.filterSize.x <= 6 && p.filterSize.y <= 6)
        return pick_mode_filter<T,6,6,Mode>(p);
    if (p.filterSize.x <= 8 && p.filterSize.y <= 8)
        return pick_mode_filter<T,8,8,Mode>(p);
    return {nullptr,-1,-1,1,1,256};
}

template <class T>
resample_r3gan_modern_debug_kernel_spec choose_resample_r3gan_modern_debug_kernel(const resample_r3gan_modern_debug_kernel_params& p)
{
    switch (p.mode)
    {
        case 100: return pick_filter<T,100>(p);
        case 101: return pick_filter<T,101>(p);
        case 102: return pick_filter<T,102>(p);
        case 103: return pick_filter<T,103>(p);
        case 104: return pick_filter<T,104>(p);
        case 105: return pick_filter<T,105>(p);
        case 106: return pick_filter<T,106>(p);
        case 107: return pick_filter<T,107>(p);
        case 108: return pick_filter<T,108>(p);
        case 109: return pick_filter<T,109>(p);
        case 110: return pick_filter<T,110>(p);
        case 111: return pick_filter<T,111>(p);
        case 112: return pick_filter<T,112>(p);
        case 113: return pick_filter<T,113>(p);
        case 114: return pick_filter<T,114>(p);
        case 115: return pick_filter<T,115>(p);
        case 116: return pick_filter<T,116>(p);
        case 117: return pick_filter<T,117>(p);
        case 118: return pick_filter<T,118>(p);
        case 119: return pick_filter<T,119>(p);
        case 120: return pick_filter<T,120>(p);
        case 121: return pick_filter<T,121>(p);
        case 122: return pick_filter<T,122>(p);
        case 123: return pick_filter<T,123>(p);
        default: return {nullptr,-1,-1,1,1,256};
    }
}




static inline bool is_strict_nhwc_contiguous(torch::Tensor t);

// ---------------------------------------------------------------------------
// Fused output-gain resample + channel-gain gradient.
// Computes base_fp32 = R(x)_fp32 once inside the output-gain resample kernel.
// The output is cast(base_fp32 * channel_gain * gain).  The gain gradient uses
// sum(other * (base_fp32 * gain)) from the same per-output FP32 base, without
// materializing base_fp32 as a tensor.  Only the small partial-sum buffer is
// materialized.
// ---------------------------------------------------------------------------

template <int THREADS>
static __global__ void kernel_outgain_both_partial_reduce(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int launchMajor,
    int launchMinor,
    int numSpatialBlocks,
    int loopMinor,
    int C)
{
    int c = blockIdx.x;
    if (c >= C) return;
    int minorTile = c / loopMinor;
    int relC = c - minorTile * loopMinor;
    float sum = 0.0f;
    int64_t num = (int64_t)launchMajor * numSpatialBlocks;
    for (int64_t idx = threadIdx.x; idx < num; idx += blockDim.x)
    {
        int spatial = (int)(idx % numSpatialBlocks);
        int major = (int)(idx / numSpatialBlocks);
        int64_t off = ((((int64_t)major * launchMinor + minorTile) * numSpatialBlocks + spatial) * loopMinor + relC);
        sum += partial[off];
    }
    __shared__ float sh[THREADS];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[c] = sh[0];
}

static inline bool storage_footprint_fits_int32_cuda(const torch::Tensor& t)
{
    int64_t max_offset = 0;
    for (int d = 0; d < t.dim(); d++)
        max_offset += (t.size(d) - 1) * t.stride(d);
    return max_offset <= std::numeric_limits<int>::max();
}

static inline bool mode_uses_int32_cuda(int mode)
{
    switch (mode)
    {
        case 102: case 103: case 105: case 108: case 109: case 111:
        case 113: case 115: case 117: case 119: case 121: case 123:
            return false;
        default:
            return true;
    }
}

static inline int mode_int64_equivalent_cuda(int mode)
{
    switch (mode)
    {
        case 100: return 102; case 101: return 103; case 104: return 105;
        case 106: return 108; case 107: return 109; case 110: return 111;
        case 112: return 113; case 114: return 115; case 116: return 117;
        case 118: return 119; case 120: return 121; case 122: return 123;
        default: return mode;
    }
}

static inline bool mode_is_output_gain_cuda(int mode)
{
    switch (mode)
    {
        case 106: case 107: case 108: case 109: case 110: case 111:
        case 120: case 121: case 122: case 123:
            return true;
        default:
            return false;
    }
}

std::vector<torch::Tensor> resample_r3gan_modern_outgain_both_cuda(
    torch::Tensor x,
    torch::Tensor other,
    torch::Tensor f,
    torch::Tensor channel_gain,
    int upx,
    int upy,
    int downx,
    int downy,
    int padx0,
    int padx1,
    int pady0,
    int pady1,
    bool flip,
    float gain,
    int mode)
{
    TORCH_CHECK(x.is_cuda() && other.is_cuda() && f.is_cuda() && channel_gain.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.device() == other.device() && x.device() == f.device() && x.device() == channel_gain.device(), "all tensors must be on same device");
    TORCH_CHECK(x.scalar_type() == other.scalar_type(), "x and other must have same dtype");
    TORCH_CHECK(f.dtype() == torch::kFloat32, "f must be float32");
    TORCH_CHECK(channel_gain.dtype() == torch::kFloat32, "channel_gain must be float32");
    TORCH_CHECK(x.dim() == 4 && f.dim() == 2 && other.dim() == 4, "x/other must be rank 4 and f rank 2");
    TORCH_CHECK(is_strict_nhwc_contiguous(x), "x must be strict NHWC/channels_last contiguous");
    TORCH_CHECK(is_strict_nhwc_contiguous(other), "other must be strict NHWC/channels_last contiguous");
    TORCH_CHECK(mode_is_output_gain_cuda(mode), "fused outgain-both requires an output-gain mode");
    TORCH_CHECK(channel_gain.numel() == x.size(1), "fused outgain-both currently supports vector gain [C] only");
    TORCH_CHECK(channel_gain.is_contiguous(), "channel_gain must be contiguous");
    const bool is_up2 = (upx == 2 && upy == 2 && downx == 1 && downy == 1);
    const bool is_down2 = (upx == 1 && upy == 1 && downx == 2 && downy == 2);
    TORCH_CHECK(is_up2 || is_down2, "fused outgain-both supports 2x up or 2x down only");
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1 && f.size(0) <= 8 && f.size(1) <= 8, "filter must be 1..8 in each dimension");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int64_t outW64 = (x.size(3) * (int64_t)upx + padx0 + padx1 - f.size(1) + downx) / downx;
    int64_t outH64 = (x.size(2) * (int64_t)upy + pady0 + pady1 - f.size(0) + downy) / downy;
    TORCH_CHECK(outW64 >= 1 && outH64 >= 1, "output must be at least 1x1");
    TORCH_CHECK(other.size(0) == x.size(0) && other.size(1) == x.size(1) && other.size(2) == outH64 && other.size(3) == outW64,
        "other must match output shape [N,C,outH,outW]");

    auto y = torch::empty({x.size(0), x.size(1), outH64, outW64}, x.options(), x.suggest_memory_format());
    TORCH_CHECK(is_strict_nhwc_contiguous(y), "fused output y must be strict NHWC contiguous");
    if (mode_uses_int32_cuda(mode) && (!storage_footprint_fits_int32_cuda(x) || !storage_footprint_fits_int32_cuda(y)))
        mode = mode_int64_equivalent_cuda(mode);

    resample_r3gan_modern_debug_kernel_params p;
    p.x                 = x.data_ptr();
    p.f                 = f.data_ptr<float>();
    p.channelGain       = channel_gain.data_ptr<float>();
    p.other             = other.data_ptr();
    p.y                 = y.data_ptr();
    p.dgainPartial      = nullptr;
    p.up                = make_int2(upx, upy);
    p.down              = make_int2(downx, downy);
    p.pad0              = make_int2(padx0, pady0);
    p.flip              = flip ? 1 : 0;
    p.gain              = gain;
    p.inSize            = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.inStride          = make_int64_4_modern_debug(x.stride(3), x.stride(2), x.stride(1), x.stride(0));
    p.inStride32        = make_int4((int)x.stride(3), (int)x.stride(2), (int)x.stride(1), (int)x.stride(0));
    p.filterSize        = make_int2((int)f.size(1), (int)f.size(0));
    p.filterStride      = make_int2((int)f.stride(1), (int)f.stride(0));
    p.outSize           = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.outStride         = make_int64_4_modern_debug(y.stride(3), y.stride(2), y.stride(1), y.stride(0));
    p.outStride32       = make_int4((int)y.stride(3), (int)y.stride(2), (int)y.stride(1), (int)y.stride(0));
    p.sizeMajor         = (p.inStride.z == 1) ? (int64_t)p.inSize.w : (int64_t)p.inSize.w * (int64_t)p.inSize.z;
    p.sizeMinor         = (p.inStride.z == 1) ? p.inSize.z : 1;
    p.channelGainStride = channel_gain.stride(0);
    p.mode              = mode;

    resample_r3gan_modern_debug_kernel_spec spec;
    switch (x.scalar_type())
    {
        case at::ScalarType::BFloat16: spec = choose_resample_r3gan_modern_debug_kernel<c10::BFloat16>(p); break;
        case at::ScalarType::Half:     spec = choose_resample_r3gan_modern_debug_kernel<c10::Half>(p); break;
        case at::ScalarType::Float:    spec = choose_resample_r3gan_modern_debug_kernel<float>(p); break;
        default: TORCH_CHECK(false, "resample_r3gan_modern_outgain_both supports only bf16/fp16/fp32");
    }
    TORCH_CHECK(spec.kernel != nullptr, "unsupported fused outgain-both mode/filter/layout combination");
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor     = spec.loopMinor;
    p.loopX         = spec.loopX;
    p.launchMinor   = (p.sizeMinor - 1) / p.loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;
    dim3 blockSize(spec.blockX, 1, 1);
    dim3 gridSize(
        ((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
        (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1,
        (unsigned int)p.launchMajor);
    int tileYBlocks = gridSize.x / p.launchMinor;
    int numSpatialBlocks = tileYBlocks * (int)gridSize.y;
    p.partialNumSpatialBlocks = numSpatialBlocks;
    auto partial = torch::empty({(int64_t)p.sizeMajor * p.launchMinor * numSpatialBlocks, p.loopMinor}, x.options().dtype(torch::kFloat32));
    auto dgain = torch::empty({x.size(1)}, x.options().dtype(torch::kFloat32));
    p.dgainPartial = partial.data_ptr<float>();

    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    AT_CUDA_CHECK(cudaGetLastError());
    kernel_outgain_both_partial_reduce<256><<<dim3((unsigned int)x.size(1),1,1), dim3(256,1,1), 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), dgain.data_ptr<float>(), (int)p.sizeMajor, p.launchMinor, numSpatialBlocks, p.loopMinor, (int)x.size(1));
    AT_CUDA_CHECK(cudaGetLastError());
    return {y, dgain};
}

// ---------------------------------------------------------------------------
// Gain-gradient reduction for y = R(x * gain).  Backward computes the spatial
// adjoint base_dx = R^T(dy).  This kernel computes
//     dGain[c] = sum_{n,h,w} base_dx[n,c,h,w] * x[n,c,h,w]
// without materializing the product tensor.  It is generic NHWC/NCHW-stride
// aware, but optimized for channels_last where stride(1)==1.
// ---------------------------------------------------------------------------

template <class T>
static __global__ void kernel_gain_grad_channel(
    const T* __restrict__ base,
    const T* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int64_t bs0, int64_t bs1, int64_t bs2, int64_t bs3,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3)
{
    int c = blockIdx.x;
    float sum = 0.0f;
    int64_t total = (int64_t)N * H * W;
    for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        int w = (int)(idx % W);
        int64_t t = idx / W;
        int h = (int)(t % H);
        int n = (int)(t / H);
        int64_t bo = (int64_t)n * bs0 + (int64_t)c * bs1 + (int64_t)h * bs2 + (int64_t)w * bs3;
        int64_t xo = (int64_t)n * xs0 + (int64_t)c * xs1 + (int64_t)h * xs2 + (int64_t)w * xs3;
        sum += (float)base[bo] * (float)x[xo];
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[c] = sh[0];
}

template <class T>
static __global__ void kernel_gain_grad_scalar(
    const T* __restrict__ base,
    const T* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int64_t bs0, int64_t bs1, int64_t bs2, int64_t bs3,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3)
{
    int64_t total = (int64_t)N * C * H * W;
    float sum = 0.0f;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x)
    {
        int w = (int)(idx % W);
        int64_t t = idx / W;
        int h = (int)(t % H);
        t /= H;
        int c = (int)(t % C);
        int n = (int)(t / C);
        int64_t bo = (int64_t)n * bs0 + (int64_t)c * bs1 + (int64_t)h * bs2 + (int64_t)w * bs3;
        int64_t xo = (int64_t)n * xs0 + (int64_t)c * xs1 + (int64_t)h * xs2 + (int64_t)w * xs3;
        sum += (float)base[bo] * (float)x[xo];
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, sh[0]);
}


// NHWC coalesced vector-gain gradient.  The old kernel assigns one block per
// channel and scans NHW with stride C, which is pathological for channels_last.
// This version assigns each block to a channel tile and an NHW chunk, so a warp
// reads contiguous C values for the same (n,h,w).  It writes partial sums
// [num_chunks, C], then a second small kernel reduces chunks over each channel.
template <class T, int CTILE, int CHUNK_NHW>
static __global__ void kernel_gain_grad_nhwc_partial(
    const T* __restrict__ base,
    const T* __restrict__ x,
    float* __restrict__ partial,
    int C,
    int64_t NHW)
{
    constexpr int THREADS = 256;
    constexpr int LANES_PER_C = THREADS / CTILE;
    static_assert(THREADS % CTILE == 0, "THREADS must be divisible by CTILE");
    int tid = threadIdx.x;
    int relC = tid % CTILE;
    int lane = tid / CTILE;
    int c = blockIdx.x * CTILE + relC;
    int64_t start = (int64_t)blockIdx.y * CHUNK_NHW;
    int64_t end = start + CHUNK_NHW;
    if (end > NHW) end = NHW;
    float sum = 0.0f;
    if (c < C)
    {
        for (int64_t t = start + lane; t < end; t += LANES_PER_C)
        {
            int64_t off = t * (int64_t)C + c;
            sum += (float)base[off] * (float)x[off];
        }
    }
    __shared__ float sh[THREADS];
    sh[tid] = sum;
    __syncthreads();
    for (int step = LANES_PER_C >> 1; step > 0; step >>= 1)
    {
        if (lane < step)
            sh[tid] += sh[tid + step * CTILE];
        __syncthreads();
    }
    if (lane == 0 && c < C)
        partial[(int64_t)blockIdx.y * C + c] = sh[tid];
}

template <int CHUNK_THREADS>
static __global__ void kernel_gain_grad_partial_reduce(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int C,
    int num_chunks)
{
    int c = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int k = tid; k < num_chunks; k += CHUNK_THREADS)
        sum += partial[(int64_t)k * C + c];
    __shared__ float sh[CHUNK_THREADS];
    sh[tid] = sum;
    __syncthreads();
    for (int stride = CHUNK_THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        out[c] = sh[0];
}

static inline bool is_strict_nhwc_contiguous(torch::Tensor t)
{
    return t.dim() == 4 &&
           t.stride(1) == 1 &&
           t.stride(3) == t.size(1) &&
           t.stride(2) == t.size(3) * t.size(1) &&
           t.stride(0) == t.size(2) * t.size(3) * t.size(1);
}

template <int CTILE, int CHUNK_NHW>
static torch::Tensor gain_grad_coalesced_impl(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel)
{
    TORCH_CHECK(base_dx.is_cuda() && x.is_cuda(), "base_dx and x must be CUDA tensors");
    TORCH_CHECK(base_dx.device() == x.device(), "base_dx and x must be on same device");
    TORCH_CHECK(base_dx.scalar_type() == x.scalar_type(), "base_dx and x must have same dtype");
    TORCH_CHECK(base_dx.dim() == 4 && x.dim() == 4, "base_dx and x must be rank 4");
    TORCH_CHECK(base_dx.sizes() == x.sizes(), "base_dx and x must have same shape");
    TORCH_CHECK(gain_numel == 1 || gain_numel == x.size(1), "gain_numel must be 1 or C");
    // Coalesced implementation is for vector gain and strict channels_last contiguous tensors.
    // The public wrapper falls back to the old generic implementation otherwise.
    TORCH_CHECK(gain_numel == x.size(1), "coalesced dGain supports vector gain only");
    TORCH_CHECK(is_strict_nhwc_contiguous(base_dx) && is_strict_nhwc_contiguous(x),
        "coalesced dGain requires strict contiguous channels_last/NHWC tensors");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int C = (int)x.size(1);
    int64_t NHW = (int64_t)x.size(0) * x.size(2) * x.size(3);
    int num_chunks = (int)((NHW + CHUNK_NHW - 1) / CHUNK_NHW);
    auto partial = torch::empty({num_chunks, C}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty({gain_numel}, x.options().dtype(torch::kFloat32));
    dim3 block(256, 1, 1);
    dim3 grid((C + CTILE - 1) / CTILE, num_chunks, 1);
    switch (x.scalar_type())
    {
        case at::ScalarType::BFloat16:
            kernel_gain_grad_nhwc_partial<c10::BFloat16, CTILE, CHUNK_NHW><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const c10::BFloat16*)base_dx.data_ptr(), (const c10::BFloat16*)x.data_ptr(), partial.data_ptr<float>(), C, NHW);
            break;
        case at::ScalarType::Half:
            kernel_gain_grad_nhwc_partial<c10::Half, CTILE, CHUNK_NHW><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const c10::Half*)base_dx.data_ptr(), (const c10::Half*)x.data_ptr(), partial.data_ptr<float>(), C, NHW);
            break;
        case at::ScalarType::Float:
            kernel_gain_grad_nhwc_partial<float, CTILE, CHUNK_NHW><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const float*)base_dx.data_ptr(), (const float*)x.data_ptr(), partial.data_ptr<float>(), C, NHW);
            break;
        default:
            TORCH_CHECK(false, "coalesced dGain supports only bfloat16, float16, and float32");
    }
    AT_CUDA_CHECK(cudaGetLastError());
    kernel_gain_grad_partial_reduce<256><<<dim3(C,1,1), dim3(256,1,1), 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), out.data_ptr<float>(), C, num_chunks);
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}


torch::Tensor resample_r3gan_modern_gain_grad_old_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel)
{
    TORCH_CHECK(base_dx.is_cuda() && x.is_cuda(), "base_dx and x must be CUDA tensors");
    TORCH_CHECK(base_dx.device() == x.device(), "base_dx and x must be on same device");
    TORCH_CHECK(base_dx.scalar_type() == x.scalar_type(), "base_dx and x must have same dtype");
    TORCH_CHECK(base_dx.dim() == 4 && x.dim() == 4, "base_dx and x must be rank 4");
    TORCH_CHECK(base_dx.sizes() == x.sizes(), "base_dx and x must have same shape");
    TORCH_CHECK(gain_numel == 1 || gain_numel == x.size(1), "gain_numel must be 1 or C");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::zeros({gain_numel}, x.options().dtype(torch::kFloat32));
    int N = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
    dim3 block(256, 1, 1);
    if (gain_numel == C)
    {
        dim3 grid(C, 1, 1);
        switch (x.scalar_type())
        {
            case at::ScalarType::BFloat16:
                kernel_gain_grad_channel<c10::BFloat16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    (const c10::BFloat16*)base_dx.data_ptr(), (const c10::BFloat16*)x.data_ptr(), out.data_ptr<float>(),
                    N,C,H,W, base_dx.stride(0),base_dx.stride(1),base_dx.stride(2),base_dx.stride(3), x.stride(0),x.stride(1),x.stride(2),x.stride(3));
                break;
            case at::ScalarType::Half:
                kernel_gain_grad_channel<c10::Half><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    (const c10::Half*)base_dx.data_ptr(), (const c10::Half*)x.data_ptr(), out.data_ptr<float>(),
                    N,C,H,W, base_dx.stride(0),base_dx.stride(1),base_dx.stride(2),base_dx.stride(3), x.stride(0),x.stride(1),x.stride(2),x.stride(3));
                break;
            case at::ScalarType::Float:
                kernel_gain_grad_channel<float><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    (const float*)base_dx.data_ptr(), (const float*)x.data_ptr(), out.data_ptr<float>(),
                    N,C,H,W, base_dx.stride(0),base_dx.stride(1),base_dx.stride(2),base_dx.stride(3), x.stride(0),x.stride(1),x.stride(2),x.stride(3));
                break;
            default:
                TORCH_CHECK(false, "gain_grad supports only bfloat16, float16, and float32");
        }
    }
    else
    {
        int64_t total = (int64_t)N * C * H * W;
        int blocks = (int)std::min<int64_t>(4096, (total + 255) / 256);
        dim3 grid(blocks, 1, 1);
        switch (x.scalar_type())
        {
            case at::ScalarType::BFloat16:
                kernel_gain_grad_scalar<c10::BFloat16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    (const c10::BFloat16*)base_dx.data_ptr(), (const c10::BFloat16*)x.data_ptr(), out.data_ptr<float>(),
                    N,C,H,W, base_dx.stride(0),base_dx.stride(1),base_dx.stride(2),base_dx.stride(3), x.stride(0),x.stride(1),x.stride(2),x.stride(3));
                break;
            case at::ScalarType::Half:
                kernel_gain_grad_scalar<c10::Half><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    (const c10::Half*)base_dx.data_ptr(), (const c10::Half*)x.data_ptr(), out.data_ptr<float>(),
                    N,C,H,W, base_dx.stride(0),base_dx.stride(1),base_dx.stride(2),base_dx.stride(3), x.stride(0),x.stride(1),x.stride(2),x.stride(3));
                break;
            case at::ScalarType::Float:
                kernel_gain_grad_scalar<float><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    (const float*)base_dx.data_ptr(), (const float*)x.data_ptr(), out.data_ptr<float>(),
                    N,C,H,W, base_dx.stride(0),base_dx.stride(1),base_dx.stride(2),base_dx.stride(3), x.stride(0),x.stride(1),x.stride(2),x.stride(3));
                break;
            default:
                TORCH_CHECK(false, "gain_grad supports only bfloat16, float16, and float32");
        }
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}



torch::Tensor resample_r3gan_modern_gain_grad_coalesced16_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel)
{
    // CTILE=16 gives more per-channel lanes; good fallback if CTILE=32 is too serial.
    return gain_grad_coalesced_impl<16, 512>(base_dx, x, gain_numel);
}

torch::Tensor resample_r3gan_modern_gain_grad_coalesced32_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel)
{
    // CTILE=32 makes each warp load one full contiguous channel vector for a fixed NHW position.
    return gain_grad_coalesced_impl<32, 512>(base_dx, x, gain_numel);
}

torch::Tensor resample_r3gan_modern_gain_grad_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel)
{
    // Production/default guess for B200: coalesced vector-gain NHWC reduction.
    // Fall back to the old generic strided kernel for scalar gain or non-strict layouts.
    if (gain_numel == x.size(1) && is_strict_nhwc_contiguous(base_dx) && is_strict_nhwc_contiguous(x))
        return resample_r3gan_modern_gain_grad_coalesced32_cuda(base_dx, x, gain_numel);
    return resample_r3gan_modern_gain_grad_old_cuda(base_dx, x, gain_numel);
}


// FP32 gain application with a single output quantization.  This is used by
// backward-both paths to avoid the PyTorch pattern base_dx * gain.to(base_dx.dtype),
// which would round gain before multiplication.
template <class T>
static __global__ void kernel_apply_gain_nhwc(
    const T* __restrict__ x,
    const float* __restrict__ gain,
    T* __restrict__ y,
    int N, int C, int H, int W,
    int64_t gain_stride)
{
    int64_t total = (int64_t)N * C * H * W;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x)
    {
        int c = (int)(idx % C); // strict NHWC contiguous storage order [N,H,W,C]
        float g = gain[(gain_stride == 0) ? 0 : (int64_t)c * gain_stride];
        float v = (float)x[idx] * g;
        y[idx] = (T)v;
    }
}

torch::Tensor resample_r3gan_modern_apply_gain_cuda(torch::Tensor x, torch::Tensor channel_gain)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(channel_gain.is_cuda(), "channel_gain must be CUDA");
    TORCH_CHECK(channel_gain.device() == x.device(), "channel_gain must be on same device as x");
    TORCH_CHECK(channel_gain.dtype() == torch::kFloat32, "channel_gain must be float32");
    TORCH_CHECK(x.dim() == 4 && is_strict_nhwc_contiguous(x), "apply_gain requires strict NHWC contiguous tensor");
    TORCH_CHECK(channel_gain.numel() == 1 || channel_gain.numel() == x.size(1), "channel_gain must have 1 or C elements");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto y = torch::empty_like(x, x.suggest_memory_format());
    int N=(int)x.size(0), C=(int)x.size(1), H=(int)x.size(2), W=(int)x.size(3);
    int64_t total = (int64_t)N*C*H*W;
    int blocks = (int)std::min<int64_t>(65535, (total + 255) / 256);
    dim3 block(256,1,1), grid(blocks,1,1);
    int64_t gstride = (channel_gain.numel() == 1) ? 0 : channel_gain.stride(0);
    switch (x.scalar_type())
    {
        case at::ScalarType::BFloat16:
            kernel_apply_gain_nhwc<c10::BFloat16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const c10::BFloat16*)x.data_ptr(), channel_gain.data_ptr<float>(), (c10::BFloat16*)y.data_ptr(), N,C,H,W,gstride);
            break;
        case at::ScalarType::Half:
            kernel_apply_gain_nhwc<c10::Half><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const c10::Half*)x.data_ptr(), channel_gain.data_ptr<float>(), (c10::Half*)y.data_ptr(), N,C,H,W,gstride);
            break;
        case at::ScalarType::Float:
            kernel_apply_gain_nhwc<float><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const float*)x.data_ptr(), channel_gain.data_ptr<float>(), (float*)y.data_ptr(), N,C,H,W,gstride);
            break;
        default:
            TORCH_CHECK(false, "apply_gain supports only bfloat16, float16, and float32");
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

// Explicit template instantiations for the types that the C++ launcher calls.
// The template definition lives in this .cu translation unit, while the launcher
// is compiled from .cpp, so relying on implicit cross-TU instantiation gives an
// undefined symbol at import time.
template resample_r3gan_modern_debug_kernel_spec choose_resample_r3gan_modern_debug_kernel<c10::BFloat16>(const resample_r3gan_modern_debug_kernel_params& p);
template resample_r3gan_modern_debug_kernel_spec choose_resample_r3gan_modern_debug_kernel<c10::Half>(const resample_r3gan_modern_debug_kernel_params& p);
template resample_r3gan_modern_debug_kernel_spec choose_resample_r3gan_modern_debug_kernel<float>(const resample_r3gan_modern_debug_kernel_params& p);

