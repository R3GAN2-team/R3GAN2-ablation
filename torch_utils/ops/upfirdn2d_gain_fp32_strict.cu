// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include "upfirdn2d_gain_fp32_strict.h"

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };
template <> struct InternalType<c10::BFloat16>  { typedef float  scalar_t; };

static __device__ __forceinline__ int floor_div(int a, int b)
{
    int t = 1 - a / b;
    return (a + t * b) / b - t;
}


// Load an input pixel, optionally applying per-channel gain in float32
// before the existing StyleGAN accumulation.  Unlike the gain_exact ablation,
// the product is NOT rounded back to T before filtering:
//
//     contribution = float32(x) * float32(channel_gain[c]) * float32(filter)
//
// This removes the activation-sized x*gain materialization while making the
// gain quantization boundary consistent with the fused FP32-gain semantics.
template <class T, class scalar_t>
static __device__ __forceinline__ scalar_t load_input_with_fp32_gain(const upfirdn2d_kernel_params& p, int offset, int c)
{
    scalar_t xv = (scalar_t)((const T*)p.x)[offset];
    if (p.hasChannelGain && p.channelGainMode == 1)
    {
        int gc = (p.channelGainSize == 1) ? 0 : c;
        xv *= (scalar_t)p.channelGain[gc];
    }
    return xv;
}

template <class scalar_t>
static __device__ __forceinline__ scalar_t apply_output_fp32_gain(const upfirdn2d_kernel_params& p, scalar_t v, int c)
{
    if (p.hasChannelGain && p.channelGainMode == 2)
    {
        int gc = (p.channelGainSize == 1) ? 0 : c;
        v *= (scalar_t)p.channelGain[gc];
    }
    return v;
}

//------------------------------------------------------------------------
// Generic CUDA implementation for large filters.

template <class T> static __global__ void upfirdn2d_kernel_large(upfirdn2d_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;

    // Calculate thread index.
    int minorBase = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = minorBase / p.launchMinor;
    minorBase -= outY * p.launchMinor;
    int outXBase = blockIdx.y * p.loopX * blockDim.y + threadIdx.y;
    int majorBase = blockIdx.z * p.loopMajor;
    if (outXBase >= p.outSize.x | outY >= p.outSize.y | majorBase >= p.sizeMajor)
        return;

    // Setup Y receptive field.
    int midY = outY * p.down.y + p.up.y - 1 - p.pad0.y;
    int inY = min(max(floor_div(midY, p.up.y), 0), p.inSize.y);
    int h = min(max(floor_div(midY + p.filterSize.y, p.up.y), 0), p.inSize.y) - inY;
    int filterY = midY + p.filterSize.y - (inY + 1) * p.up.y;
    if (p.flip)
        filterY = p.filterSize.y - 1 - filterY;

    // Loop over major, minor, and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    for (int minorIdx = 0, minor = minorBase; minorIdx < p.loopMinor & minor < p.sizeMinor; minorIdx++, minor += p.launchMinor)
    {
        int nc = major * p.sizeMinor + minor;
        int n = nc / p.inSize.z;
        int c = nc - n * p.inSize.z;
        for (int loopX = 0, outX = outXBase; loopX < p.loopX & outX < p.outSize.x; loopX++, outX += blockDim.y)
        {
            // Setup X receptive field.
            int midX = outX * p.down.x + p.up.x - 1 - p.pad0.x;
            int inX = min(max(floor_div(midX, p.up.x), 0), p.inSize.x);
            int w = min(max(floor_div(midX + p.filterSize.x, p.up.x), 0), p.inSize.x) - inX;
            int filterX = midX + p.filterSize.x - (inX + 1) * p.up.x;
            if (p.flip)
                filterX = p.filterSize.x - 1 - filterX;

            // Initialize pointers.
            int inputOffset = inX * p.inStride.x + inY * p.inStride.y + c * p.inStride.z + n * p.inStride.w;
            const float* fp = &p.f[filterX * p.filterStride.x + filterY * p.filterStride.y];
            int filterStepX = ((p.flip) ? p.up.x : -p.up.x) * p.filterStride.x;
            int filterStepY = ((p.flip) ? p.up.y : -p.up.y) * p.filterStride.y;

            // Inner loop.
            scalar_t v = 0;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    v += load_input_with_fp32_gain<T, scalar_t>(p, inputOffset, c) * (scalar_t)(*fp);
                    inputOffset += p.inStride.x;
                    fp += filterStepX;
                }
                inputOffset += p.inStride.y - w * p.inStride.x;
                fp += filterStepY - w * filterStepX;
            }

            // Store result.
            v *= p.gain;
            v = apply_output_fp32_gain<scalar_t>(p, v, c);
            ((T*)p.y)[outX * p.outStride.x + outY * p.outStride.y + c * p.outStride.z + n * p.outStride.w] = (T)v;
        }
    }
}

//------------------------------------------------------------------------
// Specialized CUDA implementation for small filters.

template <class T, int upx, int upy, int downx, int downy, int filterW, int filterH, int tileOutW, int tileOutH, int loopMinor>
static __global__ void upfirdn2d_kernel_small(upfirdn2d_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;
    __shared__ volatile scalar_t sf[filterH][filterW];
    __shared__ volatile scalar_t sx[tileInH][tileInW][loopMinor];

    // Calculate tile index.
    int minorBase = blockIdx.x;
    int tileOutY = minorBase / p.launchMinor;
    minorBase -= tileOutY * p.launchMinor;
    minorBase *= loopMinor;
    tileOutY *= tileOutH;
    int tileOutXBase = blockIdx.y * p.loopX * tileOutW;
    int majorBase = blockIdx.z * p.loopMajor;
    if (tileOutXBase >= p.outSize.x | tileOutY >= p.outSize.y | majorBase >= p.sizeMajor)
        return;

    // Load filter (flipped).
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

    // Loop over major and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    {
        int baseNC = major * p.sizeMinor + minorBase;
        int n = baseNC / p.inSize.z;
        int baseC = baseNC - n * p.inSize.z;
        for (int loopX = 0, tileOutX = tileOutXBase; loopX < p.loopX & tileOutX < p.outSize.x; loopX++, tileOutX += tileOutW)
        {
            // Load input pixels.
            int tileMidX = tileOutX * downx + upx - 1 - p.pad0.x;
            int tileMidY = tileOutY * downy + upy - 1 - p.pad0.y;
            int tileInX = floor_div(tileMidX, upx);
            int tileInY = floor_div(tileMidY, upy);
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
                    v = load_input_with_fp32_gain<T, scalar_t>(p, inX * p.inStride.x + inY * p.inStride.y + c * p.inStride.z + n * p.inStride.w, c);
                sx[relInY][relInX][relC] = v;
            }

            // Loop over output pixels.
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

                // Setup receptive field.
                int midX = tileMidX + relOutX * downx;
                int midY = tileMidY + relOutY * downy;
                int inX = floor_div(midX, upx);
                int inY = floor_div(midY, upy);
                int relInX = inX - tileInX;
                int relInY = inY - tileInY;
                int filterX = (inX + 1) * upx - midX - 1; // flipped
                int filterY = (inY + 1) * upy - midY - 1; // flipped

                // Inner loop.
                if (outX < p.outSize.x & outY < p.outSize.y & c < p.outSize.z)
                {
                    scalar_t v = 0;
                    #pragma unroll
                    for (int y = 0; y < filterH / upy; y++)
                        #pragma unroll
                        for (int x = 0; x < filterW / upx; x++)
                            v += sx[relInY + y][relInX + x][relC] * sf[filterY + y * upy][filterX + x * upx];
                    v *= p.gain;
                    v = apply_output_fp32_gain<scalar_t>(p, v, c);
                    ((T*)p.y)[outX * p.outStride.x + outY * p.outStride.y + c * p.outStride.z + n * p.outStride.w] = (T)v;
                }
            }
        }
    }
}

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> upfirdn2d_kernel_spec choose_upfirdn2d_kernel(const upfirdn2d_kernel_params& p)
{
    int s = p.inStride.z, fx = p.filterSize.x, fy = p.filterSize.y;
    upfirdn2d_kernel_spec spec = {(void*)upfirdn2d_kernel_large<T>, -1,-1,1, 4}; // contiguous
    if (s == 1)           spec = {(void*)upfirdn2d_kernel_large<T>, -1,-1,4, 1}; // channels_last

    // No up/downsampling.
    if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 24 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 24,24, 64,32,1>, 64,32,1, 1};
        if (s != 1 && fx <= 16 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 16,16, 64,32,1>, 64,32,1, 1};
        if (s != 1 && fx <= 7  && fy <= 7 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 7,7,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 6  && fy <= 6 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 6,6,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 5  && fy <= 5 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 5,5,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 4  && fy <= 4 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 4,4,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 3  && fy <= 3 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 3,3,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 24 && fy <= 1 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 24,1,  128,8,1>, 128,8,1, 1};
        if (s != 1 && fx <= 16 && fy <= 1 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 16,1,  128,8,1>, 128,8,1, 1};
        if (s != 1 && fx <= 8  && fy <= 1 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 8,1,   128,8,1>, 128,8,1, 1};
        if (s != 1 && fx <= 1  && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 1,24,  32,32,1>, 32,32,1, 1};
        if (s != 1 && fx <= 1  && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 1,16,  32,32,1>, 32,32,1, 1};
        if (s != 1 && fx <= 1  && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 1,8,   32,32,1>, 32,32,1, 1};
        // channels_last
        if (s == 1 && fx <= 24 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 24,24, 32,32,1>,  32,32,1,  1};
        if (s == 1 && fx <= 16 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 16,16, 32,32,1>,  32,32,1,  1};
        if (s == 1 && fx <= 7  && fy <= 7 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 7,7,   16,16,8>,  16,16,8,  1};
        if (s == 1 && fx <= 6  && fy <= 6 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 6,6,   16,16,8>,  16,16,8,  1};
        if (s == 1 && fx <= 5  && fy <= 5 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 5,5,   16,16,8>,  16,16,8,  1};
        if (s == 1 && fx <= 4  && fy <= 4 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 4,4,   16,16,8>,  16,16,8,  1};
        if (s == 1 && fx <= 3  && fy <= 3 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 3,3,   16,16,8>,  16,16,8,  1};
        if (s == 1 && fx <= 24 && fy <= 1 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 24,1,  128,1,16>, 128,1,16, 1};
        if (s == 1 && fx <= 16 && fy <= 1 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 16,1,  128,1,16>, 128,1,16, 1};
        if (s == 1 && fx <= 8  && fy <= 1 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 8,1,   128,1,16>, 128,1,16, 1};
        if (s == 1 && fx <= 1  && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 1,24,  1,128,16>, 1,128,16, 1};
        if (s == 1 && fx <= 1  && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 1,16,  1,128,16>, 1,128,16, 1};
        if (s == 1 && fx <= 1  && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,1, 1,8,   1,128,16>, 1,128,16, 1};
    }

    // 2x upsampling.
    if (p.up.x == 2 && p.up.y == 2 && p.down.x == 1 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 24 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 24,24, 64,32,1>, 64,32,1, 1};
        if (s != 1 && fx <= 16 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 16,16, 64,32,1>, 64,32,1, 1};
        if (s != 1 && fx <= 8  && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 8,8,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 6  && fy <= 6 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 6,6,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 4  && fy <= 4 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 4,4,   64,16,1>, 64,16,1, 1};
        if (s != 1 && fx <= 2  && fy <= 2 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 2,2,   64,16,1>, 64,16,1, 1};
        // channels_last
        if (s == 1 && fx <= 24 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 24,24, 32,32,1>, 32,32,1, 1};
        if (s == 1 && fx <= 16 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 16,16, 32,32,1>, 32,32,1, 1};
        if (s == 1 && fx <= 8  && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 8,8,   16,16,8>, 16,16,8, 1};
        if (s == 1 && fx <= 6  && fy <= 6 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 6,6,   16,16,8>, 16,16,8, 1};
        if (s == 1 && fx <= 4  && fy <= 4 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 4,4,   16,16,8>, 16,16,8, 1};
        if (s == 1 && fx <= 2  && fy <= 2 ) spec = {(void*)upfirdn2d_kernel_small<T, 2,2, 1,1, 2,2,   16,16,8>, 16,16,8, 1};
    }
    if (p.up.x == 2 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 24 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 2,1, 1,1, 24,1, 128,8,1>, 128,8,1, 1};
        if (s != 1 && fx <= 16 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 2,1, 1,1, 16,1, 128,8,1>, 128,8,1, 1};
        if (s != 1 && fx <= 8  && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 2,1, 1,1, 8,1,  128,8,1>, 128,8,1, 1};
        // channels_last
        if (s == 1 && fx <= 24 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 2,1, 1,1, 24,1, 128,1,16>, 128,1,16, 1};
        if (s == 1 && fx <= 16 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 2,1, 1,1, 16,1, 128,1,16>, 128,1,16, 1};
        if (s == 1 && fx <= 8  && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 2,1, 1,1, 8,1,  128,1,16>, 128,1,16, 1};
    }
    if (p.up.x == 1 && p.up.y == 2 && p.down.x == 1 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 1 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,2, 1,1, 1,24, 32,32,1>, 32,32,1, 1};
        if (s != 1 && fx <= 1 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,2, 1,1, 1,16, 32,32,1>, 32,32,1, 1};
        if (s != 1 && fx <= 1 && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,2, 1,1, 1,8,  32,32,1>, 32,32,1, 1};
        // channels_last
        if (s == 1 && fx <= 1 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,2, 1,1, 1,24, 1,128,16>, 1,128,16, 1};
        if (s == 1 && fx <= 1 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,2, 1,1, 1,16, 1,128,16>, 1,128,16, 1};
        if (s == 1 && fx <= 1 && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,2, 1,1, 1,8,  1,128,16>, 1,128,16, 1};
    }

    // 2x downsampling.
    if (p.up.x == 1 && p.up.y == 1 && p.down.x == 2 && p.down.y == 2)
    {
        // contiguous
        if (s != 1 && fx <= 24 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 24,24, 32,16,1>, 32,16,1, 1};
        if (s != 1 && fx <= 16 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 16,16, 32,16,1>, 32,16,1, 1};
        if (s != 1 && fx <= 8  && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 8,8,   32,8,1>,  32,8,1,  1};
        if (s != 1 && fx <= 6  && fy <= 6 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 6,6,   32,8,1>,  32,8,1,  1};
        if (s != 1 && fx <= 4  && fy <= 4 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 4,4,   32,8,1>,  32,8,1,  1};
        if (s != 1 && fx <= 2  && fy <= 2 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 2,2,   32,8,1>,  32,8,1,  1};
        // channels_last
        if (s == 1 && fx <= 24 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 24,24, 16,16,1>, 16,16,1, 1};
        if (s == 1 && fx <= 16 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 16,16, 16,16,1>, 16,16,1, 1};
        if (s == 1 && fx <= 8  && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 8,8,   8,8,8>,   8,8,8,   1};
        if (s == 1 && fx <= 6  && fy <= 6 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 6,6,   8,8,8>,   8,8,8,   1};
        if (s == 1 && fx <= 4  && fy <= 4 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 4,4,   8,8,8>,   8,8,8,   1};
        if (s == 1 && fx <= 2  && fy <= 2 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,2, 2,2,   8,8,8>,   8,8,8,   1};
    }
    if (p.up.x == 1 && p.up.y == 1 && p.down.x == 2 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 24 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,1, 24,1, 64,8,1>, 64,8,1, 1};
        if (s != 1 && fx <= 16 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,1, 16,1, 64,8,1>, 64,8,1, 1};
        if (s != 1 && fx <= 8  && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,1, 8,1,  64,8,1>, 64,8,1, 1};
        // channels_last
        if (s == 1 && fx <= 24 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,1, 24,1, 64,1,8>, 64,1,8, 1};
        if (s == 1 && fx <= 16 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,1, 16,1, 64,1,8>, 64,1,8, 1};
        if (s == 1 && fx <= 8  && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 2,1, 8,1,  64,1,8>, 64,1,8, 1};
    }
    if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 2)
    {
        // contiguous
        if (s != 1 && fx <= 1 && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,2, 1,24, 32,16,1>, 32,16,1, 1};
        if (s != 1 && fx <= 1 && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,2, 1,16, 32,16,1>, 32,16,1, 1};
        if (s != 1 && fx <= 1 && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,2, 1,8,  32,16,1>, 32,16,1, 1};
        // channels_last
        if (s == 1 && fx <= 1  && fy <= 24) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,2, 1,24, 1,64,8>, 1,64,8, 1};
        if (s == 1 && fx <= 1  && fy <= 16) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,2, 1,16, 1,64,8>, 1,64,8, 1};
        if (s == 1 && fx <= 1  && fy <= 8 ) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,2, 1,8,  1,64,8>, 1,64,8, 1};
    }

    // 4x upsampling.
    if (p.up.x == 4 && p.up.y == 4 && p.down.x == 1 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 48 && fy <= 48) spec = {(void*)upfirdn2d_kernel_small<T, 4,4, 1,1, 48,48, 64,32,1>, 64,32,1, 1};
        if (s != 1 && fx <= 32 && fy <= 32) spec = {(void*)upfirdn2d_kernel_small<T, 4,4, 1,1, 32,32, 64,32,1>, 64,32,1, 1};
        // channels_last
        if (s == 1 && fx <= 48 && fy <= 48) spec = {(void*)upfirdn2d_kernel_small<T, 4,4, 1,1, 48,48, 32,32,1>, 32,32,1, 1};
        if (s == 1 && fx <= 32 && fy <= 32) spec = {(void*)upfirdn2d_kernel_small<T, 4,4, 1,1, 32,32, 32,32,1>, 32,32,1, 1};
    }
    if (p.up.x == 4 && p.up.y == 1 && p.down.x == 1 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 48 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 4,1, 1,1, 48,1, 128,8,1>, 128,8,1, 1};
        if (s != 1 && fx <= 32 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 4,1, 1,1, 32,1, 128,8,1>, 128,8,1, 1};
        // channels_last
        if (s == 1 && fx <= 48 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 4,1, 1,1, 48,1, 128,1,16>, 128,1,16, 1};
        if (s == 1 && fx <= 32 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 4,1, 1,1, 32,1, 128,1,16>, 128,1,16, 1};
    }
    if (p.up.x == 1 && p.up.y == 4 && p.down.x == 1 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 1 && fy <= 48) spec = {(void*)upfirdn2d_kernel_small<T, 1,4, 1,1, 1,48, 32,32,1>, 32,32,1, 1};
        if (s != 1 && fx <= 1 && fy <= 32) spec = {(void*)upfirdn2d_kernel_small<T, 1,4, 1,1, 1,32, 32,32,1>, 32,32,1, 1};
        // channels_last
        if (s == 1 && fx <= 1 && fy <= 48) spec = {(void*)upfirdn2d_kernel_small<T, 1,4, 1,1, 1,48, 1,128,16>, 1,128,16, 1};
        if (s == 1 && fx <= 1 && fy <= 32) spec = {(void*)upfirdn2d_kernel_small<T, 1,4, 1,1, 1,32, 1,128,16>, 1,128,16, 1};
    }

    // 4x downsampling (inefficient).
    if (p.up.x == 1 && p.up.y == 1 && p.down.x == 4 && p.down.y == 1)
    {
        // contiguous
        if (s != 1 && fx <= 48 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 4,1, 48,1, 32,8,1>, 32,8,1, 1};
        if (s != 1 && fx <= 32 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 4,1, 32,1, 32,8,1>, 32,8,1, 1};
        // channels_last
        if (s == 1 && fx <= 48 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 4,1, 48,1, 32,1,8>, 32,1,8, 1};
        if (s == 1 && fx <= 32 && fy <= 1) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 4,1, 32,1, 32,1,8>, 32,1,8, 1};
    }
    if (p.up.x == 1 && p.up.y == 1 && p.down.x == 1 && p.down.y == 4)
    {
        // contiguous
        if (s != 1 && fx <= 1 && fy <= 48) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,4, 1,48, 32,8,1>, 32,8,1, 1};
        if (s != 1 && fx <= 1 && fy <= 32) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,4, 1,32, 32,8,1>, 32,8,1, 1};
        // channels_last
        if (s == 1 && fx <= 1  && fy <= 48) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,4, 1,48, 1,32,8>, 1,32,8, 1};
        if (s == 1 && fx <= 1  && fy <= 32) spec = {(void*)upfirdn2d_kernel_small<T, 1,1, 1,4, 1,32, 1,32,8>, 1,32,8, 1};
    }
    return spec;
}

//------------------------------------------------------------------------
// Template specializations.

template upfirdn2d_kernel_spec choose_upfirdn2d_kernel<double>   (const upfirdn2d_kernel_params& p);
template upfirdn2d_kernel_spec choose_upfirdn2d_kernel<float>    (const upfirdn2d_kernel_params& p);
template upfirdn2d_kernel_spec choose_upfirdn2d_kernel<c10::Half>(const upfirdn2d_kernel_params& p);
template upfirdn2d_kernel_spec choose_upfirdn2d_kernel<c10::BFloat16>(const upfirdn2d_kernel_params& p);

//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Exact FP32 channel-gain gradient without materializing R^T(dy) or
// (x * R^T(dy)).  This is intentionally a simple reduction kernel, not an
// optimized replacement for the StyleGAN resampling kernels.

template <class T>
static __device__ __forceinline__ float read_tensor4(const T* ptr, int64_t n, int64_t c, int64_t h, int64_t w,
    int64_t sN, int64_t sC, int64_t sH, int64_t sW)
{
    return (float)ptr[n * sN + c * sC + h * sH + w * sW];
}

struct channel_gain_grad_params
{
    const void* x;
    const void* dy;
    const float* f;
    float* partial;

    int64_t N;
    int64_t C;
    int64_t inH;
    int64_t inW;
    int64_t outH;
    int64_t outW;

    int64_t xStrideN;
    int64_t xStrideC;
    int64_t xStrideH;
    int64_t xStrideW;
    int64_t dyStrideN;
    int64_t dyStrideC;
    int64_t dyStrideH;
    int64_t dyStrideW;
    int64_t fStrideY;
    int64_t fStrideX;

    int filterH;
    int filterW;
    int2 up;       // reverse-op up
    int2 down;     // reverse-op down
    int2 pad0;     // reverse-op pad0
    int flip;      // reverse-op flip
    float gain;    // reverse-op gain
    int gainSize;
    int chunks;
};

template <class T, int BLOCK>
static __global__ void channel_gain_grad_kernel(channel_gain_grad_params p)
{
    int gidx = blockIdx.x;
    int chunk = blockIdx.y;
    int64_t total = (p.gainSize == 1) ? (p.N * p.C * p.inH * p.inW) : (p.N * p.inH * p.inW);
    int64_t start = (total * chunk) / p.chunks;
    int64_t end = (total * (chunk + 1)) / p.chunks;

    float sum = 0.0f;
    for (int64_t linear = start + threadIdx.x; linear < end; linear += BLOCK)
    {
        int64_t n, c, iy, ix;
        if (p.gainSize == 1)
        {
            int64_t t = linear;
            ix = t % p.inW; t /= p.inW;
            iy = t % p.inH; t /= p.inH;
            c  = t % p.C;   t /= p.C;
            n  = t;
        }
        else
        {
            c = gidx;
            int64_t t = linear;
            ix = t % p.inW; t /= p.inW;
            iy = t % p.inH; t /= p.inH;
            n  = t;
        }

        // Compute one output pixel of the reverse upfirdn2d op at [n,c,iy,ix].
        int outX = (int)ix;
        int outY = (int)iy;

        int midY = outY * p.down.y + p.up.y - 1 - p.pad0.y;
        int inY = min(max(floor_div(midY, p.up.y), 0), (int)p.outH);
        int h = min(max(floor_div(midY + p.filterH, p.up.y), 0), (int)p.outH) - inY;
        int filterY = midY + p.filterH - (inY + 1) * p.up.y;
        if (p.flip)
            filterY = p.filterH - 1 - filterY;

        int midX = outX * p.down.x + p.up.x - 1 - p.pad0.x;
        int inX = min(max(floor_div(midX, p.up.x), 0), (int)p.outW);
        int w = min(max(floor_div(midX + p.filterW, p.up.x), 0), (int)p.outW) - inX;
        int filterX = midX + p.filterW - (inX + 1) * p.up.x;
        if (p.flip)
            filterX = p.filterW - 1 - filterX;

        int filterStepX = ((p.flip) ? p.up.x : -p.up.x);
        int filterStepY = ((p.flip) ? p.up.y : -p.up.y);

        float base = 0.0f;
        int fx0 = filterX;
        int fy = filterY;
        for (int yy = 0; yy < h; yy++)
        {
            int fx = fx0;
            for (int xx = 0; xx < w; xx++)
            {
                float dyv = read_tensor4((const T*)p.dy, n, c, inY + yy, inX + xx,
                    p.dyStrideN, p.dyStrideC, p.dyStrideH, p.dyStrideW);
                float fv = p.f[fy * p.fStrideY + fx * p.fStrideX];
                base += dyv * fv;
                fx += filterStepX;
            }
            fy += filterStepY;
        }
        base *= p.gain;

        float xv = read_tensor4((const T*)p.x, n, c, iy, ix,
            p.xStrideN, p.xStrideC, p.xStrideH, p.xStrideW);
        sum += xv * base;
    }

    __shared__ float sh[BLOCK];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        p.partial[gidx * p.chunks + chunk] = sh[0];
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor channel_gain_grad(torch::Tensor dy, torch::Tensor x, torch::Tensor f,
    int upx, int upy, int downx, int downy,
    int padx0, int padx1, int pady0, int pady1,
    bool flip, float gain, int64_t gain_size)
{
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(dy.is_cuda(), "dy must reside on CUDA device");
    TORCH_CHECK(f.is_cuda(), "f must reside on CUDA device");
    TORCH_CHECK(x.device() == dy.device() && x.device() == f.device(), "x, dy, f must be on same device");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.dim() == 4 && dy.dim() == 4 && f.dim() == 2, "x/dy must be rank 4 and f rank 2");
    TORCH_CHECK(gain_size == 1 || gain_size == x.size(1), "gain_size must be 1 or num_channels");
    TORCH_CHECK(x.size(0) == dy.size(0) && x.size(1) == dy.size(1), "x/dy N,C mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int64_t outC = gain_size;
    int chunks = 32;
    torch::Tensor partial = torch::empty({outC, chunks}, x.options().dtype(torch::kFloat));

    channel_gain_grad_params p;
    p.x = x.data_ptr();
    p.dy = dy.data_ptr();
    p.f = f.data_ptr<float>();
    p.partial = partial.data_ptr<float>();
    p.N = x.size(0);
    p.C = x.size(1);
    p.inH = x.size(2);
    p.inW = x.size(3);
    p.outH = dy.size(2);
    p.outW = dy.size(3);
    p.xStrideN = x.stride(0);
    p.xStrideC = x.stride(1);
    p.xStrideH = x.stride(2);
    p.xStrideW = x.stride(3);
    p.dyStrideN = dy.stride(0);
    p.dyStrideC = dy.stride(1);
    p.dyStrideH = dy.stride(2);
    p.dyStrideW = dy.stride(3);
    p.fStrideY = f.stride(0);
    p.fStrideX = f.stride(1);
    p.filterH = f.size(0);
    p.filterW = f.size(1);
    p.up = make_int2(upx, upy);
    p.down = make_int2(downx, downy);
    p.pad0 = make_int2(padx0, pady0);
    p.flip = flip ? 1 : 0;
    p.gain = gain;
    p.gainSize = (int)gain_size;
    p.chunks = chunks;

    constexpr int BLOCK = 256;
    dim3 grid((unsigned)outC, (unsigned)chunks, 1);
    dim3 block(BLOCK, 1, 1);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "channel_gain_grad", [&]
    {
        channel_gain_grad_kernel<scalar_t, BLOCK><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(p);
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return partial.sum(1);
}
