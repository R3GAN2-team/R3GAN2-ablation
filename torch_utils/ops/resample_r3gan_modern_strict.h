// Modern R3GAN resampling plugin.
// Restricted to NHWC/channels-last 2x upsampling with small filters <= 8x8.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

struct int64_4_modern_debug
{
    int64_t x, y, z, w;
};

static __host__ __device__ __forceinline__ int64_4_modern_debug make_int64_4_modern_debug(int64_t x, int64_t y, int64_t z, int64_t w)
{
    int64_4_modern_debug v; v.x=x; v.y=y; v.z=z; v.w=w; return v;
}

struct resample_r3gan_modern_debug_kernel_params
{
    const void*     x;
    const float*    f;
    const float*    channelGain; // float32 scalar or [C]
    const void*     other;       // optional tensor with output shape for fused dGain
    void*           y;
    float*          dgainPartial; // optional [launchMajor, launchMinor, numSpatialBlocks, loopMinor]

    int2            up;
    int2            down;
    int2            pad0;
    int             flip;
    float           gain;

    int4            inSize;       // [W,H,C,N]
    int64_4_modern_debug inStride;
    int4            inStride32;
    int2            filterSize;   // [W,H]
    int2            filterStride;
    int4            outSize;      // [W,H,C,N]
    int64_4_modern_debug outStride;
    int4            outStride32;

    int             sizeMinor;
    int64_t         sizeMajor;
    int             loopMinor;
    int64_t         loopMajor;
    int             loopX;
    int             launchMinor;
    int64_t         launchMajor;

    int64_t         channelGainStride; // 0 for scalar
    int             mode;
    int             partialNumSpatialBlocks;
};

struct resample_r3gan_modern_debug_kernel_spec
{
    void* kernel;
    int tileOutW;
    int tileOutH;
    int loopMinor;
    int loopX;
    int blockX;
};

template <class T>
resample_r3gan_modern_debug_kernel_spec choose_resample_r3gan_modern_debug_kernel(const resample_r3gan_modern_debug_kernel_params& p);
