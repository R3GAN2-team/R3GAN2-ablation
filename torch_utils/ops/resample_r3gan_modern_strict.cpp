// Strict modern R3GAN resampling plugin: optimized kernels with strict FP32 gain boundaries.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <algorithm>
#include <vector>
#include "resample_r3gan_modern_strict.h"

// Implemented in the CUDA translation unit.
torch::Tensor resample_r3gan_modern_gain_grad_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel);
torch::Tensor resample_r3gan_modern_gain_grad_old_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel);
torch::Tensor resample_r3gan_modern_gain_grad_coalesced16_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel);
torch::Tensor resample_r3gan_modern_gain_grad_coalesced32_cuda(torch::Tensor base_dx, torch::Tensor x, int64_t gain_numel);
torch::Tensor resample_r3gan_modern_apply_gain_cuda(torch::Tensor x, torch::Tensor channel_gain);
std::vector<torch::Tensor> resample_r3gan_modern_outgain_both_cuda(torch::Tensor x, torch::Tensor other, torch::Tensor f, torch::Tensor channel_gain, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain, int mode);

static inline void check_dim_int32(int64_t value, const char* name)
{
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " must fit int32");
}

static inline bool storage_footprint_fits_int32(const torch::Tensor& t)
{
    int64_t max_offset = 0;
    for (int d = 0; d < t.dim(); d++)
        max_offset += (t.size(d) - 1) * t.stride(d);
    return max_offset <= std::numeric_limits<int>::max();
}

static inline bool mode_uses_int32(int mode)
{
    // Int64 modes are exact twins of the production int32 modes with only addressing changed.
    switch (mode)
    {
        case 102: case 103: case 105: case 108: case 109: case 111:
        case 113: case 115: case 117: case 119: case 121: case 123:
            return false;
        default:
            return true;
    }
}

static inline int mode_int64_equivalent(int mode)
{
    switch (mode)
    {
        case 100: return 102;
        case 101: return 103;
        case 104: return 105;
        case 106: return 108;
        case 107: return 109;
        case 110: return 111;
        case 112: return 113;
        case 114: return 115;
        case 116: return 117;
        case 118: return 119;
        case 120: return 121;
        case 122: return 123;
        default: return mode;
    }
}

static inline bool mode_requires_gain(int mode)
{
    return !(mode == 112 || mode == 113 || mode == 114 || mode == 115);
}

static inline int modern_vec4s_required_loop_minor(int mode)
{
    switch (mode)
    {
        case 100: case 102: case 106: case 108:
        case 101: case 103: case 107: case 109:
            return 32;
        default:
            return 0;
    }
}

static inline bool mode_is_modern_vec4s(int mode)
{
    return modern_vec4s_required_loop_minor(mode) != 0;
}

static inline bool mode_is_up2_mode(int mode)
{
    switch (mode)
    {
        case 100: case 101: case 102: case 103:
        case 106: case 107: case 108: case 109:
        case 112: case 113: case 116: case 117:
        case 120: case 121:
            return true;
        default:
            return false;
    }
}

static inline bool mode_is_down2_mode(int mode)
{
    switch (mode)
    {
        case 104: case 105: case 110: case 111:
        case 114: case 115: case 118: case 119:
        case 122: case 123:
            return true;
        default:
            return false;
    }
}

static torch::Tensor resample_r3gan_modern_debug(
    torch::Tensor x,
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
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(f.device() == x.device(), "f must reside on same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.numel() > 0, "x has zero size");
    TORCH_CHECK(f.numel() > 0, "f has zero size");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(f.dim() == 2, "f must be rank 2");
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");
    TORCH_CHECK(f.size(0) <= 8 && f.size(1) <= 8, "modern resample plugin only supports filters <= 8x8");
    const bool is_up2 = (upx == 2 && upy == 2 && downx == 1 && downy == 1);
    const bool is_down2 = (upx == 1 && upy == 1 && downx == 2 && downy == 2);
    TORCH_CHECK(is_up2 || is_down2,
        "modern resample plugin is restricted to 2x upsampling or 2x downsampling");
    TORCH_CHECK((mode_is_up2_mode(mode) && is_up2) || (mode_is_down2_mode(mode) && is_down2),
        "selected mode is incompatible with requested up/down factors");
    TORCH_CHECK(x.stride(1) == 1, "modern resample plugin requires channels-last/NHWC input");
    TORCH_CHECK(mode >= 100 && mode <= 123, "unknown modern resample mode");
    int lm = modern_vec4s_required_loop_minor(mode);
    if (lm != 0)
    {
        TORCH_CHECK(x.size(1) % lm == 0, "experimental vec4s no-tail mode requires C % loopMinor == 0");
        TORCH_CHECK(x.stride(1) == 1, "experimental vec4s mode requires contiguous channel stride");
        TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Float,
            "experimental vec4s mode supports only bf16/fp16/fp32");
    }

    check_dim_int32(x.size(0), "x batch dimension");
    check_dim_int32(x.size(1), "x channel dimension");
    check_dim_int32(x.size(2), "x height dimension");
    check_dim_int32(x.size(3), "x width dimension");
    check_dim_int32(f.size(0), "filter height dimension");
    check_dim_int32(f.size(1), "filter width dimension");

    const bool need_gain = mode_requires_gain(mode);
    if (need_gain)
    {
        TORCH_CHECK(channel_gain.defined() && channel_gain.numel() > 0, "selected mode requires channel_gain");
        TORCH_CHECK(channel_gain.is_cuda(), "channel_gain must reside on CUDA device");
        TORCH_CHECK(channel_gain.device() == x.device(), "channel_gain must be on same device as x");
        TORCH_CHECK(channel_gain.dtype() == torch::kFloat, "channel_gain must be float32");
        TORCH_CHECK(channel_gain.numel() == 1 || channel_gain.numel() == x.size(1), "channel_gain must have 1 or C elements");
        TORCH_CHECK(channel_gain.is_contiguous(), "channel_gain must be contiguous");
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int64_t outW64 = (x.size(3) * (int64_t)upx + padx0 + padx1 - f.size(1) + downx) / downx;
    int64_t outH64 = (x.size(2) * (int64_t)upy + pady0 + pady1 - f.size(0) + downy) / downy;
    TORCH_CHECK(outW64 >= 1 && outH64 >= 1, "output must be at least 1x1");
    check_dim_int32(outW64, "output width dimension");
    check_dim_int32(outH64, "output height dimension");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH64, outW64}, x.options(), x.suggest_memory_format());
    if (mode_is_modern_vec4s(mode))
        TORCH_CHECK(y.stride(1) == 1, "experimental vec4s mode requires channels-last output");

    if (mode_uses_int32(mode) && (!storage_footprint_fits_int32(x) || !storage_footprint_fits_int32(y)))
        mode = mode_int64_equivalent(mode);
    if (mode_uses_int32(mode))
    {
        TORCH_CHECK(storage_footprint_fits_int32(x), "int32 mode requested but x storage footprint exceeds int32");
        TORCH_CHECK(storage_footprint_fits_int32(y), "int32 mode requested but y storage footprint exceeds int32");
    }

    resample_r3gan_modern_debug_kernel_params p;
    p.x                 = x.data_ptr();
    p.f                 = f.data_ptr<float>();
    p.channelGain       = (need_gain) ? channel_gain.data_ptr<float>() : nullptr;
    p.other             = nullptr;
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
    p.channelGainStride = (need_gain && channel_gain.numel() != 1) ? channel_gain.stride(0) : 0;
    p.mode              = mode;
    p.partialNumSpatialBlocks = 0;

    resample_r3gan_modern_debug_kernel_spec spec;
    switch (x.scalar_type())
    {
        case at::ScalarType::BFloat16:
            spec = choose_resample_r3gan_modern_debug_kernel<c10::BFloat16>(p);
            break;
        case at::ScalarType::Half:
            spec = choose_resample_r3gan_modern_debug_kernel<c10::Half>(p);
            break;
        case at::ScalarType::Float:
            spec = choose_resample_r3gan_modern_debug_kernel<float>(p);
            break;
        default:
            TORCH_CHECK(false, "resample_r3gan_modern_debug supports only bfloat16, float16, and float32");
    }

    TORCH_CHECK(spec.kernel != nullptr, "unsupported modern resample mode/filter/layout combination");
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor     = spec.loopMinor;
    p.loopX         = spec.loopX;
    p.launchMinor   = (p.sizeMinor - 1) / p.loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;
    TORCH_CHECK(p.launchMajor <= std::numeric_limits<unsigned int>::max(), "CUDA grid z dimension too large");

    dim3 blockSize(spec.blockX, 1, 1);
    dim3 gridSize;
    gridSize = dim3(
        ((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
        (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1,
        (unsigned int)p.launchMajor);

    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("resample_r3gan_modern_debug", &resample_r3gan_modern_debug);
    m.def("resample_r3gan_modern_gain_grad", &resample_r3gan_modern_gain_grad_cuda);
    m.def("resample_r3gan_modern_gain_grad_old", &resample_r3gan_modern_gain_grad_old_cuda);
    m.def("resample_r3gan_modern_gain_grad_coalesced16", &resample_r3gan_modern_gain_grad_coalesced16_cuda);
    m.def("resample_r3gan_modern_gain_grad_coalesced32", &resample_r3gan_modern_gain_grad_coalesced32_cuda);
    m.def("resample_r3gan_modern_apply_gain", &resample_r3gan_modern_apply_gain_cuda);
    m.def("resample_r3gan_modern_outgain_both", &resample_r3gan_modern_outgain_both_cuda);
}
