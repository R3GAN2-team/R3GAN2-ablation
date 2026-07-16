// gconv32_binding.cpp -- torch-facing TU. Deliberately contains NO kernel
// code: torch/extension.h sharing a TU with the kernel templates made ptxas
// spill 4 of 8 instantiations at 254 regs (~1.8x, measured). All device work
// lives in gconv32_kernel.cu behind these extern "C" entry points.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>

extern "C" void gconv32_run(void const* x, void const* wpk, void* y,
                            int N, int H, int W, int C, int gslab,
                            float alpha, uint32_t* bits, void const* slope_src,
                            cudaStream_t stream);
extern "C" void gconv32_pack_order_fill(long long* p);

static void checks(torch::Tensor const& x, torch::Tensor const& wpk, int64_t gslab)
{
  TORCH_CHECK(x.is_cuda() && wpk.is_cuda(), "gconv32: CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && wpk.scalar_type() == torch::kBFloat16,
              "gconv32: bf16 required");
  TORCH_CHECK(x.dim() == 4 && x.is_contiguous(at::MemoryFormat::ChannelsLast),
              "gconv32: NCHW logical + channels_last (NHWC physical) required");
  TORCH_CHECK(wpk.dim() == 2 && wpk.size(1) == 288 && wpk.is_contiguous(),
              "gconv32: packed weight [C,288] required");
  TORCH_CHECK(gslab == 2 || gslab == 4, "gconv32: gslab in {2,4}");
  int C = x.size(1), H = x.size(2), W = x.size(3);
  TORCH_CHECK(C % 128 == 0 && wpk.size(0) == C, "gconv32: C%128==0 required");
  if (H == 8) { TORCH_CHECK(W == 8 && x.size(0) % 2 == 0, "gconv32: 8x8 path needs W==8, N even"); }
  else        { TORCH_CHECK(H % 16 == 0 && W % 8 == 0, "gconv32: H%16==0, W%8==0"); }
}

torch::Tensor gconv32_fprop(torch::Tensor x, torch::Tensor wpk, int64_t gslab)
{
  const c10::cuda::CUDAGuard _dev_guard(x.device());  // footgun 31

  checks(x, wpk, gslab);
  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  auto y = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::ChannelsLast));
  gconv32_run(x.data_ptr(), wpk.data_ptr(), y.data_ptr(),
              N, H, W, C, (int)gslab, 0.2f, nullptr, nullptr,
              at::cuda::getCurrentCUDAStream());
  return y;
}

std::vector<torch::Tensor> gconv32_fprop_act(torch::Tensor x, torch::Tensor wpk,
                                             int64_t gslab, double alpha)
{
  const c10::cuda::CUDAGuard _dev_guard(x.device());  // footgun 31

  checks(x, wpk, gslab);
  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  auto y = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::ChannelsLast));
  // sign bitmap of pre-activation y2: NHWC-linear, word j covers out elements
  // [32j, 32j+32) -> [N, H, W, C/32] i32; pair-interleaved bit order (see gconv32.py)
  auto bits = torch::empty({N, H, W, C / 32}, x.options().dtype(torch::kInt32));
  gconv32_run(x.data_ptr(), wpk.data_ptr(), y.data_ptr(),
              N, H, W, C, (int)gslab, (float)alpha,
              reinterpret_cast<uint32_t*>(bits.data_ptr()), nullptr,
              at::cuda::getCurrentCUDAStream());
  return {y, bits};
}

torch::Tensor gconv32_fprop_slope(torch::Tensor x, torch::Tensor wpk,
                                  int64_t gslab, double alpha, torch::Tensor slope_src)
{
  const c10::cuda::CUDAGuard _dev_guard(x.device());  // footgun 31

  // y = conv(x, wpk) * slope(slope_src): slope = 1 if src sign bit clear else
  // alpha. For the FFN dgrad: x = dz2 (pre-act grad wrt y2), wpk = dgrad pack,
  // slope_src = a1 (sign(a1) == sign(y1), alpha > 0). Output = dy1.
  checks(x, wpk, gslab);
  TORCH_CHECK(slope_src.is_cuda() && slope_src.scalar_type() == torch::kBFloat16
              && slope_src.sizes() == x.sizes()
              && slope_src.is_contiguous(at::MemoryFormat::ChannelsLast),
              "gconv32: slope_src must be bf16 channels_last, same shape as x");
  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  auto y = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::ChannelsLast));
  gconv32_run(x.data_ptr(), wpk.data_ptr(), y.data_ptr(),
              N, H, W, C, (int)gslab, (float)alpha, nullptr,
              slope_src.data_ptr(), at::cuda::getCurrentCUDAStream());
  return y;
}

torch::Tensor gconv32_pack_order()
{
  auto out = torch::empty({32 * 288}, torch::dtype(torch::kInt64));
  gconv32_pack_order_fill(reinterpret_cast<long long*>(out.data_ptr<int64_t>()));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fprop", &gconv32_fprop, "grouped 3x3 conv fprop (SM100)",
        py::arg("x"), py::arg("wpk"), py::arg("gslab") = 4);
  m.def("fprop_act", &gconv32_fprop_act, "fused fprop + lrelu + sign bitmap (SM100)",
        py::arg("x"), py::arg("wpk"), py::arg("gslab") = 4, py::arg("alpha") = 0.2);
  m.def("fprop_slope", &gconv32_fprop_slope, "conv * slope(src) fused dgrad epilogue",
        py::arg("x"), py::arg("wpk"), py::arg("gslab") = 4, py::arg("alpha") = 0.2,
        py::arg("slope_src") = torch::Tensor());
  m.def("pack_order", &gconv32_pack_order, "SMEM-order gather indices");
}
