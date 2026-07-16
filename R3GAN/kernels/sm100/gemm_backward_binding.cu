// R3GAN/kernels/sm100/bind_ffn_backward.cu
// Pybind layer for sm100_ffn_backward_instances.cu.
// Requires sm100_gemm_ffn_backward_evt.cuh to be present in this directory.
#include <torch/extension.h>
#include "gemm_backward_kernel.cuh"

// Prototypes for the instance symbols defined in sm100_ffn_backward_instances.cu.
DX_SIG(dx_256x128_2x1);
DX_SIG(dx_256x256_2x1);
DX_SIG(dx_256x256_2x2);
DW_SIG(dw_256x256_2sm);
DW_SIG(dw_256x128_2sm);
DW_SIG(dw_256x64_2sm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dx_256x128_2x1", &dx_256x128_2x1, "SM100 FFN dX 256x128 cluster 2x1");
  m.def("dx_256x256_2x1", &dx_256x256_2x1, "SM100 FFN dX 256x256 cluster 2x1");
  m.def("dx_256x256_2x2", &dx_256x256_2x2, "SM100 FFN dX 256x256 cluster 2x2");
  m.def("dw_256x256_2sm", &dw_256x256_2sm, "SM100 FFN dW 256x256 2SM");
  m.def("dw_256x128_2sm", &dw_256x128_2sm, "SM100 FFN dW 256x128 2SM");
  m.def("dw_256x64_2sm",  &dw_256x64_2sm,  "SM100 FFN dW 256x64 2SM");
}
