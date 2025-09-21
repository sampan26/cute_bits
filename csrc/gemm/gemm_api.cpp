#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
                      
// Template declaration for the function in matmul.cu
template<class TA, class TB, class TC>
void run_fp8_gemm(TA const *A, TB const *B, TC *C, int m, int n, int k, cudaStream_t stream);


torch::Tensor cutlass_matmul(torch::Tensor a, torch::Tensor b){
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    // Get tensor dimensions properly
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);
    
    at::cuda::CUDAGuard device_guard{(char)a.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts = a.options();

    // Create output tensor with proper dimensions (no batch dimension for 2D matmul)
    auto out = torch::empty({m, n}, opts.dtype(torch::kBFloat16));
    
    // Call the templated function with proper type casting
    run_fp8_gemm(
        reinterpret_cast<nv_bfloat16 const*>(a.data_ptr<at::BFloat16>()), 
        reinterpret_cast<nv_bfloat16 const*>(b.data_ptr<at::BFloat16>()), 
        reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()), 
        m, n, k, stream);
    
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("cutlass_matmul", &cutlass_matmul,
          "cute matmul");
}