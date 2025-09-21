#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <cute/tensor.hpp>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)                                                           \
                      
void run_gemm(void *A, void *B, void *C, int BA, int BB, int M, int N, int K, cudaStream_t stream);


torch::Tensor cutlass_matmul(torch::Tensor a, torch::Tensor b){
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    int m, n, k;

    // batch size
    int a_ndim = a.sizes().size();
    int b_ndim = b.sizes().size();

    m, k = a.size();
    n, k = b.size();
    
    at::cuda::CUDAGuard device_guard{(char)a.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts = a.options();

    auto out = torch::empty({batch, m, n}, opts.dtype(kBfloat16));
    run_gemm(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, stream);
    
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("cutlass_matmul", &cutlass_matmul,
          "cute matmul");
}