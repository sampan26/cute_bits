#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
void check_tensor(const torch::Tensor& x,
                  c10::ScalarType dtype,
                  bool require_cuda = true) {
  TORCH_CHECK(x.is_contiguous(), "Tensor must be contiguous");
  TORCH_CHECK(x.dtype() == dtype, "Unexpected dtype for tensor");
  if (require_cuda) {
    TORCH_CHECK(x.is_cuda(), "Tensor must be on CUDA device");
  }
}


void flash_attention_bf16(torch::Tensor Q,
                          torch::Tensor K,
                          torch::Tensor V,
                          torch::Tensor O,
                          int64_t B,
                          int64_t T,
                          int64_t NH,
                          int64_t HEAD_DIM)
{
  using c10::ScalarType;

  check_tensor(Q, ScalarType::BFloat16);
  check_tensor(K, ScalarType::BFloat16);
  check_tensor(V, ScalarType::BFloat16);
  check_tensor(O, ScalarType::BFloat16);

  TORCH_CHECK(Q.sizes().size() == 3 &&
              K.sizes().size() == 3 &&
              V.sizes().size() == 3 &&
              O.sizes().size() == 3,
              "All tensors must be 3D [B, T, NH*HEAD_DIM]");

  TORCH_CHECK(Q.size(0) == B && Q.size(1) == T && Q.size(2) == NH * HEAD_DIM, "Q shape mismatch");
  TORCH_CHECK(K.sizes() == Q.sizes(), "K shape mismatch vs Q");
  TORCH_CHECK(V.sizes() == Q.sizes(), "V shape mismatch vs Q");
  TORCH_CHECK(O.sizes() == Q.sizes(), "O shape mismatch vs Q");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Raw pointers
  const void* q_ptr = Q.data_ptr();
  const void* k_ptr = K.data_ptr();
  const void* v_ptr = V.data_ptr();
  void*       o_ptr = O.data_ptr();

  // Call the CUDA launcher (defined in flash_attn_kernel.cu)
  extern void flash_attention_bf16_launcher(
      int B, int T, int NH, int HEAD_DIM,
      const void* Q, const void* K, const void* V, void* O,
      cudaStream_t stream);

  flash_attention_bf16_launcher(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(NH),
      static_cast<int>(HEAD_DIM),
      q_ptr, k_ptr, v_ptr, o_ptr, stream);
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attention_bf16", &flash_attention_bf16,
        "Flash Attention (bf16) — expects [B,T,NH*D] bfloat16 CUDA tensors");
}
