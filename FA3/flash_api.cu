#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "flash.h"
// #include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor mha_fwd(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    const float softmax_scale,
    bool is_casaul
) {
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kBFloat16, "only supports bfloat16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and key must have same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen = sizes[1];
    const int num_heads = sizes[2];
    const int head_dim = sizes[3];

    TORCH_CHECK(k.size(1) == seqlen, "Only supports same seqlen for keys and query");
    TORCH_CHECK(k.size(2) == num_heads, "Only supports same number of heads for keys and query");

    TORCH_CHECK(head_dim % 8 == 0, "head dim should be a multiple of 8");
    TORCH_CHECK(head_dim == 128, "Supports only dim 128");

    at::Tensor o = torch::empty_like(q);

    Flash_fwd_params params = {};
    params.b = batch_size;
    params.seqlen = seqlen;
    params.h = num_heads;
    params.d = head_dim;
    params.is_causal = is_casaul;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    //ptrs
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = o.data_ptr();
    params.q_batch_stride=q.stride(0);
    params.k_batch_stride=k.stride(0);
    params.v_batch_stride=v.stride(0);
    params.o_batch_stride=o.stride(0);
    params.q_row_stride=q.stride(1);
    params.k_row_stride=k.stride(1);
    params.v_row_stride=v.stride(1);
    params.o_head_stride=o.stride(1);
    params.q_head_stride=q.stride(2);
    params.k_head_stride=k.stride(2);
    params.v_head_stride=v.stride(2);
    params.o_head_stride=o.stride(2);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd<cutlass::bfloat16_t, 128>(params, stream);
    
    return o;

}

PYBIND11_MODULE(TORCH_EXTENTION_NAME, m) {
    m.doc() = "FlashAttention3";
    m.def("fwd", &mha_fwd);
}
