#pragma once

#include <cmath>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "utils.h"

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ auto scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
    return tensor;
}


template <int NUM_ROWS_PER_THREAD>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<NUM_ROWS_PER_THREAD>>{}));
    TensorT scores_max, scores_sum, scores_scale;
    float sm_scale_log2;

    CUTLASS_DEVICE Softmax() {}

    CUTLASS_DEVICE Softmax(float sm_scale_log2) : sm_scale_log2(sm_scale_log2) {
        clear(scores_scale);
      };    

    template<bool init, bool check_inf, typename Tensor0>
    __forceinline__ __device__ void update(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        if constexpr (init) {
            reduce_max<true>(scores, scores_max);
            scale_apply_exp2(scores, scores_max, sm_scale_log2);
            reduce_sum<true, false>(scores, scores_sum);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(scores_max);
            cute::copy(scores_max, scores_max_prev);
            reduce_max<false>(scores, scores_max);
            #pragma unroll
            for (int mi = 0; mi < size(scores_max); ++mi) {
                float scores_max_cur = !check_inf ? scores_max(mi) : (scores_max(mi) == -INFINITY ? 0.0f : scores_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * sm_scale_log2);
                scores_sum(mi) *= scores_scale(mi);
            }
            scale_apply_exp2(scores, scores_max, sm_scale_log2);
        }
    };
    template <typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1& acc_o) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == NUM_ROWS_PER_THREAD);
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
        #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale(mi);
            }
        }
    };

    template <typename Tensor0>
    __forceinline__ __device__ void finalize(Tensor0& acc_s, float pv_scale = 1.f) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == NUM_ROWS_PER_THREAD);
        SumOp<float> sum_op;
        quad_allreduce_(scores_sum, scores_sum, sum_op);
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            float sum = scores_sum(mi);
            float inv_sum = pv_scale / sum;
            scores_scale(mi) = inv_sum;
            scores_sum(mi) = scores_max(mi) * sm_scale_log2 + ptx_log2(sum);
        }
    };
    
};