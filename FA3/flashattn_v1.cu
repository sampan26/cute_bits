#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/arch/copy.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;



template <class TQ, class TK, class TV, class TO, int D>
void run_flash_attn(int B, int T, int NH,
                    TQ const* Q, 
                    TK const* K, 
                    TV const* V, 
                    TO const* O,
                    cudaStream_t stream = 0) 
{
    auto batch_size = int(B);
    auto seq_len = int(T);
    auto n_heads = int(NH);

    static constexpr int bM = 256;
    static constexpr int bN = 128;
    static constexpr int HEAD_DIM = D;
    static constexpr int bP = 3;
    static constexpr int CLUSER_M = 1;

    static constexpr int q_head_dim_stride = 1;  // stride-1 in head dimension
    static constexpr int q_seq_stride = HEAD_DIM;
    int q_head_stride = HEAD_DIM * seq_len;
    int q_batch_stride = HEAD_DIM * seq_len * n_heads;

    static constexpr int k_head_dim_stride = 1;  // stride-1 in head dimension
    static constexpr int k_seq_stride = HEAD_DIM;
    int k_head_stride = HEAD_DIM * seq_len;
    int k_batch_stride = HEAD_DIM * seq_len * n_heads;

    static constexpr int v_head_dim_stride = 1;  // stride-1 in head dimension
    static constexpr int v_seq_stride = HEAD_DIM;
    int v_head_stride = HEAD_DIM * seq_len;
    int v_batch_stride = HEAD_DIM * seq_len * n_heads;
    
    static constexpr int o_head_dim_stride = 1;  // stride-1 in head dimension
    static constexpr int o_seq_stride = HEAD_DIM;
    int o_head_stride = HEAD_DIM * seq_len;
    int o_batch_stride = HEAD_DIM * seq_len * n_heads;

    using TiledShape_MNK = Shape<Int<bM>, Int<bN>, Int<HEAD_DIM>>;

    auto SmemLayoutAtomQ = cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, TQ, Int<bM>, Int<HEAD_DIM>>();
    auto SmemLayoutAtomK = cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, TK, Int<bN>, Int<HEAD_DIM>>();
    auto SmemLayoutAtomV = cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, TV, Int<bN>, Int<HEAD_DIM>>();
    auto SmemLayoutAtomO = cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, TO, Int<bM>, Int<HEAD_DIM>>();

    auto SmemLayoutQ = tile_to_shape(SmemLayoutAtomQ, make_shape(Int<bM>{}, Int<HEAD_DIM>{}));
    auto SmemLayoutK = tile_to_shape(SmemLayoutAtomK, make_shape(Int<bN>{}, Int<HEAD_DIM>{}, Int<bP>{}));
    auto SmemLayoutO = tile_to_shape(SmemLayoutAtomQ, make_shape(Int<bM>{}, Int<HEAD_DIM>{}));

    auto SmemLayoutV = SmemLayoutK;
    // auto SmemLayoutVt = cute::composition(SmemLayoutV, make_layout(make_shape(Int<HEAD_DIM>{}, Int<bN>{}, Int<bP>{}), make_stride(Int<bN>{}, _1{}, Int<size(SmemLayoutV(_,_,_,_0{})>))));
    auto SmemLayoutVt = cute::composition(SmemLayoutV, make_ordered_layout(make_shape(Int<HEAD_DIM>{}, Int<bN>{}, Int<bP>{}), Step<_2, _1, _3>{}));

    auto LayoutQ = make_layout(
        make_shape(seq_len, HEAD_DIM, n_heads, batch_size),
        make_stride(q_seq_stride, q_head_dim_stride, q_head_stride, q_batch_stride)                                   // stride for B (batch_size)
    );

    auto LayoutK = make_layout(
        make_shape(seq_len, HEAD_DIM, n_heads, batch_size),
        make_stride(k_seq_stride, k_head_dim_stride, k_head_stride, k_batch_stride)                                   // stride for B (batch_size)
    );

    auto LayoutV = make_layout(
        make_shape(seq_len, HEAD_DIM, n_heads, batch_size),
        make_stride(v_seq_stride, v_head_dim_stride, v_head_stride, v_batch_stride)                                   // stride for B (batch_size)
    );

    Tensor mQ = make_tensor(make_gmem_ptr(Q), LayoutQ);
    Tensor mK = make_tensor(make_gmem_ptr(K), LayoutK);
    Tensor mV = make_tensor(make_gmem_ptr(V), LayoutV);

    auto tma_q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        mQ,
        SmemLayoutQ,
        make_shape(Int<bM>{}, Int<HEAD_DIM>{}),
        _1{}
    );

    auto tma_k = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        mK,
        SmemLayoutK(_,_,_0{}),
        make_shape(Int<bN>{}, Int<HEAD_DIM>{}),
        _1{}
    );

    auto tma_v = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        mV,
        SmemLayoutV(_,_,_0{}),
        make_shape(Int<bN>{}, Int<HEAD_DIM>{}),
        _1{}
    );

    using AtomLayoutMNK = Layout<Shape<Int<bM / 64>, _1, _1>>;    

    TiledMMA TiledMmaQK = make_tiled_mma(GMMA::ss_op_selector<TQ, TK, float, TiledShape_MNK>(), AtomLayoutMNK{});
    TiledMMA TiledMmaPV = make_tiled_mma(GMMA::ss_op_selector<TK, TV, float, TiledShape_MNK, GMMA::Major::K, GMMA::Major::MN>(), 
                                        AtomLayoutMNK{});
    
    static constexpr int NUM_WARPS = ((bM / 64) + 1) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int NUM_WARPGROUPS = NUM_WARPS / 4;
    static constexpr int NUM_CONSUMER_GROUPS = NUM_WARPGROUPS - 1;
    static constexpr int NUM_PRODUCER_GROUPS = 1;
    static constexpr int NUM_CONSUMER_THREADS = NUM_CONSUMER_GROUPS * cutlass::NumThreadsPerWarp;
    static constexpr int NUM_PRODUCER_THREADS = cutlass::NumThreadsPerWarp;

    int num_tiles_q = cutlass::ceil_div(seq_len, bM);

    void const* kernel = reinterpret_cast<void const*>()
}

template <class TQ, class TK, class TV, class TO>
void flash_attention(int B, int T, int NH, int D,
    TQ const* Q,
    TK const* K,
    TV const* V,
    TO const* O,
    cudaStream_t stream = 0)
{
    run_flash_attn<TQ, TK, TV, TO, 128>(B, T, NH, Q, K, V, O, stream);
}



int main(int argc, char** argv) {
    int B = 8;
    int T = 2048;      // sequence length
    int NH = 8;      // number of heads
    int HEAD_DIM = 128;     // head size
    int D = NH * HEAD_DIM; // model dim per token

    using TQ = cute::bfloat16_t;
    using TK = cute::bfloat16_t;
    using TV = cute::bfloat16_t;
    using TO = cute::bfloat16_t;

    // Replace lines 141-144 with:
    thrust::host_vector<TQ> h_Q(B*T*D);
    thrust::host_vector<TK> h_K(B*T*D);
    thrust::host_vector<TV> h_V(B*T*D);
    thrust::host_vector<TO> h_O(B*T*D);

    for (int j = 0; j < B*T*D; ++j) h_Q[j] = TQ(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < B*T*D; ++j) h_K[j] = TK(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < B*T*D; ++j) h_V[j] = TV(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < B*T*D; ++j) h_O[j] = TV(0);

    thrust::device_vector<TQ> d_Q = h_Q;
    thrust::device_vector<TK> d_K = h_K;
    thrust::device_vector<TV> d_V = h_V;
    thrust::device_vector<TV> d_O = h_O;
    thrust::device_vector<TO> d_O_ref = h_O;  // Reference result

    // // Initialize cuBLAS
    // cublasHandle_t cublas_handle;
    // cublasCreate(&cublas_handle);

    // printf("Running correctness verification...\n");
    
    // // Run cuBLAS reference
    // d_C_ref = h_C;  // Reset
    // run_cublas_gemm(cublas_handle, transA, transB, m, n, k, 
    //                static_cast<float>(alpha),
    //                d_A.data().get(), ldA,
    //                d_B.data().get(), ldB,
    //                static_cast<float>(beta),
    //                d_C_ref.data().get(), ldC);
    
    // Run CuTe implementation
    d_O = h_O;  // Reset
    flash_attention<TQ, TK, TV, TO>(B, T, NH, HEAD_DIM,
        d_Q.data().get(),
        d_K.data().get(),
        d_V.data().get(),
        d_O.data().get());
    
    // Copy results back to host for verification
    thrust::host_vector<TO> cute_result = d_O;
    thrust::host_vector<TO> cudnn_result = d_O_ref;
    
    // Verify correctness
    // bool passed = verify_matrix(cudnn_result, cute_result, B, T, D);
    // if (passed) {
    //     printf("✓ Correctness verification passed!\n");
    // } else {
    //     printf("✗ Correctness verification FAILED!\n");
    //     cublasDestroy(cublas_handle);
    //     return 1;
    // }
}
