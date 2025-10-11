#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/arch/copy.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/util/helper_cuda.hpp"


template <class TQ, class TK, class TV, int HEAD_DIM_QK, int HEAD_DIM_VO>
void run_flash_attn(int B, int T, int NH, int HS,
                    TQ const* Q, 
                    TK const* K, 
                    TV const* V, 
                    TO const* O,
                    cudaStream_t stream = 0) 
{

    using LayoutQ = decltype(make_layout(make_shape(params.qo_len, params.num_qo_heads, HEAD_DIM_QK),
                                         make_stride(params.q_stride_n, params.q_stride_h)));
    
    auto CTA_Q = Int<128>{};
    auto CTA_KV = Int<128>{};
    auto STAGES = Int<2>{};

    auto TiledShape_QKD = make_shape(CTA_Q, CTA_KV, HEAD_DIM_QK);
    auto TiledShape_PDV = make_shape(CTA_Q, HEAD_DIM_VO, CTA_KV);
    auto AtomLayoutQKD = make_shape(CTA_Q/64, _1{}, _1{});

    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<TQ, TK, float, TiledShape_QKD>(), AtomLayoutQKD));
    
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<TK, TV, float, TiledShape_PDV, GMMA::Major::K, GMMA::Major::MN>(), AtomLayoutQKD));

    static constexpr int NUM_MMA_THREADS = size(TiledMmaQK{});

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                    GMMA::Major::K, TQ, CTA_Q, HEAD_DIM_QK>());

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                    GMMA::Major::K, TK, CTA_KV, HEAD_DIM_QK>());

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                    GMMA::Major::K, TQ, CTA_KV, HEAD_DIM_VO>());

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                    GMMA::Major::K, TO, CTA_Q, HEAD_DIM_VO>());
    
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}), make_shape(CTA_Q, HEAD_DIM_QK));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}), make_shape(CTA_KV, HEAD_DIM_QK, STAGES));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}), make_shape(CTA_KV, HEAD_DIM_VO, STAGES));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomV{}), make_ordered_layout(make_shape(HEAD_DIM_VO, CTA_KV, STAGES), Step<_2,_1,_3>{}));
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}), make_shape(CTA_Q, HEAD_DIM_VO));

    using TMA_Q = decltype(make_tma_copy(
        SM90_TMA_LOAD{}, 
        make_tensor(make_gmem_ptr(Q), LayoutQ{}),
        SmemLayoutQ{},
        make_shape(CTA_Q, HEAD_DIM_QK),
        _1{}
    ));

    using TMA_K = decltype(make_tma_copy(
        SM90_TMA_LOAD{}, 
        make_tensor(make_gmem_ptr(K, make_shape(B, T, NH, NS), make_layout(stride_KB,stride_KT,stride_KH,stride_KD))),
        SmemLayoutK{}(_, _, _0{}),
        make_shape(CTA_KV, HEAD_DIM_QK),
        _1{}
    ));

    using TMA_V = decltype(make_tma_copy(
        SM90_TMA_LOAD{}, 
        make_tensor(make_gmem_ptr(B, make_shape(B, T, NH, NS), make_layout(stride_VB,stride_VT,stride_VH,stride_VD))),
        SmemLayoutV{}(_, _, _0{}),
        make_shape(CTA_KV, HEAD_DIM_VO),
        _1{}
    ));

    static constexpr int kWarps = ((CTA_Q / 64) + 1) * 4;
    static constexpr int kWarpGroups = kWarps/4;
    static constexpr int kThreads = kWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kProducerThreads = cutlass::NumThreadsPerWarp;

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    int num_tiles_q = cutlass::ceil_div(T, CTA_Q);
    dim3 dimGrid(num_tiles_q, NH);
    dim3 dimBlock(kThreads);
    int smem_bytes = int(sizeof(SharedStorage<TQ, TK, TV, TO, decltype(SmemLayoutQ{}), decltype(SmemLayoutK{}), decltype(SmemLayoutV{}), decltype(SmemLayoutO{}), 2>));
    void const* kernel_ptr = reinterpret_cast<void const*>(
                                                            &flash_device<TQ, TK, TV, TO, decltype(SmemLayoutQ{}), decltype(SmemLayoutK{}), decltype(SmemLayoutV{}), decltype(SmemLayoutVt{}), decltype(SmemLayoutO{}),
                                                                        decltype(TiledMmaQK{}), decltype(TiledMmaPV{}),
                                                                        decltype(TMA_Q{}), decltype(TMA_K{}), decltype(TMA_V{}), decltype(TMA_O{}),
                                                                        kWarps, kWarpGroups, kConsumerWGs, kThreads,
                                                                        CTA_Q, CTA_KV, HEAD_DIM_QK, HEAD_DIM_VO,
                                                                        decltype(alpha), decltype(beta)>);
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
        num_tiles_q,
        Q, TMA_Q{},
        K, TMA_K{},
        V, TMA_V{},
        O,
        TiledMmaQK{},
        TiledMmaPV{},
        alpha, beta);
}

template <class TQ, class TK, class TV>
void flash_attention(int B, int T, int NH, int HS, int D,
    TQ const* Q, int ldQ,
    TK const* K, int ldK,
    TV const* V, int ldV,
    TO const* O, int ldO,
    cudaStream_t stream = 0)
{
    run_flash_attn(B, T, NH, HS, D, Q, ldQ, K, ldK, V, ldV, O, ldO, stream);
}



int main(int argc, char** argv) {
    int B = 8;
    int T = 2048;      // sequence length
    int NH = 4;      // number of heads
    int HS = 32;     // head size
    int D = NH * HS; // model dim per token

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
    thrust::device_vector<TO> d_O_ref = h_O;  // Reference result

   
    int ldQ = 0, ldK = 0, ldV = 0, ldO = 0;

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
    flash_attention(B, T, NH, HS, D,
        d_Q.data().get(), ldQ,
        d_K.data().get(), ldK,
        d_V.data().get(), ldV,
        d_O.data().get(), ldO);
    
    // Copy results back to host for verification
    thrust::host_vector<TC> cute_result = d_O;
    thrust::host_vector<TC> cudnn_result = d_O_ref;
    
    // Verify correctness
    bool passed = verify_matrix(cudnn_result, cute_result, B, T, D);
    if (passed) {
        printf("✓ Correctness verification passed!\n");
    } else {
        printf("✗ Correctness verification FAILED!\n");
        cublasDestroy(cublas_handle);
        return 1;
    }
}
