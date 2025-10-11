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


template <class TQ, class TK, class TV, int HEAD_DIM>
void run_flash_attn(int B, int T, int NH, int HS,
                    TQ const* Q, 
                    TK const* K, 
                    TV const* V, 
                    TO const* O,
                    cudaStream_t stream = 0) 
{
    
}

template <class TQ, class TK, class TV>
void flash_attention(int B, int T, int NH, int D,
    TQ const* Q 
    TK const* K
    TV const* V
    TO const* O
    cudaStream_t stream = 0)
{
    run_flash_attn(B, T, NH, D, Q, K, V, O, stream);
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
    flash_attention(B, T, NH, D,
        d_Q.data().get(),
        d_K.data().get(),
        d_V.data().get(),
        d_O.data().get());
    
    // Copy results back to host for verification
    thrust::host_vector<TC> cute_result = d_O;
    thrust::host_vector<TC> cudnn_result = d_O_ref;
    
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
