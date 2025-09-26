#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

// Add cuBLAS for verification
#include <cublas_v2.h>
#include <cmath>

using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage
{
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];   
};


template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N));

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)


    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                       group_modes<0,2>(sA), group_modes<0,2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                        group_modes<0,2>(sB), group_modes<0,2>(gB));
    
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));
    auto K_PIPE_MAX = size<1>(tAsA);
    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;
    
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;
    CUTE_UNROLL
    for (int pipe=0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx==0)&&lane_predicate) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }
    cluster_sync();

    CUTE_UNROLL
    for (int pipe=0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx==0) && lane_predicate) {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        ++k_tile;
        --k_tile_count;
    }

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>();

    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        warpgroup_arrive();
        gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_, _,_,read_pipe), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if ((warp_idx==0)&&lane_predicate)
        {
            int pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());

            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }

    axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void gemm_nt(int m, int n, int k,
             Alpha alpha, 
             TA const* A, int ldA,
             TB const* B, int ldB,
             Beta beta,
             TC *C, int ldC,
             cudaStream_t stream=0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

    dim3 dimBlock(size(tiled_mma));
    dim3 dimCluster(2, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(M, bM)), dimCluster.x),
             round_up(size(ceil_div(N, bN)), dimCluster.y));
    int smem_bytes = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    void const* kernel_ptr = reinterpret_cast<void const*>(
                                &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                            TA, decltype(sA), decltype(tmaA),
                                            TB, decltype(sB), decltype(tmaB),
                                            TC, decltype(dC), decltype(tiled_mma),
                                            decltype(alpha), decltype(beta)>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                prob_shape, cta_tiler,
                                            A, tmaA,
                                            B, tmaB,
                                            C, dC, tiled_mma,
                                            alpha, beta);


}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void gemm_tn(int m, int n, int k,
             Alpha alpha, 
             TA const* A, int ldA,
             TB const* B, int ldB,
             Beta beta,
             TC *C, int ldC,
             cudaStream_t stream=0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};

    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

    dim3 dimBlock(size(tiled_mma));
    dim3 dimCluster(2, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
    int smem_bytes = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    void const* kernel_ptr = reinterpret_cast<void const*>(
                            &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                         TA, decltype(sA), decltype(tmaA),
                                         TB, decltype(sB), decltype(tmaB),
                                         TC, decltype(dC), decltype(tiled_mma),
                                         decltype(alpha), decltype(beta)>);
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                                prob_shape, cta_tiler,
                                                                A, tmaA,
                                                                B, tmaB,
                                                                C, dC, tiled_mma,
                                                                alpha, beta);


}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k,
    Alpha alpha,
    TA const* A, int ldA,
    TB const* B, int ldB,
    Beta beta,
    TC * C, int ldC,
    cudaStream_t stream = 0)
{
    if (transA == 'T' && transB == 'N') {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else if (transA == 'N' && transB == 'T') {
        return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    } 
    assert(false && "Not implemented");
}

// Verification function
bool verify_matrix(const thrust::host_vector<cute::half_t>& ref, 
                  const thrust::host_vector<cute::half_t>& test, 
                  int m, int n, float tolerance = 0.1f)
{
    for (int i = 0; i < m * n; i++) {
        float ref_val = static_cast<float>(ref[i]);
        float test_val = static_cast<float>(test[i]);
        float diff = std::abs(ref_val - test_val);
        
        if (diff > tolerance) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   ref_val, test_val, diff, i);
            return false;
        }
    }
    return true;
}

// cuBLAS reference implementation
void run_cublas_gemm(cublasHandle_t handle, char transA, char transB, 
                     int m, int n, int k, float alpha,
                     const cute::half_t* A, int ldA,
                     const cute::half_t* B, int ldB, 
                     float beta, cute::half_t* C, int ldC)
{
    cublasOperation_t opA = (transA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = (transB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    cublasStatus_t status = cublasGemmEx(handle, 
                                        opA, opB, 
                                        m, n, k, 
                                        &alpha, 
                                        A, CUDA_R_16F, ldA,
                                        B, CUDA_R_16F, ldB, 
                                        &beta, 
                                        C, CUDA_R_16F, ldC,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %d\n", status);
        exit(1);
    }
}

int main(int argc, char** argv) {
    int m = 4096;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);
    int n = 4096;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);
    int k = 4096;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    char transA = 'T';
    if (argc >= 5)
        sscanf(argv[4], "%c", &transA);
    char transB = 'N';
    if (argc >= 6)
        sscanf(argv[5], "%c", &transB);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TI = cute::half_t;

    TI alpha = TI(1.0f);
    TI beta = TI(0.0f);

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n*k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TC> d_C_ref = h_C;  // Reference result

    double gflops = (2.0*m*n*k) * 1e-9;

    const int timing_iterations = 100;
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;
    if (transA == 'T') {
        ldA = k;
    }
    else {
        ldA = m;
    }
    if (transB == 'N') {
        ldB = k;
    }
    else {
        ldB = n;
    }

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    printf("Running correctness verification...\n");
    
    // Run cuBLAS reference
    d_C_ref = h_C;  // Reset
    run_cublas_gemm(cublas_handle, transA, transB, m, n, k, 
                   static_cast<float>(alpha),
                   d_A.data().get(), ldA,
                   d_B.data().get(), ldB,
                   static_cast<float>(beta),
                   d_C_ref.data().get(), ldC);
    
    // Run CuTe implementation
    d_C = h_C;  // Reset
    gemm(transA, transB, m, n, k,
        alpha,
        d_A.data().get(), ldA,
        d_B.data().get(), ldB,
        beta,
        d_C.data().get(), ldC);
    
    // Copy results back to host for verification
    thrust::host_vector<TC> cute_result = d_C;
    thrust::host_vector<TC> cublas_result = d_C_ref;
    
    // Verify correctness
    bool passed = verify_matrix(cublas_result, cute_result, m, n);
    if (passed) {
        printf("✓ Correctness verification passed!\n");
    } else {
        printf("✗ Correctness verification FAILED!\n");
        cublasDestroy(cublas_handle);
        return 1;
    }

    printf("Running performance benchmark...\n");
    
    // Performance timing
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(transA, transB, m, n, k,
            alpha,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            beta,
            d_C.data().get(), ldC);
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
    
    // Cleanup
    cublasDestroy(cublas_handle);
    return 0;
}
