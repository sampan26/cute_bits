#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/include/cutlass/arch/reg_reconfig.h"
#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

// Add cuBLAS for verification
#include <cublas_v2.h>
#include <cmath>

using namespace cute;


template <typename T, int PIPE, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage
{
    array_aligned<T, cosize_v<SmemLayoutA>, 128> A;
    array_aligned<T, cosize_v<SmemLayoutB>, 128> B;

    typename cutlass::PipelineTmaAsync<PIPE>::SharedStorage pipeline;
};


template <typename T, class ProblemShape, class CtaTiler,
          class SmemLayoutA, class TmaA,
          class SmemLayoutB, class TmaB,
          class SmemLayoutC, class TmaC,
          class SmemCopyAtomC, 
          class CStride, class TiledMma,
          class Alpha, class Beta, 
          int kWarps, int kWarpGroups, int kConsumerWGs, int kThreads,
          int bM, int bN, int bK, int PIPE>
__global__ static
__launch_bounds__(kThreads)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, 
            T const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            T const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            T      * C, CUTLASS_GRID_CONSTANT TmaC const tma_c,
            CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
    auto [M, N, K] = shape_MNK;
    using MainloopPipeline = cutlass::PipelineTmaAsync<PIPE>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    typename MainloopPipeline::Params params;

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<T, PIPE, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage *>(shared_memory);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    int lane_predicate = cute::elect_one_sync();

    if (warp_idx==0 && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
    }

    params.role = warp_group_idx==0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    params.is_leader = (threadIdx.x % 128) == 0;
    params.num_consumers = kConsumerWGs * 128;

    static constexpr size_t kBytesA = sizeof(cute::ArrayEngine<T, size(SmemLayoutA{}(_,_,0))>);
    static constexpr size_t kBytesB = sizeof(cute::ArrayEngine<T, size(SmemLayoutB{}(_,_,0))>);
    params.transaction_bytes = kBytesA + kBytesB;

    MainloopPipeline pipeline(smem.pipeline, params, Shape<_1, _1, _1>{});

    int k_tile_count = ceil_div(K, bK);
    cluster_sync();

    if (warp_group_idx == 0) { // Producer
        cutlass::arch::warpgroup_reg_dealloc<24>();
        int warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        if (warp_idx_in_wg == 0) 
        {
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

            Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
            Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));

            auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
            Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
            Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

            Tensor sA = make_tensor(make_smem_ptr(smem.A.data()), SmemLayoutA{});
            Tensor sB = make_tensor(make_smem_ptr(smem.B.data()), SmemLayoutB{});

            auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                group_modes<0,2>(sA), group_modes<0,2>(gA));
            auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                 group_modes<0,2>(sB), group_modes<0,2>(gB));

            int lane_predicate = cute::elect_one_sync();
            if (lane_predicate) {
                #pragma unroll 1
                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    pipeline.producer_acquire(smem_pipe_write);
                    uint64_t* full_barrier = pipeline.producer_get_barrier(smem_pipe_write);

                    auto pipe = smem_pipe_write.index();
                    copy(tma_a.with(*full_barrier, 0), tAgA(_,k_tile), tAsA(_,pipe));
                    copy(tma_b.with(*full_barrier, 0), tBgB(_,k_tile), tBsB(_,pipe));
                    ++smem_pipe_write;
                }
            }
        }
    }
    else 
    {
        cutlass::arch::warpgroup_reg_alloc<240>();
        PipelineState smem_pipe_read;
        auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

        Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

        auto thr_mma = mma.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);
        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_B(sB);

        Tensor tCrC = partition_fragment_C(TiledMma{}, Shape<Int<bM>, Int<bK>>{});
        clear(tCrC);
        
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);
            
        #pragma unroll 1
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            pipeline.consumer_wait(smem_pipe_read);
            auto read_stage = smem_pipe_read.index();
            warpgroup_arrive();
            gemm(mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<0>();

            pipeline.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
        }
        cutlass::arch::NamedBarrier::sync(kConsumerWGs * 32 * 4, 0);

        // Epilogue Store
        {
            Tensor sC = make_tensor(make_smem_ptr(C), SmemLayoutC{});
            auto r2s_tiled_copy = make_tiled_copy_C(SmemCopyAtomC{}, mma);
            auto r2s_thr_copy = r2s_tiled_copy.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);

            Tensor tAccCrC = r2s_thr_copy.retile_S(tCrC);
            Tensor tAccCsC = r2s_thr_copy.partition_D(sC);

            copy(r2s_thr_copy, tAccCrC, tAccCsC);
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::arrive(kConsumerWGs * 32 * 4 + cutlass::NumThreadsPerWarp, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            Tensor mC = tma_c.get_tma_tensor(make_shape(M, N));
            auto cta_coord = make_coord(blockIdx.x, blockIdx.y);
            Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});
            
            auto s2g_thr_copy = tma_c.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);
            Tensor tCsC = s2g_thr_copy.partition_S(sC);
            Tensor tCgC = s2g_thr_copy.partition_D(gC);

            int write_warp_idx = kWarps - 1;
            int const warp_idx = cutlass::canonical_warp_idx_sync();
            int const lane_predicate = cute::elect_one_sync();
            if (warp_idx == write_warp_idx) {
                cutlass::arch::NamedBarrier::sync(
                    kConsumerWGs * 32 * 4 + cutlass::NumThreadsPerWarp,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
                );
            }
            if (warp_idx == write_warp_idx && lane_predicate) {
                copy(tma_c, tCsC, tCgC);
                tma_store_arrive();
            }
        }
        tma_store_wait<0>();
    }
}

template <typename T, class Alpha, class Beta>
void gemm_nt(int m, int n, int k,
             Alpha alpha, 
             T const* A, int ldA,
             T const* B, int ldB,
             Beta beta,
             T *C, int ldC,
             cudaStream_t stream=0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<256>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<4>{};

    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(bN, bK, bP));
    auto sC = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(bM, bN));

    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);
    Tensor mC = make_tensor(C, make_shape(M, N), dC);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));
    Copy_Atom tmaC = make_tma_atom(SM90_TMA_STORE{}, mC, sC(_, _), make_shape(bM, bN));

    Copy_Atom copy_c = Copy_Atom<SM90_U16x8_STSM_T, T>{};

    static constexpr int kWarps = 12;
    static constexpr int kWarpGroups = kWarps/4;
    static constexpr int kConsumerWGs = kWarpGroups - 1;
    static constexpr int kThreads = kWarps * 32;

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

    dim3 dimBlock(kThreads);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));
    int smem_bytes = int(sizeof(SharedStorage<T, bP, decltype(sA), decltype(sB)>));
    void const* kernel_ptr = reinterpret_cast<void const*>(
                            &gemm_device<T, decltype(prob_shape), decltype(cta_tiler),
                                         decltype(sA), decltype(tmaA),
                                         decltype(sB), decltype(tmaB),
                                         decltype(sC), decltype(tmaC),
                                         decltype(copy_c),
                                         decltype(dC), decltype(tiled_mma),
                                         decltype(alpha), decltype(beta),
                                         kWarps, kWarpGroups, kConsumerWGs, kThreads,
                                         bM, bN, bK, bP>);
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                                prob_shape, cta_tiler,
                                                                A, tmaA,
                                                                B, tmaB,
                                                                C, tmaC, dC, tiled_mma,
                                                                alpha, beta);


}

template <typename T,
          class Alpha, class Beta>
void gemm_tn(int m, int n, int k,
             Alpha alpha, 
             T const* A, int ldA,
             T const* B, int ldB,
             Beta beta,
             T *C, int ldC,
             cudaStream_t stream=0)
{
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<256>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<4>{};

    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(bN, bK, bP));
    auto sC = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(bM, bN));

    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);
    Tensor mC = make_tnesor(C, make_shape(M, N), dC);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));
    Copy_Atom tmaC = make_tma_atom(SM90_TMA_STORE{}, mC, sC(_, _), make_shape(bM, bN));

    Copy_Atom copy_c = Copy_Atom<SM90_U16x8_STSM_T, T>{};

    //TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

    static constexpr int kWarps = 12;
    static constexpr int kWarpGroups = kWarps/4;
    static constexpr int kConsumerWGs = kWarpGroups - 1;
    static constexpr int kThreads = kWarps * 32;
    using mma_op = decltype(
        SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{}
    );
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = kConsumerWGs;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int MmaVM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int MmaVN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int MmaVK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(
        make_layout(
            make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})
        )
    );

    // Thread workload repetition 1x1x1
    // Each mode of this shape can be a layout to do permutation on the corresponding layout mode
    using MMA_V_RepeatT = decltype(
        make_shape(Int<MmaVM>{}, Int<MmaVN>{}, Int<MmaVK>{})
    );
    TiledMMA tiled_mma = make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{});

    dim3 dimBlock(kThreads);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));
    int smem_bytes = int(sizeof(SharedStorage<T, bP, decltype(sA), decltype(sB)>));
    void const* kernel_ptr = reinterpret_cast<void const*>(
                            &gemm_device<T, decltype(prob_shape), decltype(cta_tiler),
                                         decltype(sA), decltype(tmaA),
                                         decltype(sB), decltype(tmaB),
                                         decltype(sC), decltype(tmaC),
                                         decltype(copy_c),
                                         decltype(dC), decltype(tiled_mma),
                                         decltype(alpha), decltype(beta),
                                         kWarps, kWarpGroups, kConsumerWGs, kThreads,
                                         bM, bN, bK, bP>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                                prob_shape, cta_tiler,
                                                                A, tmaA,
                                                                B, tmaB,
                                                                C, tmaC, dC, tiled_mma,
                                                                alpha, beta);


}

template <typename T, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k,
    Alpha alpha,
    T const* A, int ldA,
    T const* B, int ldB,
    Beta beta,
    T * C, int ldC,
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

    using T = cute::half_t;
    using TI = cute::half_t;

    TI alpha = TI(1.0f);
    TI beta = TI(0.0f);

    thrust::host_vector<T> h_A(m*k);
    thrust::host_vector<T> h_B(n*k);
    thrust::host_vector<T> h_C(m*n);

    for (int j = 0; j < m*k; ++j) h_A[j] = T(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n*k; ++j) h_B[j] = T(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < m*n; ++j) h_C[j] = T(0);

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;
    thrust::device_vector<T> d_C_ref = h_C;  // Reference result

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

    // // Initialize cuBLAS
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
    thrust::host_vector<T> cute_result = d_C;
    thrust::host_vector<T> cublas_result = d_C_ref;
    
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
    
    // // Cleanup
    cublasDestroy(cublas_handle);
    return 0;
}