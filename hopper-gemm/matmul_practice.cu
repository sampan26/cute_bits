#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

// Add cuBLAS for verification
#include <cublas_v2.h>
#include <cmath>

using namespace cute;

__device__ __forceinline__ 
void calculate_tile_coordinates(int tile_idx, int num_tiles_group, int group_size_m, int group_size_n, int num_blocks_m,
                               int& tile_m, int& tile_n, int& grid_m, int& grid_n) {
    int tile_group_idx = tile_idx / num_tiles_group;
    int tile_in_group = tile_idx % num_tiles_group;
    tile_m = tile_in_group % group_size_m;
    tile_n = tile_in_group / group_size_m;
    grid_m = tile_group_idx % (num_blocks_m / group_size_m);
    grid_n = tile_group_idx / (num_blocks_m / group_size_m);
}

template <typename T, int PIPE, class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
struct SharedStorage
{
    array_aligned<T, cosize_v<SmemLayoutA>, 128> smem_A;
    array_aligned<T, cosize_v<SmemLayoutB>, 128> smem_B;
    array_aligned<T, cosize_v<SmemLayoutC>, 128> smem_C;

    typename cutlass::PipelineTmaAsync<PIPE>::SharedStorage pipeline;
};



template <typename T, class ProblemShape, class CtaTiler,
          class SmemLayoutA, class TmaA,
          class SmemLayoutB, class TmaB,
          class SmemLayoutC, class TmaC,
          class SmemCopyAtomC, 
          class TiledMma,
          int kWarps, int kWarpGroups, int kConsumerWGs, int kThreads,
          int bM, int bN, int bK, int PIPE, int cluster_M>
__global__ static 
__launch_bounds__(kThreads)
void 
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, 
    int num_blocks_m, int num_blocks_n,
    T const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    T const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
    T      * C, CUTLASS_GRID_CONSTANT TmaC const tma_c,
    TiledMma mma)
{
    constexpr int group_size_m = 16;
    constexpr int group_size_n = 8;
    constexpr int num_tiles_group = group_size_m * group_size_n;
    const int num_blocks = num_blocks_m * num_blocks_n;

    auto [M, N, K] = shape_MNK;
    int num_k_tiles = cute::ceil_div(K, bK);

    using MainloopPipeline = cutlass::PipelineTmaAsync<PIPE>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    typename MainloopPipeline::Params params;

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<T, PIPE, SmemLayoutA, SmemLayoutB, SmemLayoutC>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    int lane_predicate = cutlass::elect_one_sync();

    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_c.get_tma_descriptor());
    }

    params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    params.is_leader = threadIdx.x % 128 == 0;
    params.num_consumers = kConsumerWGs * 128;

    static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(size(take<0,2>(SmemLayoutA{})) * 2);
    static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(size(take<0,2>(SmemLayoutB{})) * 2);
    params.transaction_bytes = TmaTransactionBytesA + TmaTransactionBytesB;

    MainloopPipeline pipeline(smem.pipeline, params, Shape<Int<cluster_M>, _1, _1>{});

    cluster_arrive_relaxed();
    cluster_wait();

    if (warp_group_idx==0) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        if (warp_idx_in_warpgroup==0)
        {
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

            Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
            Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));

            Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{});
            Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{});
            int tile_m, tile_n, grid_m, grid_n;
            for (int tile_idx = blockIdx.x; tile_idx < num_blocks; tile_idx+=gridDim.x)
            {
                calculate_tile_coordinates(tile_idx, num_tiles_group, group_size_m, group_size_n, num_blocks_m,
                                            tile_m, tile_n, grid_m, grid_n);
                auto cta_coord = make_coord(grid_m * group_size_m + tile_m, grid_n * group_size_n + tile_n, _);

                Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
                Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
                
                auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0,2>(gA));
                auto block_in_cluster = cute::block_rank_in_cluster();
                auto [tBgB, tBsB] = tma_partition(tma_b, block_in_cluster, Layout<Shape<Int<cluster_M>, _1, _1>>{}, group_modes<0,2>(sB), group_modes<0,2>(gB));

                constexpr uint16_t mcast_mask_B = (uint16_t(1) << cluster_M) - 1;
                int lane_predicate = cute::elect_one_sync();
                if (lane_predicate) {
                    #pragma unroll
                    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                        pipeline.producer_acquire(smem_pipe_write);
                        uint64_t* full_barrier = pipeline.producer_get_barrier(smem_pipe_write);
                        int pipe = smem_pipe_write.index();
                        copy(tma_a.with(*full_barrier, 0), tAgA(_, k_tile), tAsA(_, pipe));
                        copy(tma_b.with(*full_barrier, mcast_mask_B), tBgB(_, k_tile), tBsB(_, pipe));
                        ++smem_pipe_write;
                    }
                }
            }
            if (lane_predicate) {
                pipeline.producer_tail(smem_pipe_write);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<240>();
        PipelineState smem_pipe_read;
        int tid = threadIdx.x - cutlass::NumThreadsPerWarpGroup;

        Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{});

        auto thr_mma = mma.get_thread_slice(tid);
        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_B(sB);

        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);
        Tensor tCrC = partition_fragment_C(mma, Shape<Int<bM>, Int<bN>>{});

        mma.accumulate_ = GMMA::ScaleOut::Zero;

        for (int tile_idx=blockIdx.x; tile_idx < num_blocks; tile_idx+=gridDim.x) {
            clear(tCrC);
            int tile_m, tile_n, grid_m, grid_n;
            calculate_tile_coordinates(tile_idx, num_tiles_group, group_size_m, group_size_n, num_blocks_m,
                tile_m, tile_n, grid_m, grid_n);
            
            auto cta_coord = make_coord(grid_m * group_size_m + tile_m, grid_n * group_size_n + tile_n, _);
            #pragma unroll
            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                pipeline.consumer_wait(smem_pipe_read);
                warpgroup_arrive();
                gemm(mma, tCrA(_,_,_,smem_pipe_read.index()),tCrB(_,_,_,smem_pipe_read.index()), tCrC);
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                ++smem_pipe_read;
            }
            cutlass::arch::NamedBarrier::sync(kConsumerWGs * cutlass::NumThreadsPerWarpGroup, 0);

            {
                auto r2s_tiled_copy = make_tiled_copy_C(SmemCopyAtomC{}, mma);
                auto r2s_thr_copy = r2s_tiled_copy.get_thread_slice(tid);
                Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutC{});
                Tensor tAccCrC = r2s_thr_copy.retile_S(tCrC);
                Tensor tAccCsC = r2s_thr_copy.partition_D(sC);
                copy(r2s_tiled_copy, tAccCrC, tAccCsC);
                cute::fence_view_async_shared();

                cutlass::arch::NamedBarrier::arrive(
                    kConsumerWGs * 128 + cutlass::NumThreadsPerWarp, 
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
                );

                Tensor mC = tma_c.get_tma_tensor(make_shape(M, N));
                Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

                auto thr_tma_c = tma_c.get_slice(_0{});
                Tensor tCsC = thr_tma_c.partition_S(sC);
                Tensor tCgC = thr_tma_c.partition_D(gC);

                if (warp_idx == kWarps - 1) {
                    cutlass::arch::NamedBarrier::sync(
                        kConsumerWGs * 128 + cutlass::NumThreadsPerWarp, 
                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
                    );
                    int lane_predicate = cute::elect_one_sync();
                    if (lane_predicate) {
                        copy(tma_c, tCsC, tCgC);
                        tma_store_arrive();
                    }
                }
            }
        }
        tma_store_wait<0>();
    }
}


template <typename T, class Alpha, class Beta>
void gemm_tn(int m, int n, int k,
             Alpha alpha, 
             T const* A, int ldA,
             T const* B, int ldB,
             Beta beta,
             T *C, int ldC,
             cudaStream_t stream=0)
{
    int M = int(m);
    int N = int(m);
    int K = int(m);
    
    auto problem_shape = make_shape(M, N, K);

    auto bM = Int<256>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto bP = Int<3>{};
    auto cluster_M = Int<2>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    static constexpr int kWarps = 12;
    static constexpr int kWarpGroups = kWarps / 4;
    static constexpr int kThreads = kWarpGroups * 128;
    static constexpr int kConsumerWGs = kWarps - 1;

    int num_blocks_m = M / bM;
    int num_blocks_n = N / bN;

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(bN, bK, bP));
    auto sC = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(bM, bN));

    auto tmaA = make_tma_copy(
        SM90_TMA_LOAD{},
        mA,
        sA(_,_,0),
        select<0,2>(cta_tiler),
        _1{}
    );

    auto tmaB = make_tma_copy(
        SM90_TMA_LOAD_MULTICAST{},
        mB,
        sB(_,_,0),
        select<1,2>(cta_tiler),
        cluster_M
    );

    auto tmaC = make_tma_copy(
        SM90_TMA_STORE{},
        mC,
        sC,
        select<0,1>(cta_tiler),
        _1{}
    );

    Copy_Atom SmemCopyC = Copy_Atom<SM90_U16x8_STSM_T, T>{};
    
    using mma_op = decltype(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using mma_shape = mma_traits::Shape_MNK;

    static constexpr int MmaVM = kConsumerWGs * get<0>(mma_shape{});
    static constexpr int MmaVN = get<1>(mma_shape{});
    static constexpr int MmaVK = get<2>(mma_shape{});

    TiledMMA tiled_mma = make_tiled_mma(
        mma_atom{}, 
        Layout<Shape<Int<kConsumerWGs>, Int<1>, Int<1>>>{},
        Shape<Int<MmaVM>, Int<MmaVN>, Int<MmaVK>>{}
    );

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    dim3 dimGrid(128);
    dim3 dimBlock(kThreads);
    dim3 dimCluster(cluster_M, 1, 1);
    
    int smem_bytes = int(sizeof(SharedStorage<T, bP, decltype(sA), decltype(sB), decltype(sB)>));

    void* kernel_ptr = reinterpret_cast<void*>(
        &gemm_device<
            T, decltype(problem_shape), decltype(cta_tiler),
            decltype(sA), decltype(tmaA),
            decltype(sB), decltype(tmaB),
            decltype(sC), decltype(tmaC), decltype(SmemCopyC),
            decltype(tiled_mma), 
            kWarps, kWarpGroups, kConsumerWGs, kThreads,
            bM, bN, bK, bP, cluster_M
        >
    );
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel_ptr, 
        problem_shape, cta_tiler, 
        num_blocks_m, num_blocks_n,
        A, tmaA, 
        B, tmaB, 
        C, tmaC,
        tiled_mma
    );
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
    assert(false && "Only T-N transpose configuration is supported");
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

    printf("Warming up...\n");
    for (int i = 0; i < 5; ++i) {
        gemm(transA, transB, m, n, k,
            alpha,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            beta,
            d_C.data().get(), ldC);
        
        run_cublas_gemm(cublas_handle, transA, transB, m, n, k, 
            static_cast<float>(alpha),
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            static_cast<float>(beta),
            d_C_ref.data().get(), ldC);
    }
    CUTE_CHECK_LAST();
    cudaDeviceSynchronize(); // Ensure all warmup kernels complete
    
    // Time CUTE implementation
    cudaDeviceSynchronize(); // Ensure GPU is idle
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(transA, transB, m, n, k,
            alpha,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            beta,
            d_C.data().get(), ldC);
    }
    cudaDeviceSynchronize(); // Wait for all kernels to complete
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
    
    // Time cuBLAS implementation
    cudaDeviceSynchronize(); // Ensure GPU is idle
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        run_cublas_gemm(cublas_handle, transA, transB, m, n, k, 
            static_cast<float>(alpha),
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            static_cast<float>(beta),
            d_C_ref.data().get(), ldC);
    }
    cudaDeviceSynchronize(); // Wait for all kernels to complete
    double cublas_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("cuBLAS:        [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time, cublas_time*1000);
    
    // Cleanup
    cublasDestroy(cublas_handle);
    return 0;
}
