#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>
#include <stdexcept>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>

#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/trace.h>

// Helper for checking CUDA errors
#define CUTE_CHECK_ERROR(error) \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    }

#define CUTE_CHECK_LAST() CUTE_CHECK_ERROR(cudaGetLastError())

// Your CUTE GEMM Kernel Implementation
namespace gemm_hopper_v00
{

using namespace cute;

template <typename T>
struct Params 
{
    int M, N, K;
    T *C;
    const T alpha;
    const T beta;
};

// shared storage
template <
    typename T,
    typename SmemLayoutA,
    typename SmemLayoutB
>
struct SharedStorage
{
    // data storage
    array_aligned<T, cosize_v<SmemLayoutA>, 128> smem_A;
    array_aligned<T, cosize_v<SmemLayoutB>, 128> smem_B;

    // barrier
    uint64_t smem_A_barrier;
    uint64_t smem_B_barrier;
};

// kernel traits
template <
    typename T,
    int kBlockM_,
    int kBlockN_,
    int kBlockK_
>
struct KernelTraits
{
    using Element = T;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;

    // TiledMMA
    using mma_op = decltype(
        SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{}
    );
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    // thread repetition
    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    // thread workload repetition
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int MmaVM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int MmaVN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int MmaVK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    
    using MMA_EU_RepeatT = decltype(
        make_layout(
            make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})
        )
    );

    using MMA_V_RepeatT = decltype(
        make_shape(Int<MmaVM>{}, Int<MmaVN>{}, Int<MmaVK>{})
    );

    using TiledMMA = decltype(
        make_tiled_mma(
            mma_atom{},
            MMA_EU_RepeatT{},
            MMA_V_RepeatT{}
        )
    );

    using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<T>;

    using SmemLayoutA = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<kBlockM>{}, Int<kBlockK>{})
        )
    );

    using SmemLayoutB = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<kBlockN>{}, Int<kBlockK>{})
        )
    );

    // SharedStorage
    using SharedStorage = SharedStorage<T, SmemLayoutA, SmemLayoutB>;

    static constexpr int smem_size = sizeof(SharedStorage);
};

// kernel
template <
    typename ParamsT,
    typename Kernel_traits,
    typename TmaLoadA,
    typename TmaLoadB

>
__global__ void cute_hopper_gemm_v00(
    ParamsT params,
    CUTE_GRID_CONSTANT TmaLoadA const tma_load_A,
    CUTE_GRID_CONSTANT TmaLoadB const tma_load_B
)
{   
    using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
    using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kBlockK = Kernel_traits::kBlockK;

    Tensor mA = tma_load_A.get_tma_tensor(make_shape(params.M, params.K));
    Tensor mB = tma_load_B.get_tma_tensor(make_shape(params.N, params.K));
    Tensor mC = make_tensor(
        make_gmem_ptr(params.C),
        make_shape(params.M, params.N),
        make_stride(_1{}, params.M)
    );

    auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    Tensor gA = local_tile(
        mA,
        cta_tiler,
        cta_coord,
        Step<_1, X, _1>{}
    ); 
    Tensor gB = local_tile(
        mB,
        cta_tiler,
        cta_coord,
        Step<X, _1, _1>{}
    ); 
    Tensor gC = local_tile(
        mC,
        cta_tiler,
        cta_coord,
        Step<_1, _1, X>{}
    ); 

    using SharedStorage = typename Kernel_traits::SharedStorage;
    extern __shared__ char smem_[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    Tensor sA = make_tensor(
        make_smem_ptr(shared_storage.smem_A.data()),
        SmemLayoutA{}
    );
    Tensor sB = make_tensor(
        make_smem_ptr(shared_storage.smem_B.data()),
        SmemLayoutB{}
    ); 
    
    auto [tAgA, tAsA] = tma_partition(
        tma_load_A, Int<0>{}, Layout<_1>{},
        group_modes<0,2>(sA), group_modes<0,2>(gA)
    );
    auto [tBgB, tBsB] = tma_partition(
        tma_load_B, Int<0>{}, Layout<_1>{},
        group_modes<0,2>(sB), group_modes<0,2>(gB)
    ); 

    typename Kernel_traits::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;
    using T = typename Kernel_traits::Element;
    constexpr int kTmaTransactionBytesA = cute::cosize_v<SmemLayoutA> * sizeof_bits_v<T> / 8;
    constexpr int kTmaTransactionBytesB = cute::cosize_v<SmemLayoutB> * sizeof_bits_v<T> / 8;

    uint64_t& smem_A_barrier = shared_storage.smem_A_barrier;
    uint64_t& smem_B_barrier = shared_storage.smem_B_barrier;

    if (warp_idx == 0 && lane_predicate)
    {
        TransactionBarrier::init(&smem_A_barrier, 1);
        TransactionBarrier::init(&smem_B_barrier, 1);
    }

    __syncthreads();

    auto NUM_TILES_K = size<2>(gA);

    #pragma unroll 1
    for (int k_tile = 0; k_tile < NUM_TILES_K; ++k_tile) {
        if (warp_idx == 0 && lane_predicate)
        {
            TransactionBarrier::arrive_and_expect_tx(&smem_A_barrier, kTmaTransactionBytesA);
            TransactionBarrier::arrive_and_expect_tx(&smem_B_barrier, kTmaTransactionBytesB);
            copy(tma_load_A.with(smem_A_barrier), tAgA(_, k_tile), tAsA);
            copy(tma_load_B.with(smem_B_barrier), tBgB(_, k_tile), tBsB);
        }

        TransactionBarrier::wait(&smem_A_barrier, (k_tile + 1) % 2);
        TransactionBarrier::wait(&smem_B_barrier, (k_tile + 1) % 2);

        cute::warpgroup_arrive();
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        cute::warpgroup_commit_batch();
        cute::warpgroup_wait<0>();
    }

    axpby(params.alpha, tCrC, params.beta, tCgC);
}

// launch
template<typename T>
void launch_cute_hopper_gemm_kernel_v00(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream
)
{   
    using ParamsT = Params<T>;
    ParamsT params = {int(m), int(n), int(k), C, *alpha, *beta};

    constexpr int kBlockM_ = 256;
    constexpr int kBlockN_ = 128;
    constexpr int kBlockK_ = 64;

    using Kernel_traits = KernelTraits<T, kBlockM_, kBlockN_, kBlockK_>;
    using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
    using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
    using TiledMMA = typename Kernel_traits::TiledMMA;

    Tensor mA = make_tensor(
        make_gmem_ptr(A),
        make_shape(params.M, params.K),
        make_stride(lda, _1{})
    );
    Tensor mB = make_tensor(
        make_gmem_ptr(B),
        make_shape(params.N, params.K),
        make_stride(ldb, _1{})
    );

    auto tma_load_A = make_tma_copy(
        SM90_TMA_LOAD{},
        mA,
        SmemLayoutA{}
    );

    auto tma_load_B = make_tma_copy(
        SM90_TMA_LOAD{},
        mB,
        SmemLayoutB{}
    );

    constexpr int smem_size = Kernel_traits::smem_size;
    dim3 block{cute::size(TiledMMA{}), 1U, 1U};
    dim3 cluster{1, 1, 1};
    dim3 grid{
        cute::size(ceil_div(params.N, kBlockN_)),
        cute::size(ceil_div(params.M, kBlockM_)),
        1U
    };

    cutlass::ClusterLaunchParams launch_params{grid, block, cluster, smem_size, stream};

    void const* kernel = reinterpret_cast<void const*>(&cute_hopper_gemm_v00 <
            ParamsT,
            Kernel_traits,
            decltype(tma_load_A),
            decltype(tma_load_B)
        >
    );

    if (smem_size >= (48 * 1024))
    {
        CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params,
        kernel,
        params,
        tma_load_A,
        tma_load_B
    );
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "Kernel launch failed!" << std::endl;
    }
}

// explicit instantiation                      
template void launch_cute_hopper_gemm_kernel_v00<cute::half_t>(size_t m, size_t n, size_t k,
                                    const cute::half_t *alpha,
                                    const cute::half_t *A, size_t lda,
                                    const cute::half_t *B, size_t ldb,
                                    const cute::half_t *beta,
                                    cute::half_t *C, size_t ldc,
                                    cudaStream_t stream);                                    

} // namespace gemm_hopper_v00

// GPU Timer
struct GPU_Clock {
    cudaEvent_t start_event, stop_event;

    GPU_Clock() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GPU_Clock() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, 0);
    }

    float stop() {
        float elapsed_time;
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        return elapsed_time;
    }
    
    // Returns time in seconds
    double seconds() {
        return static_cast<double>(stop()) / 1000.0;
    }
};

// Verification function
bool verify_matrix(const thrust::host_vector<cute::half_t>& ref, 
                  const thrust::host_vector<cute::half_t>& test, 
                  int m, int n, float tolerance = 0.1f)
{
    for (int i = 0; i < m * n; ++i) {
        float ref_val = __half2float(ref[i]);
        float test_val = __half2float(test[i]);
        float diff = std::abs(ref_val - test_val);
        
        if (diff > tolerance) {
            printf("Divergence at index %d! Reference: %5.2f, Test: %5.2f, Diff: %5.2f\n",
                   i, ref_val, test_val, diff);
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
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %d\n", status);
        exit(1);
    }
}


int main() {
    // Problem size
    int m = 4096;
    int n = 4096;
    int k = 4096;
    
    cute::half_t alpha = cute::float_to_half(1.0f);
    cute::half_t beta = cute::float_to_half(0.0f);

    // Assuming column-major layout like cuBLAS (NN)
    // A (m,k), B(k,n), C(m,n)
    // CUTE kernel expects A(m, k) and B(n, k) -> we must transpose B
    char transA = 'N';
    char transB = 'N';
    int ldA = m;
    int ldB = k;
    int ldC = m;

    printf("Problem: M=%d, N=%d, K=%d, alpha=1.0, beta=0.0\n", m, n, k);

    // Allocate host memory
    thrust::host_vector<cute::half_t> h_A(m * k);
    thrust::host_vector<cute::half_t> h_B(k * n);
    thrust::host_vector<cute::half_t> h_C(m * n, cute::float_to_half(0.0f));

    // Initialize host matrices with random data
    srand(time(0));
    for(int i = 0; i < m * k; ++i) h_A[i] = cute::float_to_half((float)(rand() % 100) / 10.0f);
    for(int i = 0; i < k * n; ++i) h_B[i] = cute::float_to_half((float)(rand() % 100) / 10.0f);

    // Allocate device memory
    thrust::device_vector<cute::half_t> d_A = h_A;
    thrust::device_vector<cute::half_t> d_B = h_B;
    thrust::device_vector<cute::half_t> d_C = h_C;
    thrust::device_vector<cute::half_t> d_C_ref = h_C;

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    // --- Correctness Verification ---
    printf("Running correctness verification...\n");

    // Run CUTE GEMM
    gemm_hopper_v00::launch_cute_hopper_gemm_kernel_v00(
        m, n, k,
        &alpha,
        d_A.data().get(), m, // For A(m,k) col-major, stride is m
        d_B.data().get(), k, // For B(k,n) col-major, stride is k, but kernel wants (n,k)
        &beta,
        d_C.data().get(), m, // For C(m,n) col-major, stride is m
        0
    );
    cudaDeviceSynchronize();

    // Run cuBLAS reference
    run_cublas_gemm(cublas_handle, transA, transB, m, n, k, 
                   1.0f,
                   d_A.data().get(), ldA,
                   d_B.data().get(), ldB,
                   0.0f,
                   d_C_ref.data().get(), ldC);
    cudaDeviceSynchronize();

    // Copy results back to host for verification
    thrust::host_vector<cute::half_t> cute_result = d_C;
    thrust::host_vector<cute::half_t> cublas_result = d_C_ref;
    
    // Verify correctness
    bool passed = verify_matrix(cublas_result, cute_result, m, n);
    if (passed) {
        printf("✓ Correctness verification passed!\n\n");
    } else {
        printf("✗ Correctness verification FAILED!\n");
        cublasDestroy(cublas_handle);
        return 1;
    }

    // --- Performance Benchmarking ---
    const int timing_iterations = 100;
    GPU_Clock timer;
    double gflops = 2.0 * m * n * k / 1e9;
    
    printf("Running performance benchmark (%d iterations)...\n", timing_iterations);

    // Time CUTE implementation
    cudaDeviceSynchronize();
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm_hopper_v00::launch_cute_hopper_gemm_kernel_v00(
            m, n, k, &alpha,
            d_A.data().get(), m,
            d_B.data().get(), k,
            &beta,
            d_C.data().get(), m, 0);
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f] GFLOP/s  (%6.4f ms)\n", gflops / cute_time, cute_time * 1000);
    
    // Time cuBLAS implementation
    cudaDeviceSynchronize();
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        run_cublas_gemm(cublas_handle, transA, transB, m, n, k, 
            1.0f,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            0.0f,
            d_C_ref.data().get(), ldC);
    }
    double cublas_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("cuBLAS:        [%6.1f] GFLOP/s  (%6.4f ms)\n", gflops / cublas_time, cublas_time * 1000);

    // Cleanup
    cublasDestroy(cublas_handle);
    return 0;
}