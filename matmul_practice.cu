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


template <typename T, int PIPE, class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
struct SharedStorage {
    array_aligned<T, cosize_v(SmemLayoutA), 128> smem_A;
    array_aligned<T, cosize_v(SmemLayoutB), 128> smem_B;
    array_aligned<T, cosize_v(SmemLayoutC), 128> smem_C;

    typename cutlass::PipelineTmaAsync<PIPE>::SharedStorage pipeline;
}


template <typename T, class ProblemShape, class CtaTiler,
          class SmemLayoutA, class TmaA,
          class SmemLayoutB, class TmaB,
          class SmemLayoutC, class TmaC,
          class SmemCopyAtomC, 
          class CStride, class TiledMma,
          class Alpha, class Beta, 
          int kWarps, int kWarpGroups, int kConsumerWGs, int kThreads,
          int bM, int bN, int bK, int PIPE, int cluster_M>
__global__ static 
__launch_bounds__()
void 
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, 
    int num_blocks_m, int num_blocks_n,
    T const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    T const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
    T      * C, CUTLASS_GRID_CONSTANT TmaC const tma_c,
    CStride dC, TiledMma mma,
    Alpha alpha, Beta beta)
{
    auto [M, N, K] = shape_MNK;
    constexpr int group_size_m = 16;
    constexpr int group_size_n = 8;
    constexpr int num_tiles_group = group_size_m * group_size_n;
    const int num_blocks = num_blocks_m * num_blocks_n;

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<T, PIPE, SmemLayoutA, SmemLayoutB, SmemLayoutC>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    using MainloopPipeline = cutlass::PipelineTmaAsync<PIPE>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    typename MainloopPipeline::Params params;

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int warp_group_idx = cutlass::canonical_warp_group_idx();


    params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    params.is_leader = (threadIdx.x % 128) == 0;
    params.num_consumers = kConsumerWGs * 128; 
    params.transaction_bytes = sizeof(ArrayEngine<T, size(SmemLayoutA{}(_, _, 0))>) + sizeof(ArrayEngine<T, size(SmemLayoutB{}(_, _, 0))>);

    MainloopPipeline pipeline(smem.pipeline, params, Shpae<Int<cluster_M>{}, _1{}, _1{}>);

    int k_tile_count = K / bK;
    cluster_arrive_relaxed();
    cluster_wait();
    
    if (warp_group_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        int warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        if (warp_idx_in_wg == 0) 
        {
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

            Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
            Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));

            Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{});
            Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{});

            for (tile_idx = blockIdx.x; tile_idx < num_blocks; tile_idx+=gridDim.x)
            {
                int tile_group_idx = tile_idx / num_tiles_group;
                int tile_in_group = tile_idx % num_tiles_group;
                int tile_m = tile_in_group % group_size_m;
                int tile_n = tile_in_group / group_size_m;
                int group_m = tile_group_idx % (num_blocks_m / group_size_m);
                int group_n = tile_group_idx / (num_blocks_m / group_size_m);
                auto cta_coord = make_coord(group_m * group_size_m + tile_m, group_n * group_size_n + tile_n, _);

                Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
                Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

                Tensor [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0,2>(gA));
                auto block_in_cluster = cutlass:block_rank_in_cluster()
                Tensor [tBgB, tBsB] = tma_partition(tma_b, block_in_cluster, Layout<Shape<cluster_m, _1, _1>>{}, group_modes<0,2>(sB), group_modes<0,2>(gB));

                int lane_predicate = cute::elect_one_sync();
                constexpr uint16_t mcast_mask = (uint16_t(1) << cluster_M) - 1;

                if (lane_predicate) 
                {
                    #pragma unroll 1
                    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                        pipeline.producer_acquire(smem_pipe_write);
                        uint64_t* full_barrier = pipeline.producer_get_barrier(smem_pipe_write);
                        
                        auto pipe_lane = smem_pipe_write.index();

                        copy(tma_a.with(*full_barrier, 0), tAgA(_, k_tile), tAsA(_, pipe_lane));
                        copy(tma_b.with(*full_barrier, mcast_mask), tBgB(_, k_tile), tBsB(_, pipe_lane));
                        ++smem_pipe_write;
                    }
                }
            }
            int lane_predicate = cute::elect_one_sync();
            if (lane_predicate) { 
                pipeline.producer_tail(smem_pipe_write);
            }
        }
    }
    else
    {
        cutlass::warpgroup_reg_alloc<240>();
        PipelineState smem_pipe_read;
        Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{});

        auto thr_mma = mma.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);
        Tensor tCsA = thr_mma.partition_A(sA);
        Tensor tCsB = thr_mma.partition_A(sB);

        Tensor tCrA = thr_mma.make_fragment_A(tCrA);
        Tensor tCrB = thr_mma.make_fragment_A(tCrB);

        Tensor tCrC = partition_fragment_C(TiledMMA{}, Shape<Int<bM>, Int<bK>>{});
        
        for (int tile_idx = blockIdx.x; tile_idx < num_blocks; tile_idx+=gridDim.x) {
            int tile_group_idx = tile_idx / num_tiles_group;
            int tile_in_group = tile_idx % num_tiles_group;
            int tile_m = tile_in_group % group_size_m;
            int tile_n = tile_in_group / group_size_m;
            int group_m = tile_group_idx % (num_blocks_m / group_size_m);
            int group_n = tile_group_idx / (num_blocks_m / group_size_m);
            auto cta_coord = make_coord(group_m * group_size_m + tile_m, group_n * group_size_n + tile_n, _);

            #pragma unroll 1
            for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                pipeline.consumer_wait(smem_pipe_write);
                auto pipe_lane = smem_pipe_write.index();

                warpgroup_arrive();
                gemm(mma, tCrA(_, _, _, pipe_lane), tCrB(_, _, _, pipe_lane), tCrC);
                warpgroup_commit_batch();
                warpgroup_wait<0>();

                pipeline.consumer_release(smem_pipe_write);
                ++smem_pipe_write;
            }
        }

        {
            Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutC{});

            TiledMMA tiled_mma;
            auto r2s_tiled_copy = make_tiled_copy_C(S2R_COPY_C{}, tiled_mma);
            auto r2s_thr_copy = r2s_thr_copy.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);
            Tensor tAccCrC = r2s_thr_copy.retile_S(tCrC);
            Tensor tAccCsC = r2s_thr_copy.partition_D(sA);
            copy(r2s_tiled_copy, tAccCrC, tAccCsC);
            cutlass::arch::fence_view_async_shared();

            cutlass::arch::NamedBarrier::arrive(
                kConsumerWGs * 4 * 32 + cutlas::NumThreadsPerWarp,
                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
            );

            Tensor mC = tma_c.get_tma_tensor(make_shape(M, N));
            Tensor gC_full = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

            auto cta_tma = tma_c.get_slice(threadIdx.x - cutlass:NumThreadsPerWarpGroup);
            Tensor tCsC = cta_tma.partition_S(sC);
            Tensor tCgC = cta_tma.partition_D(gC_full);

            int write_warp_idx = kWarps - 1;

            if (warp_idx == write_warp_idx && write_warp_idx == warp_idx) {
                cutlss::cutlass::arch::NamedBarrier::arrive(
                    kConsumerWGs * 4 * 32 + cutlas::NumThreadsPerWarp,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
                );
            }
            int lane_predicate = cute::elect_one_sync();
            if (lane_predicate) {
                copy(tma_c, tCsC, tCgC);
                tma_store_arrive();
            }
            tma_store_wait<0>();
        }
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

    static constexpr int kWarps = 12;
    static constexpr int kWarpGroups = kWarps / 4;
    static constexpr int kThreads = kWarpGroups * 128;
    static constexpr int kConsumerWGs = kWarps - 1;

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<128>{};
    auto bP = Int<3>{};
    auto cluter_m = Int<2>{};

    auto prob_shape = make_shape(M, N, K);
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA =  tile_to_shape(GMMA::Layout_K_SW128_Atom<T>, make_shape(bM, bK, bP));  /// *remember tile to shape
    auto sB =  tile_to_shape(GMMA::Layout_K_SW128_Atom<T>, make_shape(bN, bK, bP));
    auto sC =  tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>, make_shape(bM, bN));

    auto mA = make_tensor(A, make_shape(M, K), dA);
    auto mB = make_tensor(B, make_shape(N, K), dB);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK), _1{});
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK), cluster_m);
    Copy_Atom tmaC = make_tma_copy(SM90_TMA_LOAD{}, mC, sC(_, _, 0), make_shape(bM, bN), _1{});

    Copy_Atom S2R_COPY_C = Copy_Atom<SM90_U16x8_STSM_T, T>{};

    using mma_op = decltype(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});
    using mma_trait = MMA_Traits<mma_op>{};
    using mma_atom = MMA_Atom<mma_trait>{};
    
    static constexpr int kMmaEURepeatM = kConsumerWGs;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    using mma_shape = mma_trait::Shape_MNK;
    static constexpr int MmaVM = kMmaEURepeatM * get<0>(mma_atom{});
    static constexpr int MmaVN = kMmaEURepeatN * get<1>(mma_atom{});
    static constexpr int MmaVK = kMmaEURepeatK * get<2>(mma_atom{});

    using MMA_EU_RepeatT = decltype(
        make_layout(
            make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})
        )
    );

    using MMA_V_RepeatT = decltype(
        make_shape(Int<MmaVM>{}, Int<MmaVN>{}, Int<MmaVK>{})
    );

    TiledMMA tiled_mma = make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{});

    int num_blocks_m = M / bM;
    int num_blocks_n = N / bN;

    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    dim3 dimGrid(num_sms);
    dim3 dimBlock(kThreads);
    dim3 dimCluster(cluster_m, 1, 1);
    int smem_size = int(sizeof(SharedStorage<T, decltype(sA), decltype(sB), decltype(sC)>));

    void const* kernel_ptr = reinterpret_cast<void const*>(
                                &gemm_device<T, decltype(prob_shape), decltype(cta_tiler),
                                            decltype(sA), decltype(tmaA),
                                            decltype(sB), decltype(tmaB),
                                            decltype(sC), decltype(tmaC),
                                            decltype(S2R_COPY_C),
                                            decltype(dC), decltype(tiled_mma),
                                            decltype(alpha), decltype(beta),
                                            kWarps, kWarpGroups, kConsumerWGs, kThreads,
                                            bM, bN, bK, bP, cluster_m>
                            );
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_bytes};

    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                                prob_shape, cta_tiler,
                                                                num_blocks_m, num_blocks_n,
                                                                A, tmaA,
                                                                B, tmaB,
                                                                C, tmaC, dC, tiled_mma,
                                                                alpha, beta);
}