#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>

using namespace cute;

template<class ElementA,
         class ElementB,
         class SmemLayoutA,
         class SmemLayoutB>
struct SharedStorage
{
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;

    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutB{})];
}

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma>
__global__ static
__launch__bounds(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC *C, CStride dC, TiledMmma mma)
{
    auto [M, N, K] = shape_MNK;
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));

    auto cta_coords = make_coords(blockIdx.x, blockIdx.y, _);
    auto gA = local_tile(mA, cta_tiler, cta_coords, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_tiler, cta_coords, Step<X, _1, _1>{});
    auto gC = local_tile(mC, cta_tiler, cta_coords, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{});

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 2>(sB), group_modes<0, 2>(gB));

    constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
                                         CUTE_STATIC_V(size<0>(tBsA));
    
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
        if ((warp_idx == 0) && lane_predicate) {
            ProducerBarType::init(producer_mbar[pipe], 1);
            ConsumerBarType::init(consumer_mbar[pipe], 128);
        }
    }
    cluster_sync();

    CUTE_UNROLL
    for (int pipe=0; pipe<K_PIPE_MAX; ++pipe) {
        if ((warp_idx==0) && lane_predicate) {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tArA = thr_mma.make_fragment_A(tCsA);
    Tensor tBrB = thr_mma.make_fragment_C(tCsB);

    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>();

    CUTE_UNROLL
    while (k_tile_count > -K_PIPE_MAX) 
    {
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        warpgroup_arrive();
        gemm(mma, tCsA(_, _, _, read_pipe), tCsB(_, _, _, read_pipe), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if ((warp_idx==0) && lane_predicate) {
            int pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }
    
}


template<class TA, class TB, class TC>
void run_fp8_gemm(TA const *A, TB const *B, TC *C,  int m, int n, int k, cudaStream_t stream){
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto prob_shape = make_shape(M, N, K);

    //Strides for TN
    auto dA = make_stride(K, Int<1>{}); 
    auto dB = make_stride(K, Int<1>{});
    auto dA = make_stride(k, Int<1>{});

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto bP = Int<3>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>{});
    
    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(tiled_mma));
    dim3 dimCluster(2, 1, 1);
    dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
                round_up(size(ceil_div(n, bN)), dimCluster.y));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const*>(
                                &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                            TA, decltype(sA, )decltype(tmaA),
                                            TB, decltype(sB), decltype(tmaB),
                                            TC, decltype(dC), decltype(tiled_mma)
                                );
    cutlass::Status status = cutlass:launch_kernek_on_cluster(params, kernel_ptr,
                                                                prob_shape, cta_tiler,
                                                                A, tmaA,
                                                                B, tmaB,
                                                                C, dC, tiled_mma);


}
