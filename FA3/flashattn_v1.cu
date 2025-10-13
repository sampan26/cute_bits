#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/arch/copy.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/util/helper_cuda.hpp"
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"

using namespace cute;

template<bool init=false, int wg_wait=0, typename TensorA, typename TensorB, typename TensorC, typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma& mma, TensorA const& tCrA, TensorB const& tCrB, TensorC& tCrC) {
    constexpr bool Is_RS = 
        !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    if constexpr(Is_RS) {
        warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
    }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    if constexpr (init) {
        mma.accumulate_ = GMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            mma.accumulate_ = GMMA::ScaleOut::One;
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    warpgroup_commit_batch();
    if constexpr(wg_wait >= 0) {
        warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) {
        warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
    }
}

template <int STAGES, class TQ, class TK, class TV, class TO, 
         class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorage {
    cute::array_aligned<TQ, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<TK, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
        cute::array_aligned<TV, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<TO, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename cutlass::PipelineTmaAsync<STAGES>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<STAGES>::SharedStorage pipeline_v;
    };
};

// Device kernel template
template <typename TQ, typename TK, typename TV, typename TO, 
            class CtaTiler,
            class LayoutQ, class LayoutK, class LayoutV,
            class SmemLayoutQ, class TmaQ,
            class SmemLayoutK, class TmaK,
            class SmemLayoutV, class TmaV,
            class SmemLayoutVt, class SmemLayoutO, 
            class TiledMmaQK, class TiledMmaPV,
            int kWarps, int kWarpGroups, int kConsumerThreads, int kThreads,
            int bM, int bN, int bK, int PIPE, int cluster_M>
__global__ static
__launch_bounds__(kThreads, 1)
void
flash_attn_device(CtaTiler cta_tiler, 
                    int num_blocks_m, int num_blocks_n,
                    TQ const* Q, CUTLASS_GRID_CONSTANT TmaQ const tma_q,
                    TK const* K, CUTLASS_GRID_CONSTANT TmaK const tma_k,
                    TV const* V, CUTLASS_GRID_CONSTANT TmaV const tma_v,
                    TO* O, 
                    int bs, int seq_len, int NUM_HEADS, int head_dim)
{
    const int q_tile_idx = blockIdx.x;
    const int batch_head_idx = blockIdx.y;
    const int batch_idx = batch_head_idx / NUM_HEADS;
    const int head_idx = batch_head_idx % NUM_HEADS;
    const int num_kv_tiles = cute::ceil_div((q_tile_idx + 1) * get<0>(cta_tiler), get<1>(cta_tiler));

    using MainloopPipeline = cutlass::PipelineTmaAsync<PIPE>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    typename MainloopPipeline::Params params;

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<PIPE, TQ, TK, TV, TO, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int warp_group_idx = cutlass::canonical_warp_group_idx();
    const int lane_predicate = cute::elect_one_sync();

    if (warp_idx == 0 && lane_predicate) {
        prefetch_tma_descriptor(tma_q.get_tma_descriptor());
        prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        prefetch_tma_descriptor(tma_v.get_tma_descriptor());
    }

    params.is_leader = threadIdx.x % 128 == 0;
    params.num_consumers = kConsumerThreads;
    params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<TQ> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0,2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<TQ> / 8); // faster than init a new aligned_array or array_engine
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(take<0,2>(SmemLayoutV{})) * cutlass::sizeof_bits_v<TQ> / 8); 

    params.transaction_bytes = TmaTransactionBytesK;

    if (warp_idx == 0 && lane_predicate) {
        smem.barrier_Q.init(1);
    }

    MainloopPipeline pipeline_k(smem.pipeline_k, params, Shape<_1, _1, _1>{});
    MainloopPipeline pipeline_v(smem.pipeline_v, params, Shape<_1, _1, _1>{});

    __syncthreads();

    if (warp_group_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<24>();

        int warp_idx_in_wg = __shfl_sync(0xfffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_wg == 0) {
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.data()), SmemLayoutQ{});
            Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.data()), SmemLayoutK{});
            Tensor sV = make_tensor(make_smem_ptr(smem.smem_v.data()), SmemLayoutV{});

            Tensor mQ = tma_q.get_tma_tensor(LayoutQ{}.shape());
            Tensor mK = tma_k.get_tma_tensor(LayoutK{}.shape());
            Tensor mV = tma_v.get_tma_tensor(LayoutV{}.shape());

            Tensor gQ = local_tile(mQ(_, _, head_idx, batch_idx), select<0, 2>(cta_tiler), make_coord(q_tile_idx, _0{}));
            Tensor gK = local_tile(mK(_, _, head_idx, batch_idx), select<1, 2>(cta_tiler), make_coord(_, _0{}));
            Tensor gV = local_tile(mV(_, _, head_idx, batch_idx), select<2, 2>(cta_tiler), make_coord(_, _0{}));

            Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
            Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));

            auto [tQgQ, tQsQ] = tma_partition(tma_q, _0{}, Layout<_1>{}, group_modes<0,2>(sQ_x), group_modes<0,2>(gQ_x));
            auto [tKgK, tKsK] = tma_partition(tma_k, _0{}, Layout<_1>{}, group_modes<0,2>(sK), group_modes<0,2>(gK));
            auto [tVgV, tVsV] = tma_partition(tma_v, _0{}, Layout<_1>{}, group_modes<0,2>(sV), group_modes<0,2>(gV));

            int kv_tile_idx = num_kv_tiles - 1;

            int lane_predicate = cute::elect_one_sync();
            if (lane_predicate) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(tma_k.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0), 
                    tKgK(_, kv_tile_idx), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
            }

            cutlass::arch::NamedBarrier::sync(kConsumerThreads + cutlass::NumThreadsPerWarp, 1);

            if (lane_predicate) {
                smem.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
                copy(tma_q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                    smem.barrier_Q), 0),
                    tQgQ, tQsQ);
            }

            if (lane_predicate) {
                #pragma unroll2
                for (; kv_tile_idx > 0; --kv_tile_idx) {
                    pipeline_k.producer_acquire(smem_pipe_write_k);
                    copy(tma_k.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0), 
                         tKgK(_, kv_tile_idx-1), tKsK(_, smem_pipe_write_k.index()));
                    ++smem_pipe_write_k;
                    pipeline_v.producer_acquire(smem_pipe_write_v);
                    copy(tma_v.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0), 
                         tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
                    ++smem_pipe_write_v;
                }
            }
            if (lane_predicate) {
                pipeline_v.producer_acquire(smem_pipe_write_v);
                    copy(tma_v.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0), 
                         tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
                    ++smem_pipe_write_v;
            }

            if (warp_idx_in_wg == 0 && lane_predicate) {
                pipeline_k.producer_tail(smem_pipe_write_k);
                pipeline_v.producer_tail(smem_pipe_write_v);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<240>();
        const int tid = threadIdx.x - cutlass::NumThreadsPerWarpGroup;
        PipelineState smem_pipe_read_k, smem_pipe_read_v;

        cutlass::arch::NamedBarrier::arrive(kConsumerThreads + cutlass::NumThreadsPerWarp, 1);
        if (warp_group_idx > 1) {
            cutlass::arch::NamedBarrier::arrive(kConsumerThreads, 3 + 1);
        }
        if constexpr (kConsumerThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
            if (warp_group_idx > 2) {
                cutlass::arch::NamedBarrier::arrive(kConsumerThreads, 3 + 2);
            }
        }
        static constexpr int NUM_ROW_PER_THREAD = 2 * (2 * bM / kConsumerThreads);
        using TensorT = decltype(make_tensor<float>(Shape<Int<NUM_ROW_PER_THREAD>>{}));
        TensorT score_max;
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;

        Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(smem.smem_v.data()), SmemLayoutVt{});

        auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
        auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

        Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
        Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
        Tensor tOrV = thr_mma_pv.partition_fragment_B(sVt);
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0,1>(cta_tiler));
        Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0,2>(cta_tiler));

        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
        int kv_tile_idx = num_kv_tiles - 1; 

        cutlass::ConsumerToken token = static_cast<cutlass::BarrierStatus>(smem.barrier_Q.try_wait(0));
        if (token == cutlass::BarrierStatus::WaitAgain) {
            smem.barrier_Q.wait(0);
        }

        pipeline_k.consumer_wait(smem_pipe_read_k, pipeline_k.consumer_try_wait(smem_pipe_read_k));
        cutlass::arch::NamedBarrier::sync(kConsumerThreads, 3 + warp_group_idx);
        gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_read_k.index()), tSrS);
        if constexpr (kConsumerThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
            cutlass::arch::NamedBarrier::arrive(kConsumerThreads, 3 + (3 - warp_group_idx) /*id*/);
        } else {
            cutlass::arch::NamedBarrier::arrive(kConsumerThreads, warp_group_idx <= 2 ? 3 + warp_group_idx + 1 : 3 + warp_group_idx + 1 - 3  /*id*/);
            cutlass::arch::NamedBarrier::arrive(kConsumerThreads, warp_group_idx <= 1 ? 3 + warp_group_idx + 2 : 3 + warp_group_idx + 2 - 3  /*id*/);
        }

        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        Tensor cS = make_identity_tensor(select<0,1>(cta_tiler));
        Tensor tScS = thr_mma_qk.partition_C(cS);

        for (int i = 0; i < size(tScS); ++i) {
            int qo_idx = get<0>(tScS(i)) + q_tile_idx * bM;
            int kv_idx = get<1>(tScS(i)) + kv_tile_idx * bK;
            if (kv_idx >= qo_idx + 1) {
                tSrS(i) = -INFINITY;
            }
        }

        {
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS); ++mi) {
                score_max(mi) = tSrS(mi, 0);
                // for (int ni = 0; ni < size<1>(tSrS); ++ni) {

                // }
            }
        }
        

    }
}

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

    using TiledMmaQK = decltype(make_tiled_mma(GMMA::ss_op_selector<TQ, TK, float, TiledShape_MNK>(), AtomLayoutMNK{}));
    using TiledMmaPV = decltype(make_tiled_mma(GMMA::ss_op_selector<TK, TV, float, TiledShape_MNK, GMMA::Major::K, GMMA::Major::MN>(), 
                                        AtomLayoutMNK{}));
    
    static constexpr int NUM_WARPS = ((bM / 64) + 1) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int NUM_WARPGROUPS = NUM_WARPS / 4;
    static constexpr int NUM_CONSUMER_THREADS = size(TiledMmaQK{});
    static constexpr int NUM_PRODUCER_THREADS = cutlass::NumThreadsPerWarp; 

    int num_tiles_q = cutlass::ceil_div(seq_len, bM);
    void const* kernel = reinterpret_cast<void const*>(
        &flash_attn_device<TQ, TK, TV, TO, 
                          decltype(TiledShape_MNK{}),
                          decltype(LayoutQ), decltype(LayoutK), decltype(LayoutV),
                          decltype(SmemLayoutQ), decltype(tma_q),
                          decltype(SmemLayoutK), decltype(tma_k),
                          decltype(SmemLayoutV), decltype(tma_v),
                          decltype(SmemLayoutVt), decltype(SmemLayoutO), 
                          TiledMmaQK, TiledMmaPV,
                          NUM_WARPS, NUM_WARPGROUPS, NUM_CONSUMER_THREADS,  NUM_THREADS,
                          bM, bN, HEAD_DIM, bP, CLUSER_M>);

    dim3 dimBlock(NUM_THREADS);
    dim3 dimCluster(CLUSER_M, 1, 1);
    dim3 dimGrid(num_tiles_q, n_heads * batch_size);
    
    int smem_bytes = int(sizeof(SharedStorage<bP, TQ, TK, TV, TO,
                                            decltype(SmemLayoutQ), decltype(SmemLayoutK),
                                            decltype(SmemLayoutV), decltype(SmemLayoutO)>));
    
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel, 
                                                            TiledShape_MNK{},
                                                            num_tiles_q, num_tiles_q,
                                                            Q, tma_q,
                                                            K, tma_k, 
                                                            V, tma_v,
                                                            O, 
                                                            batch_size, seq_len, n_heads, HEAD_DIM);
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
