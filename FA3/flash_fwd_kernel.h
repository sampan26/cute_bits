#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "utils.h"
#include "static_switch.h"

using namespace cute;


template<int NumMmaThreads>
CUTLASS_DEVICE inline void scheduler_barrier_arrive(int wg_idx) {
    if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, 3 + (3 - wg_idx) /*id*/);
    } else {
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, wg_idx <= 2 ? 3 + wg_idx + 1 : 3 + wg_idx + 1 - 3);
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, wg_idx <= 1 ? 3 + wg_idx + 2 : 3 + wg_idx + 2 - 3);
    }
}

template<typename Kernel_traits>
struct SharedStorage {
    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK>> smem_k;
    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutV>> smem_v;
    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename MainloopPipeline::SharedStorage pipeline_k;
        typename MainloopPipeline::SharedStorage pipeline_v;
    }
}


template <int HEAD_DIM_, int BM_, int BN_, int NUM_STAGES_, typename T>
struct AttentionKernelTraits {
    using Element = T;
    using index_t = int64_t;

    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int HEAD_DIM = HEAD_DIM_;
    
    static constexpr int NUM_WARPS = ((BM / 64) + 1) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int NUM_PRODUCER_THREADS = cutlass::NumThreadsPerWarp;
    static constexpr int NUM_STAGES = NUM_STAGES_;

    using TileShape = Shape<Int<BM>, Int<BN>, Int<HEAD_DIM>>;

    using TiledMmaQK = decltype(make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, float, TileShape{}>(),
        Layout<Shape<Int<BM/64>, _1>, _1>{}));

    using TiledMmaPV = decltype(make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, float, decltype(select<0,2,1>(TileShape{})),
                                   GMMA::Major::K, GMMA::Major::MN>(), // we use Vt which is N major
        Layout<Shape<Int<BM/64>, _1>, _1>{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<0>(TileShape{})), decltype(get<2>(TileShape{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<1>(TileShape{})), decltype(get<2>(TileShape{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<1>(TileShape{})), decltype(get<2>(TileShape{}))>());
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<0>(TileShape{})), decltype(get<2>(TileShape{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0,2>(TileShape{})));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{}, 
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<NUM_STAGES>{})));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{}, 
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<NUM_STAGES>{})));

    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        make_ordered_layout(make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<NUM_STAGES>{}),
                            Step<_2,_1,_3>)));
    
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0,2>(TileShape{})));

    using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, Element>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<NUM_STAGES>;
    using PipelineState = typename MainloopPipeline::PipelineState<NUM_STAGES>;
    
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
};

template <typename Kernel_traits, bool Is_causal, typename SharedStorage>
__global__ void __launch_bounds__(Kernel_traits::NUM_THREADS, 1 , 1) 
flash_fwd_kernel(__grid_constant__ const Flash_fwd_params params) {
    const int q_tile_idx = blockIdx.x;
    const int batch_head_idx = blockIdx.y;
    const int batch_idx = batch_head_idx / NUM_HEADS;
    const int head_idx = batch_head_idx % NUM_HEADS;

    constexpr int BM = Kernel_traits::BM;
    constexpr int BN = Kernel_traits::BN;
    constexpr int HEAD_DIM = Kernel_traits::HEAD_DIM;
    constexpr int NUM_THREADS = Kernel_traits::NUM_THREADS;
    constexpr int NUM_PRODUCER_THREADS = Kernel_traits::NUM_PRODUCER_THREADS;
    constexpr int NUM_CONSUMER_THREADS = size(typename Kernel_traits::TiledMmaQK{});
    constexpr int NUM_STAGES = Kernel_traits::NUM_STAGES;
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;
    
    using TileShape = Kernel_traits::TileShape;
    const int num_kv_tiles = cute::ceil_div((q_tile_idx + 1) * get<0>(TileShape{}), get<1>(TileShape{}));

    using MainloopPipeline = typename Kernel_traits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    auto LayoutQ = make_layout(
        make_shape(params.seqlen, params.d, params.h, params.b), 
        make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride);
    )

    auto LayoutK = make_layout(
        make_shape(params.seqlen, params.d, params.h, params.b), 
        make_stride(params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride);
    )
    auto LayoutV = make_layout(
        make_shape(params.seqlen, params.d, params.h, params.b), 
        make_stride(params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride);
    )
    auto LayoutO = make_layout(
        make_shape(params.seqlen, params.d, params.h, params.b), 
        make_stride(params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride);
    )

    auto TMA_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(params.q_ptr), LayoutQ),
        Kernel_traits::SmemLayoutQ{},
        make_shape(Int<bM>{}, Int<HEAD_DIM>{}),
        _1{}
    );

    auto TMA_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(params.k_ptr), LayoutK),
        Kernel_traits::SmemLayoutK{}(_,_,_0{}),
        make_shape(Int<BN>{}, Int<HEAD_DIM>{}),
        _1{}
    );

    auto TMA_V = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(params.v_ptr), LayoutV),
        Kernel_traits::SmemLayoutV{}(_,_,_0{}),
        make_shape(Int<BN>{}, Int<HEAD_DIM>{}),
        _1{}
    );

    auto TMA_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr(params.o_ptr), LayoutO),
        Kernel_traits::SmemLayoutO{},
        make_shape(Int<BM>{}, Int<HEAD_DIM>{}),
        _1{}
    );

    extern __shared__ char shared_memory[];
    auto &storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(storage.smem_v.data()), typename Kernel_traits::SmemLayoutVt{});
    Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), typename Kernel_traits::SmemLayoutO{});

    const int lane_predicate = cute::elect_one();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();

    if (warp_idx == 0 && lane_predicate) {
        prefetch_tma_descriptor(TMA_Q.get_tma_descriptor());
        prefetch_tma_descriptor(TMA_K.get_tma_descriptor());
        prefetch_tma_descriptor(TMA_V.get_tma_descriptor());
        prefetch_tma_descriptor(TMA_O.get_tma_descriptor());
    }

    PipelineParams params;
    params.is_leader = threadIdx.x % 128 == 0;
    params.num_consumers = NUM_CONSUMER_THREADS;
    params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    params.transaction_bytes = Kernel_traits::TmaTransactionBytesK;

    if (warp_idx == 0 && lane_predicate) {
        storage.barrier_Q.init(1);
        storage.barrier_O.init(1);
    }

    MainloopPipeline pipeline_k(storage.pipeline_k, params, Shape<_1, _1, _1>{});
    MainloopPipeline pipeline_v(storage.pipeline_v, params, Shape<_1, _1, _1>{});

    __syncthreads();

    if (warp_group_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        int warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_wg == 0) {
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            Tensor mQ = TMA_Q.get_tma_tensor(LayoutQ.shape());
            Tensor mK = TMA_K.get_tma_tensor(LayoutK.shape());
            Tensor mV = TMA_V.get_tma_tensor(LayoutV.shape());

            Tensor gQ = local_tile(mQ(_, _, head_idx, batch_idx), select<0, 2>(TileShape{}), make_coord(q_tile_idx, _0{}));
            Tensor gK = local_tile(mK(_, _, head_idx, batch_idx), select<1, 2>(TileShape{}), make_coord(_, _0{}));
            Tensor gV = local_tile(mV(_, _, head_idx, batch_idx), select<2, 2>(TileShape{}), make_coord(_, _0{}));
            
            Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
            Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));

            auto [tQgQ, tQsQ] = tma_partition(TMA_Q, _0{}, Layout<_1>{}, group_modes<0,2>(sQ_x), group_modes<0,2>(gQ_x));
            auto [tKgK, tKsK] = tma_partition(TMA_K, _0{}, Layout<_1>{}, group_modes<0,2>(sK), group_modes<0,2>(gK));
            auto [tVgV, tVsV] = tma_partition(TMA_V, _0{}, Layout<_1>{}, group_modes<0,2>(sV), group_modes<0,2>(gV));
            
            int kv_tile_idx = num_kv_tiles - 1;
            int lane_predicate = cute::elect_one_sync();
            if (lane_predicate) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(TMA_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0), 
                    tKgK(_, kv_tile_idx), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
            }

            cutlass::arch::NamedBarrier::sync(NUM_CONSUMER_THREADS + cutlass::NumThreadsPerWarp, 1);
            if (lane_predicate) {
                storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
                copy(TMA_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                    storage.barrier_Q), 0),
                    tQgQ, tQsQ);
            }

            if (lane_predicate) {
                #pragma unroll 2
                for (; kv_tile_idx > 0; --kv_tile_idx) {
                    pipeline_k.producer_acquire(smem_pipe_write_k);
                    copy(TMA_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0), 
                         tKgK(_, kv_tile_idx-1), tKsK(_, smem_pipe_write_k.index()));
                    ++smem_pipe_write_k;
                    pipeline_v.producer_acquire(smem_pipe_write_v);
                    copy(TMA_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0), 
                         tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
                    ++smem_pipe_write_v;
                }
            }

            if (lane_predicate) {
                pipeline_v.producer_acquire(smem_pipe_write_v);
                    copy(TMA_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0), 
                         tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
                    ++smem_pipe_write_v;
            }
        }
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_k.producer_tail(smem_pipe_write_k);
            pipeline_v.producer_tail(smem_pipe_write_v);
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<240>();
        const int tid = threadIdx.x - cutlass::NumThreadsPerWarpGroup;
        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        
        cutlass::arch::NamedBarrier::arrive(NUM_CONSUMER_THREADS + cutlass::NumThreadsPerWarp, 1);
        if (warp_group_idx > 1) {
            cutlass::arch::NamedBarrier::arrive(NUM_CONSUMER_THREADS, 3 + 1);
        }
        if constexpr (NUM_CONSUMER_THREADS == 3 * 128) {
            if (warp_group_idx > 2) {
                cutlass::arch::NamedBarrier::arrive(NUM_CONSUMER_THREADS, 3 + 2);
            }
        }

        Softmax<2 * (2 * BM / NUM_CONSUMER_THREADS)>(params.softmax_scale_log2) softmax;
        
        typename Kernel_traits::TiledMmaQK tiled_mma_qk;
        typename Kernel_traits::TiledMmaPV tiled_mma_pv;
        auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
        auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

        Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
        Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
        Tensor tOrV = thr_mma_pv.partition_fragment_B(sVt);
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShape{}));
        Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0,2>(TileShape{}));
        
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
        int kv_tile_idx = num_kv_tiles - 1; 
        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(storage.barrier_Q.try_wait(0));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
            storage.barrier_Q.wait(0);
        }

        pipeline_k.consumer_wait(smem_pipe_read_k, pipeline_k.consumer_try_wait(smem_pipe_read_k));
        cutlass::arch::NamedBarrier::sync(NUM_CONSUMER_THREADS, 3 + warp_group_idx);
        gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_read_k.index()),tSrS);
        scheduler_barrier_arrive<NUM_CONSUMER_THREADS>(warp_group_idx);
        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;
        {
            Tensor cS = cute::make_identity_tensor(select<0,1>(TileShape{}));
            Tensor tScS = thr_mma_qk.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tScS); ++i) {
                int qo_idx = get<0>(tScS(i)) + q_tile_idx * BM;
                int kv_idx = get<1>(tScS(i)) + kv_tile_idx * BN;
                if (kv_idx >= qo_idx + 1) {
                    tSrS(i) = -INFINITY;
                }
            }
        }

        softmax.update<true>(tSrS);
        Tensor tOrP = make_tensor(convert_type<Element>(tSrS.data()), convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaPV>(tSrS.layout()));
        constexpr int masking_steps = ceil_div(BM, BN);
        
        #pragma unroll
        for (int mask_step = 0; mask_step < masking_steps; ++mask_step, --kv_tile_idx) {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShape{})); //clear(tSrS);
            pipeline_k.consumer_wait(smem_pipe_read_k, pipeline_k.consumer_try_wait(smem_pipe_read_k));
            cutlass::arch::NamedBarrier::sync(NUM_CONSUMER_THREADS, 3 + warp_group_idx);
            gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_read_k.index()),tSrS);            
            if (mask_step > 0) {
                softmax.rescale_o(tOrO);
            }

            pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v));
            gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(_,_,_,smem_pipe_read_v.index()),tOrO);
            scheduler_barrier_arrive<NUM_CONSUMER_THREADS>(warp_group_idx);
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);
            {
                Tensor cS = cute::make_identity_tensor(select<0,1>(TileShape{}));
                Tensor tScS = thr_mma_qk.partition_C(cS);
                #pragma unroll
                for (int i = 0; i < size(tScS); ++i) {
                    int qo_idx = get<0>(tScS(i)) + q_tile_idx * BM;
                    int kv_idx = get<1>(tScS(i)) + kv_tile_idx * BN;
                    if (kv_idx >= qo_idx + 1) {
                        tSrS(i) = -INFINITY;
                    }
                }
            }
            softmax.update<false>(tSrS);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(),convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaPV>(tSrS.layout())), tOrP);
        }
        #pragma unroll 1
        for (; kv_tile_idx > 0; --kv_tile_idx) {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShape{})); //clear(tSrS);
            pipeline_k.consumer_wait(smem_pipe_read_k, pipeline_k.consumer_try_wait(smem_pipe_read_k));
            cutlass::arch::NamedBarrier::sync(NUM_CONSUMER_THREADS, 3 + warp_group_idx);
            gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_read_k.index()),tSrS);            
            softmax.rescale_o(tOrO);
            pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v));
            gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(_,_,_,smem_pipe_read_v.index()),tOrO);
            scheduler_barrier_arrive<NUM_CONSUMER_THREADS>(warp_group_idx);
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);
            softmax.update<false>(tSrS);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(),convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaPV>(tSrS.layout())), tOrP);
        }
        softmax.rescale_o(tOrO);
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v));
        gemm<false, -1>(tiled_mma_pv, tOrP, tOrV(_,_,_,smem_pipe_read_v.index()),tOrO);
        softmax.finalize(tSrS);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
        softmax.rescale_o(tOrO);

        {
            auto smem_tiled_copy_O = make_tiled_copy_C(Kernel_traits::SmemCopyAtomO{}, tiled_mma_pv);
            auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);

            Tensor tOrO_out = convert_type<Element>(tOrO);
            Tensor taccOrO = retile_S(tOrO_out);
            Tensor taccOsO = partition_D(sO);

            // cutlass::arch::NamedBarrier::sync(NUM_CONSUMER_THREADS, 0);
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::arrive(NUM_CONSUMER_THREADS + cutlass::NumThreadsPerWarp, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            
            Tensor mO = TMA_O.get_tma_tensor(LayoutO.shape());
            Tensor gO = local_tile(mO(_,_,head_idx, batch_idx), select<0,2>(TileShape{}), make_coord(q_tile_idx, _0{}));
            auto block_tma_O = TMA_O.get_slice(_0{});
            Tensor tOgO = block_tma_O.partition_D(gO);
            Tensor tOsO = block_tma_O.partition_S(sO);

            if (warp_idx == NUM_WARPS - 1) {
                cutlass::arch::NamedBarrier::sync(NUM_CONSUMER_THREADS + cutlass::NumThreadsPerWarp, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                int lane_predicate = cute::elect_one_sync();
                if (lane_predicate) {
                    cute::copy(TMA_O, tOsO, tOgO);
                    tma_store_arrive();
                }   
            }
        }
        tma_store_wait<0>();
    }
}



template<typename AttentionKernelTraits, typename SharedStorage, bool Is_causal>
void run_flash_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    const int num_q_blocks = cute::ceil_div(params.seqlen, Kernel_traits::BM)
    auto& kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, SharedStorage>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(num_q_blocks, params.h, params.b), Kernel_traits::NUM_THREADS, smem_size, stream>>>(params);
}


template <typename T, int Headdim>
void run_mha_fwd(Flash_fwd_params params, cudaStream_t stream) {
    static_assert(Headdim == 128);
    BOOL_SWITCH(params.is_casaul, Is_causal, [&] {
        using Kernel_traits = AttentionKernelTraits<128, 128, Is_causal ? 128 : 178, Is_causal ? 2 : 1, T>;
        run_flash_mha_fwd<Kernel_traits, SharedStorage<Kernel_traits>, Is_causal>(params, stream);
    });
}
