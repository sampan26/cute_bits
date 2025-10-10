#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<class TQ, class TK, class TV, class TO, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutO, int PIPE>
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
        typename cutlass::PipelineTmaAsync<PIPE>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<PIPE>::SharedStorage pipeline_v;
    }
};

enum class NamedBarriers {
    kQueryEmpty = 0,
    kValueEmpty = 1,
    kWarpSchedulerWG1 = 2,
    kWarpSchedulerWG2 = 3,
    kWarpSchedulerWG3 = 4,
    kPrefetchIndices = 5,
    kProducerWG = 6
  };
  


template <class TQ, class TK, class TV, class TO, 
          class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutVt, class SmemLayoutO,
          class TiledMmaQK, class TiledMmaPV,
          class TMA_Q, class TMA_K, class TMA_V, class TMA_O,
          int kWarps, int kWarpGroups, int kConsumerWGs, int kThreads,
          int CTA_Q, int CTA_KV, int HEAD_DIM_QK, int HEAD_DIM_VO, int PIPE>
void flash_device(int num_tiles_q,
                  TQ const* Q, TMA_Q tma_q,
                  TK const* K, TMA_K tma_k,
                  TV const* V, TMA_V tma_v,
                  TO* O,
                  TiledMmaQK tiled_mma_qk,
                  TiledMmaPV tiled_mma_pv)
{
    using MainloopPipeline = cutlass::PipelineTmaAsync<PIPE>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using PipelineParams =  typename MainloopPipeline::Params;

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TQ, TK, TV, TO, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO, PIPE>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory)lane_predicate
    
    const int lane_predicate = cute::elect_one_sync();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    const int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

    const int q_tile_idx = blockIdx.x;
    const int qo_head_idx = blockIdx.y;
    const int kv_head_idx = q_kv_head_ratio / qo_head_idx;
    const int num_kv_tiles = std::min(ceil_div(kv_len, CTA_KV), ceil_div((q_tile_idx + 1) * CTA_Q + kv_len - qo_len, CTA_KV));


    if (warp_idx==0 && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_v.get_tma_descriptor());
    }

    PipelineParams params;
    params.role = warp_group_idx==0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
    params.is_leader = threadIdx.x % 128 == 0;
    params.num_consumers = NUM_MMA_THREADS;
    
    static constexpr uint32_t kBytesQ = sizeof(cute::ArrayEngine<T, size(SmemLayoutQ{})>);
    static constexpr uint32_t kBytesK = sizeof(cute::ArrayEngine<T, size(SmemLayoutK{}(_, _, 0))>);
    static constexpr uint32_t kBytesV = sizeof(cute::ArrayEngine<T, size(SmemLayoutV{}(_, _, 0))>);
    params.transaction_bytes = ;
    __syncthreads();

    if (warp_group_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
        int warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        if (warp_idx_in_wg == 0) {
            int work_idx = 0;
            int kv_tile_idx = num_kv_tiles - 1;
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();
            
            Tensor sQ = make_tensor(make_smem_ptr(smem.smemQ.data()), SmemLayoutAtomQ{});
            Tensor sK = make_tensor(make_smem_ptr(smem.smemK.data()), SmemLayoutAtomK{});
            Tensor sV = make_tensor(make_smem_ptr(smem.smemV.data()), SmemLayoutAtomV{});

            Tensor mQ = tma_q.get_tma_tensor(LayoutQ.shape());
            Tensor mK = tma_k.get_tma_tensor(LayoutK.shape());
            Tensor mV = tma_v.get_tma_tensor(LayoutV.shape());

            Tensor gQ = local_tile(mQ, select<0,2>(TiledShape_QKD{}), qo_head_idx, 0, qo_len)(_, _, q_tile_idx);
            Tensor gK = local_tile(mK, select<1,2>(TiledShape_QKD{}), kv_head_idx, 0, kv_len);
            Tensor gV = local_tile(mV, select<2,1>(TiledShape_QKD{}), kv_head_idx, 0, kv_len);

            Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
            Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));

            auto [tQgQ, tQsQ] = tma_partition(tma_q, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));
            auto [tKgK, tKsK] = tma_partition(tma_q, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));
            auto [tVgV, tVsV] = tma_partition(tma_q, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));
            
            int lane_predicate = cute::elect_one_sync();
            if (lane_predicate) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(tma_k.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0), tKgK(_, kv_tile_idx), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
            }

            cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + kProducerThreads,
                static_cast<int>(NamedBarriers::kQueryEmpty));

            if (lane_predicate) {
                smem.barrier_Q.arrive_and_expect_tx(kBytesQ);
                copy(tma_q.with(*smem.barrier_Q, 0), tQsQ, tQgQ);
            }

            smem.barrier_O.wait((work_idx+1) % 2);

            if (lane_predicate) {
#pragma unroll
                for ( ; kv_tile_idx > 0; --kv_tile_idx) {
                    pipeline_k.producer_acquire(smem_pipe_write_k);
                    copy(tma_k.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), 0), tKgK(_, kv_tile_idx-1), tKsK(_, smem_pipe_write_k.index()));
                    ++smem_pipe_write_k;
                    pipeline_v.producer_acquire(smem_pipe_write_v);
                    copy(tma_v.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0), tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
                    ++smem_pipe_write_v;
                }
            }
            if (lane_predicate) {
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(tma_v.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), 0), tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }

            if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                pipeline_k.producer_tail(smem_pipe_write_k);
                pipeline_v.producer_tail(smem_pipe_write_v);
            }
        }
        else {            
            cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS + kProducerThreads,
                /*id=*/static_cast<int>(NamedBarriers::kQueryEmpty));
                static_assert(NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup ||
                    NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup);
            if (warp_group_idx> 1) {
                cutlass::arch::NamedBarrier::arrive(
                    NUM_MMA_THREADS, /*id=*/static_cast<int>(NamedBarriers::kWarpSchedulerWG1));
            }
            if constexpr (NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup) {
                if (warp_group_idx > 2) {
                cutlass::arch::NamedBarrier::arrive(
                    NUM_MMA_THREADS, /*id=*/static_cast<int>(NamedBarriers::kWarpSchedulerWG2));
                }
            }
            
            int work_idx = 0;
            cutlass::arch::warpgroup_reg_alloc<240>();
            PipelineState smem_pipe_read_k, smem_pipe_read_v;

            Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.data()), SmemLayoutAtomQ{});
            Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.data()), SmemLayoutAtomK{});
            Tensor sVt = make_tensor(make_smem_ptr(smem.smem_v.data()), SmemLayoutAtomVt{});

            TiledMmaQK tiled_mma_qk;
            TiledMmaPV tiled_mma_pv;
            auto thr_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);
            auto thr_mma_pv = tiled_mma_pv.get_thread_slice(threadIdx.x - cutlass::NumThreadsPerWarpGroup);
            
            Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
            Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
            Tensor tOrV = thr_mma_pv.partition_fragment_B(sVt);

            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
            Tensor tSrS = partition_fragment_C(tiled_mma_qk{}, select<0,1>(TiledShape_QKD{}));
            Tensor tOrO = partition_fragment_C(tiled_mma_pv{}, select<0,1>(TiledShape_PDV{}));

            int kv_tile_idx = num_kv_tiles - 1;
            
            cutlass::ConsumerToken barrier_token =
                static_cast<cutlass::BarrierStatus>(smem.barrier_Q.try_wait(work_idx % 2));
            if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
                smem.barrier_Q.wait(work_idx % 2);
            }

            pipeline_k.consumer_wait(smem_pipe_read_k, pipeline_k.consumer_try_wait(smem_pipe_read_k));
            cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS, NamedBarriers::kWarpSchedulerWG1 + warp_group_idx - 1);

            gemm(tiled_mma_qk, tSrQ, tSrK(_,_,_,smem_pipe_read_k.index()), tSrS);

            cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS, (warp_group_idx + 1) % 3 + 1);
            cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS, (warp_group_idx % 3) + 1);

            if (work_idx != 0) {
                int lane_predicate = cute::elect_one_sync();
                if (cutlass::canonical_warp_idx_sync() == kWarps - 1 && lane_predicate) {
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < 1; ++cta_id) {
                        smem.barrier_O.arrive(cta_id, lane_predicate);
                      }
                }
            }

            warpgroup_wait<0>();
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;

            Tensor cS = cute::make_identity_tensor(select<0, 1>(TiledShape_QKD{}));
            Tensor tScS = threadMmaQK.partition_C(cS);

            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                int qo_idx = get<0>(tScS(i)) + q_tile_idx * CTA_Q;
                int kv_idx = get<1>(tScS(i)) + kv_tile_idx * CTA_KV;

                if (qo_idx + 1 + kv_len - qo_len < kv_idx) {
                    tScS(i) = -INFINITY;
                }

                if (kv_idx < qo_idx + kv_len - qo_len) {
                    tSrS(i) = softmax(tSrS(i));
                }
            }

        }
    }


}

template <class TQ, class TK, class TV, int HEAD_DIM_QK, int HEAD_DIM_VO>
void run_flash_attn(int B, int T, int NH, int HS,
                    TQ const* Q, int ldQ,
                    TK const* K, int ldK,
                    TV const* V, int ldV,
                    TO const* O, int ldO,
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

    thrust::host_vector<TA> h_Q(B*T*D);
    thrust::host_vector<TB> h_K(B*T*D);
    thrust::host_vector<TC> h_V(B*T*D);
    thrust::host_vector<TC> h_O(B*T*D);

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
