jluiv
import cutlass 
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.runtime import from_dlpack
import torch
import math

import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda

class HopperGemm:
    def __init__(self):
        self.acc_dtype = cutlass.Float32

        self.BM = 256
        self.BN = 128
        self.BK = 64
        self.num_ab_pipeline_stages = 4
        self.cta_tiler = (self.BM, self.BN, self.BK)

        self.atom_layout_mnk = (2,1,1)

        self.consumer_warp_groups = math.prod(self.atom_layout_mnk)
        self.producer_warp_groups = 1
              
    @cute.jit
    def __call__(
        self, 
        A: cute.Tensor, 
        B: cute.Tensor, 
        C: cute.Tensor, 
        stream: cuda.CUstream):
        self.a_dtype = A.element_type
        self.b_dtype = B.element_type
        self.c_dtype = C.element_type
        


        A_smemload_atom_struct = sm90_utils.get_smem_layout_atom(
            cutlass.utils.layout.LayoutEnum.ROW_MAJOR,
            self.a_dtype,
            self.BK
        )
        A_smemload_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            A_smemload_atom_struct,
            self.a_dtype
        )
        B_smemload_atom_struct = sm90_utils.get_smem_layout_atom(
            cutlass.utils.layout.LayoutEnum.ROW_MAJOR,
            self.b_dtype,
            self.BK
        )
        B_smemload_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            B_smemload_atom_struct,
            self.b_dtype
        )

        sA_layout = cute.tile_to_shape(
            A_smemload_atom,
            (self.BM, self.BK, self.num_ab_pipeline_stages),
            order=(0,1,2)
        )
        sB_layout = cute.tile_to_shape(
            B_smemload_atom,
            (self.BN, self.BK, self.num_ab_pipeline_stages),
            order=(0,1,2)
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_pipeline_stages * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(sA_layout)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(sB_layout)],
                1024,
            ]
        self.shared_storage = SharedStorage

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.BN)
        )

        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            A,
            cute.slice_(sA_layout, (None, None, 0)),
            (self.BM, self.BK),
            num_multicast=1
        )

        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            B,
            cute.slice_(sB_layout, (None, None, 0)),
            (self.BN, self.BK),
            num_multicast=1
        )

        tile_schedule_params, grid = self._compute_grid(
            M,
            N,
            self.BM,
            self.BN
        )

        num_threads = 128 * (self.consumer_warp_groups + self.producer_warp_groups)
        self.kernel(
            tiled_mma,
            tma_tensor_a,
            tma_atom_a,
            tma_tensor_b,
            tma_atom_b,
            C,
            sA_layout,
            sB_layout,
            tile_schedule_params
        ).launch(
            grid=grid,
            block=(num_threads, 1, 1),
            cluster=(1,1,1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        mA: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tile_schedule_params: utils.PersistentTileSchedulerParams
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        bdim, _, _ = cute.arch.block_dim()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_warp_group = cute.arch.make_warp_uniform(warp_idx % 4)
        warp_group_idx = cute.arch.make_warp_uniform(warp_idx // 4)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        is_consumer = warp_group_idx == 0 or warp_group_idx == 1
        is_producer = not is_consumer

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        tma_load_bytes = (self.BM * self.BK + self.BN * self.BK) * 2

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.num_ab_pipeline_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_load_bytes,
        )


        gA = cute.local_tile(mA, (self.BM, self.BK), (None, None))
        gB = cute.local_tile(mB, (self.BN, self.BK), (None, None))
        gC = cute.local_tile(mC, (self.BM, self.BN), (None, bidy))

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2)
        )

        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2)
        )

        num_tiles_k = cute.size(gA, mode=[2])

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        print(tCgC) #tensor<ptr<f32, gmem, align<8>> o ((2,2,16),2,1,16):((1,32768,8),524288,0,1048576)>

        tCrA = thr_mma.make_fragment_A(tCsA);
        tCrB = thr_mma.make_fragment_B(tCsB);
        
        tCrC = thr_mma.make_fragment_C(tCgC.shape);
        tCrC.fill(0.0)

        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_fragment(acc_shape, self.acc_dtype)

        cute.arch.sync_threads()

        if is_producer:
            cute.arch.warpgroup_reg_dealloc(24)
            if warp_idx_in_warp_group == 0:
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_schedule_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )

                work_tile = tile_sched.initial_work_tile_info()
                producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_pipeline_stages)

                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    producer_state.reset_count()

                    for tile_k in range(num_tiles_k):
                        mainloop_pipeline.producer_acquire(producer_state)
                        tAgA_mk = tAgA[(None, tile_coord_mnl[0], tile_k)]
                        tBgB_nk = tBgB[(None, tile_coord_mnl[1], tile_k)]
                        
                        cute.copy(
                            tma_atom_a,
                            tAgA_mk,
                            tAsA[(None, producer_state.index)], 
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                        )

                        cute.copy(
                            tma_atom_b,
                            tBgB_nk,
                            tBsB[(None, producer_state.index)], 
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                        )

                        mainloop_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                    
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                
                mainloop_pipeline.producer_tail(producer_state)
        
        if is_consumer:
            cute.arch.warpgroup_reg_alloc(240)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_schedule_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            work_tile = tile_sched.initial_work_tile_info()
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_pipeline_stages)

            num_block_k = cute.size(tCrA, mode=[2])

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx

                consumer_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True) 
                cute.nvgpu.warpgroup.fence()

                for tile_k in range(num_tiles_k):
                    mainloop_pipeline.consumer_wait (consumer_state)

                    for k_idx in range(num_block_k):
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[(None, None, k_idx, consumer_state.index)],
                            tCrB[(None, None, k_idx, consumer_state.index)],
                            accumulators
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    mainloop_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()
                
                store_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
                # cute.copy(store_copy, accumulators, tCgC[None, None, None, tile_coord_mnl[0], tile_coord_mnl[1]])

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


    @staticmethod
    def _compute_grid(
        M: int,
        N: int,
        BM: int,
        BN: int
    ):
        num_ctas_mnl = (M // BM, N // BN, 1)
        cluster_shape_mnl = (1,1,1)
        max_active_clusters = cutlass.const_expr(132)

        tile_schedule_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )

        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_schedule_params, max_active_clusters
        )
        return tile_schedule_params, grid

if __name__ == "__main__":
    M, N, K = 4096,4096,4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    C = torch.randn(M, K, device="cuda", dtype=torch.float32)
    
    A_dlpack = from_dlpack(A, assumed_align=16)
    B_dlpack = from_dlpack(B, assumed_align=16)
    C_dlpack = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    gemm = HopperGemm()
    compiled_kernel = cute.compile(gemm, A_dlpack, B_dlpack, C_dlpack, stream)
    torch_stream.synchronize()

    compiled_kernel(A_dlpack, B_dlpack, C_dlpack, stream)
    # C = C.to(torch.bfloat16)
    # C_ref = torch.matmult(A, B.t())
    # assert torch.allclose(C, C_ref, atol=1e-3)
