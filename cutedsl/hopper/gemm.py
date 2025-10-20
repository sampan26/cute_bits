import cutlass 
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.runtime import from_dlpack
import torch
import math
from typing import Tuple

import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda

class HopperGemm:
    def __init__(self):
        self.acc_dtype = cutlass.Float32
        self.BM = 128
        self.BN = 256
        self.BK = 64
        self.pipeline_stages = 3

        self.cta_tiler = (self.BM, self.BN, self.BK)
        self.atom_layout_mnk = (2,1,1)

        self.num_consumer_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_producer_warp_groups = 1
              
    @cute.jit
    def __call__(
        self, 
        A: cute.Tensor, 
        B: cute.Tensor, 
        C: cute.Tensor, 
        stream: cuda.CUstream
    ):

        self.a_dtype = A.element_type
        self.b_dtype = B.element_type
        self.c_dtype = C.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(A)
        self.b_layout = utils.LayoutEnum.from_tensor(B)
        self.c_layout = utils.LayoutEnum.from_tensor(C)
         
        self.a_smem_layout_staged = _make_smem_layout(
            (self.cta_tiler[0], self.cta_tiler[2]),
            self.a_layout,
            self.a_dtype,
            self.pipeline_stages,
        )

        self.b_smem_layout_staged = _make_smem_layout(
            (self.cta_tiler[1], self.cta_tiler[2]),
            self.b_layout,
            self.b_dtype,
            self.pipeline_stages,
        )

        self.c_smem_layout_staged = _make_smem_layout(
            (self.cta_tiler[0], self.cta_tiler[1]),
            self.c_layout,
            self.c_dtype,
            1,
        )

        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            A,
            cute.slice_(self.a_smem_layout_staged, (None, None, 0)),
            (self.cta_tiler[0], self.cta_tiler[2])
        )

        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            B,
            cute.slice_(self.b_smem_layout_staged, (None, None, 0)),
            (self.cta_tiler[1], self.cta_tiler[2])
        )

        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            C,
            cute.slice_(self.a_smem_layout_staged, (None, None, 0)),
            (self.cta_tiler[0], self.cta_tiler[2])
        )

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            cute.nvgpu.warpgroup.OperandMajorMode.K, # A is K major
            cute.nvgpu.warpgroup.OperandMajorMode.K, # B is K major
            self.acc_dtype,
            (self.num_consumer_warp_groups, 1, 1), # atom_mnk
            (self.cta_tiler[0] // self.num_consumer_warp_groups, self.cta_tiler[1]), # Has to be 64 for M-Major
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_ptr: cute.struct.MemRange[cutlass.Int64, self.pipeline_stages * 2]
            smem_A: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                1024,
            ]
            smem_B: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                1024,
            ]

        self.shared_storage = SharedStorage

        tile_schedule_params, grid = self._compute_grid(
            C,
            self.cta_tiler[0], 
            self.cta_tiler[1],
        )

        num_threads = 128 * (self.num_consumer_warp_groups + self.num_producer_warp_groups)

        self.kernel(
            tma_atom_a,
            tma_atom_b,
            tma_atom_c,
            tma_tensor_a,
            tma_tensor_b,
            tma_tensor_c,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            tile_schedule_params
        ).launch(
            grid=grid,
            block=[num_threads, 1, 1],
            cluster=[1,1,1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        tile_schedule_params: utils.PersistentTileSchedulerParams
    ):
        tidx, _, _ = cute.arch.thread_idx()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        warp_idx_in_warpgroup = cute.arch.make_warp_uniform((tidx // 128) % 4)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // 128)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        is_producer = warpgroup_idx < self.num_producer_warp_groups

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)        

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        
        
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, cute.slice_(a_smem_layout_staged, (None, None, 0))) \
                        + cute.size_in_bytes(self.b_dtype, cute.slice_(b_smem_layout_staged, (None, None, 0)))

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.pipeline_stages,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
        )

        sA = storage.smem_A.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        
        sB = storage.smem_B.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, c_smem_layout_staged.inner, dtype=self.c_dtype),
            c_smem_layout_staged.outer
        )
        
        gA = cute.local_tile(
            mA,
            cute.slice_(self.cta_tiler, (None, 0, None)),
            (None, None)
        )
        gB = cute.local_tile(
            mB,
            cute.slice_(self.cta_tiler, (0, None, None)),
            (None, None)
        )
        gC = cute.local_tile(
            mC,
            (self.cta_tiler[0], self.cta_tiler[1]),
            (None, None)
        )
        
        cta_crd = (0,)
        cta_layout = cute.make_layout((1,))
        tAsA, tAgA  = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cta_crd,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2)
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            cta_crd,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2)
        )

        num_tiles_k = cute.size(tAgA, mode=[2])
        
        # mma_warp_group_thread_layout = cute.make_layout(
        #     self.num_consumer_warp_groups, stride=128
        # )
        # thr_mma = tiled_mma.get_slice(
        #     mma_warp_group_thread_layout(warpgroup_idx - self.num_producer_warp_groups)
        # )

        thr_mma = tiled_mma.get_slice(tidx - self.num_producer_warp_groups * 128)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = thr_mma.make_fragment_A(tCsA)
        tCrB = thr_mma.make_fragment_B(tCsB)

        tCgC = thr_mma.partition_C(gC)
        accumulator = cute.make_fragment(tCgC.shape[:3], self.acc_dtype)

        cute.arch.sync_threads()

        if is_producer:
            cute.arch.warpgroup_reg_dealloc(24)

        if warp_idx == 0:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_schedule_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.pipeline_stages
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_k = tAgA[(None, tile_coord_mnl[0], None)]
                tBgB_k = tBgB[(None, tile_coord_mnl[1], None)]

                mainloop_producer_state.reset_count()

                for k_tile in range(num_tiles_k):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    cute.copy(
                        tma_atom_a,
                        tAgA_k[(None, mainloop_producer_state.count)],
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        )
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k[(None, mainloop_producer_state.count)],
                        tBsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        )
                    )

                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        
        if not is_producer:
            cute.arch.warpgroup_reg_alloc(240);
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_schedule_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.pipeline_stages
            )

            num_wgmma_tiles_k = cute.size(tCrA, mode=[2])

            # copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            #     self.c_layout,
            #     elem_ty_d=self.c_dtype,
            #     elem_ty_acc=self.acc_dtype
            # )

            # tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_mma)
            # thr_copy_r2s = tiled_copy_r2s.get_slice(tidx - self.num_producer_warp_groups * 128)
            # tRS_sD = thr_copy_r2s.partition_D(sC)
            
            # tRS_rAcc = thr_copy_r2s.retile(accumulator)

            # rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            # tRS_rD_layout = cute.make_layout(rD_shape[:3])
            # tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            # tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            # size_tRS_rD = cute.size(tRS_rD)


            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx

                mainloop_consumer_state.reset_count()
                accumulator.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence() # try without

                for k_tile in range(num_tiles_k):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_state)

                    for k_block_idx in cutlass.range_constexpr(num_wgmma_tiles_k):
                        k_block_coord = (None, None, k_block_idx, mainloop_consumer_state.index)
                        cute.gemm(
                            tiled_mma,
                            accumulator,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulator,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    mainloop_pipeline.consumer_release(mainloop_consumer_state)
                    mainloop_consumer_state.advance()
                
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                    
    @staticmethod
    def _make_smem_layout(
        smem_shape: Tuple[int, int],
        layout: utils.LayoutEnum,
        type: type[cutlass.Numeric],
        stages: int,
    ):
        smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                layout,
                type,
                smem_shape[1]
            ),
            type
        ) 

        return cute.tile_to_shape(
            smem_layout_atom,
            cute.append(smem_shape, stages),
            order=(0,1,2)
        )

    @staticmethod
    def _compute_grid(
        C: cute.Tensor,
        BM: int,
        BN: int,
    ):
        gc = cute.zipped_divide(C, (BM, BN))
        num_ctas_mnl = cute.append(gc[(0, (None, None))].shape, 1)
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
    C = torch.zeros(M, K, device="cuda", dtype=torch.float32)
    C_out = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    C_ref = torch.matmul(A, B.t())
    
    A_dlpack = from_dlpack(A, assumed_align=16)
    B_dlpack = from_dlpack(B, assumed_align=16)
    C_dlpack = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    gemm = HopperGemm()
    compiled_kernel = cute.compile(gemm, A_dlpack, B_dlpack, C_dlpack, stream)

    compiled_kernel(A_dlpack, B_dlpack, C_dlpack, stream)

    print(C[0][0])
    # for i in range(C.shape[0]):
    #     for j in range(C.shape[1]):
    #         a = float(C[0][0])
            # a = C[0][0].to(torch.bfloat16)
    # assert torch.allclose(C_out, C_ref, atol=1e-3)
