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

        self.consumer_warp_groups = math.prod(self.atom_layout_mnk)
        self.producer_warp_groups = 1
              
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
            (self.consumer_warp_groups, 1, 1), # atom_mnk
            (self.cta_tiler[0] // self.consumer_warp_groups, self.cta_tiler[1]), # Has to be 64 for M-Major
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

        num_threads = 128 * (self.consumer_warp_groups + self.producer_warp_groups)

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
        bidx, bidy, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        warp_idx_in_warpgroup = cute.arch.warp_idx() % 4
        warp_idx = cute.arch.make_warp_uniform(warp_idx_in_warpgroup)
        warpgroup_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() // 4)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        is_consumer = warpgroup_idx == 0 or warpgroup_idx == 1
        is_producer = not is_consumer

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)        

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_ptr.data_ptr()
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        
        
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

        print(cute.slice_(self.cta_tiler, (None, 0, None)))
        
        gA = cute.local_tile(
            mA,
            (self.cta_tiler[0], self.cta_tiler[2]),
            (None, None)
        )
        gB = cute.local_tile(
            mB,
            (self.cta_tiler[1], self.cta_tiler[2]),
            (None, None)
        )
        gC = cute.local_tile(
            mC,
            (self.cta_tiler[0], self.cta_tiler[1]),
            (None, None)
        )
        
        


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
