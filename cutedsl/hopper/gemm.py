import cutlass 
import cutlass.cute as cute
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
            cutlass.utils.layout.ROW_MAJOR,
            self.a_dtype,
            self.BK
        )
        A_smemload_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            A_smemload_atom_struct,
            self.a_dtype
        )
        B_smemload_atom_struct = sm90_utils.get_smem_layout_atom(
            cutlass.utils.layout.ROW_MAJOR,
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
            ab_barrier_full: cute.struct.MemRange[cutlass.Int64, self.num_ab_pipeline_stages]
            ab_barrier_empty: cute.struct.MemRange[cutlass.Int64, self.num_ab_pipeline_stages]

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

        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tma_tiled_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileGS2Op(),
            A,
            cute.slice_(sA_layout, (None, None, 0)),
            (self.BM, self.BK),
            num_multicast=1
        )

        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tma_tiled_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileGS2Op(),
            B,
            cute.slice_(sB_layout, (None, None, 0)),
            (self.BN, self.BK),
            num_multicast=1
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
        ).launch(
            grid=(1,1,1),
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
        gC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        bdim, _, _ = cute.arch.block_dim()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.outer)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_warp_group = cute.arch.make_warp_uniform(warp_idx % 4)
        warp_group_idx = cute.arch.make_warp_uniform(warp_idx // 4)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
        





if __name__ == "__main__":
    M, N, K = 4096,4096,4096
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    C = torch.randn(M, K, device="cuda", dtype=torch.float32)
    
    A_dlpack = from_dlpack(A, assumed_align=16)
    B_dlpack = from_dlpack(B, assumed_align=16)
    C_dlpack = from_dlpack(C, assumed_align=16)

    gemm = HopperGemm()
