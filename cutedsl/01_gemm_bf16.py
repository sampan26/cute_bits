import torch
import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)

ab_stages = 4
epi_stages = 1

# @cute.jit
# def launch_gemm(
#     a: cute.Tensor,
#     b: cute.Tensor,
#     c: cute.Tensor,
# ):
#     op = tcgen05.MmaF16BF16Op(
#         io_dtype,
#         acc_dtype,
#         mma_inst_shape_mnk,
#         tcgen05.CtaGroup.One,
#         tcgen05.OperandSource.SMEM,
#         tcgen05.OperandMajorMode.K,
#         tcgen05.OperandMajorMode.K
#     )

#     tiled_mma = cute.make_tiled_mma(op)

#     a_smem_layout = sm100_utils.make_smem_layout_a(
#         tiled_mma,
#         mma_tiler_mnk,
#         io_dtype,
#         ab_stages,
#     )

#     b_smem_layout = sm100_utils.make_smem_layout_b(
#         tiled_mma,
#         mma_tiler_mnk,
#         io_dtype,
#         ab_stages,
#     )

#     print(a_smem_layout)
#     a_smem_layout_staged = cute.select(a_smem_layout, mode=[0,1,2])
#     b_smem_layout_staged = cute.select(b_smem_layout, mode=[0,1,2])

#     a_tma_atom, a_tma_tensor = cute.nvgpu.make_tma_atom_A(
        
#     )



def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
):
    M, N, K = mnk
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check GPU access in container.")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    

    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    A = make_tensors(M, K, cutlass_torch.dtype(io_dtype))
    B = make_tensors(N, K, cutlass_torch.dtype(io_dtype))
    C = make_tensors(M, N, cutlass_torch.dtype(io_dtype))

    A_dlpack = from_dlpack(A, assumed_align=32).make_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, divisibility=k)
    B_dlpack = from_dlpack(B, assumed_align=32).make_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, divisibility=k)
    C_dlpack = from_dlpack(C, assumed_align=32).make_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, divisibility=n)

    # launch_gemm(
    #     A_dlpack,
    #     B_dlpack,
    #     C_dlpack,
    #     no_cache=True
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnk",
        default=[8192, 8192, 8192],
        help="MNK dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    args = parser.parse_args()

    run_dense_gemm(
        args.mnk,
        args.tolerance,
    )

    print("PASS")
