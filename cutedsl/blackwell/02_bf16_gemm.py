import argparse
import torch
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
cluster_mnk = (2,1,1)
mma_mnk = (256, 256, 16)
mma_tiler = (256, 256, 64)
threads_per_cta = 128

ab_stages = 7
acc_stage = 1

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32
    tmem_dealloc_mbar_ptr: cutlass.Int64


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    mC: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    coords = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),
        bidx // cute.size(cta_layout_vmnk, mode=[0]),
        bidy, 
        None
    )
    coords_mnk = coords[1:]


    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
    
    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0,1,2])) +
        cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0,1,2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])
    num_mcast_ctas_a = cute.size(cta_layout_vmnk, mode=[2])
    num_mcast_ctas_b = cute.size(cta_layout_vmnk, mode=[1])
    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1


    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        ),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            cute.size(cta_layout_vmnk, mode=[0]) * threads_per_cta,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk
    ).make_participants()

    gA = cute.local_tile(mA, mma_tiler, coords_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB, mma_tiler, coords_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC, mma_tiler, coords_mnk, proj=(1, 1, None))

    print(gA.shape)

    thr_mma = tiled_mma.get_slice(coords[0])
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = thr_mma.make_fragment_A(sA)
    tCrB = thr_mma.make_fragment_B(sB)

    print(tCgA.shape)
    print(tCgB.shape)

    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    ) 

    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    ) 

    tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )

    tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

    
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )

    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        is_two_cta=cute.size(cta_layout_vmnk, mode=[0]),
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
    )

    num_cols = 512
    tmem.allocate(num_cols)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)


    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )

    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32
    )
    
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)  # Uses tDgC, not tDtC
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    is_leader_cta = coords[0] == 0
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        if is_leader_cta:
            acc_producer.acquire_and_advance()
        for k_tile in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a, 
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_a,
            )
            cute.copy(
                tma_atom_b, 
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=tma_mcast_mask_b,
            )
            
            if is_leader_cta:
                ab_full = ab_consumer.wait_and_advance()
                num_k_blocks = cute.size(tCrA, mode=[2])
                for block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coords = (None, None, block_idx, ab_full.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coords],
                        tCrB[k_block_coords],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            
                ab_full.release()
        if is_leader_cta:
            acc_producer.commit()

    tmem.relinquish_alloc_permit()

    acc_full = acc_consumer.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    
    acc_full.release()

    if warp_idx == 0:
        ab_producer.tail()
        if is_leader_cta:
            acc_producer.tail()

    pipeline.sync(barrier_id=1)    
    tmem.free(tmem_ptr)

@cute.jit
def gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor
):
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_mnk,
        tcgen05.CtaGroup.TWO,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler,
        a.element_type,
        ab_stages
    )

    print(a_smem_layout)

    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler,
        b.element_type,
        ab_stages
    )
    a_smem_layout_staged = cute.select(a_smem_layout, mode=[0,1,2])
    b_smem_layout_staged = cute.select(b_smem_layout, mode=[0,1,2])

    cta_layout_mnk = cute.make_layout(cluster_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))
    print(cta_layout_vmnk)

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_staged,
        mma_tiler,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_staged,
        mma_tiler,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    grid_shape = cute.round_up(
        cute.ceil_div(
            (*c.layout.shape, 1), (mma_tiler[0] // 2, *mma_tiler[1:]),
        ),
        cluster_mnk
    )
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
        cta_layout_vmnk,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_mnk
    )

def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
):
    m, n, k = mnk
    torch.manual_seed(42)

    def make_tensor(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    a = make_tensor(m, k, cutlass_torch.dtype(io_dtype))
    b = make_tensor(n, k, cutlass_torch.dtype(io_dtype))
    c = make_tensor(m, n, cutlass_torch.dtype(io_dtype))
    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=n)
    )

    gemm(a_tensor, b_tensor, c_tensor, no_cache=True)

    ref = (torch.einsum("mk, nk -> mn", a.to(torch.float32), b.to(torch.float32))).cpu()
    torch.testing.assert_close(c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=tolerance, rtol=1e-05)


def benchmark(func, *args, warmup_steps=10, timing_steps=10, **kwargs):
    # Warmup steps to avoid initial overhead
    for _ in range(warmup_steps):
        func(*args, **kwargs)
    
    # Create CUDA events for accurate timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timing_steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timing_steps)]
    
    # Timing loop with proper CUDA events
    for i in range(timing_steps):
        # Flush cache to avoid cache hits affecting timing
        flush_cache()
        
        # Sleep to saturate command queue and avoid GPU "outrunning" CPU
        torch.cuda._sleep(1_000_000)
        
        # Record timing events around the kernel execution
        start_events[i].record()
        result = func(*args, **kwargs)
        end_events[i].record()
    
    # Synchronize before calculating elapsed times
    torch.cuda.synchronize()
    
    # Calculate elapsed times in milliseconds
    times = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    
    # Return median time
    return sorted(times)[len(times) // 2]


def calculate_flops(M, N, K, time_ms):
    """Calculate FLOPS for attention computation"""
    flops = M * N * K
    
    # With causal mask, only ~half the computation is done
    
    flops_per_second = flops / (time_ms / 1000) 
    return flops_per_second / 1e12
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackwell bf16 gemm")
    mnk = (8192, 8192, 8192)
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )

    args = parser.parse_args()
    run_dense_gemm(
        mnk,
        args.tolerance,
    )

    