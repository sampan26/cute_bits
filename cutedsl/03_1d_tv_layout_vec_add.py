import cutlass
import cutlass.cute as cute

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = (None, bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord] 
    blkC = gC[blk_coord] 

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord] 
    thrB = tidfrgB[thr_coord] 
    thrC = tidfrgC[thr_coord]  

    thrC[None] = thrA.load() + thrB.load()

@cute.jit
def vec_add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, n: cute.Int32):
    thr_layout = cute.make_layout(shape=(512), stride=(1))
    val_layout = cute.group_modes(cute.make_layout(shape=(4,4), stride=(4,1)), 0, 1)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    tiler_1d = cute.group_modes((cute.size(tiler_mn)), 0, 1)

    gA = cute.zipped_divide(A, tiler_1d)
    gB = cute.zipped_divide(B, tiler_1d)
    gC = cute.zipped_divide(C, tiler_1d)

    elementwise_add_kernel(
        gA, gB, gC, tv_layout
    ).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
