import cutlass
import cutlass.cute as cute

@cute.kernel
def naive_elementwise_add_kernel(
  gA: cute.Tensor,
  gB: cute.Tensor,
  gC: cute.Tensor,
):
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  bdim, _, _ = cute.arch.block_dim()

  thread_idx = tidx + bidx * bdim
  n = gA.shape

  a_val = gA[thread_idx]
  b_val = gB[thread_idx]

  gC[thread_idx] = a_val + b_val

@cute.jit
def vec_add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, n: cute.Int32):
  num_threads_per_block = 256
  num_blocks = max(n // num_threads_per_block, 1)

  kernel = naive_elementwise_add_kernel(A, B, C)
  kernel.launch(grid=(num_blocks, 1, 1),
                block=(num_threads_per_block, 1, 1))
  return C