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
def solution(d_input1: cute.Tensor, d_input2: cute.Tensor, d_output: cute.Tensor, n: cute.Int32):
  num_threads_per_block = 256
  num_blocks = max(n // num_threads_per_block, 1)

  kernel = naive_elementwise_add_kernel(d_input1, d_input2, d_output)
  kernel.launch(grid=(num_blocks, 1, 1),
                block=(num_threads_per_block, 1, 1))
  return d_output