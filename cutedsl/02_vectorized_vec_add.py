import cutlass
import cutlass.cute as cute

@cute.kernel
def vectorized_elementwise_add_kernel(
  gA: cute.Tensor,
  gB: cute.Tensor,
  gC: cute.Tensor,
):
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  bdim, _, _ = cute.arch.block_dim()

  thread_idx = tidx + bidx * bdim
  n = gA.shape

  a_val = gA[(None, thread_idx)].load()
  b_val = gB[(None, thread_idx)].load()

  gC[(None, thread_idx)] = a_val + b_val

@cute.jit
def solution(d_input1: cute.Tensor, d_input2: cute.Tensor, d_output: cute.Tensor, n: cute.Int32):
  num_threads_per_block = 256

  L = cute.make_layout(shape=(4), stride=(1))

  gA = cute.zipped_divide(d_input1, L)
  gB = cute.zipped_divide(d_input2, L)
  gC = cute.zipped_divide(d_output, L)

  num_blocks = max(cute.shape(gA, mode=[1]) // num_threads_per_block, 1)
  vectorized_elementwise_add_kernel(gA, gB, gC).launch(grid=(num_blocks, 1, 1),
                block=(num_threads_per_block, 1, 1))
  return d_output