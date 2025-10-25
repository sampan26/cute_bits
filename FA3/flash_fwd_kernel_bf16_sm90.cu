#include "flash_fwd_kernel.h"

template void run_mha_fwd<cutlass::bfloat16_t, 128>(Flash_fwd_params params, cudaStream_t stream);