#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

// Add cuBLAS for verification
#include <cublas_v2.h>
#include <cmath>

using namespace cute;


template <typename T, class ProblemShape, class CtaTiler,
          class SmemLayoutA, class TmaA,
          class SmemLayoutB, class TmaB,
          class SmemLayoutC, class TmaC,
          class SmemCopyAtomC, 
          class CStride, class TiledMma,
          class Alpha, class Beta, 
          int kWarps, int kWarpGroups, int kConsumerWGs, int kThreads,
          int bM, int bN, int bK, int PIPE, int cluster_M>
__global__ static 
__launch_bounds__()
void 
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, 
    int num_blocks_m, int num_blocks_n,
    T const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    T const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
    T      * C, CUTLASS_GRID_CONSTANT TmaC const tma_c,
    CStride dC, TiledMma mma,
    Alpha alpha, Beta beta)
{
    
}


template <typename T, class Alpha, class Beta>
void gemm_tn(int m, int n, int k,
             Alpha alpha, 
             T const* A, int ldA,
             T const* B, int ldB,
             Beta beta,
             T *C, int ldC,
             cudaStream_t stream=0)
{
    int M = int(m);
    int N = int(m);
    int K = int(m);

    auto bM = Int<256>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto bP = Int<3>{};
    auto cluter_M = Int<2>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    static constexpr int kWarps = 12;
    static constexpr int kWarpGroups = kWarps / 4;
    static constexpr int kThreads = kWarpGroups * 128;
    static constexpr int kConsumerWGs = kWarps - 1;

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);
    
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom{}, make_shape(bN, bK, bP));
    auto sC = tile_to_shape(GMMA::Layout_MN_SW128_Atom{}, make_shape(bM, bN));

    auto tmaA = make_tma_copy(
        SM90_TMA_LOAD{},
        mA,
        sA(_,_,0),
        
    );

    
    Tensor sA = 
}