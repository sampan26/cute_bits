#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>

using namespace cute;

template<class TA, class TB, class TC>
void run_fp8_gemm(TA const *A, TB const *B, TC *C,  int m, int n, int k, cudaStream_t stream){
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    //Strides for TN
    auto dA = make_stride(K, Int<1>{}); 
    auto dB = make_stride(K, Int<1>{});
    auto dA = make_stride(k, Int<1>{});

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto bP = Int<3>{};

    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_BF16BF16BF16<GMMA::Major::MN, GMMA::Major::MN>{});
    
    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));
}
