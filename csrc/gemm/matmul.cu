#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>

using namespace cute;

void run_fp8_gemm(void *A, void *B, void *C,  int m, int n, int k, cudaStream_t stream){
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

    
}
