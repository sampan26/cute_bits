#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "static_switch.h"

using namespace cute;




template <int HEAD_DIM_, int BM_, int BN_, int NUM_STAGES_, typename T>
struct AttentionKernelTraits {
    using Element = T;
    using index_t = int64_t;

    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int HEAD_DIM = HEAD_DIM_;
    
    static constexpr int NUM_WARPS = ((BM / 64) + 1) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int NUM_PRODUCER_THREADS = cutlass::NumThreadsPerWarp;
    static constexpr int NUM_STAGES = NUM_STAGES_;

    using TileShape = Shape<Int<BM>, Int<BN>, Int<HEAD_DIM>>;

    using TiledMmaQK = decltype(make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, float, TileShape{}>(),
        Layout<Shape<Int<BM/64>, _1>, _1>{}));

    using TiledMmaPV = decltype(make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, float, decltype(select<0,2,1>(TileShape{})),
                                   GMMA::Major::K, GMMA::Major::MN>(), // we use Vt which is N major
        Layout<Shape<Int<BM/64>, _1>, _1>{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<0>(TileShape{})), decltype(get<2>(TileShape{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<1>(TileShape{})), decltype(get<2>(TileShape{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<1>(TileShape{})), decltype(get<2>(TileShape{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
                                    decltype(get<0>(TileShape{})), decltype(get<2>(TileShape{}))>());
    
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0,2>(TileShape{})));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{}, 
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<NUM_STAGES>{})));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{}, 
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<NUM_STAGES>{})));

    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        make_ordered_layout(make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<NUM_STAGES>{}),
                            Step<_2,_1,_3>);
    
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0,2>(TileShape{})));

    using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, Element>;

    ))
};




template<typename AttentionKernelTraits, typename SharedStorage>
void run_flash_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {

}


template <typename T, int Headdim>
void run_mha_fwd(Flash_fwd_params params, cudaStream_t stream) {
    static_assert(Headdim == 128);
    BOOL_SWITCH(params.is_casaul, Is_causal, [&] {
        using Kernel_traits = AttentionKernelTraits<128, 128, Is_causal ? 128 : 178, Is_causal ? 2 : 1, T>;
        run_flash_mha_fwd<Kernel_traits, SharedStorage<Kernel_traits>>(params, stream);
    });
}
