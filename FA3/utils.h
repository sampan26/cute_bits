#pragma once

#include <cmath>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "utils.h"


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};



template<bool init=false, int wg_wait=0, typename TensorA, typename TensorB, typename TensorC, typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma& mma, TensorA const& tCrA, TensorB const& tCrB, TensorC& tCrC) {
    constexpr bool Is_RS = 
        !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    if constexpr(Is_RS) {
        warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
    }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    if constexpr (init) {
        mma.accumulate_ = GMMA::ScaleOut::Zero;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            mma.accumulate_ = GMMA::ScaleOut::One;
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    warpgroup_commit_batch();
    if constexpr(wg_wait >= 0) {
        warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) {
        warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
    }
}

// For SM90, convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V,
// MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = acc_layout;
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                        make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
};

// For SM90, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16,
// MMA_N))
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});  // (2, 2, (2, N / 16)))
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout),
                     make_layout(get<2, 1>(l), get<2>(acc_layout)));
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel,
                                 cutlass::FloatRoundStyle::round_to_nearest>
      convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
