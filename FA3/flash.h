#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params {
    using index_t = int64_t;

    int b, seqlen, h, d;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;
};

template<typename T, int Headdim>
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream);