#ifndef SILVANN__PACKAGES_NN_CONTRACTS_ABI_GPU_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_ABI_GPU_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stddef.h>
#include <stdint.h>

/* ⭐ SEVERAL MATRICES FOR ONE LAUNCH — what the grouped gemv doors take, by value. Addresses are numbers, so
 * the table means the same thing to every family's launch (a kernel argument, an OpenCL argument, a CPU
 * block's struct); `count` says how many rows of the table are used. `out_at` is where a group's rows begin
 * in the door's `out`, in elements; the sum door reads `x` for group `i` at `i · cols` and its weight at
 * `w[out_at[i]]`, so a sum over some of a router's picks names each one's weight without moving the weights. */
#define NN__TURBOQUANT__GROUPS_MAX 12u
typedef struct nn__turboquant__groups {
    uint64_t codes[NN__TURBOQUANT__GROUPS_MAX];      /* each matrix's codes, and its row scales */
    uint64_t luts[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t rows[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t d[NN__TURBOQUANT__GROUPS_MAX];          /* its width: 4, 5, 8 or 16 */
    uint64_t out_at[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t count;
} nn__turboquant__groups;

/* ⭐ SEVERAL EXPERTS, EACH OVER ITS OWN ROWS, FOR ONE LAUNCH — what `expert_groups` takes, by value. Group
 * `i` is a matrix of `out_rows[i]` rows at width `d[i]`, applied to the rows its pairs name: its pairs are
 * `pairs[i]` entries of the door's `rows` list starting at pair `pairs_at[i]`, and its answers go to
 * `out + out_at[i]`, a row of `out_rows[i]` for each out-row a pair names. */
#define NN__EXPERT__GROUPS_MAX 12u
typedef struct nn__expert__groups {
    uint64_t codes[NN__EXPERT__GROUPS_MAX];          /* each matrix's codes, and its row scales */
    uint64_t luts[NN__EXPERT__GROUPS_MAX];
    uint64_t out_rows[NN__EXPERT__GROUPS_MAX];
    uint64_t d[NN__EXPERT__GROUPS_MAX];              /* its width: 4, 5, 8 or 16 */
    uint64_t out_at[NN__EXPERT__GROUPS_MAX];
    uint64_t pairs_at[NN__EXPERT__GROUPS_MAX];
    uint64_t pairs[NN__EXPERT__GROUPS_MAX];
    uint64_t count;
} nn__expert__groups;

/* ⭐⭐ HOW A SILICON LIKES ITS GEMM TILED — ⚖ *"the actual gemm primitive become a silicon capability over the
 * generic code. the point is that we can use methods to get the silicons preferred params for things like
 * gemm tiling size"*. The expert-major bodies (`kernels/kernels.cuh`) are written once against these numbers,
 * and a family changes how they run by answering `nn__silicon__gemm_tile` below.
 *     w_rows   rows of the matrix a block works on at once: each lane decodes one code of each
 *     x_rows   rows of `x` each decoded code is multiplied into before the next is decoded — the reuse that
 *              makes a batch cheaper than its rows one at a time
 * A tile holds `w_rows · x_rows` running sums in each lane, so the two are capped by what a body can keep in
 * registers. */
#define NN__GEMM__W_ROWS_MAX 4u
#define NN__GEMM__X_ROWS_MAX 8u
typedef struct nn__gemm__tile {
    uint32_t w_rows;
    uint32_t x_rows;
} nn__gemm__tile;

/* ══ ⭐⭐⭐ nn's DOORS — THE ONE LIST BOTH SIDES OF A FAMILY AGREE ON ═════════════════════════════════════
 *
 * ⚖ *"doors gathered by macros like the verbs."* One row per door: the package forwarded in, the name the
 * program calls it by, the function that is the door (`nn/gpu/doors.cuh`), and its parameters. From
 * this one list come the table's SHAPE (`nn__doors`, below), each family's FILLING of it (in the
 * family's binary, where those doors are compiled against the family's launch), and the
 * program's CALLS through it — so the three cannot disagree about an order or a signature.
 * ⛳ A ROW IS ONLY EVER APPENDED. The table carries its size, and a program reads no further than the
 *   size it was built with, so a door added at the end is invisible to an older program rather than
 *   misread by it.
 * ⛳ A ROW'S PARAMETERS ARE ITS KERNEL'S, TYPE FOR TYPE, because the door hands each one to the kernel by
 *   address and the kernel reads it at its own width. Written once here, the compiler holds the program's
 *   call, the family's door and the kernel to the same list. Nothing in them names a vendor: `uint16_t*`
 *   is `stdint.h`. ▶ `nn/gpu/doors.cuh` for what each door runs, and the three that answer a value. */
#define nn__CONTRACT__DOORS(X, PKG) \
    X(PKG, void, vector_zero,              nn__vector__zzabi_launch_zero,              (uint16_t* v, uint64_t n)) \
    X(PKG, void, vector_exp,               nn__vector__zzabi_launch_exp,               (uint16_t* o, const uint16_t* x, uint64_t n, unsigned int* over)) \
    X(PKG, void, vector_softplus,          nn__vector__zzabi_launch_softplus,          (uint16_t* o, const uint16_t* x, uint64_t n, unsigned int* over)) \
    X(PKG, void, vector_scale,             nn__vector__zzabi_launch_scale,             (uint16_t* o, const uint16_t* x, uint64_t n, float factor, unsigned int* over)) \
    X(PKG, void, vector_add,               nn__vector__zzabi_launch_add,               (uint16_t* z, const uint16_t* x, const uint16_t* y, uint64_t n, unsigned int* over)) \
    X(PKG, void, vector_pointwise_mul,     nn__vector__zzabi_launch_pointwise_mul,     (uint16_t* z, const uint16_t* x, const uint16_t* y, uint64_t n, unsigned int* over)) \
    X(PKG, void, rmsnorm,                  nn__rmsnorm__zzabi_launch,                  (uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over)) \
    X(PKG, void, vector_l2norm,            nn__vector__zzabi_launch_l2norm,            (uint16_t* o, const uint16_t* x, uint64_t n, unsigned int* over)) \
    X(PKG, void, softmax,                  nn__softmax__zzabi_launch,                  (uint16_t* o, const uint16_t* x, uint64_t n, unsigned int* over)) \
    X(PKG, void, sigmoid,                  nn__sigmoid__zzabi_launch,                  (uint16_t* o, const uint16_t* x, uint64_t n, unsigned int* over)) \
    X(PKG, void, vector_scale_at,          nn__vector__zzabi_launch_scale_at,          (uint16_t* o, const uint16_t* x, const uint16_t* a, uint64_t which, uint64_t n, unsigned int* over)) \
    X(PKG, void, matrix_transpose,         nn__matrix__zzabi_launch_transpose,         (uint16_t* d, const uint16_t* s, uint64_t rows, uint64_t cols)) \
    X(PKG, void, matrix_matvec_transposed, nn__matrix__zzabi_launch_matvec_transposed, (uint16_t* out, const uint16_t* W, const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, argmax,                   nn__argmax__zzabi_launch,                   (uint64_t* out, uint64_t stamp, const uint16_t* x, uint64_t n)) \
    X(PKG, void, vector_dot_product,       nn__vector__zzabi_launch_dot_product,       (uint64_t* out, uint64_t stamp, const uint16_t* a, const uint16_t* b, uint64_t n)) \
    X(PKG, void, vector_at,                nn__vector__zzabi_launch_at,                (uint64_t* out, uint64_t stamp, const uint16_t* x, uint64_t i)) \
    X(PKG, void, swiglu_combine,           nn__swiglu__zzabi_launch_combine,           (uint16_t* o, const uint16_t* g, const uint16_t* u, uint64_t n, unsigned int* over)) \
    X(PKG, void, deltanet_rank_1_update,   nn__deltanet__zzabi_launch_rank_1_update,   (float* S, const uint16_t* k, const uint16_t* d, float g, uint64_t k_dim, uint64_t v_dim)) \
    X(PKG, void, deltanet_readout,         nn__deltanet__zzabi_launch_readout,         (uint16_t* o, const float* S, const uint16_t* x, uint64_t k_dim, uint64_t v_dim, unsigned int* over)) \
    X(PKG, void, deltanet_conv_step,       nn__deltanet__zzabi_launch_conv_step,       (uint16_t* o, uint16_t* s, const uint16_t* w, const uint16_t* x, uint64_t ch, unsigned int* over)) \
    X(PKG, void, expert_multiply_fp16,     nn__expert__zzabi_launch_multiply_fp16,     (uint16_t* out, const uint16_t* W, const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, turboquant_decode,        nn__turboquant__zzabi_launch_decode,        (uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, uint64_t d, uint64_t rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, turboquant_gemv,          nn__turboquant__zzabi_launch_gemv,          (uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, hadamard_rotate,          nn__hadamard__zzabi_launch_rotate,          (uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, unsigned int* over)) \
    X(PKG, void, rope,                     nn__rope__zzabi_launch,                     (uint16_t* o, const uint16_t* x, const float* cs, uint64_t heads, uint64_t head_dim, unsigned int* over)) \
    X(PKG, void, attention_scores,         nn__attention__zzabi_launch_scores,         (float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(PKG, void, attention_mix,            nn__attention__zzabi_launch_mix,            (uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, unsigned int* over)) \
    X(PKG, void, rope_angles,              nn__rope__zzabi_launch_angles,              (float* cs, uint64_t position, float theta, uint64_t head_dim)) \
    X(PKG, void, attention_weights,        nn__attention__zzabi_launch_weights,        (float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(PKG, void, attention_residual_mix,   nn__attention__zzabi_launch_residual_mix,   (float* res, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(PKG, void, attention_merge,          nn__attention__zzabi_launch_merge,          (float* into, const float* a, const float* b, uint64_t q_heads, uint64_t head_dim)) \
    X(PKG, void, attention_finish,         nn__attention__zzabi_launch_finish,         (uint16_t* o, const float* res, uint64_t q_heads, uint64_t head_dim, unsigned int* over)) \
    X(PKG, void, vector_top_k,             nn__vector__zzabi_launch_top_k,             (uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, uint64_t n, uint64_t k)) \
    X(PKG, void, turboquant_gemv_int8,     nn__turboquant__zzabi_launch_gemv_int8,     (uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, deltanet_step,            nn__deltanet__zzabi_launch_step,            (float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t head_dim, unsigned int* over)) \
    X(PKG, void, turboquant_gemv_groups,   nn__turboquant__zzabi_launch_gemv_groups,   (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, unsigned int* over)) \
    X(PKG, void, turboquant_gemv_groups_sum, nn__turboquant__zzabi_launch_gemv_groups_sum, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w, const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, turboquant_gemv_groups_int8, nn__turboquant__zzabi_launch_gemv_groups_int8, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, unsigned int* over)) \
    X(PKG, void, turboquant_gemv_groups_sum_int8, nn__turboquant__zzabi_launch_gemv_groups_sum_int8, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w, const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, swiglu_pairs,             nn__swiglu__zzabi_launch_pairs,             (uint16_t* o, const uint16_t* gu, uint64_t pairs, uint64_t n, unsigned int* over)) \
    X(PKG, void, expert_rows,              nn__expert__zzabi_launch_rows,              (uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, expert_rows_sum,          nn__expert__zzabi_launch_rows_sum,          (float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols)) \
    X(PKG, void, expert_groups,            nn__expert__zzabi_launch_groups,            (uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, expert_rows_finish,       nn__expert__zzabi_launch_rows_finish,       (uint16_t* out, float* acc, const uint16_t* residual, uint64_t n, unsigned int* over)) \
    X(PKG, void, expert_rows_reduce,       nn__expert__zzabi_launch_rows_reduce,       (uint16_t* out, float* acc, const uint16_t* residual, uint64_t n, uint64_t k, uint64_t cols, unsigned int* over)) \
    X(PKG, void, rmsnorm_rows,             nn__rmsnorm__zzabi_launch_rows,             (uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, uint64_t rows, uint64_t x_stride, uint64_t o_stride, float eps, unsigned int* over)) \
    X(PKG, void, rope_rows,                nn__rope__zzabi_launch_rows,                (uint16_t* o, const uint16_t* x, const float* cs, uint64_t rows, uint64_t heads, uint64_t head_stride, uint64_t row_stride, uint64_t rot, unsigned int* over)) \
    X(PKG, void, rope_angles_rows,         nn__rope__zzabi_launch_angles_rows,         (float* cs, uint64_t first, float theta, uint64_t head_dim, uint64_t rows)) \
    X(PKG, void, attention_causal_scores,  nn__attention__zzabi_launch_causal_scores,  (float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows)) \
    X(PKG, void, attention_causal_mix,     nn__attention__zzabi_launch_causal_mix,     (uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows, unsigned int* over)) \
    X(PKG, void, attention_gate_rows,      nn__attention__zzabi_launch_gate_rows,      (uint16_t* o, const uint16_t* att, const uint16_t* qg, uint64_t heads, uint64_t head_dim, unsigned int* over)) \
    X(PKG, void, deltanet_gates_rows,      nn__deltanet__zzabi_launch_gates_rows,      (uint16_t* beta, uint16_t* g, const uint16_t* a, const uint16_t* b, const uint16_t* a_log, const uint16_t* dt_bias, uint64_t heads, uint64_t rows, unsigned int* over)) \
    X(PKG, void, deltanet_conv_steps,      nn__deltanet__zzabi_launch_conv_steps,      (uint16_t* o, uint16_t* s, const uint16_t* w, const uint16_t* x, uint64_t ch, uint64_t rows, unsigned int* over)) \
    X(PKG, void, deltanet_steps,           nn__deltanet__zzabi_launch_steps,           (float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t head_dim, uint64_t rows, unsigned int* over)) \
    X(PKG, void, hadamard_blocks,          nn__hadamard__zzabi_launch_blocks,          (uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, uint64_t block, uint64_t inverse, unsigned int* over)) \
    X(PKG, void, turboquant_encode_rows,   nn__turboquant__zzabi_launch_encode_rows,   (uint8_t* codes, uint16_t* scales, const uint16_t* x, uint64_t rows, uint64_t cols, uint64_t d, unsigned int* over)) \
    X(PKG, void, attention_weights_tq,     nn__attention__zzabi_launch_weights_tq,     (float* p, float* res, const uint16_t* q, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d)) \
    X(PKG, void, attention_residual_mix_tq, nn__attention__zzabi_launch_residual_mix_tq, (float* res, const float* p, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d)) \
    X(PKG, void, hyper_logits,             nn__hyper__zzabi_launch_logits,             (float* lg, const uint16_t* streams, const uint16_t* fn, uint64_t hidden, uint64_t mult, float norm_eps)) \
    X(PKG, void, hyper_pre,                nn__hyper__zzabi_launch_pre,                (uint16_t* out, float* mix, const uint16_t* streams, const uint16_t* base, const uint16_t* scale, uint64_t hidden, uint64_t mult, uint64_t iters, float eps, unsigned int* over)) \
    X(PKG, void, hyper_post,               nn__hyper__zzabi_launch_post,               (uint16_t* out, const uint16_t* streams, const uint16_t* y, const float* mix, uint64_t hidden, uint64_t mult, unsigned int* over)) \
    X(PKG, void, kda_step,                 nn__kda__zzabi_launch_step,                 (float* S, const uint16_t* conved, const uint16_t* f, const uint16_t* b, const uint16_t* dt, const uint16_t* a_log, const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads, uint64_t head_dim, float lower, float eps, unsigned int* over)) \
    X(PKG, void, vector_top_k_biased,      nn__vector__zzabi_launch_top_k_biased,      (uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, const uint16_t* bias, uint64_t n, uint64_t k, float scale)) \
    X(PKG, void, swiglu_clamped,           nn__swiglu__zzabi_launch_clamped,           (uint16_t* act, const uint16_t* gu, uint64_t pairs, uint64_t n, float limit, unsigned int* over)) \
    X(PKG, void, attention_absorb,         nn__attention__zzabi_launch_absorb,         (uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope, uint64_t latent, uint64_t stride, float scale, unsigned int* over)) \
    X(PKG, void, attention_expand,         nn__attention__zzabi_launch_expand,         (uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset, unsigned int* over)) \
    X(PKG, void, index_layernorm,          nn__index__zzabi_launch_layernorm,          (uint16_t* o, const uint16_t* x, const uint16_t* w, const uint16_t* b, uint64_t n, float eps, unsigned int* over)) \
    X(PKG, void, index_pool,               nn__index__zzabi_launch_pool,               (uint16_t* pooled, const uint16_t* keys, const uint16_t* gates, const uint16_t* ape, uint64_t first, uint64_t pools, uint64_t dim, uint64_t kpool, unsigned int* over)) \
    X(PKG, void, index_scores,             nn__index__zzabi_launch_scores,             (float* scores, const uint16_t* q, const uint16_t* w, const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t pools, float scale)) \
    X(PKG, void, index_select,             nn__index__zzabi_launch_select,             (uint32_t* index, uint32_t* count, const float* scores, uint64_t pools, uint64_t k, uint64_t kpool, uint64_t tail_first, uint64_t tail)) \
    X(PKG, void, index_gather,             nn__index__zzabi_launch_gather,             (uint16_t* out, const uint16_t* src, const uint32_t* index, uint64_t n, uint64_t row)) \
    X(PKG, void, vector_copy,              nn__vector__zzabi_launch_copy,              (uint16_t* out, const uint16_t* in, uint64_t n)) \
    X(PKG, void, hyper_logits_rows,        nn__hyper__zzabi_launch_logits_rows,        (float* lg, const uint16_t* streams, const uint16_t* fn, uint64_t hidden, uint64_t mult, float norm_eps, uint64_t rows, uint64_t lg_stride)) \
    X(PKG, void, hyper_pre_rows,           nn__hyper__zzabi_launch_pre_rows,           (uint16_t* out, float* mix, const uint16_t* streams, const uint16_t* base, const uint16_t* scale, uint64_t hidden, uint64_t mult, uint64_t iters, float eps, uint64_t rows, uint64_t mix_stride, unsigned int* over)) \
    X(PKG, void, hyper_post_rows,          nn__hyper__zzabi_launch_post_rows,          (uint16_t* out, const uint16_t* streams, const uint16_t* y, const float* mix, uint64_t hidden, uint64_t mult, uint64_t rows, uint64_t mix_stride, unsigned int* over)) \
    X(PKG, void, kda_steps,                nn__kda__zzabi_launch_steps,                (float* S, const uint16_t* conved, const uint16_t* f, const uint16_t* b, const uint16_t* dt, const uint16_t* a_log, const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads, uint64_t head_dim, float lower, float eps, uint64_t rows, unsigned int* over)) \
    X(PKG, void, expert_groups_int8,       nn__expert__zzabi_launch_groups_int8,       (uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows, uint64_t cols, unsigned int* over)) \
    X(PKG, void, expert_rows_sum_int8,     nn__expert__zzabi_launch_rows_sum_int8,     (float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols)) \
    X(PKG, void, rank1_dots,               nn__rank1__zzabi_launch_dots,               (float* s, const uint16_t* v, const uint32_t* which, const uint16_t* x, const uint16_t* w, const uint32_t* w_at, uint64_t n, uint64_t cols, uint64_t x_stride, uint64_t w_stride)) \
    X(PKG, void, rank1_add,                nn__rank1__zzabi_launch_add,                (uint16_t* out, const uint16_t* r, const float* s, uint64_t n, uint64_t per, uint64_t rows, uint64_t out_stride, unsigned int* over)) \
    X(PKG, void, rank1_spread,             nn__rank1__zzabi_launch_spread,             (float* acc, const uint16_t* r, const float* s, const uint32_t* row_of, uint64_t n, uint64_t rows, uint64_t row_stride)) \
    X(PKG, void, attention_scores_grouped, nn__attention__zzabi_launch_scores_grouped, (float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(PKG, void, attention_softmax_rows,   nn__attention__zzabi_launch_softmax_rows,   (float* p, uint64_t rows, uint64_t length)) \
    X(PKG, void, attention_causal_scores_grouped, nn__attention__zzabi_launch_causal_scores_grouped, (float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows)) \
    X(PKG, void, attention_causal_softmax, nn__attention__zzabi_launch_causal_softmax, (float* p, uint64_t q_heads, uint64_t first, uint64_t rows)) \
    X(PKG, void, vector_penalize,          nn__vector__zzabi_launch_penalize,          (uint16_t* x, uint64_t n, const uint64_t* ids, uint64_t stride, uint64_t count, float penalty, unsigned int* over)) \
    X(PKG, void, vector_draw,              nn__vector__zzabi_launch_draw,              (uint64_t* out, uint64_t stamp, const uint16_t* w, uint64_t k, float top_p, uint64_t seed, uint64_t pos))

/* ⛳ THE TABLE AND THE LAUNCH ARE THE HOST'S SIDE OF A FAMILY, so a family whose kernels are compiled as
 * OpenCL C does not see them there: that language has no function pointers, and `kernel` is a keyword. */
#ifndef __OPENCL_C_VERSION__
#define NN__DOORS__ZZPRIVATE_FIELD(PKG, RET, NAME, FN, PARAMS)  RET (*NAME) PARAMS;
typedef struct nn__doors {
    uint32_t size;                                   /* sizeof this table as the family built it */
    nn__CONTRACT__DOORS(NN__DOORS__ZZPRIVATE_FIELD, nn)
} nn__doors;
#undef NN__DOORS__ZZPRIVATE_FIELD
#endif

/* ══ ⭐⭐ AND THE ARITHMETIC nn's KERNELS ASK FOR — EVERY FAMILY ANSWERS IT, THE HOST'S INCLUDED ═══════════
 *
 * ⚖ *"primitives with the package"* — the functions a compute body calls that a family must provide, declared
 * by the package whose bodies call them, and the launch its doors call. Each family defines them beside its doors (`silicon_families/
 * <family>/nn.cuh`), `__device__` on a card and a plain function on the host, where the same bodies are
 * compiled into the program.
 *
 * ⭐⭐ THEY WERE THE TREE'S FIRST FLOATING POINT. A node is six `uint64_t` and the language computes with
 * integers; nothing in it was arithmetic. ⇒ ★ A LANGUAGE CAN BE COMPLETE AND STILL NOT BE ABLE TO ADD TWO
 * NUMBERS THE WAY A MODEL MEANS IT.
 * ⛳ ONLY WHAT DIFFERS BETWEEN SILICON, AND THE REST IS WRITTEN IN C. A clamp is two comparisons and a
 * multiply is a multiply — neither differs between families. What does is the transcendental, so that is
 * what is asked for. ⛔ ADDING MORE "FOR SYMMETRY" WOULD BE VOCABULARY EVERY FAMILY MUST KEEP IDENTICAL FOR
 * NO REASON, which is the cost the conformance rule makes visible.
 *
 *     expf    ⛔ THE PRECISE ONE BY DEFAULT, NOT `__expf`, a deliberate divergence from the old tree, whose
 *             SwiGLU called the hardware approximation. Matching it would make this engine agree with the old one
 *             and with nothing else. ⇒ ★ AN ORACLE IS ONLY WORTH RUNNING AGAINST A REFERENCE SOMEBODY ELSE
 *             COMPUTED, so the arithmetic is the one numpy and PyTorch can be asked about. The fast path is
 *             a dial, `NN__ARITHMETIC__FAST` in `contracts/defaults.cuh`, which turns `expf` and `logf` to
 *             the card's fast ones — off until the measurement it names turns it.
 *     sqrtf   for rmsnorm. ⛔ NOT `rsqrtf`, though rmsnorm wants the reciprocal — the same ruling, one
 *             function later. `1.0f / sqrtf(v)` is two operations where `rsqrtf` is one; that is a
 *             performance difference, unmeasured, so not yet a reason.
 *     logf    for softplus, which spends it as `log(1 + y)` with `y = exp(-|x|)` in `(0, 1]`. ⛳ NOT
 *             `log1pf`: the two differ only for `y` below fp32's `2^-24`, where this answers `0` — an
 *             absolute error under `6e-8` against an output stored in HALF, whose spacing near 1 is
 *             `1e-3`. ⇒ ★ A PRECISION ARGUMENT IS ONLY WORTH MAKING AGAINST THE WIDTH THE ANSWER IS
 *             STORED IN.
 *     cosf    for RoPE's angles — ⚖ *"yes add cos and sin"*. ⛔ ALWAYS THE PRECISE ONES, AND THE FAST DIAL
 *     sinf    DOES NOT REACH THEM: an angle is a position times a frequency, so at position 100,000 it is
 *             100,000 radians, and a fast sine reduces that range badly enough to be wrong in its first
 *             digit. An `expf` is fast-able because its argument is small; these two cannot promise that.
 * ⚠ A HOST AND A CARD ARE NOT BIT-IDENTICAL HERE: a card's `expf` is its own, not libm's, so an oracle
 *   comparing softmax or rmsnorm across the two needs a tolerance. */
/* ⭐ AND THE ONE CALL THAT STARTS A KERNEL — ⚖ *"the wrapper over hiplaunchkernelGGL"*. Start a kernel on
 * `blocks` blocks of `threads` threads, `args` holding a pointer to each of its `count` arguments in order
 * and `sizes` each one's width; it returns once the launch is queued. A host function, and only a card
 * family answers it: the host runs no kernel. ▶ `nn/gpu/doors.cuh`, the one place it is called.
 * ⛳ THE KERNEL IS GIVEN TWO WAYS, AND A FAMILY TAKES THE ONE ITS RUNTIME USES: `kernel`, its address, for
 *   a runtime that launches by pointer (HIP, CUDA); `name`, for one that finds kernels by name in a
 *   program it built (OpenCL) — where `kernel` is null, because no such function was compiled. */
#ifndef __OPENCL_C_VERSION__
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks, uint32_t threads,
                                       void** args, const size_t* sizes, uint32_t count);
#endif

static __device__ inline float nn__silicon__expf(float x);
static __device__ inline float nn__silicon__sqrtf(float x);
static __device__ inline float nn__silicon__logf(float x);
static __device__ inline float nn__silicon__cosf(float x);
static __device__ inline float nn__silicon__sinf(float x);

/* ⭐ WHICH LANE OF ITS BLOCK A BODY IS RUNNING ON, AND HOW MANY LANES THE BLOCK HAS — ⚖ *"go with
 * nn__silicon__lane and lanes"*. A body takes its share of a loop as `i = lane; i < n; i += lanes`, and it
 * asks these rather than reading `threadIdx` and `blockDim`, which are HIP's and CUDA's spelling and not
 * OpenCL's (`get_local_id`, `get_local_size`) — the same reason the evaluator asks `sys__silicon__block_id`.
 * On the host, which compiles the same bodies as the reference a card is checked against, a body is one
 * lane of one. */
static __device__ inline uint32_t nn__silicon__lane(void);
static __device__ inline uint32_t nn__silicon__lanes(void);

/* ⭐ AND WHICH BLOCK, OF HOW MANY — ⚖ ruled. A body spread over several blocks takes its share
 * of the OUTER loop as `r = block; r < rows; r += blocks` and its share of the inner one by lane, so the
 * same body is right at any width, including the host's one lane of one block. */
static __device__ inline uint32_t nn__silicon__block(void);
static __device__ inline uint32_t nn__silicon__blocks(void);

/* ⭐⭐ THE COMBINE STEP — every lane of the block hands in its partial and every lane gets back the
 * block's sum (or largest). ⚖ ruled, and for the reason that decides the matvec: one block
 * per row, lanes along the row, so the weights are read once and in order while the activation, which
 * every block reads, stays where it is — and the row's answer is then the sum of the lanes' parts.
 * ⛔ EVERY LANE OF THE BLOCK MUST CALL IT, AND AT THE SAME POINT: it waits for all of them inside, so a
 * call under a branch some lanes skip waits forever. The wait is inside these two so that no body ever
 * writes one. ⚠ The order the parts are added in is the family's, not the loop's, so the answer can
 * differ from a serial sum in its last bits. */
static __device__ inline float nn__silicon__lanes_sum(float part);
static __device__ inline float nn__silicon__lanes_max(float part);
/* ⭐ AND SEVERAL SUMS AT ONCE: `parts[0..n)` of every lane replaced by the block's sums, each folded in the order
 * `lanes_sum` folds it, so each answer is the bits one call of it gives — for a body that ends a step with a
 * tile of sums, where a call each would wait for the block once a sum. The same rule: every lane calls it. */
static __device__ inline void nn__silicon__lanes_sums(float* parts, uint32_t n);

/* ⭐⭐ A HALF, AS THE FLOAT IT IS — ⚖ ruled. Every half is exactly a float, so every correct
 * conversion gives the same bits and nothing about the ANSWER differs between families; what differs is
 * the cost. (⛳ One exception, and not a value: a signalling NaN comes back QUIET from a card's instruction —
 * still a NaN. `MEASURED` over all 65,536 halves, `test/src_nn_fold_on_device.cu`.) `MEASURED` on gfx906 at the 7B's gate/up shape: the fp16 matvec ran at 188 GB/s converting in
 * software (a branch per element, and a loop for a subnormal, so the lanes of a wave diverge) and at 570
 * GB/s with the card's own instruction. ⛔ ONLY THIS DIRECTION: float -> half is one conversion per
 * output, not per weight, and it is where overflow is noticed, so it stays nn's own. */
static __device__ inline float nn__silicon__half_to_float(uint16_t h);

/* ⭐ TWO HALVES BY TWO HALVES, ONTO A FLOAT: `c + a.lo·b.lo + a.hi·b.hi`, each word two halves with the low one first.
 * Every product of two halves is exact in a float, so what differs between families is the order of the two additions:
 * one lane's families keep `(c + lo) + hi`, the order a sum over the columns has, and a card's instruction may not. */
static __device__ inline float nn__silicon__dot2(uint32_t a, uint32_t b, float c);

/* ⭐⭐ AND THE TILE IT LIKES ITS GEMM IN — the numbers above `nn__gemm__tile`, answered as constants, so a body
 * that asks is compiled with them folded in and its tile loops unrolled. Each must be at least 1 and at most
 * its `_MAX`. */
static __device__ inline nn__gemm__tile nn__silicon__gemm_tile(void);

/* ⚖ *"cap at 256"* — the most lanes a door may start in one block. It is the narrowest family's limit:
 * `MEASURED` with `clinfo` on the MI50, OpenCL reports a largest work-group of 256 where HIP
 * allows 1,024, and a door past it would not launch there. Every door's width is checked against it when
 * the doors are compiled (`gpu/doors.cuh`). */
#define NN__SILICON__LANES_MAX  256u

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_ABI_GPU_CUH */
