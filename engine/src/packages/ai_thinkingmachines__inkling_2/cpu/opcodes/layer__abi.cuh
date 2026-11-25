#ifndef SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_LAYER__ABI_CUH
#define SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_LAYER__ABI_CUH
/* ══ INKLING 2's LAYER, IN TWO VERBS AROUND ITS EXPERTS ════════════════════════════════════════════════════════════════════
 * ⚖ 2026-11-24, the architect: *"the threshold that makes a breakage earn its keep is if we do have a parameter to pass
 * that the evaluator can change during the execution"*. A layer's application hands the evaluator one such thing: where
 * its experts run — this worker, another, a file behind them. So an application is two verbs, and the experts' between
 * them (▶ `experts__abi.cuh`):
 *   `before_experts`  the stream to the router's picks: what the attention reads (a learned query's mix over the block
 *                     states, the messages of the pass before through their gates), the attention with its bias a
 *                     distance and its sink, the short convolutions, the experts' own mix, the router — the hand
 *                     filled and the picks chosen;
 *   `after_experts`   the experts' sum through its short convolution, back into the stream.
 * Every step is one of nn's doors, launched from here in the order the model's forward has them; the arithmetic is the
 * doors' — a short convolution one launch, a message gate's end one, a value's heads normed as rows of one. ▶ `contracts/objects/layer.cuh` for the three tables, lisp/inkling2.lisp for the program around the verbs. */

#include "../../../nn/cpu/doors.cuh"            /* nn's doors, a buffer's room, a plane table */
#include "../../../nn/cpu/routed__header.cuh"   /* the router's top-k, as nn writes a position's picks */

/* What a verb holds while it runs: the three tables read, the shape, and where each part of the scratch begins. */
typedef struct ai_thinkingmachines__inkling_2__layer__site {
    const nn__doors* doors;
    unsigned int* over;
    uint64_t layer[AI_THINKINGMACHINES__INKLING_2__LAYER__TABLE], layer_room[AI_THINKINGMACHINES__INKLING_2__LAYER__TABLE];
    uint64_t layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__TABLE];
    uint64_t own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__TABLE], own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__TABLE];
    uint64_t work[AI_THINKINGMACHINES__INKLING_2__WORK__TABLE], work_room[AI_THINKINGMACHINES__INKLING_2__WORK__TABLE];
    uint64_t work_v[AI_THINKINGMACHINES__INKLING_2__WORK__TABLE];
    float work_r[AI_THINKINGMACHINES__INKLING_2__WORK__TABLE];
    uint64_t hidden, query_heads, key_value_heads, head_width, query_width, key_value_width, bias_rank, bias_width;
    uint64_t gate_rank, experts, top_k, taps;
    float epsilon;
    /* the scratch's parts, each where it begins */
    uint64_t gated_inputs, query_input, key_input, value_input, query_input_rotated, key_input_rotated, value_input_rotated;
    uint64_t gate_pair, gate_pair_rotated, gate_rank_values, gate_rank_rotated, forget, gain;
    uint64_t query, query_clipped, key, value, key_convolved, value_convolved, made_taps;
    uint64_t bias_state, bias_state_clipped, bias_query, sink;
    uint64_t attended, output_gate, attended_gated, attended_gated_rotated;
    uint64_t experts_input, experts_normed, router_logits, routed_convolved;
} ai_thinkingmachines__inkling_2__layer__site;

/* `halves` of the scratch from `*next`, which moves past them. */
static inline uint64_t ai_thinkingmachines__inkling_2__layer__zzprivate_take(uint64_t* next, uint64_t halves) {
    const uint64_t at = *next;
    *next += 2ull * halves;
    return at;
}

/* The three tables into `s`, the shape checked, the scratch laid out and every shared buffer's room asked for. */
static uint64_t ai_thinkingmachines__inkling_2__layer__zzprivate_read(const sys__heap_node* argv, sys__engine__ctx* ctx,
                                                                      ai_thinkingmachines__inkling_2__layer__site* s) {
    for (unsigned a = 0u; a < 3u; ++a)
        if (!sys__heap_node__carries_reference(argv[a].dtype) || !sys__node_array__is(argv[a].args[0])) return SYS__OPCODES__FAULT_TYPE;
    float none[AI_THINKINGMACHINES__INKLING_2__LAYER__TABLE];
    if (!nn__primitives__table(argv[0].args[0], AI_THINKINGMACHINES__INKLING_2__LAYER__PLANES, AI_THINKINGMACHINES__INKLING_2__LAYER__TABLE,
                               AI_THINKINGMACHINES__INKLING_2__LAYER__TABLE, s->layer, s->layer_room, s->layer_v, none)
     || !nn__primitives__table(argv[1].args[0], AI_THINKINGMACHINES__INKLING_2__APPLICATION__TABLE, AI_THINKINGMACHINES__INKLING_2__APPLICATION__TABLE,
                               AI_THINKINGMACHINES__INKLING_2__APPLICATION__TABLE, s->own, s->own_room, s->layer_v, none)
     || !nn__primitives__table(argv[2].args[0], AI_THINKINGMACHINES__INKLING_2__WORK__PLANES, AI_THINKINGMACHINES__INKLING_2__WORK__REALS,
                               AI_THINKINGMACHINES__INKLING_2__WORK__TABLE, s->work, s->work_room, s->work_v, s->work_r))
        return AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE;
    const uint64_t H = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__HIDDEN], QH = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__QUERY_HEADS];
    const uint64_t KVH = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__KEY_VALUE_HEADS], HW = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__HEAD_WIDTH];
    const uint64_t BR = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_RANK], GR = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__GATE_RANK];
    const uint64_t E = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__EXPERTS], K = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__TOP_K];
    const uint64_t T = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__TAPS];
    const uint64_t most = 1ull << 20;
    if (H == 0u || H > most || QH == 0u || QH > 4096u || KVH == 0u || KVH > QH || QH % KVH != 0u || HW == 0u || HW > most
     || BR == 0u || BR > most || GR == 0u || GR > most || E == 0u || E > most || K == 0u || K > E
     || K > AI_THINKINGMACHINES__INKLING_2__HAND__WEIGHTS || T < 2u || T > 16u
     || s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_ROW] > 64u
     || s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_ATTENDED] + 1u > NN__ATTENTION__STATES_MAX
     || s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_MIXED] + 1u > NN__ATTENTION__STATES_MAX)
        return AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE;
    s->hidden = H; s->query_heads = QH; s->key_value_heads = KVH; s->head_width = HW;
    s->query_width = QH * HW; s->key_value_width = KVH * HW; s->bias_rank = BR; s->bias_width = QH * BR;
    s->gate_rank = GR; s->experts = E; s->top_k = K; s->taps = T;
    s->epsilon = s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__EPSILON];
    const uint64_t QW = s->query_width, KVW = s->key_value_width, RW = s->bias_width;
    uint64_t next = s->work[AI_THINKINGMACHINES__INKLING_2__WORK__SCRATCH];
#define AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(name, halves) s->name = ai_thinkingmachines__inkling_2__layer__zzprivate_take(&next, halves)
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(gated_inputs, 3u * H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(query_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(key_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(value_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(query_input_rotated, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(key_input_rotated, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(value_input_rotated, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(gate_pair, 2u * H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(gate_pair_rotated, 2u * H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(gate_rank_values, GR);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(gate_rank_rotated, GR);
    /* a gate's forget and gain, and the products of the query's input, each in its matrices' order (▶ `times_joined`) */
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(forget, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(gain, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(query, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(bias_state, RW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(sink, QH);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(output_gate, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(key, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(value, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(query_clipped, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(key_convolved, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(value_convolved, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(made_taps, KVH * T);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(bias_state_clipped, RW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(bias_query, RW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(attended, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(attended_gated, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(attended_gated_rotated, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(experts_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(experts_normed, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(router_logits, E);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART(routed_convolved, H);
#undef AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_PART
    const uint64_t* wr = s->work_room;
    const uint64_t* lr = s->layer_room;
    const uint64_t rows = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_MIXED] + 1u;
    if (next - s->work[AI_THINKINGMACHINES__INKLING_2__WORK__SCRATCH] > wr[AI_THINKINGMACHINES__INKLING_2__WORK__SCRATCH]
     || !nn__primitives__fits(H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__STREAM])
     || !nn__primitives__fits(rows * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__STATES])
     || !nn__primitives__fits((s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_ATTENDED] + 1u) * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__STATES])
     || !nn__primitives__fits((s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_ROW] + 1u) * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__STATES])
     || !nn__primitives__fits(2u * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__PASS_STATES])
     || !nn__primitives__fits(4u * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__MESSAGES])
     || !nn__primitives__fits(AI_THINKINGMACHINES__INKLING_2__HAND__HALVES(H), wr[AI_THINKINGMACHINES__INKLING_2__WORK__HAND])
     || !nn__primitives__fits(H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__RESIDUAL])
     || !nn__primitives__fits(3u * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__INPUTS])
     || !nn__primitives__fits(H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__ATTENTION_ANSWER])
     || !nn__primitives__fits(H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__ATTENTION_CONVOLVED])
     || !nn__primitives__fits(H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__EXPERTS_MIX])
     || !nn__primitives__fits(2u * 4u * (rows > 2u ? rows : 2u), wr[AI_THINKINGMACHINES__INKLING_2__WORK__MIX_SCORES])
     || !nn__primitives__fits(2u * 4u * (s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_ATTENDED] + 1u), wr[AI_THINKINGMACHINES__INKLING_2__WORK__MIX_SCORES])
     || !nn__primitives__fits(E, lr[AI_THINKINGMACHINES__INKLING_2__LAYER__ROUTER_BIAS])
     || !nn__primitives__fits(512u, wr[AI_THINKINGMACHINES__INKLING_2__WORK__SIGNS])
     || !nn__primitives__fits(H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__ONES])
     || !nn__primitives__fits(3u * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEY_HISTORY])
     || !nn__primitives__fits(3u * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUE_HISTORY])
     || !nn__primitives__fits(3u * H, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__ATTENTION_HISTORY])
     || !nn__primitives__fits(3u * H, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__EXPERTS_HISTORY])
     || T != 4u
     || !nn__primitives__fits(T * KVW, lr[AI_THINKINGMACHINES__INKLING_2__LAYER__KEY_TAPS])
     || !nn__primitives__fits(T * KVW, lr[AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE_TAPS])
     || !nn__primitives__fits(T * H, lr[AI_THINKINGMACHINES__INKLING_2__LAYER__ATTENTION_TAPS])
     || !nn__primitives__fits(T * H, lr[AI_THINKINGMACHINES__INKLING_2__LAYER__EXPERTS_TAPS]))
        return AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE;
    s->doors = nn__doors_for(ctx);
    s->over = ctx->fault_word;
    return s->doors == 0 ? NN__PRIMITIVES__FAULT_NO_DEVICE : 0ull;
}

/* Matrix `m` of the layer's table times a vector: the rotated vector where the matrix is quantised, the plain one
 * where it is kept exact. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_times(const ai_thinkingmachines__inkling_2__layer__site* s, unsigned m,
                                                                   uint64_t plain, uint64_t rotated, uint64_t out, uint64_t rows,
                                                                   uint64_t cols) {
    const uint64_t d = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BITS + m];
    s->doors->turboquant_gemv((uint16_t*)(uintptr_t)(out), (const uint8_t*)(uintptr_t)s->layer[2u * m],
                              s->layer_room[2u * m], (const uint8_t*)(uintptr_t)s->layer[2u * m + 1u], s->layer_room[2u * m + 1u],
                              (const uint16_t*)(uintptr_t)(d == 16ull ? plain : rotated), d, rows, cols, s->over);
}

/* Matrices `m[0 … count)` of the layer's table, `rows[i]` rows each, times one vector, their answers one after another
 * from `out`. Where a matrix's codes begin where the one before's end, and its scales where that one's end, at the same
 * width — as the load lays the matrices of one input — the two are one product: each row is the same sum, so the
 * answers are the separate products' to the bit, in fewer launches. A run at 16 bits stays a product a matrix: its
 * scale is one for the whole matrix. */
/* Whether matrix `next` lies right after a run from matrix `first` of `codes` and `scales` bytes, at its width — not 16. */
static bool ai_thinkingmachines__inkling_2__layer__zzprivate_abuts(const ai_thinkingmachines__inkling_2__layer__site* s, unsigned first,
                                                                   uint64_t codes, uint64_t scales, unsigned next) {
    const uint64_t d = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BITS + first];
    return d != 16ull && s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BITS + next] == d
        && s->layer[2u * next] == s->layer[2u * first] + codes && s->layer[2u * next + 1u] == s->layer[2u * first + 1u] + scales;
}
static void ai_thinkingmachines__inkling_2__layer__zzprivate_times_joined(const ai_thinkingmachines__inkling_2__layer__site* s,
                                                                          const unsigned* m, const uint64_t* rows, unsigned count,
                                                                          uint64_t plain, uint64_t rotated, uint64_t out, uint64_t cols) {
    for (unsigned i = 0u; i < count;) {
        const uint64_t d = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BITS + m[i]];
        uint64_t codes = s->layer_room[2u * m[i]], scales = s->layer_room[2u * m[i] + 1u], run = rows[i];
        unsigned j = i + 1u;
        for (; j < count && ai_thinkingmachines__inkling_2__layer__zzprivate_abuts(s, m[i], codes, scales, m[j]); ++j) {
            codes += s->layer_room[2u * m[j]];
            scales += s->layer_room[2u * m[j] + 1u];
            run += rows[j];
        }
        s->doors->turboquant_gemv((uint16_t*)(uintptr_t)(out), (const uint8_t*)(uintptr_t)s->layer[2u * m[i]], codes,
                                  (const uint8_t*)(uintptr_t)s->layer[2u * m[i] + 1u], scales,
                                  (const uint16_t*)(uintptr_t)(d == 16ull ? plain : rotated), d, run, cols, s->over);
        out += 2u * run;
        i = j;
    }
}

#define AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(name) s->work[AI_THINKINGMACHINES__INKLING_2__WORK__##name]
#define AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(name) s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__##name]

/* A learned query's mix, at one position: `how_many` queries of the model's width over the states' first `blocks` rows
 * — the block states, normed when they were made — and `current`, normed: a query's scores over √hidden, their
 * softmax, the states weighed by it. An attention of one head as wide as the model, one launch (nn's
 * `attention_states`). */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_mix(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t queries, uint64_t how_many, uint64_t blocks, uint64_t current, uint64_t into) {
    s->doors->attention_states((uint16_t*)(uintptr_t)(into), (const uint16_t*)(uintptr_t)(queries),
                               (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STATES)), (const uint16_t*)(uintptr_t)(current),
                               how_many, s->hidden, blocks, 1u, s->epsilon, s->over);
}

/* A short convolution: this position and the three before it, a tap a lag, the input added back, over √(taps + 1) —
 * nn's `convolution_step`, which also slides the history: this input in, the oldest out. `taps` a row a lag, this
 * position's first; `history` the three inputs before this one, the newest first. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_convolve(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t input, uint64_t history, uint64_t taps, uint64_t width, uint64_t out) {
    s->doors->convolution_step((uint16_t*)(uintptr_t)(out), (uint16_t*)(uintptr_t)(history), (const uint16_t*)(uintptr_t)(taps),
                               (const uint16_t*)(uintptr_t)(taps), (const uint16_t*)(uintptr_t)(input), width, s->taps, 0u,
                               s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__CONVOLUTION_SCALE], s->over);
}

/* …and its output added to `add` into `sum`, in the same launch (nn's `convolution_step_add`): the convolution kept in
 * `out`, the stream it joins made. With `add` zero `sum` is the convolution itself. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_add(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t input, uint64_t history,
                                                                          uint64_t taps, uint64_t width, uint64_t out, uint64_t add, uint64_t sum) {
    s->doors->convolution_step_add((uint16_t*)(uintptr_t)(out), (uint16_t*)(uintptr_t)(sum), (uint16_t*)(uintptr_t)(history),
                                   (const uint16_t*)(uintptr_t)(taps), (const uint16_t*)(uintptr_t)(taps), (const uint16_t*)(uintptr_t)(input),
                                   (const uint16_t*)(uintptr_t)(add), width, s->taps, 0u,
                                   s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__CONVOLUTION_SCALE], s->over);
}

/* A full layer's key or value: a head's taps are also made from the position's own state — matrix `made` over `state`
 * gives a head its four, head g's lag the (taps·g + lag)-th — and join the layer's own; the sum is over √(2·taps + 1). */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_made(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t input, uint64_t history, uint64_t taps, unsigned made,
                                                                           uint64_t state, uint64_t out) {
    ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, made, state, state, s->made_taps, s->key_value_heads * s->taps, s->hidden);
    s->doors->convolution_step((uint16_t*)(uintptr_t)(out), (uint16_t*)(uintptr_t)(history), (const uint16_t*)(uintptr_t)(taps),
                               (const uint16_t*)(uintptr_t)(s->made_taps), (const uint16_t*)(uintptr_t)(input), s->key_value_width,
                               s->taps, s->head_width, s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__MADE_SCALE], s->over);
}

/* A loop layer's message gate, in its second and third pass: the input and the message, each normed, through the
 * controller's map of the gate's rank — matrices `controller`, +1 and +2: down, forget, gain — then
 * (1 − tanh(forget))·input + tanh(gain)·message. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_gate(const ai_thinkingmachines__inkling_2__layer__site* s, unsigned controller,
                                                                  uint64_t input, uint64_t message, uint64_t out) {
    const nn__doors* doors = s->doors;
    const uint64_t H = s->hidden, GR = s->gate_rank;
    const uint64_t ones = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ONES), signs = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(SIGNS);
    /* the two norms and the pair's rotation, nn's one door: a family's one launch, or the three doors' */
    doors->rmsnorm_pair_rotate((uint16_t*)(uintptr_t)(s->gate_pair_rotated), (uint16_t*)(uintptr_t)(s->gate_pair),
                               (const uint16_t*)(uintptr_t)(input), (const uint16_t*)(uintptr_t)(message),
                               (const uint16_t*)(uintptr_t)(ones), (const uint16_t*)(uintptr_t)(signs), H, s->epsilon, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, controller, s->gate_pair, s->gate_pair_rotated, s->gate_rank_values, GR, 2u * H);
    /* the rank rotated, forget's and gain's products over it and the gate: nn's one door where the load laid forget and
     * gain one after the other — a family's one launch, or the three doors' */
    const unsigned forget = controller + 1u, gain = controller + 2u;
    const uint64_t codes = s->layer_room[2u * forget], scales = s->layer_room[2u * forget + 1u];
    if (ai_thinkingmachines__inkling_2__layer__zzprivate_abuts(s, forget, codes, scales, gain)) {
        doors->vector_gate_tanh_rank((uint16_t*)(uintptr_t)(out), (uint16_t*)(uintptr_t)(s->gate_rank_rotated), (uint16_t*)(uintptr_t)(s->forget),
                                     (const uint16_t*)(uintptr_t)(input), (const uint16_t*)(uintptr_t)(message),
                                     (const uint16_t*)(uintptr_t)(s->gate_rank_values), (const uint16_t*)(uintptr_t)(signs),
                                     (const uint8_t*)(uintptr_t)s->layer[2u * forget], codes + s->layer_room[2u * gain],
                                     (const uint8_t*)(uintptr_t)s->layer[2u * forget + 1u], scales + s->layer_room[2u * gain + 1u],
                                     s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BITS + forget], H, GR, s->over);
        return;
    }
    doors->hadamard_blocks((uint16_t*)(uintptr_t)(s->gate_rank_rotated), (const uint16_t*)(uintptr_t)(s->gate_rank_values),
                           (const uint16_t*)(uintptr_t)(signs), GR, GR, 0u, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, forget, s->gate_rank_values, s->gate_rank_rotated, s->forget, H, GR);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, gain, s->gate_rank_values, s->gate_rank_rotated, s->gain, H, GR);
    doors->vector_gate_tanh((uint16_t*)(uintptr_t)(out), (const uint16_t*)(uintptr_t)(input), (const uint16_t*)(uintptr_t)(message),
                            (const uint16_t*)(uintptr_t)(s->forget), (const uint16_t*)(uintptr_t)(s->gain), H, s->over);
}

/* What the attention reads — the query's, the key's and the value's input, each normed and rotated. The first layer
 * reads the stream; another its query's mix over the block states and the stream, a full layer three queries' (a
 * sliding layer's one mix serves the three). With messages each goes through its gate first. An input that is the
 * query's own — the same vector under the same norm — is read where the query's is, not made again. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_inputs(ai_thinkingmachines__inkling_2__layer__site* s, bool messages) {
    const nn__doors* doors = s->doors;
    const uint64_t H = s->hidden, stream = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STREAM);
    const uint64_t inputs = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(INPUTS), told = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(MESSAGES);
    const uint64_t signs = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(SIGNS);
    const bool full = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FULL] == 1u;
    uint64_t from[3] = { stream, stream, stream };
    if (s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FIRST] != 1u) {
        ai_thinkingmachines__inkling_2__layer__zzprivate_mix(s, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(ATTENTION_QUERY), full ? 3u : 1u, s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_ATTENDED], stream, inputs);
        for (unsigned i = 0u; i < 3u; ++i) from[i] = full ? inputs + 2u * i * H : inputs;
    }
    if (messages)
        for (unsigned i = 0u; i < 3u; ++i) {
            ai_thinkingmachines__inkling_2__layer__zzprivate_gate(s, AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLERS + i * AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLER, from[i], told + 2u * i * H, s->gated_inputs + 2u * i * H);
            from[i] = s->gated_inputs + 2u * i * H;
        }
    uint64_t* normed[3] = { &s->query_input, &s->key_input, &s->value_input };
    uint64_t* rotated[3] = { &s->query_input_rotated, &s->key_input_rotated, &s->value_input_rotated };
    for (unsigned i = 0u; i < 3u; ++i) {
        if (i != 0u && from[i] == from[0] && s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_NORM + i] == s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_NORM]) {
            *normed[i] = *normed[0];
            *rotated[i] = *rotated[0];
            continue;
        }
        /* the norm and its rotation, nn's one door: a family's one launch, or the two doors' */
        doors->rmsnorm_rotate((uint16_t*)(uintptr_t)(*rotated[i]), (uint16_t*)(uintptr_t)(*normed[i]), (const uint16_t*)(uintptr_t)(from[i]),
                              (const uint16_t*)(uintptr_t)(s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_NORM + i]),
                              (const uint16_t*)(uintptr_t)(signs), H, s->epsilon, s->over);
    }
}

/* The attention, into the work's answer. The query scaled a head and clipped to a length; the key and the value through
 * their short convolutions — the key then clipped, the value normed a head — into the application's cache at `row`; the
 * position's bias, a head's values (scaled, clipped, times √rank since nn's scores divide by it) against the table laid
 * a row a distance from `furthest`; the scores over the cache's `rows`, the bias added, the sink in the softmax, the
 * values mixed; the answer gated by a sigmoid of its own projection, then the output's. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_attention(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t pos,
                                                                       uint64_t row, uint64_t rows, uint64_t furthest) {
    const nn__doors* doors = s->doors;
    const uint64_t H = s->hidden, QH = s->query_heads, KVH = s->key_value_heads, HW = s->head_width;
    const uint64_t QW = s->query_width, KVW = s->key_value_width, BR = s->bias_rank, RW = s->bias_width;
    const uint64_t keys = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS], values = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES];
    const uint64_t key_history = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEY_HISTORY];
    const uint64_t value_history = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUE_HISTORY];
    const uint64_t ones = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ONES), signs = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(SIGNS);
    const uint64_t scores = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(SCORES), bias = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(BIAS);
    const float clip = s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__QUERY_KEY_LENGTH];
    const bool full = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FULL] == 1u;
    /* past `POSITION_SCALE_FROM` a full layer's query and bias state are also scaled for the position, after their clip */
    const bool position_scaled = full && pos >= s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE_FROM];
    /* the products of the query's input — the query's, the bias state's, the sink's, the output gate's, and the key's and
     * the value's where their input is the query's — one launch where the load laid them one after another */
    const bool shared = s->key_input == s->query_input && s->value_input == s->query_input;
    const unsigned of_query[6] = { AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY, AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_STATE,
                                   AI_THINKINGMACHINES__INKLING_2__LAYER__SINK, AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT_GATE,
                                   AI_THINKINGMACHINES__INKLING_2__LAYER__KEY, AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE };
    const uint64_t of_query_rows[6] = { QW, RW, QH, QW, KVW, KVW };
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_joined(s, of_query, of_query_rows, shared ? 6u : 4u, s->query_input, s->query_input_rotated,
                                                                  s->query, H);
    if (!shared) {
        ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__KEY, s->key_input, s->key_input_rotated, s->key, KVW, H);
        ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE, s->value_input, s->value_input_rotated, s->value, KVW, H);
    }
    doors->vector_scale_l2clip((uint16_t*)(uintptr_t)(s->query_clipped), (const uint16_t*)(uintptr_t)(s->query),
                               (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(QUERY_SCALE)), QH, HW, clip, 1.0f,
                               s->over);
    if (full) {
        ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_made(s, s->key, key_history, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(KEY_TAPS),
                                                                       AI_THINKINGMACHINES__INKLING_2__LAYER__KEY_TAPS_MADE, s->key_input, s->key_convolved);
        ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_made(s, s->value, value_history, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(VALUE_TAPS),
                                                                       AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE_TAPS_MADE, s->value_input, s->value_convolved);
    } else {
        ai_thinkingmachines__inkling_2__layer__zzprivate_convolve(s, s->key, key_history, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(KEY_TAPS), KVW, s->key_convolved);
        ai_thinkingmachines__inkling_2__layer__zzprivate_convolve(s, s->value, value_history, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(VALUE_TAPS), KVW,
                                                                  s->value_convolved);
    }
    doors->vector_l2clip((uint16_t*)(uintptr_t)(keys + 2u * row * KVW), (const uint16_t*)(uintptr_t)(s->key_convolved), KVH, HW,
                         clip, s->over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(values + 2u * row * KVW), (const uint16_t*)(uintptr_t)(s->value_convolved),
                        (const uint16_t*)(uintptr_t)(ones), HW, KVH, HW, HW, s->epsilon, s->over);
    /* the bias state scaled, clipped and times √rank in one launch — or, past `POSITION_SCALE_FROM`, scaled and clipped,
     * then scaled for the position, then times √rank */
    const float rank_root = s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_RANK_ROOT];
    doors->vector_scale_l2clip((uint16_t*)(uintptr_t)(position_scaled ? s->bias_state_clipped : s->bias_query), (const uint16_t*)(uintptr_t)(s->bias_state),
                               (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(BIAS_SCALE)), QH, BR,
                               s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_LENGTH], position_scaled ? 1.0f : rank_root, s->over);
    if (position_scaled) {
        const uint64_t scale = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(POSITION_SCALE);
        doors->vector_scale_at((uint16_t*)(uintptr_t)(s->query_clipped), (const uint16_t*)(uintptr_t)(s->query_clipped),
                               (const uint16_t*)(uintptr_t)(scale), pos, QW, s->over);
        doors->vector_scale_at((uint16_t*)(uintptr_t)(s->bias_state_clipped), (const uint16_t*)(uintptr_t)(s->bias_state_clipped),
                               (const uint16_t*)(uintptr_t)(scale), pos, RW, s->over);
        doors->vector_scale((uint16_t*)(uintptr_t)(s->bias_query), (const uint16_t*)(uintptr_t)(s->bias_state_clipped), RW, rank_root, s->over);
    }
    doors->attention_scores_grouped((float*)(uintptr_t)(bias), (const uint16_t*)(uintptr_t)(s->bias_query),
                                    (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(BIAS_TABLE) + 2u * furthest * BR),
                                    QH, 1u, BR, rows);
    doors->attention_scores_grouped((float*)(uintptr_t)(scores), (const uint16_t*)(uintptr_t)(s->query_clipped),
                                    (const uint16_t*)(uintptr_t)(keys), QH, KVH, HW, rows);
    doors->attention_softmax_sink_bias((float*)(uintptr_t)(scores), (const float*)(uintptr_t)(bias), (const uint16_t*)(uintptr_t)(s->sink),
                                       QH, rows);
    doors->attention_mix((uint16_t*)(uintptr_t)(s->attended), (float*)(uintptr_t)(scores),
                         (const uint16_t*)(uintptr_t)(values), QH, KVH, HW, rows, s->over);
    doors->vector_gate_sigmoid((uint16_t*)(uintptr_t)(s->attended_gated), (const uint16_t*)(uintptr_t)(s->attended),
                               (const uint16_t*)(uintptr_t)(s->output_gate), QW, s->over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(s->attended_gated_rotated), (const uint16_t*)(uintptr_t)(s->attended_gated),
                           (const uint16_t*)(uintptr_t)(signs), QW, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT, s->attended_gated, s->attended_gated_rotated,
                                                           AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ATTENTION_ANSWER), H, QW);
}

/* `(ai_thinkingmachines__inkling_2__before_experts layer application work with_messages pos ring_row picks)` -> `picks`:
 * an application of a layer from the stream to its router's choice. `with_messages`: 1 in a loop layer's second or
 * third pass, which first mixes its four messages from what the pass before left; 0 otherwise. `pos` the position; `ring_row` its place in
 * a sliding layer's ring. Leaves the hand filled — the experts' input rotated, the eight weights — the picks in `picks`,
 * and the residual the experts' sum is added to. */
static sys__heap_node ai_thinkingmachines__inkling_2__before_experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[3].args[0] > 1u
     || argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT
     || !sys__heap_node__carries_reference(argv[6].dtype) || !sys__node_array__is(argv[6].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_thinkingmachines__inkling_2__layer__site site;
    ai_thinkingmachines__inkling_2__layer__site* s = &site;
    const uint64_t fault = ai_thinkingmachines__inkling_2__layer__zzprivate_read(argv, ctx, s);
    if (fault != 0u) return sys__engine__abi__error(fault);
    const nn__doors* doors = s->doors;
    const bool messages = argv[3].args[0] == 1u;
    const uint64_t pos = argv[4].args[0], ring_row = argv[5].args[0], picks = argv[6].args[0];
    const uint64_t H = s->hidden, QH = s->query_heads, KVW = s->key_value_width, BR = s->bias_rank;
    const uint64_t window = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__WINDOW], context = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__CONTEXT];
    const bool full = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FULL] == 1u;
    /* A full layer keeps every position: row `pos`, `pos + 1` rows, its table read from `context − 1 − pos`. A sliding
     * layer keeps the window: row `ring_row`, the rows the ring holds, its table — twice over — from `window − 1 − ring_row`. */
    if (window == 0u || pos >= context || ring_row >= window || sys__node_array__length(picks) < s->top_k)
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE);
    const uint64_t row = full ? pos : ring_row;
    const uint64_t rows = full ? pos + 1u : (pos < window ? pos + 1u : window);
    const uint64_t furthest = full ? context - 1u - pos : window - 1u - ring_row;
    if (!nn__primitives__fits((row + 1u) * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS])
     || !nn__primitives__fits(rows * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS])
     || !nn__primitives__fits((row + 1u) * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES])
     || !nn__primitives__fits(rows * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES])
     || !nn__primitives__fits((furthest + rows) * BR, s->layer_room[AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_TABLE])
     || !nn__primitives__fits(2u * QH * rows, s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__SCORES])
     || !nn__primitives__fits(2u * QH * rows, s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__BIAS])
     || (full && pos >= s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE_FROM]
         && !nn__primitives__fits(pos + 1u, s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE])))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE);

    const uint64_t stream = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STREAM), states = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STATES);
    const uint64_t residual = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(RESIDUAL), hand = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(HAND);
    const uint64_t answer = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ATTENTION_ANSWER);
    const uint64_t convolved = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ATTENTION_CONVOLVED);
    const uint64_t experts_mix = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(EXPERTS_MIX), told = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(MESSAGES);
    if (messages) {             /* its four messages — to its query, its key, its value and its experts: four queries' mix */
        const uint64_t left = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(PASS_STATES);
        doors->attention_states((uint16_t*)(uintptr_t)(told), (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(MESSAGE_QUERIES)),
                                (const uint16_t*)(uintptr_t)(left), (const uint16_t*)(uintptr_t)(left), 4u, H, 2u, 0u, s->epsilon, s->over);
    }
    ai_thinkingmachines__inkling_2__layer__zzprivate_inputs(s, messages);
    ai_thinkingmachines__inkling_2__layer__zzprivate_attention(s, pos, row, rows, furthest);
    /* at a block's first layer the stream joins the block states, and restarts from the attention's answer through its
     * convolution; at another the convolution is added to the stream — one launch either way */
    const bool block_start = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_START] == 1u;
    if (block_start)
        doors->rmsnorm((uint16_t*)(uintptr_t)(states + 2u * s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_ROW] * H),
                       (const uint16_t*)(uintptr_t)(stream), (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ONES)),
                       H, s->epsilon, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_add(s, answer, s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__ATTENTION_HISTORY],
                                                                  AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(ATTENTION_TAPS), H, convolved,
                                                                  block_start ? 0u : stream, residual);
    ai_thinkingmachines__inkling_2__layer__zzprivate_mix(s, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(EXPERTS_QUERY), 1u,
                                                         s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_MIXED], residual, experts_mix);
    uint64_t read = experts_mix;
    if (messages) {
        ai_thinkingmachines__inkling_2__layer__zzprivate_gate(s, AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLERS + 3u * AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLER, experts_mix, told + 2u * 3u * H, s->experts_input);
        read = s->experts_input;
    }
    /* the router: its score a sigmoid, its bias in the choice alone, the eight's scores over their sum times the scale;
     * the hand is the experts' input rotated, then the weights */
    doors->rmsnorm_rotate((uint16_t*)(uintptr_t)(hand), (uint16_t*)(uintptr_t)(s->experts_normed), (const uint16_t*)(uintptr_t)(read),
                          (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(EXPERTS_NORM)),
                          (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(SIGNS)), H, s->epsilon, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__ROUTER, s->experts_normed, hand, s->router_logits, s->experts, H);
    const uint64_t chosen = nn__routed__top_k_scored(ctx, doors, picks, hand + 2u * H, s->router_logits, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(ROUTER_BIAS),
                                                     s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__ROUTE_SCALE], s->experts, s->top_k, NN__ROUTED__BIASED, 0);
    return chosen != 0u ? sys__engine__abi__error(chosen) : nn__doors_answer(&argv[6]);
}

/* `(ai_thinkingmachines__inkling_2__after_experts layer application work routed)` -> `routed`: the experts' sum through
 * its short convolution, added to the residual `before_experts` kept: the application's output, the stream. */
static sys__heap_node ai_thinkingmachines__inkling_2__after_experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t routed = 0, room = 0;
    if (!nn__primitives__room(&argv[3], &routed, &room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_thinkingmachines__inkling_2__layer__site site;
    ai_thinkingmachines__inkling_2__layer__site* s = &site;
    const uint64_t fault = ai_thinkingmachines__inkling_2__layer__zzprivate_read(argv, ctx, s);
    if (fault != 0u) return sys__engine__abi__error(fault);
    if (!nn__primitives__fits(s->hidden, room)) return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE);
    ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_add(s, routed, s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__EXPERTS_HISTORY],
                                                                  AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(EXPERTS_TAPS), s->hidden, s->routed_convolved,
                                                                  AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(RESIDUAL),
                                                                  AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STREAM));
    return nn__doors_answer(&argv[3]);
}

#undef AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W
#undef AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L

SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__before_experts__zzabi_adapter, ai_thinkingmachines__inkling_2__before_experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__after_experts__zzabi_adapter, ai_thinkingmachines__inkling_2__after_experts__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_LAYER__ABI_CUH */
