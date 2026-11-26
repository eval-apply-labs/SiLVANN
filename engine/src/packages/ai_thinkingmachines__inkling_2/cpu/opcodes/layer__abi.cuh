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
    uint64_t scratch_end;                   /* where the parts above end: a chunk's rows are laid from here (▶ `rows`) */
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
    s->scratch_end = next;
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

/* A full layer's key or value: a head's taps are also made from the position's own state — matrix `made` over the
 * state gives a head its four, head g's lag the (taps·g + lag)-th, `made_taps` — and join the layer's own; the sum is
 * over √(2·taps + 1). */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_made(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t input, uint64_t history, uint64_t taps,
                                                                           uint64_t made_taps, uint64_t out) {
    s->doors->convolution_step((uint16_t*)(uintptr_t)(out), (uint16_t*)(uintptr_t)(history), (const uint16_t*)(uintptr_t)(taps),
                               (const uint16_t*)(uintptr_t)(made_taps), (const uint16_t*)(uintptr_t)(input), s->key_value_width,
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

/* The products of the query's input — the query's, the bias state's, the sink's, the output gate's, and the key's and
 * the value's where their input is the query's — one launch where the load laid them one after another; the key's and
 * the value's of their own inputs otherwise. Into `s->query` and the parts after it, in the matrices' order. */
static const unsigned ai_thinkingmachines__inkling_2__layer__zzprivate_of_query[6] = {
    AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY, AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_STATE, AI_THINKINGMACHINES__INKLING_2__LAYER__SINK,
    AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT_GATE, AI_THINKINGMACHINES__INKLING_2__LAYER__KEY, AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE };
static void ai_thinkingmachines__inkling_2__layer__zzprivate_attention_products(const ai_thinkingmachines__inkling_2__layer__site* s) {
    const uint64_t H = s->hidden, QW = s->query_width, KVW = s->key_value_width;
    const bool shared = s->key_input == s->query_input && s->value_input == s->query_input;
    const uint64_t of_query_rows[6] = { QW, s->bias_width, s->query_heads, QW, KVW, KVW };
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_joined(s, ai_thinkingmachines__inkling_2__layer__zzprivate_of_query, of_query_rows, shared ? 6u : 4u,
                                                                  s->query_input, s->query_input_rotated, s->query, H);
    if (!shared) {
        ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__KEY, s->key_input, s->key_input_rotated, s->key, KVW, H);
        ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE, s->value_input, s->value_input_rotated, s->value, KVW, H);
    }
}

/* The attention at position `pos`, from its products to the answer gated, `s->attended_gated`. The query scaled a head
 * and clipped to a length; the key and the value through their short convolutions — the key then clipped, the value
 * normed a head — into the application's cache at `row`; the position's bias, a head's values (scaled, clipped, times
 * √rank since nn's scores divide by it) against the table laid a row a distance from `furthest`; the scores over the
 * cache's `rows`, the bias added, the sink in the softmax, the values mixed; the answer gated by a sigmoid of its own
 * projection. A full layer's made taps: `made_key` and `made_value` where a chunk's were made beforehand, else made
 * here, each before its convolution. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_attention_position(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t pos,
                                                                                uint64_t row, uint64_t rows, uint64_t furthest,
                                                                                uint64_t made_key, uint64_t made_value) {
    const nn__doors* doors = s->doors;
    const uint64_t QH = s->query_heads, KVH = s->key_value_heads, HW = s->head_width;
    const uint64_t QW = s->query_width, KVW = s->key_value_width, BR = s->bias_rank, RW = s->bias_width;
    const uint64_t keys = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS], values = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES];
    const uint64_t key_history = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEY_HISTORY];
    const uint64_t value_history = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUE_HISTORY];
    const uint64_t ones = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ONES);
    const uint64_t scores = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(SCORES), bias = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(BIAS);
    const float clip = s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__QUERY_KEY_LENGTH];
    const bool full = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FULL] == 1u;
    /* past `POSITION_SCALE_FROM` a full layer's query and bias state are also scaled for the position, after their clip */
    const bool position_scaled = full && pos >= s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE_FROM];
    doors->vector_scale_l2clip((uint16_t*)(uintptr_t)(s->query_clipped), (const uint16_t*)(uintptr_t)(s->query),
                               (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(QUERY_SCALE)), QH, HW, clip, 1.0f,
                               s->over);
    if (full) {
        const uint64_t made_rows = KVH * s->taps;
        if (made_key == 0u)
            ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__KEY_TAPS_MADE, s->key_input, s->key_input,
                                                                   s->made_taps, made_rows, s->hidden);
        ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_made(s, s->key, key_history, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(KEY_TAPS),
                                                                       made_key != 0u ? made_key : s->made_taps, s->key_convolved);
        if (made_value == 0u)
            ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE_TAPS_MADE, s->value_input, s->value_input,
                                                                   s->made_taps, made_rows, s->hidden);
        ai_thinkingmachines__inkling_2__layer__zzprivate_convolve_made(s, s->value, value_history, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(VALUE_TAPS),
                                                                       made_value != 0u ? made_value : s->made_taps, s->value_convolved);
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
}

/* The attention at one position, into the work's answer: its products, the attention, the answer gated, rotated and
 * through the output's matrix. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_attention(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t pos,
                                                                       uint64_t row, uint64_t rows, uint64_t furthest) {
    const uint64_t QW = s->query_width;
    ai_thinkingmachines__inkling_2__layer__zzprivate_attention_products(s);
    ai_thinkingmachines__inkling_2__layer__zzprivate_attention_position(s, pos, row, rows, furthest, 0u, 0u);
    s->doors->hadamard_rotate((uint16_t*)(uintptr_t)(s->attended_gated_rotated), (const uint16_t*)(uintptr_t)(s->attended_gated),
                              (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(SIGNS)), QW, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times(s, AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT, s->attended_gated, s->attended_gated_rotated,
                                                           AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ATTENTION_ANSWER), s->hidden, QW);
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

/* ══ A CHUNK OF A PROMPT, AS ROWS ══════════════════════════════════════════════════════════════════════════════════════
 * The same two verbs over `n` positions from `first` (▶ `contracts/objects/layer.cuh`, "A chunk"). Every matrix reads
 * the chunk's rows in one product (nn's `expert_rows`), and the router chooses for every row in one launch; what a
 * position reads of the positions before it — the attention over the cache, the short convolutions over their inputs —
 * the learned queries' mixes and the message gates run a position at a time, in order, as the one-position verbs run
 * them. The cache and the convolutions' inputs are left as a position at a time leaves them; the products sum in their
 * own order, so the two forms agree to a bound, not to the bit. */

/* What a chunk's verb keeps beside the site: the chunk, and where each of its parts begins in the scratch — a part a
 * row a position, `n` rows of it. */
typedef struct ai_thinkingmachines__inkling_2__layer__chunk {
    uint64_t first, n, hand_row;
    uint64_t pairs;                                               /* n pairs, row t to row t: a product's rows */
    uint64_t inputs;                                              /* the attention's learned mixes, three rows a position */
    uint64_t gated;                                               /* the three gates' answers, a gate's n rows after another's */
    uint64_t gate_pair, gate_pair_rotated, gate_rank, gate_rank_rotated, forget, gain, gate_input, gate_message;
    uint64_t query_input, query_input_rotated, key_input, key_input_rotated, value_input, value_input_rotated;
    uint64_t query, bias_state, sink, output_gate, key, value, made_key, made_value;
    uint64_t query_scale, bias_scale, query_clipped, bias_clipped, bias_query;
    uint64_t attended, gated_answer, gated_answer_rotated;
    uint64_t key_convolved, value_convolved, convolved;           /* the short convolutions' answers */
    uint64_t window_keys, window_values;                          /* a sliding layer's window and the chunk, a row a position */
    uint64_t experts_mix, experts_input, normed, rotated, logits;
} ai_thinkingmachines__inkling_2__layer__chunk;

/* The chunk's parts laid out in the scratch after the one-position parts, each where it begins; false past its room. */
static bool ai_thinkingmachines__inkling_2__layer__zzprivate_chunk(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t first, uint64_t n,
                                                                   ai_thinkingmachines__inkling_2__layer__chunk* c) {
    const uint64_t H = s->hidden, QW = s->query_width, KVW = s->key_value_width, RW = s->bias_width, GR = s->gate_rank;
    const uint64_t made = s->key_value_heads * s->taps;
    c->first = first; c->n = n;
    c->hand_row = AI_THINKINGMACHINES__INKLING_2__HAND__ROW(H, s->top_k);
    uint64_t next = (s->scratch_end + 7u) & ~7ull;
#define AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(name, halves) c->name = ai_thinkingmachines__inkling_2__layer__zzprivate_take(&next, n * (halves))
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(pairs, 4u);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(inputs, 3u * H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gated, 3u * H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gate_pair, 2u * H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gate_pair_rotated, 2u * H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gate_rank, GR);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gate_rank_rotated, GR);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(forget, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gain, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gate_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gate_message, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(query_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(query_input_rotated, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(key_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(key_input_rotated, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(value_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(value_input_rotated, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(query, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(bias_state, RW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(sink, s->query_heads);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(output_gate, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(key, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(value, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(made_key, made);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(made_value, made);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(query_scale, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(bias_scale, RW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(query_clipped, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(bias_clipped, RW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(bias_query, RW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(attended, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gated_answer, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(gated_answer_rotated, QW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(key_convolved, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(value_convolved, KVW);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(convolved, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(experts_mix, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(experts_input, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(normed, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(rotated, H);
    AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS(logits, s->experts);
#undef AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_ROWS
    const uint64_t window = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__WINDOW];
    c->window_keys = ai_thinkingmachines__inkling_2__layer__zzprivate_take(&next, (window + n) * KVW);
    c->window_values = ai_thinkingmachines__inkling_2__layer__zzprivate_take(&next, (window + n) * KVW);
    return next - s->work[AI_THINKINGMACHINES__INKLING_2__WORK__SCRATCH] <= s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__SCRATCH];
}

/* Matrix `m` of the layer's table over the chunk's `x` rows (`cols` halves each) into `out`, `rows` halves a row. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(const ai_thinkingmachines__inkling_2__layer__site* s,
                                                                        const ai_thinkingmachines__inkling_2__layer__chunk* c, unsigned m,
                                                                        uint64_t plain, uint64_t rotated, uint64_t out, uint64_t rows, uint64_t cols) {
    const uint64_t d = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BITS + m];
    s->doors->expert_rows((uint16_t*)(uintptr_t)(out), (const uint8_t*)(uintptr_t)s->layer[2u * m], s->layer_room[2u * m],
                          (const uint8_t*)(uintptr_t)s->layer[2u * m + 1u], s->layer_room[2u * m + 1u],
                          (const uint16_t*)(uintptr_t)(d == 16ull ? plain : rotated), (const uint32_t*)(uintptr_t)c->pairs, c->n, d, rows, cols,
                          s->over);
}

/* Every row of `n` rotated as a one-position row of `cols` is (`hadamard_rotate`): one launch over all of them where a
 * row is whole blocks of 512 — the blocks then fall as they fall row by row — else the pack's ladder a row at a time. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_rotate_rows(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t out,
                                                                         uint64_t in, uint64_t cols, uint64_t n) {
    const uint64_t signs = s->work[AI_THINKINGMACHINES__INKLING_2__WORK__SIGNS];
    if (cols % NN__HADAMARD__WHOLE_BELOW == 0u)
        s->doors->hadamard_rotate((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)in, (const uint16_t*)(uintptr_t)signs, n * cols, s->over);
    else
        s->doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)in, (const uint16_t*)(uintptr_t)signs, cols, n, s->over);
}

/* A message gate over the chunk (▶ `gate`): `x` and `y` rows `x_stride` and `y_stride` halves apart, each normed, the
 * pair rotated and through the controller's map of the gate's rank — matrices `controller`, +1 and +2: down, forget,
 * gain — then (1 − tanh(forget))·x + tanh(gain)·y into `out`, its rows one after another. Each step the one-position
 * gate's doors over every row: the norms a launch each, the rotations, a product a matrix, the gate one launch. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_gate_rows(const ai_thinkingmachines__inkling_2__layer__site* s,
                                                                       const ai_thinkingmachines__inkling_2__layer__chunk* c, unsigned controller,
                                                                       uint64_t x, uint64_t x_stride, uint64_t y, uint64_t y_stride, uint64_t out) {
    const nn__doors* doors = s->doors;
    const uint64_t H = s->hidden, GR = s->gate_rank, n = c->n;
    const uint64_t ones = s->work[AI_THINKINGMACHINES__INKLING_2__WORK__ONES], signs = s->work[AI_THINKINGMACHINES__INKLING_2__WORK__SIGNS];
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(c->gate_pair), (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)ones, H, n, x_stride, 2u * H,
                        s->epsilon, s->over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(c->gate_pair + 2u * H), (const uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)ones, H, n, y_stride,
                        2u * H, s->epsilon, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_rotate_rows(s, c->gate_pair_rotated, c->gate_pair, 2u * H, n);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, controller, c->gate_pair, c->gate_pair_rotated, c->gate_rank, GR, 2u * H);
    doors->hadamard_blocks((uint16_t*)(uintptr_t)(c->gate_rank_rotated), (const uint16_t*)(uintptr_t)(c->gate_rank), (const uint16_t*)(uintptr_t)signs,
                           n * GR, GR, 0u, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, controller + 1u, c->gate_rank, c->gate_rank_rotated, c->forget, H, GR);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, controller + 2u, c->gate_rank, c->gate_rank_rotated, c->gain, H, GR);
    if (x_stride != H) {
        doors->vector_copy_rows((uint16_t*)(uintptr_t)(c->gate_input), H, (const uint16_t*)(uintptr_t)x, x_stride, H, n);
        x = c->gate_input;
    }
    if (y_stride != H) {
        doors->vector_copy_rows((uint16_t*)(uintptr_t)(c->gate_message), H, (const uint16_t*)(uintptr_t)y, y_stride, H, n);
        y = c->gate_message;
    }
    doors->vector_gate_tanh((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)y,
                            (const uint16_t*)(uintptr_t)(c->forget), (const uint16_t*)(uintptr_t)(c->gain), n * H, s->over);
}

/* The chunk's rows of `in`, `stride` halves apart, normed by `norm` and rotated: into `normed` and `rotated`, a row
 * after another — `rmsnorm_rotate` a row at a time, as its two doors over every row. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_norm_rotate_rows(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t n,
                                                                              uint64_t in, uint64_t stride, uint64_t norm, uint64_t normed,
                                                                              uint64_t rotated) {
    const uint64_t H = s->hidden;
    s->doors->rmsnorm_rows((uint16_t*)(uintptr_t)normed, (const uint16_t*)(uintptr_t)in, (const uint16_t*)(uintptr_t)norm, H, n, stride, H,
                           s->epsilon, s->over);
    s->doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)rotated, (const uint16_t*)(uintptr_t)normed,
                                   (const uint16_t*)(uintptr_t)(s->work[AI_THINKINGMACHINES__INKLING_2__WORK__SIGNS]), H, n, s->over);
}

/* A position's place in the application's cache, the rows it reads and where its bias table begins (▶ `before_experts`);
 * false where the cache, the table or the scores have no room for it. */
static bool ai_thinkingmachines__inkling_2__layer__zzprivate_place(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t pos,
                                                                   uint64_t* row, uint64_t* rows, uint64_t* furthest) {
    const uint64_t QH = s->query_heads, KVW = s->key_value_width, BR = s->bias_rank;
    const uint64_t window = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__WINDOW], context = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__CONTEXT];
    const bool full = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FULL] == 1u;
    if (window == 0u || pos >= context) return false;
    const uint64_t ring_row = pos % window;
    *row = full ? pos : ring_row;
    *rows = full ? pos + 1u : (pos < window ? pos + 1u : window);
    *furthest = full ? context - 1u - pos : window - 1u - ring_row;
    return nn__primitives__fits((*row + 1u) * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS])
        && nn__primitives__fits(*rows * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS])
        && nn__primitives__fits((*row + 1u) * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES])
        && nn__primitives__fits(*rows * KVW, s->own_room[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES])
        && nn__primitives__fits((*furthest + *rows) * BR, s->layer_room[AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_TABLE])
        && nn__primitives__fits(2u * QH * *rows, s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__SCORES])
        && nn__primitives__fits(2u * QH * *rows, s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__BIAS])
        && (!full || pos < s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE_FROM]
            || nn__primitives__fits(pos + 1u, s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE]));
}

/* Rows `ua … ub` of the chunk attend their windows (▶ nn's `attention_window_*`): row `u`'s `len0 + (u − ua)·step` key and
 * value rows from `k` and `v` plus `(u − ua)·k_row` halves, the bias table's row `last` distance 0; the queries clipped,
 * the bias states, the sinks and the answers the chunk's. As many rows at a time as the work's scores hold. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_attend(const ai_thinkingmachines__inkling_2__layer__site* s,
                                                                    const ai_thinkingmachines__inkling_2__layer__chunk* c, uint64_t ua, uint64_t ub,
                                                                    uint64_t k, uint64_t v, uint64_t k_row, uint64_t len0, uint64_t step,
                                                                    uint64_t last) {
    const nn__doors* doors = s->doors;
    const uint64_t QH = s->query_heads, KVH = s->key_value_heads, HW = s->head_width, QW = s->query_width, BR = s->bias_rank, RW = s->bias_width;
    const uint64_t scores = s->work[AI_THINKINGMACHINES__INKLING_2__WORK__SCORES];
    const uint64_t room = s->work_room[AI_THINKINGMACHINES__INKLING_2__WORK__SCORES];
    const uint64_t longest = len0 + (ub - ua - 1u) * step, fits = room / (4u * QH * longest);
    const uint64_t each = fits == 0u ? 1u : fits;
    for (uint64_t u = ua; u < ub; u += each) {
        const uint64_t m = ub - u < each ? ub - u : each, length = len0 + (u - ua) * step;
        const uint64_t p_row = QH * (length + (m - 1u) * step), keys = k + 2u * (u - ua) * k_row, values = v + 2u * (u - ua) * k_row;
        doors->attention_window_scores((float*)(uintptr_t)scores, p_row, (const uint16_t*)(uintptr_t)(c->query_clipped + 2u * u * QW),
                                       (const uint16_t*)(uintptr_t)keys, k_row, (const uint16_t*)(uintptr_t)(c->bias_query + 2u * u * RW),
                                       (const uint16_t*)(uintptr_t)(s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_TABLE]), last, length, step,
                                       QH, KVH, HW, BR, m);
        doors->attention_window_softmax((float*)(uintptr_t)scores, p_row, (const uint16_t*)(uintptr_t)(c->sink + 2u * u * QH), QH, length, step,
                                        QH, m);
        doors->attention_window_mix((uint16_t*)(uintptr_t)(c->attended + 2u * u * QW), QW, (const float*)(uintptr_t)scores, p_row,
                                    (const uint16_t*)(uintptr_t)values, k_row, length, step, QH, KVH, HW, m, s->over);
    }
}

/* `rows` rows of `width` halves from a ring of `ring` rows, from its row `at`, to `out` — or from `in` into the ring
 * where `into` — in at most two runs, as the ring wraps. */
static void ai_thinkingmachines__inkling_2__layer__zzprivate_ring_copy(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t ring_at,
                                                                       uint64_t ring, uint64_t at, uint64_t other, uint64_t rows, uint64_t width,
                                                                       bool into) {
    const uint64_t first_run = ring - at < rows ? ring - at : rows;
    const uint64_t runs[2][3] = { { ring_at + 2u * at * width, other, first_run }, { ring_at, other + 2u * first_run * width, rows - first_run } };
    for (unsigned i = 0u; i < 2u; ++i) {
        if (runs[i][2] == 0u) continue;
        const uint64_t from = into ? runs[i][1] : runs[i][0], to = into ? runs[i][0] : runs[i][1];
        s->doors->vector_copy((uint16_t*)(uintptr_t)to, (const uint16_t*)(uintptr_t)from, runs[i][2] * width);
    }
}

/* The work's chunk buffers hold `n` rows, and the experts' sum `routed` (a buffer, its room) too where it is given. */
static bool ai_thinkingmachines__inkling_2__layer__zzprivate_holds(const ai_thinkingmachines__inkling_2__layer__site* s, uint64_t n) {
    const uint64_t H = s->hidden, *wr = s->work_room;
    return n != 0u && n <= AI_THINKINGMACHINES__INKLING_2__ROWS_MAX
        && nn__primitives__fits(n * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__STREAM])
        && nn__primitives__fits(n * AI_THINKINGMACHINES__INKLING_2__STATES__ROWS * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__STATES])
        && nn__primitives__fits(n * 2u * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__PASS_STATES])
        && nn__primitives__fits(n * 4u * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__MESSAGES])
        && n * AI_THINKINGMACHINES__INKLING_2__HAND__ROW(H, s->top_k) <= wr[AI_THINKINGMACHINES__INKLING_2__WORK__HAND]
        && nn__primitives__fits(n * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__RESIDUAL])
        && nn__primitives__fits(n * H, wr[AI_THINKINGMACHINES__INKLING_2__WORK__ATTENTION_ANSWER]);
}

/* `(ai_thinkingmachines__inkling_2__before_experts_rows layer application work with_messages first n)` -> `work`: an
 * application over the `n` positions from `first`, the work a chunk's, as `before_experts` over each in turn. Leaves
 * each row's hand filled — the experts' input rotated, the eight weights, the picks after them as words — and its
 * residual. Every step runs over the chunk's rows at once — the short convolutions too, each lane sliding a channel's
 * inputs through the positions in turn, and the attention, each row over its own window of the keys. */
static sys__heap_node ai_thinkingmachines__inkling_2__before_experts_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[3].args[0] > 1u || argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_thinkingmachines__inkling_2__layer__site site;
    ai_thinkingmachines__inkling_2__layer__site* s = &site;
    const uint64_t fault = ai_thinkingmachines__inkling_2__layer__zzprivate_read(argv, ctx, s);
    if (fault != 0u) return sys__engine__abi__error(fault);
    const nn__doors* doors = s->doors;
    const bool messages = argv[3].args[0] == 1u;
    const uint64_t first = argv[4].args[0], n = argv[5].args[0];
    const uint64_t H = s->hidden, QH = s->query_heads, KVH = s->key_value_heads, HW = s->head_width, QW = s->query_width;
    const uint64_t KVW = s->key_value_width, BR = s->bias_rank, RW = s->bias_width, E = s->experts, K = s->top_k;
    ai_thinkingmachines__inkling_2__layer__chunk chunk;
    ai_thinkingmachines__inkling_2__layer__chunk* c = &chunk;
    if (!ai_thinkingmachines__inkling_2__layer__zzprivate_holds(s, n) || first > (1ull << 40)
     || !ai_thinkingmachines__inkling_2__layer__zzprivate_chunk(s, first, n, c))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE);
    for (uint64_t r = 0u; r < n; ++r) {                     /* every position's place, before anything is launched */
        uint64_t row = 0u, rows = 0u, furthest = 0u;
        if (!ai_thinkingmachines__inkling_2__layer__zzprivate_place(s, first + r, &row, &rows, &furthest))
            return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE);
    }
    const uint64_t ones = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ONES), hands = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(HAND);
    const uint64_t stream = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STREAM), states = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STATES);
    const uint64_t left = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(PASS_STATES), told = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(MESSAGES);
    const uint64_t residual = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(RESIDUAL), answer = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(ATTENTION_ANSWER);
    const uint64_t SH = AI_THINKINGMACHINES__INKLING_2__STATES__ROWS * H;
    const bool full = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FULL] == 1u, first_layer = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__FIRST] == 1u;
    doors->expert_pairs_rows((uint32_t*)(uintptr_t)c->pairs, n, 1u, 0u, 1u, 0u);

    /* ① what the attention reads, over every row: each position's messages and learned mix over its own states, the
     *   gates, the norms and their rotations */
    if (messages)
        doors->attention_states_rows((uint16_t*)(uintptr_t)told, (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(MESSAGE_QUERIES)),
                                     (const uint16_t*)(uintptr_t)left, (const uint16_t*)(uintptr_t)left, 4u, H, 2u, 0u, s->epsilon, n, 2u * H, 2u * H,
                                     4u * H, s->over);
    if (!first_layer)
        doors->attention_states_rows((uint16_t*)(uintptr_t)(c->inputs), (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(ATTENTION_QUERY)),
                                     (const uint16_t*)(uintptr_t)states, (const uint16_t*)(uintptr_t)stream, full ? 3u : 1u, H,
                                     s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_ATTENDED], 1u, s->epsilon, n, SH, H, 3u * H, s->over);
    uint64_t from[3], stride[3];
    for (unsigned i = 0u; i < 3u; ++i) {
        from[i] = first_layer ? stream : full ? c->inputs + 2u * i * H : c->inputs;
        stride[i] = first_layer ? H : 3u * H;
    }
    if (messages)
        for (unsigned i = 0u; i < 3u; ++i) {
            const uint64_t out = c->gated + 2u * i * n * H;
            ai_thinkingmachines__inkling_2__layer__zzprivate_gate_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLERS + i * AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLER,
                                                                       from[i], stride[i], told + 2u * i * H, 4u * H, out);
            from[i] = out; stride[i] = H;
        }
    const uint64_t normed[3] = { c->query_input, c->key_input, c->value_input };
    const uint64_t rotated[3] = { c->query_input_rotated, c->key_input_rotated, c->value_input_rotated };
    uint64_t input[3], input_rotated[3];
    for (unsigned i = 0u; i < 3u; ++i) {
        if (i != 0u && from[i] == from[0] && s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_NORM + i] == s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_NORM]) {
            input[i] = input[0]; input_rotated[i] = input_rotated[0];
            continue;
        }
        ai_thinkingmachines__inkling_2__layer__zzprivate_norm_rotate_rows(s, n, from[i], stride[i], s->layer[AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_NORM + i],
                                                                          normed[i], rotated[i]);
        input[i] = normed[i]; input_rotated[i] = rotated[i];
    }

    /* ② the products, a matrix a launch over every row */
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY, input[0], input_rotated[0], c->query, QW, H);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_STATE, input[0], input_rotated[0], c->bias_state, RW, H);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__SINK, input[0], input_rotated[0], c->sink, QH, H);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT_GATE, input[0], input_rotated[0], c->output_gate, QW, H);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__KEY, input[1], input_rotated[1], c->key, KVW, H);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE, input[2], input_rotated[2], c->value, KVW, H);
    const uint64_t made = KVH * s->taps;
    if (full) {                                             /* the made taps read the state as it is */
        ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__KEY_TAPS_MADE, input[1], input[1], c->made_key, made, H);
        ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE_TAPS_MADE, input[2], input[2], c->made_value, made, H);
    }

    /* ③ the query and the bias state scaled and clipped over every row — a layer's scale laid once a row for it — and,
     *   past `POSITION_SCALE_FROM` in a full layer, each such row scaled for its position before the bias's √rank */
    const float clip = s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__QUERY_KEY_LENGTH], rank_root = s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_RANK_ROOT];
    doors->vector_copy_rows((uint16_t*)(uintptr_t)(c->query_scale), QW, (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(QUERY_SCALE)), 0u, QW, n);
    doors->vector_copy_rows((uint16_t*)(uintptr_t)(c->bias_scale), RW, (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(BIAS_SCALE)), 0u, RW, n);
    doors->vector_scale_l2clip((uint16_t*)(uintptr_t)(c->query_clipped), (const uint16_t*)(uintptr_t)(c->query), (const uint16_t*)(uintptr_t)(c->query_scale),
                               n * QH, HW, clip, 1.0f, s->over);
    const uint64_t scaled_from = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE_FROM];
    const bool position_scaled = full && first + n > scaled_from;
    doors->vector_scale_l2clip((uint16_t*)(uintptr_t)(position_scaled ? c->bias_clipped : c->bias_query), (const uint16_t*)(uintptr_t)(c->bias_state),
                               (const uint16_t*)(uintptr_t)(c->bias_scale), n * QH, BR, s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_LENGTH],
                               position_scaled ? 1.0f : rank_root, s->over);
    if (position_scaled) {
        const uint64_t scale = AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(POSITION_SCALE);
        for (uint64_t r = 0u; r < n; ++r) {
            if (first + r < scaled_from) continue;
            doors->vector_scale_at((uint16_t*)(uintptr_t)(c->query_clipped + 2u * r * QW), (const uint16_t*)(uintptr_t)(c->query_clipped + 2u * r * QW),
                                   (const uint16_t*)(uintptr_t)(scale), first + r, QW, s->over);
            doors->vector_scale_at((uint16_t*)(uintptr_t)(c->bias_clipped + 2u * r * RW), (const uint16_t*)(uintptr_t)(c->bias_clipped + 2u * r * RW),
                                   (const uint16_t*)(uintptr_t)(scale), first + r, RW, s->over);
        }
        doors->vector_scale((uint16_t*)(uintptr_t)(c->bias_query), (const uint16_t*)(uintptr_t)(c->bias_clipped), n * RW, rank_root, s->over);
    }

    /* ④ the key and the value through their convolutions over every row, into the cache, and every row's attention over
     *   its window: a full layer's — or a sliding one's before its window is full — the cache from its start, a row a
     *   position; past that the window before the chunk and the chunk's own, laid a row a position, the chunk's then into
     *   the ring. Three launches over the chunk (nn's `attention_window_*`). */
    const uint64_t keys = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS], values = s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES];
    const float convolution_scale = s->work_r[full ? AI_THINKINGMACHINES__INKLING_2__WORK__MADE_SCALE : AI_THINKINGMACHINES__INKLING_2__WORK__CONVOLUTION_SCALE];
    doors->convolution_steps((uint16_t*)(uintptr_t)(c->key_convolved), 0, (uint16_t*)(uintptr_t)(s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEY_HISTORY]),
                             (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(KEY_TAPS)),
                             (const uint16_t*)(uintptr_t)(full ? c->made_key : 0u), (const uint16_t*)(uintptr_t)(c->key), 0, KVW, s->taps,
                             full ? HW : 0u, convolution_scale, n, made, s->over);
    doors->convolution_steps((uint16_t*)(uintptr_t)(c->value_convolved), 0, (uint16_t*)(uintptr_t)(s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUE_HISTORY]),
                             (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(VALUE_TAPS)),
                             (const uint16_t*)(uintptr_t)(full ? c->made_value : 0u), (const uint16_t*)(uintptr_t)(c->value), 0, KVW, s->taps,
                             full ? HW : 0u, convolution_scale, n, made, s->over);
    const uint64_t window = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__WINDOW], context = s->work_v[AI_THINKINGMACHINES__INKLING_2__WORK__CONTEXT];
    const uint64_t last = full ? context - 1u : window - 1u;
    if (full || first + n <= window) {                      /* the cache a row a position: each row's window its start */
        uint64_t row = 0u, rows = 0u, furthest = 0u;
        (void)ai_thinkingmachines__inkling_2__layer__zzprivate_place(s, first, &row, &rows, &furthest);
        doors->vector_l2clip((uint16_t*)(uintptr_t)(keys + 2u * row * KVW), (const uint16_t*)(uintptr_t)(c->key_convolved), n * KVH, HW, clip, s->over);
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)(values + 2u * row * KVW), (const uint16_t*)(uintptr_t)(c->value_convolved),
                            (const uint16_t*)(uintptr_t)(ones), HW, n * KVH, HW, HW, s->epsilon, s->over);
        ai_thinkingmachines__inkling_2__layer__zzprivate_attend(s, c, 0u, n, keys, values, 0u, first + 1u, 1u, last);
    } else {
        /* a sliding layer whose window fills within the chunk or before it: the window before the chunk out of the ring
         *   and the chunk's keys and values after it, a row a position — each row's window a run of them — then the
         *   chunk's into the ring */
        const uint64_t from = first >= window - 1u ? first - (window - 1u) : 0u, before = first - from;
        ai_thinkingmachines__inkling_2__layer__zzprivate_ring_copy(s, keys, window, from % window, c->window_keys, before, KVW, false);
        ai_thinkingmachines__inkling_2__layer__zzprivate_ring_copy(s, values, window, from % window, c->window_values, before, KVW, false);
        doors->vector_l2clip((uint16_t*)(uintptr_t)(c->window_keys + 2u * before * KVW), (const uint16_t*)(uintptr_t)(c->key_convolved), n * KVH, HW,
                             clip, s->over);
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)(c->window_values + 2u * before * KVW), (const uint16_t*)(uintptr_t)(c->value_convolved),
                            (const uint16_t*)(uintptr_t)(ones), HW, n * KVH, HW, HW, s->epsilon, s->over);
        const uint64_t filling = from == 0u ? window - 1u - first : 0u;       /* the rows whose window is not yet full */
        if (filling != 0u)
            ai_thinkingmachines__inkling_2__layer__zzprivate_attend(s, c, 0u, filling, c->window_keys, c->window_values, 0u, first + 1u, 1u, last);
        ai_thinkingmachines__inkling_2__layer__zzprivate_attend(s, c, filling, n, c->window_keys, c->window_values, KVW, window, 0u, last);
        ai_thinkingmachines__inkling_2__layer__zzprivate_ring_copy(s, keys, window, first % window, c->window_keys + 2u * before * KVW, n, KVW, true);
        ai_thinkingmachines__inkling_2__layer__zzprivate_ring_copy(s, values, window, first % window, c->window_values + 2u * before * KVW, n, KVW, true);
    }

    /* ⑤ the answers gated, rotated and through the output's matrix, over every row */
    doors->vector_gate_sigmoid((uint16_t*)(uintptr_t)(c->gated_answer), (const uint16_t*)(uintptr_t)(c->attended),
                               (const uint16_t*)(uintptr_t)(c->output_gate), n * QW, s->over);
    ai_thinkingmachines__inkling_2__layer__zzprivate_rotate_rows(s, c->gated_answer_rotated, c->gated_answer, QW, n);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT, c->gated_answer, c->gated_answer_rotated,
                                                                answer, H, QW);

    /* ⑥ at a block's first layer every row's stream joins its block states; the answers through their convolution into
     *   the residual, and the experts' mix over each position's own states, over every row */
    const bool block_start = s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_START] == 1u;
    if (block_start)
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)(states + 2u * s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_ROW] * H),
                            (const uint16_t*)(uintptr_t)(stream), (const uint16_t*)(uintptr_t)(ones), H, n, H, SH, s->epsilon, s->over);
    doors->convolution_steps((uint16_t*)(uintptr_t)(c->convolved), (uint16_t*)(uintptr_t)residual,
                             (uint16_t*)(uintptr_t)(s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__ATTENTION_HISTORY]),
                             (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(ATTENTION_TAPS)), 0, (const uint16_t*)(uintptr_t)answer,
                             (const uint16_t*)(uintptr_t)(block_start ? 0u : stream), H, s->taps, 0u,
                             s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__CONVOLUTION_SCALE], n, 0u, s->over);
    doors->attention_states_rows((uint16_t*)(uintptr_t)(c->experts_mix), (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(EXPERTS_QUERY)),
                                 (const uint16_t*)(uintptr_t)states, (const uint16_t*)(uintptr_t)residual, 1u, H,
                                 s->layer_v[AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_MIXED], 1u, s->epsilon, n, SH, H, H, s->over);

    /* ⑦ the experts' gate, norm and rotation, and the router, over every row: every row's input into its hand, its
     *   eight and their weights after it */
    uint64_t read = c->experts_mix;
    if (messages) {
        ai_thinkingmachines__inkling_2__layer__zzprivate_gate_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLERS + 3u * AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLER,
                                                                   c->experts_mix, H, told + 2u * 3u * H, 4u * H, c->experts_input);
        read = c->experts_input;
    }
    ai_thinkingmachines__inkling_2__layer__zzprivate_norm_rotate_rows(s, n, read, H, AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(EXPERTS_NORM), c->normed, c->rotated);
    ai_thinkingmachines__inkling_2__layer__zzprivate_times_rows(s, c, AI_THINKINGMACHINES__INKLING_2__LAYER__ROUTER, c->normed, c->rotated, c->logits, E, H);
    doors->vector_copy_rows((uint16_t*)(uintptr_t)(hands), c->hand_row / 2u, (const uint16_t*)(uintptr_t)(c->rotated), H, H, n);
    doors->vector_top_k_biased_rows((uint64_t*)(uintptr_t)(hands + AI_THINKINGMACHINES__INKLING_2__HAND__PICKS(H)), 1u, c->hand_row / 8u,
                                    (uint16_t*)(uintptr_t)(hands + 2u * H), c->hand_row / 2u, (const uint16_t*)(uintptr_t)(c->logits),
                                    (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(ROUTER_BIAS)), E, K,
                                    s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__ROUTE_SCALE], n);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_thinkingmachines__inkling_2__after_experts_rows layer application work routed n)` -> `routed`: each of the chunk's
 * rows' experts' sum — `routed`'s rows — through its short convolution, added to its residual: its stream. */
static sys__heap_node ai_thinkingmachines__inkling_2__after_experts_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t routed = 0, room = 0;
    if (!nn__primitives__room(&argv[3], &routed, &room) || argv[4].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_thinkingmachines__inkling_2__layer__site site;
    ai_thinkingmachines__inkling_2__layer__site* s = &site;
    const uint64_t fault = ai_thinkingmachines__inkling_2__layer__zzprivate_read(argv, ctx, s);
    if (fault != 0u) return sys__engine__abi__error(fault);
    const uint64_t n = argv[4].args[0], H = s->hidden;
    if (!ai_thinkingmachines__inkling_2__layer__zzprivate_holds(s, n) || !nn__primitives__fits(n * H, room))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE);
    ai_thinkingmachines__inkling_2__layer__chunk chunk;
    if (!ai_thinkingmachines__inkling_2__layer__zzprivate_chunk(s, 0u, n, &chunk))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE);
    s->doors->convolution_steps((uint16_t*)(uintptr_t)(chunk.convolved), (uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(STREAM)),
                                (uint16_t*)(uintptr_t)(s->own[AI_THINKINGMACHINES__INKLING_2__APPLICATION__EXPERTS_HISTORY]),
                                (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L(EXPERTS_TAPS)), 0, (const uint16_t*)(uintptr_t)routed,
                                (const uint16_t*)(uintptr_t)(AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W(RESIDUAL)), H, s->taps, 0u,
                                s->work_r[AI_THINKINGMACHINES__INKLING_2__WORK__CONVOLUTION_SCALE], n, 0u, s->over);
    return nn__doors_answer(&argv[3]);
}

#undef AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_W
#undef AI_THINKINGMACHINES__INKLING_2__LAYER__ZZPRIVATE_L

SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__before_experts__zzabi_adapter, ai_thinkingmachines__inkling_2__before_experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__after_experts__zzabi_adapter, ai_thinkingmachines__inkling_2__after_experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__before_experts_rows__zzabi_adapter, ai_thinkingmachines__inkling_2__before_experts_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__after_experts_rows__zzabi_adapter, ai_thinkingmachines__inkling_2__after_experts_rows__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_LAYER__ABI_CUH */
