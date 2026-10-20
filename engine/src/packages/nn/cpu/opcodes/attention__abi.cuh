#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_ATTENTION__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_ATTENTION__ABI_CUH
/* ══ RoPE AND ATTENTION, ON THE PROGRAM'S SIDE ═══════════════════════════════════════════════════════
 * One position at a time: the rotation of every head, and one query's attention over the positions a
 * cache holds. The checks are here; the arithmetic is the doors'. ▶ `vector__abi.cuh` for the shape and
 * the refusals, and `gpu/kernels/kernels.cuh` for the bodies.
 *
 * ⚖ PROPOSED FOR THE ARCHITECT — RoPE, attention and the KV cache were his to design, and each
 * choice here is the simplest one that is correct:
 *   P1  THE CACHE IS TWO fp16 PLANES A LAYER, `[position][kv_head][head_dim]`, whatever holds them — the
 *       7B's driver keeps them as one more slot type in the weight arena. The cartridge's tiers (fp32 hot
 *       window, TurboQuant tail) are not used by these verbs.
 *   P2  THE CACHE HAS NO WRITER VERB. A program points `v_proj`'s output, and RoPE's output for `k`, at
 *       the cache's row for the position (`nn__vector__range`), so each lands where it is kept.
 *   P3  COS AND SIN ARE A BUFFER, filled once a position by `nn__rope__angles` and read by every layer's
 *       `nn__rope__apply`, as HF's rotary computes them once a forward. ⚖ *"yes add cos and sin"* — the
 *       seam has `cosf`/`sinf`.
 *   P4  ONE QUERY AT A TIME, OVER A RANGE OF KEYS `[from, to)` — ⚖ ruled the range. A prompt is fed a
 *       position at a time; queries over a range of positions (a prefill) come with batched projections.
 *       Today a program attends `0` to the current token; several ranges are one `residual` call each,
 *       merged — ▶ RESIDUALS below. */

#include "../doors.cuh"
#include "../../contracts/objects/attention.cuh"

/* The most a head count, a head width or a length may be. Past it a product of three could wrap, and
 * no model has a dimension near it. */
#define NN__ATTENTION__DIM_MAX (1ull << 20)

static inline bool nn__attention__zzprivate_dim(const sys__heap_node* v, uint64_t* out) {
    if (v->dtype != SYS__KIND__VALUE_INT || v->args[0] == 0ull || v->args[0] > NN__ATTENTION__DIM_MAX) return false;
    *out = v->args[0];
    return true;
}

/* `(nn__rope__apply x cs out heads head_dim)` — every head of `x` rotated at the position `cs` holds:
 * its `head_dim/2` cosines then its `head_dim/2` sines, in fp32. `out` may be `x`. */
static sys__heap_node nn__rope__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, c_at = 0, o_at = 0, x_room = 0, c_room = 0, o_room = 0, heads = 0, head_dim = 0;
    if (!nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &c_at, &c_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)
     || !nn__attention__zzprivate_dim(&argv[3], &heads)
     || !nn__attention__zzprivate_dim(&argv[4], &head_dim)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    /* a head is pairs, and `cs` is `head_dim` floats — twice as many halves */
    if (head_dim % 2ull != 0ull || !nn__primitives__fits(heads * head_dim, x_room)
     || !nn__primitives__fits(heads * head_dim, o_room)
     || !nn__primitives__fits(2ull * head_dim, c_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->rope((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, (const float*)(uintptr_t)c_at, heads, head_dim,
                ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__rope__interleaved x cs out heads head_stride rot inverse)` — the first `rot` halves of each of `heads` heads, a head
 * `head_stride` halves on, turned on adjacent pairs at the position `cs` holds (`rot/2` cosines then `rot/2` sines, fp32);
 * `inverse` 1 turns them back. `x` and `out` point at head 0's first turned half — a head's tail is reached through a range
 * — and `out` may be `x`. */
static sys__heap_node nn__rope__zzabi_apply_interleaved(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, c_at = 0, o_at = 0, x_room = 0, c_room = 0, o_room = 0, heads = 0, stride = 0, rot = 0;
    if (!nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &c_at, &c_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)
     || !nn__attention__zzprivate_dim(&argv[3], &heads)
     || !nn__attention__zzprivate_dim(&argv[4], &stride)
     || !nn__attention__zzprivate_dim(&argv[5], &rot)
     || argv[6].dtype != SYS__KIND__VALUE_INT || argv[6].args[0] > 1ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t span = (heads - 1ull) * stride + rot;
    if (rot % 2ull != 0ull || rot > stride || !nn__primitives__fits(span, x_room) || !nn__primitives__fits(span, o_room)
     || !nn__primitives__fits(2ull * rot, c_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->rope_interleaved((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, (const float*)(uintptr_t)c_at, heads, stride,
                            rot, argv[6].args[0], ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__attention__softmax_sink p sink rows length)` — each of `rows` rows of `length` scores (fp32) made weights, its sum
 * taking one more term `exp(sink[r])` (`sink` rows halves): DeepSeek-V4's attention sink. In place. */
static sys__heap_node nn__attention__zzabi_apply_softmax_sink(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t p_at = 0, p_room = 0, s_at = 0, s_room = 0, rows = 0, length = 0;
    if (!nn__primitives__room(&argv[0], &p_at, &p_room) || !nn__primitives__room(&argv[1], &s_at, &s_room)
     || !nn__attention__zzprivate_dim(&argv[2], &rows) || !nn__attention__zzprivate_dim(&argv[3], &length))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__primitives__fits(2ull * rows * length, p_room) || !nn__primitives__fits(rows, s_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->attention_softmax_sink((float*)(uintptr_t)p_at, (const uint16_t*)(uintptr_t)s_at, rows, length);
    return nn__doors_answer(&argv[0]);
}

/* `(nn__rope__angles cs position theta head_dim)` — the cosines and sines of one position into `cs`, fp32:
 * `head_dim/2` cosines then `head_dim/2` sines, which is what `nn__rope__apply` reads. One call a position
 * serves every layer, as HF's rotary does once a forward. `theta` is an integer or a float — every base in
 * use is a whole number, and a program has no float literal to write one with. */
static sys__heap_node nn__rope__zzabi_apply_angles(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t c_at = 0, c_room = 0, head_dim = 0;
    if (argv[1].dtype != SYS__KIND__VALUE_INT
     || (argv[2].dtype != SYS__KIND__VALUE_INT && argv[2].dtype != SYS__KIND__VALUE_FLOAT)
     || !nn__primitives__room(&argv[0], &c_at, &c_room)
     || !nn__attention__zzprivate_dim(&argv[3], &head_dim)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const float theta = argv[2].dtype == SYS__KIND__VALUE_INT ? (float)argv[2].args[0]
                                                              : (float)sys__heap_node__real(argv[2].args[0]);
    if (head_dim % 2ull != 0ull || !(theta > 1.0f)
     || !nn__primitives__fits(2ull * head_dim, c_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->rope_angles((float*)(uintptr_t)c_at, argv[1].args[0], theta, head_dim);
    return nn__doors_answer(&argv[0]);
}

/* `(nn__attention__decode q k v scores out q_heads kv_heads head_dim from to)` — one query's attention over
 * positions `from` up to but not including `to` of a cache. `k` and `v` are `[position][kv_head][head_dim]`;
 * query head `h` reads kv head `h / (q_heads / kv_heads)`. `scores` is `q_heads * (to - from)` floats the
 * two doors pass between them — each head's weights, which the first writes and the second reads.
 * ⭐ THE RANGE IS WHAT LETS A SENTENCE BE A RANGE OF ONE CACHE — ⚖ *"by having a from-to method we can do
 *   the sentences without having to make it into the pages"*. A sentence is `[start, end)` of the token
 *   axis, and attending to it is attending to that range; nothing is copied or paged to make it one.
 * ⛳ AND IT COSTS THE DOORS NOTHING: the window is where `k` and `v` start, so the doors are handed the
 *   cache from row `from` and a length, exactly as for a cache that begins there. */
static sys__heap_node nn__attention__zzabi_apply_decode(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 10u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t q_at = 0, k_at = 0, v_at = 0, p_at = 0, o_at = 0, q_room = 0, k_room = 0, v_room = 0, p_room = 0, o_room = 0;
    uint64_t q_heads = 0, kv_heads = 0, head_dim = 0, to = 0;
    if (!nn__primitives__room(&argv[0], &q_at, &q_room)
     || !nn__primitives__room(&argv[1], &k_at, &k_room)
     || !nn__primitives__room(&argv[2], &v_at, &v_room)
     || !nn__primitives__room(&argv[3], &p_at, &p_room)
     || !nn__primitives__room(&argv[4], &o_at, &o_room)
     || !nn__attention__zzprivate_dim(&argv[5], &q_heads)
     || !nn__attention__zzprivate_dim(&argv[6], &kv_heads)
     || !nn__attention__zzprivate_dim(&argv[7], &head_dim)
     || argv[8].dtype != SYS__KIND__VALUE_INT
     || !nn__attention__zzprivate_dim(&argv[9], &to)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    /* an empty range has no softmax, so `from` must be below `to` */
    const uint64_t from = argv[8].args[0], row = kv_heads * head_dim;
    if (from >= to) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const uint64_t length = to - from, cached = to * row;
    if (q_heads % kv_heads != 0ull
     || !nn__primitives__fits(q_heads * head_dim, q_room) || !nn__primitives__fits(q_heads * head_dim, o_room)
     || !nn__primitives__fits(cached, k_room) || !nn__primitives__fits(cached, v_room)
     || !nn__primitives__fits(2ull * q_heads * length, p_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t skip = from * row * NN__PRIMITIVES__ELEMENT_BYTES;
    doors->attention_scores((float*)(uintptr_t)p_at, (const uint16_t*)(uintptr_t)q_at,
                            (const uint16_t*)(uintptr_t)(k_at + skip), q_heads, kv_heads, head_dim, length);
    doors->attention_mix((uint16_t*)(uintptr_t)o_at, (const float*)(uintptr_t)p_at,
                         (const uint16_t*)(uintptr_t)(v_at + skip), q_heads, kv_heads, head_dim, length, ctx->fault_word);
    return nn__doors_answer(&argv[4]);
}

/* ── RESIDUALS — SEVERAL RANGES, ONE SOFTMAX ────────────────────────────────────────────────────────
 * ⚖ *"in the future we can add residuals"*. A sentence evicted from the middle of a conversation leaves
 * two ranges, and each is one call: `residual` over a range, `merge` to fold two residuals into one, and
 * `finish` for the answer. ▶ `gpu/kernels/kernels.cuh` for what a residual holds and why the merge is
 * exact. A residual is `q_heads * (head_dim + 2)` floats. */
static inline bool nn__attention__zzprivate_residual_room(const sys__heap_node* v, uint64_t q_heads, uint64_t head_dim,
                                                          uint64_t* at) {
    uint64_t room = 0;
    return nn__primitives__room(v, at, &room)
        && nn__primitives__fits(2ull * q_heads * NN__ATTENTION__RESIDUAL_FLOATS(head_dim), room);
}

/* `(nn__attention__residual q k v scores res q_heads kv_heads head_dim from to)` — the residual of one query
 * over positions `from` up to but not including `to`, into `res`. The operands and refusals are
 * `nn__attention__decode`'s, with `res` where that verb has `out`. */
static sys__heap_node nn__attention__zzabi_apply_residual(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 10u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t q_at = 0, k_at = 0, v_at = 0, p_at = 0, r_at = 0, q_room = 0, k_room = 0, v_room = 0, p_room = 0, r_room = 0;
    uint64_t q_heads = 0, kv_heads = 0, head_dim = 0, to = 0;
    if (!nn__primitives__room(&argv[0], &q_at, &q_room)
     || !nn__primitives__room(&argv[1], &k_at, &k_room)
     || !nn__primitives__room(&argv[2], &v_at, &v_room)
     || !nn__primitives__room(&argv[3], &p_at, &p_room)
     || !nn__primitives__room(&argv[4], &r_at, &r_room)
     || !nn__attention__zzprivate_dim(&argv[5], &q_heads)
     || !nn__attention__zzprivate_dim(&argv[6], &kv_heads)
     || !nn__attention__zzprivate_dim(&argv[7], &head_dim)
     || argv[8].dtype != SYS__KIND__VALUE_INT
     || !nn__attention__zzprivate_dim(&argv[9], &to)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t from = argv[8].args[0], row = kv_heads * head_dim;
    if (from >= to) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const uint64_t length = to - from, cached = to * row;
    if (q_heads % kv_heads != 0ull || !nn__primitives__fits(q_heads * head_dim, q_room)
     || !nn__primitives__fits(cached, k_room) || !nn__primitives__fits(cached, v_room)
     || !nn__primitives__fits(2ull * q_heads * length, p_room)
     || !nn__primitives__fits(2ull * q_heads * NN__ATTENTION__RESIDUAL_FLOATS(head_dim), r_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t skip = from * row * NN__PRIMITIVES__ELEMENT_BYTES;
    doors->attention_weights((float*)(uintptr_t)p_at, (float*)(uintptr_t)r_at, (const uint16_t*)(uintptr_t)q_at,
                             (const uint16_t*)(uintptr_t)(k_at + skip), q_heads, kv_heads, head_dim, length);
    doors->attention_residual_mix((float*)(uintptr_t)r_at, (const float*)(uintptr_t)p_at,
                                  (const uint16_t*)(uintptr_t)(v_at + skip), q_heads, kv_heads, head_dim, length);
    return nn__doors_answer(&argv[4]);
}

/* `(nn__attention__merge a b into q_heads head_dim)` — two residuals as one, into `into`, which may be
 * `a` or `b`; answers `into`, so merges chain. */
static sys__heap_node nn__attention__zzabi_apply_merge(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t q_heads = 0, head_dim = 0, a_at = 0, b_at = 0, i_at = 0;
    if (!nn__attention__zzprivate_dim(&argv[3], &q_heads)
     || !nn__attention__zzprivate_dim(&argv[4], &head_dim)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__attention__zzprivate_residual_room(&argv[0], q_heads, head_dim, &a_at)
     || !nn__attention__zzprivate_residual_room(&argv[1], q_heads, head_dim, &b_at)
     || !nn__attention__zzprivate_residual_room(&argv[2], q_heads, head_dim, &i_at))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->attention_merge((float*)(uintptr_t)i_at, (const float*)(uintptr_t)a_at, (const float*)(uintptr_t)b_at,
                           q_heads, head_dim);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__attention__finish res out q_heads head_dim)` — the attention a residual stands for, as halves. */
static sys__heap_node nn__attention__zzabi_apply_finish(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t q_heads = 0, head_dim = 0, r_at = 0, o_at = 0, o_room = 0;
    if (!nn__primitives__room(&argv[1], &o_at, &o_room)
     || !nn__attention__zzprivate_dim(&argv[2], &q_heads)
     || !nn__attention__zzprivate_dim(&argv[3], &head_dim)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__attention__zzprivate_residual_room(&argv[0], q_heads, head_dim, &r_at)
     || !nn__primitives__fits(q_heads * head_dim, o_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->attention_finish((uint16_t*)(uintptr_t)o_at, (const float*)(uintptr_t)r_at, q_heads, head_dim, ctx->fault_word);
    return nn__doors_answer(&argv[1]);
}

/* `(nn__attention__residual_tq q kcodes kscales vcodes vscales scores res q_heads kv_heads head_dim d count)` — the
 * residual of one query over `count` positions of a TurboQuant tier of the cache: key and value rows `t · kv_heads + g`,
 * each `head_dim` wide at width `d`, both rotated as `q` is. `res` as `nn__attention__residual` leaves it, so the
 * tiers' residuals merge as any other. */
static sys__heap_node nn__attention__zzabi_apply_residual_tq(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 12u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at[7] = {0}, room[7] = {0};
    for (unsigned i = 0u; i < 7u; ++i)
        if (!nn__primitives__room(&argv[i], &at[i], &room[i])) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t q_heads = 0, kv_heads = 0, head_dim = 0, d = 0, count = 0;
    if (!nn__attention__zzprivate_dim(&argv[7], &q_heads) || !nn__attention__zzprivate_dim(&argv[8], &kv_heads)
     || !nn__attention__zzprivate_dim(&argv[9], &head_dim) || !nn__attention__zzprivate_dim(&argv[10], &d)
     || !nn__attention__zzprivate_dim(&argv[11], &count)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if ((d != 4ull && d != 8ull) || q_heads % kv_heads != 0ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = count * kv_heads, code_bytes = rows * nn__turboquant__row_bytes(d, head_dim);
    if (!nn__primitives__fits(q_heads * head_dim, room[0]) || code_bytes > room[1] || !nn__primitives__fits(rows, room[2])
     || code_bytes > room[3] || !nn__primitives__fits(rows, room[4]) || !nn__primitives__fits(2ull * q_heads * count, room[5])
     || !nn__primitives__fits(2ull * q_heads * NN__ATTENTION__RESIDUAL_FLOATS(head_dim), room[6]))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->attention_weights_tq((float*)(uintptr_t)at[5], (float*)(uintptr_t)at[6], (const uint16_t*)(uintptr_t)at[0],
                                (const uint8_t*)(uintptr_t)at[1], (const uint16_t*)(uintptr_t)at[2], q_heads, kv_heads, head_dim,
                                count, d);
    doors->attention_residual_mix_tq((float*)(uintptr_t)at[6], (const float*)(uintptr_t)at[5], (const uint8_t*)(uintptr_t)at[3],
                                     (const uint16_t*)(uintptr_t)at[4], q_heads, kv_heads, head_dim, count, d);
    return nn__doors_answer(&argv[6]);
}

SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_residual_tq, nn__attention__zzabi_apply_residual_tq)
SYS__ENGINE__ABI__BRIDGE(nn__rope__zzabi_adapter,             nn__rope__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_decode, nn__attention__zzabi_apply_decode)
SYS__ENGINE__ABI__BRIDGE(nn__rope__zzabi_adapter_angles,      nn__rope__zzabi_apply_angles)
SYS__ENGINE__ABI__BRIDGE(nn__rope__zzabi_adapter_interleaved, nn__rope__zzabi_apply_interleaved)
SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_softmax_sink, nn__attention__zzabi_apply_softmax_sink)
SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_residual, nn__attention__zzabi_apply_residual)
SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_merge,    nn__attention__zzabi_apply_merge)
SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_finish,   nn__attention__zzabi_apply_finish)

/* ── MULTI-HEAD LATENT ATTENTION ────────────────────────────────────────────────────────────────────────
 * The cache holds one latent a position, `latent` halves, and it is both the keys and the values: a head's key is
 * its key rows of `W` times the latent, its value its value rows times it. So a query is absorbed into the latent
 * first, `nn__attention__decode` attends with every head over the latent as ONE kv head, and the answer — a latent a
 * head — is expanded by the head's value rows. `W` is `heads · stride` rows of `latent` halves: a head's key rows
 * from its first, its value rows from `offset`. */

/* `(nn__attention__absorb q W out heads nope latent stride scale)` — each head's `nope` query halves absorbed into the
 * latent: `out[h] = scale · W_hᵀ q_h`, `latent` halves a head. `scale` is a float; the attention's own `1/√latent`
 * and the model's `1/√nope` differ by it. */
static sys__heap_node nn__attention__zzabi_apply_absorb(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 8u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at[3] = {0}, room[3] = {0}, heads = 0, nope = 0, latent = 0, stride = 0;
    if (argv[7].dtype != SYS__KIND__VALUE_FLOAT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    for (unsigned i = 0u; i < 3u; ++i)
        if (!nn__primitives__room(&argv[i], &at[i], &room[i])) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__attention__zzprivate_dim(&argv[3], &heads) || !nn__attention__zzprivate_dim(&argv[4], &nope)
     || !nn__attention__zzprivate_dim(&argv[5], &latent) || !nn__attention__zzprivate_dim(&argv[6], &stride))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (nope > stride || !nn__primitives__fits(heads * nope, room[0]) || !nn__primitives__fits(heads * stride * latent, room[1])
     || !nn__primitives__fits(heads * latent, room[2])) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->attention_absorb((uint16_t*)(uintptr_t)at[2], (const uint16_t*)(uintptr_t)at[0], (const uint16_t*)(uintptr_t)at[1],
                            heads, nope, latent, stride, (float)sys__heap_node__real(argv[7].args[0]), ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__attention__expand o W out heads width latent stride offset)` — each head's latent answer expanded by its
 * `width` value rows, those from row `offset` of its `stride`: `out[h] = W_h,v · o_h`, `width` halves a head. */
static sys__heap_node nn__attention__zzabi_apply_expand(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 8u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at[3] = {0}, room[3] = {0}, heads = 0, width = 0, latent = 0, stride = 0;
    for (unsigned i = 0u; i < 3u; ++i)
        if (!nn__primitives__room(&argv[i], &at[i], &room[i])) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__attention__zzprivate_dim(&argv[3], &heads) || !nn__attention__zzprivate_dim(&argv[4], &width)
     || !nn__attention__zzprivate_dim(&argv[5], &latent) || !nn__attention__zzprivate_dim(&argv[6], &stride)
     || argv[7].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t offset = argv[7].args[0];
    if (offset > stride || width > stride - offset || !nn__primitives__fits(heads * latent, room[0])
     || !nn__primitives__fits(heads * stride * latent, room[1]) || !nn__primitives__fits(heads * width, room[2]))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->attention_expand((uint16_t*)(uintptr_t)at[2], (const uint16_t*)(uintptr_t)at[0], (const uint16_t*)(uintptr_t)at[1],
                            heads, width, latent, stride, offset, ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_absorb, nn__attention__zzabi_apply_absorb)
SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_expand, nn__attention__zzabi_apply_expand)

/* ══ ⭐ A PROMPT'S ROWS THROUGH A GROUPED-QUERY ATTENTION — nn's word over a plane table (▶ `contracts/objects/attention.cuh`) ══
 * `(nn__attention__rows xs table first h1s rows)` -> `h1s`: positions `first .. first + rows`, a row of `xs` each — the
 * norm, the queries and keys in one grouped launch and the values straight into their cache rows, RoPE on the queries and
 * on the keys into theirs, every key and value of the chunk written before any query reads them, row `r` attending the
 * cache's rows `0 .. first + r` — as many rows at a time as the scores' room holds — the answers rotated and through the
 * out projection, the residual added. Each step is
 * the many-row form of the one a model's one-position program takes (Ministral's), so the two agree as those doors do.
 * The scratch starts with the two lists of pairs (each row onto itself; each row onto its position), then in halves
 * h · hrot · [q · k] · att · attr · ao. */
static sys__heap_node nn__attention__zzabi_apply_rows(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, h1 = 0, h1_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[3], &h1, &h1_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[NN__ATTENTION__TABLE], room[NN__ATTENTION__TABLE], v[NN__ATTENTION__TABLE];
    float r[NN__ATTENTION__TABLE];
    if (!nn__primitives__table(argv[1].args[0], NN__ATTENTION__HIDDEN, NN__ATTENTION__EPS, NN__ATTENTION__TABLE, at, room, v, r))
        return sys__engine__abi__error(NN__ATTENTION__FAULT_TABLE);
    const uint64_t H = v[NN__ATTENTION__HIDDEN], qh = v[NN__ATTENTION__Q_HEADS], kvh = v[NN__ATTENTION__KV_HEADS];
    const uint64_t hd = v[NN__ATTENTION__HEAD_DIM], first = argv[2].args[0], T = argv[4].args[0];
    const uint64_t qw = qh * hd, kvw = kvh * hd, to = first + T;
    /* the residual and the heads' widths whole blocks of 512, so a rotation over the rows is each row's */
    if (H == 0u || qh == 0u || kvh == 0u || hd == 0u || hd % 2u != 0u || qh % kvh != 0u || H > (1ull << 24)
     || qh > (1ull << 12) || hd > (1ull << 12) || H % 512u != 0u || qw % 512u != 0u || first >= (1ull << 32)
     || T == 0u || T > NN__ATTENTION__ROWS_MAX || !nn__primitives__fits(T * H, x_room) || !nn__primitives__fits(T * H, h1_room))
        return sys__engine__abi__error(NN__ATTENTION__FAULT_TABLE);
    if (!nn__primitives__fits(H, room[NN__ATTENTION__NORM]) || !nn__primitives__fits(to * kvw, room[NN__ATTENTION__KEYS])
     || !nn__primitives__fits(to * kvw, room[NN__ATTENTION__VALUES]) || room[NN__ATTENTION__ANGLES_Q] / 4u < to * hd
     || room[NN__ATTENTION__ANGLES_K] / 4u < to * hd || room[NN__ATTENTION__SCORES] / 4u < qh * to
     || !nn__primitives__fits(512u, room[NN__ATTENTION__SIGNS]))
        return sys__engine__abi__error(NN__ATTENTION__FAULT_ROOM);
    /* the rows attend as many at a time as the scores' room holds — the last of them spans the most — at least one */
    const uint64_t pass = room[NN__ATTENTION__SCORES] / 4u / (qh * to) < T ? room[NN__ATTENTION__SCORES] / 4u / (qh * to) : T;
    const uint64_t pairs = at[NN__ATTENTION__SCRATCH], vpairs = pairs + 8u * NN__ATTENTION__ROWS_MAX, h = vpairs + 8u * NN__ATTENTION__ROWS_MAX;
    const uint64_t hrot = h + 2u * T * H, q = hrot + 2u * T * H, k = q + 2u * T * qw, att = k + 2u * T * kvw;
    const uint64_t attr = att + 2u * T * qw, ao = attr + 2u * T * qw;
    if (room[NN__ATTENTION__SCRATCH] < ao + 2u * T * H - pairs) return sys__engine__abi__error(NN__ATTENTION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    /* row j's input and output both row j; and row j's value onto the cache's row `first + j` */
    uint32_t each[2u * NN__ATTENTION__ROWS_MAX], onto[2u * NN__ATTENTION__ROWS_MAX];
    for (uint64_t j = 0u; j < T; ++j) {
        each[2u * j] = (uint32_t)j; each[2u * j + 1u] = (uint32_t)j;
        onto[2u * j] = (uint32_t)j; onto[2u * j + 1u] = (uint32_t)(first + j);
    }
    if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)pairs, each, (size_t)(8u * T))
     || !sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)vpairs, onto, (size_t)(8u * T)))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[NN__ATTENTION__SIGNS], keys = at[NN__ATTENTION__KEYS], values = at[NN__ATTENTION__VALUES];
    const uint64_t kfirst = keys + 2u * first * kvw;

    doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[NN__ATTENTION__NORM],
                        H, T, H, H, r[NN__ATTENTION__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, T * H, over);
    nn__expert__groups gr = {};
    gr.codes[0] = at[NN__ATTENTION__Q]; gr.luts[0] = at[NN__ATTENTION__Q_SCALES]; gr.out_rows[0] = qw;  gr.d[0] = v[NN__ATTENTION__D_Q];
    gr.codes[1] = at[NN__ATTENTION__K]; gr.luts[1] = at[NN__ATTENTION__K_SCALES]; gr.out_rows[1] = kvw; gr.d[1] = v[NN__ATTENTION__D_K];
    gr.out_at[0] = 0u; gr.out_at[1] = T * qw;
    gr.pairs[0] = gr.pairs[1] = T;
    gr.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)q, gr, (const uint16_t*)(uintptr_t)hrot, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->expert_rows((uint16_t*)(uintptr_t)values, (const uint8_t*)(uintptr_t)at[NN__ATTENTION__V], room[NN__ATTENTION__V],
                       (const uint8_t*)(uintptr_t)at[NN__ATTENTION__V_SCALES], room[NN__ATTENTION__V_SCALES], (const uint16_t*)(uintptr_t)hrot,
                       (const uint32_t*)(uintptr_t)vpairs, T, v[NN__ATTENTION__D_V], kvw, H, over);
    doors->rope_rows((uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)q, (const float*)(uintptr_t)(at[NN__ATTENTION__ANGLES_Q] + 4u * first * hd),
                     T, qh, hd, qw, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)kfirst, (const uint16_t*)(uintptr_t)k, (const float*)(uintptr_t)(at[NN__ATTENTION__ANGLES_K] + 4u * first * hd),
                     T, kvh, hd, kvw, hd, over);
    for (uint64_t r0 = 0u; r0 < T; r0 += pass) {
        const uint64_t n = T - r0 < pass ? T - r0 : pass;
        doors->attention_causal_scores_grouped((float*)(uintptr_t)at[NN__ATTENTION__SCORES], (const uint16_t*)(uintptr_t)(q + 2u * r0 * qw),
                                               (const uint16_t*)(uintptr_t)keys, qh, kvh, hd, first + r0, n);
        doors->attention_causal_softmax((float*)(uintptr_t)at[NN__ATTENTION__SCORES], qh, first + r0, n);
        doors->attention_causal_mix((uint16_t*)(uintptr_t)(att + 2u * r0 * qw), (const float*)(uintptr_t)at[NN__ATTENTION__SCORES],
                                    (const uint16_t*)(uintptr_t)values, qh, kvh, hd, first + r0, n, over);
    }
    doors->hadamard_rotate((uint16_t*)(uintptr_t)attr, (const uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)signs, T * qw, over);
    doors->expert_rows((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[NN__ATTENTION__O], room[NN__ATTENTION__O],
                       (const uint8_t*)(uintptr_t)at[NN__ATTENTION__O_SCALES], room[NN__ATTENTION__O_SCALES], (const uint16_t*)(uintptr_t)attr,
                       (const uint32_t*)(uintptr_t)pairs, T, v[NN__ATTENTION__D_O], H, qw, over);
    doors->vector_add((uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)ao, T * H, over);
    return nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(nn__attention__zzabi_adapter_rows, nn__attention__zzabi_apply_rows)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_ATTENTION__ABI_CUH */
