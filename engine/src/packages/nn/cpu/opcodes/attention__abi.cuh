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

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_ATTENTION__ABI_CUH */
