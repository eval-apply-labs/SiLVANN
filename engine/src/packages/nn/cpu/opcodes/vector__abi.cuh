#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_VECTOR__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_VECTOR__ABI_CUH
/* ══ THE ADAPTER'S BODY. ▶ `vector__abi__header.cuh` for what it is and why it is scaffolding. ══════ */

#include "../doors.cuh"   /* the doors this verb launches through, on the worker's silicon */

/* ── `out[i] = a[i] + b[i]` ──────────────────────────────────────────────────────────────────────────
 * ⛳ ALIASING IS ALLOWED AND IS THE POINT: passing the same buffer as `a` and `out` makes this the
 * residual accumulate the old op could only do, and passing three makes it the general add the old op
 * could not. Nothing here has to know which the caller meant. */
/* ══ `nn__vector__add`, ON THE PROGRAM'S SIDE ═══════════════════════════════════════════════════════════
 * The operands are checked here — the same bounds and the same `room()` reads as every nn verb — and the
 * arithmetic is the door's, on whichever device the calling worker bound. `(add x y out n)`.
 * ⛔ NO FAMILY FOR THE VERB'S SILICON, OR NO FAULT WORD, IS A REFUSAL AND NOT A CRASH: the door would
 * write an overflow through that word on the device, and a program run where no worker has bound a card
 * must learn that from an error value it can read. */
static sys__heap_node nn__vector__zzabi_apply_add(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, x_room = 0, y_at = 0, y_room = 0, o_at = 0, o_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &y_at, &y_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    /* ⛳ n = 0 IS A SUM OVER NO TERMS — nothing to do, and not an error, as it never was. */
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, y_room)
     || !nn__primitives__fits(n, o_room)) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->vector_add((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, (const uint16_t*)(uintptr_t)y_at, n,
                      ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* ── `(nn__vector__zero v n)` — the one fill that needs no width ───────────────────────────────────── */
/* `(nn__vector__zero v n)` — the one verb here with nothing to overflow, so it launches with no fault word. */
static sys__heap_node nn__vector__zzabi_apply_zero(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 2u) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_ARITY);
    uint64_t v_at = 0, v_room = 0;
    if (argv[1].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &v_at, &v_room))
        return nn__vector__zzabi_error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[1].args[0];
    if (!nn__primitives__fits(n, v_room)) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->vector_zero((uint16_t*)(uintptr_t)v_at, n);
    return nn__doors_answer(&argv[0]);
}

/* `(nn__vector__exp x out n)` and `(nn__vector__softplus x out n)` — one input, one output, the same
 * checks; only the door differs, so one body serves both. */
static inline sys__heap_node nn__vector__zzabi_zzprivate_unary(const sys__heap_node* argv, unsigned argc,
                                                               sys__engine__ctx* ctx, bool softplus) {
    if (argc != 3u) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, o_at = 0, x_room = 0, o_room = 0;
    if (argv[2].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &o_at, &o_room)) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[2].args[0];
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, o_room))
        return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    (softplus ? doors->vector_softplus : doors->vector_exp)((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, n,
                                                            ctx->fault_word);
    return nn__doors_answer(&argv[1]);
}
/* ── `out[i] = exp(x[i])` ────────────────────────────────────────────────────────────────────────── */
static sys__heap_node nn__vector__zzabi_apply_exp(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    return nn__vector__zzabi_zzprivate_unary(argv, argc, ctx, false);
}
/* ── `out[i] = log(1 + exp(x[i]))`, in the form that is total ───────────────────────────────────────
 * ⛳ `max(x,0) + log1p(exp(-|x|))`. For `x >= 0` that is `x + log(1 + e^-x)` and for `x < 0` it is
 * `log(1 + e^x)` — the same function on both arms, written so the exponent is never positive. */
static sys__heap_node nn__vector__zzabi_apply_softplus(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    return nn__vector__zzabi_zzprivate_unary(argv, argc, ctx, true);
}

/* ── `out[i] = x[i] * rsqrt(Σ x^2 + 1e-6)` ──────────────────────────────────────────────────────────
 * ⛔⛔ A **SUM**, NOT A MEAN — ▶ the header. `rmsnorm` twelve screens up divides by `n` and this does
 * not, and on a 128-wide head that is a factor of 11.3.
 * ⛳ TWO PASSES, AND THE ORDER IS LOAD-BEARING FOR THE SAME REASON `rmsnorm`'s IS: the caller aliases
 * `x` and `out`, so the sum must be complete before the first write. */
/* `(nn__vector__l2norm x out n)` — the same shape, and a length of zero refused: it has no norm. */
static sys__heap_node nn__vector__zzabi_apply_l2norm(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, o_at = 0, x_room = 0, o_room = 0;
    if (argv[2].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &o_at, &o_room)) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[2].args[0];
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, o_room))
        return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (n == 0ull) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->vector_l2norm((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, n, ctx->fault_word);
    return nn__doors_answer(&argv[1]);
}

/* ══ ⭐ THE ELEMENTWISE PRODUCT — `add`'s mirror, and the shared expert's gate ══════════════════════
 * ⛳ WRITTEN AS A SEPARATE FUNCTION RATHER THAN AS `add` WITH AN OPERATOR ARGUMENT. An operator
 * selector would be a value the dispatch has to branch on inside the loop, which is the register
 * pressure this package's own contract says a per-term opcode is trying to get out from under — and
 * the two bodies differ by one character. ⇒ ★ TWO SHORT FUNCTIONS BEAT ONE FUNCTION WITH A MODE. */
/* `(nn__vector__pointwise_mul a b out n)` */
static sys__heap_node nn__vector__zzabi_apply_pointwise_mul(const sys__heap_node* argv, unsigned argc,
                                                     sys__engine__ctx* ctx) {
    if (argc != 4u) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_ARITY);
    uint64_t a_at = 0, b_at = 0, o_at = 0, a_room = 0, b_room = 0, o_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &a_at, &a_room)
     || !nn__primitives__room(&argv[1], &b_at, &b_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    if (!nn__primitives__fits(n, a_room) || !nn__primitives__fits(n, b_room)
     || !nn__primitives__fits(n, o_room)) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->vector_pointwise_mul((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)a_at, (const uint16_t*)(uintptr_t)b_at, n,
                                ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* ══ ⭐⭐ SCALE BY A NUMBER THE PROGRAM KNOWS — the verb the float kind unlocked ═══════════════════
 * ⚖ *"get the scalar version too."* ⛳ A day ago this could not have been written: the language had no
 * way to spell a factor, which is why `_at` reads one out of a buffer. Both stay — one is for a factor
 * a program HAS, the other for one it must READ. ▶ the header. */
/* `(nn__vector__scale x factor out n)` — the factor is a float the program holds, so it crosses the door
 * as a value and not as a buffer. */
static sys__heap_node nn__vector__zzabi_apply_scale(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, o_at = 0, x_room = 0, o_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[1].dtype != SYS__KIND__VALUE_FLOAT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, o_room))
        return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->vector_scale((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, n,
                        (float)sys__heap_node__real(argv[1].args[0]), ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* ══ ⭐⭐ SCALE — a vector by one number, and the number comes from a buffer ════════════════════════
 * ⛳ THE FACTOR IS READ ONCE, BEFORE THE LOOP, which is what makes `alpha` safe to alias `out`: by the
 * time anything is written the scalar is already in a register. ⇒ ★ HOISTING A READ IS NOT ONLY A
 * SPEED CHOICE — IT IS WHAT MAKES AN ALIASING RULE STATEABLE. */
/* `(nn__vector__scale_at x scales which out n)` — the factor is element `which` of a buffer, so a loop over
 * experts scales by each one's score with nothing copied; that element is bounds-checked here. */
static sys__heap_node nn__vector__zzabi_apply_scale_at(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, a_at = 0, o_at = 0, x_room = 0, a_room = 0, o_room = 0;
    if (argv[4].dtype != SYS__KIND__VALUE_INT || argv[2].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &a_at, &a_room)
     || !nn__primitives__room(&argv[3], &o_at, &o_room)) return nn__vector__zzabi_error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[4].args[0];
    const uint64_t which = argv[2].args[0];
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, o_room))
        return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (!nn__primitives__fits(which + 1ull, a_room)) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return nn__vector__zzabi_error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->vector_scale_at((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, (const uint16_t*)(uintptr_t)a_at, which, n,
                           ctx->fault_word);
    return nn__doors_answer(&argv[3]);
}

/* The same bridge every `sys` verb goes through, so the operand limit, the refusal of a malformed form
 * and the hand-back of the verb's hold are written once, in `sys/verb_abi.cuh`, and this adapter cannot
 * drift from them. */
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_add,           nn__vector__zzabi_apply_add)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_zero,          nn__vector__zzabi_apply_zero)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_exp,           nn__vector__zzabi_apply_exp)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_softplus,      nn__vector__zzabi_apply_softplus)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_l2norm,        nn__vector__zzabi_apply_l2norm)
/* `(nn__vector__top_k x n k picks weights)` — the `k` largest of `n` values: their indices into the first
 * `k` elements of `picks`, a node array the program made, and their softmax into `weights` as halves;
 * answers `picks`. ⚖ *"i can allocate the return nodes beforehand and send the node array base address as
 * an object so the gpu has a base to compute its output slots"*.
 * ⛳ THE CARD WRITES THE NODES ITSELF when the boot registered the heap with it — `ctx->heap_card` — and
 *   the verb copies them in from a card buffer otherwise (OpenCL on the MI50 cannot register the heap).
 * ⛔ ONLY VALUE WORDS ARE WRITTEN, AND ONLY INTO ELEMENTS THAT HOLD NO REFERENCE: each of the `k` is asked
 *   its kind first, an int is written over, a null becomes an int here, and anything else is refused — so
 *   no hold is made or lost behind the array's back.
 * ⛳ IT WAITS FOR ITS KERNEL BEFORE ANSWERING, so a program reading `picks` next reads the answer. A
 *   stamped element a reader waits on would let it return at once; that is later work. */
/* The top-k's body, the plain softmax's or — `bias` not zero — the biased sigmoid router's: the picks into the node
 * array's value words (written on the card where the heap is registered with it, landed and copied otherwise), the
 * weights into `w`. */
static sys__heap_node nn__vector__zzprivate_top_k(const sys__heap_node* argv, sys__engine__ctx* ctx, uint64_t bias, float scale) {
    uint64_t x_at = 0, x_room = 0, w_at = 0, w_room = 0;
    if (argv[1].dtype != SYS__KIND__VALUE_INT || argv[2].dtype != SYS__KIND__VALUE_INT
     || !sys__heap_node__carries_reference(argv[3].dtype) || !sys__node_array__is(argv[3].args[0])
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[4], &w_at, &w_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[1].args[0], k = argv[2].args[0], picks = argv[3].args[0];
    /* ⛳ `n` STOPS AT 2^24 because the lowest index of a tie is found as the largest of negated indices in a
     *   float, which holds an integer exactly up to there. */
    if (n == 0ull || n > (1ull << 24) || k == 0ull || k > NN__VECTOR__TOP_K_MAX || k > n || (bias != 0ull && k > NN__VECTOR__ROUTER_K_MAX)
     || sys__node_array__length(picks) < k
     || !nn__primitives__fits(n, x_room) || !nn__primitives__fits(k, w_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t fault = nn__routed__top_k(ctx, doors, picks, w_at, x_at, bias, scale, n, k, 0);
    if (fault != 0u) return sys__engine__abi__error(fault);
    return nn__doors_answer(&argv[3]);
}

static sys__heap_node nn__vector__zzabi_apply_top_k(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    return nn__vector__zzprivate_top_k(argv, ctx, 0ull, 1.0f);
}

/* `(nn__vector__top_k_biased logits n k picks weights bias scale)` — GLM 5.3's and DeepSeek V3's router: the `k` experts
 * whose σ(logit) + bias is largest into `picks`, their σ(logit) — without the bias — normalised to one and times `scale`
 * into `weights`. The operands are `nn__vector__top_k`'s, with the bias (n halves) and the scale (a real) after them. */
static sys__heap_node nn__vector__zzabi_apply_top_k_biased(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t b_at = 0, b_room = 0;
    if (!nn__primitives__room(&argv[5], &b_at, &b_room) || argv[6].dtype != SYS__KIND__VALUE_FLOAT
     || argv[1].dtype != SYS__KIND__VALUE_INT || !nn__primitives__fits(argv[1].args[0], b_room) || b_at == 0ull)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return nn__vector__zzprivate_top_k(argv, ctx, b_at, (float)sys__heap_node__real(argv[6].args[0]));
}

/* `(nn__vector__penalize x n ids count penalty)` — the repetition penalty over the `n` logits of `x`, in place: each token
 * among the first `count` elements of `ids`, a node array of integers, has its logit divided by `penalty` when positive
 * and multiplied by it otherwise, once however often it appears. Answers `x`; a `count` of 0 runs nothing.
 * ⛳ THE CARD READS THE IDS FROM THE NODES THEMSELVES where the heap is registered with it, as `top_k` writes them, and
 *   from a copy in a card buffer otherwise — whose release waits for the kernel. */
static sys__heap_node nn__vector__zzabi_apply_penalize(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, x_room = 0;
    if (argv[1].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_FLOAT
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || !nn__primitives__room(&argv[0], &x_at, &x_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[1].args[0], ids = argv[2].args[0], count = argv[3].args[0];
    const float penalty = (float)sys__heap_node__real(argv[4].args[0]);
    if (n == 0ull || !nn__primitives__fits(n, x_room) || sys__node_array__length(ids) < count || !(penalty > 0.0f))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (count == 0ull) return nn__doors_answer(&argv[0]);
    sys__node_array_walk walk;
    if (!sys__node_array__walk(ids, 0ull, &walk)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    for (uint64_t j = 0ull; j < count; ++j, sys__node_array__next(&walk)) {
        const sys__heap_node* cell = sys__node_array__walk_cell(&walk);
        if (cell == 0 || cell->dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    }
    const sys__heap_node* cells = sys__node_array__cells(ids, count);
    const nn__doors* doors = nn__doors_for(ctx);
    if (cells == 0) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t words = count > 1ull ? (uint64_t)(&cells[1].args[0] - &cells[0].args[0]) : 1ull;
    if (ctx->heap_card != 0 && cells >= ctx->heap_host && (uint64_t)(cells - ctx->heap_host) + count <= ctx->heap_nodes) {
        const sys__heap_node* card = ctx->heap_card + (cells - ctx->heap_host);
        doors->vector_penalize((uint16_t*)(uintptr_t)x_at, n, &card[0].args[0], words, count, penalty, ctx->fault_word);
        return nn__doors_answer(&argv[0]);
    }
    uint64_t* held = (uint64_t*)malloc((size_t)(count * sizeof(uint64_t)));
    void* landing = 0;
    if (held == 0 || !sys__gpu__memory_allocate(ctx->family, &landing, count * sizeof(uint64_t))) {
        free(held);
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    }
    for (uint64_t j = 0ull; j < count; ++j) held[j] = cells[j].args[0];
    bool ok = sys__gpu__memory_write(ctx->family, landing, held, count * sizeof(uint64_t));
    if (ok) doors->vector_penalize((uint16_t*)(uintptr_t)x_at, n, (const uint64_t*)landing, 1ull, count, penalty, ctx->fault_word);
    ok = ok && sys__gpu__compute_completed(ctx->family);
    sys__gpu__memory_free(ctx->family, landing);
    free(held);
    if (!ok) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    return nn__doors_answer(&argv[0]);
}

SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_top_k,         nn__vector__zzabi_apply_top_k)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_top_k_biased,  nn__vector__zzabi_apply_top_k_biased)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_penalize,      nn__vector__zzabi_apply_penalize)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_pointwise_mul, nn__vector__zzabi_apply_pointwise_mul)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_scale,         nn__vector__zzabi_apply_scale)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_scale_at,      nn__vector__zzabi_apply_scale_at)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_VECTOR__ABI_CUH */
