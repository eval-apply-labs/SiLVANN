#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_PRIMITIVES__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_PRIMITIVES__ABI_CUH
/* ══ THE NORMALISATIONS AND THE MATRIX VERBS, ON THE PROGRAM'S SIDE ═══════════════════════════════════
 * Each checks its operands exactly as its body in `primitives__impl.cuh` did and hands the arithmetic to
 * its door, on the calling worker's silicon. ▶ `vector__abi.cuh` for the shape and the refusals. */

#include "../doors.cuh"

/* ── `out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]` ────────────────────────────────────────────
 * ⛔⛔ THE SUM IS TAKEN IN ONE PASS BEFORE ANYTHING IS WRITTEN, AND THAT ORDER IS LOAD-BEARING WHEN THE
 * CALLER ALIASES `x` AND `out` — which it will, because normalising in place is what a model does.
 * Writing as it went would fold already-scaled elements back into a sum that is still being taken, and
 * the answer would be wrong by an amount that depends on the length. ⇒ ★ A FUSED LOOP IS NOT A FASTER
 * TWO-PASS LOOP WHEN ONE OF ITS OPERANDS IS ITS OWN OUTPUT.
 * ⛳ THE ACCUMULATOR IS `float` AND NOT `double`, DELIBERATELY: the reference this is checked against
 * reduces in fp32 too, and a wider accumulator here would make the oracle disagree with the thing it is
 * supposed to agree with. ⚠ It is also the less accurate choice, and that is a real trade — a model's
 * hidden dim is thousands of terms. It is what the old tree does. */
/* `(nn__rmsnorm__apply x w out n eps)` — a length of zero has no mean, so it is refused before anything
 * divides by it. The epsilon is a real and is not defaulted: it is the model's, and a default would be
 * one model's number standing in for every other's. */
static sys__heap_node nn__rmsnorm__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, w_at = 0, o_at = 0, x_room = 0, w_room = 0, o_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_FLOAT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &w_at, &w_room)
     || !nn__primitives__room(&argv[2], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, w_room)
     || !nn__primitives__fits(n, o_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (n == 0ull) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->rmsnorm((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, (const uint16_t*)(uintptr_t)w_at, n,
                   (float)sys__heap_node__real(argv[4].args[0]), ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* ══ ⭐⭐ SOFTMAX — a reduction, then a second reduction, then a broadcast ═══════════════════════════
 * ⛳ THREE PASSES AND NOT TWO, deliberately. The one-pass "online" form that rescales a running sum as
 * a new maximum arrives is a real algorithm and it is a PERFORMANCE choice; here the door runs one
 * block of one thread and nothing is parallel, so the fused form would buy nothing and cost a reader the
 * argument for why it is correct. ▶ `nn_port.md` §⑧ — no throughput claim in this package is
 * observed, so none may be optimised for. */
/* `(nn__softmax__apply x out n)` — refuses a length of zero, as rmsnorm does. */
static sys__heap_node nn__softmax__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, o_at = 0, x_room = 0, o_room = 0;
    if (argv[2].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[2].args[0];
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, o_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (n == 0ull) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->softmax((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, n, ctx->fault_word);
    return nn__doors_answer(&argv[1]);
}

/* ══ ⭐ SIGMOID — elementwise, and the one activation `swiglu` does not expose ══════════════════════ */
/* `(nn__sigmoid__apply x out n)` — elementwise, so an empty length is simply nothing to do. */
static sys__heap_node nn__sigmoid__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, o_at = 0, x_room = 0, o_room = 0;
    if (argv[2].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &o_at, &o_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[2].args[0];
    if (!nn__primitives__fits(n, x_room) || !nn__primitives__fits(n, o_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->sigmoid((uint16_t*)(uintptr_t)o_at, (const uint16_t*)(uintptr_t)x_at, n, ctx->fault_word);
    return nn__doors_answer(&argv[1]);
}

/* ══ ⭐⭐ TRANSPOSE — `dst[c][r] = src[r][c]` ═══════════════════════════════════════════════════════
 * ⚖ *"a transpose method in nn where you pass a table and it gives you its transposed one."*
 * ⛳ THE WALK IS OVER THE *DESTINATION* IN ORDER, not over the source: writes are sequential and the
 * reads are strided. On a serial one-thread loop the two are the same work, and it is written this
 * way because the parallel version will want contiguous writes per lane. ⛔ NO SPEED IS CLAIMED —
 * ▶ `nn_port.md` §⑧, nothing in this package has been observed. */
/* `(nn__matrix__transpose src dst rows cols)` — src and dst must differ: a transpose in place would read
 * cells it has already overwritten. Nothing here can overflow, so the door takes no fault word. */
static sys__heap_node nn__matrix__zzabi_apply_transpose(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t s_at = 0, d_at = 0, s_room = 0, d_room = 0;
    if (argv[2].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &s_at, &s_room)
     || !nn__primitives__room(&argv[1], &d_at, &d_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = argv[2].args[0], cols = argv[3].args[0];
    if (rows == 0ull || cols == 0ull) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (rows > 0xFFFFFFFFFFFFFFFFull / cols) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const uint64_t cells = rows * cols;
    if (!nn__primitives__fits(cells, s_room) || !nn__primitives__fits(cells, d_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (s_at == d_at) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->matrix_transpose((uint16_t*)(uintptr_t)d_at, (const uint16_t*)(uintptr_t)s_at, rows, cols);
    return nn__doors_answer(&argv[1]);
}

/* ══ ⭐⭐⭐ THE TRANSPOSED MATVEC — `out[c] = Σ_r W[r][c]·x[r]`, WITH NO TRANSPOSE ═════════════════
 * ⚖ *"i want it to have the option to multiply a vector for a matrix and then multiply it for its
 * transposed matrix."* ⭐ `W` is read in the layout it is already stored in; only the ACCUMULATION
 * changes. ⇒ ★ A TRANSPOSE IS A CHANGE OF TRAVERSAL BEFORE IT IS A CHANGE OF STORAGE.
 * ⛔ fp16 ONLY — ▶ the header, and the 132% measurement that says why a quantised plane cannot.
 *
 * ⛳ THE LOOP IS COLUMN-OUTER AND ROW-INNER, WHICH IS THE SLOWER TRAVERSAL AND THE RIGHT ONE. The
 * other order — walk rows, scatter into `out` — would accumulate into fp16 STORAGE `cols` times over,
 * losing the fp32 accumulator this package rules for every reduction. ⇒ ★★ PICKING THE ACCESS PATTERN
 * BY CACHE BEHAVIOUR WOULD HAVE COST THE ACCUMULATOR, and the accumulator is the ruling. The door runs
 * one block of one thread and nothing here is parallel, so there is no speed claim either way. */
/* `(nn__matrix__matvec_transposed W x out rows cols)` — `out = Wᵀx`: W is rows × cols as stored, x has
 * rows elements and out has cols. */
static sys__heap_node nn__matrix__zzabi_apply_matvec_transposed(const sys__heap_node* argv, unsigned argc,
                                                         sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t w_at = 0, x_at = 0, d_at = 0, w_room = 0, x_room = 0, d_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &w_at, &w_room)
     || !nn__primitives__room(&argv[1], &x_at, &x_room)
     || !nn__primitives__room(&argv[2], &d_at, &d_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = argv[3].args[0], cols = argv[4].args[0];
    if (rows == 0ull || cols == 0ull) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (rows > 0xFFFFFFFFFFFFFFFFull / cols) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (!nn__primitives__fits(rows * cols, w_room)
     || !nn__primitives__fits(rows, x_room)
     || !nn__primitives__fits(cols, d_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->matrix_matvec_transposed((uint16_t*)(uintptr_t)d_at, (const uint16_t*)(uintptr_t)w_at, (const uint16_t*)(uintptr_t)x_at,
                                    rows, cols, ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(nn__rmsnorm__zzabi_adapter,                    nn__rmsnorm__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(nn__softmax__zzabi_adapter,                    nn__softmax__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(nn__sigmoid__zzabi_adapter,                    nn__sigmoid__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(nn__matrix__zzabi_adapter_transpose,           nn__matrix__zzabi_apply_transpose)
SYS__ENGINE__ABI__BRIDGE(nn__matrix__zzabi_adapter_matvec_transposed,   nn__matrix__zzabi_apply_matvec_transposed)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_PRIMITIVES__ABI_CUH */
