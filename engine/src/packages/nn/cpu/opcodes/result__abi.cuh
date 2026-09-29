#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_RESULT__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_RESULT__ABI_CUH
/* ══ THE VERBS THAT ANSWER A VALUE, AND THE ONE THAT READS IT ════════════════════════════════════════
 * ⚖ RULED, shape B: argmax, dot_product and at write their value into an `out` buffer and
 * answer that buffer, like every other verb — so a door that wants the value on the card takes the
 * buffer, and nothing waits. `nn__buffer__read` turns it into a value when the program needs one: in
 * mapped pinned RAM it watches the word land (~4.7 us, `MEASURED`), anywhere else it waits for the card
 * and copies it back. ▶ `contracts/objects/result.cuh` for the stamped word, and why it is stamped. */

#include "../doors.cuh"
#include "../../contracts/objects/result.cuh"

/* A fresh stamp: the kind, and a ticket no earlier call into any buffer shares (modulo 2^24, and only the
 * latest call into a buffer is ever compared). */
static inline uint64_t nn__result__zzprivate_stamp(uint64_t kind) {
    static uint64_t tickets = 0ull;
    const uint64_t t = __atomic_add_fetch(&tickets, 1ull, __ATOMIC_RELAXED) & NN__RESULT__TICKET_MASK;
    return (kind << NN__RESULT__KIND_SHIFT) | (t << NN__RESULT__TICKET_SHIFT);
}

/* The `out` operand of a value verb: somewhere a card can write, room for the word, and its stamp slot. */
static inline bool nn__result__zzprivate_out(const sys__heap_node* ref, uint64_t* at, uint64_t** slot) {
    uint64_t room = 0, host_at = 0;
    if (!nn__primitives__room(ref, at, &room)) return false;
    *slot = nn__primitives__result_slot(ref, &host_at);
    return *slot != 0 && room >= NN__RESULT__BYTES;
}

/* ── the index of the largest element ────────────────────────────────────────────────────────────────
 * ⭐⭐ IT ANSWERS AN INTEGER, WHICH IS WHY IT MATTERS MORE THAN ITS ARITHMETIC. Every other verb here
 * hands back a buffer; this hands back a NUMBER, and `eng_abi_poll` gives a program's integer answer
 * straight to the host. ⇒ a token id needs no buffer, no address and no DMA. */
/* `(nn__argmax__find x out n)` — the index of the largest of n, as a 32-bit value in `out`. */
static sys__heap_node nn__argmax__zzabi_apply_find(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, x_room = 0, o_at = 0; uint64_t* slot = 0;
    if (argv[2].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x_at, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__result__zzprivate_out(&argv[1], &o_at, &slot)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const uint64_t n = argv[2].args[0];
    if (!nn__primitives__fits(n, x_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (n == 0ull) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (n > 0xFFFFFFFFull) return sys__engine__abi__error(NN__RESULT__FAULT_TOO_LONG);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t stamp = nn__result__zzprivate_stamp(NN__RESULT__INT);
    *slot = stamp;
    doors->argmax((uint64_t*)(uintptr_t)o_at, stamp, (const uint16_t*)(uintptr_t)x_at, n);
    return nn__doors_answer(&argv[1]);
}

/* ══ ⭐⭐ THE DOT PRODUCT — one number, and the language can finally hold one ═══════════════════════
 * ⛔ A DIFFERENT OPERATION FROM THE POINTWISE PRODUCT, not another name for it: that one answers `n`
 * values, this one answers ONE. ▶ the header. */
/* `(nn__vector__dot_product a b out n)` — the fp32 sum of a·b, as a float in `out`. */
static sys__heap_node nn__vector__zzabi_apply_dot_product(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t a_at = 0, b_at = 0, a_room = 0, b_room = 0, o_at = 0; uint64_t* slot = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &a_at, &a_room)
     || !nn__primitives__room(&argv[1], &b_at, &b_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__result__zzprivate_out(&argv[2], &o_at, &slot)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const uint64_t n = argv[3].args[0];
    if (n == 0ull) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (!nn__primitives__fits(n, a_room) || !nn__primitives__fits(n, b_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t stamp = nn__result__zzprivate_stamp(NN__RESULT__FLOAT);
    *slot = stamp;
    doors->vector_dot_product((uint64_t*)(uintptr_t)o_at, stamp, (const uint16_t*)(uintptr_t)a_at, (const uint16_t*)(uintptr_t)b_at, n);
    return nn__doors_answer(&argv[2]);
}

/* ── `(nn__vector__at x i)` -> a float ─────────────────────────────────────────────────────────────── */
/* `(nn__vector__at x i out)` — element i, as a float in `out`. A kernel and not a copy: it is ordered
 * behind whatever wrote x, and landing in mapped RAM it is cheaper than a copy of the same two bytes. */
static sys__heap_node nn__vector__zzabi_apply_at(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, x_room = 0, o_at = 0; uint64_t* slot = 0;
    if (argv[1].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x_at, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__result__zzprivate_out(&argv[2], &o_at, &slot)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const uint64_t i = argv[1].args[0];
    if (i == 0xFFFFFFFFFFFFFFFFull || !nn__primitives__fits(i + 1ull, x_room))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t stamp = nn__result__zzprivate_stamp(NN__RESULT__FLOAT);
    *slot = stamp;
    doors->vector_at((uint64_t*)(uintptr_t)o_at, stamp, (const uint16_t*)(uintptr_t)x_at, i);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__buffer__read b)` — the value the last value verb wrote into b, as an INT or a FLOAT.
 * In mapped RAM the word is watched until it carries b's stamp; the watch is bounded, and past it the
 * card is waited for once, so a kernel that has not started yet is still answered and a buffer nothing was
 * ever written into is refused rather than waited on forever. Anywhere else the card is waited for and
 * the word copied back. */
static sys__heap_node nn__buffer__zzabi_apply_read(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t at = 0, room = 0, host_at = 0;
    if (!nn__primitives__room(&argv[0], &at, &room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t* slot = nn__primitives__result_slot(&argv[0], &host_at);
    if (slot == 0) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t stamp = *slot;
    if (stamp == 0ull || room < NN__RESULT__BYTES) return sys__engine__abi__error(NN__RESULT__FAULT_UNWRITTEN);
    if (nn__doors_for(ctx) == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);

    uint64_t word = 0ull;
    if (host_at != 0ull) {
        const volatile uint64_t* w = (const volatile uint64_t*)(uintptr_t)host_at;
        for (uint32_t spin = 0u; spin < (1u << 16); ++spin) {
            word = *w;
            if ((word & NN__RESULT__STAMP_MASK) == stamp) break;
        }
        if ((word & NN__RESULT__STAMP_MASK) != stamp) {
            (void)sys__gpu__compute_completed(ctx->family);
            word = *w;
        }
    } else if (!sys__gpu__compute_completed(ctx->family)
            || !sys__gpu__memory_read(ctx->family, &word, (const void*)(uintptr_t)at, sizeof word)) {
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    }
    if ((word & NN__RESULT__STAMP_MASK) != stamp) return sys__engine__abi__error(NN__RESULT__FAULT_UNWRITTEN);

    const uint32_t bits = (uint32_t)(word & 0xFFFFFFFFull);
    sys__heap_node answer = sys__heap_node__nothing();
    answer.num_args = 0u; answer.op_code = 0ull;
    if ((stamp >> NN__RESULT__KIND_SHIFT) == NN__RESULT__INT) {
        answer.dtype = SYS__KIND__VALUE_INT;
        answer.args[0] = (uint64_t)bits;
    } else {
        union { uint32_t u; float f; } cast; cast.u = bits;
        answer.dtype = SYS__KIND__VALUE_FLOAT;
        answer.args[0] = sys__heap_node__real_bits((double)cast.f);
    }
    return answer;
}

SYS__ENGINE__ABI__BRIDGE(nn__argmax__zzabi_adapter_find,        nn__argmax__zzabi_apply_find)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_dot_product, nn__vector__zzabi_apply_dot_product)
SYS__ENGINE__ABI__BRIDGE(nn__vector__zzabi_adapter_at,          nn__vector__zzabi_apply_at)
SYS__ENGINE__ABI__BRIDGE(nn__buffer__zzabi_adapter_read,        nn__buffer__zzabi_apply_read)

/* ⭐⭐ `(nn__buffer__copy from worker to bytes)` -> `to`: `bytes` of `from`, a buffer of the worker named, into `to`,
 * one of this verb's own worker. ⚖ *"so in case of a single card or a single card type the buffer remains on gpu,
 * if there are different card types the program is written that it also copies the buffer out"* — so the program
 * says when a tensor crosses, and this is the crossing: the source's device bound and waited for (its work may
 * still be running), its bytes read into RAM, this worker's device bound again and the bytes written. On node03's
 * MI50s, which have no peer access, RAM is the only way between two cards anyway.
 * ⛳ THROUGH A BOUNCE THE CALLING THREAD KEEPS, grown to the largest copy it has made. */
static __thread unsigned char* nn__buffer__zzprivate_bounce = 0;
static __thread uint64_t       nn__buffer__zzprivate_bounce_room = 0ull;
static sys__heap_node nn__buffer__zzabi_apply_copy(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t from = 0, from_room = 0, to = 0, to_room = 0;
    if (!nn__primitives__room_of_any(&argv[0], &from, &from_room) || argv[1].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[2], &to, &to_room) || argv[3].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t bytes = argv[3].args[0];
    if (bytes == 0ull || bytes > from_room || bytes > to_room) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    /* ⛳ THE SOURCE IS READ WHATEVER WORKER HOLDS IT — that is the crossing — but it must be the worker named: the
     *   bytes are read through that worker's device. */
    const uint64_t owner = nn__primitives__owner(&argv[0]);
    if (owner != 0ull && owner != argv[1].args[0] + 1ull) return sys__engine__abi__error(NN__BUFFER__FAULT_OWNER);
    const sys__engine__ctx* src = sys__engine__ctx_of((unsigned int)argv[1].args[0]);
    if (src == 0 || src->fault_word == 0 || nn__doors_for(ctx) == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (bytes > nn__buffer__zzprivate_bounce_room) {
        unsigned char* grown = (unsigned char*)realloc(nn__buffer__zzprivate_bounce, (size_t)bytes);
        if (grown == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
        nn__buffer__zzprivate_bounce = grown; nn__buffer__zzprivate_bounce_room = bytes;
    }
    const bool elsewhere = src->family != ctx->family || src->device != ctx->device;
    bool ok = !elsewhere || sys__silicon__bind_device(src->family, src->device);
    ok = ok && sys__gpu__compute_completed(src->family)
            && sys__gpu__memory_read(src->family, nn__buffer__zzprivate_bounce, (const void*)(uintptr_t)from, (size_t)bytes);
    if (elsewhere) ok = sys__silicon__bind_device(ctx->family, ctx->device) && ok;
    ok = ok && sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)to, nn__buffer__zzprivate_bounce, (size_t)bytes);
    if (!ok) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    return nn__doors_answer(&argv[2]);
}
SYS__ENGINE__ABI__BRIDGE(nn__buffer__zzabi_adapter_copy,        nn__buffer__zzabi_apply_copy)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_RESULT__ABI_CUH */
