#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_TURBOQUANT__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_TURBOQUANT__ABI_CUH
/* ══ TurboQuant's DECODE AND GEMV, ON THE PROGRAM'S SIDE ═════════════════════════════════════════════
 * The checks are their bodies' in `turboquant__impl.cuh`, plus one more: whether every row lies inside
 * the rooms it was given. It is the body's own `false`, and it depends on nothing but sizes —
 * `nn__turboquant__zzpackage_row` compares `row_bytes(d, cols)` with the two rooms, and the last row is
 * the one that reaches furthest — so it is asked here, before anything launches, and the doors carry no
 * result. ⛳ A refused call writes nothing, where a refusal inside the body would come mid-matrix, after
 * the rows before the bad one. ▶ `vector__abi.cuh` for the shape and the other refusals. */

#include "../doors.cuh"

/* Every row of a `rows`-row matrix at width `d` inside `w_room` bytes of codes and `l_room` of scales:
 * the last row's location, asked the way the kernel asks it. */
static inline bool nn__turboquant__zzabi_zzprivate_rows_fit(uint64_t w_at, uint64_t w_room, uint64_t l_at,
                                                            uint64_t l_room, uint64_t d, uint64_t rows,
                                                            uint64_t cols) {
    const uint8_t *row_base = 0, *lut_base = 0;
    return nn__turboquant__zzpackage_row((const uint8_t*)(uintptr_t)w_at, w_room, (const uint8_t*)(uintptr_t)l_at,
                                         l_room, d, rows - 1ull, cols, &row_base, &lut_base);
}

/* ── `(nn__turboquant__decode weights lut out rows cols d)` ─────────────────────────────────────────────────
 * The two planes of a family-2 matrix to a dense fp16 `[rows*cols]`.
 * ⛳ IT IS THE WHOLE MATRIX AND NOT ONE ROW, because that is the shape the port inherited. A per-row
 * verb would be the better primitive the day a caller wants one row; nothing wants one yet. */
/* `(nn__turboquant__decode codes scales out rows cols d)` — the matrix expanded to fp16. */
static sys__heap_node nn__turboquant__zzabi_apply_decode(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t w_at = 0, l_at = 0, d_at = 0, w_room = 0, l_room = 0, d_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || argv[5].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &w_at, &w_room)
     || !nn__primitives__room(&argv[1], &l_at, &l_room)
     || !nn__primitives__room(&argv[2], &d_at, &d_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = argv[3].args[0], cols = argv[4].args[0], d = argv[5].args[0];
    if (rows == 0ull || cols == 0ull) return sys__engine__abi__error(NN__TURBOQUANT__FAULT_BOUNDS);
    if (!nn__turboquant__rate_is_known(d)) return sys__engine__abi__error(NN__TURBOQUANT__FAULT_RATE);
    if ((cols != 0ull && rows > 0xFFFFFFFFFFFFFFFFull / cols) || !nn__primitives__fits(rows * cols, d_room))
        return sys__engine__abi__error(NN__TURBOQUANT__FAULT_BOUNDS);
    if (!nn__turboquant__zzabi_zzprivate_rows_fit(w_at, w_room, l_at, l_room, d, rows, cols))
        return sys__engine__abi__error(nn__turboquant__zzpackage_fault_for(d));

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->turboquant_decode((uint16_t*)(uintptr_t)d_at, (const uint8_t*)(uintptr_t)w_at, w_room, (const uint8_t*)(uintptr_t)l_at, l_room,
                             d, rows, cols, ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

/* ── ⭐⭐ THE GEMV — `out[r] = Σ_c W[r][c] * x[c]`, WITHOUT EVER MATERIALISING W ───────────────────────
 *
 * ⭐ THIS IS WHAT THE COMPRESSION IS FOR. `decode` proves the layout is read correctly and costs
 * `rows × cols` floats of scratch to do it; a 2048×4096 expert is 32 MiB dense against ~4 MiB packed.
 * ⇒ ★ A CODEC THAT MUST EXPAND BEFORE IT CAN MULTIPLY HAS SPENT THE SAVING IT EXISTS TO MAKE. The
 * weight is decoded into a REGISTER and consumed in the same expression.
 *
 * ⛔⛔ THE ACCUMULATOR IS fp32 — matching the old tree's `psum` and the reference — BUT THE ORDER IS
 * NOT SOMETHING THIS SOURCE GETS TO DECIDE, AND AN EARLIER VERSION OF THIS COMMENT CLAIMED IT WAS.
 * `MEASURED` with a fixture built to tell: `2^24` followed by seven 1.0s over a row of all
 * +1 weights. Summed in the written order the trailing ones are each lost and the answer is 16777216;
 * summed any other way they survive and it is 16777224. **The device answered 16777224**, from this
 * loop, with no `-ffast-math` anywhere in the build — only `-O3`.
 * ⇒ ★★ A REDUCTION'S ORDER IS NOT PINNED BY THE ORDER IT IS WRITTEN IN.
 * ⛳⛳ AND THAT IS A NARROWER CLAIM THAN IT FIRST SOUNDS — ⚖ the architect pushed back and was right.
 * TWO THINGS WERE BEING CONFLATED:
 * ```
 *   run-to-run coherence    TRUE.  The compiler picks an order at COMPILE time and bakes it into the
 *                                  binary. `MEASURED`: two full device-suite runs differ in exactly
 *                                  ONE line, and it is a `hipMalloc` address.
 *   source-order fidelity   FALSE. The executed order is not the written order.
 * ```
 * ⇒ ★ SO REPEATABILITY IS UNTOUCHED. What is lost is only agreement with an EXTERNAL reference that
 * assumes the written order — and the one place that bites is ACROSS BINARIES, where a different runtime
 * or compiler may choose differently. ⛳ The old tree already knew: `device_math.cuh:59` — *"Cross-arch
 * fp is never bit-identical anyway; Phase-4 verifies by data."*
 * ⭐ AND THE HARDWARE'S OWN PRIMITIVE IS ALREADY PAIRWISE: `fdot2(half2, half2, float)` folds TWO
 * products per instruction, so a linear left-to-right sum was never the shape the silicon prefers.
 * The reassociation is the arithmetic fitting the machine, not the compiler being clever.
 * ⛳ WHAT REMAINS TRUE AND IS WORTH KEEPING: the accumulator's WIDTH is a real choice (fp32, not
 * double), because a wider one would disagree with the reference by more than rounding. Width is
 * decidable here; order is not.
 *
 * ⛳ AND IT IS A GEMV AND NOT A GEMM, DELIBERATELY. A model decodes one token at a time through these
 * weights; the batched form is a different verb with a different tiling, and inventing it now would be
 * a shape nothing has asked for. ⚖ *"import it as is"* applies here too. */
/* `(nn__turboquant__gemv codes scales x out rows cols d)` — `out = W x`, each weight decoded where it is
 * used and never stored. */
static sys__heap_node nn__turboquant__zzabi_apply_gemv(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t w_at = 0, l_at = 0, x_at = 0, d_at = 0, w_room = 0, l_room = 0, x_room = 0, d_room = 0;
    if (argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT
     || argv[6].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &w_at, &w_room)
     || !nn__primitives__room(&argv[1], &l_at, &l_room)
     || !nn__primitives__room(&argv[2], &x_at, &x_room)
     || !nn__primitives__room(&argv[3], &d_at, &d_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = argv[4].args[0], cols = argv[5].args[0], d = argv[6].args[0];
    if (rows == 0ull || cols == 0ull) return sys__engine__abi__error(NN__TURBOQUANT__FAULT_BOUNDS);
    if (!nn__turboquant__rate_is_known(d)) return sys__engine__abi__error(NN__TURBOQUANT__FAULT_RATE);
    if (!nn__primitives__fits(cols, x_room) || !nn__primitives__fits(rows, d_room))
        return sys__engine__abi__error(NN__TURBOQUANT__FAULT_BOUNDS);
    if (!nn__turboquant__zzabi_zzprivate_rows_fit(w_at, w_room, l_at, l_room, d, rows, cols))
        return sys__engine__abi__error(nn__turboquant__zzpackage_fault_for(d));

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)d_at, (const uint8_t*)(uintptr_t)w_at, w_room, (const uint8_t*)(uintptr_t)l_at, l_room,
                           (const uint16_t*)(uintptr_t)x_at, d, rows, cols, ctx->fault_word);
    return nn__doors_answer(&argv[3]);
}

/* `(nn__turboquant__gemv_int8 codes scales x out rows cols d)` — the same product as `nn__turboquant__gemv`,
 * with `x` and the levels in int8 and the products in integers. ⚖ *"two different instructions so at boot
 * time the lisp can decide how to write its defun"*. ▶ `gpu/kernels/kernels.cuh` for what every family's body
 * agrees on. ⛔ It refuses the width with no table (16, whose codes are the weights) and columns that are not
 * a multiple of the 32 `x` is quantised in. */
static sys__heap_node nn__turboquant__zzabi_apply_gemv_int8(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t w_at = 0, l_at = 0, x_at = 0, d_at = 0, w_room = 0, l_room = 0, x_room = 0, d_room = 0;
    if (argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT
     || argv[6].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &w_at, &w_room)
     || !nn__primitives__room(&argv[1], &l_at, &l_room)
     || !nn__primitives__room(&argv[2], &x_at, &x_room)
     || !nn__primitives__room(&argv[3], &d_at, &d_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = argv[4].args[0], cols = argv[5].args[0], d = argv[6].args[0];
    if (rows == 0ull || cols == 0ull || cols % NN__TURBOQUANT__INT8_BLOCK != 0ull)
        return sys__engine__abi__error(NN__TURBOQUANT__FAULT_BOUNDS);
    if (!nn__turboquant__rate_is_known(d) || d == 16ull) return sys__engine__abi__error(NN__TURBOQUANT__FAULT_RATE);
    if (!nn__primitives__fits(cols, x_room) || !nn__primitives__fits(rows, d_room))
        return sys__engine__abi__error(NN__TURBOQUANT__FAULT_BOUNDS);
    if (!nn__turboquant__zzabi_zzprivate_rows_fit(w_at, w_room, l_at, l_room, d, rows, cols))
        return sys__engine__abi__error(nn__turboquant__zzpackage_fault_for(d));

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->turboquant_gemv_int8((uint16_t*)(uintptr_t)d_at, (const uint8_t*)(uintptr_t)w_at, w_room, (const uint8_t*)(uintptr_t)l_at,
                                l_room, (const uint16_t*)(uintptr_t)x_at, d, rows, cols, ctx->fault_word);
    return nn__doors_answer(&argv[3]);
}

/* `(nn__turboquant__encode x codes scales rows cols d)` — `rows` rows of `cols` halves, already rotated, as TurboQuant
 * rows at width `d` (4 or 8): the codes a row after another, a half scale a row. The KV cache's cooling. */
static sys__heap_node nn__turboquant__zzabi_apply_encode(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x_at = 0, c_at = 0, s_at = 0, x_room = 0, c_room = 0, s_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x_at, &x_room)
     || !nn__primitives__room(&argv[1], &c_at, &c_room)
     || !nn__primitives__room(&argv[2], &s_at, &s_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = argv[3].args[0], cols = argv[4].args[0], d = argv[5].args[0];
    if ((d != 4ull && d != 8ull) || rows == 0ull || cols == 0ull || rows > (1ull << 32) || cols > (1ull << 24))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (!nn__primitives__fits(rows * cols, x_room) || rows * nn__turboquant__row_bytes(d, cols) > c_room
     || !nn__primitives__fits(rows, s_room)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->turboquant_encode_rows((uint8_t*)(uintptr_t)c_at, (uint16_t*)(uintptr_t)s_at, (const uint16_t*)(uintptr_t)x_at, rows, cols,
                                  d, ctx->fault_word);
    return nn__doors_answer(&argv[1]);
}

SYS__ENGINE__ABI__BRIDGE(nn__turboquant__zzabi_adapter_encode, nn__turboquant__zzabi_apply_encode)
SYS__ENGINE__ABI__BRIDGE(nn__turboquant__zzabi_adapter_decode, nn__turboquant__zzabi_apply_decode)
SYS__ENGINE__ABI__BRIDGE(nn__turboquant__zzabi_adapter_gemv_int8, nn__turboquant__zzabi_apply_gemv_int8)
SYS__ENGINE__ABI__BRIDGE(nn__turboquant__zzabi_adapter_gemv,   nn__turboquant__zzabi_apply_gemv)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_TURBOQUANT__ABI_CUH */
