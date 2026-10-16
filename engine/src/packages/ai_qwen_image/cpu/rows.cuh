#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_ROWS_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_ROWS_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ ROWS THROUGH A MATRIX — what every verb of the package does to multiply ══════════════════════════════════════
 * A matrix's codes at 8 bits were rotated row by row in the pack's ladder, so the rows it multiplies are rotated the same
 * way first (`hadamard_ladder_rows`); at 16 bits a matrix is as it was. Every product is nn's `expert_rows` over the rows
 * `(j, j)` — the pairs at the front of a scratch. */

/* The pairs `(j, j)` for `count` rows at `at`, written from the host 256 at a time. False if the card would not take them. */
static bool ai_qwen_image__rows__zzpackage_pairs(sys__engine__ctx* ctx, uint64_t at, uint64_t count) {
    uint32_t pairs[512u];
    for (uint64_t j0 = 0u; j0 < count; j0 += 256u) {
        const uint64_t n = count - j0 < 256u ? count - j0 : 256u;
        for (uint64_t j = 0u; j < n; ++j) { pairs[2u * j] = (uint32_t)(j0 + j); pairs[2u * j + 1u] = (uint32_t)(j0 + j); }
        if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)(at + 8u * j0), pairs, (size_t)(8u * n))) return false;
    }
    return true;
}

/* The bytes the pairs of `rows` rows take at the front of a scratch, rounded to 16. */
static uint64_t ai_qwen_image__rows__zzpackage_pairs_bytes(uint64_t rows) {
    return (8u * rows + 15u) & ~15ull;
}

/* Whether a matrix of `out_rows` rows of `cols` at width `d` fits its planes — the codes, and the scales below 16 bits;
 * below 16 bits its rows a multiple of 128, as the ladder takes them. */
static bool ai_qwen_image__rows__zzpackage_holds(const uint64_t* room, unsigned codes, uint64_t d, uint64_t out_rows,
                                                 uint64_t cols) {
    if (d != 4u && d != 8u && d != 16u) return false;
    if (d != 16u && cols % 128u != 0u) return false;
    return out_rows * nn__turboquant__row_bytes(d, cols) <= room[codes] && (d == 16u || 2u * out_rows <= room[codes + 1u]);
}

/* `out = W·in` over `rows` rows of `cols`, `in` already rotated where W's codes are. Every operand a card address. */
static void ai_qwen_image__rows__zzpackage_product(const nn__doors* doors, uint64_t out, const uint64_t* at, const uint64_t* room,
                                                   unsigned codes, uint64_t d, uint64_t in, uint64_t pairs, uint64_t rows,
                                                   uint64_t out_rows, uint64_t cols, unsigned int* over) {
    doors->expert_rows((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)at[codes], room[codes],
                       (const uint8_t*)(uintptr_t)at[codes + 1u], room[codes + 1u], (const uint16_t*)(uintptr_t)in,
                       (const uint32_t*)(uintptr_t)pairs, rows, d, out_rows, cols, over);
}

/* `x` rotated into `rot` where a matrix of width `d` multiplies it (below 16 bits); the rows to multiply, either way. */
static uint64_t ai_qwen_image__rows__zzpackage_rotated(const nn__doors* doors, uint64_t x, uint64_t rot, uint64_t signs,
                                                       uint64_t d, uint64_t cols, uint64_t rows, unsigned int* over) {
    if (d == 16u) return x;
    doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)rot, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)signs,
                                cols, rows, over);
    return rot;
}

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_ROWS_CUH */
