/* ══ THE INT8 BODIES ON AVX-512 — ONE TEXT, BUILT TWICE ════════════════════════════════════════════════════════════
 * No guard: `nn.cuh` includes this file twice, after the calls, structures and helpers it reads, and says each time how
 * it is built —
 *   X86_AVX2__ZZPRIVATE_I8(name)         a body's name: `x86_avx2__zzpackage_<name>_512`, or `_vnni`
 *   X86_AVX2__ZZPRIVATE_I8_TARGET        what it is compiled for: AVX-512 F, BW, VL and DQ, or those and VNNI
 *   X86_AVX2__ZZPRIVATE_I8_DOT(u, s)     the 32-bit lanes of 64 bytes' products, unsigned `u` by signed `s`, four bytes a
 *                                        lane: `maddubs` then `madd` over ones, or VNNI's `dpbusd` from zero
 *   X86_AVX2__ZZPRIVATE_I8_SUM(a, u, s)  the same added onto `a`: and `add`, or `dpbusd` onto `a`
 * ⭐ THE SAME BYTES EITHER WAY, `REASONED`: `u` is a level's magnitude, at most 128, and `s` an activation quantised to
 *   ±127 with the level's sign, so a pair's sum in `maddubs` is at most 2 × 128 × 127 = 32,512 and never saturates its
 *   16 bits; `madd` then adds two pairs into the lane `dpbusd` adds the same four products into, exactly — integers.
 *   `MEASURED` (measurements/2026-11-06_vnni_dl380.md): the doors' outputs byte for byte, and GLM's logits bit for bit. */

/* ── the int8 gemvs ── */
/* One 4-bit row against `x` quantised in blocks of 32 — a decoded position's experts (`groups_i8`, `groups_sum_i8`).
 * The row two ahead — `cols` bytes on, a row being `cols / 2` — is fetched into L2 as this one is read: a prefetch never
 * faults, so past a thread's last row or a matrix's it costs nothing, and the arithmetic is untouched. `MEASURED`
 * (measurements/2026-10-30_nn145_threads_dl380.md, a Gold 6132 socket, GLM's gate+up over memory no call has read): one
 * core 7.8 -> 12.1 GB/s, as fast as a plain read; seven 54 -> 72; fourteen 78 either way, the socket's memory. */
static X86_AVX2__ZZPRIVATE_I8_TARGET inline float X86_AVX2__ZZPRIVATE_I8(dot_i8_4)(const uint8_t* row, const int8_t* xq, const float* xs, uint64_t cols,
                                                                                __m512i t8) {
    const __m512i nibbles = _mm512_set1_epi16(0x0F0F);
    __m512 acc = _mm512_setzero_ps();
    uint64_t k = 0;
    for (; k + 64u <= cols; k += 64u) {
        _mm_prefetch((const char*)(row + k / 2u + cols), _MM_HINT_T1);
        const __m512i u = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row + k / 2u)));
        /* (u | u << 4) & 0x0F0F: the low nibble in the word's first byte, the high one in its second — the columns' order */
        const __m512i idx = _mm512_ternarylogic_epi32(u, _mm512_slli_epi16(u, 4), nibbles, 0xA8);
        const __m512i w = _mm512_shuffle_epi8(t8, idx);
        const __m512i q = _mm512_loadu_si512((const void*)(xq + k));
        const __m512i sq = _mm512_mask_sub_epi8(q, _mm512_movepi8_mask(w), _mm512_setzero_si512(), q);
        const __m512i p = X86_AVX2__ZZPRIVATE_I8_DOT(_mm512_abs_epi8(w), sq);
        const __m512 s = _mm512_insertf32x8(_mm512_set1_ps(xs[k / 32u]), _mm256_set1_ps(xs[k / 32u + 1u]), 1);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), s, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    if (k < cols) {                                     /* a last block of 32, as the AVX2 body has it */
        const __m128i fifteen = _mm_set1_epi8(15);
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
        const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
        const __m256i w = _mm256_shuffle_epi8(_mm512_castsi512_si256(t8), _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)));
        const __m256i q = _mm256_loadu_si256((const __m256i*)(xq + k));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(q, w));
        sum += x86_avx2__zzpackage_hsum(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p16, _mm256_set1_epi16(1))), _mm256_set1_ps(xs[k / 32u])));
    }
    return sum;
}
static X86_AVX2__ZZPRIVATE_I8_TARGET inline __m512i X86_AVX2__ZZPRIVATE_I8(table_i8)(void) {
    return _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
}
static X86_AVX2__ZZPRIVATE_I8_TARGET void X86_AVX2__ZZPRIVATE_I8(int8_4)(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    const __m512i t8 = X86_AVX2__ZZPRIVATE_I8(table_i8)();
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r)
        x86_avx2__zzpackage_write(c, r, X86_AVX2__ZZPRIVATE_I8(dot_i8_4)(c->weights + r * c->row_bytes, c->xq, c->xs, c->cols, t8)
                                        * x86_avx2__zzpackage_row_scale(c->luts, r) / 127.0f);
}
static X86_AVX2__ZZPRIVATE_I8_TARGET void X86_AVX2__ZZPRIVATE_I8(groups_i8)(void* a) {
    const x86_avx2__groups_i8_call* c = (const x86_avx2__groups_i8_call*)a;
    const __m512i t8 = X86_AVX2__ZZPRIVATE_I8(table_i8)();
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->rows * b / n, t1 = c->rows * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = t - base;
        const uint8_t* row = (const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u);
        const float dot = X86_AVX2__ZZPRIVATE_I8(dot_i8_4)(row, c->xq, c->xs, c->cols, t8);
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(
            dot * x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r) / 127.0f, c->over);
    }
}
static X86_AVX2__ZZPRIVATE_I8_TARGET void X86_AVX2__ZZPRIVATE_I8(groups_sum_i8)(void* a) {
    const x86_avx2__groups_i8_call* c = (const x86_avx2__groups_i8_call*)a;
    const __m512i t8 = X86_AVX2__ZZPRIVATE_I8(table_i8)();
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->rows * b / n, r1 = c->rows * (b + 1u) / n;
    for (uint64_t r = r0; r < r1; ++r) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i) {
            const uint8_t* row = (const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u);
            const float dot = X86_AVX2__ZZPRIVATE_I8(dot_i8_4)(row, c->xq + i * c->cols, c->xs + i * (c->cols / 32u), c->cols, t8);
            sum += dot * x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r) / 127.0f * _cvtsh_ss(c->w[c->g.out_at[i]]);
        }
        c->out[r] = nn__kernels__zzabi_to_half(_cvtsh_ss(c->residual[r]) + sum, c->over);
    }
}

/* ── the expert-major GEMM in integers: the row's levels once, 64 at a step, then its pairs four at a time ── */
/* `n` pairs (at most 4) against a row's int8 levels `w8`: a block's products summed in integers, one float step a block,
 * `v[q]` pair `q`'s sum. ⛳ Called with a literal 4 for a full four, so the sums stay in registers. */
static X86_AVX2__ZZPRIVATE_I8_TARGET inline void X86_AVX2__ZZPRIVATE_I8(pairs_i8)(const int8_t* w8, uint64_t cols, const int8_t* const* xr,
                                                                               const float* const* sr, uint64_t n, float* v) {
    const uint64_t per = cols / NN__EXPERT__INT8_BLOCK;
    __m512 f[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    for (uint64_t B = 0; B < per; ++B) {
        __m512i s4[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
        for (uint64_t k = B * NN__EXPERT__INT8_BLOCK; k < (B + 1u) * NN__EXPERT__INT8_BLOCK; k += 64u) {
            const __m512i wv = _mm512_load_si512((const void*)(w8 + k)), aw = _mm512_abs_epi8(wv);
            const __mmask64 neg = _mm512_movepi8_mask(wv);
            for (uint64_t q = 0; q < n; ++q) {
                const __m512i xv = _mm512_loadu_si512((const void*)(xr[q] + k));
                s4[q] = X86_AVX2__ZZPRIVATE_I8_SUM(s4[q], aw, _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv));
            }
        }
        for (uint64_t q = 0; q < n; ++q) f[q] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s4[q]), _mm512_set1_ps(sr[q][B]), f[q]);
    }
    for (uint64_t q = 0; q < n; ++q) v[q] = _mm512_reduce_add_ps(f[q]);
}
/* One or two pairs — an expert one or two of a chunk's rows picked: the row's levels made in a register and used at once,
 * not kept in `w8`, with `pairs_i8`'s lanes, its integer sums a block, its float step a block and its reduction, so its
 * bytes; the row two ahead fetched into L2. `MEASURED` (measurements/2026-10-30_nn145_int8_bodies_dl380.md): GLM's gate+up
 * at a row an expert through `expert_groups_int8` on one core of a Gold 6132, 6.1 -> 8.9 GB/s, GLM's logits the same bit
 * for bit. */
static X86_AVX2__ZZPRIVATE_I8_TARGET inline void X86_AVX2__ZZPRIVATE_I8(row_few_i8)(const uint8_t* row, float scale, const int8_t* xq,
                                                                                 const float* xs, uint64_t x0, const uint32_t* rows,
                                                                                 const uint16_t* w, uint64_t count, uint64_t cols,
                                                                                 uint64_t out_rows, uint64_t r, uint16_t* out, float* acc,
                                                                                 unsigned int* over, __m512i t8) {
    const __m512i nibbles = _mm512_set1_epi16(0x0F0F);
    const uint64_t per = cols / NN__EXPERT__INT8_BLOCK;
    const int8_t* xr[2] = { xq, xq };
    const float* sr[2] = { xs, xs };
    for (uint64_t q = 0; q < count; ++q) {
        const uint64_t xrow = (uint64_t)rows[2u * q] - x0;
        xr[q] = xq + xrow * cols; sr[q] = xs + xrow * per;
    }
    __m512 f[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
    for (uint64_t B = 0; B < per; ++B) {
        __m512i s[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
        for (uint64_t k = B * NN__EXPERT__INT8_BLOCK; k < (B + 1u) * NN__EXPERT__INT8_BLOCK; k += 64u) {
            _mm_prefetch((const char*)(row + k / 2u + cols), _MM_HINT_T1);
            const __m512i u = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row + k / 2u)));
            const __m512i wv = _mm512_shuffle_epi8(t8, _mm512_ternarylogic_epi32(u, _mm512_slli_epi16(u, 4), nibbles, 0xA8));
            const __m512i aw = _mm512_abs_epi8(wv);
            const __mmask64 neg = _mm512_movepi8_mask(wv);
            for (uint64_t q = 0; q < count; ++q) {
                const __m512i xv = _mm512_loadu_si512((const void*)(xr[q] + k));
                s[q] = X86_AVX2__ZZPRIVATE_I8_SUM(s[q], aw, _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv));
            }
        }
        for (uint64_t q = 0; q < count; ++q) f[q] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s[q]), _mm512_set1_ps(sr[q][B]), f[q]);
    }
    for (uint64_t q = 0; q < count; ++q) {
        const float val = _mm512_reduce_add_ps(f[q]) * scale / 127.0f;
        const uint64_t at = (uint64_t)rows[2u * q + 1u] * out_rows + r;
        if (out) out[at] = nn__kernels__zzabi_to_half(val, over);
        else acc[at] += val * _cvtsh_ss(w[rows[2u * q]]);
    }
}
static X86_AVX2__ZZPRIVATE_I8_TARGET inline void X86_AVX2__ZZPRIVATE_I8(row_pairs_i8)(const uint8_t* row, float scale, const int8_t* xq,
                                                                                   const float* xs, uint64_t x0, const uint32_t* rows,
                                                                                   const uint16_t* w, uint64_t count, uint64_t cols,
                                                                                   uint64_t out_rows, uint64_t r, uint16_t* out, float* acc,
                                                                                   unsigned int* over, __m512i t8) {
    if (count <= 2u) {
        X86_AVX2__ZZPRIVATE_I8(row_few_i8)(row, scale, xq, xs, x0, rows, w, count, cols, out_rows, r, out, acc, over, t8);
        return;
    }
    int8_t w8[cols] __attribute__((aligned(64)));
    const __m512i nibbles = _mm512_set1_epi16(0x0F0F);
    for (uint64_t k = 0; k < cols; k += 64u) {
        _mm_prefetch((const char*)(row + k / 2u + cols), _MM_HINT_T1);
        const __m512i u = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row + k / 2u)));
        _mm512_store_si512((void*)(w8 + k), _mm512_shuffle_epi8(t8, _mm512_ternarylogic_epi32(u, _mm512_slli_epi16(u, 4), nibbles, 0xA8)));
    }
    const uint64_t per = cols / NN__EXPERT__INT8_BLOCK;
    for (uint64_t j = 0; j < count; j += 4u) {
        const uint64_t n = count - j < 4u ? count - j : 4u;
        const int8_t* xr[4] = { xq, xq, xq, xq };
        const float* sr[4] = { xs, xs, xs, xs };
        for (uint64_t q = 0; q < n; ++q) {
            const uint64_t xrow = (uint64_t)rows[2u * (j + q)] - x0;
            xr[q] = xq + xrow * cols; sr[q] = xs + xrow * per;
        }
        float v[4];
        if (n == 4u) X86_AVX2__ZZPRIVATE_I8(pairs_i8)(w8, cols, xr, sr, 4u, v);
        else X86_AVX2__ZZPRIVATE_I8(pairs_i8)(w8, cols, xr, sr, n, v);
        for (uint64_t q = 0; q < n; ++q) {
            const float val = v[q] * scale / 127.0f;
            const uint64_t at = (uint64_t)rows[2u * (j + q) + 1u] * out_rows + r;
            if (out) out[at] = nn__kernels__zzabi_to_half(val, over);
            else acc[at] += val * _cvtsh_ss(w[rows[2u * (j + q)]]);
        }
    }
}
static X86_AVX2__ZZPRIVATE_I8_TARGET void X86_AVX2__ZZPRIVATE_I8(expert_rows_i8)(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const __m512i t8 = X86_AVX2__ZZPRIVATE_I8(table_i8)();
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks(), row_bytes = c->cols / 2u;
    for (uint64_t r = c->out_rows * b / n; r < c->out_rows * (b + 1u) / n; ++r)
        X86_AVX2__ZZPRIVATE_I8(row_pairs_i8)(c->weights + r * row_bytes, x86_avx2__zzpackage_row_scale(c->luts, r), c->xq, c->xs, c->x0,
                                             c->rows, c->w, c->count, c->cols, c->out_rows, r, c->out, c->acc, c->over, t8);
}
static X86_AVX2__ZZPRIVATE_I8_TARGET void X86_AVX2__ZZPRIVATE_I8(expert_groups_i8)(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const __m512i t8 = X86_AVX2__ZZPRIVATE_I8(table_i8)();
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->total * b / n, t1 = c->total * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t r = t - base;
        X86_AVX2__ZZPRIVATE_I8(row_pairs_i8)((const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u),
                                             x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r), c->xq, c->xs, c->x0,
                                             c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->cols, c->g.out_rows[i], r,
                                             c->out + c->g.out_at[i], 0, c->over, t8);
    }
}
