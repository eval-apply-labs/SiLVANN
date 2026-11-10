#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <immintrin.h>
#if defined(_MSC_VER) && defined(__clang__)
/* clang-cl's immintrin.h, like MSVC's, may leave out the AVX-512 headers a build without -mavx512f does not ask for;
 * the functions below ask for AVX-512 themselves (X86_AVX2__ZZPRIVATE_AVX512), so their headers are named here.
 * ASSUMED for clang 19 until a Windows build: unneeded there, harmless (each header is guarded). */
#include <avx512fintrin.h>
#include <avx512bwintrin.h>
#include <avx512dqintrin.h>
#include <avx512vnniintrin.h>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>                 /* memset: a zero as the memory's */
#include "../primitives/pool.cuh"   /* the threads the rows are spread over */

/* ══ ⭐⭐ OVERRIDES — THE DOORS THIS FAMILY RUNS FASTER THAN THE GENERIC BODY ══════════════════════════
 * ⚖ *"a for the family bodies"*, in *"a folder for primitives and one for overrides so it is clear what is
 * expected to run and what are performance upgrades"*. Each function here has its door's signature and
 * takes its place in nn's table (`x86_avx2__override_doors`, called from `../entry.cuh` once the table is
 * built). ⛳ EVERY ONE FALLS BACK TO nn's GENERIC DOOR for what it does not cover — another width, a row
 * that is not aligned — so an override can only ever be faster, never different in what it accepts, and
 * deleting this file loses nothing but speed.
 * The kernels are the ones measured in `scripts/tq_cpu_probe.c`
 * (`measurements/2026-09-27_avx2_expert_gemv.md`): at 4 bits the levels through `pshufb` and F16C, at 8
 * bits gathered; and the int8 route's levels through one `pshufb` and `pmaddubsw`. */

/* ── the rows of one call, spread over the pool: each block a contiguous run of rows ───────────────── */
typedef struct x86_avx2__gemv_call {
    uint16_t* out; const uint8_t* weights; const uint8_t* luts; uint64_t row_bytes, rows, cols, d;
    const float* xf;                 /* x widened to floats, once a call (the exact route) */
    const int8_t* xq; const float* xs;   /* x quantised in blocks of 32, once a call (the int8 route) */
    unsigned int* over;
    const float* xd;                 /* x widened and split, each run of 32 its even columns then its odd (AVX-512) */
} x86_avx2__gemv_call;

static inline float x86_avx2__zzpackage_hsum(__m256 a) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(r);
}
static inline float x86_avx2__zzpackage_row_scale(const uint8_t* luts, uint64_t r) {
    return _cvtsh_ss((uint16_t)((uint32_t)luts[r * 2u] | ((uint32_t)luts[r * 2u + 1u] << 8)));
}
/* A GEMM row's scale: its own, or none at 16 bits, where a code is the weight (the routers). */
static inline float x86_avx2__zzprivate_gemm_scale(const uint8_t* luts, uint64_t d, uint64_t r) {
    return d == 16u ? 1.0f : x86_avx2__zzpackage_row_scale(luts, r);
}
static inline void x86_avx2__zzpackage_write(const x86_avx2__gemv_call* c, uint64_t r, float sum) {
    c->out[r] = nn__kernels__zzabi_to_half(sum, c->over);
}
/* The block's rows: an even share of them, by the pool's block number. */
static inline void x86_avx2__zzpackage_rows(const x86_avx2__gemv_call* c, uint64_t* r0, uint64_t* r1) {
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    *r0 = c->rows * b / n; *r1 = c->rows * (b + 1u) / n;
}

/* ── the exact route ──────────────────────────────────────────────────────────────────────────────── */
static void x86_avx2__zzprivate_exact4(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    const uint16_t* lev = nn__turboquant__zzabi_levels(4u);
    uint8_t lo_b[16], hi_b[16];
    for (int i = 0; i < 16; ++i) { lo_b[i] = (uint8_t)lev[i]; hi_b[i] = (uint8_t)(lev[i] >> 8); }
    const __m256i tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)lo_b));
    const __m256i thi = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)hi_b));
    const __m128i fifteen = _mm_set1_epi8(15);
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = c->weights + r * c->row_bytes;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (uint64_t k = 0; k < c->cols; k += 32u) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
            const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
            const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            const __m256i bl = _mm256_shuffle_epi8(tlo, idx), bh = _mm256_shuffle_epi8(thi, idx);
            const __m256i h0 = _mm256_unpacklo_epi8(bl, bh), h1 = _mm256_unpackhi_epi8(bl, bh);
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h0)),      _mm256_loadu_ps(c->xf + k),       a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h1)),      _mm256_loadu_ps(c->xf + k + 8u),  a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h0, 1)), _mm256_loadu_ps(c->xf + k + 16u), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h1, 1)), _mm256_loadu_ps(c->xf + k + 24u), a3);
        }
        const float sum = x86_avx2__zzpackage_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        x86_avx2__zzpackage_write(c, r, sum * x86_avx2__zzpackage_row_scale(c->luts, r));
    }
}
static float x86_avx2__zzprivate_levels8[256];
static inline float x86_avx2__zzprivate_dot8(const uint8_t* row, const float* xf, uint64_t cols);
static void x86_avx2__zzprivate_exact8(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r)
        x86_avx2__zzpackage_write(c, r, x86_avx2__zzprivate_dot8(c->weights + r * c->row_bytes, c->xf, c->cols)
                                        * x86_avx2__zzpackage_row_scale(c->luts, r));
}

/* 16 bits: a code is the weight as a half, and the row has no scale (the routers). */
static inline float x86_avx2__zzprivate_dot16(const uint16_t* row, const float* xf, uint64_t cols) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (uint64_t k = 0; k < cols; k += 32u) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + k))),       _mm256_loadu_ps(xf + k),       a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + k + 8u))),  _mm256_loadu_ps(xf + k + 8u),  a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + k + 16u))), _mm256_loadu_ps(xf + k + 16u), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + k + 24u))), _mm256_loadu_ps(xf + k + 24u), a3);
    }
    return x86_avx2__zzpackage_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}
static void x86_avx2__zzprivate_exact16(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r)
        x86_avx2__zzpackage_write(c, r, x86_avx2__zzprivate_dot16((const uint16_t*)(const void*)(c->weights + r * c->row_bytes), c->xf, c->cols));
}

/* ── the int8 route: nn's definition (`kernels.cuh`), x quantised once a call ─────────────────────── */
static void x86_avx2__zzprivate_int8_4(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    const int8_t* lev = nn__turboquant__zzabi_levels_i8(4u);
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)lev));
    const __m128i fifteen = _mm_set1_epi8(15);
    const __m256i ones = _mm256_set1_epi16(1);
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = c->weights + r * c->row_bytes;
        __m256 acc = _mm256_setzero_ps();
        for (uint64_t k = 0; k < c->cols; k += 32u) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
            const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
            const __m256i w = _mm256_shuffle_epi8(t8, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)));
            const __m256i xq = _mm256_loadu_si256((const __m256i*)(c->xq + k));
            const __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(xq, w));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones)), _mm256_set1_ps(c->xs[k / 32u]), acc);
        }
        x86_avx2__zzpackage_write(c, r, x86_avx2__zzpackage_hsum(acc) * x86_avx2__zzpackage_row_scale(c->luts, r) / 127.0f);
    }
}

/* ── the call's own scratch: x widened or quantised, grown as needed, one a calling thread ─────────── */
static __thread float*  x86_avx2__zzprivate_xf = 0;
static __thread float*  x86_avx2__zzprivate_xd = 0;
static __thread int8_t* x86_avx2__zzprivate_xq = 0;
static __thread float*  x86_avx2__zzprivate_xs = 0;
static __thread uint64_t x86_avx2__zzprivate_x_room = 0;
static __thread float*   x86_avx2__zzprivate_rows_room = 0;
static __thread uint64_t x86_avx2__zzprivate_rows_room_n = 0;

/* ⛔ AND GIVEN BACK WHEN ITS THREAD ENDS. The engine runs a launch's blocks on threads it starts for that launch and joins
 *   after it (`eng__launch__spread`), so a scratch a thread keeps lasts one launch, and kept past it is lost: `MEASURED`
 *   2026-10-14 on node03 (measurements/2026-10-14_v080rc1_on_node03.md), the 35B with its experts on the CPU lost 4.6 MB a
 *   prompt's chunk and 37 KB a token, without end. A key's destructor frees a thread's scratch as the thread ends; the
 *   family is never unloaded, so the destructor outlives every thread that names it. */
static sys__os__key   x86_avx2__zzprivate_scratch_key;
static sys__os__once x86_avx2__zzprivate_scratch_once = SYS__OS__ONCE_INIT;
static void x86_avx2__zzprivate_scratch_free(void* held) {
    (void)held;
    sys__os__memory_aligned_free(x86_avx2__zzprivate_xf); sys__os__memory_aligned_free(x86_avx2__zzprivate_xd);
    sys__os__memory_aligned_free(x86_avx2__zzprivate_xq); sys__os__memory_aligned_free(x86_avx2__zzprivate_xs);
    sys__os__memory_aligned_free(x86_avx2__zzprivate_rows_room);
    x86_avx2__zzprivate_xf = 0; x86_avx2__zzprivate_xd = 0; x86_avx2__zzprivate_xq = 0; x86_avx2__zzprivate_xs = 0;
    x86_avx2__zzprivate_x_room = 0; x86_avx2__zzprivate_rows_room = 0; x86_avx2__zzprivate_rows_room_n = 0;
}
static void x86_avx2__zzprivate_scratch_key_make(void) {
    (void)sys__os__key_make(&x86_avx2__zzprivate_scratch_key, x86_avx2__zzprivate_scratch_free);
}
static inline void x86_avx2__zzprivate_scratch_held(void) {
    sys__os__once_run(&x86_avx2__zzprivate_scratch_once, x86_avx2__zzprivate_scratch_key_make);
    sys__os__key_set(x86_avx2__zzprivate_scratch_key, (void*)1);
}

static inline bool x86_avx2__zzprivate_x_room_for(uint64_t cols) {
    if (cols <= x86_avx2__zzprivate_x_room) return true;
    x86_avx2__zzprivate_scratch_held();
    sys__os__memory_aligned_free(x86_avx2__zzprivate_xf); sys__os__memory_aligned_free(x86_avx2__zzprivate_xd);
    sys__os__memory_aligned_free(x86_avx2__zzprivate_xq); sys__os__memory_aligned_free(x86_avx2__zzprivate_xs);
    x86_avx2__zzprivate_xf = (float*)sys__os__memory_aligned(64, ((cols * 4u + 63u) / 64u) * 64u);
    x86_avx2__zzprivate_xd = (float*)sys__os__memory_aligned(64, ((cols * 4u + 63u) / 64u) * 64u);
    x86_avx2__zzprivate_xq = (int8_t*)sys__os__memory_aligned(64, ((cols + 63u) / 64u) * 64u);
    x86_avx2__zzprivate_xs = (float*)sys__os__memory_aligned(64, ((cols / 32u * 4u + 64u) / 64u) * 64u);
    x86_avx2__zzprivate_x_room = (x86_avx2__zzprivate_xf && x86_avx2__zzprivate_xd && x86_avx2__zzprivate_xq && x86_avx2__zzprivate_xs)
                               ? cols : 0u;
    return x86_avx2__zzprivate_x_room != 0u;
}

/* ⛳ A GEMM'S DECODED ROWS IN A SCRATCH OF THE THREAD'S OWN, kept and grown across calls and fetched once a tile: six rows
 *   of 32,768 columns are 768 KiB, no size for a stack frame — one is probed a page at a time on every call where GCC's
 *   stack-clash protection is on (by default here), and an array sized at run time is not portable C++. Null when there is
 *   no room, and the tile then takes its rows from the codes. */
static inline float* x86_avx2__zzprivate_rows_scratch(uint64_t floats) {
    if (floats > x86_avx2__zzprivate_rows_room_n) {
        x86_avx2__zzprivate_scratch_held();
        sys__os__memory_aligned_free(x86_avx2__zzprivate_rows_room);
        x86_avx2__zzprivate_rows_room = (float*)sys__os__memory_aligned(64, ((floats * 4u + 63u) / 64u) * 64u);
        x86_avx2__zzprivate_rows_room_n = x86_avx2__zzprivate_rows_room ? floats : 0u;
    }
    return x86_avx2__zzprivate_rows_room;
}

/* ── THE AVX-512 BODIES (below, ▶ `THE SAME DOORS ON AVX-512`), and whether they are installed ─────────────── */
/* ⛳ TUNED FOR THE AVX-512 CORES, as well as compiled for them: under generic tuning GCC folds a load into every FMA that
 *   uses it, so the GEMM's tile loaded each `x` once a row and ran out of loads — `MEASURED` (the tile alone, 4 threads,
 *   12 experts 1024 x 2048, 16 pairs): 250 GFLOP/s generic, 294 tuned. Tuning orders and picks instructions; any processor
 *   with the four extensions runs the code. */
#define X86_AVX2__ZZPRIVATE_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,tune=skylake-avx512")))
static bool x86_avx2__zzprivate_wide = false;    /* the processor has AVX-512 F, BW, VL and DQ, and it was not refused */
static bool x86_avx2__zzprivate_wide8 = false;   /* ...and the 8-bit levels are symmetric, as the 8-bit bodies need */
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_exact4_512(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_exact8_512(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_exact16_512(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_groups_512(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_groups_sum_512(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_expert_rows_512(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_expert_groups_512(void* a);
/* ⭐ THE INT8 BODIES, AND THEIR VNNI BUILD — one text, `int8_512.cuh`, built twice under these names (▶ where it is
 *   included). VNNI does a multiply-add of bytes in one instruction where AVX-512 takes three, the same sums:
 *   `MEASURED` (measurements/2026-11-06_vnni_dl380.md, a Gold 6252) a prompt's rows through the experts 1.2-1.3x. */
#define X86_AVX2__ZZPRIVATE_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,tune=skylake-avx512")))
#define X86_AVX2__ZZPRIVATE_I8_512(name)  x86_avx2__zzpackage_##name##_512
#define X86_AVX2__ZZPRIVATE_I8_VNNI(name) x86_avx2__zzpackage_##name##_vnni
static bool x86_avx2__zzprivate_vnni = false;    /* ...and AVX-512 VNNI, and `SILVANN_CPU_VNNI` is not `0` */
static X86_AVX2__ZZPRIVATE_AVX512 void X86_AVX2__ZZPRIVATE_I8_512(int8_4)(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void X86_AVX2__ZZPRIVATE_I8_512(groups_i8)(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void X86_AVX2__ZZPRIVATE_I8_512(groups_sum_i8)(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void X86_AVX2__ZZPRIVATE_I8_512(expert_rows_i8)(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void X86_AVX2__ZZPRIVATE_I8_512(expert_groups_i8)(void* a);
static X86_AVX2__ZZPRIVATE_VNNI void X86_AVX2__ZZPRIVATE_I8_VNNI(int8_4)(void* a);
static X86_AVX2__ZZPRIVATE_VNNI void X86_AVX2__ZZPRIVATE_I8_VNNI(groups_i8)(void* a);
static X86_AVX2__ZZPRIVATE_VNNI void X86_AVX2__ZZPRIVATE_I8_VNNI(groups_sum_i8)(void* a);
static X86_AVX2__ZZPRIVATE_VNNI void X86_AVX2__ZZPRIVATE_I8_VNNI(expert_rows_i8)(void* a);
static X86_AVX2__ZZPRIVATE_VNNI void X86_AVX2__ZZPRIVATE_I8_VNNI(expert_groups_i8)(void* a);
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_split_512(float* to, const uint16_t* x, uint64_t n);

/* `n` halves of `x` widened into `xf`, and with the AVX-512 bodies split into `xd` as well. */
static inline void x86_avx2__zzprivate_widen_x(const uint16_t* x, uint64_t n) {
    for (uint64_t k = 0; k < n; ++k) x86_avx2__zzprivate_xf[k] = _cvtsh_ss(x[k]);
    if (x86_avx2__zzprivate_wide) x86_avx2__zzprivate_split_512(x86_avx2__zzprivate_xd, x, n);
}
/* How many blocks to cut a call's rows into: a few a thread, so a slow one does not hold the rest.
 * ⛔⛔ NEVER NONE: a socket's pool is sized at boot to the cores then free, and on a busy machine that is ZERO — its
 *   thread count then 0, and `4 × 0` blocks is a launch that computes NOTHING. `MEASURED` (NN-59, 2026-10-06): booted
 *   beside a busy card or busy CPUs, GLM's experts on the CPUs were silently skipped — the layer added whatever its
 *   output buffer held — and the decode came out different, and exactly repeatable, against a quiet boot's. With an
 *   empty pool the caller runs the blocks itself (▶ `x86_avx2__zzprivate_run_body`), slowly and correctly. */
static inline uint32_t x86_avx2__zzprivate_blocks_for(uint64_t rows) {
    const uint32_t threads = x86_avx2__zzpackage_threads();
    const uint64_t want = 4ull * (threads > 0u ? threads : 1u);
    return (uint32_t)(rows < want ? rows : want);
}

/* ⭐ THE EXACT GEMV: 4, 8 and 16 bits with whole 32-column runs and a row a multiple of 16 bytes; anything else
 *   is the generic door. */
static void x86_avx2__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                               uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                               unsigned int* over) {
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols);
    const bool fits = (d == 4u || d == 8u || d == 16u) && cols % 32u == 0u && row_bytes % 16u == 0u
                   && row_bytes != 0u && rows <= w_room / row_bytes && (d == 16u || rows <= l_room / 2u)
                   && x86_avx2__zzprivate_x_room_for(cols);
    if (!fits) { nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over); return; }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_widen_x(x, cols);
    x86_avx2__gemv_call c = { out, weights, luts, row_bytes, rows, cols, d, x86_avx2__zzprivate_xf, 0, 0, over, x86_avx2__zzprivate_xd };
    void (*body)(void*) = d == 4u ? x86_avx2__zzprivate_exact4 : (d == 8u ? x86_avx2__zzprivate_exact8 : x86_avx2__zzprivate_exact16);
    if (x86_avx2__zzprivate_wide)
        body = d == 4u ? x86_avx2__zzprivate_exact4_512 : (d == 8u ? x86_avx2__zzprivate_exact8_512 : x86_avx2__zzprivate_exact16_512);
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows), body, &c);
}

/* A block `x[0..n)` quantised into `q` as nn's definition says — its largest magnitude over 127 the scale, each element
 * divided by it and rounded half away from zero — `n` a whole number of 32s; its scale answered. The gemvs' blocks of
 * 32 and the GEMM's of 256 (`NN__EXPERT__INT8_BLOCK`) alike: nn's scalar body took ~0.15 ms of a DeepSeek layer's 2 ms
 * on the CPU (measurements/2026-10-21_cpu_experts_node03.md). */
static inline float x86_avx2__zzprivate_quantise_run(const uint16_t* x, int8_t* q, uint32_t n) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 mx = _mm256_setzero_ps();
    for (uint32_t k = 0; k < n; k += 8u)
        mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + k))), absmask));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    const float m = _mm_cvtss_f32(m4), inv = m > 0.0f ? 127.0f / m : 0.0f;
    const __m256 vinv = _mm256_set1_ps(inv), half = _mm256_set1_ps(0.5f);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (uint32_t k = 0; k < n; k += 32u) {
        __m256i v[4];
        for (uint32_t i = 0; i < 4u; ++i) {
            const __m256 t = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + k + 8u * i))), vinv);
            /* half away from zero: the magnitude plus a half, truncated, the sign put back */
            v[i] = _mm256_sign_epi32(_mm256_cvttps_epi32(_mm256_add_ps(_mm256_and_ps(t, absmask), half)), _mm256_castps_si256(t));
        }
        const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
        _mm256_storeu_si256((__m256i*)(q + k), _mm256_permutevar8x32_epi32(p, order));
    }
    return m / 127.0f;
}

/* `x` quantised a block of 32 at a time into the call's scratch, as nn's int8 gemv says. */
static inline void x86_avx2__zzprivate_quantise_x(const uint16_t* x, uint64_t cols) {
    for (uint64_t b = 0; b < cols / 32u; ++b)
        x86_avx2__zzprivate_xs[b] = x86_avx2__zzprivate_quantise_run(x + b * 32u, x86_avx2__zzprivate_xq + b * 32u, 32u);
}

/* ⭐ THE int8 GEMV: 4 bits (the levels through one `pshufb`); anything else is the generic door. x is
 *   quantised here exactly as nn's definition says — blocks of 32, max|x|/127, half away from zero. */
static void x86_avx2__override_turboquant_gemv_int8(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                    unsigned int* over) {
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols);
    const bool fits = d == 4u && cols % 32u == 0u && row_bytes % 16u == 0u
                   && row_bytes != 0u && rows <= w_room / row_bytes && rows <= l_room / 2u
                   && x86_avx2__zzprivate_x_room_for(cols);
    if (!fits) { nn__turboquant__zzabi_launch_gemv_int8(out, weights, w_room, luts, l_room, x, d, rows, cols, over); return; }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_quantise_x(x, cols);
    x86_avx2__gemv_call c = { out, weights, luts, row_bytes, rows, cols, d, 0, x86_avx2__zzprivate_xq, x86_avx2__zzprivate_xs, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows),
                            x86_avx2__zzprivate_vnni ? X86_AVX2__ZZPRIVATE_I8_VNNI(int8_4) : x86_avx2__zzprivate_wide ? X86_AVX2__ZZPRIVATE_I8_512(int8_4)
                            : x86_avx2__zzprivate_int8_4, &c);
}

/* ══ ⭐⭐ THE LAYER'S OTHER DOORS — the grouped gemvs, the rotation, the DeltaNet step ═════════════════════════
 * ⚖ *"lets also do the avx2 bodies for the CPU family"*. `MEASURED` before them: a 35B token on this
 * family 333 ms, against ~22 ms for the bytes it reads — these four were nn's generic bodies. */

/* One row's product over `cols`, at 4 and at 8 bits, with `xf` the activation widened to floats. */
static inline float x86_avx2__zzprivate_dot4(const uint8_t* row, const float* xf, uint64_t cols, __m256i tlo, __m256i thi) {
    const __m128i fifteen = _mm_set1_epi8(15);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (uint64_t k = 0; k < cols; k += 32u) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
        const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
        const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
        const __m256i bl = _mm256_shuffle_epi8(tlo, idx), bh = _mm256_shuffle_epi8(thi, idx);
        const __m256i h0 = _mm256_unpacklo_epi8(bl, bh), h1 = _mm256_unpackhi_epi8(bl, bh);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h0)),      _mm256_loadu_ps(xf + k),       a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h1)),      _mm256_loadu_ps(xf + k + 8u),  a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h0, 1)), _mm256_loadu_ps(xf + k + 16u), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h1, 1)), _mm256_loadu_ps(xf + k + 24u), a3);
    }
    return x86_avx2__zzpackage_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}
/* ⛳ FOUR GATHERS AND FOUR SUMS A STEP of 32 columns (the fit check holds `cols` to whole runs of 32): with two, each sum
 *   waited on the FMA before it and the gathers could not overlap — `MEASURED` on the 35B on the CPU alone, ▶ the commit. */
static inline float x86_avx2__zzprivate_dot8(const uint8_t* row, const float* xf, uint64_t cols) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (uint64_t k = 0; k < cols; k += 32u) {
        const __m128i b = _mm_loadu_si128((const __m128i*)(row + k)), c = _mm_loadu_si128((const __m128i*)(row + k + 16u));
        const __m256i i0 = _mm256_cvtepu8_epi32(b), i1 = _mm256_cvtepu8_epi32(_mm_srli_si128(b, 8));
        const __m256i i2 = _mm256_cvtepu8_epi32(c), i3 = _mm256_cvtepu8_epi32(_mm_srli_si128(c, 8));
        a0 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i0, 4), _mm256_loadu_ps(xf + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i1, 4), _mm256_loadu_ps(xf + k + 8u), a1);
        a2 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i2, 4), _mm256_loadu_ps(xf + k + 16u), a2);
        a3 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i3, 4), _mm256_loadu_ps(xf + k + 24u), a3);
    }
    return x86_avx2__zzpackage_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}
static inline void x86_avx2__zzprivate_tables4(__m256i* tlo, __m256i* thi) {
    const uint16_t* lev = nn__turboquant__zzabi_levels(4u);
    uint8_t lo_b[16], hi_b[16];
    for (int i = 0; i < 16; ++i) { lo_b[i] = (uint8_t)lev[i]; hi_b[i] = (uint8_t)(lev[i] >> 8); }
    *tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)lo_b));
    *thi = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)hi_b));
}

typedef struct x86_avx2__groups_call {
    uint16_t* out; nn__turboquant__groups g; const float* xf; const uint16_t* w; const uint16_t* residual;
    uint64_t rows, cols; unsigned int* over;
    const float* xd;                 /* as `x86_avx2__gemv_call`'s */
} x86_avx2__groups_call;

static inline float x86_avx2__zzprivate_group_row(const nn__turboquant__groups* g, uint64_t i, uint64_t r, const float* xf,
                                                  uint64_t cols, __m256i tlo, __m256i thi) {
    const uint8_t* row = (const uint8_t*)(uintptr_t)g->codes[i] + r * (cols * g->d[i] / 8u);
    const float dot = g->d[i] == 4u ? x86_avx2__zzprivate_dot4(row, xf, cols, tlo, thi) : x86_avx2__zzprivate_dot8(row, xf, cols);
    return dot * x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)g->luts[i], r);
}

/* the gate_ups: every group's rows, one `x`, the flat row range of this block */
static void x86_avx2__zzprivate_groups(void* a) {
    const x86_avx2__groups_call* c = (const x86_avx2__groups_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->rows * b / n, t1 = c->rows * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = t - base;
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(x86_avx2__zzprivate_group_row(&c->g, i, r, c->xf, c->cols, tlo, thi), c->over);
    }
}
/* the downs: every group's row `r` weighted onto the residual */
static void x86_avx2__zzprivate_groups_sum(void* a) {
    const x86_avx2__groups_call* c = (const x86_avx2__groups_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->rows * b / n, r1 = c->rows * (b + 1u) / n;
    for (uint64_t r = r0; r < r1; ++r) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i)
            sum += x86_avx2__zzprivate_group_row(&c->g, i, r, c->xf + i * c->cols, c->cols, tlo, thi) * _cvtsh_ss(c->w[c->g.out_at[i]]);
        c->out[r] = nn__kernels__zzabi_to_half(_cvtsh_ss(c->residual[r]) + sum, c->over);
    }
}
static inline bool x86_avx2__zzprivate_groups_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i) if ((g->d[i] != 4u && g->d[i] != 8u) || g->rows[i] == 0u) return false;
    return true;
}
static void x86_avx2__override_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(cols)) {
        nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_widen_x(x, cols);
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    x86_avx2__groups_call c = { out, g, x86_avx2__zzprivate_xf, 0, 0, total, cols, over, x86_avx2__zzprivate_xd };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total), x86_avx2__zzprivate_wide ? x86_avx2__zzprivate_groups_512 : x86_avx2__zzprivate_groups, &c);
}

/* ⭐ THE GROUPS OVER `n` INPUTS (NN-161): each input through the one-input door above — input `k` at `x + k·cols`, its
 *   answers where nn's `groups_input` puts them — so each answer is the one-input door's bytes, and `n` inputs cost what `n`
 *   one-input calls do. nn's body went an input at a time over its own one-input body: nn's sums, and a GPU's loops. */
static void x86_avx2__override_gemv_groups_n(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                             uint64_t n, uint64_t out_stride, unsigned int* over) {
    for (uint64_t k = 0u; k < n; ++k) {
        uint16_t* o = out;
        const nn__turboquant__groups gk = nn__turboquant__groups_input(&g, &o, k, out_stride);
        x86_avx2__override_gemv_groups(o, gk, x + k * cols, cols, over);
    }
}
static void x86_avx2__override_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                               const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(g.count * cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_widen_x(x, g.count * cols);
    x86_avx2__groups_call c = { out, g, x86_avx2__zzprivate_xf, w, residual, rows, cols, over, x86_avx2__zzprivate_xd };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows),
                            x86_avx2__zzprivate_wide ? x86_avx2__zzprivate_groups_sum_512 : x86_avx2__zzprivate_groups_sum, &c);
}

/* ── THE GROUPED GEMVS IN INTEGERS — nn's int8 definition, as `turboquant_gemv_int8`'s override has it: `x` quantised once
 *   a call (every group's own `x` for the weighted sum), the 4-bit levels through one `pshufb`, int8 × int8 in `maddubs`. */
static inline float x86_avx2__zzprivate_dot_i8_4(const uint8_t* row, const int8_t* xq, const float* xs, uint64_t cols, __m256i t8) {
    const __m128i fifteen = _mm_set1_epi8(15);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (uint64_t k = 0; k < cols; k += 32u) {
        _mm_prefetch((const char*)(row + k / 2u + 1024u), _MM_HINT_T0);
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
        const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
        const __m256i w = _mm256_shuffle_epi8(t8, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)));
        const __m256i q = _mm256_loadu_si256((const __m256i*)(xq + k));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(q, w));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones)), _mm256_set1_ps(xs[k / 32u]), acc);
    }
    return x86_avx2__zzpackage_hsum(acc);
}
typedef struct x86_avx2__groups_i8_call {
    uint16_t* out; nn__turboquant__groups g; const int8_t* xq; const float* xs; const uint16_t* w; const uint16_t* residual;
    uint64_t rows, cols; unsigned int* over;
} x86_avx2__groups_i8_call;
static void x86_avx2__zzprivate_groups_i8(void* a) {
    const x86_avx2__groups_i8_call* c = (const x86_avx2__groups_i8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->rows * b / n, t1 = c->rows * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = t - base;
        const uint8_t* row = (const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u);
        const float dot = x86_avx2__zzprivate_dot_i8_4(row, c->xq, c->xs, c->cols, t8);
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(
            dot * x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r) / 127.0f, c->over);
    }
}
static void x86_avx2__zzprivate_groups_sum_i8(void* a) {
    const x86_avx2__groups_i8_call* c = (const x86_avx2__groups_i8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->rows * b / n, r1 = c->rows * (b + 1u) / n;
    for (uint64_t r = r0; r < r1; ++r) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i) {
            const uint8_t* row = (const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u);
            const float dot = x86_avx2__zzprivate_dot_i8_4(row, c->xq + i * c->cols, c->xs + i * (c->cols / 32u), c->cols, t8);
            sum += dot * x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r) / 127.0f * _cvtsh_ss(c->w[c->g.out_at[i]]);
        }
        c->out[r] = nn__kernels__zzabi_to_half(_cvtsh_ss(c->residual[r]) + sum, c->over);
    }
}
static inline bool x86_avx2__zzprivate_groups_i8_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i) if (g->d[i] != 4u || g->rows[i] == 0u) return false;
    return true;
}
static void x86_avx2__override_gemv_groups_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_i8_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_int8(out, g, x, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_quantise_x(x, cols);
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    x86_avx2__groups_i8_call c = { out, g, x86_avx2__zzprivate_xq, x86_avx2__zzprivate_xs, 0, 0, total, cols, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total),
                            x86_avx2__zzprivate_vnni ? X86_AVX2__ZZPRIVATE_I8_VNNI(groups_i8) : x86_avx2__zzprivate_wide ? X86_AVX2__ZZPRIVATE_I8_512(groups_i8)
                            : x86_avx2__zzprivate_groups_i8, &c);
}
static void x86_avx2__override_gemv_groups_sum_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                    const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_i8_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(g.count * cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_sum_int8(out, g, x, w, residual, rows, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_quantise_x(x, g.count * cols);
    x86_avx2__groups_i8_call c = { out, g, x86_avx2__zzprivate_xq, x86_avx2__zzprivate_xs, w, residual, rows, cols, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows),
                            x86_avx2__zzprivate_vnni ? X86_AVX2__ZZPRIVATE_I8_VNNI(groups_sum_i8) : x86_avx2__zzprivate_wide ? X86_AVX2__ZZPRIVATE_I8_512(groups_sum_i8)
                            : x86_avx2__zzprivate_groups_sum_i8, &c);
}

/* ── THE ROTATION: an fp32 butterfly a 512-block, a block a pool block, rounded to a half once ───────────── */
typedef struct x86_avx2__rotate_call {
    uint16_t* out; const uint16_t* in; const uint16_t* sign; unsigned int* over;
    uint64_t whole, n;             /* the whole 512-blocks' halves, and every half */
} x86_avx2__rotate_call;
/* Pool block `b`'s place and size: the whole 512-blocks first, then nn's tail — a 256-block where 256 or more are left, then
 *   a 128-block (▶ nn's `nn__hadamard__zzabi_body_rotate`). */
static inline void x86_avx2__zzprivate_rotate_block(const x86_avx2__rotate_call* c, uint64_t b, uint64_t* at, uint32_t* size) {
    const uint64_t whole_blocks = c->whole / 512u, rest = c->n - c->whole;
    if (b < whole_blocks) { *at = b * 512u; *size = 512u; return; }
    if (b == whole_blocks && rest >= 256u) { *at = c->whole; *size = 256u; return; }
    *at = c->whole + (rest >= 256u ? 256u : 0u); *size = 128u;
}
/* Block `b` of row `row` (rows of `n`, one after another): its butterfly in floats, rounded once. */
static inline void x86_avx2__zzprivate_rotate_one(const x86_avx2__rotate_call* c, uint64_t b, uint64_t row) {
    uint64_t at = 0u; uint32_t B = 512u;
    x86_avx2__zzprivate_rotate_block(c, b, &at, &B);
    at += row * c->n;
    float v[512] __attribute__((aligned(32)));
    for (uint32_t i = 0u; i < B; ++i) v[i] = _cvtsh_ss(c->in[at + i]) * _cvtsh_ss(c->sign[i]);
    for (uint32_t h = 1u; h < 8u; h <<= 1)
        for (uint32_t i = 0u; i < B; i += 2u * h)
            for (uint32_t j = i; j < i + h; ++j) { const float x = v[j], y = v[j + h]; v[j] = x + y; v[j + h] = x - y; }
    for (uint32_t h = 8u; h < B; h <<= 1)
        for (uint32_t i = 0u; i < B; i += 2u * h)
            for (uint32_t j = i; j < i + h; j += 8u) {
                const __m256 x = _mm256_load_ps(v + j), y = _mm256_load_ps(v + j + h);
                _mm256_store_ps(v + j, _mm256_add_ps(x, y)); _mm256_store_ps(v + j + h, _mm256_sub_ps(x, y));
            }
    const float inv = 1.0f / sqrtf((float)B);
    for (uint32_t i = 0u; i < B; ++i) c->out[at + i] = nn__kernels__zzabi_to_half(v[i] * inv, c->over);
}
static void x86_avx2__zzprivate_rotate(void* a) {
    x86_avx2__zzprivate_rotate_one((const x86_avx2__rotate_call*)a, (uint64_t)nn__silicon__block(), 0u);
}
/* The pool blocks one row of `n` takes: its whole 512-blocks, then nn's 256 and 128 tail. */
static inline uint64_t x86_avx2__zzprivate_rotate_blocks(uint64_t n) {
    const uint64_t rest = n % 512u;
    return n / 512u + (rest >= 256u ? 1u : 0u) + (rest % 256u == 128u ? 1u : 0u);
}
/* ⭐ Whole 512-blocks and nn's 256 and 128 tail, `out` apart from `in`: in place, nn's definition is its one-lane butterfly,
 *   rounded a stage at a time, so that is the generic door's. ⚖ THE TAIL ON THE FAST PATH TOO — `MEASURED` 2026-10-19 node-01:
 *   DeepSeek-V4.1's experts are 2304 wide (four 512s and a 256) and rotated a pick at a time, and the generic door took
 *   ~0.7 ms a pick on the CPU (a dot product an element), 4.0 of the experts' 6.1 ms a layer; whole 512-blocks, 13,824
 *   halves, 0.19 ms. */
static void x86_avx2__override_hadamard_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out;
    if (n < 512u || n % 128u != 0u || !apart) { nn__hadamard__zzabi_launch_rotate(out, in, sign, n, over); return; }
    const uint64_t whole = n / 512u * 512u, rest = n - whole;
    x86_avx2__rotate_call c = { out, in, sign, over, whole, n };
    (void)rest;
    x86_avx2__zzpackage_run((uint32_t)x86_avx2__zzprivate_rotate_blocks(n), x86_avx2__zzprivate_rotate, &c);
}

/* ⭐ THE LADDER ROW BY ROW — `hadamard_ladder_rows`: `rows` rows of `cols`, each rotated in nn's ladder (512s, a 256, a
 *   128), every row's blocks one pool launch, a row's arithmetic the rotation's above — so a row here is the bytes
 *   `hadamard_rotate` gives it alone. Where a launch a row took one: the picks of an expert 640 wide (512 and 128) are a
 *   launch each through the rotation, 10.5 µs each past the evaluator on node03's pool, ten picks 105 µs where one
 *   launch of 6,400 is 13 (`MEASURED` 2026-10-31, measurements/2026-10-31_nn141_flash_next_node03.md). Rows of 128 or
 *   wider and whole 128s, `out` apart from `in`; anything else is nn's generic door. */
typedef struct x86_avx2__ladder_call {
    x86_avx2__rotate_call row;     /* one row's: `n` its width */
    uint64_t per_row;              /* the pool blocks a row takes */
} x86_avx2__ladder_call;
static void x86_avx2__zzprivate_ladder(void* a) {
    const x86_avx2__ladder_call* c = (const x86_avx2__ladder_call*)a;
    const uint64_t b = (uint64_t)nn__silicon__block();
    x86_avx2__zzprivate_rotate_one(&c->row, b % c->per_row, b / c->per_row);
}
static void x86_avx2__override_hadamard_ladder_rows(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t cols,
                                                   uint64_t rows, unsigned int* over) {
    const uint64_t n = cols * rows;
    const bool apart = out + n <= in || in + n <= out;
    const uint64_t per_row = cols >= 128u && cols % 128u == 0u ? x86_avx2__zzprivate_rotate_blocks(cols) : 0u;
    if (per_row == 0u || rows == 0u || !apart || per_row * rows > 0xFFFFFFFFull) {
        nn__hadamard__zzabi_launch_ladder_rows(out, in, sign, cols, rows, over);
        return;
    }
    x86_avx2__ladder_call c = { { out, in, sign, over, cols / 512u * 512u, cols }, per_row };
    x86_avx2__zzpackage_run((uint32_t)(per_row * rows), x86_avx2__zzprivate_ladder, &c);
}

/* ⭐ A ZERO AND AN ADD AS THE MEMORY'S, NOT AS A LANE AN ELEMENT — nn's bodies give item `i` to block `i mod blocks`, so
 *   every cache line is written by every thread: on node03's pool a zero of 12,800 halves took 99 µs and an add of 6,400
 *   43 µs past the evaluator (`MEASURED` 2026-10-31, measurements/2026-10-31_nn141_flash_next_node03.md), each launched
 *   twice a layer by ai_qwen_3's experts split over the sockets. Here a vector the caller covers in a few microseconds is
 *   the caller's own, and a larger one is cut into runs of whole cache lines, a run a pool block. The add's arithmetic is
 *   nn's — both halves widened, summed in a float, rounded once by nn's own conversion, its overflow in the fault word —
 *   so its bytes are nn's body's. */
#define X86_AVX2__ZZPRIVATE_CALLER_HALVES (64u * 1024u)    /* 128 KB: under it the caller's, over it the pool's */
#define X86_AVX2__ZZPRIVATE_RUN_HALVES    (16u * 1024u)    /* a pool block's run, 32 KB */
typedef struct x86_avx2__pointwise_call {
    uint16_t* z; const uint16_t* x; const uint16_t* y; uint64_t n; unsigned int* over;
} x86_avx2__pointwise_call;
/* Eight at a time through F16C: its conversion with the immediate's rounding is IEEE's to the nearest even, subnormals and
 * NaNs included, which is nn's (`nn__primitives__zzpackage_float_to_half`) — and nn flags a finite sum that becomes an
 * infinite half, as here. The tail is nn's conversion itself. */
static inline void x86_avx2__zzprivate_add_run(const x86_avx2__pointwise_call* c, uint64_t first, uint64_t end) {
    const __m256 inf = _mm256_set1_ps(INFINITY), magnitude = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int hit = 0;
    uint64_t i = first;
    for (; i + 8u <= end; i += 8u) {
        const __m256 sum = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(c->x + i))),
                                         _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(c->y + i))));
        const __m128i h = _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(c->z + i), h);
        const __m256 finite = _mm256_cmp_ps(_mm256_and_ps(sum, magnitude), inf, _CMP_LT_OQ);
        const __m256 infinite = _mm256_cmp_ps(_mm256_and_ps(_mm256_cvtph_ps(h), magnitude), inf, _CMP_EQ_OQ);
        hit |= _mm256_movemask_ps(_mm256_and_ps(finite, infinite));
    }
    for (; i < end; ++i) c->z[i] = nn__kernels__zzabi_to_half(_cvtsh_ss(c->x[i]) + _cvtsh_ss(c->y[i]), c->over);
    if (hit) *c->over = NN__KERNELS__OVERFLOWED;
}
static void x86_avx2__zzprivate_add(void* a) {
    const x86_avx2__pointwise_call* c = (const x86_avx2__pointwise_call*)a;
    const uint64_t first = (uint64_t)nn__silicon__block() * X86_AVX2__ZZPRIVATE_RUN_HALVES;
    x86_avx2__zzprivate_add_run(c, first, first + X86_AVX2__ZZPRIVATE_RUN_HALVES < c->n ? first + X86_AVX2__ZZPRIVATE_RUN_HALVES : c->n);
}
static void x86_avx2__zzprivate_zero(void* a) {
    const x86_avx2__pointwise_call* c = (const x86_avx2__pointwise_call*)a;
    const uint64_t first = (uint64_t)nn__silicon__block() * X86_AVX2__ZZPRIVATE_RUN_HALVES;
    const uint64_t end = first + X86_AVX2__ZZPRIVATE_RUN_HALVES < c->n ? first + X86_AVX2__ZZPRIVATE_RUN_HALVES : c->n;
    memset(c->z + first, 0, (size_t)(2u * (end - first)));
}
static void x86_avx2__override_vector_zero(uint16_t* v, uint64_t n) {
    if (n <= X86_AVX2__ZZPRIVATE_CALLER_HALVES) { memset(v, 0, (size_t)(2u * n)); return; }
    x86_avx2__pointwise_call c = { v, 0, 0, n, 0 };
    x86_avx2__zzpackage_run((uint32_t)((n + X86_AVX2__ZZPRIVATE_RUN_HALVES - 1u) / X86_AVX2__ZZPRIVATE_RUN_HALVES), x86_avx2__zzprivate_zero, &c);
}
static void x86_avx2__override_vector_add(uint16_t* z, const uint16_t* x, const uint16_t* y, uint64_t n, unsigned int* over) {
    x86_avx2__pointwise_call c = { z, x, y, n, over };
    if (n <= X86_AVX2__ZZPRIVATE_CALLER_HALVES) { x86_avx2__zzprivate_add_run(&c, 0u, n); return; }
    x86_avx2__zzpackage_run((uint32_t)((n + X86_AVX2__ZZPRIVATE_RUN_HALVES - 1u) / X86_AVX2__ZZPRIVATE_RUN_HALVES), x86_avx2__zzprivate_add, &c);
}

/* ⭐ THE CLAMPED SWIGLU A RUN OF ELEMENTS A POOL BLOCK — `act = silu(min(g, limit)) · clamp(u, −limit, limit)` for `pairs`
 *   pairs of `n`. nn's body gives element `i` to block `i mod blocks` (a block a thread here, ▶ the zero and the add above)
 *   and divides for its pair at every element; here a block takes a run of X86_AVX2__ZZPRIVATE_SWIGLU_RUN elements in
 *   order, eight at a time through F16C. The arithmetic is nn's: the clamps (a NaN kept, as nn's comparisons keep it — MINPS
 *   and MAXPS answer their second operand for a NaN), `nn__silicon__expf` an element (libm's `expf` here), the quotient and
 *   the product in floats, rounded once — so its bytes are nn's body's. */
#define X86_AVX2__ZZPRIVATE_SWIGLU_RUN 512u
typedef struct x86_avx2__swiglu_call {
    uint16_t* act; const uint16_t* gu; uint64_t pairs, n; float limit; unsigned int* over;
} x86_avx2__swiglu_call;
static void x86_avx2__zzprivate_swiglu_clamped(void* a) {
    const x86_avx2__swiglu_call* c = (const x86_avx2__swiglu_call*)a;
    const uint64_t all = c->pairs * c->n, first = (uint64_t)nn__silicon__block() * X86_AVX2__ZZPRIVATE_SWIGLU_RUN;
    const uint64_t end = first + X86_AVX2__ZZPRIVATE_SWIGLU_RUN < all ? first + X86_AVX2__ZZPRIVATE_SWIGLU_RUN : all;
    const __m256 top = _mm256_set1_ps(c->limit), bottom = _mm256_set1_ps(-c->limit), one = _mm256_set1_ps(1.0f);
    const __m256 inf = _mm256_set1_ps(INFINITY), magnitude = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int hit = 0;
    for (uint64_t i = first; i < end;) {
        const uint64_t p = i / c->n, stop = (p + 1u) * c->n < end ? (p + 1u) * c->n : end, len = stop - i;
        const uint16_t* g = c->gu + i + p * c->n;        /* element i's gate: pair p's gates at i − p·n, its ups n on */
        const uint16_t* u = g + c->n;
        uint16_t* o = c->act + i;
        uint64_t k = 0u;
        for (; k + 8u <= len; k += 8u) {
            const __m256 gv = _mm256_min_ps(top, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(g + k))));
            const __m256 uv = _mm256_max_ps(bottom, _mm256_min_ps(top, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(u + k)))));
            float gf[8], e[8];
            _mm256_storeu_ps(gf, gv);
            for (int t = 0; t < 8; ++t) e[t] = nn__silicon__expf(-gf[t]);
            const __m256 v = _mm256_mul_ps(_mm256_div_ps(gv, _mm256_add_ps(one, _mm256_loadu_ps(e))), uv);
            const __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i*)(o + k), h);
            const __m256 finite = _mm256_cmp_ps(_mm256_and_ps(v, magnitude), inf, _CMP_LT_OQ);
            const __m256 infinite = _mm256_cmp_ps(_mm256_and_ps(_mm256_cvtph_ps(h), magnitude), inf, _CMP_EQ_OQ);
            hit |= _mm256_movemask_ps(_mm256_and_ps(finite, infinite));
        }
        for (; k < len; ++k) {
            float gs = _cvtsh_ss(g[k]), us = _cvtsh_ss(u[k]);
            gs = gs > c->limit ? c->limit : gs;
            us = us > c->limit ? c->limit : (us < -c->limit ? -c->limit : us);
            o[k] = nn__kernels__zzabi_to_half(gs / (1.0f + nn__silicon__expf(-gs)) * us, c->over);
        }
        i = stop;
    }
    if (hit) *c->over = NN__KERNELS__OVERFLOWED;
}
static void x86_avx2__override_swiglu_clamped(uint16_t* act, const uint16_t* gu, uint64_t pairs, uint64_t n, float limit,
                                              unsigned int* over) {
    if (pairs == 0u || n == 0u) {
        nn__swiglu__zzabi_launch_clamped(act, gu, pairs, n, limit, over);
        return;
    }
    x86_avx2__swiglu_call c = { act, gu, pairs, n, limit, over };
    x86_avx2__zzpackage_run((uint32_t)((pairs * n + X86_AVX2__ZZPRIVATE_SWIGLU_RUN - 1u) / X86_AVX2__ZZPRIVATE_SWIGLU_RUN),
                            x86_avx2__zzprivate_swiglu_clamped, &c);
}

/* ── THE DELTANET STEP: a head a pool block, eight of its state's columns at a time ──────────────────────── */
typedef struct x86_avx2__step_call {
    float* S; const uint16_t* conved; const uint16_t* z; const uint16_t* beta; const uint16_t* g; const uint16_t* w;
    uint16_t* out; uint64_t k_heads, v_heads, d, rows; unsigned int gate; unsigned int* over;
} x86_avx2__step_call;
/* One value head's step: its state `Sh`, its `q`, `k`, `v` and `zh`, its gate and decay, its readout gated by silu(z) or
 * σ(z) as `gate` says (`NN__DELTANET__GATE_*`, nn's). ⛳ Shared by the one-position door, the many-row one and the one
 * with its gate named, so a position decoded and the same position read as a prompt row are the same bytes, and the
 * gate's door with silu(z) the plain door's. */
static inline void x86_avx2__zzprivate_step_head(float* Sh, const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* zh,
                                                 uint16_t beta_h, uint16_t g_h, const uint16_t* w, uint16_t* oh, uint64_t d,
                                                 unsigned int gate, unsigned int* over) {
    float qf[256], kf[256];
    float qq = 0.0f, kk = 0.0f;
    for (uint64_t i = 0u; i < d; ++i) { qf[i] = _cvtsh_ss(q[i]); kf[i] = _cvtsh_ss(k[i]); qq += qf[i] * qf[i]; kk += kf[i] * kf[i]; }
    const float inv_q = 1.0f / sqrtf(qq + 1e-6f) / sqrtf((float)d), inv_k = 1.0f / sqrtf(kk + 1e-6f);
    const __m256 decay = _mm256_set1_ps(expf(_cvtsh_ss(g_h))), b = _mm256_set1_ps(_cvtsh_ss(beta_h));
    float oo = 0.0f;
    for (uint64_t j = 0u; j < d; j += 8u) {
        __m256 kv = _mm256_setzero_ps();
        for (uint64_t i = 0u; i < d; ++i) kv = _mm256_fmadd_ps(_mm256_loadu_ps(Sh + i * d + j), _mm256_set1_ps(kf[i]), kv);
        float vj[8];
        for (int t = 0; t < 8; ++t) vj[t] = _cvtsh_ss(v[j + t]);
        const __m256 delta = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(vj), _mm256_mul_ps(decay, _mm256_mul_ps(kv, _mm256_set1_ps(inv_k)))), b);
        __m256 o = _mm256_setzero_ps();
        for (uint64_t i = 0u; i < d; ++i) {
            float* row = Sh + i * d + j;
            const __m256 s = _mm256_fmadd_ps(decay, _mm256_loadu_ps(row), _mm256_mul_ps(_mm256_set1_ps(kf[i] * inv_k), delta));
            _mm256_storeu_ps(row, s);
            o = _mm256_fmadd_ps(s, _mm256_set1_ps(qf[i]), o);
        }
        float of[8];
        _mm256_storeu_ps(of, _mm256_mul_ps(o, _mm256_set1_ps(inv_q)));
        for (int t = 0; t < 8; ++t) {
            oh[j + t] = nn__kernels__zzabi_to_half(of[t], over);
            const float r = _cvtsh_ss(oh[j + t]);
            oo += r * r;
        }
    }
    const float scale = 1.0f / sqrtf(oo / (float)d + 1e-6f);
    for (uint64_t j = 0u; j < d; ++j) {
        const float zr = _cvtsh_ss(zh[j]);
        const float zc = zr < -88.0f ? -88.0f : (zr > 88.0f ? 88.0f : zr);
        const float gz = gate == NN__DELTANET__GATE_SIGMOID ? 1.0f / (1.0f + expf(-zc)) : zc / (1.0f + expf(-zc));
        oh[j] = nn__kernels__zzabi_to_half(_cvtsh_ss(oh[j]) * scale * _cvtsh_ss(w[j]) * gz, over);
    }
}
/* Head `block` through each of the call's rows in order, its state carried from one to the next in place: row `t` reads
 * `conved`, `z`, `beta` and `g` at its own row and writes `out`'s. One row is the one-position step. */
static void x86_avx2__zzprivate_steps(void* a) {
    const x86_avx2__step_call* c = (const x86_avx2__step_call*)a;
    const uint64_t h = nn__silicon__block(), d = c->d, rep = c->v_heads / c->k_heads;
    const uint64_t ch = (2ull * c->k_heads + c->v_heads) * d, vd = c->v_heads * d;
    for (uint64_t t = 0u; t < c->rows; ++t) {
        const uint16_t* r = c->conved + t * ch;
        x86_avx2__zzprivate_step_head(c->S + h * d * d, r + (h / rep) * d, r + c->k_heads * d + (h / rep) * d, r + 2ull * c->k_heads * d + h * d,
                                      c->z + t * vd + h * d, c->beta[t * c->v_heads + h], c->g[t * c->v_heads + h], c->w,
                                      c->out + t * vd + h * d, d, c->gate, c->over);
    }
}
static inline bool x86_avx2__zzprivate_step_fit(uint64_t k_heads, uint64_t v_heads, uint64_t head_dim) {
    return head_dim != 0u && head_dim % 8u == 0u && head_dim <= 256u && k_heads != 0u && v_heads % k_heads == 0u;
}
static void x86_avx2__override_deltanet_step(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g,
                                             const uint16_t* w, uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t head_dim,
                                             unsigned int* over) {
    if (!x86_avx2__zzprivate_step_fit(k_heads, v_heads, head_dim)) {
        nn__deltanet__zzabi_launch_step(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, over); return;
    }
    x86_avx2__step_call c = { S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, 1u, NN__DELTANET__GATE_SILU, over };
    x86_avx2__zzpackage_run((uint32_t)v_heads, x86_avx2__zzprivate_steps, &c);
}
/* ⭐ The prompt's rows: the step's own arithmetic a row at a time, so the two doors agree byte for byte. */
static void x86_avx2__override_deltanet_steps(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g,
                                              const uint16_t* w, uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t head_dim,
                                              uint64_t rows, unsigned int* over) {
    if (!x86_avx2__zzprivate_step_fit(k_heads, v_heads, head_dim)) {
        nn__deltanet__zzabi_launch_steps(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, over); return;
    }
    x86_avx2__step_call c = { S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, NN__DELTANET__GATE_SILU, over };
    x86_avx2__zzpackage_run((uint32_t)v_heads, x86_avx2__zzprivate_steps, &c);
}
/* ⭐ The same rows with the readout's gate named (Flash-Next's σ(z)): the one body, so silu(z) here is the plain door's
 * bytes. */
static void x86_avx2__override_deltanet_steps_gate(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                   const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                   uint64_t v_heads, uint64_t head_dim, uint64_t rows, unsigned int gate,
                                                   unsigned int* over) {
    if (!x86_avx2__zzprivate_step_fit(k_heads, v_heads, head_dim)
        || (gate != NN__DELTANET__GATE_SILU && gate != NN__DELTANET__GATE_SIGMOID)) {
        nn__deltanet__zzabi_launch_steps_gate(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, gate, over); return;
    }
    x86_avx2__step_call c = { S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, gate, over };
    x86_avx2__zzpackage_run((uint32_t)v_heads, x86_avx2__zzprivate_steps, &c);
}

/* ── THE MAP'S LOGITS OVER STREAMS OF FLOATS (DeepSeek's `hyper_logits_fp32`): a row a pool block, its sums in double ──
 * A row's logit is `Σ fn·s / rms(s)` over every stream's every element — 20,480 at DeepSeek-V4.1's shape — and a stream
 * may be as large as a sink token's (300,000 and more). nn's body on this family's one lane adds them in turn in a float:
 * 4.7e-05 of the largest from the sums in double (`MEASURED` node03 2026-11-08, test/src_gemv_27b_bench.cpp, nn's host
 * body's line). Here a row's two sums, the squares and the dot, are doubles — a half's float and a stream's float
 * multiply exactly in one — over four lanes and two sums each, folded in one order, and the logit rounded to a float
 * once. The squares are each row's own, the same sum in every row. */
typedef struct x86_avx2__hyper_logits_call {
    float* lg; const float* streams; const uint16_t* fn; uint64_t n; double eps;
} x86_avx2__hyper_logits_call;
static void x86_avx2__zzprivate_hyper_logits_fp32(void* a) {
    const x86_avx2__hyper_logits_call* c = (const x86_avx2__hyper_logits_call*)a;
    const uint64_t r = nn__silicon__block(), n = c->n;
    const float* s = c->streams;
    const uint16_t* f = c->fn + r * n;
    __m256d q0 = _mm256_setzero_pd(), q1 = _mm256_setzero_pd(), d0 = _mm256_setzero_pd(), d1 = _mm256_setzero_pd();
    uint64_t i = 0u;
    for (; i + 8u <= n; i += 8u) {
        const __m256 sv = _mm256_loadu_ps(s + i), fv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(f + i)));
        const __m256d slo = _mm256_cvtps_pd(_mm256_castps256_ps128(sv)), shi = _mm256_cvtps_pd(_mm256_extractf128_ps(sv, 1));
        const __m256d flo = _mm256_cvtps_pd(_mm256_castps256_ps128(fv)), fhi = _mm256_cvtps_pd(_mm256_extractf128_ps(fv, 1));
        q0 = _mm256_fmadd_pd(slo, slo, q0); q1 = _mm256_fmadd_pd(shi, shi, q1);
        d0 = _mm256_fmadd_pd(flo, slo, d0); d1 = _mm256_fmadd_pd(fhi, shi, d1);
    }
    double qs[4], ds[4];
    _mm256_storeu_pd(qs, _mm256_add_pd(q0, q1)); _mm256_storeu_pd(ds, _mm256_add_pd(d0, d1));
    double sq = (qs[0] + qs[1]) + (qs[2] + qs[3]), dot = (ds[0] + ds[1]) + (ds[2] + ds[3]);
    for (; i < n; ++i) {
        const double v = s[i];
        sq += v * v;
        dot += (double)_cvtsh_ss(f[i]) * v;
    }
    c->lg[r] = (float)(dot / sqrt(sq / (double)n + c->eps));
}
static void x86_avx2__override_hyper_logits_fp32(float* lg, const float* streams, const uint16_t* fn, uint64_t hidden, uint64_t mult,
                                                 float norm_eps) {
    const uint64_t rows = NN__HYPER__MIX(mult);
    if (hidden == 0u || mult == 0u || mult > NN__HYPER__MULT_MAX) {
        nn__hyper__zzabi_launch_logits_fp32(lg, streams, fn, hidden, mult, norm_eps); return;
    }
    x86_avx2__hyper_logits_call c = { lg, streams, fn, mult * hidden, (double)norm_eps };
    x86_avx2__zzpackage_run((uint32_t)rows, x86_avx2__zzprivate_hyper_logits_fp32, &c);
}

/* ── THE DELTANET'S CONV: a pool block a run of channels, the positions in turn ─────────────────────────────────────────
 * nn's definition (`kernels.cuh`): each channel's window of the last `NN__DELTANET__CONV_WINDOW` inputs and its taps, the
 * output `silu(Σ taps · window + tap · x)` as a half, the window slid by the input. nn's body gives a lane a channel and
 * strides the channels across the blocks, so on one lane a block read each row of `x` a channel at a time — `MEASURED` (the
 * family's launch profile, the 35B's prompt of 1,037 on the CPU alone) 2.8 s, 19 ms a call. Here a block takes 64
 * channels: their taps and windows once as floats, then each position's 64 inputs together, eight channels a vector, the
 * sum in nn's order a channel (each tap a multiply-add onto it), libm's `expf` for the silu as nn's. ⛳ The one-position
 * door is the same body at one row, so the two forms agree to the bit. */
#define X86_AVX2__ZZPRIVATE_CONV_RUN 64u
typedef struct x86_avx2__conv_call {
    uint16_t* o; uint16_t* s; const uint16_t* w; const uint16_t* x; uint64_t ch, rows; unsigned int* over;
} x86_avx2__conv_call;
static void x86_avx2__zzprivate_conv(void* a) {
    const x86_avx2__conv_call* c = (const x86_avx2__conv_call*)a;
    const uint64_t c0 = (uint64_t)nn__silicon__block() * X86_AVX2__ZZPRIVATE_CONV_RUN;
    const uint64_t n = c->ch - c0 < X86_AVX2__ZZPRIVATE_CONV_RUN ? c->ch - c0 : X86_AVX2__ZZPRIVATE_CONV_RUN;
    float wt[NN__DELTANET__CONV_TAPS][X86_AVX2__ZZPRIVATE_CONV_RUN] __attribute__((aligned(32)));
    float st[NN__DELTANET__CONV_WINDOW][X86_AVX2__ZZPRIVATE_CONV_RUN] __attribute__((aligned(32)));
    for (uint64_t j = 0; j < n; ++j) {
        for (unsigned t = 0u; t < NN__DELTANET__CONV_TAPS; ++t) wt[t][j] = _cvtsh_ss(c->w[(c0 + j) * NN__DELTANET__CONV_TAPS + t]);
        for (unsigned t = 0u; t < NN__DELTANET__CONV_WINDOW; ++t) st[t][j] = _cvtsh_ss(c->s[(c0 + j) * NN__DELTANET__CONV_WINDOW + t]);
    }
    float acc[X86_AVX2__ZZPRIVATE_CONV_RUN] __attribute__((aligned(32))), xf[X86_AVX2__ZZPRIVATE_CONV_RUN] __attribute__((aligned(32)));
    for (uint64_t r = 0; r < c->rows; ++r) {
        const uint16_t* xr = c->x + r * c->ch + c0;
        uint16_t* orow = c->o + r * c->ch + c0;
        uint64_t j = 0;
        for (; j + 8u <= n; j += 8u) {
            const __m256 xv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(xr + j)));
            __m256 s = _mm256_fmadd_ps(_mm256_load_ps(wt[0] + j), _mm256_load_ps(st[0] + j), _mm256_setzero_ps());
            for (unsigned t = 1u; t < NN__DELTANET__CONV_WINDOW; ++t) s = _mm256_fmadd_ps(_mm256_load_ps(wt[t] + j), _mm256_load_ps(st[t] + j), s);
            s = _mm256_fmadd_ps(_mm256_load_ps(wt[NN__DELTANET__CONV_WINDOW] + j), xv, s);
            _mm256_store_ps(acc + j, s); _mm256_store_ps(xf + j, xv);
        }
        for (; j < n; ++j) {
            xf[j] = _cvtsh_ss(xr[j]);
            float s = 0.0f;
            for (unsigned t = 0u; t < NN__DELTANET__CONV_WINDOW; ++t) s = fmaf(wt[t][j], st[t][j], s);
            acc[j] = fmaf(wt[NN__DELTANET__CONV_WINDOW][j], xf[j], s);
        }
        for (j = 0; j < n; ++j) {
            orow[j] = nn__kernels__zzabi_to_half(acc[j] / (1.0f + expf(-acc[j])), c->over);
            for (unsigned t = 0u; t + 1u < NN__DELTANET__CONV_WINDOW; ++t) st[t][j] = st[t + 1u][j];
            st[NN__DELTANET__CONV_WINDOW - 1u][j] = xf[j];
        }
    }
    for (uint64_t j = 0; j < n; ++j)
        for (unsigned t = 0u; t < NN__DELTANET__CONV_WINDOW; ++t) c->s[(c0 + j) * NN__DELTANET__CONV_WINDOW + t] = _cvtss_sh(st[t][j], 0);
}
static void x86_avx2__override_deltanet_conv_steps(uint16_t* o, uint16_t* s, const uint16_t* w, const uint16_t* x, uint64_t ch,
                                                   uint64_t rows, unsigned int* over) {
    if (ch == 0u || rows == 0u) { nn__deltanet__zzabi_launch_conv_steps(o, s, w, x, ch, rows, over); return; }
    x86_avx2__zzpackage_start();
    x86_avx2__conv_call c = { o, s, w, x, ch, rows, over };
    x86_avx2__zzpackage_run((uint32_t)((ch + X86_AVX2__ZZPRIVATE_CONV_RUN - 1u) / X86_AVX2__ZZPRIVATE_CONV_RUN), x86_avx2__zzprivate_conv, &c);
}
static void x86_avx2__override_deltanet_conv_step(uint16_t* o, uint16_t* s, const uint16_t* w, const uint16_t* x, uint64_t ch,
                                                  unsigned int* over) {
    x86_avx2__override_deltanet_conv_steps(o, s, w, x, ch, 1u, over);
}

/* ── ⭐⭐ THE EXPERT-MAJOR GEMM — a row of the matrix decoded once to floats, then every pair's dot with it ───────
 * nn's definitions (`kernels.cuh`, the expert-major GEMM). A pool block takes a run of the matrix's rows; for
 * each it decodes the row into floats once — the levels as the gemv finds them — and then takes the row's dot
 * product with every pair's `x`, which was widened to floats once for the call. So a row's codes are read and
 * decoded once however many pairs use it, and one block owns a row for every pair, so the fp32 sum needs no
 * lock. Covers 4, 8 and 16 bits (the routers: a code is the weight, no scale) over whole blocks of 32 columns up to
 * 32,768; anything else is the generic door. */
#define X86_AVX2__ZZPRIVATE_GEMM_COLS 32768u
/* ⛳ THE PAIRS IN BLOCKS THAT FIT HALF THE L2: a matrix row meets every pair's `x`, and a prompt's 256 pairs of 2,048 to
 *   17,408 columns are 2-17 MB — out of any core's L2, so every row's tile read them from L3 (`MEASURED`, the sweep's dense
 *   prompt tiles, 256 rows: 0.06-0.12 TFLOP/s against the experts' 0.22). A block of pairs whose `x` fits half the L2
 *   (as the processor reports it, 256 KiB where it does not), at least twelve, goes through every row of the block's run
 *   before the next; a row's dots are the same arithmetic, so the answer is the same bits. */
static uint64_t x86_avx2__zzprivate_l2 = 256u * 1024u;
/* ⛳ THE INT8 GEMM'S ROW OF LEVELS ON THE STACK AT THE ROW'S OWN WIDTH (`cols` bytes): a frame of the widest row would be
 *   probed a page at a time on every call where GCC's stack-clash protection is on (by default here), and a buffer a thread
 *   in its place was slower — the int8 GEMM's weighted sums 368 -> 574 us (`MEASURED`, the sweep). The float GEMM's six
 *   decoded rows, 768 KiB at the widest, take the thread's scratch (`x86_avx2__zzprivate_rows_scratch`), fetched once a
 *   tile. */
static inline uint64_t x86_avx2__zzprivate_pair_block(uint64_t cols) {
    const uint64_t fit = x86_avx2__zzprivate_l2 / 2u / (cols * 4u);
    return fit < 12u ? 12u : fit - fit % 12u;
}

typedef struct x86_avx2__rows_call {
    uint16_t* out; float* acc; const uint8_t* weights; const uint8_t* luts; const float* xf; const uint32_t* rows;
    const uint16_t* w; uint64_t count, d, out_rows, cols; unsigned int* over;
    nn__expert__groups g; uint64_t total;   /* several matrices: `g`, and all their rows */
    uint64_t xf_at[NN__EXPERT__GROUPS_MAX]; /* where each group's widened pairs begin in `xf` */
} x86_avx2__rows_call;

/* Row `r` of a matrix decoded into `wf`, `cols` floats. */
static inline void x86_avx2__zzprivate_decode_row(uint64_t d, const uint8_t* row, uint64_t cols, float* wf, __m256i tlo, __m256i thi) {
    if (d == 16u) {
        for (uint64_t k = 0; k < cols; k += 8u)
            _mm256_store_ps(wf + k, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + 2u * k))));
    } else if (d == 4u) {
        const __m128i fifteen = _mm_set1_epi8(15);
        for (uint64_t k = 0; k < cols; k += 32u) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
            const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
            const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            const __m256i bl = _mm256_shuffle_epi8(tlo, idx), bh = _mm256_shuffle_epi8(thi, idx);
            const __m256i h0 = _mm256_unpacklo_epi8(bl, bh), h1 = _mm256_unpackhi_epi8(bl, bh);
            _mm256_store_ps(wf + k,       _mm256_cvtph_ps(_mm256_castsi256_si128(h0)));
            _mm256_store_ps(wf + k + 8u,  _mm256_cvtph_ps(_mm256_castsi256_si128(h1)));
            _mm256_store_ps(wf + k + 16u, _mm256_cvtph_ps(_mm256_extracti128_si256(h0, 1)));
            _mm256_store_ps(wf + k + 24u, _mm256_cvtph_ps(_mm256_extracti128_si256(h1, 1)));
        }
    } else {
        for (uint64_t k = 0; k < cols; k += 8u)
            _mm256_store_ps(wf + k, _mm256_i32gather_ps(x86_avx2__zzprivate_levels8,
                                                        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(row + k))), 4));
    }
}
static inline float x86_avx2__zzprivate_dotf(const float* a, const float* b, uint64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (uint64_t k = 0; k < n; k += 32u) {
        a0 = _mm256_fmadd_ps(_mm256_load_ps(a + k),       _mm256_loadu_ps(b + k),       a0);
        a1 = _mm256_fmadd_ps(_mm256_load_ps(a + k + 8u),  _mm256_loadu_ps(b + k + 8u),  a1);
        a2 = _mm256_fmadd_ps(_mm256_load_ps(a + k + 16u), _mm256_loadu_ps(b + k + 16u), a2);
        a3 = _mm256_fmadd_ps(_mm256_load_ps(a + k + 24u), _mm256_loadu_ps(b + k + 24u), a3);
    }
    return x86_avx2__zzpackage_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}
/* A pair's product with a decoded row, weighted into the running sum or rounded into `out`. */
static inline void x86_avx2__zzprivate_put(float v, const uint32_t* rows, const uint16_t* w, uint64_t j, uint64_t out_rows, uint64_t r,
                                           uint16_t* out, float* acc, unsigned int* over) {
    const uint64_t at = (uint64_t)rows[2u * j + 1u] * out_rows + r;
    if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
    else acc[at] += v * _cvtsh_ss(w[rows[2u * j]]);
}
/* One matrix row against its pairs, their `x` widened at `xf`, pair `j`'s at `xf + j·cols`: decoded into the thread's
 * scratch, or a pair at a time from the codes when there is no room — the gemv's dot either way. */
static inline void x86_avx2__zzprivate_row_pairs(uint64_t d, const uint8_t* row, float scale, const float* xf, const uint32_t* rows,
                                                 const uint16_t* w, uint64_t count, uint64_t cols, uint64_t out_rows, uint64_t r,
                                                 uint16_t* out, float* acc, unsigned int* over, __m256i tlo, __m256i thi) {
    if (count == 0u) return;
    float* wf = x86_avx2__zzprivate_rows_scratch(cols);
    if (!wf) {
        for (uint64_t j = 0; j < count; ++j) {
            const float* x = xf + j * cols;
            const float v = d == 4u ? x86_avx2__zzprivate_dot4(row, x, cols, tlo, thi) : d == 8u ? x86_avx2__zzprivate_dot8(row, x, cols)
                          : x86_avx2__zzprivate_dot16((const uint16_t*)(const void*)row, x, cols);
            x86_avx2__zzprivate_put(v * scale, rows, w, j, out_rows, r, out, acc, over);
        }
        return;
    }
    x86_avx2__zzprivate_decode_row(d, row, cols, wf, tlo, thi);
    for (uint64_t j = 0; j < count; ++j)
        x86_avx2__zzprivate_put(x86_avx2__zzprivate_dotf(wf, xf + j * cols, cols) * scale, rows, w, j, out_rows, r, out, acc, over);
}
/* ⭐ THREE ROWS BY FOUR PAIRS, A SUM A PASS — the AVX-512 tile's arrangement in sixteen registers. A dot sums chunk `c` of 8
 *   columns into sum `c % 4`, so a pass over chunks e, e + 4, … gives every (row, pair) of the tile its sum `e` — 12 in
 *   flight, each load of a row serving four products and each load of a pair's `x` three — and the four passes' sums are
 *   added as `x86_avx2__zzprivate_dotf` adds its own: the gemv's bits. ⛳ Called with a literal `n`. */
#define X86_AVX2__ZZPRIVATE_TILE_ROWS2 3u
static inline void x86_avx2__zzprivate_three_rows(const float* const* w, const float* const* xq, uint64_t n, uint64_t cols, float v[3][4]) {
    __m256 keep[4][3][4];
    for (uint64_t e = 0; e < 4u; ++e) {
        __m256 s[3][4];
        for (int a = 0; a < 3; ++a) for (uint64_t q = 0; q < n; ++q) s[a][q] = _mm256_setzero_ps();
        for (uint64_t k = 8u * e; k < cols; k += 32u) {
            const __m256 w0 = _mm256_load_ps(w[0] + k), w1 = _mm256_load_ps(w[1] + k), w2 = _mm256_load_ps(w[2] + k);
            for (uint64_t q = 0; q < n; ++q) {
                const __m256 x = _mm256_loadu_ps(xq[q] + k);
                s[0][q] = _mm256_fmadd_ps(w0, x, s[0][q]); s[1][q] = _mm256_fmadd_ps(w1, x, s[1][q]); s[2][q] = _mm256_fmadd_ps(w2, x, s[2][q]);
            }
        }
        for (int a = 0; a < 3; ++a) for (uint64_t q = 0; q < n; ++q) keep[e][a][q] = s[a][q];
    }
    for (int a = 0; a < 3; ++a)
        for (uint64_t q = 0; q < n; ++q)
            v[a][q] = x86_avx2__zzpackage_hsum(_mm256_add_ps(_mm256_add_ps(keep[0][a][q], keep[1][a][q]), _mm256_add_ps(keep[2][a][q], keep[3][a][q])));
}
/* Rows `r` to `r + 2` of a matrix against their pairs, decoded once into the thread's scratch. */
static inline void x86_avx2__zzprivate_rows3_pairs(uint64_t d, const uint8_t* const* row, const float* scale, const float* xf,
                                                   const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t cols, uint64_t out_rows,
                                                   uint64_t r, uint16_t* out, float* acc, unsigned int* over, __m256i tlo, __m256i thi) {
    float* room = count != 0u ? x86_avx2__zzprivate_rows_scratch(X86_AVX2__ZZPRIVATE_TILE_ROWS2 * cols) : 0;
    if (!room) {
        for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS2; ++a)
            x86_avx2__zzprivate_row_pairs(d, row[a], scale[a], xf, rows, w, count, cols, out_rows, r + a, out, acc, over, tlo, thi);
        return;
    }
    const float* wf[3];
    for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS2; ++a) {
        x86_avx2__zzprivate_decode_row(d, row[a], cols, room + a * cols, tlo, thi);
        wf[a] = room + a * cols;
    }
    for (uint64_t j = 0; j < count; j += 4u) {
        const uint64_t n = count - j < 4u ? count - j : 4u;
        const float* xq[4] = { xf + j * cols, xf + (j + (n > 1u ? 1u : 0u)) * cols, xf + (j + (n > 2u ? 2u : 0u)) * cols,
                               xf + (j + (n > 3u ? 3u : 0u)) * cols };
        float v[3][4];
        if (n == 4u) x86_avx2__zzprivate_three_rows(wf, xq, 4u, cols, v);
        else if (n == 3u) x86_avx2__zzprivate_three_rows(wf, xq, 3u, cols, v);
        else if (n == 2u) x86_avx2__zzprivate_three_rows(wf, xq, 2u, cols, v);
        else x86_avx2__zzprivate_three_rows(wf, xq, 1u, cols, v);
        for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS2; ++a)
            for (uint64_t q = 0; q < n; ++q) x86_avx2__zzprivate_put(v[a][q] * scale[a], rows, w, j + q, out_rows, r + a, out, acc, over);
    }
}
static void x86_avx2__zzprivate_expert_rows(void* a) {
    const x86_avx2__rows_call* c = (const x86_avx2__rows_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->out_rows * b / n, r1 = c->out_rows * (b + 1u) / n, row_bytes = c->cols * c->d / 8u;
    const uint64_t block = x86_avx2__zzprivate_pair_block(c->cols);
    for (uint64_t j0 = 0; j0 < c->count; j0 += block) {
        const uint64_t m = c->count - j0 < block ? c->count - j0 : block;
        uint64_t r = r0;
        for (; r + X86_AVX2__ZZPRIVATE_TILE_ROWS2 <= r1; r += X86_AVX2__ZZPRIVATE_TILE_ROWS2) {
            const uint8_t* row[3]; float scale[3];
            for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS2; ++a) {
                row[a] = c->weights + (r + a) * row_bytes; scale[a] = x86_avx2__zzprivate_gemm_scale(c->luts, c->d, r + a);
            }
            x86_avx2__zzprivate_rows3_pairs(c->d, row, scale, c->xf + j0 * c->cols, c->rows + 2u * j0, c->w, m, c->cols, c->out_rows, r,
                                            c->out, c->acc, c->over, tlo, thi);
        }
        for (; r < r1; ++r)
            x86_avx2__zzprivate_row_pairs(c->d, c->weights + r * row_bytes, x86_avx2__zzprivate_gemm_scale(c->luts, c->d, r), c->xf + j0 * c->cols,
                                          c->rows + 2u * j0, c->w, m, c->cols, c->out_rows, r, c->out, c->acc, c->over, tlo, thi);
    }
}
static void x86_avx2__zzprivate_expert_groups(void* a) {
    const x86_avx2__rows_call* c = (const x86_avx2__rows_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->total * b / n, t1 = c->total * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ) {
        while (t >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t r = t - base, row_bytes = c->cols * c->g.d[i] / 8u;
        const uint8_t* codes = (const uint8_t*)(uintptr_t)c->g.codes[i];
        const uint8_t* luts = (const uint8_t*)(uintptr_t)c->g.luts[i];
        if (t + X86_AVX2__ZZPRIVATE_TILE_ROWS2 <= t1 && r + X86_AVX2__ZZPRIVATE_TILE_ROWS2 <= c->g.out_rows[i]) {   /* three rows of one matrix */
            const uint8_t* row[3]; float scale[3];
            for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS2; ++a) {
                row[a] = codes + (r + a) * row_bytes; scale[a] = x86_avx2__zzprivate_gemm_scale(luts, c->g.d[i], r + a);
            }
            x86_avx2__zzprivate_rows3_pairs(c->g.d[i], row, scale, c->xf + c->xf_at[i], c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i],
                                            c->cols, c->g.out_rows[i], r, c->out + c->g.out_at[i], 0, c->over, tlo, thi);
            t += X86_AVX2__ZZPRIVATE_TILE_ROWS2;
        } else {
            x86_avx2__zzprivate_row_pairs(c->g.d[i], codes + r * row_bytes, x86_avx2__zzprivate_gemm_scale(luts, c->g.d[i], r), c->xf + c->xf_at[i],
                                          c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->cols, c->g.out_rows[i], r,
                                          c->out + c->g.out_at[i], 0, c->over, tlo, thi);
            t += 1u;
        }
    }
}
/* Pairs' `x` rows widened to floats, pair `j` at `xf + j·cols` — for a 4-bit matrix under the AVX-512 bodies split as
 * their decoded rows are, each run of 32 its even columns then its odd. */
static inline void x86_avx2__zzprivate_widen_pairs(float* xf, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d,
                                                   uint64_t cols) {
    for (uint64_t j = 0; j < count; ++j) {
        const uint16_t* xr = x + (uint64_t)rows[2u * j] * cols;
        if (x86_avx2__zzprivate_wide && d == 4u) { x86_avx2__zzprivate_split_512(xf + j * cols, xr, cols); continue; }
        for (uint64_t k = 0; k < cols; k += 8u)
            _mm256_storeu_ps(xf + j * cols + k, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(xr + k))));
    }
}
static inline bool x86_avx2__zzprivate_gemm_fit(uint64_t d, uint64_t cols) {
    return (d == 4u || d == 8u || d == 16u) && cols != 0u && cols % 32u == 0u && cols <= X86_AVX2__ZZPRIVATE_GEMM_COLS;
}
static void x86_avx2__override_expert_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                           const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d, uint64_t out_rows,
                                           uint64_t cols, unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (!x86_avx2__zzprivate_gemm_fit(d, cols) || count == 0u || out_rows > w_room / row_bytes || (d != 16u && out_rows > l_room / 2u)
     || !x86_avx2__zzprivate_x_room_for(count * cols)) {
        nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_widen_pairs(x86_avx2__zzprivate_xf, x, rows, count, d, cols);
    x86_avx2__rows_call c = {};
    c.out = out; c.weights = weights; c.luts = luts; c.xf = x86_avx2__zzprivate_xf; c.rows = rows;
    c.count = count; c.d = d; c.out_rows = out_rows; c.cols = cols; c.over = over;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(out_rows),
                            x86_avx2__zzprivate_wide ? x86_avx2__zzprivate_expert_rows_512 : x86_avx2__zzprivate_expert_rows, &c);
}
static void x86_avx2__override_expert_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                               const uint16_t* x, const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t d,
                                               uint64_t out_rows, uint64_t cols) {
    const uint64_t row_bytes = cols * d / 8u;
    if (!x86_avx2__zzprivate_gemm_fit(d, cols) || count == 0u || out_rows > w_room / row_bytes || (d != 16u && out_rows > l_room / 2u)
     || !x86_avx2__zzprivate_x_room_for(count * cols)) {
        nn__expert__zzabi_launch_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_widen_pairs(x86_avx2__zzprivate_xf, x, rows, count, d, cols);
    x86_avx2__rows_call c = {};
    c.acc = acc; c.weights = weights; c.luts = luts; c.xf = x86_avx2__zzprivate_xf; c.rows = rows; c.w = w;
    c.count = count; c.d = d; c.out_rows = out_rows; c.cols = cols;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(out_rows),
                            x86_avx2__zzprivate_wide ? x86_avx2__zzprivate_expert_rows_512 : x86_avx2__zzprivate_expert_rows, &c);
}
static void x86_avx2__override_expert_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows, uint64_t cols,
                                             unsigned int* over) {
    bool fit = g.count != 0u && g.count <= NN__EXPERT__GROUPS_MAX;
    uint64_t total = 0u, pairs = 0u;
    for (uint64_t i = 0u; fit && i < g.count; ++i) {
        fit = x86_avx2__zzprivate_gemm_fit(g.d[i], cols) && g.out_rows[i] != 0u;
        total += g.out_rows[i]; pairs += g.pairs[i];
    }
    if (!fit || !x86_avx2__zzprivate_x_room_for(pairs * cols)) { nn__expert__zzabi_launch_groups(out, g, x, rows, cols, over); return; }
    x86_avx2__zzpackage_start();
    x86_avx2__rows_call c = {};
    uint64_t placed = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) {
        c.xf_at[i] = placed * cols;
        x86_avx2__zzprivate_widen_pairs(x86_avx2__zzprivate_xf + placed * cols, x, rows + 2u * g.pairs_at[i], g.pairs[i], g.d[i], cols);
        placed += g.pairs[i];
    }
    c.out = out; c.xf = x86_avx2__zzprivate_xf; c.rows = rows; c.cols = cols; c.over = over; c.g = g; c.total = total;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total),
                            x86_avx2__zzprivate_wide ? x86_avx2__zzprivate_expert_groups_512 : x86_avx2__zzprivate_expert_groups, &c);
}


/* ══ ⭐⭐ THE EXPERT-MAJOR GEMM IN INTEGERS — nn's `expert_groups_int8` / `expert_rows_sum_int8` definition: each row of
 *   `x` quantised once a call, a block of `NN__EXPERT__INT8_BLOCK` at a time; a matrix row's 4-bit codes to their int8
 *   levels once (one `pshufb`), then four pairs at a time, `abs(w)·sign(x, w)` in `maddubs`, integer sums a block, one
 *   float step a block. `MEASURED` (the probe beside NN-35's measurements, 28 threads): 300-545 GMAC/s against the
 *   fp32 GEMM's 100-132. */
typedef struct x86_avx2__int8_call {
    const uint16_t* x; int8_t* xq; float* xs; uint64_t cols, x0, nx;                  /* rows x0 .. x0 + nx, quantised */
    uint16_t* out; float* acc; const uint8_t* weights; const uint8_t* luts; const uint32_t* rows; const uint16_t* w;
    uint64_t count, out_rows;                                                          /* one matrix */
    nn__expert__groups g; uint64_t total;                                              /* several */
    unsigned int* over;
} x86_avx2__int8_call;

/* Block `x[0..256)` quantised into `q` as nn's definition says; its scale answered. */
static inline float x86_avx2__zzprivate_quantise256(const uint16_t* x, int8_t* q) {
    return x86_avx2__zzprivate_quantise_run(x, q, NN__EXPERT__INT8_BLOCK);
}
static void x86_avx2__zzprivate_quantise_rows(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks(), per = c->cols / NN__EXPERT__INT8_BLOCK;
    for (uint64_t r = c->nx * b / n; r < c->nx * (b + 1u) / n; ++r)
        for (uint64_t k = 0; k < per; ++k)
            c->xs[r * per + k] = x86_avx2__zzprivate_quantise256(c->x + (c->x0 + r) * c->cols + k * NN__EXPERT__INT8_BLOCK,
                                                                   c->xq + r * c->cols + k * NN__EXPERT__INT8_BLOCK);
}
/* One or two pairs — an expert one or two of a chunk's rows picked: the row's levels made in a register and used at once,
 * not kept in `w8`, with the four-pair loop's lanes, its integer sums a block, its float step a block and its sum, so its
 * bytes; the row two ahead — `cols` bytes on — fetched into L2 as it is read. `MEASURED`
 * (measurements/2026-10-30_nn145_int8_bodies_dl380.md): GLM's gate+up at a row an expert through `expert_groups_int8` on
 * one core of an E5-2680 v4, 4.5 -> 5.9 GB/s. ▶ `row_few_i8` in `int8_512.cuh`. */
static inline void x86_avx2__zzprivate_row_few_i8(const uint8_t* row, float scale, const int8_t* xq, const float* xs, uint64_t x0,
                                                  const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t cols,
                                                  uint64_t out_rows, uint64_t r, uint16_t* out, float* acc, unsigned int* over,
                                                  __m256i t8) {
    const __m128i fifteen = _mm_set1_epi8(15);
    const __m256i ones = _mm256_set1_epi16(1);
    const uint64_t per = cols / NN__EXPERT__INT8_BLOCK;
    const int8_t* xr[2] = { xq, xq };
    const float* sr[2] = { xs, xs };
    for (uint64_t q = 0; q < count; ++q) {
        const uint64_t xrow = (uint64_t)rows[2u * q] - x0;
        xr[q] = xq + xrow * cols; sr[q] = xs + xrow * per;
    }
    __m256 f[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    for (uint64_t B = 0; B < per; ++B) {
        __m256i s[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
        for (uint64_t k = B * NN__EXPERT__INT8_BLOCK; k < (B + 1u) * NN__EXPERT__INT8_BLOCK; k += 32u) {
            _mm_prefetch((const char*)(row + k / 2u + cols), _MM_HINT_T1);
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
            const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
            const __m256i wv = _mm256_shuffle_epi8(t8, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)));
            const __m256i aw = _mm256_abs_epi8(wv);
            for (uint64_t q = 0; q < count; ++q)
                s[q] = _mm256_add_epi32(s[q], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(
                                           _mm256_loadu_si256((const __m256i*)(xr[q] + k)), wv)), ones));
        }
        for (uint64_t q = 0; q < count; ++q) f[q] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[q]), _mm256_set1_ps(sr[q][B]), f[q]);
    }
    for (uint64_t q = 0; q < count; ++q) {
        const float v = x86_avx2__zzpackage_hsum(f[q]) * scale / 127.0f;
        const uint64_t at = (uint64_t)rows[2u * q + 1u] * out_rows + r;
        if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
        else acc[at] += v * _cvtsh_ss(w[rows[2u * q]]);
    }
}
/* One matrix row against its pairs: the codes to int8 levels once, then the pairs four at a time — one or two of them
 * `row_few_i8`'s. */
static inline void x86_avx2__zzprivate_row_pairs_i8(const uint8_t* row, float scale, const int8_t* xq, const float* xs, uint64_t x0,
                                                    const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t cols,
                                                    uint64_t out_rows, uint64_t r, uint16_t* out, float* acc, unsigned int* over,
                                                    __m256i t8) {
    if (count <= 2u) {
        x86_avx2__zzprivate_row_few_i8(row, scale, xq, xs, x0, rows, w, count, cols, out_rows, r, out, acc, over, t8);
        return;
    }
    int8_t w8[cols] __attribute__((aligned(32)));
    const __m128i fifteen = _mm_set1_epi8(15);
    for (uint64_t k = 0; k < cols; k += 32u) {
        _mm_prefetch((const char*)(row + k / 2u + cols), _MM_HINT_T1);
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
        const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
        _mm256_store_si256((__m256i*)(w8 + k), _mm256_shuffle_epi8(t8, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi))));
    }
    const __m256i ones = _mm256_set1_epi16(1);
    const uint64_t per = cols / NN__EXPERT__INT8_BLOCK;
    for (uint64_t j = 0; j < count; j += 4u) {
        const uint64_t n = count - j < 4u ? count - j : 4u;
        const int8_t* xr[4];
        const float* sr[4];
        for (uint64_t q = 0; q < n; ++q) {
            const uint64_t xrow = (uint64_t)rows[2u * (j + q)] - x0;
            xr[q] = xq + xrow * cols; sr[q] = xs + xrow * per;
        }
        __m256 f[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (uint64_t B = 0; B < per; ++B) {
            __m256i s4[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
            for (uint64_t k = B * NN__EXPERT__INT8_BLOCK; k < (B + 1u) * NN__EXPERT__INT8_BLOCK; k += 32u) {
                const __m256i wv = _mm256_load_si256((const __m256i*)(w8 + k)), aw = _mm256_abs_epi8(wv);
                for (uint64_t q = 0; q < n; ++q)
                    s4[q] = _mm256_add_epi32(s4[q], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(
                                                 _mm256_loadu_si256((const __m256i*)(xr[q] + k)), wv)), ones));
            }
            for (uint64_t q = 0; q < n; ++q) f[q] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s4[q]), _mm256_set1_ps(sr[q][B]), f[q]);
        }
        for (uint64_t q = 0; q < n; ++q) {
            const float v = x86_avx2__zzpackage_hsum(f[q]) * scale / 127.0f;
            const uint64_t at = (uint64_t)rows[2u * (j + q) + 1u] * out_rows + r;
            if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
            else acc[at] += v * _cvtsh_ss(w[rows[2u * (j + q)]]);
        }
    }
}
static void x86_avx2__zzprivate_expert_rows_i8(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks(), row_bytes = c->cols / 2u;
    for (uint64_t r = c->out_rows * b / n; r < c->out_rows * (b + 1u) / n; ++r)
        x86_avx2__zzprivate_row_pairs_i8(c->weights + r * row_bytes, x86_avx2__zzpackage_row_scale(c->luts, r), c->xq, c->xs, c->x0, c->rows,
                                         c->w, c->count, c->cols, c->out_rows, r, c->out, c->acc, c->over, t8);
}
static void x86_avx2__zzprivate_expert_groups_i8(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->total * b / n, t1 = c->total * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t r = t - base;
        x86_avx2__zzprivate_row_pairs_i8((const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u),
                                         x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r), c->xq, c->xs, c->x0,
                                         c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->cols, c->g.out_rows[i], r,
                                         c->out + c->g.out_at[i], 0, c->over, t8);
    }
}
/* The rows of `x` the pairs name — the run from the lowest to the highest — quantised once, on the pool. False when
 * there is no room for them. */
static inline bool x86_avx2__zzprivate_quantise_pairs(x86_avx2__int8_call* c, const uint16_t* x, const uint32_t* rows,
                                                      uint64_t pairs, uint64_t cols) {
    uint64_t lo = ~0ull, hi = 0u;
    for (uint64_t j = 0; j < pairs; ++j) {
        if ((uint64_t)rows[2u * j] < lo) lo = rows[2u * j];
        if ((uint64_t)rows[2u * j] + 1u > hi) hi = (uint64_t)rows[2u * j] + 1u;
    }
    if (pairs == 0u || !x86_avx2__zzprivate_x_room_for((hi - lo) * cols)) return false;
    c->x = x; c->xq = x86_avx2__zzprivate_xq; c->xs = x86_avx2__zzprivate_xs; c->cols = cols; c->x0 = lo; c->nx = hi - lo;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(c->nx), x86_avx2__zzprivate_quantise_rows, c);
    return true;
}
static inline bool x86_avx2__zzprivate_int8_fit(uint64_t d, uint64_t cols) {
    return d == 4u && cols != 0u && cols % NN__EXPERT__INT8_BLOCK == 0u && cols <= X86_AVX2__ZZPRIVATE_GEMM_COLS;
}
static void x86_avx2__override_expert_rows_sum_int8(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                    uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols) {
    x86_avx2__int8_call c = {};
    if (!x86_avx2__zzprivate_int8_fit(d, cols) || count == 0u || out_rows > w_room / (cols / 2u) || out_rows > l_room / 2u) {
        nn__expert__zzabi_launch_rows_sum_int8(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    x86_avx2__zzpackage_start();
    if (!x86_avx2__zzprivate_quantise_pairs(&c, x, rows, count, cols)) {
        nn__expert__zzabi_launch_rows_sum_int8(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    c.acc = acc; c.weights = weights; c.luts = luts; c.rows = rows; c.w = w; c.count = count; c.out_rows = out_rows;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(out_rows),
                            x86_avx2__zzprivate_vnni ? X86_AVX2__ZZPRIVATE_I8_VNNI(expert_rows_i8) : x86_avx2__zzprivate_wide ? X86_AVX2__ZZPRIVATE_I8_512(expert_rows_i8)
                            : x86_avx2__zzprivate_expert_rows_i8, &c);
}
static void x86_avx2__override_expert_groups_int8(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                  uint64_t cols, unsigned int* over) {
    bool fit = g.count != 0u && g.count <= NN__EXPERT__GROUPS_MAX;
    uint64_t total = 0u, last = 0u;
    for (uint64_t i = 0u; fit && i < g.count; ++i) {
        fit = x86_avx2__zzprivate_int8_fit(g.d[i], cols) && g.out_rows[i] != 0u;
        total += g.out_rows[i];
        if (g.pairs_at[i] + g.pairs[i] > last) last = g.pairs_at[i] + g.pairs[i];
    }
    x86_avx2__int8_call c = {};
    if (fit) x86_avx2__zzpackage_start();
    if (!fit || !x86_avx2__zzprivate_quantise_pairs(&c, x, rows, last, cols)) {
        nn__expert__zzabi_launch_groups_int8(out, g, x, rows, cols, over); return;
    }
    c.out = out; c.rows = rows; c.g = g; c.total = total; c.over = over;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total),
                            x86_avx2__zzprivate_vnni ? X86_AVX2__ZZPRIVATE_I8_VNNI(expert_groups_i8) : x86_avx2__zzprivate_wide ? X86_AVX2__ZZPRIVATE_I8_512(expert_groups_i8)
                            : x86_avx2__zzprivate_expert_groups_i8, &c);
}

/* ── THE ROTATION A HEAD AT A TIME — `hadamard_blocks`, a butterfly a block ─────────────────────────────────────────
 * nn's definition (`kernels.cuh`): blocks of `block` (a power of two, at most 512), each `out = H·(S ⊙ in) / √block`,
 * or with `inverse` `S ⊙ H·in / √block`; the generic body sums each output directly over its block, O(b²), with a lane an
 * output. Here a pool block takes whole Hadamard blocks and runs the butterfly in floats — O(b log b), the same matrix
 * (Sylvester's order, `(-1)^popcount(r & c)`) — and rounds each output to a half once, as the generic body does; only the
 * order of the float sums differs. `MEASURED` before: ~9 ms of the 35B's ~87 ms a token on the CPU alone. */
typedef struct x86_avx2__hblocks_call {
    uint16_t* out; const uint16_t* in; const uint16_t* sign; uint64_t n, block, inverse; unsigned int* over;
} x86_avx2__hblocks_call;
static void x86_avx2__zzprivate_hblocks(void* a) {
    const x86_avx2__hblocks_call* c = (const x86_avx2__hblocks_call*)a;
    const uint64_t count = c->n / c->block, b = nn__silicon__block(), nb = nn__silicon__blocks();
    const uint64_t q0 = count * b / nb, q1 = count * (b + 1u) / nb, B = c->block;
    const float norm = 1.0f / sqrtf((float)B);
    float v[512];
    for (uint64_t q = q0; q < q1; ++q) {
        const uint64_t at = q * B;
        for (uint64_t e = 0u; e < B; ++e) v[e] = _cvtsh_ss(c->in[at + e]) * (c->inverse ? 1.0f : _cvtsh_ss(c->sign[e]));
        for (uint64_t h = 1u; h < B; h <<= 1)
            for (uint64_t i = 0u; i < B; i += 2u * h)
                for (uint64_t j = i; j < i + h; ++j) { const float x = v[j], y = v[j + h]; v[j] = x + y; v[j + h] = x - y; }
        for (uint64_t e = 0u; e < B; ++e)
            c->out[at + e] = nn__kernels__zzabi_to_half(v[e] * norm * (c->inverse ? _cvtsh_ss(c->sign[e]) : 1.0f), c->over);
    }
}
/* ⭐ Blocks of a power of two up to 512, `out` apart from `in` or exactly `in` — a pool block reads each Hadamard block
 *   whole before it writes it, so a rotation in place is the same arithmetic; anything else is the generic door. */
static void x86_avx2__override_hadamard_blocks(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, uint64_t block,
                                               uint64_t inverse, unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out || out == in;
    if (block == 0u || block > 512u || (block & (block - 1u)) != 0u || n == 0u || n % block != 0u || !apart) {
        nn__hadamard__zzabi_launch_blocks(out, in, sign, n, block, inverse, over);
        return;
    }
    x86_avx2__hblocks_call c = { out, in, sign, n, block, inverse, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(n / block), x86_avx2__zzprivate_hblocks, &c);
}

/* ══ ⭐⭐ THE SAME DOORS ON AVX-512 — installed over the AVX2 bodies when the processor has AVX-512 F, BW, VL and DQ ═════════
 * ⚖ *"go on with the cuda work first, then commit, then avx512"*. Each body is compiled for AVX-512 on its own
 * (`X86_AVX2__ZZPRIVATE_AVX512`), so the library still loads, and runs its AVX2 bodies, on a processor without it;
 * `SILVANN_CPU_AVX512=0` keeps the AVX2 ones on one that has it. The same fit tests, the same fallbacks, the same pool.
 *   4 bits   a run of 32 codes is 16 bytes widened to 16 dwords. `vpermps` reads the low 4 bits of an index, so the low
 *            nibbles index the 16 levels as they lie and the high ones after a shift: 16 levels an instruction, in floats,
 *            no half to widen and no byte to interleave. The run's even columns come out first and its odd ones second,
 *            so `x` is split the same way once a call (`xd`), and a row the GEMM decodes is laid out so too.
 *   8 bits   the levels are symmetric, `level[255 − c] = −level[c]` as halves (checked when the doors are installed; they are
 *            gathered otherwise): the first 128 levels are four registers of halves, a code past 127 is inverted onto
 *            255 − c, two `vpermt2w` and a blend on bit 6 give 32 halves, the code's bit 7 goes onto their sign, and two
 *            `vcvtph2ps` widen them. No gather: on Skylake under the microcode's `gather_data_sampling` mitigation the AVX2
 *            gathers read 2.9-4.1 GB/s.
 *   int8     64 codes a step: the codes widened to 16 bits put each nibble in a byte of its own with one shift and one
 *            `vpternlog`, in the columns' order, and one `vpshufb` gives their levels. There is no `vpsignb` at 512 bits,
 *            so `x` is negated under the levels' sign mask; the integer sums are the AVX2 bodies'.
 * ⛳ EVERY 4- AND 8-BIT DOT SUMS ITS ROW IN CHUNKS OF 16 COLUMNS, CHUNK `c` INTO SUM `c % 4`, the four added `(0 + 1) +
 *   (2 + 3)` and then across their lanes — in the gemv and in the GEMM alike, so a row the GEMM decoded once and the same
 *   row through the gemv give the same bits.
 * `MEASURED` (precision, Xeon W-2123, 4 cores, two DDR4-2400 sticks; `scripts/tq_cpu_avx512_probe.c`, 512 MiB of codes,
 *   4 threads, an idle machine): memory reads 31.6-36.1 GB/s; 4-bit rows 14.7 GB/s on AVX2, 33.2 on AVX-512; 8-bit rows 4.1
 *   (gathers), 23.6 (the levels as floats, eight registers), 29.0 (as halves). */
static float x86_avx2__zzprivate_levels4[16];

/* The 4-bit levels as floats, and the first 128 8-bit levels as halves, 32 a register. */
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_tables_512(__m512* t4, __m512i* t8) {
    *t4 = _mm512_loadu_ps(x86_avx2__zzprivate_levels4);
    const uint16_t* lev = nn__turboquant__zzabi_levels(8u);
    for (int i = 0; i < 4; ++i) t8[i] = _mm512_loadu_si512((const void*)(lev + 32 * i));
}
/* `n` halves (whole runs of 32) widened and split: each run's 16 even columns, then its 16 odd ones. */
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_split_512(float* to, const uint16_t* x, uint64_t n) {
    const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    for (uint64_t k = 0; k < n; k += 32u) {
        const __m512 a = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(x + k)));
        const __m512 b = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(x + k + 16u)));
        _mm512_storeu_ps(to + k, _mm512_permutex2var_ps(a, even, b));
        _mm512_storeu_ps(to + k + 16u, _mm512_permutex2var_ps(a, odd, b));
    }
}
/* 32 codes' 8-bit levels, the codes one a 16-bit word, as two runs of 16 floats. A code past 127 is inverted — its low
 * 7 bits are then 255 − c's — `vpermt2w` reads the low 6 bits and bit 6 picks the pair of registers, and the code's bit 7
 * goes onto the half's sign. */
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_lev8_512(__m512i c, const __m512i* t8, __m512* lo, __m512* hi) {
    const __m512i up = _mm512_slli_epi16(c, 8);
    const __m512i f = _mm512_xor_si512(c, _mm512_srai_epi16(up, 15));
    const __mmask32 b6 = _mm512_test_epi16_mask(f, _mm512_set1_epi16(64));
    __m512i h = _mm512_mask_blend_epi16(b6, _mm512_permutex2var_epi16(t8[0], f, t8[1]), _mm512_permutex2var_epi16(t8[2], f, t8[3]));
    h = _mm512_ternarylogic_epi32(h, up, _mm512_set1_epi16((short)0x8000), 0x78);          /* h ^ (up & 0x8000) */
    *lo = _mm512_cvtph_ps(_mm512_castsi512_si256(h));
    *hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(h, 1));
}
static X86_AVX2__ZZPRIVATE_AVX512 inline float x86_avx2__zzprivate_sum4_512(__m512 a0, __m512 a1, __m512 a2, __m512 a3) {
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
}
/* One 4-bit row's product with `xd`, `x` split. */
static X86_AVX2__ZZPRIVATE_AVX512 inline float x86_avx2__zzprivate_dot4_512(const uint8_t* row, const float* xd, uint64_t cols, __m512 t4) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    uint64_t k = 0;
    for (; k + 64u <= cols; k += 64u) {
        const __m512i b0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(row + k / 2u)));
        const __m512i b1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(row + k / 2u + 16u)));
        a0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(b0, t4), _mm512_loadu_ps(xd + k), a0);
        a1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(b0, 4), t4), _mm512_loadu_ps(xd + k + 16u), a1);
        a2 = _mm512_fmadd_ps(_mm512_permutexvar_ps(b1, t4), _mm512_loadu_ps(xd + k + 32u), a2);
        a3 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(b1, 4), t4), _mm512_loadu_ps(xd + k + 48u), a3);
    }
    if (k < cols) {                                     /* a last run of 32: chunks 4m and 4m + 1 */
        const __m512i b0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(row + k / 2u)));
        a0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(b0, t4), _mm512_loadu_ps(xd + k), a0);
        a1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(b0, 4), t4), _mm512_loadu_ps(xd + k + 16u), a1);
    }
    return x86_avx2__zzprivate_sum4_512(a0, a1, a2, a3);
}
/* 32 codes' 8-bit levels as two runs of 16 floats: through the halves' registers, or gathered from the floats where the
 * levels are not symmetric — the same values either way, so a dot's sums do not depend on which. */
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_codes8_512(const uint8_t* codes, const __m512i* t8, __m512* lo, __m512* hi) {
    if (x86_avx2__zzprivate_wide8) {
        x86_avx2__zzprivate_lev8_512(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)codes)), t8, lo, hi);
        return;
    }
    *lo = _mm512_i32gather_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)codes)), x86_avx2__zzprivate_levels8, 4);
    *hi = _mm512_i32gather_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(codes + 16u))), x86_avx2__zzprivate_levels8, 4);
}
/* One 8-bit row's product with `xf`. */
static X86_AVX2__ZZPRIVATE_AVX512 inline float x86_avx2__zzprivate_dot8_512(const uint8_t* row, const float* xf, uint64_t cols, const __m512i* t8) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps(), l, h;
    uint64_t k = 0;
    for (; k + 64u <= cols; k += 64u) {
        x86_avx2__zzprivate_codes8_512(row + k, t8, &l, &h);
        a0 = _mm512_fmadd_ps(l, _mm512_loadu_ps(xf + k), a0);
        a1 = _mm512_fmadd_ps(h, _mm512_loadu_ps(xf + k + 16u), a1);
        x86_avx2__zzprivate_codes8_512(row + k + 32u, t8, &l, &h);
        a2 = _mm512_fmadd_ps(l, _mm512_loadu_ps(xf + k + 32u), a2);
        a3 = _mm512_fmadd_ps(h, _mm512_loadu_ps(xf + k + 48u), a3);
    }
    if (k < cols) {
        x86_avx2__zzprivate_codes8_512(row + k, t8, &l, &h);
        a0 = _mm512_fmadd_ps(l, _mm512_loadu_ps(xf + k), a0);
        a1 = _mm512_fmadd_ps(h, _mm512_loadu_ps(xf + k + 16u), a1);
    }
    return x86_avx2__zzprivate_sum4_512(a0, a1, a2, a3);
}

/* ── the exact gemvs ── */
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_exact4_512(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    __m512 t4; __m512i t8[4]; x86_avx2__zzprivate_tables_512(&t4, t8);
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r)
        x86_avx2__zzpackage_write(c, r, x86_avx2__zzprivate_dot4_512(c->weights + r * c->row_bytes, c->xd, c->cols, t4)
                                        * x86_avx2__zzpackage_row_scale(c->luts, r));
}
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_exact8_512(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    __m512 t4; __m512i t8[4]; x86_avx2__zzprivate_tables_512(&t4, t8);
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r)
        x86_avx2__zzpackage_write(c, r, x86_avx2__zzprivate_dot8_512(c->weights + r * c->row_bytes, c->xf, c->cols, t8)
                                        * x86_avx2__zzpackage_row_scale(c->luts, r));
}
/* One 16-bit row's product with `xf`. */
static X86_AVX2__ZZPRIVATE_AVX512 inline float x86_avx2__zzprivate_dot16_512(const uint16_t* row, const float* xf, uint64_t cols) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    uint64_t k = 0;
    for (; k + 64u <= cols; k += 64u) {
        a0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + k))),       _mm512_loadu_ps(xf + k),       a0);
        a1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + k + 16u))), _mm512_loadu_ps(xf + k + 16u), a1);
        a2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + k + 32u))), _mm512_loadu_ps(xf + k + 32u), a2);
        a3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + k + 48u))), _mm512_loadu_ps(xf + k + 48u), a3);
    }
    if (k < cols) {
        a0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + k))),       _mm512_loadu_ps(xf + k),       a0);
        a1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + k + 16u))), _mm512_loadu_ps(xf + k + 16u), a1);
    }
    return x86_avx2__zzprivate_sum4_512(a0, a1, a2, a3);
}
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_exact16_512(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    uint64_t r0, r1; x86_avx2__zzpackage_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r)
        x86_avx2__zzpackage_write(c, r, x86_avx2__zzprivate_dot16_512((const uint16_t*)(const void*)(c->weights + r * c->row_bytes), c->xf, c->cols));
}
static X86_AVX2__ZZPRIVATE_AVX512 inline float x86_avx2__zzprivate_group_row_512(const nn__turboquant__groups* g, uint64_t i, uint64_t r,
                                                                                 const float* xf, const float* xd, uint64_t cols,
                                                                                 __m512 t4, const __m512i* t8) {
    const uint8_t* row = (const uint8_t*)(uintptr_t)g->codes[i] + r * (cols * g->d[i] / 8u);
    const float dot = g->d[i] == 4u ? x86_avx2__zzprivate_dot4_512(row, xd, cols, t4) : x86_avx2__zzprivate_dot8_512(row, xf, cols, t8);
    return dot * x86_avx2__zzpackage_row_scale((const uint8_t*)(uintptr_t)g->luts[i], r);
}
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_groups_512(void* a) {
    const x86_avx2__groups_call* c = (const x86_avx2__groups_call*)a;
    __m512 t4; __m512i t8[4]; x86_avx2__zzprivate_tables_512(&t4, t8);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->rows * b / n, t1 = c->rows * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = t - base;
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(
            x86_avx2__zzprivate_group_row_512(&c->g, i, r, c->xf, c->xd, c->cols, t4, t8), c->over);
    }
}
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_groups_sum_512(void* a) {
    const x86_avx2__groups_call* c = (const x86_avx2__groups_call*)a;
    __m512 t4; __m512i t8[4]; x86_avx2__zzprivate_tables_512(&t4, t8);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->rows * b / n, r1 = c->rows * (b + 1u) / n;
    for (uint64_t r = r0; r < r1; ++r) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i)
            sum += x86_avx2__zzprivate_group_row_512(&c->g, i, r, c->xf + i * c->cols, c->xd + i * c->cols, c->cols, t4, t8)
                 * _cvtsh_ss(c->w[c->g.out_at[i]]);
        c->out[r] = nn__kernels__zzabi_to_half(_cvtsh_ss(c->residual[r]) + sum, c->over);
    }
}

/* ── the expert-major GEMM: a row decoded once — at 4 bits split as `xd` is — then its pairs four at a time ── */
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_decode_row_512(uint64_t d, const uint8_t* row, uint64_t cols, float* wf,
                                                                                 __m512 t4, const __m512i* t8) {
    if (d == 4u) {
        for (uint64_t k = 0; k < cols; k += 32u) {
            const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(row + k / 2u)));
            _mm512_store_ps(wf + k, _mm512_permutexvar_ps(b, t4));
            _mm512_store_ps(wf + k + 16u, _mm512_permutexvar_ps(_mm512_srli_epi32(b, 4), t4));
        }
    } else if (d == 16u) {
        for (uint64_t k = 0; k < cols; k += 16u)
            _mm512_store_ps(wf + k, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + 2u * k))));
    } else {
        for (uint64_t k = 0; k < cols; k += 32u) {
            __m512 l, h;
            x86_avx2__zzprivate_codes8_512(row + k, t8, &l, &h);
            _mm512_store_ps(wf + k, l); _mm512_store_ps(wf + k + 16u, h);
        }
    }
}
/* A decoded row's product with one pair's `x`: the dots' four sums. */
static X86_AVX2__ZZPRIVATE_AVX512 inline float x86_avx2__zzprivate_dotf_512(const float* wf, const float* x, uint64_t cols) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    uint64_t k = 0;
    for (; k + 64u <= cols; k += 64u) {
        a0 = _mm512_fmadd_ps(_mm512_load_ps(wf + k),       _mm512_loadu_ps(x + k),       a0);
        a1 = _mm512_fmadd_ps(_mm512_load_ps(wf + k + 16u), _mm512_loadu_ps(x + k + 16u), a1);
        a2 = _mm512_fmadd_ps(_mm512_load_ps(wf + k + 32u), _mm512_loadu_ps(x + k + 32u), a2);
        a3 = _mm512_fmadd_ps(_mm512_load_ps(wf + k + 48u), _mm512_loadu_ps(x + k + 48u), a3);
    }
    if (k < cols) {
        a0 = _mm512_fmadd_ps(_mm512_load_ps(wf + k),       _mm512_loadu_ps(x + k),       a0);
        a1 = _mm512_fmadd_ps(_mm512_load_ps(wf + k + 16u), _mm512_loadu_ps(x + k + 16u), a1);
    }
    return x86_avx2__zzprivate_sum4_512(a0, a1, a2, a3);
}
/* Four pairs against one chunk `c` of the decoded row: each pair's sum `c % 4`. */
#define X86_AVX2__ZZPRIVATE_FOUR(S, C, AT) do {                                                                   \
        const __m512 w_ = _mm512_load_ps(wf + (AT));                                                           \
        S[0][C] = _mm512_fmadd_ps(w_, _mm512_loadu_ps(x0 + (AT)), S[0][C]);                                    \
        S[1][C] = _mm512_fmadd_ps(w_, _mm512_loadu_ps(x1 + (AT)), S[1][C]);                                    \
        S[2][C] = _mm512_fmadd_ps(w_, _mm512_loadu_ps(x2 + (AT)), S[2][C]);                                    \
        S[3][C] = _mm512_fmadd_ps(w_, _mm512_loadu_ps(x3 + (AT)), S[3][C]);                                    \
    } while (0)
/* ⛳ A FEW PAIRS: STRAIGHT FROM THE CODES. Routing gives an expert a handful of a chunk's rows — the 35B's, 1 to 21 a call
 *   and most of them under five — and at so few a row's decode into floats and its reads back cost more than the products
 *   (`MEASURED`, the expert GEMM at 1 to 16 pairs, ▶ the record). So up to four pairs take a row from its codes: at 4 bits
 *   decoded into registers once and multiplied into each pair's four sums, at 8 and 16 the gemv's dot a pair. The sums are
 *   the tiles' — chunk `c` into sum `c % 4` — so the answer is the same bits whichever way a row goes. */
#define X86_AVX2__ZZPRIVATE_FUSED_PAIRS 4u
/* A 4-bit row against `n` pairs (1 to 4), their split `x` at `x[q]`: `v[q]`. ⛳ Called with a literal `n`, so the sums stay
 * in registers. */
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_dot4_pairs_512(const uint8_t* row, const float* const* x, uint64_t n,
                                                                                 uint64_t cols, __m512 t4, float* v) {
    __m512 s[4][4];
    for (int q = 0; q < 4; ++q) for (int e = 0; e < 4; ++e) s[q][e] = _mm512_setzero_ps();
    uint64_t k = 0;
    for (; k + 64u <= cols; k += 64u) {
        const __m512i b0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(row + k / 2u)));
        const __m512i b1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(row + k / 2u + 16u)));
        const __m512 l0 = _mm512_permutexvar_ps(b0, t4), h0 = _mm512_permutexvar_ps(_mm512_srli_epi32(b0, 4), t4);
        const __m512 l1 = _mm512_permutexvar_ps(b1, t4), h1 = _mm512_permutexvar_ps(_mm512_srli_epi32(b1, 4), t4);
        for (uint64_t q = 0; q < n; ++q) {
            s[q][0] = _mm512_fmadd_ps(l0, _mm512_loadu_ps(x[q] + k), s[q][0]);
            s[q][1] = _mm512_fmadd_ps(h0, _mm512_loadu_ps(x[q] + k + 16u), s[q][1]);
            s[q][2] = _mm512_fmadd_ps(l1, _mm512_loadu_ps(x[q] + k + 32u), s[q][2]);
            s[q][3] = _mm512_fmadd_ps(h1, _mm512_loadu_ps(x[q] + k + 48u), s[q][3]);
        }
    }
    if (k < cols) {
        const __m512i b0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(row + k / 2u)));
        const __m512 l0 = _mm512_permutexvar_ps(b0, t4), h0 = _mm512_permutexvar_ps(_mm512_srli_epi32(b0, 4), t4);
        for (uint64_t q = 0; q < n; ++q) {
            s[q][0] = _mm512_fmadd_ps(l0, _mm512_loadu_ps(x[q] + k), s[q][0]);
            s[q][1] = _mm512_fmadd_ps(h0, _mm512_loadu_ps(x[q] + k + 16u), s[q][1]);
        }
    }
    for (uint64_t q = 0; q < n; ++q) v[q] = x86_avx2__zzprivate_sum4_512(s[q][0], s[q][1], s[q][2], s[q][3]);
}
static X86_AVX2__ZZPRIVATE_AVX512 inline float x86_avx2__zzprivate_pair_dot_512(uint64_t d, const uint8_t* row, const float* x, uint64_t cols,
                                                                                __m512 t4, const __m512i* t8) {
    if (d == 4u) return x86_avx2__zzprivate_dot4_512(row, x, cols, t4);
    if (d == 8u) return x86_avx2__zzprivate_dot8_512(row, x, cols, t8);
    return x86_avx2__zzprivate_dot16_512((const uint16_t*)(const void*)row, x, cols);
}
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_row_dots_512(uint64_t d, const uint8_t* row, float scale, const float* xf,
                                                                               const uint32_t* rows, const uint16_t* w, uint64_t count,
                                                                               uint64_t cols, uint64_t out_rows, uint64_t r, uint16_t* out,
                                                                               float* acc, unsigned int* over, __m512 t4, const __m512i* t8) {
    float v4[4];
    if (count == 0u) return;
    if (d == 4u) {
        const float* x[4] = { xf, xf + (count > 1u ? cols : 0u), xf + (count > 2u ? 2u * cols : 0u), xf + (count > 3u ? 3u * cols : 0u) };
        if (count == 1u) v4[0] = x86_avx2__zzprivate_dot4_512(row, xf, cols, t4);
        else if (count == 2u) x86_avx2__zzprivate_dot4_pairs_512(row, x, 2u, cols, t4, v4);
        else if (count == 3u) x86_avx2__zzprivate_dot4_pairs_512(row, x, 3u, cols, t4, v4);
        else x86_avx2__zzprivate_dot4_pairs_512(row, x, 4u, cols, t4, v4);
    } else {
        for (uint64_t j = 0; j < count; ++j) v4[j] = x86_avx2__zzprivate_pair_dot_512(d, row, xf + j * cols, cols, t4, t8);
    }
    for (uint64_t j = 0; j < count; ++j) {
        const float v = v4[j] * scale;
        const uint64_t at = (uint64_t)rows[2u * j + 1u] * out_rows + r;
        if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
        else acc[at] += v * _cvtsh_ss(w[rows[2u * j]]);
    }
}
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_row_pairs_512(uint64_t d, const uint8_t* row, float scale, const float* xf,
                                                                                const uint32_t* rows, const uint16_t* w, uint64_t count,
                                                                                uint64_t cols, uint64_t out_rows, uint64_t r, uint16_t* out,
                                                                                float* acc, unsigned int* over, __m512 t4, const __m512i* t8) {
    float* wf = count > X86_AVX2__ZZPRIVATE_FUSED_PAIRS ? x86_avx2__zzprivate_rows_scratch(cols) : 0;
    if (!wf) {
        for (uint64_t j = 0; j < count; j += X86_AVX2__ZZPRIVATE_FUSED_PAIRS) {
            const uint64_t n = count - j < X86_AVX2__ZZPRIVATE_FUSED_PAIRS ? count - j : X86_AVX2__ZZPRIVATE_FUSED_PAIRS;
            x86_avx2__zzprivate_row_dots_512(d, row, scale, xf + j * cols, rows + 2u * j, w, n, cols, out_rows, r, out, acc, over, t4, t8);
        }
        return;
    }
    x86_avx2__zzprivate_decode_row_512(d, row, cols, wf, t4, t8);
    float v4[4];
    uint64_t j = 0;
    for (; j + 4u <= count; j += 4u) {
        const float* x0 = xf + j * cols;
        const float* x1 = x0 + cols;
        const float* x2 = x1 + cols;
        const float* x3 = x2 + cols;
        __m512 s[4][4];
        for (int q = 0; q < 4; ++q) for (int e = 0; e < 4; ++e) s[q][e] = _mm512_setzero_ps();
        uint64_t k = 0;
        for (; k + 64u <= cols; k += 64u) {
            X86_AVX2__ZZPRIVATE_FOUR(s, 0, k);
            X86_AVX2__ZZPRIVATE_FOUR(s, 1, k + 16u);
            X86_AVX2__ZZPRIVATE_FOUR(s, 2, k + 32u);
            X86_AVX2__ZZPRIVATE_FOUR(s, 3, k + 48u);
        }
        if (k < cols) {
            X86_AVX2__ZZPRIVATE_FOUR(s, 0, k);
            X86_AVX2__ZZPRIVATE_FOUR(s, 1, k + 16u);
        }
        for (int q = 0; q < 4; ++q) v4[q] = x86_avx2__zzprivate_sum4_512(s[q][0], s[q][1], s[q][2], s[q][3]);
        for (uint64_t q = 0; q < 4u; ++q) {
            const float v = v4[q] * scale;
            const uint64_t at = (uint64_t)rows[2u * (j + q) + 1u] * out_rows + r;
            if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
            else acc[at] += v * _cvtsh_ss(w[rows[2u * (j + q)]]);
        }
    }
    for (; j < count; ++j) {
        const float v = x86_avx2__zzprivate_dotf_512(wf, xf + j * cols, cols) * scale;
        const uint64_t at = (uint64_t)rows[2u * j + 1u] * out_rows + r;
        if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
        else acc[at] += v * _cvtsh_ss(w[rows[2u * j]]);
    }
}
#undef X86_AVX2__ZZPRIVATE_FOUR
/* ⭐ SIX ROWS BY FOUR PAIRS, A SUM A PASS. A dot sums chunk `c` of 16 columns into sum `c % 4`, and sum `e` sees only the
 *   chunks e, e + 4, e + 8, … — so one pass over those gives every (row, pair) of the tile its sum `e` in one register: 24
 *   sums in flight, each load of a decoded row serving four products and each load of a pair's `x` six. The four passes'
 *   sums are kept and added as the dots above add theirs, `(0 + 1) + (2 + 3)` and across the lanes: the gemv's bits.
 *   `MEASURED` (`scripts/tq_cpu_gemm_tiles_probe.c`, 4 threads, every answer the four-sum gemv's to the bit): the 35B's dense
 *   12,352 × 2,048 over 256 pairs 324 -> 430 GFLOP/s and the 27B's gate and up, 34,816 × 5,120, 286 -> 421, against two
 *   rows by three pairs with all four sums in registers. ⛳ Called with a literal `n`, so the sums stay in registers. */
#define X86_AVX2__ZZPRIVATE_TILE_ROWS 6u
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_six_rows_512(const float* const* w, const float* const* xq, uint64_t n,
                                                                               uint64_t cols, float v[6][4]) {
    __m512 keep[4][6][4];
    for (uint64_t e = 0; e < 4u; ++e) {
        __m512 s[6][4];
        for (int a = 0; a < 6; ++a) for (uint64_t q = 0; q < n; ++q) s[a][q] = _mm512_setzero_ps();
        for (uint64_t k = 16u * e; k < cols; k += 64u) {
            __m512 x[4];
            for (uint64_t q = 0; q < n; ++q) x[q] = _mm512_loadu_ps(xq[q] + k);
            for (int a = 0; a < 6; ++a) {
                const __m512 w_ = _mm512_load_ps(w[a] + k);
                for (uint64_t q = 0; q < n; ++q) s[a][q] = _mm512_fmadd_ps(w_, x[q], s[a][q]);
            }
        }
        for (int a = 0; a < 6; ++a) for (uint64_t q = 0; q < n; ++q) keep[e][a][q] = s[a][q];
    }
    for (int a = 0; a < 6; ++a)
        for (uint64_t q = 0; q < n; ++q) v[a][q] = x86_avx2__zzprivate_sum4_512(keep[0][a][q], keep[1][a][q], keep[2][a][q], keep[3][a][q]);
}
/* Rows `r` to `r + 5` of a matrix against their pairs, as `x86_avx2__zzprivate_row_pairs_512` takes one: from the codes a row
 * at a time for a few pairs, else decoded into the thread's scratch and four pairs at a time. */
static X86_AVX2__ZZPRIVATE_AVX512 inline void x86_avx2__zzprivate_rows6_pairs_512(uint64_t d, const uint8_t* const* row, const float* scale,
                                                                                  const float* xf, const uint32_t* rows, const uint16_t* w,
                                                                                  uint64_t count, uint64_t cols, uint64_t out_rows, uint64_t r,
                                                                                  uint16_t* out, float* acc, unsigned int* over, __m512 t4,
                                                                                  const __m512i* t8) {
    float* room = count > X86_AVX2__ZZPRIVATE_FUSED_PAIRS ? x86_avx2__zzprivate_rows_scratch(X86_AVX2__ZZPRIVATE_TILE_ROWS * cols) : 0;
    if (!room) {
        for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS; ++a)
            x86_avx2__zzprivate_row_pairs_512(d, row[a], scale[a], xf, rows, w, count, cols, out_rows, r + a, out, acc, over, t4, t8);
        return;
    }
    const float* wf[6];
    for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS; ++a) {
        x86_avx2__zzprivate_decode_row_512(d, row[a], cols, room + a * cols, t4, t8);
        wf[a] = room + a * cols;
    }
    for (uint64_t j = 0; j < count; j += 4u) {
        const uint64_t n = count - j < 4u ? count - j : 4u;
        const float* xq[4] = { xf + j * cols, xf + (j + (n > 1u ? 1u : 0u)) * cols, xf + (j + (n > 2u ? 2u : 0u)) * cols,
                               xf + (j + (n > 3u ? 3u : 0u)) * cols };
        float v[6][4];
        if (n == 4u) x86_avx2__zzprivate_six_rows_512(wf, xq, 4u, cols, v);
        else if (n == 3u) x86_avx2__zzprivate_six_rows_512(wf, xq, 3u, cols, v);
        else if (n == 2u) x86_avx2__zzprivate_six_rows_512(wf, xq, 2u, cols, v);
        else x86_avx2__zzprivate_six_rows_512(wf, xq, 1u, cols, v);
        for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS; ++a)
            for (uint64_t q = 0; q < n; ++q) {
                const float val = v[a][q] * scale[a];
                const uint64_t at = (uint64_t)rows[2u * (j + q) + 1u] * out_rows + r + a;
                if (out) out[at] = nn__kernels__zzabi_to_half(val, over);
                else acc[at] += val * _cvtsh_ss(w[rows[2u * (j + q)]]);
            }
    }
}
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_expert_rows_512(void* a) {
    const x86_avx2__rows_call* c = (const x86_avx2__rows_call*)a;
    __m512 t4; __m512i t8[4]; x86_avx2__zzprivate_tables_512(&t4, t8);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->out_rows * b / n, r1 = c->out_rows * (b + 1u) / n, row_bytes = c->cols * c->d / 8u;
    const uint64_t block = x86_avx2__zzprivate_pair_block(c->cols);
    for (uint64_t j0 = 0; j0 < c->count; j0 += block) {
        const uint64_t m = c->count - j0 < block ? c->count - j0 : block;
        const float* xf = c->xf + j0 * c->cols;
        const uint32_t* rows = c->rows + 2u * j0;
        uint64_t r = r0;
        for (; r + X86_AVX2__ZZPRIVATE_TILE_ROWS <= r1; r += X86_AVX2__ZZPRIVATE_TILE_ROWS) {
            const uint8_t* row[6]; float scale[6];
            for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS; ++a) {
                row[a] = c->weights + (r + a) * row_bytes; scale[a] = x86_avx2__zzprivate_gemm_scale(c->luts, c->d, r + a);
            }
            x86_avx2__zzprivate_rows6_pairs_512(c->d, row, scale, xf, rows, c->w, m, c->cols, c->out_rows, r, c->out, c->acc, c->over, t4, t8);
        }
        for (; r < r1; ++r)
            x86_avx2__zzprivate_row_pairs_512(c->d, c->weights + r * row_bytes, x86_avx2__zzprivate_gemm_scale(c->luts, c->d, r), xf, rows, c->w,
                                              m, c->cols, c->out_rows, r, c->out, c->acc, c->over, t4, t8);
    }
}
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_expert_groups_512(void* a) {
    const x86_avx2__rows_call* c = (const x86_avx2__rows_call*)a;
    __m512 t4; __m512i t8[4]; x86_avx2__zzprivate_tables_512(&t4, t8);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->total * b / n, t1 = c->total * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ) {
        while (t >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t r = t - base, row_bytes = c->cols * c->g.d[i] / 8u;
        const uint8_t* codes = (const uint8_t*)(uintptr_t)c->g.codes[i];
        const uint8_t* luts = (const uint8_t*)(uintptr_t)c->g.luts[i];
        if (t + X86_AVX2__ZZPRIVATE_TILE_ROWS <= t1 && r + X86_AVX2__ZZPRIVATE_TILE_ROWS <= c->g.out_rows[i]) {   /* six rows of one matrix */
            const uint8_t* row[6]; float scale[6];
            for (uint64_t a = 0; a < X86_AVX2__ZZPRIVATE_TILE_ROWS; ++a) {
                row[a] = codes + (r + a) * row_bytes; scale[a] = x86_avx2__zzprivate_gemm_scale(luts, c->g.d[i], r + a);
            }
            x86_avx2__zzprivate_rows6_pairs_512(c->g.d[i], row, scale, c->xf + c->xf_at[i], c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i],
                                                c->cols, c->g.out_rows[i], r, c->out + c->g.out_at[i], 0, c->over, t4, t8);
            t += X86_AVX2__ZZPRIVATE_TILE_ROWS;
        } else {
            x86_avx2__zzprivate_row_pairs_512(c->g.d[i], codes + r * row_bytes, x86_avx2__zzprivate_gemm_scale(luts, c->g.d[i], r), c->xf + c->xf_at[i],
                                              c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->cols, c->g.out_rows[i], r,
                                              c->out + c->g.out_at[i], 0, c->over, t4, t8);
            t += 1u;
        }
    }
}

/* ── THE INT8 BODIES ON AVX-512, AND ON VNNI (▶ int8_512.cuh): one text built twice — the multiply-add of bytes as
 *   AVX-512 has it, and as VNNI's `dpbusd` does it in one instruction; VNNI's where the processor has it. ── */
#define X86_AVX2__ZZPRIVATE_I8(name)          X86_AVX2__ZZPRIVATE_I8_512(name)
#define X86_AVX2__ZZPRIVATE_I8_TARGET         X86_AVX2__ZZPRIVATE_AVX512
#define X86_AVX2__ZZPRIVATE_I8_DOT(u, s)      _mm512_madd_epi16(_mm512_maddubs_epi16((u), (s)), _mm512_set1_epi16(1))
#define X86_AVX2__ZZPRIVATE_I8_SUM(a, u, s)   _mm512_add_epi32((a), X86_AVX2__ZZPRIVATE_I8_DOT(u, s))
#include "int8_512.cuh"
#undef X86_AVX2__ZZPRIVATE_I8
#undef X86_AVX2__ZZPRIVATE_I8_TARGET
#undef X86_AVX2__ZZPRIVATE_I8_DOT
#undef X86_AVX2__ZZPRIVATE_I8_SUM
#define X86_AVX2__ZZPRIVATE_I8(name)          X86_AVX2__ZZPRIVATE_I8_VNNI(name)
#define X86_AVX2__ZZPRIVATE_I8_TARGET         X86_AVX2__ZZPRIVATE_VNNI
#define X86_AVX2__ZZPRIVATE_I8_DOT(u, s)      _mm512_dpbusd_epi32(_mm512_setzero_si512(), (u), (s))
#define X86_AVX2__ZZPRIVATE_I8_SUM(a, u, s)   _mm512_dpbusd_epi32((a), (u), (s))
#include "int8_512.cuh"
#undef X86_AVX2__ZZPRIVATE_I8
#undef X86_AVX2__ZZPRIVATE_I8_TARGET
#undef X86_AVX2__ZZPRIVATE_I8_DOT
#undef X86_AVX2__ZZPRIVATE_I8_SUM

/* ══ ⭐⭐ ATTENTION OVER THE CACHE, ONE QUERY — the tiered decode's four doors ═══════════════════════════════════════
 * `attention_weights` and `attention_residual_mix` over the fp16 tier, `attention_weights_tq` and
 * `attention_residual_mix_tq` over the TurboQuant ones — nn's definitions (`kernels.cuh`), what a decoded position reads
 * its cache with. nn's bodies give a query head a block, and on one lane that block reads its kv head's rows for itself
 * and walks each value element down the whole cache: `MEASURED` (the doors bench, 24 heads over 4, 256 wide, 1,000
 * positions) 1.8 ms for the weights and 3.4 ms for the mix — the 35B on the CPU alone took 220 ms a token after 2,035
 * positions against 111 at the first. Here a pool block takes a kv head:
 *   the weights   a run of positions; each key row widened (or decoded) once and its dot taken with every query head of
 *                 the group, then a block a query head for the softmax's largest, its exponentials and their sum, in
 *                 nn's order (libm's `expf`, the sum over the positions in turn);
 *   the mix       16 elements of the head, the group's query heads four at a time against each value row: each
 *                 element's sum adds the positions one after another from the first, as nn's does, so the mix is
 *                 nn's to the bit wherever a multiply-add is one rounding in both.
 * Heads whole runs of 32 up to 512 wide, 4- and 8-bit tiers; anything else is the generic door. */
#define X86_AVX2__ZZPRIVATE_ATTEND_SPAN 64u      /* positions a block of the weights takes */
#define X86_AVX2__ZZPRIVATE_ATTEND_WIDTH 512u    /* the widest head */

typedef struct x86_avx2__attend_call {
    float* p; float* res; const float* qf; const uint16_t* kv; const uint8_t* codes; const uint16_t* scales;
    uint64_t q_heads, kv_heads, head_dim, length, d;
} x86_avx2__attend_call;

/* `n` 8-bit codes' levels times `scale` into `out` (whole runs of 32), through the halves' registers (AVX-512, symmetric
 * levels): the same floats as the table's. */
static X86_AVX2__ZZPRIVATE_AVX512 void x86_avx2__zzprivate_attend_row8_512(float* out, const uint8_t* codes, float scale, uint64_t n) {
    __m512i t8[4];
    const uint16_t* lev = nn__turboquant__zzabi_levels(8u);
    for (int i = 0; i < 4; ++i) t8[i] = _mm512_loadu_si512((const void*)(lev + 32 * i));
    const __m512 s = _mm512_set1_ps(scale);
    for (uint64_t k = 0; k < n; k += 32u) {
        __m512 lo, hi;
        x86_avx2__zzprivate_codes8_512(codes + k, t8, &lo, &hi);
        _mm512_storeu_ps(out + k, _mm512_mul_ps(s, lo));
        _mm512_storeu_ps(out + k + 16u, _mm512_mul_ps(s, hi));
    }
}
/* Elements `e0 .. e0 + n` of row `row` of a TurboQuant tier into `out`: each `scale · level`, as nn's element is (`e0` and
 * `n` multiples of 8). 8 bits: the halves' registers where they serve, else the table a code at a time — a gather of
 * eight is slower here (▶ the 8-bit rows' rates above); 4 bits: the 16 levels in two registers, eight codes a step. */
static inline void x86_avx2__zzprivate_attend_row_tq(float* out, const x86_avx2__attend_call* c, uint64_t row, uint64_t e0, uint64_t n) {
    const uint8_t* r = c->codes + row * nn__turboquant__row_bytes(c->d, c->head_dim);
    const float scale = _cvtsh_ss(c->scales[row]);
    if (c->d == 8u) {
        if (x86_avx2__zzprivate_wide8 && n % 32u == 0u) { x86_avx2__zzprivate_attend_row8_512(out, r + e0, scale, n); return; }
        for (uint64_t e = e0; e < e0 + n; ++e) out[e - e0] = scale * x86_avx2__zzprivate_levels8[r[e]];
        return;
    }
    const __m256 lo = _mm256_loadu_ps(x86_avx2__zzprivate_levels4), hi = _mm256_loadu_ps(x86_avx2__zzprivate_levels4 + 8);
    const __m256 sc = _mm256_set1_ps(scale);
    const __m256i shift = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4), low = _mm256_set1_epi32(15), seven = _mm256_set1_epi32(7);
    for (uint64_t e = e0; e < e0 + n; e += 8u) {
        const uint8_t* q = r + e / 2u;
        const __m128i pairs = _mm_cvtsi32_si128((int)((uint32_t)q[0] | (uint32_t)q[1] << 8 | (uint32_t)q[2] << 16 | (uint32_t)q[3] << 24));
        const __m256i at = _mm256_and_si256(_mm256_srlv_epi32(_mm256_cvtepu8_epi32(_mm_unpacklo_epi8(pairs, pairs)), shift), low);
        const __m256 level = _mm256_blendv_ps(_mm256_permutevar8x32_ps(lo, at), _mm256_permutevar8x32_ps(hi, at),
                                              _mm256_castsi256_ps(_mm256_cmpgt_epi32(at, seven)));
        _mm256_storeu_ps(out + (e - e0), _mm256_mul_ps(sc, level));
    }
}
/* Row `row` of the fp16 tier into `out`. */
static inline void x86_avx2__zzprivate_attend_row(float* out, const x86_avx2__attend_call* c, uint64_t row) {
    const uint16_t* r = c->kv + row * c->head_dim;
    for (uint64_t e = 0; e < c->head_dim; e += 8u) _mm256_storeu_ps(out + e, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r + e))));
}

/* The scores, a block a (kv head, run of positions): `p[h·length + t]`, scaled. */
static void x86_avx2__zzprivate_attend_scores(void* a) {
    const x86_avx2__attend_call* c = (const x86_avx2__attend_call*)a;
    const uint64_t runs = (c->length + X86_AVX2__ZZPRIVATE_ATTEND_SPAN - 1u) / X86_AVX2__ZZPRIVATE_ATTEND_SPAN;
    const uint64_t b = nn__silicon__block(), g = b / runs, t0 = (b % runs) * X86_AVX2__ZZPRIVATE_ATTEND_SPAN;
    const uint64_t t1 = t0 + X86_AVX2__ZZPRIVATE_ATTEND_SPAN < c->length ? t0 + X86_AVX2__ZZPRIVATE_ATTEND_SPAN : c->length;
    const uint64_t group = c->q_heads / c->kv_heads;
    const float scale = 1.0f / sqrtf((float)(uint32_t)c->head_dim);
    float kf[X86_AVX2__ZZPRIVATE_ATTEND_WIDTH] __attribute__((aligned(32)));
    for (uint64_t t = t0; t < t1; ++t) {
        if (c->codes) x86_avx2__zzprivate_attend_row_tq(kf, c, t * c->kv_heads + g, 0u, c->head_dim);
        else x86_avx2__zzprivate_attend_row(kf, c, t * c->kv_heads + g);
        for (uint64_t hg = 0; hg < group; ++hg) {
            const uint64_t h = g * group + hg;
            c->p[h * c->length + t] = x86_avx2__zzprivate_dotf(kf, c->qf + h * c->head_dim, c->head_dim) * scale;
        }
    }
}
/* The softmax's first half, a block a query head: its largest, `e^(score − largest)` in place and their sum, as nn's. */
static void x86_avx2__zzprivate_attend_softmax(void* a) {
    const x86_avx2__attend_call* c = (const x86_avx2__attend_call*)a;
    const uint64_t h = nn__silicon__block();
    float* ph = c->p + h * c->length;
    float top = -NN__KERNELS__INFINITY;
    for (uint64_t t = 0; t < c->length; ++t) if (ph[t] > top) top = ph[t];
    float sum = 0.0f;
    for (uint64_t t = 0; t < c->length; ++t) { ph[t] = expf(ph[t] - top); sum += ph[t]; }
    c->res[h * NN__ATTENTION__RESIDUAL_FLOATS(c->head_dim)] = top;
    c->res[h * NN__ATTENTION__RESIDUAL_FLOATS(c->head_dim) + 1u] = sum;
}
/* The mix, a block a (kv head, 16 elements): the group's heads four at a time, each value row read once a four. */
static void x86_avx2__zzprivate_attend_mix(void* a) {
    const x86_avx2__attend_call* c = (const x86_avx2__attend_call*)a;
    const uint64_t chunks = c->head_dim / 16u, b = nn__silicon__block(), g = b / chunks, e0 = (b % chunks) * 16u;
    const uint64_t group = c->q_heads / c->kv_heads, w = NN__ATTENTION__RESIDUAL_FLOATS(c->head_dim);
    float row[16] __attribute__((aligned(32)));
    for (uint64_t h0 = 0; h0 < group; h0 += 4u) {
        /* four heads' weights; past the group's end a head repeats the last, and its sums are not stored */
        const float* ph[4];
        for (uint64_t j = 0; j < 4u; ++j) ph[j] = c->p + (g * group + (h0 + j < group ? h0 + j : group - 1u)) * c->length;
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        for (uint64_t t = 0; t < c->length; ++t) {
            __m256 v0, v1;
            if (c->codes) {
                x86_avx2__zzprivate_attend_row_tq(row, c, t * c->kv_heads + g, e0, 16u);
                v0 = _mm256_load_ps(row); v1 = _mm256_load_ps(row + 8u);
            } else {
                const uint16_t* r = c->kv + (t * c->kv_heads + g) * c->head_dim + e0;
                v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)r));
                v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r + 8u)));
            }
            const __m256 w0 = _mm256_set1_ps(ph[0][t]), w1 = _mm256_set1_ps(ph[1][t]);
            const __m256 w2 = _mm256_set1_ps(ph[2][t]), w3 = _mm256_set1_ps(ph[3][t]);
            s0 = _mm256_fmadd_ps(w0, v0, s0); s1 = _mm256_fmadd_ps(w0, v1, s1);
            s2 = _mm256_fmadd_ps(w1, v0, s2); s3 = _mm256_fmadd_ps(w1, v1, s3);
            s4 = _mm256_fmadd_ps(w2, v0, s4); s5 = _mm256_fmadd_ps(w2, v1, s5);
            s6 = _mm256_fmadd_ps(w3, v0, s6); s7 = _mm256_fmadd_ps(w3, v1, s7);
        }
        const __m256 lo[4] = { s0, s2, s4, s6 }, hi[4] = { s1, s3, s5, s7 };
        for (uint64_t j = 0; j < 4u && h0 + j < group; ++j) {
            float* acc = c->res + (g * group + h0 + j) * w + 2u + e0;
            _mm256_storeu_ps(acc, lo[j]); _mm256_storeu_ps(acc + 8u, hi[j]);
        }
    }
}
/* ⭐ THE MIX OVER A LONG CACHE, a block a (kv head, `width` elements): a cache line of a row (64 elements at 8 bits) is
 *   read by one block, not four, and a tile of `X86_AVX2__ZZPRIVATE_MIX_TILE` rows is decoded once, then its 16-element
 *   slices summed over the tile in turn — each element's sum still the positions one after another from the first, so
 *   the bytes are the 16-element mix's. `MEASURED` the motive: the 4B on the CPU alone 116 ms a token after 4,147
 *   positions, 529 after 32,807 (measurements/2026-10-29_x86_attend_long_precision.md). */
#define X86_AVX2__ZZPRIVATE_MIX_TILE 32u
static void x86_avx2__zzprivate_attend_mix_wide(void* a) {
    const x86_avx2__attend_call* c = (const x86_avx2__attend_call*)a;
    const uint64_t width = c->d >> 8u, chunks = c->head_dim / width, b = nn__silicon__block(), g = b / chunks, e0 = (b % chunks) * width;
    const uint64_t d = c->d & 255u, group = c->q_heads / c->kv_heads, w = NN__ATTENTION__RESIDUAL_FLOATS(c->head_dim);
    float tile[X86_AVX2__ZZPRIVATE_MIX_TILE * 64u] __attribute__((aligned(32)));
    float acc[4u * 64u] __attribute__((aligned(32)));
    for (uint64_t h0 = 0; h0 < group; h0 += 4u) {
        /* four heads' weights; past the group's end a head repeats the last, and its sums are not stored */
        const float* ph[4];
        for (uint64_t j = 0; j < 4u; ++j) ph[j] = c->p + (g * group + (h0 + j < group ? h0 + j : group - 1u)) * c->length;
        for (uint64_t i = 0; i < 4u * 64u; i += 8u) _mm256_store_ps(acc + i, _mm256_setzero_ps());
        for (uint64_t t0 = 0; t0 < c->length; t0 += X86_AVX2__ZZPRIVATE_MIX_TILE) {
            const uint64_t t1 = t0 + X86_AVX2__ZZPRIVATE_MIX_TILE < c->length ? t0 + X86_AVX2__ZZPRIVATE_MIX_TILE : c->length;
            for (uint64_t t = t0; t < t1; ++t) {
                float* out = tile + (t - t0) * width;
                if (d) {
                    const x86_avx2__attend_call at = { 0, 0, 0, 0, c->codes, c->scales, c->q_heads, c->kv_heads, c->head_dim, 0, d };
                    x86_avx2__zzprivate_attend_row_tq(out, &at, t * c->kv_heads + g, e0, width);
                } else {
                    const uint16_t* r = c->kv + (t * c->kv_heads + g) * c->head_dim + e0;
                    for (uint64_t e = 0; e < width; e += 8u) _mm256_store_ps(out + e, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(r + e))));
                }
            }
            for (uint64_t q = 0; q < width; q += 16u) {
                __m256 s0 = _mm256_load_ps(acc + q), s1 = _mm256_load_ps(acc + q + 8u);
                __m256 s2 = _mm256_load_ps(acc + 64u + q), s3 = _mm256_load_ps(acc + 64u + q + 8u);
                __m256 s4 = _mm256_load_ps(acc + 128u + q), s5 = _mm256_load_ps(acc + 128u + q + 8u);
                __m256 s6 = _mm256_load_ps(acc + 192u + q), s7 = _mm256_load_ps(acc + 192u + q + 8u);
                for (uint64_t t = t0; t < t1; ++t) {
                    const __m256 v0 = _mm256_load_ps(tile + (t - t0) * width + q), v1 = _mm256_load_ps(tile + (t - t0) * width + q + 8u);
                    const __m256 w0 = _mm256_set1_ps(ph[0][t]), w1 = _mm256_set1_ps(ph[1][t]);
                    const __m256 w2 = _mm256_set1_ps(ph[2][t]), w3 = _mm256_set1_ps(ph[3][t]);
                    s0 = _mm256_fmadd_ps(w0, v0, s0); s1 = _mm256_fmadd_ps(w0, v1, s1);
                    s2 = _mm256_fmadd_ps(w1, v0, s2); s3 = _mm256_fmadd_ps(w1, v1, s3);
                    s4 = _mm256_fmadd_ps(w2, v0, s4); s5 = _mm256_fmadd_ps(w2, v1, s5);
                    s6 = _mm256_fmadd_ps(w3, v0, s6); s7 = _mm256_fmadd_ps(w3, v1, s7);
                }
                _mm256_store_ps(acc + q, s0); _mm256_store_ps(acc + q + 8u, s1);
                _mm256_store_ps(acc + 64u + q, s2); _mm256_store_ps(acc + 64u + q + 8u, s3);
                _mm256_store_ps(acc + 128u + q, s4); _mm256_store_ps(acc + 128u + q + 8u, s5);
                _mm256_store_ps(acc + 192u + q, s6); _mm256_store_ps(acc + 192u + q + 8u, s7);
            }
        }
        for (uint64_t j = 0; j < 4u && h0 + j < group; ++j)
            for (uint64_t e = 0; e < width; e += 8u) _mm256_storeu_ps(c->res + (g * group + h0 + j) * w + 2u + e0 + e, _mm256_load_ps(acc + 64u * j + e));
    }
}
/* The mix on the pool: the widest of 64, 32 and 16 elements a block that still gives every thread a block. */
static inline void x86_avx2__zzprivate_attend_mix_run(x86_avx2__attend_call* c) {
    const uint64_t threads = x86_avx2__zzpackage_threads();
    uint64_t width = 64u;
    while (width > 16u && (c->head_dim % width != 0u || c->kv_heads * (c->head_dim / width) < threads)) width /= 2u;
    if (width == 16u) {
        x86_avx2__zzpackage_run((uint32_t)(c->kv_heads * (c->head_dim / 16u)), x86_avx2__zzprivate_attend_mix, c);
        return;
    }
    c->d |= width << 8u;                       /* the block's width rides above the tier's bits */
    x86_avx2__zzpackage_run((uint32_t)(c->kv_heads * (c->head_dim / width)), x86_avx2__zzprivate_attend_mix_wide, c);
}
static inline bool x86_avx2__zzprivate_attend_fit(uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d) {
    return kv_heads != 0u && q_heads % kv_heads == 0u && head_dim != 0u && head_dim % 32u == 0u
        && head_dim <= X86_AVX2__ZZPRIVATE_ATTEND_WIDTH && length != 0u && (d == 0u || d == 4u || d == 8u)
        && x86_avx2__zzprivate_x_room_for(q_heads * head_dim);
}
/* The weights: the scores on the pool, then a block a query head for the softmax's first half. */
static inline void x86_avx2__zzprivate_attend_weights(x86_avx2__attend_call* c, const uint16_t* q) {
    x86_avx2__zzpackage_start();
    for (uint64_t i = 0; i < c->q_heads * c->head_dim; ++i) x86_avx2__zzprivate_xf[i] = _cvtsh_ss(q[i]);
    c->qf = x86_avx2__zzprivate_xf;
    const uint64_t runs = (c->length + X86_AVX2__ZZPRIVATE_ATTEND_SPAN - 1u) / X86_AVX2__ZZPRIVATE_ATTEND_SPAN;
    x86_avx2__zzpackage_run((uint32_t)(c->kv_heads * runs), x86_avx2__zzprivate_attend_scores, c);
    x86_avx2__zzpackage_run((uint32_t)c->q_heads, x86_avx2__zzprivate_attend_softmax, c);
}
static void x86_avx2__override_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                 uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!x86_avx2__zzprivate_attend_fit(q_heads, kv_heads, head_dim, length, 0u)) {
        nn__attention__zzabi_launch_weights(p, res, q, k, q_heads, kv_heads, head_dim, length); return;
    }
    x86_avx2__attend_call c = { p, res, 0, k, 0, 0, q_heads, kv_heads, head_dim, length, 0u };
    x86_avx2__zzprivate_attend_weights(&c, q);
}
static void x86_avx2__override_attention_weights_tq(float* p, float* res, const uint16_t* q, const uint8_t* codes, const uint16_t* scales,
                                                    uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d) {
    if (!x86_avx2__zzprivate_attend_fit(q_heads, kv_heads, head_dim, length, d) || d == 0u) {
        nn__attention__zzabi_launch_weights_tq(p, res, q, codes, scales, q_heads, kv_heads, head_dim, length, d); return;
    }
    x86_avx2__attend_call c = { p, res, 0, 0, codes, scales, q_heads, kv_heads, head_dim, length, d };
    x86_avx2__zzprivate_attend_weights(&c, q);
}
static void x86_avx2__override_attention_residual_mix(float* res, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                                      uint64_t head_dim, uint64_t length) {
    if (!x86_avx2__zzprivate_attend_fit(q_heads, kv_heads, head_dim, length, 0u)) {
        nn__attention__zzabi_launch_residual_mix(res, p, v, q_heads, kv_heads, head_dim, length); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__attend_call c = { (float*)(uintptr_t)p, res, 0, v, 0, 0, q_heads, kv_heads, head_dim, length, 0u };
    x86_avx2__zzprivate_attend_mix_run(&c);
}
static void x86_avx2__override_attention_residual_mix_tq(float* res, const float* p, const uint8_t* codes, const uint16_t* scales,
                                                         uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                         uint64_t d) {
    if (!x86_avx2__zzprivate_attend_fit(q_heads, kv_heads, head_dim, length, d) || d == 0u) {
        nn__attention__zzabi_launch_residual_mix_tq(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__attend_call c = { (float*)(uintptr_t)p, res, 0, 0, codes, scales, q_heads, kv_heads, head_dim, length, d };
    x86_avx2__zzprivate_attend_mix_run(&c);
}

/* ══ ⭐⭐ THE PROMPT'S ATTENTION — the grouped causal scores, and every mix of nn's softmax ════════════════════════════
 * `attention_causal_scores_grouped`, `attention_causal_mix` and `attention_mix`, nn's definitions: a prompt's `rows`
 * queries at positions `first ..`, row `r` over the cache up to its own position. nn's bodies walked each value element
 * down the cache on one lane, and the scores a position at a time — `REASONED` from the 35B's prompt on the CPU alone,
 * 91 ms a position with ~25 of it the GEMMs. Here:
 *   the scores   a pool block a (kv head, run of 64 positions): each key row widened once, its dot taken with the query
 *                heads of the group in every row that reaches it;
 *   a mix        a block a (row, kv head, 16 elements), the group's heads four at a time, each element's sum the
 *                positions in turn from the first — nn's order. ⛳ `attention_mix` is the same body at one row, so a
 *                position decoded and the same position read as a prompt row are the same bytes.
 * The softmax between them is nn's. Heads whole runs of 32 up to 512 wide; anything else is the generic door. */
typedef struct x86_avx2__causal_call {
    float* p; const float* qf; const uint16_t* k; const uint16_t* v; uint16_t* o; unsigned int* over;
    uint64_t q_heads, kv_heads, head_dim, first, rows, span;   /* `span`: a (row, head)'s weights apart in `p` */
} x86_avx2__causal_call;

static void x86_avx2__zzprivate_causal_scores(void* a) {
    const x86_avx2__causal_call* c = (const x86_avx2__causal_call*)a;
    const uint64_t runs = (c->span + X86_AVX2__ZZPRIVATE_ATTEND_SPAN - 1u) / X86_AVX2__ZZPRIVATE_ATTEND_SPAN;
    const uint64_t b = nn__silicon__block(), g = b / runs, t0 = (b % runs) * X86_AVX2__ZZPRIVATE_ATTEND_SPAN;
    const uint64_t t1 = t0 + X86_AVX2__ZZPRIVATE_ATTEND_SPAN < c->span ? t0 + X86_AVX2__ZZPRIVATE_ATTEND_SPAN : c->span;
    const uint64_t group = c->q_heads / c->kv_heads, hd = c->head_dim;
    const float scale = 1.0f / sqrtf((float)(uint32_t)hd);
    float kf[X86_AVX2__ZZPRIVATE_ATTEND_WIDTH] __attribute__((aligned(32)));
    for (uint64_t t = t0; t < t1; ++t) {
        const uint16_t* kr = c->k + (t * c->kv_heads + g) * hd;
        for (uint64_t e = 0; e < hd; e += 8u) _mm256_store_ps(kf + e, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(kr + e))));
        for (uint64_t r = t > c->first ? t - c->first : 0u; r < c->rows; ++r)        /* the rows that reach position t */
            for (uint64_t hg = 0; hg < group; ++hg) {
                const uint64_t i = r * c->q_heads + g * group + hg;
                c->p[i * c->span + t] = x86_avx2__zzprivate_dotf(kf, c->qf + i * hd, hd) * scale;
            }
    }
}
static void x86_avx2__zzprivate_causal_mix(void* a) {
    const x86_avx2__causal_call* c = (const x86_avx2__causal_call*)a;
    const uint64_t hd = c->head_dim, chunks = hd / 16u, b = nn__silicon__block(), e0 = (b % chunks) * 16u;
    const uint64_t g = (b / chunks) % c->kv_heads, r = b / chunks / c->kv_heads, len = c->first + r + 1u;
    const uint64_t group = c->q_heads / c->kv_heads;
    for (uint64_t h0 = 0; h0 < group; h0 += 4u) {
        /* four heads' weights; past the group's end a head repeats the last, and its answer is not written */
        const float* ph[4];
        for (uint64_t j = 0; j < 4u; ++j) ph[j] = c->p + (r * c->q_heads + g * group + (h0 + j < group ? h0 + j : group - 1u)) * c->span;
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        for (uint64_t t = 0; t < len; ++t) {
            const uint16_t* vr = c->v + (t * c->kv_heads + g) * hd + e0;
            const __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)vr));
            const __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(vr + 8u)));
            const __m256 w0 = _mm256_set1_ps(ph[0][t]), w1 = _mm256_set1_ps(ph[1][t]);
            const __m256 w2 = _mm256_set1_ps(ph[2][t]), w3 = _mm256_set1_ps(ph[3][t]);
            s0 = _mm256_fmadd_ps(w0, v0, s0); s1 = _mm256_fmadd_ps(w0, v1, s1);
            s2 = _mm256_fmadd_ps(w1, v0, s2); s3 = _mm256_fmadd_ps(w1, v1, s3);
            s4 = _mm256_fmadd_ps(w2, v0, s4); s5 = _mm256_fmadd_ps(w2, v1, s5);
            s6 = _mm256_fmadd_ps(w3, v0, s6); s7 = _mm256_fmadd_ps(w3, v1, s7);
        }
        float sums[4][16];
        _mm256_storeu_ps(sums[0], s0); _mm256_storeu_ps(sums[0] + 8, s1); _mm256_storeu_ps(sums[1], s2); _mm256_storeu_ps(sums[1] + 8, s3);
        _mm256_storeu_ps(sums[2], s4); _mm256_storeu_ps(sums[2] + 8, s5); _mm256_storeu_ps(sums[3], s6); _mm256_storeu_ps(sums[3] + 8, s7);
        for (uint64_t j = 0; j < 4u && h0 + j < group; ++j) {
            uint16_t* oh = c->o + (r * c->q_heads + g * group + h0 + j) * hd + e0;
            for (uint64_t e = 0; e < 16u; ++e) oh[e] = nn__kernels__zzabi_to_half(sums[j][e], c->over);
        }
    }
}
static inline bool x86_avx2__zzprivate_causal_fit(uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim) {
    return kv_heads != 0u && q_heads % kv_heads == 0u && head_dim != 0u && head_dim % 32u == 0u
        && head_dim <= X86_AVX2__ZZPRIVATE_ATTEND_WIDTH;
}
static void x86_avx2__override_attention_causal_scores_grouped(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                               uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows) {
    if (!x86_avx2__zzprivate_causal_fit(q_heads, kv_heads, head_dim) || rows == 0u || !x86_avx2__zzprivate_x_room_for(rows * q_heads * head_dim)) {
        nn__attention__zzabi_launch_causal_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, first, rows); return;
    }
    x86_avx2__zzpackage_start();
    for (uint64_t i = 0; i < rows * q_heads * head_dim; ++i) x86_avx2__zzprivate_xf[i] = _cvtsh_ss(q[i]);
    x86_avx2__causal_call c = { p, x86_avx2__zzprivate_xf, k, 0, 0, 0, q_heads, kv_heads, head_dim, first, rows, first + rows };
    const uint64_t runs = (first + rows + X86_AVX2__ZZPRIVATE_ATTEND_SPAN - 1u) / X86_AVX2__ZZPRIVATE_ATTEND_SPAN;
    x86_avx2__zzpackage_run((uint32_t)(kv_heads * runs), x86_avx2__zzprivate_causal_scores, &c);
}
static void x86_avx2__override_attention_causal_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                                    uint64_t head_dim, uint64_t first, uint64_t rows, unsigned int* over) {
    if (!x86_avx2__zzprivate_causal_fit(q_heads, kv_heads, head_dim) || rows == 0u) {
        nn__attention__zzabi_launch_causal_mix(o, p, v, q_heads, kv_heads, head_dim, first, rows, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__causal_call c = { (float*)(uintptr_t)p, 0, 0, v, o, over, q_heads, kv_heads, head_dim, first, rows, first + rows };
    x86_avx2__zzpackage_run((uint32_t)(rows * kv_heads * (head_dim / 16u)), x86_avx2__zzprivate_causal_mix, &c);
}
static void x86_avx2__override_attention_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                             uint64_t head_dim, uint64_t length, unsigned int* over) {
    if (!x86_avx2__zzprivate_causal_fit(q_heads, kv_heads, head_dim) || length == 0u) {
        nn__attention__zzabi_launch_mix(o, p, v, q_heads, kv_heads, head_dim, length, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__causal_call c = { (float*)(uintptr_t)p, 0, 0, v, o, over, q_heads, kv_heads, head_dim, length - 1u, 1u, length };
    x86_avx2__zzpackage_run((uint32_t)(kv_heads * (head_dim / 16u)), x86_avx2__zzprivate_causal_mix, &c);
}

/* ── THE `k` LARGEST: a sampler's top k over a vocabulary, a router's over its experts ──────────────────────────────── */
/* ⭐ ONE PASS, WHERE nn's ONE LANE TAKES `k`. nn's rounds each read every value and ask it whether an earlier round took
 *   it — on one lane ~29 ms a token for 248,320 logits top 20. Here the `k` largest so far are held in order, largest
 *   first, as WORDS that order as nn's rounds pick: a value's order above (the two zeros one), its index taken from all
 *   ones below, so of equal values the first is the larger word; a NaN is never held. Eight values at a time are widened
 *   and compared with the `k`-th held one, and only those at or above it are looked at. The picks are the held words in
 *   their order, and each weight is nn's expression over a sum added in the rounds' order, so the indices and the
 *   weights are nn's to the bit. */
static inline uint64_t x86_avx2__zzprivate_top_k_word(uint16_t h, uint64_t i) {
    const uint32_t z = (h & 0x7FFFu) == 0u ? 0u : h;
    const uint32_t order = (z & 0x8000u) != 0u ? (~z & 0xFFFFu) : (z | 0x8000u);
    return ((uint64_t)order << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)i);
}
/* Value `i`, not a NaN, among the held words if it is one of the `k` largest so far; the floor the `k`-th held one's
 * value once `k` are held. */
static inline void x86_avx2__zzprivate_top_k_offer(uint64_t* held, uint64_t* count, uint64_t k, float* floor_v,
                                                   const uint16_t* x, uint64_t i) {
    const uint64_t word = x86_avx2__zzprivate_top_k_word(x[i], i);
    uint64_t j = *count;
    if (j == k) {
        if (word <= held[k - 1u]) return;
        j = k - 1u;
    } else {
        *count = j + 1u;
    }
    for (; j > 0u && held[j - 1u] < word; --j) held[j] = held[j - 1u];
    held[j] = word;
    if (*count == k) *floor_v = _cvtsh_ss(x[0xFFFFFFFFu - (uint32_t)held[k - 1u]]);
}
static void x86_avx2__override_vector_top_k(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, uint64_t n,
                                            uint64_t k) {
    if (n == 0u || n > NN__KERNELS__INDEX_EXACT || k == 0u || k > NN__VECTOR__TOP_K_MAX || stride == 0u) {
        nn__vector__zzabi_launch_top_k(values, stride, weights, x, n, k);
        return;
    }
    uint64_t held[NN__VECTOR__TOP_K_MAX];
    float picked[NN__VECTOR__TOP_K_MAX];
    uint64_t count = 0u;
    float floor_v = -NN__KERNELS__INFINITY;
    uint64_t i = 0u;
    for (; i + 8u <= n; i += 8u) {
        const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(const void*)(x + i)));
        uint32_t near = (uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_set1_ps(floor_v), _CMP_GE_OQ));
        for (; near != 0u; near &= near - 1u)
            x86_avx2__zzprivate_top_k_offer(held, &count, k, &floor_v, x, i + (uint64_t)__builtin_ctz(near));
    }
    for (; i < n; ++i)
        if (_cvtsh_ss(x[i]) >= floor_v) x86_avx2__zzprivate_top_k_offer(held, &count, k, &floor_v, x, i);
    for (uint64_t r = 0u; r < k; ++r) {
        const uint64_t at = r < count ? (uint64_t)(0xFFFFFFFFu - (uint32_t)held[r]) : n;
        values[r * stride] = at;
        picked[r] = r < count ? nn__silicon__half_to_float(x[at]) : -NN__KERNELS__INFINITY;
    }
    float sum = 0.0f;
    for (uint64_t r = 0u; r < k; ++r) sum += nn__silicon__expf(picked[r] - picked[0]);
    unsigned int unreported = 0u;                                        /* nn's top_k raises no overflow either */
    for (uint64_t r = 0u; r < k; ++r) weights[r] = nn__kernels__zzabi_to_half(nn__silicon__expf(picked[r] - picked[0]) / sum, &unreported);
}

/* Whether the AVX-512 bodies go in: the processor has the four extensions they are compiled for, and
 * `SILVANN_CPU_AVX512` is not `0`; the 8-bit ones also need the levels symmetric; the int8 ones' VNNI build, VNNI too
 * and `SILVANN_CPU_VNNI` not `0`. */
static inline void x86_avx2__zzprivate_install_wide(void) {
    const uint16_t* lev = nn__turboquant__zzabi_levels(4u);
    for (int i = 0; i < 16; ++i) x86_avx2__zzprivate_levels4[i] = _cvtsh_ss(lev[i]);
    const char* want = getenv("SILVANN_CPU_AVX512");
    __builtin_cpu_init();
    x86_avx2__zzprivate_wide = !(want != 0 && want[0] == '0')
                            && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
                            && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
    const uint16_t* lev8 = nn__turboquant__zzabi_levels(8u);
    bool symmetric = true;
    for (int i = 0; i < 128; ++i) symmetric = symmetric && lev8[255 - i] == (uint16_t)(lev8[i] ^ 0x8000u);
    x86_avx2__zzprivate_wide8 = x86_avx2__zzprivate_wide && symmetric;
    const char* vnni = getenv("SILVANN_CPU_VNNI");
    x86_avx2__zzprivate_vnni = x86_avx2__zzprivate_wide && !(vnni != 0 && vnni[0] == '0') && __builtin_cpu_supports("avx512vnni");
}

/* Into nn's table, once, when the family is first asked for it. */
static inline void x86_avx2__override_doors(nn__doors* doors) {
    const uint16_t* lev = nn__turboquant__zzabi_levels(8u);
    for (int i = 0; i < 256; ++i) x86_avx2__zzprivate_levels8[i] = _cvtsh_ss(lev[i]);
    x86_avx2__zzprivate_install_wide();
    const long l2 = sys__os__cpu_l2_bytes();   /* the core's L2, for the GEMM's blocks of pairs */
    if (l2 > 0) x86_avx2__zzprivate_l2 = (uint64_t)l2;
    doors->turboquant_gemv      = x86_avx2__override_turboquant_gemv;
    doors->turboquant_gemv_int8 = x86_avx2__override_turboquant_gemv_int8;
    doors->turboquant_gemv_groups     = x86_avx2__override_gemv_groups;
    doors->turboquant_gemv_groups_n   = x86_avx2__override_gemv_groups_n;
    doors->turboquant_gemv_groups_sum = x86_avx2__override_gemv_groups_sum;
    doors->turboquant_gemv_groups_int8     = x86_avx2__override_gemv_groups_int8;
    doors->turboquant_gemv_groups_sum_int8 = x86_avx2__override_gemv_groups_sum_int8;
    doors->hadamard_rotate      = x86_avx2__override_hadamard_rotate;
    doors->hadamard_ladder_rows = x86_avx2__override_hadamard_ladder_rows;
    doors->vector_zero          = x86_avx2__override_vector_zero;
    doors->vector_add           = x86_avx2__override_vector_add;
    doors->swiglu_clamped       = x86_avx2__override_swiglu_clamped;
    doors->hadamard_blocks      = x86_avx2__override_hadamard_blocks;
    doors->deltanet_step        = x86_avx2__override_deltanet_step;
    doors->deltanet_steps       = x86_avx2__override_deltanet_steps;
    doors->deltanet_steps_gate  = x86_avx2__override_deltanet_steps_gate;
    doors->deltanet_conv_step   = x86_avx2__override_deltanet_conv_step;
    doors->deltanet_conv_steps  = x86_avx2__override_deltanet_conv_steps;
    doors->hyper_logits_fp32    = x86_avx2__override_hyper_logits_fp32;
    doors->expert_rows          = x86_avx2__override_expert_rows;
    doors->expert_rows_sum      = x86_avx2__override_expert_rows_sum;
    doors->expert_groups        = x86_avx2__override_expert_groups;
    doors->expert_groups_int8   = x86_avx2__override_expert_groups_int8;
    doors->expert_rows_sum_int8 = x86_avx2__override_expert_rows_sum_int8;
    doors->attention_weights         = x86_avx2__override_attention_weights;
    doors->attention_residual_mix    = x86_avx2__override_attention_residual_mix;
    doors->attention_weights_tq      = x86_avx2__override_attention_weights_tq;
    doors->attention_residual_mix_tq = x86_avx2__override_attention_residual_mix_tq;
    doors->attention_causal_scores_grouped = x86_avx2__override_attention_causal_scores_grouped;
    doors->attention_causal_mix            = x86_avx2__override_attention_causal_mix;
    doors->attention_mix                   = x86_avx2__override_attention_mix;
    doors->vector_top_k                    = x86_avx2__override_vector_top_k;
}

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_OVERRIDES_NN_CUH */
