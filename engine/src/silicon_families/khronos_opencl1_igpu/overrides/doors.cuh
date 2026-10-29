#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_DOORS_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_DOORS_CUH
/* ══ THE OVERRIDES, INTO nn's TABLE — khronos_opencl1_igpu ════════════════════════════════════════════════════════════
 * Each door below launches its kernel from kernels.cuh by name, where the call fits it, and is nn's generic door for
 * everything else; every other door stays nn's. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../sys.cuh"           /* the cards, their devices, the bound one */

/* ⛳ AN INTEGRATED GPU OR A DISCRETE CARD — the two shapes of kernels.cuh: integrated where the device says its memory is the
 *   host's (`CL_DEVICE_HOST_UNIFIED_MEMORY`, asked once a card, whatever `SILVANN_OPENCL1_UNIFIED` says of the spans), and
 *   `SILVANN_OPENCL1_SHAPE=igpu` or `=discrete` asks for either — to check both shapes on one device. */
static inline bool khronos_opencl1_igpu__zzprivate_integrated(void) {
    static int kind[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];   /* 0 not yet asked, 1 discrete, 2 integrated */
    const char* say = getenv("SILVANN_OPENCL1_SHAPE");
    if (say && strcmp(say, "igpu") == 0) return true;
    if (say && strcmp(say, "discrete") == 0) return false;
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    const uint32_t card = khronos_opencl1_igpu__bound_device();
    if (s == 0 || card >= s->count || card >= KHRONOS_OPENCL1_IGPU__DEVICES_MAX) return false;
    if (kind[card] == 0) {
        cl_bool unified = CL_FALSE;
        kind[card] = clGetDeviceInfo(s->device[card], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof unified, &unified, 0) == CL_SUCCESS
                  && unified == CL_TRUE ? 2 : 1;
    }
    return kind[card] == 2;
}
static inline uint32_t khronos_opencl1_igpu__zzprivate_blocks(uint64_t items) {
    return items == 0u ? 1u : items < NN__KERNELS__BLOCKS_MAX ? (uint32_t)items : NN__KERNELS__BLOCKS_MAX;
}

/* KDA's steps: a head of 128, `out` and the state apart from what the steps read. */
static void khronos_opencl1_igpu__override_kda_steps_decayed(float* S, const uint16_t* conved, const float* eg, const uint16_t* b,
                                                             const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads,
                                                             uint64_t head_dim, float eps, uint64_t rows, unsigned int* over) {
    const uint64_t width = heads * head_dim;
    const uint64_t c0 = (uint64_t)(uintptr_t)conved, c1 = c0 + 3u * rows * width * 2u;
    const uint64_t e0 = (uint64_t)(uintptr_t)eg, e1 = e0 + rows * width * 4u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * width * 2u;
    const uint64_t s0 = (uint64_t)(uintptr_t)S, s1 = s0 + heads * head_dim * head_dim * 4u;
    const bool apart = (o1 <= c0 || c1 <= o0) && (s1 <= c0 || c1 <= s0) && (o1 <= e0 || e1 <= o0) && (s1 <= e0 || e1 <= s0);
    if (head_dim != 128u || heads == 0u || !apart) {
        nn__kda__zzabi_launch_steps_decayed(S, conved, eg, b, gate, w, out, heads, head_dim, eps, rows, over);
        return;
    }
    void* args[] = { &S, &conved, &eg, &b, &gate, &w, &out, &heads, &eps, &rows, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof eg, sizeof b, sizeof gate, sizeof w, sizeof out, sizeof heads, sizeof eps,
                             sizeof rows, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_kda_steps__kernel", khronos_opencl1_igpu__zzprivate_blocks(heads), 128u,
                        args, sizes, 11u);
}

/* MLA's absorb: `out` apart from the queries. */
static void khronos_opencl1_igpu__zzprivate_absorb_launch(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope,
                                                          uint64_t latent, uint64_t stride, float scale, uint64_t rows, unsigned int* over) {
    /* a discrete card a row a work item: its queries are re-read from memory for every row either way, with no block-shared
     * memory to stage them in, so more rows a work item would only make fewer of them (`MEASURED`, the bench's 37 rows on
     * the RTX 5060 Ti: 824.0 us at eight against nn's 524.2) */
    const uint64_t R = rows == 1u || !khronos_opencl1_igpu__zzprivate_integrated() ? 1u : 16u;
    const char* name = R == 1u ? "khronos_opencl1_igpu__zzprivate_absorb1__kernel" : "khronos_opencl1_igpu__zzprivate_absorb16__kernel";
    const uint64_t items = heads * ((latent + 255u) / 256u) * ((rows + R - 1u) / R);
    void* args[] = { &out, &q, &W, &heads, &nope, &latent, &stride, &scale, &rows, &over };
    const size_t sizes[] = { sizeof out, sizeof q, sizeof W, sizeof heads, sizeof nope, sizeof latent, sizeof stride, sizeof scale,
                             sizeof rows, sizeof over };
    nn__silicon__launch(0, name, khronos_opencl1_igpu__zzprivate_blocks(items), 256u, args, sizes, 10u);
}
static bool khronos_opencl1_igpu__zzprivate_absorb_fits(const uint16_t* out, const uint16_t* q, uint64_t heads, uint64_t nope,
                                                        uint64_t latent, uint64_t rows) {
    const uint64_t q0 = (uint64_t)(uintptr_t)q, q1 = q0 + rows * heads * nope * 2u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * heads * latent * 2u;
    return heads != 0u && rows != 0u && nope != 0u && latent != 0u && (o1 <= q0 || q1 <= o0);
}
static void khronos_opencl1_igpu__override_attention_absorb(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads,
                                                            uint64_t nope, uint64_t latent, uint64_t stride, float scale, unsigned int* over) {
    if (!khronos_opencl1_igpu__zzprivate_absorb_fits(out, q, heads, nope, latent, 1u)) {
        nn__attention__zzabi_launch_absorb(out, q, W, heads, nope, latent, stride, scale, over);
        return;
    }
    khronos_opencl1_igpu__zzprivate_absorb_launch(out, q, W, heads, nope, latent, stride, scale, 1u, over);
}
static void khronos_opencl1_igpu__override_attention_absorb_rows(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads,
                                                                 uint64_t nope, uint64_t latent, uint64_t stride, float scale, uint64_t rows,
                                                                 unsigned int* over) {
    if (!khronos_opencl1_igpu__zzprivate_absorb_fits(out, q, heads, nope, latent, rows)) {
        nn__attention__zzabi_launch_absorb_rows(out, q, W, heads, nope, latent, stride, scale, rows, over);
        return;
    }
    khronos_opencl1_igpu__zzprivate_absorb_launch(out, q, W, heads, nope, latent, stride, scale, rows, over);
}

/* MLA's expand: a latent of at most 512 (nn's 256 lanes two columns each at most), outputs in whole eights. */
static void khronos_opencl1_igpu__zzprivate_expand_launch(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width,
                                                          uint64_t latent, uint64_t stride, uint64_t offset, uint64_t rows, unsigned int* over) {
    const uint64_t R = khronos_opencl1_igpu__zzprivate_integrated() ? 32u : 16u;
    const uint64_t items = heads * (width / 8u) * ((rows + R - 1u) / R);
    void* args[] = { &out, &o, &W, &heads, &width, &latent, &stride, &offset, &rows, &over };
    const size_t sizes[] = { sizeof out, sizeof o, sizeof W, sizeof heads, sizeof width, sizeof latent, sizeof stride, sizeof offset,
                             sizeof rows, sizeof over };
    nn__silicon__launch(0, R == 16u ? "khronos_opencl1_igpu__zzprivate_expand16__kernel" : "khronos_opencl1_igpu__zzprivate_expand32__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks(items), 256u, args, sizes, 10u);
}
static void khronos_opencl1_igpu__override_attention_expand(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                            uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset, unsigned int* over) {
    if (latent == 0u || latent > 512u || heads == 0u || width == 0u || width % 8u != 0u) {
        nn__attention__zzabi_launch_expand(out, o, W, heads, width, latent, stride, offset, over);
        return;
    }
    khronos_opencl1_igpu__zzprivate_expand_launch(out, o, W, heads, width, latent, stride, offset, 1u, over);
}
static void khronos_opencl1_igpu__override_attention_expand_rows(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                                 uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset,
                                                                 uint64_t rows, unsigned int* over) {
    if (latent == 0u || latent > 512u || heads == 0u || width == 0u || rows == 0u || width % 8u != 0u) {
        nn__attention__zzabi_launch_expand_rows(out, o, W, heads, width, latent, stride, offset, rows, over);
        return;
    }
    khronos_opencl1_igpu__zzprivate_expand_launch(out, o, W, heads, width, latent, stride, offset, rows, over);
}

/* The indexer's pool scores: at most 64 heads, nn's lanes for them. */
static void khronos_opencl1_igpu__zzprivate_index_scores_launch(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                                const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0,
                                                                uint64_t kpool, uint64_t rows, float scale) {
    const uint64_t G = khronos_opencl1_igpu__zzprivate_integrated() ? 16u : 8u;
    const uint64_t items = rows * (((len0 + rows - 1u) / kpool + G - 1u) / G);
    void* args[] = { &scores, &scores_row, &q, &w, &pooled, &heads, &dim, &len0, &kpool, &rows, &scale };
    const size_t sizes[] = { sizeof scores, sizeof scores_row, sizeof q, sizeof w, sizeof pooled, sizeof heads, sizeof dim, sizeof len0,
                             sizeof kpool, sizeof rows, sizeof scale };
    nn__silicon__launch(0, G == 8u ? "khronos_opencl1_igpu__zzprivate_index_scores8__kernel" : "khronos_opencl1_igpu__zzprivate_index_scores16__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks(items), 64u, args, sizes, 11u);
}
static void khronos_opencl1_igpu__override_index_scores(float* scores, const uint16_t* q, const uint16_t* w, const uint16_t* pooled,
                                                        uint64_t heads, uint64_t dim, uint64_t pools, float scale) {
    /* one row's pools a block each on a discrete card, as nn's: a row's few hundred pools in blocks of eight leave it idle
     * (`MEASURED`, the bench's sparse rows a launch a row on the RTX 5060 Ti: 3019.3 us against nn's 2700.6) */
    if (pools == 0u || heads == 0u || heads > 64u || !khronos_opencl1_igpu__zzprivate_integrated()) {
        nn__index__zzabi_launch_scores(scores, q, w, pooled, heads, dim, pools, scale);
        return;
    }
    khronos_opencl1_igpu__zzprivate_index_scores_launch(scores, 0u, q, w, pooled, heads, dim, pools, 1u, 1u, scale);
}
static void khronos_opencl1_igpu__override_index_scores_rows(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                             const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0,
                                                             uint64_t kpool, uint64_t rows, float scale) {
    if (rows == 0u || kpool == 0u || (len0 + rows - 1u) / kpool == 0u || heads == 0u || heads > 64u) {
        nn__index__zzabi_launch_scores_rows(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale);
        return;
    }
    khronos_opencl1_igpu__zzprivate_index_scores_launch(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale);
}

/* The mix over rows of their own: whole groups. */
static void khronos_opencl1_igpu__override_attention_mix_rows(uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row, const uint16_t* v,
                                                              uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
                                                              uint64_t head_dim, uint64_t rows, unsigned int* over) {
    if (rows == 0u || kv_heads == 0u || q_heads % kv_heads != 0u || head_dim == 0u) {
        nn__attention__zzabi_launch_mix_rows(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, head_dim, rows, over);
        return;
    }
    const uint64_t Q = khronos_opencl1_igpu__zzprivate_integrated() ? 16u : 8u;
    const uint64_t items = rows * kv_heads * ((q_heads / kv_heads + Q - 1u) / Q) * ((head_dim + 255u) / 256u);
    void* args[] = { &o, &o_row, &p, &p_row, &v, &v_row, &lens, &q_heads, &kv_heads, &head_dim, &rows, &over };
    const size_t sizes[] = { sizeof o, sizeof o_row, sizeof p, sizeof p_row, sizeof v, sizeof v_row, sizeof lens, sizeof q_heads,
                             sizeof kv_heads, sizeof head_dim, sizeof rows, sizeof over };
    nn__silicon__launch(0, Q == 8u ? "khronos_opencl1_igpu__zzprivate_mix_rows8__kernel" : "khronos_opencl1_igpu__zzprivate_mix_rows16__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks(items), 256u, args, sizes, 12u);
}

/* Into nn's table, once, when the family is first asked for it. */
/* ⭐ THE GEMV (▶ kernels.cuh): nn's levels as floats in the bound card's memory, made once a card — the 8 bits' 256 then the 4
 *   bits' 16 — and the groups launched when every matrix fits: 4 or 8 bits, rows of whole sixteen bytes, codes and `x` on
 *   sixteen; anything else is nn's door. */
static const float* khronos_opencl1_igpu__zzprivate_gemv_levels(void) {
    static void* at[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];
    static sys__os__lock lock = SYS__OS__LOCK_INIT;
    const uint32_t card = khronos_opencl1_igpu__bound_device();
    if (card >= KHRONOS_OPENCL1_IGPU__DEVICES_MAX) return 0;
    sys__os__lock_take(&lock);
    if (at[card] == 0) {
        float v[272];
        for (uint32_t e = 0u; e < 256u; ++e) v[e] = nn__silicon__half_to_float(nn__turboquant__zzabi_levels(8u)[e]);
        for (uint32_t e = 0u; e < 16u; ++e) v[256u + e] = nn__silicon__half_to_float(nn__turboquant__zzabi_levels(4u)[e]);
        void* made = 0;
        if (khronos_opencl1_igpu__memory_allocate(&made, sizeof v) && khronos_opencl1_igpu__memory_write(made, v, sizeof v)) at[card] = made;
    }
    sys__os__lock_give(&lock);
    return (const float*)at[card];
}
#define KHRONOS_OPENCL1_IGPU__GEMV_FOUR 128u       /* sixteens of rows from which a run takes four */
#define KHRONOS_OPENCL1_IGPU__GEMV_TWO  48u        /* and two */
#define KHRONOS_OPENCL1_IGPU__GEMV_SUM_FOUR 256u   /* the weighted sum's four */
static bool khronos_opencl1_igpu__zzprivate_gemv_launch(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                        unsigned int* over) {
    if (g.count == 0u || g.count > NN__TURBOQUANT__GROUPS_MAX || ((uint64_t)(uintptr_t)x) % 16u != 0u) return false;
    uint64_t units = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) {
        const uint64_t d = g.d[i];
        if ((d != 4u && d != 8u) || (cols * d / 8u) % 16u != 0u || g.codes[i] % 16u != 0u) return false;
        units += (g.rows[i] + 15u) / 16u;
    }
    const float* levels = khronos_opencl1_igpu__zzprivate_gemv_levels();
    if (levels == 0) return false;
    if (units == 0u) return true;
    /* a run `R` rows (▶ kernels.cuh) by the matrices' sixteens of rows: four from 2,048 rows, two from 768, else one —
     * `MEASURED` on the RTX 5060 Ti, the bench at each (measurements/2026-10-28_opencl1_gemv_rows_precision.md): GLM's
     * attention output 4096 x 16384 at 4 bits 150 -> 87 us at four, 1,001 rows best at two, 517 at one. Bytes the same
     * at every `R`; `SILVANN_OPENCL1_GEMV_ROWS` names one */
    const uint64_t rule = units >= KHRONOS_OPENCL1_IGPU__GEMV_FOUR ? 4u : units >= KHRONOS_OPENCL1_IGPU__GEMV_TWO ? 2u : 1u;
    uint64_t R = rule;
    const char* say = getenv("SILVANN_OPENCL1_GEMV_ROWS");
    if (say && (say[0] == '1' || say[0] == '2' || say[0] == '4') && say[1] == 0) R = (uint64_t)(say[0] - '0');
    uint64_t blocks = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) blocks += (g.rows[i] + 16u * R - 1u) / (16u * R);
    void* args[] = { &out, &g, &x, &cols, &levels, &over };
    const size_t sizes[] = { sizeof out, sizeof g, sizeof x, sizeof cols, sizeof levels, sizeof over };
    nn__silicon__launch(0, R == 1u ? "khronos_opencl1_igpu__zzprivate_gemv__kernel" : R == 2u ? "khronos_opencl1_igpu__zzprivate_gemv2__kernel"
                                                                                           : "khronos_opencl1_igpu__zzprivate_gemv4__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks(blocks), 256u, args, sizes, 6u);
    return true;
}
static void khronos_opencl1_igpu__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                           uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                           unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (d != 16u && row_bytes != 0u && rows <= w_room / row_bytes && rows <= l_room / 2u) {
        nn__turboquant__groups g = {};
        g.codes[0] = (uint64_t)(uintptr_t)weights; g.luts[0] = (uint64_t)(uintptr_t)luts; g.rows[0] = rows; g.d[0] = d; g.out_at[0] = 0u;
        g.count = 1u;
        if (khronos_opencl1_igpu__zzprivate_gemv_launch(out, g, x, cols, over)) return;
    }
    nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over);
}
static void khronos_opencl1_igpu__override_turboquant_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                                  unsigned int* over) {
    if (khronos_opencl1_igpu__zzprivate_gemv_launch(out, g, x, cols, over)) return;
    nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over);
}

/* The groups over `n` inputs: the one input's kernel for each, so each input's answers are the one-input door's bytes. */
static void khronos_opencl1_igpu__override_turboquant_gemv_groups_n(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                                    uint64_t n, uint64_t out_stride, unsigned int* over) {
    bool fits = ((uint64_t)(uintptr_t)x) % 16u == 0u && (cols * 2u) % 16u == 0u;
    for (uint64_t i = 0u; fits && i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i)
        fits = (g.d[i] == 4u || g.d[i] == 8u) && (cols * g.d[i] / 8u) % 16u == 0u && g.codes[i] % 16u == 0u;
    if (fits && g.count != 0u && khronos_opencl1_igpu__zzprivate_gemv_levels() != 0) {
        for (uint64_t k = 0u; k < n; ++k) {
            uint16_t* o = out;
            const nn__turboquant__groups gk = nn__turboquant__groups_input(&g, &o, k, out_stride);
            (void)khronos_opencl1_igpu__zzprivate_gemv_launch(o, gk, x + k * cols, cols, over);
        }
        return;
    }
    nn__turboquant__zzabi_launch_gemv_groups_n(out, g, x, cols, n, out_stride, over);
}
/* The weighted sum onto the residual, every matrix fitting as the gemv's, `x_i` on sixteen too. */
static void khronos_opencl1_igpu__override_turboquant_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                      const uint16_t* w, const uint16_t* residual, uint64_t rows,
                                                                      uint64_t cols, unsigned int* over) {
    bool fits = g.count != 0u && g.count <= NN__TURBOQUANT__GROUPS_MAX && ((uint64_t)(uintptr_t)x) % 16u == 0u && (cols * 2u) % 16u == 0u;
    for (uint64_t i = 0u; fits && i < g.count; ++i)
        fits = (g.d[i] == 4u || g.d[i] == 8u) && (cols * g.d[i] / 8u) % 16u == 0u && g.codes[i] % 16u == 0u;
    const float* levels = fits ? khronos_opencl1_igpu__zzprivate_gemv_levels() : 0;
    if (levels == 0) {
        nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over);
        return;
    }
    /* a run `R` rows: four from 4,096 rows, two from 768, else one — a row here is every matrix's, so the blocks are the
     * rows' alone (`MEASURED` on the RTX 5060 Ti, measurements/2026-10-28_opencl1_gemv_rows_precision.md: the 35B's MoE
     * down, 2048 rows of nine matrices, 29.1 / 20.3 / 24.9 us at one / two / four rows, a 5120-row MLP down 175.6 / 154.6
     * / 142.6); `SILVANN_OPENCL1_GEMV_ROWS` names one */
    const uint64_t units = (rows + 15u) / 16u;
    uint64_t R = units >= KHRONOS_OPENCL1_IGPU__GEMV_SUM_FOUR ? 4u : units >= KHRONOS_OPENCL1_IGPU__GEMV_TWO ? 2u : 1u;
    const char* say = getenv("SILVANN_OPENCL1_GEMV_ROWS");
    if (say && (say[0] == '1' || say[0] == '2' || say[0] == '4') && say[1] == 0) R = (uint64_t)(say[0] - '0');
    void* args[] = { &out, &g, &x, &w, &residual, &rows, &cols, &levels, &over };
    const size_t sizes[] = { sizeof out, sizeof g, sizeof x, sizeof w, sizeof residual, sizeof rows, sizeof cols, sizeof levels, sizeof over };
    nn__silicon__launch(0, R == 1u ? "khronos_opencl1_igpu__zzprivate_gemv_sum__kernel" : R == 2u ? "khronos_opencl1_igpu__zzprivate_gemv_sum2__kernel"
                                                                                               : "khronos_opencl1_igpu__zzprivate_gemv_sum4__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks((rows + 16u * R - 1u) / (16u * R)), 256u,
                        args, sizes, 9u);
}

/* ⭐ THE HADAMARD ROTATION (▶ kernels.cuh): `out` apart from `in` — rotating in place is nn's door, as nn's body runs it
 *   on one lane then — and a length nn's ladder places; a block of 256 lanes a block of the ladder. */
static void khronos_opencl1_igpu__override_hadamard_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n,
                                                           unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out;
    /* a length the ladder places as nn's: whole 512s then a 256 and a 128, or one block of a power of two up to 256 */
    const bool ladder = n >= 512u ? (n % 128u == 0u) : (n >= 2u && (n & (n - 1u)) == 0u);
    if (!apart || !ladder) {
        nn__hadamard__zzabi_launch_rotate(out, in, sign, n, over);
        return;
    }
    void* args[] = { &out, &in, &sign, &n, &over };
    const size_t sizes[] = { sizeof out, sizeof in, sizeof sign, sizeof n, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_rotate__kernel", khronos_opencl1_igpu__zzprivate_blocks((n + 511u) / 512u + 2u), 256u,
                        args, sizes, 5u);
}

/* KDA's step a row: a head of 128, `out` and the state apart from the `q`, `k` and `v` the step reads. */
static void khronos_opencl1_igpu__override_kda_step(float* S, const uint16_t* conved, const uint16_t* f, const uint16_t* b, const uint16_t* dt,
                                                    const uint16_t* a_log, const uint16_t* gate, const uint16_t* w, uint16_t* out,
                                                    uint64_t heads, uint64_t head_dim, float lower, float eps, unsigned int* over) {
    const uint64_t width = heads * head_dim;
    const uint64_t c0 = (uint64_t)(uintptr_t)conved, c1 = c0 + 3u * width * 2u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + width * 2u;
    const uint64_t s0 = (uint64_t)(uintptr_t)S, s1 = s0 + heads * head_dim * head_dim * 4u;
    const bool apart = (o1 <= c0 || c1 <= o0) && (s1 <= c0 || c1 <= s0) && (s1 <= o0 || o1 <= s0);
    if (head_dim != 128u || heads == 0u || !apart) {
        nn__kda__zzabi_launch_step(S, conved, f, b, dt, a_log, gate, w, out, heads, head_dim, lower, eps, over);
        return;
    }
    void* args[] = { &S, &conved, &f, &b, &dt, &a_log, &gate, &w, &out, &heads, &lower, &eps, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof f, sizeof b, sizeof dt, sizeof a_log, sizeof gate, sizeof w, sizeof out,
                             sizeof heads, sizeof lower, sizeof eps, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_kda_step__kernel", khronos_opencl1_igpu__zzprivate_blocks(heads), 128u,
                        args, sizes, 13u);
}

/* A site's map's logits: nn's door's blocks and lanes, so a lane's stride and the fold are nn's; any count of streams. */
static void khronos_opencl1_igpu__override_hyper_logits(float* lg, const uint16_t* streams, const uint16_t* fn, uint64_t hidden, uint64_t mult,
                                                        float norm_eps) {
    const uint64_t rows = NN__HYPER__MIX(mult);
    if (rows == 0u || hidden == 0u) {
        nn__hyper__zzabi_launch_logits(lg, streams, fn, hidden, mult, norm_eps);
        return;
    }
    void* args[] = { &lg, &streams, &fn, &hidden, &mult, &norm_eps };
    const size_t sizes[] = { sizeof lg, sizeof streams, sizeof fn, sizeof hidden, sizeof mult, sizeof norm_eps };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_hyper_logits__kernel", khronos_opencl1_igpu__zzprivate_blocks(rows), 256u,
                        args, sizes, 6u);
}

/* The map's logits over streams of floats: nn's door's blocks and lanes, as the halves' above. */
static void khronos_opencl1_igpu__override_hyper_logits_fp32(float* lg, const float* streams, const uint16_t* fn, uint64_t hidden,
                                                             uint64_t mult, float norm_eps) {
    const uint64_t rows = NN__HYPER__MIX(mult);
    if (rows == 0u || hidden == 0u) {
        nn__hyper__zzabi_launch_logits_fp32(lg, streams, fn, hidden, mult, norm_eps);
        return;
    }
    void* args[] = { &lg, &streams, &fn, &hidden, &mult, &norm_eps };
    const size_t sizes[] = { sizeof lg, sizeof streams, sizeof fn, sizeof hidden, sizeof mult, sizeof norm_eps };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_hyper_logits_fp32__kernel", khronos_opencl1_igpu__zzprivate_blocks(rows), 256u,
                        args, sizes, 6u);
}

/* The norm of a row of halves: nn's door's one block of 256 lanes; a row normed in place, or apart from its output. */
static void khronos_opencl1_igpu__override_rmsnorm(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    const bool apart = o == x || o + n <= x || x + n <= o;
    if (n == 0u || !apart) {
        nn__rmsnorm__zzabi_launch(o, x, w, n, eps, over);
        return;
    }
    void* args[] = { &o, &x, &w, &n, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof eps, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_norm__kernel", 1u, 256u, args, sizes, 6u);
}

/* The norm of a row of floats: nn's door's one block of 256 lanes; a row written over its own floats is nn's door's. */
static void khronos_opencl1_igpu__override_rmsnorm_fp32(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, float eps,
                                                        unsigned int* over) {
    const uint64_t o0 = (uint64_t)(uintptr_t)o, x0 = (uint64_t)(uintptr_t)x;
    const bool apart = o0 + n * 2u <= x0 || x0 + n * 4u <= o0;
    if (n == 0u || !apart) {
        nn__rmsnorm__zzabi_launch_fp32(o, x, w, n, eps, over);
        return;
    }
    void* args[] = { &o, &x, &w, &n, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof eps, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_norm_fp32__kernel", 1u, 256u, args, sizes, 6u);
}

/* The scored router: up to four scores a lane of nn's 256, one to `NN__VECTOR__ROUTER_K_MAX` picks; anything else is nn's. */
static void khronos_opencl1_igpu__override_vector_top_k_scored(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x,
                                                               const uint16_t* bias, uint64_t n, uint64_t k, float scale, uint64_t score) {
    if (n == 0u || n > 4u * 256u || k == 0u || k > NN__VECTOR__ROUTER_K_MAX) {
        nn__vector__zzabi_launch_top_k_scored(values, stride, weights, x, bias, n, k, scale, score);
        return;
    }
    void* args[] = { &values, &stride, &weights, &x, &bias, &n, &k, &scale, &score };
    const size_t sizes[] = { sizeof values, sizeof stride, sizeof weights, sizeof x, sizeof bias, sizeof n, sizeof k, sizeof scale,
                             sizeof score };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_top_k_scored__kernel", 1u, 256u, args, sizes, 9u);
}

/* A site's `pre` at four streams, over halves and over floats; any other count is nn's. */
static void khronos_opencl1_igpu__override_hyper_pre(uint16_t* out, float* mix, const uint16_t* streams, const uint16_t* base,
                                                     const uint16_t* scale, uint64_t hidden, uint64_t mult, uint64_t iters, float eps,
                                                     unsigned int* over) {
    if (mult != 4u || hidden == 0u) {
        nn__hyper__zzabi_launch_pre(out, mix, streams, base, scale, hidden, mult, iters, eps, over);
        return;
    }
    void* args[] = { &out, &mix, &streams, &base, &scale, &hidden, &iters, &eps, &over };
    const size_t sizes[] = { sizeof out, sizeof mix, sizeof streams, sizeof base, sizeof scale, sizeof hidden, sizeof iters, sizeof eps,
                             sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_hyper_pre4__kernel", khronos_opencl1_igpu__zzprivate_blocks((hidden + 255u) / 256u),
                        256u, args, sizes, 9u);
}
static void khronos_opencl1_igpu__override_hyper_pre_fp32(float* out, float* mix, const float* streams, const uint16_t* base,
                                                          const uint16_t* scale, uint64_t hidden, uint64_t mult, uint64_t iters, float eps) {
    if (mult != 4u || hidden == 0u) {
        nn__hyper__zzabi_launch_pre_fp32(out, mix, streams, base, scale, hidden, mult, iters, eps);
        return;
    }
    void* args[] = { &out, &mix, &streams, &base, &scale, &hidden, &iters, &eps };
    const size_t sizes[] = { sizeof out, sizeof mix, sizeof streams, sizeof base, sizeof scale, sizeof hidden, sizeof iters, sizeof eps };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_hyper_pre4_fp32__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks((hidden + 255u) / 256u), 256u, args, sizes, 8u);
}

/* A one-query attention's grouped scores: the widest tile of one, two or four positions that still makes `ITEMS` items,
 *   and nn's door, whose tiles are eight, where eight do; nn's `dots` reading halves two to a word — an even width, `q`
 *   and `k` on four bytes — and the heads a whole number of groups, or nn's door. */
#define KHRONOS_OPENCL1_IGPU__SCORES_ITEMS 8192u    /* items enough to fill a card: 32 blocks of 256 */
static void khronos_opencl1_igpu__override_attention_scores_grouped(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                    uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    uint64_t T = 1u;
    while (T < 8u && q_heads * ((length + 2u * T - 1u) / (2u * T)) >= KHRONOS_OPENCL1_IGPU__SCORES_ITEMS) T *= 2u;
    if (T == 8u || kv_heads == 0u || q_heads % kv_heads != 0u || head_dim % 2u != 0u || ((uint64_t)(uintptr_t)q) % 4u != 0u
     || ((uint64_t)(uintptr_t)k) % 4u != 0u || q_heads * length == 0u) {
        nn__attention__zzabi_launch_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    const char* name = T == 1u ? "khronos_opencl1_igpu__zzprivate_scores1__kernel"
                     : T == 2u ? "khronos_opencl1_igpu__zzprivate_scores2__kernel" : "khronos_opencl1_igpu__zzprivate_scores4__kernel";
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length };
    nn__silicon__launch(0, name, khronos_opencl1_igpu__zzprivate_blocks((q_heads * ((length + T - 1u) / T) + 255u) / 256u), 256u,
                        args, sizes, 7u);
}

/* ⭐ THE PROMPT'S GEMM — `expert_rows` through the runs of 32 (▶ kernels.cuh), 4- and 8-bit codes two at a time as nn's
 *   pairs are — a width of whole containers, `x` on four bytes, the codes on two — and `expert_groups` the same a group at
 *   a time, which is nn's body's order; a group or a call outside that is nn's. */
static inline bool khronos_opencl1_igpu__zzprivate_gemm_fits(const uint8_t* weights, uint64_t w_room, uint64_t l_room, const uint16_t* x,
                                                             uint64_t d, uint64_t cols, uint64_t count, uint64_t out_rows) {
    return (d == 4u || d == 8u) && cols != 0u && cols % (16u / d) == 0u && ((uint64_t)(uintptr_t)x) % 4u == 0u
        && ((uint64_t)(uintptr_t)weights) % 2u == 0u && w_room >= nn__turboquant__row_bytes(d, cols) && l_room >= 2u
        && count != 0u && out_rows != 0u;
}
static void khronos_opencl1_igpu__zzprivate_gemm_launch(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                        uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d,
                                                        uint64_t out_rows, uint64_t cols, unsigned int* over) {
    const float* table = khronos_opencl1_igpu__zzprivate_gemv_levels();
    if (table == 0) {
        nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over);
        return;
    }
    const uint64_t items = ((out_rows + 1u) / 2u) * ((count + 3u) / 4u);   /* tiles of 2 rows by 4 pairs (▶ kernels.cuh) */
    void* args[] = { &out, &weights, &w_room, &luts, &l_room, &x, &rows, &count, &out_rows, &cols, &table, &over };
    const size_t sizes[] = { sizeof out, sizeof weights, sizeof w_room, sizeof luts, sizeof l_room, sizeof x, sizeof rows, sizeof count,
                             sizeof out_rows, sizeof cols, sizeof table, sizeof over };
    nn__silicon__launch(0, d == 4u ? "khronos_opencl1_igpu__zzprivate_gemm4__kernel" : "khronos_opencl1_igpu__zzprivate_gemm8__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks((items + 7u) / 8u), 256u, args, sizes, 12u);
}
static void khronos_opencl1_igpu__override_expert_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                                       const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d, uint64_t out_rows,
                                                       uint64_t cols, unsigned int* over) {
    if (!khronos_opencl1_igpu__zzprivate_gemm_fits(weights, w_room, l_room, x, d, cols, count, out_rows)) {
        nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over);
        return;
    }
    khronos_opencl1_igpu__zzprivate_gemm_launch(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over);
}
static void khronos_opencl1_igpu__override_expert_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                         uint64_t cols, unsigned int* over) {
    for (uint64_t i = 0u; i < g.count && i < NN__EXPERT__GROUPS_MAX; ++i) {
        const uint64_t row_bytes = nn__turboquant__row_bytes(g.d[i], cols);
        khronos_opencl1_igpu__override_expert_rows(out + g.out_at[i], (const uint8_t*)(uintptr_t)g.codes[i], g.out_rows[i] * row_bytes,
                                                   (const uint8_t*)(uintptr_t)g.luts[i], g.out_rows[i] * 2u, x, rows + 2u * g.pairs_at[i],
                                                   g.pairs[i], g.d[i], g.out_rows[i], cols, over);
    }
}

/* ⭐ A RANGE'S RESIDUAL, ONE QUERY (▶ kernels.cuh): the halves' scores the grouped ones above, a TurboQuant tier's an item a
 *   (head, position) — 4 or 8 bits, a head of whole containers on two bytes — then nn's softmax first half a block a head;
 *   each mix an item a (head, element). Anything else is nn's. */
static inline bool khronos_opencl1_igpu__zzprivate_residual_fits(uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d,
                                                                 const uint8_t* codes) {
    return kv_heads != 0u && q_heads % kv_heads == 0u && head_dim != 0u && length != 0u
        && (d == 0u || ((d == 4u || d == 8u) && head_dim % (16u / d) == 0u && ((uint64_t)(uintptr_t)codes) % 2u == 0u))
        && khronos_opencl1_igpu__zzprivate_gemv_levels() != 0;
}
static void khronos_opencl1_igpu__zzprivate_residual_head(float* p, float* res, uint64_t q_heads, uint64_t head_dim, uint64_t length) {
    void* args[] = { &p, &res, &q_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof p, sizeof res, sizeof q_heads, sizeof head_dim, sizeof length };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_residual_weights__kernel", khronos_opencl1_igpu__zzprivate_blocks(q_heads), 256u,
                        args, sizes, 5u);
}
static void khronos_opencl1_igpu__override_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                             uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!khronos_opencl1_igpu__zzprivate_residual_fits(q_heads, kv_heads, head_dim, length, 0u, 0)) {
        nn__attention__zzabi_launch_weights(p, res, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    khronos_opencl1_igpu__override_attention_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, length);
    khronos_opencl1_igpu__zzprivate_residual_head(p, res, q_heads, head_dim, length);
}
static void khronos_opencl1_igpu__override_attention_weights_tq(float* p, float* res, const uint16_t* q, const uint8_t* codes, const uint16_t* scales,
                                                                uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d) {
    if (d == 0u || !khronos_opencl1_igpu__zzprivate_residual_fits(q_heads, kv_heads, head_dim, length, d, codes)) {
        nn__attention__zzabi_launch_weights_tq(p, res, q, codes, scales, q_heads, kv_heads, head_dim, length, d);
        return;
    }
    const float* table = khronos_opencl1_igpu__zzprivate_gemv_levels();
    void* args[] = { &p, &q, &codes, &scales, &q_heads, &kv_heads, &head_dim, &length, &d, &table };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length,
                             sizeof d, sizeof table };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_scores_tq__kernel", khronos_opencl1_igpu__zzprivate_blocks((q_heads * length + 255u) / 256u),
                        256u, args, sizes, 10u);
    khronos_opencl1_igpu__zzprivate_residual_head(p, res, q_heads, head_dim, length);
}
static void khronos_opencl1_igpu__override_attention_residual_mix(float* res, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                                                  uint64_t head_dim, uint64_t length) {
    if (!khronos_opencl1_igpu__zzprivate_residual_fits(q_heads, kv_heads, head_dim, length, 0u, 0)) {
        nn__attention__zzabi_launch_residual_mix(res, p, v, q_heads, kv_heads, head_dim, length);
        return;
    }
    void* args[] = { &res, &p, &v, &q_heads, &kv_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_residual_mix16__kernel", khronos_opencl1_igpu__zzprivate_blocks((q_heads * head_dim + 255u) / 256u),
                        256u, args, sizes, 7u);
}
static void khronos_opencl1_igpu__override_attention_residual_mix_tq(float* res, const float* p, const uint8_t* codes, const uint16_t* scales,
                                                                     uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                     uint64_t d) {
    if (d == 0u || !khronos_opencl1_igpu__zzprivate_residual_fits(q_heads, kv_heads, head_dim, length, d, codes)) {
        nn__attention__zzabi_launch_residual_mix_tq(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d);
        return;
    }
    const float* table = khronos_opencl1_igpu__zzprivate_gemv_levels();
    void* args[] = { &res, &p, &codes, &scales, &q_heads, &kv_heads, &head_dim, &length, &d, &table };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length,
                             sizeof d, sizeof table };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_residual_mix_tq__kernel", khronos_opencl1_igpu__zzprivate_blocks((q_heads * head_dim + 255u) / 256u),
                        256u, args, sizes, 10u);
}

/* ⭐ DECODE'S ONE QUERY — `attention_scores` as the grouped scores above, then nn's own softmax a row a head (the one-call
 *   door's second half, the same arithmetic), and `attention_mix` an item a (head, element), nn's order. */
static void khronos_opencl1_igpu__override_attention_scores(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads,
                                                            uint64_t head_dim, uint64_t length) {
    if (kv_heads == 0u || q_heads % kv_heads != 0u || length == 0u) {
        nn__attention__zzabi_launch_scores(p, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    khronos_opencl1_igpu__override_attention_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, length);
    nn__attention__zzabi_launch_softmax_rows(p, q_heads, length);
}
static void khronos_opencl1_igpu__override_attention_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                                         uint64_t head_dim, uint64_t length, unsigned int* over) {
    if (kv_heads == 0u || q_heads % kv_heads != 0u || length == 0u || q_heads * head_dim == 0u) {
        nn__attention__zzabi_launch_mix(o, p, v, q_heads, kv_heads, head_dim, length, over);
        return;
    }
    void* args[] = { &o, &p, &v, &q_heads, &kv_heads, &head_dim, &length, &over };
    const size_t sizes[] = { sizeof o, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_mix__kernel",
                        khronos_opencl1_igpu__zzprivate_blocks((q_heads * head_dim + 255u) / 256u), 256u, args, sizes, 8u);
}

/* A sampler's top k: one to `NN__VECTOR__TOP_K_MAX` picks over more values than a router's, every index exact as a float
 *   (nn's folds take them so); anything else is nn's. */
#define KHRONOS_OPENCL1_IGPU__TOP_K_SHORT 1024u     /* up to this many values, nn's rounds: a router's few */
static void khronos_opencl1_igpu__override_vector_top_k(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, uint64_t n,
                                                        uint64_t k) {
    if (n <= KHRONOS_OPENCL1_IGPU__TOP_K_SHORT || n > NN__KERNELS__INDEX_EXACT || k == 0u || k > NN__VECTOR__TOP_K_MAX) {
        nn__vector__zzabi_launch_top_k(values, stride, weights, x, n, k);
        return;
    }
    void* args[] = { &values, &stride, &weights, &x, &n, &k };
    const size_t sizes[] = { sizeof values, sizeof stride, sizeof weights, sizeof x, sizeof n, sizeof k };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_top_k__kernel", 1u, 256u, args, sizes, 6u);
}

static inline void khronos_opencl1_igpu__override_doors(nn__doors* doors) {
    doors->kda_step              = khronos_opencl1_igpu__override_kda_step;
    doors->hyper_logits          = khronos_opencl1_igpu__override_hyper_logits;
    doors->hyper_pre             = khronos_opencl1_igpu__override_hyper_pre;
    doors->hyper_pre_fp32        = khronos_opencl1_igpu__override_hyper_pre_fp32;
    doors->hyper_logits_fp32     = khronos_opencl1_igpu__override_hyper_logits_fp32;
    doors->rmsnorm_fp32          = khronos_opencl1_igpu__override_rmsnorm_fp32;
    doors->rmsnorm               = khronos_opencl1_igpu__override_rmsnorm;
    doors->vector_top_k_scored   = khronos_opencl1_igpu__override_vector_top_k_scored;
    doors->vector_top_k          = khronos_opencl1_igpu__override_vector_top_k;
    doors->attention_scores_grouped = khronos_opencl1_igpu__override_attention_scores_grouped;
    doors->attention_scores         = khronos_opencl1_igpu__override_attention_scores;
    doors->attention_mix            = khronos_opencl1_igpu__override_attention_mix;
    doors->expert_rows              = khronos_opencl1_igpu__override_expert_rows;
    doors->expert_groups            = khronos_opencl1_igpu__override_expert_groups;
    doors->attention_weights         = khronos_opencl1_igpu__override_attention_weights;
    doors->attention_weights_tq      = khronos_opencl1_igpu__override_attention_weights_tq;
    doors->attention_residual_mix    = khronos_opencl1_igpu__override_attention_residual_mix;
    doors->attention_residual_mix_tq = khronos_opencl1_igpu__override_attention_residual_mix_tq;
    doors->hadamard_rotate            = khronos_opencl1_igpu__override_hadamard_rotate;
    doors->turboquant_gemv_groups_n   = khronos_opencl1_igpu__override_turboquant_gemv_groups_n;
    doors->turboquant_gemv_groups_sum = khronos_opencl1_igpu__override_turboquant_gemv_groups_sum;
    doors->turboquant_gemv        = khronos_opencl1_igpu__override_turboquant_gemv;
    doors->turboquant_gemv_groups = khronos_opencl1_igpu__override_turboquant_gemv_groups;
    doors->kda_steps_decayed     = khronos_opencl1_igpu__override_kda_steps_decayed;
    doors->attention_absorb      = khronos_opencl1_igpu__override_attention_absorb;
    doors->attention_absorb_rows = khronos_opencl1_igpu__override_attention_absorb_rows;
    doors->attention_expand      = khronos_opencl1_igpu__override_attention_expand;
    doors->attention_expand_rows = khronos_opencl1_igpu__override_attention_expand_rows;
    doors->index_scores          = khronos_opencl1_igpu__override_index_scores;
    doors->index_scores_rows     = khronos_opencl1_igpu__override_index_scores_rows;
    doors->attention_mix_rows    = khronos_opencl1_igpu__override_attention_mix_rows;
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_DOORS_CUH */
