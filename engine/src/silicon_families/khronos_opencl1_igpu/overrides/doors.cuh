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
    void* args[] = { &out, &g, &x, &cols, &levels, &over };
    const size_t sizes[] = { sizeof out, sizeof g, sizeof x, sizeof cols, sizeof levels, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_gemv__kernel", khronos_opencl1_igpu__zzprivate_blocks(units), 256u,
                        args, sizes, 6u);
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
    void* args[] = { &out, &g, &x, &w, &residual, &rows, &cols, &levels, &over };
    const size_t sizes[] = { sizeof out, sizeof g, sizeof x, sizeof w, sizeof residual, sizeof rows, sizeof cols, sizeof levels, sizeof over };
    nn__silicon__launch(0, "khronos_opencl1_igpu__zzprivate_gemv_sum__kernel", khronos_opencl1_igpu__zzprivate_blocks((rows + 15u) / 16u), 256u,
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

static inline void khronos_opencl1_igpu__override_doors(nn__doors* doors) {
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
