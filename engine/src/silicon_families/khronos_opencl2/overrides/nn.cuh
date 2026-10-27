#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_NN_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include <stdlib.h>                          /* getenv, for the shape asked for */
#include <string.h>                          /* memset, for a tile job built on the stack; strcmp */
/* ══ ⭐⭐ OVERRIDES — THE DOORS THIS FAMILY RUNS FASTER THAN THE GENERIC BODY ══════════════════════════
 * The host side of `kernels.cuh` beside this: each function has its door's signature, takes its place in nn's table
 * (`khronos_opencl2__override_doors`, called from `../../entry.cuh` once the table is built) and launches its kernel by
 * name. ⛳ EVERY ONE FALLS BACK TO nn's GENERIC DOOR for what it does not cover — another width, columns that are not
 * whole blocks of 32 or wider than a work-group holds, codes that are not 8-byte aligned — so an override can only ever
 * be faster, never different in what it accepts, and deleting this folder loses nothing but speed. ▶ the ROCm family's
 * own, which these follow in shape. */
#define KHRONOS_OPENCL2__ZZPRIVATE_GROUP      256u     /* lanes a work-group: four sub-groups of 64 on the MI50 */
#define KHRONOS_OPENCL2__ZZPRIVATE_ROWS       4u       /* rows a work-group takes at a time, a sub-group each, where they are 64 */
#define KHRONOS_OPENCL2__ZZPRIVATE_HOST_COLS  8192u    /* the kernels' `KHRONOS_OPENCL2__ZZPRIVATE_COLS_MAX` */

/* ⭐ THE WORK-GROUPS THE DEVICE RUNS AT ONCE — its compute units, a few each. A work-group stages `x` in local memory
 *   before its first row, so a grid larger than the device holds paid that again for every group that followed, for a
 *   row or two each (▶ the ROCm family's `zzpackage_resident`, where it measured 158 -> 73 us). Kept per thread. */
/* `MEASURED` on the MI50 (test/src_gemv_27b_bench.cpp, 1/2/3/4/8 a unit): three is the best for every 27B shape — the
 * out-projection 114/88/84/97/97 us. `ASSUMED` for other devices, an Intel iGPU's units among them: what would change it
 * is a device whose work-groups a unit differ, and a run of the bench there settles it. */
#define KHRONOS_OPENCL2__ZZPRIVATE_EACH  3u
static inline uint32_t khronos_opencl2__zzprivate_resident(void) {
    static __thread uint32_t seen_device = 0xFFFFFFFFu, seen = 0u;
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    const uint32_t device = khronos_opencl2__bound_device();
    if (s == 0 || device >= s->count) return NN__KERNELS__BLOCKS_MAX;
    if (device != seen_device) {
        cl_uint units = 0u;
        if (clGetDeviceInfo(s->device[device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof units, &units, 0) != CL_SUCCESS || units == 0u) units = 1u;
        seen = (uint32_t)units * KHRONOS_OPENCL2__ZZPRIVATE_EACH; seen_device = device;
    }
    return seen;
}
/* As many work-groups as `rows` needs at four a group, up to what the device runs at once. */
static inline uint32_t khronos_opencl2__zzprivate_groups_for(uint64_t rows) {
    const uint64_t want = (rows + KHRONOS_OPENCL2__ZZPRIVATE_ROWS - 1u) / KHRONOS_OPENCL2__ZZPRIVATE_ROWS;
    const uint32_t most = khronos_opencl2__zzprivate_resident();
    return want == 0u ? 1u : want < most ? (uint32_t)want : most;
}

/* A width these kernels read: 4 or 8 bits, whole blocks of 32 columns no wider than a work-group holds. */
static inline bool khronos_opencl2__zzprivate_cols_fit(uint64_t d, uint64_t cols) {
    return (d == 4u || d == 8u) && cols != 0u && cols % 32u == 0u && cols <= KHRONOS_OPENCL2__ZZPRIVATE_HOST_COLS;
}

/* ⭐ THE GEMV at 4 and 8 bits; anything else is the generic door. */
static void khronos_opencl2__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room,
                                                      const uint8_t* luts, uint64_t l_room, const uint16_t* x,
                                                      uint64_t d, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__gemv")
     || !khronos_opencl2__zzprivate_cols_fit(d, cols) || ((uint64_t)(uintptr_t)weights) % 8u != 0u
     || rows > w_room / (cols * d / 8u) || rows > l_room / 2u) {
        nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over);
        return;
    }
    void* args[] = { &out, &weights, &luts, &x, &d, &rows, &cols, &over };
    const size_t sizes[] = { sizeof out, sizeof weights, sizeof luts, sizeof x, sizeof d, sizeof rows, sizeof cols, sizeof over };
    nn__silicon__launch(0, "khronos_opencl2__gemv", khronos_opencl2__zzprivate_groups_for(rows), KHRONOS_OPENCL2__ZZPRIVATE_GROUP,
                        args, sizes, 8u);
}

/* Every group one these kernels read: 4 or 8 bits, 8-byte aligned codes, some rows. */
static inline bool khronos_opencl2__zzprivate_groups_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX) return false;
    for (uint64_t i = 0u; i < g->count; ++i)
        if (!khronos_opencl2__zzprivate_cols_fit(g->d[i], cols) || g->codes[i] % 8u != 0u || g->rows[i] == 0u) return false;
    return true;
}

/* ⭐ THE GROUPED GEMV; anything else is the generic door. */
static void khronos_opencl2__override_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                  unsigned int* over) {
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__gemv_groups") || !khronos_opencl2__zzprivate_groups_fit(&g, cols)) {
        nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over);
        return;
    }
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    void* args[] = { &out, &g, &x, &cols, &over };
    const size_t sizes[] = { sizeof out, sizeof g, sizeof x, sizeof cols, sizeof over };
    nn__silicon__launch(0, "khronos_opencl2__gemv_groups", khronos_opencl2__zzprivate_groups_for(total), KHRONOS_OPENCL2__ZZPRIVATE_GROUP,
                        args, sizes, 5u);
}

/* ⭐ THE GROUPED GEMV ONTO A RESIDUAL; anything else is the generic door. Its kernel reads each group's `x` from global
 *   memory and holds none of it, so no width is too wide for it — the 27B's down projection is 17,408 columns, which the
 *   work-group's cap sent to the generic door. */
static inline bool khronos_opencl2__zzprivate_sum_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i)
        if ((g->d[i] != 4u && g->d[i] != 8u) || g->codes[i] % 8u != 0u || g->rows[i] == 0u) return false;
    return true;
}
static void khronos_opencl2__override_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                      const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__gemv_groups_sum") || !khronos_opencl2__zzprivate_sum_fit(&g, cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over);
        return;
    }
    void* args[] = { &out, &g, &x, &w, &residual, &rows, &cols, &over };
    const size_t sizes[] = { sizeof out, sizeof g, sizeof x, sizeof w, sizeof residual, sizeof rows, sizeof cols, sizeof over };
    nn__silicon__launch(0, "khronos_opencl2__gemv_groups_sum", khronos_opencl2__zzprivate_groups_for(rows), KHRONOS_OPENCL2__ZZPRIVATE_GROUP,
                        args, sizes, 8u);
}

/* ⭐ THE ROTATION of whole 512-blocks, `out` apart from `in`; anything else is the generic door. */
static void khronos_opencl2__override_hadamard_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n,
                                                      unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out;
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__rotate") || n == 0u || n % 512u != 0u || !apart) { nn__hadamard__zzabi_launch_rotate(out, in, sign, n, over); return; }
    uint64_t blocks = n / 512u;
    const uint32_t groups = blocks < NN__KERNELS__BLOCKS_MAX ? (uint32_t)blocks : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &out, &in, &sign, &blocks, &over };
    const size_t sizes[] = { sizeof out, sizeof in, sizeof sign, sizeof blocks, sizeof over };
    nn__silicon__launch(0, "khronos_opencl2__rotate", groups, KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 5u);
}

/* ⭐ ATTENTION'S RESIDUAL PASSES; anything a kernel cannot take is the generic door. */
static void khronos_opencl2__override_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                        uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__attention_weights") || kv_heads == 0u || q_heads % kv_heads != 0u
     || head_dim == 0u) {
        nn__attention__zzabi_launch_weights(p, res, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    void* args[] = { &p, &res, &q, &k, &q_heads, &kv_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof p, sizeof res, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length };
    const uint32_t groups = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    nn__silicon__launch(0, "khronos_opencl2__attention_weights", groups, KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 8u);
}
static void khronos_opencl2__override_attention_residual_mix(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
                                                             uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__attention_mix") || kv_heads == 0u || q_heads % kv_heads != 0u
     || head_dim == 0u) {
        nn__attention__zzabi_launch_residual_mix(res, p, v, q_heads, kv_heads, head_dim, length);
        return;
    }
    void* args[] = { &res, &p, &v, &q_heads, &kv_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length };
    const uint32_t groups = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    nn__silicon__launch(0, "khronos_opencl2__attention_mix", groups, KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 7u);
}

/* ⭐ A PROMPT'S ROWS AGAINST A MATRIX, IN TILES (`kernels.cuh`). Each matrix is checked as the generic body would — 4 or 8
 *   bits, whole slices of 32 columns, the rows its rooms hold — and the shape is the one its pairs want; false sends the
 *   caller to nn's generic door. */
static inline bool khronos_opencl2__zzprivate_tiles(nn__expert__groups* g, uint16_t* out, float* acc, const uint16_t* x,
                                                    const uint32_t* rows, const uint16_t* w, uint64_t cols, unsigned int* over) {
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__tile_wide") || !khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__tile_few")
     || g->count == 0u || g->count > NN__EXPERT__GROUPS_MAX || cols == 0u || cols % 32u != 0u)
        return false;
    uint64_t most = 0u;
    for (uint64_t i = 0u; i < g->count; ++i) {
        if ((g->d[i] != 4u && g->d[i] != 8u) || g->out_rows[i] == 0u || g->codes[i] % 2u != 0u) return false;
        if (g->pairs[i] > most) most = g->pairs[i];
    }
    const bool few = most <= 16u;
    const uint64_t TP = few ? 16u : 64u;
    uint64_t tiles = 0u;
    for (uint64_t i = 0u; i < g->count; ++i) tiles += ((g->out_rows[i] + 63u) / 64u) * ((g->pairs[i] + TP - 1u) / TP);
    if (tiles == 0u) return true;
    const uint32_t groups = tiles < NN__KERNELS__BLOCKS_MAX ? (uint32_t)tiles : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { g, &out, &acc, &x, &rows, &w, &cols, &over };
    const size_t sizes[] = { sizeof *g, sizeof out, sizeof acc, sizeof x, sizeof rows, sizeof w, sizeof cols, sizeof over };
    nn__silicon__launch(0, few ? "khronos_opencl2__tile_few" : "khronos_opencl2__tile_wide", groups, KHRONOS_OPENCL2__ZZPRIVATE_GROUP,
                        args, sizes, 8u);
    return true;
}
/* One matrix's call as a tile job, its rooms checked first. */
static inline bool khronos_opencl2__zzprivate_tiles_one(uint16_t* out, float* acc, const uint8_t* weights, uint64_t w_room,
                                                        const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows,
                                                        const uint16_t* w, uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols,
                                                        unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (count == 0u || row_bytes == 0u || out_rows > w_room / row_bytes || out_rows > l_room / 2u) return false;
    nn__expert__groups g;
    memset(&g, 0, sizeof g);
    g.codes[0] = (uint64_t)(uintptr_t)weights; g.luts[0] = (uint64_t)(uintptr_t)luts; g.out_rows[0] = out_rows; g.d[0] = d;
    g.pairs[0] = count; g.count = 1u;
    return khronos_opencl2__zzprivate_tiles(&g, out, acc, x, rows, w, cols, over);
}
static void khronos_opencl2__override_expert_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                  uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d,
                                                  uint64_t out_rows, uint64_t cols, unsigned int* over) {
    if (khronos_opencl2__zzprivate_tiles_one(out, 0, weights, w_room, luts, l_room, x, rows, 0, count, d, out_rows, cols, over)) return;
    nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over);
}
static void khronos_opencl2__override_expert_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                      uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                      uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols) {
    if (khronos_opencl2__zzprivate_tiles_one(0, acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols, 0)) return;
    nn__expert__zzabi_launch_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols);
}
static void khronos_opencl2__override_expert_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                    uint64_t cols, unsigned int* over) {
    if (khronos_opencl2__zzprivate_tiles(&g, out, 0, x, rows, 0, cols, over)) return;
    nn__expert__zzabi_launch_groups(out, g, x, rows, cols, over);
}

/* ⭐ A SAMPLER'S TOP K over more than `TOP_K_SHORT` values, one to `NN__VECTOR__TOP_K_MAX` picks, each its own word
 *   (`stride` not 0); anything else — a router's few among them — is the generic door. */
#define KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_SHORT 1024u
static void khronos_opencl2__override_vector_top_k(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, uint64_t n,
                                                   uint64_t k) {
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__top_k_long") || n <= KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_SHORT
     || n > NN__KERNELS__INDEX_EXACT || k == 0u || k > NN__VECTOR__TOP_K_MAX || stride == 0u) {
        nn__vector__zzabi_launch_top_k(values, stride, weights, x, n, k);
        return;
    }
    void* args[] = { &values, &stride, &weights, &x, &n, &k };
    const size_t sizes[] = { sizeof values, sizeof stride, sizeof weights, sizeof x, sizeof n, sizeof k };
    nn__silicon__launch(0, "khronos_opencl2__top_k_long", 1u, KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 6u);
}

/* ══ ⭐ A PROMPT'S ROWS THROUGH A LATENT'S ATTENTION AND KDA — `kernels.cuh`'s last section, nn's outputs to the bit ════════
 * ⛳ AN INTEGRATED GPU OR A DISCRETE CARD — the two shapes there: integrated where the device says its memory is the host's
 *   (`CL_DEVICE_HOST_UNIFIED_MEMORY`, asked once a card), and `SILVANN_OPENCL2_SHAPE=igpu` or `=discrete` asks for either —
 *   to check both shapes on one device. */
static inline bool khronos_opencl2__zzprivate_integrated(void) {
    static int kind[KHRONOS_OPENCL2__DEVICES_MAX];   /* 0 not yet asked, 1 discrete, 2 integrated */
    const char* say = getenv("SILVANN_OPENCL2_SHAPE");
    if (say && strcmp(say, "igpu") == 0) return true;
    if (say && strcmp(say, "discrete") == 0) return false;
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    const uint32_t card = khronos_opencl2__bound_device();
    if (s == 0 || card >= s->count) return false;
    if (kind[card] == 0) {
        cl_bool unified = CL_FALSE;
        kind[card] = clGetDeviceInfo(s->device[card], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof unified, &unified, 0) == CL_SUCCESS
                  && unified == CL_TRUE ? 2 : 1;
    }
    return kind[card] == 2;
}
static inline uint32_t khronos_opencl2__zzprivate_blocks(uint64_t items) {
    return items == 0u ? 1u : items < NN__KERNELS__BLOCKS_MAX ? (uint32_t)items : NN__KERNELS__BLOCKS_MAX;
}

/* KDA's steps: a head of 128, `out` and the state apart from what the steps read. */
static void khronos_opencl2__override_kda_steps_decayed(float* S, const uint16_t* conved, const float* eg, const uint16_t* b,
                                                        const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads,
                                                        uint64_t head_dim, float eps, uint64_t rows, unsigned int* over) {
    const uint64_t width = heads * head_dim;
    const uint64_t c0 = (uint64_t)(uintptr_t)conved, c1 = c0 + 3u * rows * width * 2u;
    const uint64_t e0 = (uint64_t)(uintptr_t)eg, e1 = e0 + rows * width * 4u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * width * 2u;
    const uint64_t s0 = (uint64_t)(uintptr_t)S, s1 = s0 + heads * head_dim * head_dim * 4u;
    const bool apart = (o1 <= c0 || c1 <= o0) && (s1 <= c0 || c1 <= s0) && (o1 <= e0 || e1 <= o0) && (s1 <= e0 || e1 <= s0);
    if (!khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__kda_steps") || head_dim != 128u || heads == 0u || !apart) {
        nn__kda__zzabi_launch_steps_decayed(S, conved, eg, b, gate, w, out, heads, head_dim, eps, rows, over);
        return;
    }
    void* args[] = { &S, &conved, &eg, &b, &gate, &w, &out, &heads, &eps, &rows, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof eg, sizeof b, sizeof gate, sizeof w, sizeof out, sizeof heads, sizeof eps,
                             sizeof rows, sizeof over };
    nn__silicon__launch(0, "khronos_opencl2__kda_steps", khronos_opencl2__zzprivate_blocks(heads), 128u, args, sizes, 11u);
}

/* MLA's absorb: `out` apart from the queries, and the rows' queries of a head within what a work-group stages. One row is
 * its own kernel; rows take 8 a work-group on a discrete card and 16 on an integrated GPU. */
#define KHRONOS_OPENCL2__ZZPRIVATE_HOST_ABSORB_STAGE 4096u   /* the kernels' `KHRONOS_OPENCL2__ZZPRIVATE_ABSORB_STAGE` */
static bool khronos_opencl2__zzprivate_absorb_launch(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope,
                                              uint64_t latent, uint64_t stride, float scale, uint64_t rows, unsigned int* over) {
    const uint64_t R = rows == 1u ? 1u : khronos_opencl2__zzprivate_integrated() ? 16u : 8u;
    const char* name = R == 1u ? "khronos_opencl2__absorb1" : R == 8u ? "khronos_opencl2__absorb8" : "khronos_opencl2__absorb16";
    const uint64_t q0 = (uint64_t)(uintptr_t)q, q1 = q0 + rows * heads * nope * 2u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * heads * latent * 2u;
    if (heads == 0u || rows == 0u || nope == 0u || latent == 0u || R * nope > KHRONOS_OPENCL2__ZZPRIVATE_HOST_ABSORB_STAGE
     || !(o1 <= q0 || q1 <= o0) || !khronos_opencl2__zzpackage_has_kernel(name))
        return false;
    const uint64_t items = heads * ((latent + KHRONOS_OPENCL2__ZZPRIVATE_GROUP - 1u) / KHRONOS_OPENCL2__ZZPRIVATE_GROUP) * ((rows + R - 1u) / R);
    void* args[] = { &out, &q, &W, &heads, &nope, &latent, &stride, &scale, &rows, &over };
    const size_t sizes[] = { sizeof out, sizeof q, sizeof W, sizeof heads, sizeof nope, sizeof latent, sizeof stride, sizeof scale,
                             sizeof rows, sizeof over };
    nn__silicon__launch(0, name, khronos_opencl2__zzprivate_blocks(items), KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 10u);
    return true;
}
static void khronos_opencl2__override_attention_absorb(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope,
                                                       uint64_t latent, uint64_t stride, float scale, unsigned int* over) {
    if (khronos_opencl2__zzprivate_absorb_launch(out, q, W, heads, nope, latent, stride, scale, 1u, over)) return;
    nn__attention__zzabi_launch_absorb(out, q, W, heads, nope, latent, stride, scale, over);
}
static void khronos_opencl2__override_attention_absorb_rows(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads,
                                                            uint64_t nope, uint64_t latent, uint64_t stride, float scale, uint64_t rows,
                                                            unsigned int* over) {
    if (khronos_opencl2__zzprivate_absorb_launch(out, q, W, heads, nope, latent, stride, scale, rows, over)) return;
    nn__attention__zzabi_launch_absorb_rows(out, q, W, heads, nope, latent, stride, scale, rows, over);
}

/* MLA's expand: a latent of at most 512 (nn's 256 lanes two columns each at most), outputs in whole eights, and more than one
 * row — one row's eight outputs share no loads, and a work-group making eight is only an eighth as many of them (`MEASURED`,
 * the bench's 37 rows a launch a row on the MI50: 3992.5 us against nn's 3525.7); 16 rows a work-group on a discrete card,
 * 32 on an integrated GPU. */
static bool khronos_opencl2__zzprivate_expand_launch(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width,
                                              uint64_t latent, uint64_t stride, uint64_t offset, uint64_t rows, unsigned int* over) {
    const uint64_t R = khronos_opencl2__zzprivate_integrated() ? 32u : 16u;
    const char* name = R == 16u ? "khronos_opencl2__expand16" : "khronos_opencl2__expand32";
    if (latent == 0u || latent > 2u * KHRONOS_OPENCL2__ZZPRIVATE_GROUP || heads == 0u || width == 0u || rows < 2u || width % 8u != 0u
     || !khronos_opencl2__zzpackage_has_kernel(name))
        return false;
    const uint64_t items = heads * (width / 8u) * ((rows + R - 1u) / R);
    void* args[] = { &out, &o, &W, &heads, &width, &latent, &stride, &offset, &rows, &over };
    const size_t sizes[] = { sizeof out, sizeof o, sizeof W, sizeof heads, sizeof width, sizeof latent, sizeof stride, sizeof offset,
                             sizeof rows, sizeof over };
    nn__silicon__launch(0, name, khronos_opencl2__zzprivate_blocks(items), KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 10u);
    return true;
}
static void khronos_opencl2__override_attention_expand(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width,
                                                       uint64_t latent, uint64_t stride, uint64_t offset, unsigned int* over) {
    if (khronos_opencl2__zzprivate_expand_launch(out, o, W, heads, width, latent, stride, offset, 1u, over)) return;
    nn__attention__zzabi_launch_expand(out, o, W, heads, width, latent, stride, offset, over);
}
static void khronos_opencl2__override_attention_expand_rows(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                            uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset,
                                                            uint64_t rows, unsigned int* over) {
    if (khronos_opencl2__zzprivate_expand_launch(out, o, W, heads, width, latent, stride, offset, rows, over)) return;
    nn__attention__zzabi_launch_expand_rows(out, o, W, heads, width, latent, stride, offset, rows, over);
}

/* The latent's grouped scores in tiles: at least 32 query heads a kv head, the width whole slices of 64, every query and key
 * row on 4 bytes for the slices' reads. */
static inline bool khronos_opencl2__zzprivate_latent_scores_fit(const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads,
                                                               uint64_t head_dim) {
    return kv_heads != 0u && q_heads % kv_heads == 0u && q_heads / kv_heads >= 32u && head_dim != 0u && head_dim % 64u == 0u
        && ((uint64_t)(uintptr_t)q) % 4u == 0u && ((uint64_t)(uintptr_t)k) % 4u == 0u;
}
static void khronos_opencl2__override_attention_scores_grouped(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                               uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (length == 0u || !khronos_opencl2__zzprivate_latent_scores_fit(q, k, q_heads, kv_heads, head_dim)
     || !khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__latent_scores")) {
        nn__attention__zzabi_launch_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    const uint64_t tiles = kv_heads * ((q_heads / kv_heads + 63u) / 64u) * ((length + 63u) / 64u);
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length };
    nn__silicon__launch(0, "khronos_opencl2__latent_scores", khronos_opencl2__zzprivate_blocks(tiles), KHRONOS_OPENCL2__ZZPRIVATE_GROUP,
                        args, sizes, 7u);
}
static void khronos_opencl2__override_attention_scores_grouped_rows(float* p, uint64_t p_row, const uint16_t* q, uint64_t q_row,
                                                                    const uint16_t* k, uint64_t k_row, const uint32_t* lens, uint64_t q_heads,
                                                                    uint64_t kv_heads, uint64_t head_dim, uint64_t most, uint64_t rows) {
    if (rows == 0u || most == 0u || q_row % 2u != 0u || k_row % 2u != 0u
     || !khronos_opencl2__zzprivate_latent_scores_fit(q, k, q_heads, kv_heads, head_dim)
     || !khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__latent_scores_rows")) {
        nn__attention__zzabi_launch_scores_grouped_rows(p, p_row, q, q_row, k, k_row, lens, q_heads, kv_heads, head_dim, most, rows);
        return;
    }
    const uint64_t tiles = rows * kv_heads * ((q_heads / kv_heads + 63u) / 64u) * ((most + 63u) / 64u);
    void* args[] = { &p, &p_row, &q, &q_row, &k, &k_row, &lens, &q_heads, &kv_heads, &head_dim, &most, &rows };
    const size_t sizes[] = { sizeof p, sizeof p_row, sizeof q, sizeof q_row, sizeof k, sizeof k_row, sizeof lens, sizeof q_heads,
                             sizeof kv_heads, sizeof head_dim, sizeof most, sizeof rows };
    nn__silicon__launch(0, "khronos_opencl2__latent_scores_rows", khronos_opencl2__zzprivate_blocks(tiles), KHRONOS_OPENCL2__ZZPRIVATE_GROUP,
                        args, sizes, 12u);
}

/* The mix over rows of their own: whole groups; 8 heads a work-group on a discrete card, 16 on an integrated GPU. */
static void khronos_opencl2__override_attention_mix_rows(uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row, const uint16_t* v,
                                                         uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
                                                         uint64_t head_dim, uint64_t rows, unsigned int* over) {
    const uint64_t Q = khronos_opencl2__zzprivate_integrated() ? 16u : 8u;
    const char* name = Q == 8u ? "khronos_opencl2__mix_rows8" : "khronos_opencl2__mix_rows16";
    if (rows == 0u || kv_heads == 0u || q_heads % kv_heads != 0u || head_dim == 0u || !khronos_opencl2__zzpackage_has_kernel(name)) {
        nn__attention__zzabi_launch_mix_rows(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, head_dim, rows, over);
        return;
    }
    const uint64_t items = rows * kv_heads * ((q_heads / kv_heads + Q - 1u) / Q)
                         * ((head_dim + KHRONOS_OPENCL2__ZZPRIVATE_GROUP - 1u) / KHRONOS_OPENCL2__ZZPRIVATE_GROUP);
    void* args[] = { &o, &o_row, &p, &p_row, &v, &v_row, &lens, &q_heads, &kv_heads, &head_dim, &rows, &over };
    const size_t sizes[] = { sizeof o, sizeof o_row, sizeof p, sizeof p_row, sizeof v, sizeof v_row, sizeof lens, sizeof q_heads,
                             sizeof kv_heads, sizeof head_dim, sizeof rows, sizeof over };
    nn__silicon__launch(0, name, khronos_opencl2__zzprivate_blocks(items), KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 12u);
}

/* The indexer's pool scores: at most 64 heads, nn's lanes for them; 8 pools a work-group on a discrete card, 16 on an
 * integrated GPU. */
static bool khronos_opencl2__zzprivate_index_scores_launch(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                    const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0, uint64_t kpool,
                                                    uint64_t rows, float scale) {
    const uint64_t G = khronos_opencl2__zzprivate_integrated() ? 16u : 8u;
    const char* name = G == 8u ? "khronos_opencl2__index_scores8" : "khronos_opencl2__index_scores16";
    if (rows == 0u || kpool == 0u || (len0 + rows - 1u) / kpool == 0u || heads == 0u || heads > 64u
     || !khronos_opencl2__zzpackage_has_kernel(name))
        return false;
    const uint64_t items = rows * (((len0 + rows - 1u) / kpool + G - 1u) / G);
    void* args[] = { &scores, &scores_row, &q, &w, &pooled, &heads, &dim, &len0, &kpool, &rows, &scale };
    const size_t sizes[] = { sizeof scores, sizeof scores_row, sizeof q, sizeof w, sizeof pooled, sizeof heads, sizeof dim, sizeof len0,
                             sizeof kpool, sizeof rows, sizeof scale };
    nn__silicon__launch(0, name, khronos_opencl2__zzprivate_blocks(items), 64u, args, sizes, 11u);
    return true;
}
/* one row's pools a work-group each on a discrete card, as nn's: a row's few hundred pools eight a work-group leave it idle
 * (`MEASURED`, the bench's sparse rows a launch a row on the MI50: 5096.7 us against nn's 3637.1) */
static void khronos_opencl2__override_index_scores(float* scores, const uint16_t* q, const uint16_t* w, const uint16_t* pooled,
                                                   uint64_t heads, uint64_t dim, uint64_t pools, float scale) {
    if (pools != 0u && khronos_opencl2__zzprivate_integrated()
     && khronos_opencl2__zzprivate_index_scores_launch(scores, 0u, q, w, pooled, heads, dim, pools, 1u, 1u, scale)) return;
    nn__index__zzabi_launch_scores(scores, q, w, pooled, heads, dim, pools, scale);
}
static void khronos_opencl2__override_index_scores_rows(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                        const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0,
                                                        uint64_t kpool, uint64_t rows, float scale) {
    if (khronos_opencl2__zzprivate_index_scores_launch(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale)) return;
    nn__index__zzabi_launch_scores_rows(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale);
}

/* The indexer's choice: its counts in 32 bits. */
static void khronos_opencl2__override_index_select(uint32_t* index, uint32_t* count, const float* scores, uint64_t pools, uint64_t k,
                                                   uint64_t kpool, uint64_t tail_first, uint64_t tail) {
    if (pools > 0xFFFFFFFFu || k > 0xFFFFFFFFu || !khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__index_select")) {
        nn__index__zzabi_launch_select(index, count, scores, pools, k, kpool, tail_first, tail);
        return;
    }
    void* args[] = { &index, &count, &scores, &pools, &k, &kpool, &tail_first, &tail };
    const size_t sizes[] = { sizeof index, sizeof count, sizeof scores, sizeof pools, sizeof k, sizeof kpool, sizeof tail_first, sizeof tail };
    nn__silicon__launch(0, "khronos_opencl2__index_select", 1u, KHRONOS_OPENCL2__ZZPRIVATE_GROUP, args, sizes, 8u);
}
static void khronos_opencl2__override_index_select_rows(uint32_t* index, uint64_t index_row, uint32_t* counts, const float* scores,
                                                        uint64_t scores_row, uint64_t k, uint64_t kpool, uint64_t len0, uint64_t rows) {
    if (rows == 0u || kpool == 0u || (len0 + rows) / kpool > 0xFFFFFFFFu || k > 0xFFFFFFFFu
     || !khronos_opencl2__zzpackage_has_kernel("khronos_opencl2__index_select_rows")) {
        nn__index__zzabi_launch_select_rows(index, index_row, counts, scores, scores_row, k, kpool, len0, rows);
        return;
    }
    void* args[] = { &index, &index_row, &counts, &scores, &scores_row, &k, &kpool, &len0, &rows };
    const size_t sizes[] = { sizeof index, sizeof index_row, sizeof counts, sizeof scores, sizeof scores_row, sizeof k, sizeof kpool,
                             sizeof len0, sizeof rows };
    nn__silicon__launch(0, "khronos_opencl2__index_select_rows", khronos_opencl2__zzprivate_blocks(rows), KHRONOS_OPENCL2__ZZPRIVATE_GROUP,
                        args, sizes, 9u);
}

static inline void khronos_opencl2__override_doors(nn__doors* doors) {
    doors->turboquant_gemv            = khronos_opencl2__override_turboquant_gemv;
    doors->turboquant_gemv_groups     = khronos_opencl2__override_gemv_groups;
    doors->turboquant_gemv_groups_sum = khronos_opencl2__override_gemv_groups_sum;
    doors->hadamard_rotate            = khronos_opencl2__override_hadamard_rotate;
    doors->attention_weights          = khronos_opencl2__override_attention_weights;
    doors->attention_residual_mix     = khronos_opencl2__override_attention_residual_mix;
    doors->expert_rows                = khronos_opencl2__override_expert_rows;
    doors->expert_rows_sum            = khronos_opencl2__override_expert_rows_sum;
    doors->expert_groups              = khronos_opencl2__override_expert_groups;
    doors->vector_top_k               = khronos_opencl2__override_vector_top_k;
    doors->kda_steps_decayed             = khronos_opencl2__override_kda_steps_decayed;
    doors->attention_absorb              = khronos_opencl2__override_attention_absorb;
    doors->attention_absorb_rows         = khronos_opencl2__override_attention_absorb_rows;
    doors->attention_expand              = khronos_opencl2__override_attention_expand;
    doors->attention_expand_rows         = khronos_opencl2__override_attention_expand_rows;
    doors->attention_scores_grouped      = khronos_opencl2__override_attention_scores_grouped;
    doors->attention_scores_grouped_rows = khronos_opencl2__override_attention_scores_grouped_rows;
    doors->attention_mix_rows            = khronos_opencl2__override_attention_mix_rows;
    doors->index_scores                  = khronos_opencl2__override_index_scores;
    doors->index_scores_rows             = khronos_opencl2__override_index_scores_rows;
    doors->index_select                  = khronos_opencl2__override_index_select;
    doors->index_select_rows             = khronos_opencl2__override_index_select_rows;
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_NN_CUH */
