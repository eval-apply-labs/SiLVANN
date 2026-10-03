#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_NN_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include <string.h>                          /* memset, for a tile job built on the stack */
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
 *   row or two each (▶ the ROCm family's `zzprivate_resident`, where it measured 158 -> 73 us). Kept per thread. */
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
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_NN_CUH */
