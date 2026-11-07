#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_PRIMITIVES_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_PRIMITIVES_CUH
/* ══ primitives' DOOR, RUN FASTER — nvidia_cuda12 ═════════════════════════════════════════════════════════════════════
 * nn's primitives doors (`nn/cpu/opcodes/primitives__abi.cuh`), where this family covers them: rmsnorm over one row and
 * over several, of halves and of floats, and the norm with its rotation in one launch. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */
#include "hadamard.cuh"     /* a warp's block of the rotation, the one `hadamard_rotate` runs */

/* ══ ⭐ rmsnorm, A ROW A BLOCK, EIGHT HALVES A READ ═══════════════════════════════════════════════════════════════════
 * nn's body takes a row's squares one half at a time, `lane`, `lane + lanes`, …: at the 27B's 5120 twenty dependent reads
 * a lane. Here, as amd_rocm6_wave64's block form, each of the block's 256 lanes reads eight halves at once (16 bytes), so
 * the row is three reads deep, and the sum goes through the family's fold. The arithmetic is nn's — the mean of the squares
 * in fp32, `1 / sqrt(mean + eps)`, each element `(x · scale) · w` rounded once — with the squares grouped eight to a lane,
 * so the sum's last bits are this kernel's own. ⛳ BOTH DOORS TAKE IT, the one-row and the rows, so a prompt's rows and its
 * positions norm alike; and `rmsnorm_rotate` below takes the same sum. Rows whose width is not whole groups of eight, or
 * whose memory is not 16-byte aligned, are nn's. AMD's other form, a row cut across waves that each take the whole sum, is
 * gfx906's answer to a decoded token's one row on one block; here a row stays one block's (`REASONED`: a row of 5120 is
 * 10 KB, and the launch, not the block, sets its time — to be measured). A row normed in place is safe: every lane reads
 * the row for the sum before the fold's waits, and afterwards reads and writes only its own groups. */
#define NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES 256u       /* nn's door's */
#define NVIDIA_CUDA12__ZZPRIVATE_NORM_BOUNDS __launch_bounds__(NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES)
/* A group's eight squares onto `part`, in order, each a fused multiply-add written as one, so every form sums alike. */
static __device__ __forceinline__ float nvidia_cuda12__zzprivate_squares8(uint4 q, float part) {
    const uint32_t words[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
    for (uint32_t k = 0u; k < 4u; ++k) {
        const float a = nn__silicon__half_to_float((uint16_t)(words[k] & 0xFFFFu)), b = nn__silicon__half_to_float((uint16_t)(words[k] >> 16));
        part = __fmaf_rn(a, a, part);
        part = __fmaf_rn(b, b, part);
    }
    return part;
}
/* One element normed, `(x · scale) · w` rounded once: the expression every form here writes with. */
static __device__ __forceinline__ uint32_t nvidia_cuda12__zzprivate_normed(uint32_t x, uint32_t w, float scale, unsigned int* over) {
    return nvidia_cuda12__zzpackage_to_half((nn__silicon__half_to_float((uint16_t)x) * scale) * nn__silicon__half_to_float((uint16_t)w), over);
}
static __device__ __forceinline__ uint4 nvidia_cuda12__zzprivate_normed8(uint4 q, uint4 r, float scale, unsigned int* over) {
    const uint32_t xs[4] = { q.x, q.y, q.z, q.w }, ws[4] = { r.x, r.y, r.z, r.w };
    uint32_t out[4];
#pragma unroll
    for (uint32_t k = 0u; k < 4u; ++k)
        out[k] = nvidia_cuda12__zzprivate_normed(xs[k] & 0xFFFFu, ws[k] & 0xFFFFu, scale, over)
               | (nvidia_cuda12__zzprivate_normed(xs[k] >> 16, ws[k] >> 16, scale, over) << 16);
    return make_uint4(out[0], out[1], out[2], out[3]);
}
/* A row's scale, `1 / sqrt(mean + eps)`, by the whole block: lane `L` of 256 sums its groups `L`, `L + 256`, … */
static __device__ __forceinline__ float nvidia_cuda12__zzprivate_norm_scale(const uint16_t* x, uint64_t n, float eps) {
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint64_t groups = n / 8u;
    float part = 0.0f;
    for (uint64_t g = lane; g < groups; g += lanes) part = nvidia_cuda12__zzprivate_squares8(((const uint4*)(const void*)x)[g], part);
    return 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(part) / (float)n + eps);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_NORM_BOUNDS void nvidia_cuda12__zzprivate_norm_rows(uint16_t* o, const uint16_t* x,
        const uint16_t* w, uint64_t n, uint64_t rows, uint64_t x_stride, uint64_t o_stride, float eps, unsigned int* over) {
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint64_t groups = n / 8u;
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        const uint16_t* xr = x + r * x_stride;
        uint4* orow = (uint4*)(void*)(o + r * o_stride);
        const float scale = nvidia_cuda12__zzprivate_norm_scale(xr, n, eps);
        for (uint64_t g = lane; g < groups; g += lanes)
            orow[g] = nvidia_cuda12__zzprivate_normed8(((const uint4*)(const void*)xr)[g], ((const uint4*)(const void*)w)[g], scale, over);
    }
}
static inline bool nvidia_cuda12__zzprivate_norm_fit(const void* o, const void* x, const void* w, uint64_t n, uint64_t x_stride,
                                                      uint64_t o_stride) {
    return n != 0u && n % 8u == 0u && x_stride % 8u == 0u && o_stride % 8u == 0u && ((uint64_t)(uintptr_t)o) % 16u == 0u
        && ((uint64_t)(uintptr_t)x) % 16u == 0u && ((uint64_t)(uintptr_t)w) % 16u == 0u;
}
static void nvidia_cuda12__override_rmsnorm_rows(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, uint64_t rows,
                                                 uint64_t x_stride, uint64_t o_stride, float eps, unsigned int* over) {
    if (!nvidia_cuda12__zzprivate_norm_fit(o, x, w, n, x_stride, o_stride) || rows == 0u) {
        nn__rmsnorm__zzabi_launch_rows(o, x, w, n, rows, x_stride, o_stride, eps, over);
        return;
    }
    const uint32_t blocks = rows < NN__KERNELS__BLOCKS_MAX ? (uint32_t)rows : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &x, &w, &n, &rows, &x_stride, &o_stride, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof rows, sizeof x_stride, sizeof o_stride, sizeof eps, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_norm_rows, "nvidia_cuda12__zzprivate_norm_rows", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES, args, sizes, 9u);
}
static void nvidia_cuda12__override_rmsnorm(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    if (!nvidia_cuda12__zzprivate_norm_fit(o, x, w, n, n, n)) { nn__rmsnorm__zzabi_launch(o, x, w, n, eps, over); return; }
    nvidia_cuda12__override_rmsnorm_rows(o, x, w, n, 1u, n, n, eps, over);
}

/* ⭐ THE NORM AND ITS ROTATION IN ONE LAUNCH — `rmsnorm_rotate`. nn's definition is the norm's door and then the
 *   rotation's (`nn/gpu/doors.cuh`), two launches and the normed row read back. Here every block of 256 lanes takes the
 *   row's whole scale as the norm above takes it — the same lanes, the same groups, the same fold, so the same scale — and
 *   then each of its warps a 512-block of the row: a lane writes `o` at its elements `l, l + 32, …` as the norm writes them,
 *   and the warp rotates them with `hadamard_rotate`'s own `rotation_at`, whose lane reads exactly the elements it has just
 *   written, so no lane waits on another. `o` and `rot` are then, to the bit, the two doors' (`REASONED` from the shared
 *   code; to be checked by the door bench). Rows `n` a multiple of 512, `o`, `x` and `w` on 16 bytes, the three vectors
 *   apart; anything else is nn's two doors. */
#define NVIDIA_CUDA12__ZZPRIVATE_NORM_ROT_WARPS 8u     /* 256 lanes: the norm's block */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_NORM_BOUNDS void nvidia_cuda12__zzprivate_norm_rotate(uint16_t* rot, uint16_t* o,
        const uint16_t* x, const uint16_t* w, const uint16_t* sign, uint64_t n, float eps, unsigned int* over) {
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint32_t warp = (uint32_t)nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const float scale = nvidia_cuda12__zzprivate_norm_scale(x, n, eps);
    const uint64_t units = n / 512u;
    for (uint64_t u = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_NORM_ROT_WARPS + warp; u < units;
         u += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_NORM_ROT_WARPS) {
        const uint64_t at = u * 512u;
#pragma unroll
        for (uint32_t e = 0u; e < 16u; ++e) {
            const uint64_t i = at + lane + NVIDIA_CUDA12__ZZPACKAGE_WARP * e;
            o[i] = (uint16_t)nvidia_cuda12__zzprivate_normed(x[i], w[i], scale, over);
        }
        nvidia_cuda12__zzpackage_rotation_at(rot, o, at, sign, 0u, over, 16u);
    }
}
static void nvidia_cuda12__override_rmsnorm_rotate(uint16_t* rot, uint16_t* o, const uint16_t* x, const uint16_t* w, const uint16_t* sign,
                                                   uint64_t n, float eps, unsigned int* over) {
    const bool apart = (rot + n <= o || o + n <= rot) && (rot + n <= x || x + n <= rot) && (o + n <= x || x + n <= o);
    if (n == 0u || n % 512u != 0u || !apart || !nvidia_cuda12__zzprivate_norm_fit(o, x, w, n, n, n)) {
        nn__rmsnorm__zzabi_launch_rotate(rot, o, x, w, sign, n, eps, over);
        return;
    }
    const uint64_t want = (n / 512u + NVIDIA_CUDA12__ZZPRIVATE_NORM_ROT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_NORM_ROT_WARPS;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &rot, &o, &x, &w, &sign, &n, &eps, &over };
    const size_t sizes[] = { sizeof rot, sizeof o, sizeof x, sizeof w, sizeof sign, sizeof n, sizeof eps, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_norm_rotate, "nvidia_cuda12__zzprivate_norm_rotate", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES, args, sizes, 8u);
}

/* ⭐ THE NORM OF A ROW OF FLOATS, INTO HALVES — nn's `rmsnorm_fp32`, a hyper-connection's collapse (DeepSeek's streams are
 *   floats): nn's block, lanes and fold, a lane's squares added element by element as nn's lane adds them, each element
 *   `(x · scale) · w` rounded once, its overflow in the fault word — so nn's bits. What is this family's own: a lane's
 *   elements read RUN at a time, all in flight together, where nn's body waits for each before the next, twice. */
#define NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN   16u
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_norm_fp32_row(uint16_t* o, const float* x, const uint16_t* w, uint64_t n,
                                                                              float eps, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint32_t RUN = NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN;
    float part = 0.0f;
    uint64_t i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN];
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) v[u] = x[i + u * lanes];
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) part += v[u] * v[u];
    }
    for (; i < n; i += lanes) part += x[i] * x[i];
    const float mean  = nn__silicon__lanes_sum(part) / (float)n;
    const float scale = 1.0f / nn__silicon__sqrtf(mean + eps);
    i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN], g[NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN];
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) {
            v[u] = x[i + u * lanes];
            g[u] = nn__silicon__half_to_float(w[i + u * lanes]);
        }
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) o[i + u * lanes] = nn__kernels__zzabi_to_half((v[u] * scale) * g[u], over);
    }
    for (; i < n; i += lanes) o[i] = nn__kernels__zzabi_to_half((x[i] * scale) * nn__silicon__half_to_float(w[i]), over);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_NORM_BOUNDS void nvidia_cuda12__zzprivate_norm_fp32(uint16_t* o, const float* x, const uint16_t* w,
                                                                                               uint64_t n, float eps, unsigned int* over) {
    nvidia_cuda12__zzprivate_norm_fp32_row(o, x, w, n, eps, over);
}
/* nn's door's one block of 256 lanes, so a lane's stride and the fold are nn's; a row written over its own floats is nn's
 * door's (a lane reads its elements again after the fold, and writes halves over another lane's floats). */
static void nvidia_cuda12__override_rmsnorm_fp32(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    const bool apart = (const char*)(o + n) <= (const char*)x || (const char*)(x + n) <= (const char*)o;
    if (n == 0u || !apart) {
        nn__rmsnorm__zzabi_launch_fp32(o, x, w, n, eps, over);
        return;
    }
    void* args[] = { &o, &x, &w, &n, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof eps, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_norm_fp32, "nvidia_cuda12__zzprivate_norm_fp32", 1u,
                        NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES, args, sizes, 6u);
}

/* ⭐ …AND `rows` OF THEM, nn's `rmsnorm_fp32_rows`: row `r` of `n` floats normed by the gain's row `r mod period`, a block
 *   a row, each the one-row kernel's body — so nn's bits, row by row, where nn's body launched the same block a row with
 *   an element at a time. Every row's floats apart from every row's halves and `period` not 0 (AMD's fit: nn's body takes
 *   the gain's first row there); anything else is nn's. */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_NORM_BOUNDS void nvidia_cuda12__zzprivate_norm_fp32_rows(uint16_t* o, const float* x,
                                                                                                    const uint16_t* w, uint64_t n,
                                                                                                    uint64_t rows, uint64_t period,
                                                                                                    float eps, unsigned int* over) {
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks())
        nvidia_cuda12__zzprivate_norm_fp32_row(o + r * n, x + r * n, w + (r % period) * n, n, eps, over);
}
static void nvidia_cuda12__override_rmsnorm_fp32_rows(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, uint64_t rows,
                                                      uint64_t period, float eps, unsigned int* over) {
    const bool apart = (const char*)(o + rows * n) <= (const char*)x || (const char*)(x + rows * n) <= (const char*)o;
    if (n == 0u || rows == 0u || period == 0u || !apart) {
        nn__rmsnorm__zzabi_launch_fp32_rows(o, x, w, n, rows, period, eps, over);
        return;
    }
    const uint32_t blocks = rows < NN__KERNELS__BLOCKS_MAX ? (uint32_t)rows : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &x, &w, &n, &rows, &period, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof rows, sizeof period, sizeof eps, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_norm_fp32_rows, "nvidia_cuda12__zzprivate_norm_fp32_rows", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES, args, sizes, 8u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_PRIMITIVES_CUH */
