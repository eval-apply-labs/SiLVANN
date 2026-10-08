#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_PRIMITIVES_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_PRIMITIVES_CUH
/* ══ primitives' DOORS, RUN FASTER — amd_rocm6_wave64 ═════════════════════════════════════════════════════════════════
 * nn's primitives doors (`nn/cpu/opcodes/primitives__abi.cuh`), where this family covers them: rmsnorm, over one row
 * and over several. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* The device's compute units, kept per thread as the blocks above are. */
static inline uint32_t amd_rocm6_wave64__zzprivate_compute_units(void) {
    static thread_local int seen_device = -1;
    static thread_local uint32_t seen_units = 1u;
    int device = 0, units = 0;
    (void)hipGetDevice(&device);
    if (device != seen_device) {
        if (hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess || units < 1) units = 1;
        seen_device = device; seen_units = (uint32_t)units;
    }
    return seen_units;
}

/* ══ ⭐ rmsnorm, A ROW A BLOCK, EIGHT HALVES A READ ═══════════════════════════════════════════════════════════
 * nn's body takes a row's squares one half at a time, `lane`, `lane + lanes`, …: at the 27B's 5120 that is twenty dependent
 * reads a lane, and a call cost ~24 us on the card (`MEASURED`, rocprof over a 27B decode) where its bytes are 30 KB. Here
 * each lane reads eight halves at once (16 bytes), so the row is three reads deep; the sum goes through the family's own
 * fold. The arithmetic is nn's — the mean of the squares in fp32, `1 / sqrt(mean + eps)`, each element `(x · scale) · w`
 * rounded once — with the squares grouped eight to a lane, so the sum's last bits are this kernel's own. ⛳ BOTH DOORS TAKE
 * IT, the one-row and the rows — or the wave form below, the same bits — so a prompt's rows and its positions norm alike. Rows whose width is not whole groups of
 * eight, or whose memory is not 16-byte aligned, take nn's generic door. */
/* A group's eight squares onto `part`, in order, each a fused multiply-add written as one: the one expression both forms
 * below sum with. ⛳ Written, not left to the compiler: on gfx90a, which has packed fp32, it splits `part += a * a` into a
 * packed multiply and an add in one form and fuses it in the other (`REASONED`, the ISA of a 3-arch build), and the two
 * forms must agree. */
static __device__ inline float amd_rocm6_wave64__zzprivate_squares8(uint4 q, float part) {
    const uint32_t words[4] = { q.x, q.y, q.z, q.w };
    for (uint32_t k = 0u; k < 4u; ++k) {
        const float a = nn__silicon__half_to_float((uint16_t)(words[k] & 0xFFFFu)), b = nn__silicon__half_to_float((uint16_t)(words[k] >> 16));
        part = __builtin_fmaf(a, a, part);
        part = __builtin_fmaf(b, b, part);
    }
    return part;
}
/* A group's eight outputs, `(x · scale) · w` each rounded once. */
static __device__ inline uint4 amd_rocm6_wave64__zzprivate_normed8(uint4 q, uint4 r, float scale, unsigned int* over) {
    const uint32_t xs[4] = { q.x, q.y, q.z, q.w }, ws[4] = { r.x, r.y, r.z, r.w };
    uint32_t out[4];
    for (uint32_t k = 0u; k < 4u; ++k) {
        const uint32_t lo = amd_rocm6_wave64__zzpackage_to_half((nn__silicon__half_to_float((uint16_t)(xs[k] & 0xFFFFu)) * scale)
                                                                * nn__silicon__half_to_float((uint16_t)(ws[k] & 0xFFFFu)), over);
        const uint32_t hi = amd_rocm6_wave64__zzpackage_to_half((nn__silicon__half_to_float((uint16_t)(xs[k] >> 16)) * scale)
                                                                * nn__silicon__half_to_float((uint16_t)(ws[k] >> 16)), over);
        out[k] = lo | (hi << 16);
    }
    return make_uint4(out[0], out[1], out[2], out[3]);
}
static __device__ inline void amd_rocm6_wave64__zzprivate_rmsnorm_row(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n,
                                                                       float eps, unsigned int* over) {
    const uint32_t lane = (uint32_t)threadIdx.x, lanes = (uint32_t)blockDim.x;
    const uint64_t groups = n / 8u;
    float part = 0.0f;
    for (uint64_t g = lane; g < groups; g += lanes) part = amd_rocm6_wave64__zzprivate_squares8(((const uint4*)(const void*)x)[g], part);
    const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(part) / (float)n + eps);
    for (uint64_t g = lane; g < groups; g += lanes)
        ((uint4*)(void*)o)[g] = amd_rocm6_wave64__zzprivate_normed8(((const uint4*)(const void*)x)[g], ((const uint4*)(const void*)w)[g], scale, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_rmsnorm_rows(uint16_t* o, const uint16_t* x,
        const uint16_t* w, uint64_t n, uint64_t rows, uint64_t x_stride, uint64_t o_stride, float eps, unsigned int* over) {
    for (uint64_t r = blockIdx.x; r < rows; r += gridDim.x)
        amd_rocm6_wave64__zzprivate_rmsnorm_row(o + r * o_stride, x + r * x_stride, w, n, eps, over);
}

/* ⭐ THE SAME NORM, A WAVE A SLICE OF A ROW — the block above, run by one wave. The block's arithmetic is lane `L` of 256
 *   summing its groups `L`, `L + 256`, … and the fold adding the 256 parts pairwise, neighbours first. Lane `l` of a wave
 *   keeps the parts of lanes `l`, `l + 64`, `l + 128` and `l + 192`: the fold's first six rounds are the wave's shuffles on
 *   each of the four, and its last two are `(p0 + p1) + (p2 + p3)` — the same additions in the same order, so the same
 *   bits, and no shared memory and no barrier. A wave then needs no other wave's sum, so a row is cut across waves (`cuts`),
 *   each taking the whole sum itself and writing its own groups: a decoded token's one row of 5120 is ten waves, not one
 *   block. All of a row's reads are issued before the first is waited on, so a row waits on memory once (`REASONED`, the ISA:
 *   sixteen global loads ahead of the first wait in the kernel of twelve); the writes are a loop of one group at a time, so
 *   the kernel holds one copy of them however wide the row — unrolled, the writes made the kernel larger and slower. Up to
 *   64 · `HELD` groups a row; wider rows are the block's. ⛔ A ROW NORMED IN PLACE IS ONE WAVE'S: cut across waves, a slice
 *   another wave writes could land before this wave has read it for its sum — `MEASURED` racing under load before the
 *   check, 0 of 4,000 after it — and nn's verb promises in place.
 *   `MEASURED` (a sweep run once, back to back, gfx906, against the block with the same conversion): rows of 5120 5.4-7.7
 *   us over 4 to 48 rows where the block is 7.5-9.1, and past the compute units' count of rows the block is faster — so the
 *   slices are taken while the rows are fewer than the compute units, and the block past that. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_NORM_HELD 12u      /* groups of eight a lane holds: rows up to 6,144 halves */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_norm_slices(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n,
                                                                       uint64_t rows, uint64_t x_stride, uint64_t o_stride,
                                                                       uint64_t cuts, float eps, unsigned int* over) {
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t waves = (uint32_t)(nn__silicon__lanes() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t groups = n / 8u, held = (groups + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t per = (held + cuts - 1u) / cuts;    /* a cut's groups a lane */
    const uint4 none = make_uint4(0u, 0u, 0u, 0u);     /* a group past the row: squares of +0, which add nothing */
    const uint4* wr = (const uint4*)(const void*)w;
    for (uint64_t t = (uint64_t)nn__silicon__block() * waves + nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
         t < rows * cuts; t += (uint64_t)nn__silicon__blocks() * waves) {
        const uint64_t r = t / cuts, k0 = (t % cuts) * per, k1 = k0 + per < held ? k0 + per : held;
        const uint4* xr = (const uint4*)(const void*)(x + r * x_stride);
        /* the slice's first group of `x` and `w` read with the sum's, so its write waits on nothing more */
        uint64_t g = lane + k0 * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        uint4 xa = k0 < k1 && g < groups ? xr[g] : none, wa = k0 < k1 && g < groups ? wr[g] : none;
        uint4 q[HELD];
#pragma unroll
        for (uint32_t k = 0u; k < HELD; ++k) {
            const uint64_t gk = lane + (uint64_t)k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
            q[k] = k < held && gk < groups ? xr[gk] : none;
        }
        float part[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (uint32_t k = 0u; k < HELD; ++k)
            if (k < held) part[k % 4u] = amd_rocm6_wave64__zzprivate_squares8(q[k], part[k % 4u]);
#pragma unroll
        for (uint32_t off = 1u; off < AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE; off <<= 1)
#pragma unroll
            for (uint32_t k = 0u; k < (HELD < 4u ? HELD : 4u); ++k) part[k] += amd_rocm6_wave64__zzpackage_xor(part[k], off);
        const float scale = 1.0f / nn__silicon__sqrtf(((part[0] + part[1]) + (part[2] + part[3])) / (float)n + eps);
        /* the slice's groups one at a time, the next one's reads in flight while this one is written: one copy of the
         * writing code however many groups a lane holds */
        uint4* orow = (uint4*)(void*)(o + r * o_stride);
#pragma unroll 1
        for (uint64_t k = k0; k < k1; ++k) {
            const uint64_t gn = g + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
            const uint4 xn = k + 1u < k1 && gn < groups ? xr[gn] : none, wn = k + 1u < k1 && gn < groups ? wr[gn] : none;
            if (g < groups) orow[g] = amd_rocm6_wave64__zzprivate_normed8(xa, wa, scale, over);
            g = gn; xa = xn; wa = wn;
        }
    }
}
/* A kernel for each count of groups a lane holds, so a row needs only the registers its width uses. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_norm_h1(uint16_t* o, const uint16_t* x, const uint16_t* w,
        uint64_t n, uint64_t rows, uint64_t x_stride, uint64_t o_stride, uint64_t cuts, float eps, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_norm_slices<1u>(o, x, w, n, rows, x_stride, o_stride, cuts, eps, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_norm_h4(uint16_t* o, const uint16_t* x, const uint16_t* w,
        uint64_t n, uint64_t rows, uint64_t x_stride, uint64_t o_stride, uint64_t cuts, float eps, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_norm_slices<4u>(o, x, w, n, rows, x_stride, o_stride, cuts, eps, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_norm_h12(uint16_t* o, const uint16_t* x, const uint16_t* w,
        uint64_t n, uint64_t rows, uint64_t x_stride, uint64_t o_stride, uint64_t cuts, float eps, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_norm_slices<12u>(o, x, w, n, rows, x_stride, o_stride, cuts, eps, over);
}

static inline bool amd_rocm6_wave64__zzprivate_norm_fit(const void* o, const void* x, const void* w, uint64_t n, uint64_t x_stride,
                                                         uint64_t o_stride) {
    return n != 0u && n % 8u == 0u && x_stride % 8u == 0u && o_stride % 8u == 0u && ((uint64_t)(uintptr_t)o) % 16u == 0u
        && ((uint64_t)(uintptr_t)x) % 16u == 0u && ((uint64_t)(uintptr_t)w) % 16u == 0u;
}
static void amd_rocm6_wave64__override_rmsnorm_rows(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, uint64_t rows,
                                                    uint64_t x_stride, uint64_t o_stride, float eps, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_norm_fit(o, x, w, n, x_stride, o_stride) || rows == 0u) {
        nn__rmsnorm__zzabi_launch_rows(o, x, w, n, rows, x_stride, o_stride, eps, over);
        return;
    }
    /* a wave's slices while the rows are fewer than the compute units, so every unit has work; a row a block beyond */
    const uint64_t held = (n / 8u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    if (held <= AMD_ROCM6_WAVE64__ZZPRIVATE_NORM_HELD && rows < amd_rocm6_wave64__zzprivate_compute_units()) {
        const void* kernel = held == 1u ? (const void*)amd_rocm6_wave64__zzprivate_norm_h1
                           : held <= 4u ? (const void*)amd_rocm6_wave64__zzprivate_norm_h4
                           :              (const void*)amd_rocm6_wave64__zzprivate_norm_h12;
        const uint32_t threads = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        /* each row cut across as many waves as the card holds at once for this many rows — rounded down, since a wave
         * the card cannot hold yet waits for a whole one to finish — and at most a wave a group */
        const uint64_t waves = (uint64_t)amd_rocm6_wave64__zzpackage_resident(kernel, threads, 0u) * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
        uint64_t cuts = waves / rows;
        cuts = cuts < 1u ? 1u : cuts > held ? held : cuts;
        /* a row normed in place is one wave's: a slice another wave writes could land before this wave has read it for its sum */
        if (!(o + (rows - 1u) * o_stride + n <= x || x + (rows - 1u) * x_stride + n <= o)) cuts = 1u;
        const uint64_t want = (rows * cuts + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
        const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
        void* args[] = { &o, &x, &w, &n, &rows, &x_stride, &o_stride, &cuts, &eps, &over };
        const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof rows, sizeof x_stride, sizeof o_stride, sizeof cuts,
                                 sizeof eps, sizeof over };
        nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_norm", blocks, threads, args, sizes, 10u);
        return;
    }
    const uint32_t blocks = rows < NN__KERNELS__BLOCKS_MAX ? (uint32_t)rows : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &x, &w, &n, &rows, &x_stride, &o_stride, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof rows, sizeof x_stride, sizeof o_stride, sizeof eps, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_rmsnorm_rows, "amd_rocm6_wave64__zzprivate_rmsnorm_rows", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 9u);
}
static void amd_rocm6_wave64__override_rmsnorm(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps,
                                               unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_norm_fit(o, x, w, n, n, n)) { nn__rmsnorm__zzabi_launch(o, x, w, n, eps, over); return; }
    amd_rocm6_wave64__override_rmsnorm_rows(o, x, w, n, 1u, n, n, eps, over);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_PRIMITIVES_CUH */
