#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_KERNELS_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_KERNELS_CUH
/* ══ ⭐⭐ THE KERNELS OF THIS FAMILY'S OVERRIDES — OpenCL C, IN nn's PROGRAM ══════════════════════════════════
 * `scripts/opencl_kernel_source.py` preprocesses this after nn's bodies into the one program the family compiles, so
 * a kernel here may use everything nn's are written with — the level tables, the half conversions, the overflow
 * word. The host side that launches them, and falls back to nn's generic door for whatever they do not cover, is
 * `nn.cuh` beside this. ⛳ OpenCL C only: a C++ compiler reading the family's host side never sees it.
 *
 * What they do that nn's generic bodies do not (▶ `measurements/2026-10-01_opencl_vs_rocm_where_the_gap_is.md`):
 *   · A SUB-GROUP A ROW. A work-group of 256 is its sub-groups (64 lanes each on the MI50), each taking a row at a
 *     time, and a row's sum is `sub_group_reduce_add` — no barrier. nn's generic body is a work-group a row and a
 *     `work_group_reduce_add`, a barrier a row.
 *   · `x` (as halves) AND THE LEVELS IN LOCAL MEMORY, loaded once for every row the work-group takes, where the generic body reads
 *     `x` again for every row and finds each level through nn's table in global memory.
 *   · A LANE A BLOCK OF 32 COLUMNS, its codes read whole — 16 bytes at 4 bits, 32 at 8.
 *   · THE ROTATION A BUTTERFLY, a 512-block a work-group in local memory — O(b log b) where the generic body sums
 *     each output directly.
 * The arithmetic is the generic bodies': fp32 sums of level · x, the row's scale once, a half rounded once; only the
 * order of the float sum differs. ⛳ `cl_khr_subgroups` is OpenCL 2.0's — `MEASURED`, `clinfo` lists it for the MI50 —
 * and the reduction is all these take from it: OpenCL 2.0 has no sub-group shuffle, so the rotation stays in local
 * memory. */
/* ⛔ ONLY WHERE THE DEVICE HAS SUB-GROUPS. The family is the path for older and smaller cards — an Intel iGPU among them —
 *   and a driver without `cl_khr_subgroups` must still build nn's program: there these kernels are left out, the host
 *   side finds them missing (`nn.cuh`) and every call takes nn's generic door. A sub-group's width is not assumed
 *   either: 64 on the MI50, 8 to 32 on Intel's, and a work-group simply takes as many rows as it has sub-groups. */
#if defined(__OPENCL_C_VERSION__) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define KHRONOS_OPENCL2__ZZPRIVATE_COLS_MAX   8192u     /* the widest `x` a work-group holds: 16 KB of halves */
#define KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK  512u

/* `x` (`cols` halves), the sixteen 4-bit levels and the 256 8-bit ones, into the work-group's local memory. */
static inline void khronos_opencl2__zzprivate_stage(__local uint16_t* xs, __local float* lev4, __local float* lev8,
                                                    __global const uint16_t* x, uint64_t cols) {
    const uint32_t lane = (uint32_t)get_local_id(0), lanes = (uint32_t)get_local_size(0);
    for (uint64_t c = lane; c < cols; c += lanes) xs[c] = x[c];
    const uint16_t* t4 = nn__turboquant__zzabi_levels(4u);
    const uint16_t* t8 = nn__turboquant__zzabi_levels(8u);
    for (uint32_t i = lane; i < 16u; i += lanes) lev4[i] = nn__silicon__half_to_float(t4[i]);
    for (uint32_t i = lane; i < 256u; i += lanes) lev8[i] = nn__silicon__half_to_float(t8[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
}

/* One row's dot with `xs` over its whole 32-column blocks, this lane's share: a block `b`, `b + lanes`, … */
static inline float khronos_opencl2__zzprivate_row_dot(uint32_t d, __global const uint8_t* row, uint64_t cols,
                                                       __local const uint16_t* xs, __local const float* lev4,
                                                       __local const float* lev8) {
    const uint32_t lane = (uint32_t)get_sub_group_local_id(), lanes = (uint32_t)get_sub_group_size();
    const uint64_t blocks = cols / 32u;
    float part = 0.0f;
    for (uint64_t b = lane; b < blocks; b += lanes) {
        __local const uint16_t* xb = xs + b * 32u;
        if (d == 4u) {
            __global const ulong* w = (__global const ulong*)(row + b * 16u);
            for (uint32_t h = 0u; h < 2u; ++h) {
                const ulong word = w[h];                        /* sixteen columns, the low nibble first */
                for (uint32_t j = 0u; j < 16u; ++j) part += lev4[(uint32_t)(word >> (4u * j)) & 0xFu] * nn__silicon__half_to_float(xb[16u * h + j]);
            }
        } else {
            __global const ulong* w = (__global const ulong*)(row + b * 32u);
            for (uint32_t h = 0u; h < 4u; ++h) {
                const ulong word = w[h];                        /* eight columns, a byte each */
                for (uint32_t j = 0u; j < 8u; ++j) part += lev8[(uint32_t)(word >> (8u * j)) & 0xFFu] * nn__silicon__half_to_float(xb[8u * h + j]);
            }
        }
    }
    return part;
}

static inline float khronos_opencl2__zzprivate_scale(__global const uint8_t* luts, uint64_t r) {
    return nn__silicon__half_to_float((uint16_t)((uint32_t)luts[r * 2ul] | ((uint32_t)luts[r * 2ul + 1ul] << 8)));
}

/* The row this sub-group takes first, and the stride to its next. */
#define KHRONOS_OPENCL2__ZZPRIVATE_FIRST_ROW  ((uint64_t)get_group_id(0) * get_num_sub_groups() + get_sub_group_id())
#define KHRONOS_OPENCL2__ZZPRIVATE_ROW_STRIDE ((uint64_t)get_num_groups(0) * get_num_sub_groups())

/* ⭐ `turboquant_gemv` at 4 and 8 bits: `out[r] = scale[r] · (W[r] · x)`. */
__kernel void khronos_opencl2__gemv(ulong out__address, ulong weights__address, ulong luts__address, ulong x__address,
                                    uint64_t d, uint64_t rows, uint64_t cols, ulong over__address) {
    __global uint16_t* out = (__global uint16_t*)out__address;
    __global const uint8_t* weights = (__global const uint8_t*)weights__address;
    __global const uint8_t* luts = (__global const uint8_t*)luts__address;
    __global unsigned int* over = (__global unsigned int*)over__address;
    __local uint16_t xs[KHRONOS_OPENCL2__ZZPRIVATE_COLS_MAX];
    __local float lev4[16];
    __local float lev8[256];
    khronos_opencl2__zzprivate_stage(xs, lev4, lev8, (__global const uint16_t*)x__address, cols);
    const uint64_t row_bytes = cols * d / 8u;
    for (uint64_t r = KHRONOS_OPENCL2__ZZPRIVATE_FIRST_ROW; r < rows; r += KHRONOS_OPENCL2__ZZPRIVATE_ROW_STRIDE) {
        const float sum = sub_group_reduce_add(khronos_opencl2__zzprivate_row_dot((uint32_t)d, weights + r * row_bytes, cols,
                                                                                  xs, lev4, lev8));
        if (get_sub_group_local_id() == 0u) out[r] = nn__kernels__zzabi_to_half(sum * khronos_opencl2__zzprivate_scale(luts, r), over);
    }
}

/* ⭐ `turboquant_gemv_groups`: every group's rows over one `x`, group `i`'s answers at `out + out_at[i]`. */
__kernel void khronos_opencl2__gemv_groups(ulong out__address, nn__turboquant__groups g, ulong x__address, uint64_t cols,
                                           ulong over__address) {
    __global uint16_t* out = (__global uint16_t*)out__address;
    __global unsigned int* over = (__global unsigned int*)over__address;
    __local uint16_t xs[KHRONOS_OPENCL2__ZZPRIVATE_COLS_MAX];
    __local float lev4[16];
    __local float lev8[256];
    khronos_opencl2__zzprivate_stage(xs, lev4, lev8, (__global const uint16_t*)x__address, cols);
    uint64_t total = 0ul;
    for (uint64_t i = 0ul; i < g.count; ++i) total += g.rows[i];
    for (uint64_t t = KHRONOS_OPENCL2__ZZPRIVATE_FIRST_ROW; t < total; t += KHRONOS_OPENCL2__ZZPRIVATE_ROW_STRIDE) {
        uint64_t i = 0ul, r = t;
        while (r >= g.rows[i]) { r -= g.rows[i]; ++i; }
        const uint32_t d = (uint32_t)g.d[i];
        __global const uint8_t* row = (__global const uint8_t*)g.codes[i] + r * (cols * d / 8u);
        const float sum = sub_group_reduce_add(khronos_opencl2__zzprivate_row_dot(d, row, cols, xs, lev4, lev8));
        if (get_sub_group_local_id() == 0u)
            out[g.out_at[i] + r] = nn__kernels__zzabi_to_half(sum * khronos_opencl2__zzprivate_scale((__global const uint8_t*)g.luts[i], r),
                                                             over);
    }
}

/* ⭐ `turboquant_gemv_groups_sum`: `out[r] = residual[r] + Σ_i w[out_at[i]] · scale_i[r] · (W_i[r] · x_i)`, every group
 * `rows` tall over its own `x_i = x + i · cols`. The `x_i` are read from global memory — they are the groups' together, up
 * to twelve of them — and a lane walks the (group, block) pairs. */
__kernel void khronos_opencl2__gemv_groups_sum(ulong out__address, nn__turboquant__groups g, ulong x__address,
                                               ulong w__address, ulong residual__address, uint64_t rows, uint64_t cols,
                                               ulong over__address) {
    __global uint16_t* out = (__global uint16_t*)out__address;
    __global const uint16_t* x = (__global const uint16_t*)x__address;
    __global const uint16_t* w = (__global const uint16_t*)w__address;
    __global const uint16_t* residual = (__global const uint16_t*)residual__address;
    __global unsigned int* over = (__global unsigned int*)over__address;
    __local float lev4[16];
    __local float lev8[256];
    const uint32_t lid = (uint32_t)get_local_id(0), lsz = (uint32_t)get_local_size(0);
    const uint16_t* t4 = nn__turboquant__zzabi_levels(4u);
    const uint16_t* t8 = nn__turboquant__zzabi_levels(8u);
    for (uint32_t i = lid; i < 16u; i += lsz) lev4[i] = nn__silicon__half_to_float(t4[i]);
    for (uint32_t i = lid; i < 256u; i += lsz) lev8[i] = nn__silicon__half_to_float(t8[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint32_t lane = (uint32_t)get_sub_group_local_id(), lanes = (uint32_t)get_sub_group_size();
    const uint64_t blocks = cols / 32u, spread = g.count * blocks;
    for (uint64_t r = KHRONOS_OPENCL2__ZZPRIVATE_FIRST_ROW; r < rows; r += KHRONOS_OPENCL2__ZZPRIVATE_ROW_STRIDE) {
        float part = 0.0f;
        for (uint64_t j = lane; j < spread; j += lanes) {
            const uint64_t i = j / blocks, b = j - i * blocks;
            const uint32_t d = (uint32_t)g.d[i];
            __global const uint8_t* row = (__global const uint8_t*)g.codes[i] + r * (cols * d / 8u);
            __global const uint16_t* xb = x + i * cols + b * 32u;
            float dot = 0.0f;
            if (d == 4u) {
                __global const ulong* wd = (__global const ulong*)(row + b * 16u);
                for (uint32_t h = 0u; h < 2u; ++h) {
                    const ulong word = wd[h];
                    for (uint32_t k = 0u; k < 16u; ++k)
                        dot += lev4[(uint32_t)(word >> (4u * k)) & 0xFu] * nn__silicon__half_to_float(xb[16u * h + k]);
                }
            } else {
                __global const ulong* wd = (__global const ulong*)(row + b * 32u);
                for (uint32_t h = 0u; h < 4u; ++h) {
                    const ulong word = wd[h];
                    for (uint32_t k = 0u; k < 8u; ++k)
                        dot += lev8[(uint32_t)(word >> (8u * k)) & 0xFFu] * nn__silicon__half_to_float(xb[8u * h + k]);
                }
            }
            part += dot * khronos_opencl2__zzprivate_scale((__global const uint8_t*)g.luts[i], r)
                  * nn__silicon__half_to_float(w[g.out_at[i]]);
        }
        const float sum = sub_group_reduce_add(part);
        if (lane == 0u) out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(residual[r]) + sum, over);
    }
}

/* ⭐ `hadamard_rotate` over whole 512-blocks, `out` apart from `in`: `out = H·(S ⊙ in) / √512` a block, the signs afresh
 * at each. A work-group a block; each of the nine stages pairs elements `h` apart in local memory, a barrier between. */
__kernel void khronos_opencl2__rotate(ulong out__address, ulong in__address, ulong sign__address, uint64_t blocks,
                                      ulong over__address) {
    __global uint16_t* out = (__global uint16_t*)out__address;
    __global const uint16_t* in = (__global const uint16_t*)in__address;
    __global const uint16_t* sign = (__global const uint16_t*)sign__address;
    __global unsigned int* over = (__global unsigned int*)over__address;
    __local float v[KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK];
    const uint32_t lid = (uint32_t)get_local_id(0), lsz = (uint32_t)get_local_size(0);
    const float inv = 1.0f / nn__silicon__sqrtf((float)KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK);
    for (uint64_t blk = get_group_id(0); blk < blocks; blk += get_num_groups(0)) {
        const uint64_t at = blk * KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK;
        for (uint32_t e = lid; e < KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK; e += lsz)
            v[e] = nn__silicon__half_to_float(in[at + e]) * nn__silicon__half_to_float(sign[e]);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint32_t h = 1u; h < KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK; h <<= 1) {
            for (uint32_t k = lid; k < KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK / 2u; k += lsz) {
                const uint32_t i = (k / h) * 2u * h + (k % h), j = i + h;
                const float a = v[i], b = v[j];
                v[i] = a + b;
                v[j] = a - b;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (uint32_t e = lid; e < KHRONOS_OPENCL2__ZZPRIVATE_ROT_BLOCK; e += lsz)
            out[at + e] = nn__kernels__zzabi_to_half(v[e] * inv, over);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/* ⭐ ATTENTION'S TWO RESIDUAL PASSES — nn's generic bodies give a lane a position and sum its whole dot product alone
 *   (the scores), and a lane an element of the head over every position (the mix), a work-group a query head. Here:
 *   · the scores: a SUB-GROUP a position, four in flight, its lanes splitting the head's dimensions, the parts summed
 *     by `sub_group_reduce_add`, which hands every lane the score; the lane that will exponentiate a position writes it,
 *     so no lane reads another's write. The largest score and the sum are the work-group's reductions, as nn's fold is.
 *   · the mix: a lane an element of the head over every position, eight in flight, so a sub-group reads a value row
 *     together and each lane owns what it sums.
 *   nn's products and fp32 sums; only the order of addition differs. Any head width; a sub-group's width is not assumed. */
__kernel void khronos_opencl2__attention_weights(ulong p__address, ulong res__address, ulong q__address, ulong k__address,
                                                 uint64_t q_heads, uint64_t kv_heads, uint64_t D, uint64_t length) {
    __global float* p = (__global float*)p__address;
    __global float* res = (__global float*)res__address;
    __global const uint16_t* q = (__global const uint16_t*)q__address;
    __global const uint16_t* k = (__global const uint16_t*)k__address;
    const uint32_t lane = (uint32_t)get_sub_group_local_id(), lanes = (uint32_t)get_sub_group_size();
    const uint32_t sg = (uint32_t)get_sub_group_id(), sgs = (uint32_t)get_num_sub_groups();
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    const float scale = 1.0f / sqrt((float)D);
    for (uint64_t h = get_group_id(0); h < q_heads; h += get_num_groups(0)) {
        __global const uint16_t* qh = q + h * D;
        __global const uint16_t* kh = k + (h / group) * D;
        __global float* ph = p + h * length;
        float top = -INFINITY;
        /* the sub-group's k-th position is `sg + k·sgs`; lane `k % lanes` writes its score and later exponentiates it */
        for (uint64_t k0 = 0ul; sg + k0 * sgs < length; k0 += 4ul) {          /* four positions in flight */
            float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint64_t e = lane; e < D; e += lanes) {
                const float qe = nn__silicon__half_to_float(qh[e]);
                for (uint32_t u = 0u; u < 4u; ++u) {
                    const uint64_t pos = sg + (k0 + u) * sgs;
                    if (pos < length) part[u] += qe * nn__silicon__half_to_float(kh[pos * row + e]);
                }
            }
            for (uint32_t u = 0u; u < 4u; ++u) {
                const uint64_t pos = sg + (k0 + u) * sgs;
                const float s = sub_group_reduce_add(part[u]) * scale;
                if (pos < length) {
                    if (lane == (uint32_t)((k0 + u) % lanes)) ph[pos] = s;
                    top = s > top ? s : top;
                }
            }
        }
        top = work_group_reduce_max(top);
        float sum = 0.0f;
        for (uint64_t k = lane; sg + k * sgs < length; k += lanes) {
            const uint64_t pos = sg + k * sgs;
            const float e = exp(ph[pos] - top);
            ph[pos] = e;
            sum += e;
        }
        sum = work_group_reduce_add(sum);
        if (get_local_id(0) == 0u) {
            res[h * (D + 2ul)] = top;
            res[h * (D + 2ul) + 1ul] = sum;
        }
    }
}

__kernel void khronos_opencl2__attention_mix(ulong res__address, ulong p__address, ulong v__address, uint64_t q_heads,
                                             uint64_t kv_heads, uint64_t D, uint64_t length) {
    __global float* res = (__global float*)res__address;
    __global const float* p = (__global const float*)p__address;
    __global const uint16_t* v = (__global const uint16_t*)v__address;
    const uint64_t t = get_local_id(0), lanes = get_local_size(0);
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    for (uint64_t h = get_group_id(0); h < q_heads; h += get_num_groups(0)) {
        __global const float* ph = p + h * length;
        __global const uint16_t* vh = v + (h / group) * D;
        for (uint64_t e = t; e < D; e += lanes) {
            float acc = 0.0f;
            uint64_t pos = 0ul;
            for (; pos + 8ul <= length; pos += 8ul) {
                float w[8], x[8];
                for (uint32_t u = 0u; u < 8u; ++u) { w[u] = ph[pos + u]; x[u] = nn__silicon__half_to_float(vh[(pos + u) * row + e]); }
                for (uint32_t u = 0u; u < 8u; ++u) acc += w[u] * x[u];
            }
            for (; pos < length; ++pos) acc += ph[pos] * nn__silicon__half_to_float(vh[pos * row + e]);
            res[h * (D + 2ul) + 2ul + e] = acc;
        }
    }
}

#endif /* __OPENCL_C_VERSION__ && cl_khr_subgroups */
#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_KERNELS_CUH */
