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

/* ⭐ THE `k` LARGEST OF A LONG VECTOR — a vocabulary's logits, for a sampler — amd_rocm6_wave64's `top_k_long` in shape,
 *   one work-group alone. nn's generic body is `k` rounds, each reading every value and asking it whether an earlier round
 *   took it: 20 rounds over 248,320 logits cost the MI50 ~50 ms a token. Here two passes. Every value is a WORD that
 *   orders as nn's rounds pick — its high half the value's order (the two zeros one), its low half the index taken from
 *   all ones, so of equal values the first is the larger word; a NaN is 0 — and the words are distinct.
 *   ① each lane's largest word; the `k`-th largest of the lanes' is a BAR the `k`-th pick reaches — the `k` lanes' words
 *     at or above it are `k` distinct values' words;
 *   ② every word at or above the bar, into local memory; a value below the bar's is passed at once.
 *   Each held word's place is then the count of held words above it, and the first `k` places are nn's `k` rounds' picks,
 *   in its order. Each weight is nn's expression over a sum added in the rounds' order, over the values read from `x`,
 *   so the indices and the weights are nn's to the bit.
 *   ⛳ MORE THAN `HELD` WORDS AT THE BAR — fewer than `k` lanes holding any valid value, and many valid values among them —
 *   and the work-group runs nn's own rounds instead, so the answer never depends on the input, only the time.
 *   ⛳ ONE WORK-GROUP, WAITING FOR ITSELF: the lanes' words, the bar, the held words, the picks' values. No other group. */
#define KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_LANES 256u     /* the host's `KHRONOS_OPENCL2__ZZPRIVATE_GROUP` */
#define KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_HELD  1024u    /* words the work-group holds at or above the bar */
#define KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_MOST  64u      /* nn's `NN__VECTOR__TOP_K_MAX`: this text joins nn's program after its macros */
/* A half's word: its order above, the index taken from all ones below; 0 for a NaN, which no round takes. */
static inline ulong khronos_opencl2__zzprivate_top_k_word(uint h, uint i) {
    const uint z = (h & 0x7FFFu) == 0u ? 0u : h;
    const uint order = (z & 0x8000u) != 0u ? (~z & 0xFFFFu) : (z | 0x8000u);
    return (h & 0x7FFFu) > 0x7C00u ? 0ul : ((ulong)order << 32) | (ulong)(0xFFFFFFFFu - i);
}
/* A word's value. */
static inline float khronos_opencl2__zzprivate_top_k_value(ulong word) {
    const uint order = (uint)(word >> 32);
    return nn__silicon__half_to_float((uint16_t)((order & 0x8000u) != 0u ? (order & 0x7FFFu) : (~order & 0xFFFFu)));
}
__kernel void khronos_opencl2__top_k_long(ulong values__address, uint64_t stride, ulong weights__address, ulong x__address,
                                          uint64_t n, uint64_t k) {
    __global uint64_t* values = (__global uint64_t*)values__address;
    __global uint16_t* weights = (__global uint16_t*)weights__address;
    __global const uint16_t* x = (__global const uint16_t*)x__address;
    __local ulong lane_word[KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_LANES];
    __local ulong held[KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_HELD];
    __local float picked[KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_MOST];
    __local ulong bar;
    __local uint count;
    const uint t = (uint)get_local_id(0), lanes = (uint)get_local_size(0), m = (uint)n;   /* n is under 2^24 here */
    /* ① the lane's largest word */
    ulong mine = 0ul;
    for (uint i = t; i < m; i += lanes) {
        const ulong word = khronos_opencl2__zzprivate_top_k_word(x[i], i);
        mine = word > mine ? word : mine;
    }
    lane_word[t] = mine;
    if (t == 0u) { count = 0u; bar = 0ul; }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* the bar: the `k`-th largest of the lanes' words — distinct, so exactly one lane has `k - 1` above it — or 0, every
     * valid value, where fewer than `k` lanes hold one */
    uint above = 0u;
    for (uint j = 0u; j < lanes; ++j) above += lane_word[j] > mine ? 1u : 0u;
    if (mine != 0ul && above == (uint)k - 1u) bar = mine;
    barrier(CLK_LOCAL_MEM_FENCE);
    const ulong floor_word = bar;
    const float floor_v = floor_word != 0ul ? khronos_opencl2__zzprivate_top_k_value(floor_word) : -INFINITY;
    /* ② every word at or above the bar: a value below the bar's (or a NaN) passed at once */
    for (uint i = t; i < m; i += lanes) {
        if (!(nn__silicon__half_to_float(x[i]) >= floor_v)) continue;
        const ulong word = khronos_opencl2__zzprivate_top_k_word(x[i], i);
        if (word < floor_word) continue;
        const uint at = atomic_inc(&count);
        if (at < KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_HELD) held[at] = word;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint c = count;
    if (c > KHRONOS_OPENCL2__ZZPRIVATE_TOP_K_HELD) {                     /* the same on every lane: nn's rounds */
        nn__vector__zzabi_body_top_k(values, stride, weights, x, n, k);
        return;
    }
    /* each held word's place among them; the first `k` places are the picks */
    for (uint j = t; j < c; j += lanes) {
        const ulong word = held[j];
        uint place = 0u;
        for (uint q = 0u; q < c; ++q) place += held[q] > word ? 1u : 0u;
        if (place < (uint)k) {
            const uint at = 0xFFFFFFFFu - (uint)word;
            values[place * stride] = (uint64_t)at;
            picked[place] = nn__silicon__half_to_float(x[at]);
        }
    }
    for (uint r = c + t; r < (uint)k; r += lanes) {                      /* fewer valid values than picks: nn's none */
        values[r * stride] = n;
        picked[r] = -INFINITY;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < (uint)k) {
        float sum = 0.0f;
        for (uint r = 0u; r < (uint)k; ++r) sum += nn__silicon__expf(picked[r] - picked[0]);
        unsigned int unreported = 0u;                                    /* nn's top_k raises no overflow either */
        weights[t] = nn__kernels__zzabi_to_half(nn__silicon__expf(picked[t] - picked[0]) / sum, &unreported);
    }
}

#endif /* __OPENCL_C_VERSION__ && cl_khr_subgroups */

/* ══ ⭐⭐ THE EXPERT-MAJOR GEMM IN TILES — a prompt's rows against a matrix, each weight and each row of `x` read once a tile ═
 * The ROCm family's tile, in OpenCL C: a work-group takes 64 matrix rows by `TP` pairs and walks the columns 32 at a
 * time, staging the tile's slice of `x` and of the matrix (its codes decoded to levels, once) in local memory, and each
 * lane sums 4 rows by `PT` pairs from there. Two shapes, as there: 64 by 64 for a matrix many rows read (a dense
 * projection over a chunk), 64 by 16 for the few rows a routed expert gets. A launch is up to twelve matrices
 * (`nn__expert__groups`), each its own tiles.
 * ⛳ THE SLICE IS STAGED AS FLOATS, where ROCm's is packed halves: OpenCL 2.0 has no two-halves dot, so a float tile keeps
 *   the inner loop to plain multiply-adds. 32 columns rather than 64 keep the two tiles near 17 KB, so more than one
 *   work-group fits a compute unit (ROCm measured 33 KB a block as one a unit).
 * ⛳ NO SUB-GROUPS, so it is outside the guard above: a device without them gets it too. The arithmetic is the generic
 *   body's: fp32 sums of level · x, the row's scale once, a half rounded once; only the order of the sum differs. */
#if defined(__OPENCL_C_VERSION__)
#define KHRONOS_OPENCL2__ZZPRIVATE_KT  32u       /* the columns a slice stages */
#define KHRONOS_OPENCL2__ZZPRIVATE_XP  68u       /* a staged column of `x`: the tile's pairs, four over so the columns' banks differ */
#define KHRONOS_OPENCL2__ZZPRIVATE_WP  68u       /* a staged column of the matrix: its 64 rows, four over */
#define KHRONOS_OPENCL2__ZZPRIVATE_TR  64u       /* matrix rows a tile */
#define KHRONOS_OPENCL2__ZZPRIVATE_RT  4u        /* of those, a lane's */

static inline void khronos_opencl2__zzprivate_tile(nn__expert__groups g, __global uint16_t* out, __global float* acc,
                                                   __global const uint16_t* x, __global const uint32_t* rows,
                                                   __global const uint16_t* w, uint64_t cols, __global unsigned int* over,
                                                   const uint32_t TP, const uint32_t PT, __local float* ws, __local float* xs,
                                                   __local float* lev4, __local float* lev8) {
    const uint32_t TR = KHRONOS_OPENCL2__ZZPRIVATE_TR, RT = KHRONOS_OPENCL2__ZZPRIVATE_RT;
    const uint32_t KT = KHRONOS_OPENCL2__ZZPRIVATE_KT, XP = KHRONOS_OPENCL2__ZZPRIVATE_XP, WP = KHRONOS_OPENCL2__ZZPRIVATE_WP;
    const uint32_t t = (uint32_t)get_local_id(0), n = (uint32_t)get_local_size(0);
    const uint16_t* t4 = nn__turboquant__zzabi_levels(4u);
    const uint16_t* t8 = nn__turboquant__zzabi_levels(8u);
    for (uint32_t e = t; e < 256u; e += n) { lev8[e] = nn__silicon__half_to_float(t8[e]); if (e < 16u) lev4[e] = nn__silicon__half_to_float(t4[e]); }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* where each matrix's tiles start among the launch's */
    uint64_t first[sizeof g.pairs / sizeof g.pairs[0] + 1u];
    first[0] = 0ul;
    for (uint64_t i = 0ul; i < g.count; ++i)
        first[i + 1u] = first[i] + ((g.out_rows[i] + TR - 1u) / TR) * ((g.pairs[i] + TP - 1u) / TP);
    const uint32_t tr = (t / (TP / PT)) * RT, tp = (t % (TP / PT)) * PT;
    for (uint64_t b = get_group_id(0); b < first[g.count]; b += get_num_groups(0)) {
        uint64_t i = 0ul;
        while (b >= first[i + 1u]) ++i;
        const uint64_t tiles_p = (g.pairs[i] + TP - 1u) / TP, within = b - first[i];
        const uint64_t r0 = (within / tiles_p) * TR, p0 = (within % tiles_p) * TP;
        const uint32_t d = (uint32_t)g.d[i], per = 16u / d, mask = (1u << d) - 1u;
        const uint64_t row_bytes = cols * d / 8u;
        __global const uint8_t* codes = (__global const uint8_t*)g.codes[i];
        __local const float* lev = d == 4u ? lev4 : lev8;
        __global const uint32_t* pr = rows + 2u * g.pairs_at[i];
        float sum[4][4];
        for (uint32_t a = 0u; a < RT; ++a)
            for (uint32_t q = 0u; q < PT; ++q) sum[a][q] = 0.0f;
        for (uint64_t k0 = 0ul; k0 < cols; k0 += KT) {
            /* `x` two halves a read, converted by the device's own instruction */
            for (uint32_t e = t; e < TP * (KT / 2u); e += n) {
                const uint32_t p = e / (KT / 2u), c = 2u * (e % (KT / 2u));
                const float2 v = p0 + p < g.pairs[i]
                    ? vload_half2(0, (__global const half*)(x + (uint64_t)pr[2u * (p0 + p)] * cols + k0 + c)) : (float2)(0.0f, 0.0f);
                xs[c * XP + p] = v.x;
                xs[(c + 1u) * XP + p] = v.y;
            }
            /* a lane a 16-bit unit of codes — `per` columns of one row — read once and decoded into all of them */
            const uint32_t units = KT / per;
            for (uint32_t e = t; e < TR * units; e += n) {
                const uint32_t r = e / units, u = e % units;
                const uint32_t unit = r0 + r < g.out_rows[i]
                    ? ((__global const uint16_t*)(codes + (r0 + r) * row_bytes))[k0 / per + u] : 0u;
                const bool live = r0 + r < g.out_rows[i];
                for (uint32_t j = 0u; j < per; ++j)
                    ws[(u * per + j) * WP + r] = live ? lev[(unit >> (j * d)) & mask] : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            /* the slices are stored a column at a time, so a lane's four rows and its pairs are each one vector read */
            for (uint32_t c = 0u; c < KT; ++c) {
                const float4 wv = vload4(0, ws + c * WP + tr);
                float xv[4];
                if (PT == 4u) { const float4 x4 = vload4(0, xs + c * XP + tp); xv[0] = x4.x; xv[1] = x4.y; xv[2] = x4.z; xv[3] = x4.w; }
                else xv[0] = xs[c * XP + tp];
                for (uint32_t q = 0u; q < PT; ++q) {
                    sum[0][q] = fma(wv.x, xv[q], sum[0][q]);
                    sum[1][q] = fma(wv.y, xv[q], sum[1][q]);
                    sum[2][q] = fma(wv.z, xv[q], sum[2][q]);
                    sum[3][q] = fma(wv.w, xv[q], sum[3][q]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        __global const uint8_t* lut = (__global const uint8_t*)g.luts[i];
        for (uint32_t a = 0u; a < RT; ++a) {
            const uint64_t r = r0 + tr + a;
            if (r >= g.out_rows[i]) continue;
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
            for (uint32_t q = 0u; q < PT; ++q) {
                const uint64_t p = p0 + tp + q;
                if (p >= g.pairs[i]) continue;
                const uint64_t at = (uint64_t)pr[2u * p + 1u] * g.out_rows[i] + r;
                const float v = sum[a][q] * scale;
                if (out != 0) (out + g.out_at[i])[at] = nn__kernels__zzabi_to_half(v, over);
                else acc[at] += v * nn__silicon__half_to_float(w[pr[2u * p]]);
            }
        }
    }
}

/* The two shapes: 64 matrix rows by 64 pairs, and by 16 for a routed expert's few. */
__kernel void khronos_opencl2__tile_wide(nn__expert__groups g, ulong out__address, ulong acc__address, ulong x__address,
                                         ulong rows__address, ulong w__address, uint64_t cols, ulong over__address) {
    __local float ws[KHRONOS_OPENCL2__ZZPRIVATE_KT * KHRONOS_OPENCL2__ZZPRIVATE_WP];
    __local float xs[KHRONOS_OPENCL2__ZZPRIVATE_KT * KHRONOS_OPENCL2__ZZPRIVATE_XP];
    __local float lev4[16];
    __local float lev8[256];
    khronos_opencl2__zzprivate_tile(g, (__global uint16_t*)out__address, (__global float*)acc__address,
                                    (__global const uint16_t*)x__address, (__global const uint32_t*)rows__address,
                                    (__global const uint16_t*)w__address, cols, (__global unsigned int*)over__address,
                                    64u, 4u, ws, xs, lev4, lev8);
}
__kernel void khronos_opencl2__tile_few(nn__expert__groups g, ulong out__address, ulong acc__address, ulong x__address,
                                        ulong rows__address, ulong w__address, uint64_t cols, ulong over__address) {
    __local float ws[KHRONOS_OPENCL2__ZZPRIVATE_KT * KHRONOS_OPENCL2__ZZPRIVATE_WP];
    __local float xs[KHRONOS_OPENCL2__ZZPRIVATE_KT * KHRONOS_OPENCL2__ZZPRIVATE_XP];
    __local float lev4[16];
    __local float lev8[256];
    khronos_opencl2__zzprivate_tile(g, (__global uint16_t*)out__address, (__global float*)acc__address,
                                    (__global const uint16_t*)x__address, (__global const uint32_t*)rows__address,
                                    (__global const uint16_t*)w__address, cols, (__global unsigned int*)over__address,
                                    16u, 1u, ws, xs, lev4, lev8);
}
#endif /* __OPENCL_C_VERSION__ */

/* ══ ⭐⭐ A PROMPT'S ROWS THROUGH A LATENT'S ATTENTION AND KDA — nn's KERNELS' OUTPUTS TO THE BIT ═══════════════════════
 * The doors a prompt read as rows brings (▶ the ROCm family's `kda.cuh`, `attention.cuh` and `index.cuh`), in OpenCL C.
 * ⭐ EACH OUTPUT IS nn's GENERIC KERNEL's, BIT FOR BIT, ON THE SAME DEVICE: the same products added in the same order as
 *   nn's body, and every sum across a work-group a `work_group_reduce_add` — nn's `lanes_sum` in this family — over the
 *   same lanes holding the same parts (or over fewer, where nn's door gives the rest zeros). What is the family's own is
 *   what a lane keeps between rows, what a work-group stages in local memory and how many outputs a work-group makes:
 *   · KDA's steps — a lane's column of the state in its registers for every row, the row's `q`, `k` and decays staged
 *   · MLA's absorb — `R` rows' queries of a head staged, a weight loaded once for them
 *   · MLA's expand — eight weight rows in the lanes' registers for `R` rows
 *   · the latent's grouped scores — 64 heads by 64 keys a work-group, the width staged a slice at a time
 *   · the mix over rows of their own — `Q` heads' weights staged, a value loaded once for them
 *   · the indexer's pool scores — `G` pools of a row a work-group, a query's element loaded once for them
 * ⛳ TWO SHAPES, AN INTEGRATED GPU'S AND A DISCRETE CARD'S, where a shape is a choice (▶ khronos_opencl1_igpu's): an
 *   integrated GPU shares the host's memory and its bandwidth, so each load serves twice as many outputs there. `nn.cuh`
 *   chooses.
 * ⛳ NO SUB-GROUPS: work-group functions and local memory are OpenCL C 2.0's own, so these are outside the guard above. */
#if defined(__OPENCL_C_VERSION__)

/* A helper whose constant arguments — `R` rows, `Q` heads, `G` pools — size its arrays, inlined into each kernel so they are
 * registers: left to the compiler, expand's took 135 registers and 12 bytes of scratch a lane against 55 and none
 * (`MEASURED`, ROCm's clang for gfx906, `-Rpass-analysis=kernel-resource-usage`), and the indexer's choice 32 bytes. */
#define KHRONOS_OPENCL2__ZZPRIVATE_INLINED __attribute__((always_inline))

/* ⭐ KDA's STEPS OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `kda_steps_decayed`: a work-group a head and a lane a column of
 *   its state, held in the lane's registers for every row of the call and written once at the end. The row's `q`, `k` and
 *   decays, the same for every lane, are staged in local memory, two rows' places taking turns so one barrier a row keeps
 *   a row's reads ahead of the next row's writes. nn's door gives a head 256 lanes, the upper 128 adding zeros to each
 *   fold. */
__kernel void khronos_opencl2__kda_steps(ulong S__address, ulong conved__address, ulong eg__address, ulong b__address,
                                         ulong gate__address, ulong w__address, ulong out__address, uint64_t heads, float eps,
                                         uint64_t rows, ulong over__address) {
    __global float* S = (__global float*)S__address;
    __global const uint16_t* conved = (__global const uint16_t*)conved__address;
    __global const float* eg = (__global const float*)eg__address;
    __global const uint16_t* b = (__global const uint16_t*)b__address;
    __global const uint16_t* gate = (__global const uint16_t*)gate__address;
    __global const uint16_t* w = (__global const uint16_t*)w__address;
    __global uint16_t* out = (__global uint16_t*)out__address;
    __global unsigned int* over = (__global unsigned int*)over__address;
    __local float qs[2][128], ks[2][128], es[2][128];
    const uint32_t D = 128u, j = (uint32_t)get_local_id(0);
    const uint64_t width = heads * D;
    __global const uint16_t* qa = conved;
    __global const uint16_t* ka = conved + rows * width;
    __global const uint16_t* va = conved + 2ul * rows * width;
    for (uint64_t h = get_group_id(0); h < heads; h += get_num_groups(0)) {
        __global float* Sh = S + h * D * D;
        float col[128];
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) col[i] = Sh[i * D + j];
        for (uint64_t t = 0ul; t < rows; ++t) {
            const uint64_t r = t * width + h * D;
            const uint32_t at = (uint32_t)(t & 1ul);
            const float qi = nn__silicon__half_to_float(qa[r + j]), ki = nn__silicon__half_to_float(ka[r + j]);
            qs[at][j] = qi; ks[at][j] = ki; es[at][j] = eg[r + j];
            barrier(CLK_LOCAL_MEM_FENCE);
            const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[t * heads + h])));
            float qq = 0.0f, kk = 0.0f;
            qq += qi * qi;
            kk += ki * ki;
            const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)D);
            const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
            float kv = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) {
                col[i] *= es[at][i];
                kv += col[i] * ks[at][i] * inv_k;
            }
            const float delta = (nn__silicon__half_to_float(va[r + j]) - kv) * beta;
            /* ⛳ the stage is read again below rather than kept from the loop above: kept, `k` takes registers beside the
             *   column's 128 and the kernel spills inside its loop over the rows */
            mem_fence(CLK_LOCAL_MEM_FENCE);
            float o = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) {
                const float s = col[i] + ks[at][i] * inv_k * delta;
                col[i] = s;
                o += s * qs[at][i];
            }
            const float oj = nn__silicon__half_to_float(nn__kernels__zzabi_to_half(o * inv_q, over));   /* rounded, as nn's readout is */
            float oo = 0.0f;
            oo += oj * oj;
            const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)D + eps);
            const float gz = nn__silicon__half_to_float(gate[r + j]);
            out[r + j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) / (1.0f + nn__silicon__expf(-gz)), over);
        }
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) Sh[i * D + j] = col[i];
    }
}

/* ⭐ MLA's ABSORB — nn's `absorb` and `absorb_rows`: a work-group a head's `lanes` latent columns over `R` rows, a lane a
 *   column, its sums `Σᵢ W[i][c] · q[i]`, `i` rising, as nn's lane adds them; the `R` rows' queries of the head staged in
 *   local memory, so every weight loaded serves `R` rows. */
#define KHRONOS_OPENCL2__ZZPRIVATE_ABSORB_STAGE 4096u    /* the query elements a work-group stages: `R` rows of `nope` */
static inline KHRONOS_OPENCL2__ZZPRIVATE_INLINED void khronos_opencl2__zzprivate_absorb(__global uint16_t* out,
                                                                                        __global const uint16_t* q,
                                                                                        __global const uint16_t* W, uint64_t heads,
                                                                                        uint64_t nope, uint64_t latent,
                                                                                        uint64_t stride, float scale, uint64_t rows,
                                                                                        __global unsigned int* over,
                                                                                        const uint32_t R, __local float* qs) {
    const uint32_t lane = (uint32_t)get_local_id(0), n = (uint32_t)get_local_size(0);
    const uint64_t cgroups = (latent + n - 1u) / n, hc_all = heads * cgroups, items = hc_all * ((rows + R - 1u) / R);
    for (uint64_t x = get_group_id(0); x < items; x += get_num_groups(0)) {
        const uint64_t hc = x % hc_all, t0 = (x / hc_all) * R, h = hc / cgroups, c = (hc % cgroups) * n + lane;
        barrier(CLK_LOCAL_MEM_FENCE);                                  /* the item before has read its stage */
        for (uint64_t e = lane; e < (uint64_t)R * nope; e += n) {
            const uint64_t r = e / nope, i = e - r * nope;
            qs[e] = t0 + r < rows ? nn__silicon__half_to_float(q[(t0 + r) * heads * nope + h * nope + i]) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (c >= latent) continue;
        __global const uint16_t* wc = W + h * stride * latent + c;
        float sum[16];
#pragma unroll
        for (uint32_t r = 0u; r < R; ++r) sum[r] = 0.0f;
        for (uint64_t i = 0u; i < nope; ++i) {
            const float wf = nn__silicon__half_to_float(wc[i * latent]);
#pragma unroll
            for (uint32_t r = 0u; r < R; ++r) sum[r] += wf * qs[r * nope + i];
        }
#pragma unroll
        for (uint32_t r = 0u; r < R; ++r)
            if (t0 + r < rows) out[(t0 + r) * heads * latent + h * latent + c] = nn__kernels__zzabi_to_half(sum[r] * scale, over);
    }
}
__kernel void khronos_opencl2__absorb1(ulong out__address, ulong q__address, ulong W__address, uint64_t heads, uint64_t nope,
                                       uint64_t latent, uint64_t stride, float scale, uint64_t rows, ulong over__address) {
    __local float qs[KHRONOS_OPENCL2__ZZPRIVATE_ABSORB_STAGE];
    khronos_opencl2__zzprivate_absorb((__global uint16_t*)out__address, (__global const uint16_t*)q__address,
                                      (__global const uint16_t*)W__address, heads, nope, latent, stride, scale, rows,
                                      (__global unsigned int*)over__address, 1u, qs);
}
__kernel void khronos_opencl2__absorb8(ulong out__address, ulong q__address, ulong W__address, uint64_t heads, uint64_t nope,
                                       uint64_t latent, uint64_t stride, float scale, uint64_t rows, ulong over__address) {
    __local float qs[KHRONOS_OPENCL2__ZZPRIVATE_ABSORB_STAGE];
    khronos_opencl2__zzprivate_absorb((__global uint16_t*)out__address, (__global const uint16_t*)q__address,
                                      (__global const uint16_t*)W__address, heads, nope, latent, stride, scale, rows,
                                      (__global unsigned int*)over__address, 8u, qs);
}
__kernel void khronos_opencl2__absorb16(ulong out__address, ulong q__address, ulong W__address, uint64_t heads, uint64_t nope,
                                       uint64_t latent, uint64_t stride, float scale, uint64_t rows, ulong over__address) {
    __local float qs[KHRONOS_OPENCL2__ZZPRIVATE_ABSORB_STAGE];
    khronos_opencl2__zzprivate_absorb((__global uint16_t*)out__address, (__global const uint16_t*)q__address,
                                      (__global const uint16_t*)W__address, heads, nope, latent, stride, scale, rows,
                                      (__global unsigned int*)over__address, 16u, qs);
}

/* ⭐ MLA's EXPAND OVER A LATENT OF AT MOST 512 — nn's `expand` and `expand_rows`: a work-group of 256 eight of a head's outputs
 *   over `R` rows, lane `L` each output's part over its columns `L` and `L + 256` as nn's lane makes it, and the eight parts
 *   folded over the work-group as nn folds each — the eight weight rows loaded once for the `R` rows. */
static inline KHRONOS_OPENCL2__ZZPRIVATE_INLINED void khronos_opencl2__zzprivate_expand(__global uint16_t* out,
                                                                                        __global const uint16_t* o,
                                                                                        __global const uint16_t* W, uint64_t heads,
                                                                                        uint64_t width, uint64_t latent,
                                                                                        uint64_t stride, uint64_t offset,
                                                                                        uint64_t rows, __global unsigned int* over,
                                                                                        const uint32_t R) {
    const uint32_t J = 8u, L = (uint32_t)get_local_id(0), n = (uint32_t)get_local_size(0);
    const uint64_t per = width / J, outs = heads * per, items = outs * ((rows + R - 1u) / R);
    for (uint64_t x = get_group_id(0); x < items; x += get_num_groups(0)) {
        const uint64_t hj = x % outs, t0 = (x / outs) * R, h = hj / per, j0 = (hj % per) * J;
        const uint64_t tn = rows - t0 < R ? rows - t0 : R;
        float wv[8][2];
#pragma unroll
        for (uint32_t jj = 0u; jj < J; ++jj) {
            __global const uint16_t* row = W + (h * stride + offset + j0 + jj) * latent;
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) wv[jj][c] = L + n * c < latent ? nn__silicon__half_to_float(row[L + n * c]) : 0.0f;
        }
        for (uint64_t r = 0u; r < tn; ++r) {
            __global const uint16_t* xr = o + (t0 + r) * heads * latent + h * latent;
            float xv[2];
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) xv[c] = L + n * c < latent ? nn__silicon__half_to_float(xr[L + n * c]) : 0.0f;
            float parts[8];
#pragma unroll
            for (uint32_t jj = 0u; jj < J; ++jj) {
                float part = 0.0f;
#pragma unroll
                for (uint32_t c = 0u; c < 2u; ++c)
                    if (L + n * c < latent) part += wv[jj][c] * xv[c];
                parts[jj] = part;
            }
            nn__silicon__lanes_sums(parts, J);
            if (L == 0u)
#pragma unroll
                for (uint32_t jj = 0u; jj < J; ++jj)
                    out[(t0 + r) * heads * width + h * width + j0 + jj] = nn__kernels__zzabi_to_half(parts[jj], over);
        }
    }
}
__kernel void khronos_opencl2__expand16(ulong out__address, ulong o__address, ulong W__address, uint64_t heads, uint64_t width,
                                        uint64_t latent, uint64_t stride, uint64_t offset, uint64_t rows, ulong over__address) {
    khronos_opencl2__zzprivate_expand((__global uint16_t*)out__address, (__global const uint16_t*)o__address,
                                      (__global const uint16_t*)W__address, heads, width, latent, stride, offset, rows,
                                      (__global unsigned int*)over__address, 16u);
}
__kernel void khronos_opencl2__expand32(ulong out__address, ulong o__address, ulong W__address, uint64_t heads, uint64_t width,
                                        uint64_t latent, uint64_t stride, uint64_t offset, uint64_t rows, ulong over__address) {
    khronos_opencl2__zzprivate_expand((__global uint16_t*)out__address, (__global const uint16_t*)o__address,
                                      (__global const uint16_t*)W__address, heads, width, latent, stride, offset, rows,
                                      (__global unsigned int*)over__address, 32u);
}

/* ⭐ THE GROUPED SCORES IN TILES — nn's `scores_grouped` and `scores_grouped_rows` where a kv head's queries are many: its work
 *   is the GEMM it is, its rows the group's query heads, its columns the key positions, summed over the head's width — 64
 *   heads by 64 positions a work-group of 256, the width 64 columns at a time staged in local memory, a lane 4 heads by 4
 *   positions 16 apart. Each dot is nn's: one chain of `dot2` over the width's pairs in order, then the scale; the keys are
 *   read once for every head of the group, where nn's lanes read them eight positions at a time. Row `u` of the rows form
 *   is the one-row form at its own length. */
#define KHRONOS_OPENCL2__ZZPRIVATE_LT_KT 64u      /* the width a slice stages; its pairs a row of 32, one over for the banks */
static inline KHRONOS_OPENCL2__ZZPRIVATE_INLINED void khronos_opencl2__zzprivate_scores_tile(__global float* p,
                                                                                             __global const uint16_t* q,
                                                                                             __global const uint16_t* k,
                                                                                             uint64_t q_heads, uint64_t kv_heads,
                                                                                             uint64_t head_dim, uint64_t length,
                                                                                             uint64_t g, uint64_t mi, uint64_t ni,
                                                                                             __local uint (*qs)[33],
                                                                                             __local uint (*ks)[33]) {
    const uint32_t t = (uint32_t)get_local_id(0), n = (uint32_t)get_local_size(0);
    const uint32_t tr = (t / 16u) * 4u, tp = t % 16u;
    const uint64_t group = q_heads / kv_heads, krow = kv_heads * head_dim, m0 = mi * 64u, n0 = ni * 64u;
    const float scale = 1.0f / nn__silicon__sqrtf((float)head_dim);
    float sum[4][4];
#pragma unroll
    for (uint32_t a = 0u; a < 4u; ++a)
#pragma unroll
        for (uint32_t c = 0u; c < 4u; ++c) sum[a][c] = 0.0f;
    for (uint64_t k0 = 0u; k0 < head_dim; k0 += KHRONOS_OPENCL2__ZZPRIVATE_LT_KT) {
        /* a slice: 64 query heads and 64 keys, eight halves an item, two items a lane */
        for (uint32_t s = 0u; s < 2u; ++s) {
            const uint32_t i = t + n * s, r = i / 8u, c = (i % 8u) * 4u;
            uint4 qv = (uint4)(0u), kv = (uint4)(0u);
            const uint64_t m = m0 + r, pos = n0 + r;
            if (m < group) qv = vload4(0, (__global const uint*)(q + (g * group + m) * head_dim + k0 + 2u * c));
            if (pos < length) kv = vload4(0, (__global const uint*)(k + pos * krow + g * head_dim + k0 + 2u * c));
            qs[r][c] = qv.x; qs[r][c + 1u] = qv.y; qs[r][c + 2u] = qv.z; qs[r][c + 3u] = qv.w;
            ks[r][c] = kv.x; ks[r][c + 1u] = kv.y; ks[r][c + 2u] = kv.z; ks[r][c + 3u] = kv.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 4
        for (uint32_t c = 0u; c < KHRONOS_OPENCL2__ZZPRIVATE_LT_KT / 2u; ++c) {
            uint qv[4], kv[4];
#pragma unroll
            for (uint32_t a = 0u; a < 4u; ++a) qv[a] = qs[tr + a][c];
#pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) kv[j] = ks[tp + 16u * j][c];
#pragma unroll
            for (uint32_t a = 0u; a < 4u; ++a)
#pragma unroll
                for (uint32_t j = 0u; j < 4u; ++j) sum[a][j] = nn__silicon__dot2(qv[a], kv[j], sum[a][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#pragma unroll
    for (uint32_t a = 0u; a < 4u; ++a) {
        const uint64_t m = m0 + tr + a;
        if (m >= group) continue;
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint64_t pos = n0 + tp + 16u * j;
            if (pos < length) p[(g * group + m) * length + pos] = sum[a][j] * scale;
        }
    }
}
__kernel void khronos_opencl2__latent_scores(ulong p__address, ulong q__address, ulong k__address, uint64_t q_heads, uint64_t kv_heads,
                                             uint64_t head_dim, uint64_t length) {
    __local uint qs[64][33];
    __local uint ks[64][33];
    __global float* p = (__global float*)p__address;
    __global const uint16_t* q = (__global const uint16_t*)q__address;
    __global const uint16_t* k = (__global const uint16_t*)k__address;
    const uint64_t mt = (q_heads / kv_heads + 63u) / 64u, nt = (length + 63u) / 64u;
    for (uint64_t b = get_group_id(0); b < kv_heads * mt * nt; b += get_num_groups(0))
        khronos_opencl2__zzprivate_scores_tile(p, q, k, q_heads, kv_heads, head_dim, length, b / (mt * nt), (b / nt) % mt, b % nt, qs, ks);
}
__kernel void khronos_opencl2__latent_scores_rows(ulong p__address, uint64_t p_row, ulong q__address, uint64_t q_row, ulong k__address,
                                                  uint64_t k_row, ulong lens__address, uint64_t q_heads, uint64_t kv_heads,
                                                  uint64_t head_dim, uint64_t most, uint64_t rows) {
    __local uint qs[64][33];
    __local uint ks[64][33];
    __global float* p = (__global float*)p__address;
    __global const uint16_t* q = (__global const uint16_t*)q__address;
    __global const uint16_t* k = (__global const uint16_t*)k__address;
    __global const uint32_t* lens = (__global const uint32_t*)lens__address;
    const uint64_t mt = (q_heads / kv_heads + 63u) / 64u, nt = (most + 63u) / 64u, per = kv_heads * mt * nt;
    for (uint64_t b = get_group_id(0); b < rows * per; b += get_num_groups(0)) {
        const uint64_t u = b / per, rest = b - u * per, length = (uint64_t)lens[u];
        const uint64_t ni = rest % nt;
        if (ni * 64u >= length) continue;                                    /* the work-group's alike: the row's keys end before it */
        khronos_opencl2__zzprivate_scores_tile(p + u * p_row, q + u * q_row, k + u * k_row, q_heads, kv_heads, head_dim, length,
                                               rest / (mt * nt), (rest / nt) % mt, ni, qs, ks);
    }
}

/* ⭐ THE MIX FOR `rows` QUERIES, EACH OVER VALUES OF ITS OWN — nn's `mix_rows`: a work-group `Q` of a row's query heads of one
 *   kv head by `lanes` of the head's columns, a lane a column, each sum over the row's positions in order as nn's lane adds
 *   it; a value loaded once for the `Q` heads, and their weights staged in local memory `T` positions at a time. */
#define KHRONOS_OPENCL2__ZZPRIVATE_MIX_T 64u
static inline KHRONOS_OPENCL2__ZZPRIVATE_INLINED void khronos_opencl2__zzprivate_mix_rows(__global uint16_t* o, uint64_t o_row,
                                                                                          __global const float* p, uint64_t p_row,
                                                                                          __global const uint16_t* v,
                                                                                          uint64_t v_row,
                                                                                          __global const uint32_t* lens,
                                                                                          uint64_t q_heads, uint64_t kv_heads,
                                                                                          uint64_t head_dim, uint64_t rows,
                                                                                          __global unsigned int* over,
                                                                                          const uint32_t Q, __local float* ps) {
    const uint32_t T = KHRONOS_OPENCL2__ZZPRIVATE_MIX_T, lane = (uint32_t)get_local_id(0), n = (uint32_t)get_local_size(0);
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim;
    const uint64_t q_tiles = (group + Q - 1u) / Q, e_tiles = (head_dim + n - 1u) / n, per = kv_heads * q_tiles * e_tiles;
    for (uint64_t b = get_group_id(0); b < rows * per; b += get_num_groups(0)) {
        const uint64_t u = b / per, rest = b - u * per, et = rest % e_tiles, qt = (rest / e_tiles) % q_tiles, g = rest / (e_tiles * q_tiles);
        const uint64_t length = (uint64_t)lens[u], e = et * n + lane, h0 = g * group + qt * Q;
        const uint64_t qn = group - qt * Q < Q ? group - qt * Q : Q;
        __global const float* pu = p + u * p_row;
        __global const uint16_t* vg = v + u * v_row + g * head_dim + e;
        float sum[16];
#pragma unroll
        for (uint32_t a = 0u; a < Q; ++a) sum[a] = 0.0f;
        for (uint64_t t0 = 0u; t0 < length; t0 += T) {
            const uint64_t span = length - t0 < T ? length - t0 : T;
            barrier(CLK_LOCAL_MEM_FENCE);                              /* the positions before have been read */
            for (uint32_t s = lane; s < Q * T; s += n) {
                const uint32_t a = s / T, i = s % T;
                ps[s] = a < qn && i < span ? pu[(h0 + a) * length + t0 + i] : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (e < head_dim)
                for (uint32_t i = 0u; i < (uint32_t)span; ++i) {
                    const float vf = nn__silicon__half_to_float(vg[(t0 + i) * row]);
#pragma unroll
                    for (uint32_t a = 0u; a < Q; ++a) sum[a] += ps[a * T + i] * vf;
                }
        }
        if (e < head_dim)
            for (uint32_t a = 0u; a < (uint32_t)qn; ++a) o[u * o_row + (h0 + a) * head_dim + e] = nn__kernels__zzabi_to_half(sum[a], over);
    }
}
__kernel void khronos_opencl2__mix_rows8(ulong o__address, uint64_t o_row, ulong p__address, uint64_t p_row, ulong v__address,
                                         uint64_t v_row, ulong lens__address, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                         uint64_t rows, ulong over__address) {
    __local float ps[8 * KHRONOS_OPENCL2__ZZPRIVATE_MIX_T];
    khronos_opencl2__zzprivate_mix_rows((__global uint16_t*)o__address, o_row, (__global const float*)p__address, p_row,
                                        (__global const uint16_t*)v__address, v_row, (__global const uint32_t*)lens__address,
                                        q_heads, kv_heads, head_dim, rows, (__global unsigned int*)over__address, 8u, ps);
}
__kernel void khronos_opencl2__mix_rows16(ulong o__address, uint64_t o_row, ulong p__address, uint64_t p_row, ulong v__address,
                                         uint64_t v_row, ulong lens__address, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                         uint64_t rows, ulong over__address) {
    __local float ps[16 * KHRONOS_OPENCL2__ZZPRIVATE_MIX_T];
    khronos_opencl2__zzprivate_mix_rows((__global uint16_t*)o__address, o_row, (__global const float*)p__address, p_row,
                                        (__global const uint16_t*)v__address, v_row, (__global const uint32_t*)lens__address,
                                        q_heads, kv_heads, head_dim, rows, (__global unsigned int*)over__address, 16u, ps);
}

/* ⭐ THE INDEXER's POOL SCORES — nn's `index_scores` and `index_scores_rows`: a work-group of 64 lanes `G` pools of a row, lane
 *   `h` a head's chain of products over each as nn's lane makes it, a query's element loaded once for the `G`, and each
 *   pool's 64 parts folded as nn folds them. Row `u` is position `len0 + u − 1`, scoring its `(len0 + u) / kpool` finished
 *   pools; the one-row door is one row at `kpool` 1. */
static inline KHRONOS_OPENCL2__ZZPRIVATE_INLINED void khronos_opencl2__zzprivate_index_scores(__global float* scores,
                                                                                              uint64_t scores_row,
                                                                                              __global const uint16_t* q,
                                                                                              __global const uint16_t* w,
                                                                                              __global const uint16_t* pooled,
                                                                                              uint64_t heads, uint64_t dim,
                                                                                              uint64_t len0, uint64_t kpool,
                                                                                              uint64_t rows, float scale,
                                                                                              const uint32_t G) {
    const uint64_t h = get_local_id(0), most = (len0 + rows - 1u) / kpool, groups = (most + G - 1u) / G;
    const float per_head = 1.0f / nn__silicon__sqrtf((float)heads);
    for (uint64_t x = get_group_id(0); x < rows * groups; x += get_num_groups(0)) {
        const uint64_t u = x / groups, base = (x - u * groups) * G, pools = (len0 + u) / kpool;
        if (base >= pools) continue;                                     /* the same for every lane of the work-group */
        const uint64_t here = pools - base < G ? pools - base : G;
        float dot[16];
#pragma unroll
        for (uint32_t i = 0u; i < G; ++i) dot[i] = 0.0f;
        if (h < heads) {
            __global const uint16_t* qh = q + u * heads * dim + h * dim;
            __global const uint16_t* pb = pooled + base * dim;
            for (uint64_t c = 0u; c < dim; ++c) {
                const float qf = nn__silicon__half_to_float(qh[c]);
#pragma unroll
                for (uint32_t i = 0u; i < G; ++i)
                    if (i < here) dot[i] += qf * nn__silicon__half_to_float(pb[i * dim + c]);
            }
        }
        const float wh = h < heads ? nn__silicon__half_to_float(w[u * heads + h]) : 0.0f;
        float parts[16];
#pragma unroll
        for (uint32_t i = 0u; i < G; ++i) {
            float part = 0.0f;
            if (h < heads && i < here) {
                const float d = dot[i] * scale;
                part += (d > 0.0f ? d : 0.0f) * wh * per_head;
            }
            parts[i] = part;
        }
        nn__silicon__lanes_sums(parts, G);
        if (h == 0u)
            for (uint32_t i = 0u; i < (uint32_t)here; ++i) scores[u * scores_row + base + i] = parts[i];
    }
}
__kernel void khronos_opencl2__index_scores8(ulong scores__address, uint64_t scores_row, ulong q__address, ulong w__address,
                                             ulong pooled__address, uint64_t heads, uint64_t dim, uint64_t len0, uint64_t kpool,
                                             uint64_t rows, float scale) {
    khronos_opencl2__zzprivate_index_scores((__global float*)scores__address, scores_row, (__global const uint16_t*)q__address,
                                            (__global const uint16_t*)w__address, (__global const uint16_t*)pooled__address,
                                            heads, dim, len0, kpool, rows, scale, 8u);
}
__kernel void khronos_opencl2__index_scores16(ulong scores__address, uint64_t scores_row, ulong q__address, ulong w__address,
                                             ulong pooled__address, uint64_t heads, uint64_t dim, uint64_t len0, uint64_t kpool,
                                             uint64_t rows, float scale) {
    khronos_opencl2__zzprivate_index_scores((__global float*)scores__address, scores_row, (__global const uint16_t*)q__address,
                                            (__global const uint16_t*)w__address, (__global const uint16_t*)pooled__address,
                                            heads, dim, len0, kpool, rows, scale, 16u);
}

/* ⭐ THE INDEXER's CHOICE — nn's `index_select` and `index_select_rows`: the same pools chosen, their positions written in the
 *   same order. The `k`-th largest key is found by four passes of a 256-bin count in local memory, eight bits a pass from
 *   the top — nn's body counts across the work-group a bit at a time, 32 sums — and the pools above it, then its ties by
 *   lowest index, are chosen by every lane over its own run of pools and placed by scans, where nn's first lane walks the
 *   pools twice alone. Every sum here is of whole numbers, so any order of adding is nn's, and the choice is nn's to the
 *   bit. A work-group of 256, a lane a bin. */
static inline uint32_t khronos_opencl2__zzprivate_key(float f) {
    const uint32_t u = as_uint(f);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}
static inline KHRONOS_OPENCL2__ZZPRIVATE_INLINED void khronos_opencl2__zzprivate_select(__global uint32_t* index,
                                                                                        __global uint32_t* count,
                                                                                        __global const float* scores,
                                                                                        uint64_t pools, uint64_t k, uint64_t kpool,
                                                                                        uint64_t tail_first, uint64_t tail,
                                                                                        __local uint* bins) {
    const uint32_t lane = (uint32_t)get_local_id(0), n = (uint32_t)get_local_size(0);
    const bool all = pools <= k;
    uint32_t cut = 0u;
    if (!all) {
        uint32_t mask = 0u, need = (uint32_t)k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            barrier(CLK_LOCAL_MEM_FENCE);                              /* the pass before has read its bins */
            bins[lane] = 0u;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint64_t p = lane; p < pools; p += n) {
                const uint32_t key = khronos_opencl2__zzprivate_key(scores[p]);
                if ((key & mask) == cut) atomic_inc(&bins[(key >> (uint32_t)shift) & 255u]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            /* lane `L` the bin of digit 255 − L: its scan the keys at that digit or above, and the digit of the `need`-th largest
             * the first lane whose scan reaches `need` */
            const uint32_t here = bins[255u - lane], at_or_above = work_group_scan_inclusive_add(here);
            const uint32_t first = work_group_reduce_min(at_or_above >= need ? lane : n);
            const uint32_t above = work_group_broadcast(at_or_above - here, (size_t)first);
            cut |= (255u - first) << (uint32_t)shift;
            mask |= 255u << (uint32_t)shift;
            need -= above;
        }
    }
    /* a lane's own run of pools, in order */
    const uint64_t per = (pools + n - 1u) / n, p0 = (uint64_t)lane * per < pools ? (uint64_t)lane * per : pools;
    const uint64_t p1 = p0 + per < pools ? p0 + per : pools;
    uint32_t above_here = 0u, ties_here = 0u;
    if (!all)
        for (uint64_t p = p0; p < p1; ++p) {
            const uint32_t key = khronos_opencl2__zzprivate_key(scores[p]);
            above_here += key > cut ? 1u : 0u;
            ties_here += key == cut ? 1u : 0u;
        }
    const uint32_t ties = all ? 0u : (uint32_t)k - work_group_reduce_add(above_here);
    const uint32_t ties_before = work_group_scan_exclusive_add(ties_here);
    const uint32_t ties_mine = ties_before >= ties ? 0u : (ties - ties_before < ties_here ? ties - ties_before : ties_here);
    const uint32_t taken_here = all ? (uint32_t)(p1 - p0) : above_here + ties_mine;
    const uint32_t before = work_group_scan_exclusive_add(taken_here), taken = work_group_reduce_add(taken_here);
    uint64_t at = (uint64_t)before * kpool;
    uint32_t ties_left = ties_mine;
    for (uint64_t p = p0; p < p1; ++p) {
        bool take = all;
        if (!take) {
            const uint32_t key = khronos_opencl2__zzprivate_key(scores[p]);
            if (key > cut) take = true;
            else if (key == cut && ties_left != 0u) { take = true; --ties_left; }
        }
        if (take)
            for (uint64_t i = 0u; i < kpool; ++i) index[at++] = (uint32_t)(p * kpool + i);
    }
    for (uint64_t i = lane; i < tail; i += n) index[(uint64_t)taken * kpool + i] = (uint32_t)(tail_first + i);
    if (lane == 0u) count[0] = (uint32_t)((uint64_t)taken * kpool + tail);
}
__kernel void khronos_opencl2__index_select(ulong index__address, ulong count__address, ulong scores__address, uint64_t pools, uint64_t k,
                                            uint64_t kpool, uint64_t tail_first, uint64_t tail) {
    __local uint bins[256];
    if (get_group_id(0) != 0u) return;
    khronos_opencl2__zzprivate_select((__global uint32_t*)index__address, (__global uint32_t*)count__address,
                                      (__global const float*)scores__address, pools, k, kpool, tail_first, tail, bins);
}
__kernel void khronos_opencl2__index_select_rows(ulong index__address, uint64_t index_row, ulong counts__address, ulong scores__address,
                                                 uint64_t scores_row, uint64_t k, uint64_t kpool, uint64_t len0, uint64_t rows) {
    __local uint bins[256];
    __global uint32_t* index = (__global uint32_t*)index__address;
    __global uint32_t* counts = (__global uint32_t*)counts__address;
    __global const float* scores = (__global const float*)scores__address;
    for (uint64_t u = get_group_id(0); u < rows; u += get_num_groups(0)) {
        const uint64_t len = len0 + u, pools = len / kpool;
        khronos_opencl2__zzprivate_select(index + u * index_row, counts + u, scores + u * scores_row, pools, k, kpool, pools * kpool,
                                          len - pools * kpool, bins);
    }
}

#endif /* __OPENCL_C_VERSION__ */

/* ══ ⭐⭐ A TOKEN'S SMALL DOORS — GLM's AND DeepSeek's: THE STEP, THE SITES' MAPS, THE ROUTER ══════════════════════════
 * The doors a decoded token meets between its GEMVs (▶ khronos_opencl1_igpu's, which these are, in OpenCL C 2.0; the ROCm
 * family's `kda.cuh`, `hyper.cuh` and `vector.cuh`). On the MI50 through nn's generic bodies they were 36 to 126 us each
 * alone (`MEASURED`, test/src_gemv_27b_bench.cpp, BENCH_ATTENTION_ONLY / ROWS_ONLY / DEEPSEEK_ONLY).
 * ⭐ EACH OUTPUT IS nn's GENERIC KERNEL's, BIT FOR BIT, ON THE SAME DEVICE: the same products added in the same order as
 *   nn's body, every sum across a work-group a `work_group_reduce_add` (or `_max`) — nn's `lanes_sum` in this family — over
 *   nn's door's lanes, or over fewer where nn's door gives the rest zeros. What is the family's own:
 *   · KDA's step a row — a lane's column of the state in its registers between the step's two passes
 *   · a site's map's logits — a lane's sixteen reads in flight together
 *   · a site's `pre` at four streams — the mix's Sinkhorn iterations sixteen lanes an entry each
 *   · the scored router — each score made once and kept, where nn's body makes every score again each round
 * ⛳ THREE STAY nn's, no faster here as the other families write them (`MEASURED` on the MI50, back to back): the norm of
 *   a row of halves, a lane's sixteen reads in flight (a row of 5,120 22.9 -> 25.0 us, every width slower), and of floats
 *   (within the noise of nn's door), and `vector_sum`, eight halves a lane (5.6 -> 6.9 us), its parts this device's own.
 * ⛳ NO SUB-GROUPS, so these are outside the guard above: an Intel card or iGPU without `cl_khr_subgroups` gets them too.
 *   nn's macros are gone from the text this joins (it is preprocessed before), so the few these need are written here. */
#if defined(__OPENCL_C_VERSION__)
#define KHRONOS_OPENCL2__ZZPRIVATE_RUN            16u   /* a lane's reads in flight */
#define KHRONOS_OPENCL2__ZZPRIVATE_MULT           4u    /* the streams of a site's `pre` here */
#define KHRONOS_OPENCL2__ZZPRIVATE_SCORE_SIGMOID  1ul   /* nn's `NN__VECTOR__SCORE_SIGMOID` */
#define KHRONOS_OPENCL2__ZZPRIVATE_SCORE_SOFTPLUS 2ul   /* nn's `NN__VECTOR__SCORE_SQRT_SOFTPLUS` */
#define KHRONOS_OPENCL2__ZZPRIVATE_ROUTER_MOST    16u   /* nn's `NN__VECTOR__ROUTER_K_MAX`, which the door holds `k` to */

/* ⭐ KDA's STEP A ROW AT A HEAD OF 128 — nn's `kda_step`: a work-group a head and a lane a column, its column of the state
 *   read once into registers and written once, where nn's body reads and writes it in memory in both passes. Each row's
 *   decay is nn's expression, made by every lane as nn's lane makes it. */
__kernel void khronos_opencl2__kda_step(ulong S__address, ulong conved__address, ulong f__address, ulong b__address, ulong dt__address,
                                        ulong a_log__address, ulong gate__address, ulong w__address, ulong out__address, uint64_t heads,
                                        float lower, float eps, ulong over__address) {
    __global float* S = (__global float*)S__address;
    __global const uint16_t* conved = (__global const uint16_t*)conved__address;
    __global const uint16_t* f = (__global const uint16_t*)f__address;
    __global const uint16_t* b = (__global const uint16_t*)b__address;
    __global const uint16_t* dt = (__global const uint16_t*)dt__address;
    __global const uint16_t* a_log = (__global const uint16_t*)a_log__address;
    __global const uint16_t* gate = (__global const uint16_t*)gate__address;
    __global const uint16_t* w = (__global const uint16_t*)w__address;
    __global uint16_t* out = (__global uint16_t*)out__address;
    __global unsigned int* over = (__global unsigned int*)over__address;
    const uint32_t D = 128u, j = (uint32_t)get_local_id(0);
    const uint64_t width = heads * D;
    for (uint64_t h = get_group_id(0); h < heads; h += get_num_groups(0)) {
        __global float* Sh = S + h * D * D;
        __global const uint16_t* q = conved + h * D;
        __global const uint16_t* k = conved + width + h * D;
        __global const uint16_t* v = conved + 2ul * width + h * D;
        __global const uint16_t* fh = f + h * D;
        __global const uint16_t* dth = dt + h * D;
        const float a = nn__silicon__expf(nn__silicon__half_to_float(a_log[h]));
        const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[h])));
        const float qi = nn__silicon__half_to_float(q[j]), ki = nn__silicon__half_to_float(k[j]);
        float qq = 0.0f, kk = 0.0f;
        qq += qi * qi;
        kk += ki * ki;
        const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)D);
        const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
        float col[128];
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) col[i] = Sh[i * D + j];
        float kv = 0.0f;
        for (uint32_t i = 0u; i < D; ++i) {
            const float gi = lower / (1.0f + nn__silicon__expf(-a * (nn__silicon__half_to_float(fh[i]) + nn__silicon__half_to_float(dth[i]))));
            col[i] *= nn__silicon__expf(gi);
            kv += col[i] * nn__silicon__half_to_float(k[i]) * inv_k;
        }
        const float delta = (nn__silicon__half_to_float(v[j]) - kv) * beta;
        float o = 0.0f;
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) {
            const float s = col[i] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
            Sh[i * D + j] = s;
            o += s * nn__silicon__half_to_float(q[i]);
        }
        const float oj = nn__silicon__half_to_float(nn__kernels__zzabi_to_half(o * inv_q, over));   /* rounded, as nn's readout is */
        float oo = 0.0f;
        oo += oj * oj;
        const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)D + eps);
        const float gz = nn__silicon__half_to_float(gate[h * D + j]);
        out[h * D + j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) / (1.0f + nn__silicon__expf(-gz)), over);
    }
}

/* ⭐ A SITE'S MAP'S LOGITS — nn's `hyper_logits` and `hyper_logits_fp32`: a work-group a logit and a lane a stride of the
 *   streams, a lane's squares and products element by element, then the work-group's fold — nn's sums in nn's order; a
 *   lane's elements read sixteen at a time, all in flight together, where nn's body waits for each before the next.
 *   `half_streams` says which the streams are: GLM's halves or DeepSeek's floats. */
static inline float khronos_opencl2__zzprivate_stream(__global const void* streams, uint64_t i, bool half_streams) {
    return half_streams ? nn__silicon__half_to_float(((__global const uint16_t*)streams)[i]) : ((__global const float*)streams)[i];
}
static inline void khronos_opencl2__zzprivate_hyper_logits(__global float* lg, __global const void* streams, __global const uint16_t* fn,
                                                           uint64_t hidden, uint64_t mult, float norm_eps, bool half_streams) {
    const uint64_t lane = get_local_id(0), lanes = get_local_size(0), n = mult * hidden, rows = (2u + mult) * mult;
    const uint64_t RUN = KHRONOS_OPENCL2__ZZPRIVATE_RUN;
    for (uint64_t r = get_group_id(0); r < rows; r += get_num_groups(0)) {
        __global const uint16_t* row = fn + r * n;
        float sq = 0.0f, dot = 0.0f;
        uint64_t c = lane;
        for (; c + (RUN - 1u) * lanes < n; c += RUN * lanes) {
            float v[16], g[16];
            for (uint64_t u = 0u; u < RUN; ++u) {
                v[u] = khronos_opencl2__zzprivate_stream(streams, c + u * lanes, half_streams);
                g[u] = nn__silicon__half_to_float(row[c + u * lanes]);
            }
            for (uint64_t u = 0u; u < RUN; ++u) {
                sq += v[u] * v[u];
                dot += g[u] * v[u];
            }
        }
        for (; c < n; c += lanes) {
            const float v = khronos_opencl2__zzprivate_stream(streams, c, half_streams);
            sq += v * v;
            dot += nn__silicon__half_to_float(row[c]) * v;
        }
        const float inv = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(sq) / (float)n + norm_eps);
        const float got = nn__silicon__lanes_sum(dot) * inv;
        if (lane == 0u) lg[r] = got;
    }
}
__kernel void khronos_opencl2__hyper_logits(ulong lg__address, ulong streams__address, ulong fn__address, uint64_t hidden, uint64_t mult,
                                            float norm_eps) {
    khronos_opencl2__zzprivate_hyper_logits((__global float*)lg__address, (__global const void*)streams__address,
                                            (__global const uint16_t*)fn__address, hidden, mult, norm_eps, true);
}
__kernel void khronos_opencl2__hyper_logits_fp32(ulong lg__address, ulong streams__address, ulong fn__address, uint64_t hidden,
                                                 uint64_t mult, float norm_eps) {
    khronos_opencl2__zzprivate_hyper_logits((__global float*)lg__address, (__global const void*)streams__address,
                                            (__global const uint16_t*)fn__address, hidden, mult, norm_eps, false);
}

/* ⭐ THE SCORED ROUTER — nn's `top_k_scored`: nn's `k` rounds of the work-group's two folds, each round's largest score + bias
 *   and the first index holding it, so nn's picks and nn's weights. A lane makes each of its scores once — an exponential, a
 *   logarithm and a root for √softplus — and keeps them; a pick leaves its lane's list. Up to four scores a lane of 256. */
static inline float khronos_opencl2__zzprivate_score(float x, uint64_t score) {   /* nn's `zzprivate_score` */
    if (score == KHRONOS_OPENCL2__ZZPRIVATE_SCORE_SIGMOID) return 1.0f / (1.0f + nn__silicon__expf(-x));
    if (score == KHRONOS_OPENCL2__ZZPRIVATE_SCORE_SOFTPLUS) return nn__silicon__sqrtf(x > 20.0f ? x : nn__silicon__logf(1.0f + nn__silicon__expf(x)));
    return x;
}
__kernel void khronos_opencl2__top_k_scored(ulong values__address, uint64_t stride, ulong weights__address, ulong x__address,
                                            ulong bias__address, uint64_t n, uint64_t k, float scale, uint64_t score) {
    if (get_group_id(0) != 0u) return;
    __global uint64_t* values = (__global uint64_t*)values__address;
    __global uint16_t* weights = (__global uint16_t*)weights__address;
    __global const uint16_t* x = (__global const uint16_t*)x__address;
    __global const uint16_t* bias = (__global const uint16_t*)bias__address;
    const uint64_t lane = get_local_id(0), lanes = get_local_size(0);
    float v[4];
    bool live[4];
    for (uint64_t j = 0u; j < 4u; ++j) {
        const uint64_t i = lane + j * lanes;
        v[j] = i < n ? khronos_opencl2__zzprivate_score(nn__silicon__half_to_float(x[i]), score) + nn__silicon__half_to_float(bias[i]) : 0.0f;
        live[j] = i < n && v[j] == v[j];
    }
    uint64_t chosen[KHRONOS_OPENCL2__ZZPRIVATE_ROUTER_MOST];
    for (uint64_t r = 0u; r < k; ++r) {
        uint64_t best_i = n;
        float best_v = -INFINITY;
        for (uint64_t j = 0u; j < 4u; ++j)
            if (live[j] && (best_i == n || v[j] > best_v)) { best_v = v[j]; best_i = lane + j * lanes; }
        const float top = nn__silicon__lanes_max(best_v);
        const float first = nn__silicon__lanes_max(best_i != n && best_v == top ? -(float)(uint32_t)best_i : -INFINITY);
        chosen[r] = first == -INFINITY ? n : (uint64_t)(-first);
        for (uint64_t j = 0u; j < 4u; ++j)
            if (lane + j * lanes == chosen[r]) live[j] = false;
    }
    if (lane != 0u) return;
    float s[KHRONOS_OPENCL2__ZZPRIVATE_ROUTER_MOST], sum = 0.0f;
    for (uint64_t r = 0u; r < k; ++r) {
        s[r] = chosen[r] < n ? khronos_opencl2__zzprivate_score(nn__silicon__half_to_float(x[chosen[r]]), score) : 0.0f;
        sum += s[r];
    }
    for (uint64_t r = 0u; r < k; ++r) {
        values[r * stride] = chosen[r];
        weights[r] = nn__kernels__zzabi_to_half(s[r] / (sum + 1e-20f) * scale, 0);   /* nn's router raises no overflow either */
    }
}

/* ⭐ A SITE'S `pre` AT FOUR STREAMS — nn's `hyper_pre` and `hyper_pre_fp32`: nn's body's first lane works the mix's Sinkhorn
 *   iterations out of memory, one lane's chain of divisions, while the rest collapse — 111 us of the call on the MI50, where
 *   the collapse is a few (`MEASURED`, the bench; one lane in registers still 75). Here the first work-group's sixteen
 *   lanes hold an entry of the 4×4 mix each, and a row's or a column's sum is read from the work-group's local memory,
 *   each lane adding its row's or its column's four in nn's order: each entry nn's expression in nn's order (▶ nn's
 *   `nn__hyper__zzprivate_weights`; the ROCm family's, which reads the sums by shuffles OpenCL 2.0 lacks). A half-step
 *   writes one of two places and waits once, every lane of the work-group reaching each wait — the category of nn's fold
 *   here, a `work_group_reduce_add`. The collapse is nn's loop. */
static inline void khronos_opencl2__zzprivate_hyper_mix4(__global float* mix, __global const float* lg, const float* pre,
                                                         __global const uint16_t* base, float s1, float s2, uint64_t iters, float eps,
                                                         __local float* sh) {
    const uint32_t M = KHRONOS_OPENCL2__ZZPRIVATE_MULT, lane = (uint32_t)get_local_id(0), l = lane % (M * M), i = l / M, j = l % M;
    float row[4], top = -INFINITY, sum = 0.0f;                  /* this lane's row: a softmax over its columns, plus eps */
    for (uint32_t c = 0u; c < M; ++c) {
        const uint32_t k = 2u * M + i * M + c;
        row[c] = lg[k] * s2 + nn__silicon__half_to_float(base[k]);
        if (row[c] > top) top = row[c];
    }
    for (uint32_t c = 0u; c < M; ++c) { row[c] = nn__silicon__expf(row[c] - top); sum += row[c]; }
    float v = row[j] / sum + eps;
    uint32_t at = 0u;
    for (uint64_t it = 0u; it < iters; ++it) {                  /* the columns, then — past the first — rows and columns */
        if (it != 0u) {
            if (lane < M * M) sh[at * 16u + l] = v;
            barrier(CLK_LOCAL_MEM_FENCE);
            float rs = 0.0f;
            for (uint32_t c = 0u; c < M; ++c) rs += sh[at * 16u + i * M + c];
            v /= rs + eps;
            at ^= 1u;
        }
        if (lane < M * M) sh[at * 16u + l] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        float cs = 0.0f;
        for (uint32_t r = 0u; r < M; ++r) cs += sh[at * 16u + r * M + j];
        v /= cs + eps;
        at ^= 1u;
    }
    if (lane < M * M) mix[2u * M + l] = v;
    if (lane < M) {
        mix[lane] = pre[lane];
        mix[M + lane] = 2.0f / (1.0f + nn__silicon__expf(-(lg[M + lane] * s1 + nn__silicon__half_to_float(base[M + lane]))));
    }
}
/* nn's `pre` — the weights every lane collapses with — and, by the first work-group alone, the mix. */
static inline void khronos_opencl2__zzprivate_hyper_pre4(float* pre, __global float* mix, __global const uint16_t* base,
                                                         __global const uint16_t* scale, uint64_t iters, float eps, __local float* sh) {
    const uint64_t M = KHRONOS_OPENCL2__ZZPRIVATE_MULT;
    __global const float* lg = mix + (2u + M) * M;
    const float s0 = nn__silicon__half_to_float(scale[0]), s1 = nn__silicon__half_to_float(scale[1]);
    const float s2 = nn__silicon__half_to_float(scale[2]);
    for (uint64_t i = 0u; i < M; ++i)
        pre[i] = 1.0f / (1.0f + nn__silicon__expf(-(lg[i] * s0 + nn__silicon__half_to_float(base[i])))) + eps;
    if (get_group_id(0) == 0u) khronos_opencl2__zzprivate_hyper_mix4(mix, lg, pre, base, s1, s2, iters, eps, sh);
}
/* GLM's, the streams halves: the collapse nn's sum in nn's order, rounded to a half as nn rounds it, its overflow in the
 *   same fault word. */
__kernel void khronos_opencl2__hyper_pre4(ulong out__address, ulong mix__address, ulong streams__address, ulong base__address,
                                          ulong scale__address, uint64_t hidden, uint64_t iters, float eps, ulong over__address) {
    __local float sh[32];
    __global uint16_t* out = (__global uint16_t*)out__address;
    __global const uint16_t* streams = (__global const uint16_t*)streams__address;
    __global unsigned int* over = (__global unsigned int*)over__address;
    float pre[4];
    khronos_opencl2__zzprivate_hyper_pre4(pre, (__global float*)mix__address, (__global const uint16_t*)base__address,
                                          (__global const uint16_t*)scale__address, iters, eps, sh);
    for (uint64_t e = get_global_id(0); e < hidden; e += get_global_size(0)) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < KHRONOS_OPENCL2__ZZPRIVATE_MULT; ++i) sum += pre[i] * nn__silicon__half_to_float(streams[i * hidden + e]);
        out[e] = nn__kernels__zzabi_to_half(sum, over);
    }
}
/* DeepSeek's, the streams floats. */
__kernel void khronos_opencl2__hyper_pre4_fp32(ulong out__address, ulong mix__address, ulong streams__address, ulong base__address,
                                               ulong scale__address, uint64_t hidden, uint64_t iters, float eps) {
    __local float sh[32];
    __global float* out = (__global float*)out__address;
    __global const float* streams = (__global const float*)streams__address;
    float pre[4];
    khronos_opencl2__zzprivate_hyper_pre4(pre, (__global float*)mix__address, (__global const uint16_t*)base__address,
                                          (__global const uint16_t*)scale__address, iters, eps, sh);
    for (uint64_t e = get_global_id(0); e < hidden; e += get_global_size(0)) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < KHRONOS_OPENCL2__ZZPRIVATE_MULT; ++i) sum += pre[i] * streams[i * hidden + e];
        out[e] = sum;
    }
}

#endif /* __OPENCL_C_VERSION__ */

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_OVERRIDES_KERNELS_CUH */
