#ifndef SILVANN__SILICON_FAMILIES_ARM_GB10_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_ARM_GB10_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <arm_neon.h>
#include <stdint.h>
#include "../../arm_base/overrides/nn.cuh"   /* the doors every Arm family runs faster: put in first */

/* ══ nn's DOORS THE DGX SPARK'S GB10 RUNS FASTER — ITS OWN, AFTER arm_base's ══════════════════════════════════════
 * The GB10's processors have what Armv8.2 has not — SVE2, and i8mm's 8-bit matrix products (`smmla`) — and two kinds of
 * core, ten Cortex-X925 and ten Cortex-A725. A door here is theirs; every other is `arm_base`'s, or nn's body. */

/* ── ⭐ THE EXPERT-MAJOR GEMM IN INTEGERS, TWO ROWS BY TWO PAIRS — `smmla` ──────────────────────────────────────────
 * arm_base's rows (`arm_base__int8_rows`) sum a matrix row against a pair sixteen products an instruction (`sdot`); i8mm's
 * `smmla` multiplies a 2×8 by an 8×2 into a 2×2 of sums, thirty-two an instruction: two matrix rows, eight columns of each
 * in turn (`wi`, their levels laid out so once a row pair), against eight columns of two pairs' quantised rows. Each of
 * its four lanes is a block's exact integer sum for one row and one pair, so the float steps after it are arm_base's own
 * (`arm_base__int8_step`, `arm_base__int8_finish`) and the answers arm_base's bytes. A last odd pair is its own second
 * (its lanes for the copy dropped), a last odd row arm_base's. */
static void arm_gb10__int8_rows(const arm_base__int8_matrix* m, uint64_t r0, uint64_t r1) {
    const int8x16_t t8 = arm_base__int8_levels();
    const uint64_t cols = m->cols, per = cols / NN__EXPERT__INT8_BLOCK, half = cols / 2u;
    int8_t wi[2u * cols] __attribute__((aligned(16)));
    uint64_t r = r0;
    for (; r + 2u <= r1; r += 2u) {
        const uint8_t* ra = m->codes + r * half;
        const uint8_t* rb = ra + half;
        for (uint64_t k = 0; k < cols; k += 32u) {
            int8x16_t a0, a1, b0, b1;
            arm_base__int8_levels32(ra + k / 2u, t8, &a0, &a1);
            arm_base__int8_levels32(rb + k / 2u, t8, &b0, &b1);
            int8_t* o = wi + 2u * k;
            vst1q_s8(o,       vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(a0), vreinterpretq_s64_s8(b0))));
            vst1q_s8(o + 16u, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(a0), vreinterpretq_s64_s8(b0))));
            vst1q_s8(o + 32u, vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(a1), vreinterpretq_s64_s8(b1))));
            vst1q_s8(o + 48u, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(a1), vreinterpretq_s64_s8(b1))));
        }
        const float sa = arm_base__int8_row_scale(m->luts, r), sb = arm_base__int8_row_scale(m->luts, r + 1u);
        for (uint64_t j = 0; j < m->count; j += 2u) {
            const uint64_t jb = j + 1u < m->count ? j + 1u : j;
            const uint64_t xa = (uint64_t)m->rows[2u * j] - m->x0, xb = (uint64_t)m->rows[2u * jb] - m->x0;
            const int8_t* pa = m->xq + xa * cols;
            const int8_t* pb = m->xq + xb * cols;
            const float* qa = m->xs + xa * per;
            const float* qb = m->xs + xb * per;
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;        /* row r·pair j, r·jb, r+1·j, r+1·jb */
            for (uint64_t B = 0; B < per; ++B) {
                int32x4_t s0 = vdupq_n_s32(0), s1 = s0;
                for (uint64_t k = B * NN__EXPERT__INT8_BLOCK; k < (B + 1u) * NN__EXPERT__INT8_BLOCK; k += 16u) {
                    s0 = vmmlaq_s32(s0, vld1q_s8(wi + 2u * k),       vcombine_s8(vld1_s8(pa + k),      vld1_s8(pb + k)));
                    s1 = vmmlaq_s32(s1, vld1q_s8(wi + 2u * k + 16u), vcombine_s8(vld1_s8(pa + k + 8u), vld1_s8(pb + k + 8u)));
                }
                const int32x4_t s = vaddq_s32(s0, s1);
                p0 = arm_base__int8_step(p0, vgetq_lane_s32(s, 0), qa[B]);
                p1 = arm_base__int8_step(p1, vgetq_lane_s32(s, 1), qb[B]);
                p2 = arm_base__int8_step(p2, vgetq_lane_s32(s, 2), qa[B]);
                p3 = arm_base__int8_step(p3, vgetq_lane_s32(s, 3), qb[B]);
            }
            arm_base__int8_finish(m, j, r, p0, sa);
            arm_base__int8_finish(m, j, r + 1u, p2, sb);
            if (jb != j) { arm_base__int8_finish(m, jb, r, p1, sa); arm_base__int8_finish(m, jb, r + 1u, p3, sb); }
        }
    }
    if (r < r1) arm_base__int8_rows(m, r, r1);
}
static void arm_gb10__override_expert_rows_sum_int8(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                    uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols) {
    arm_base__int8_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols, arm_gb10__int8_rows);
}
static void arm_gb10__override_expert_groups_int8(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                  uint64_t cols, unsigned int* over) {
    arm_base__int8_groups(out, g, x, rows, cols, over, arm_gb10__int8_rows);
}

/* arm_base's doors, then the GB10's own over them. */
static inline void arm_gb10__override_doors(nn__doors* doors) {
    arm_base__override_doors(doors);
    doors->expert_groups_int8   = arm_gb10__override_expert_groups_int8;
    doors->expert_rows_sum_int8 = arm_gb10__override_expert_rows_sum_int8;
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_GB10_OVERRIDES_NN_CUH */
