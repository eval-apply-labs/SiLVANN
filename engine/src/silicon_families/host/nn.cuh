#ifndef SILVANN__SILICON_FAMILIES_HOST_NN_CUH
#define SILVANN__SILICON_FAMILIES_HOST_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <math.h>
#include <stdint.h>
#include "../../packages/nn/contracts/abi/gpu.cuh"   /* what is answered here */

/* ══ nn's ARITHMETIC, ON THE HOST ═══════════════════════════════════════════════════════════════════════
 * The host runs no kernel, but the program compiles nn's compute bodies too — the checks run them as the
 * reference a card is measured against — so the host family answers the arithmetic they ask
 * for (`nn/contracts/abi/gpu.cuh`), from libm.
 * ⚠ NOT BIT-IDENTICAL TO A CARD'S: a device `expf` is the card's own, not libm's, so a host/device oracle
 * comparing softmax or rmsnorm to the last bit will find that out. It is the same class of difference nn's
 * fp16 conversions were written in software to avoid, and it is not avoidable here: the answer is a
 * tolerance, not a reimplementation. */
static inline float nn__silicon__expf(float x)  { return expf(x); }
static inline float nn__silicon__sqrtf(float x) { return sqrtf(x); }
static inline float nn__silicon__logf(float x)  { return logf(x); }
static inline float nn__silicon__cosf(float x)  { return cosf(x); }
static inline float nn__silicon__sinf(float x)  { return sinf(x); }

/* A body compiled for the host is one lane of one block, so it runs as the serial loop it is. */
static inline uint32_t nn__silicon__lane(void)  { return 0u; }
static inline uint32_t nn__silicon__lanes(void) { return 1u; }
static inline uint32_t nn__silicon__block(void)  { return 0u; }
static inline uint32_t nn__silicon__blocks(void) { return 1u; }
/* One lane holds the whole of the block's work, so its part is already the answer. */
static inline float nn__silicon__lanes_sum(float part) { return part; }
static inline void nn__silicon__lanes_sums(float* parts, uint32_t n) { (void)parts; (void)n; }
static inline float nn__silicon__lanes_max(float part) { return part; }
/* The GEMM's tile, on the host: small, so the reference runs the tile loops the cards run, with more than
 * one row of each. */
static inline nn__gemm__tile nn__silicon__gemm_tile(void) { nn__gemm__tile t = {2u, 4u}; return t; }

/* A half as a float, without a half type (this compiler has none): the half's exponent and mantissa, moved
 * into a float's place, are the half's value times 2^-112 — for a subnormal too — so one multiply puts it
 * right; infinity and NaN keep their payload. `MEASURED`: bit-identical to nn's own conversion on
 * all 65,536 halves. */
static inline float nn__silicon__half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16, rest = (uint32_t)h & 0x7FFFu;
    union { uint32_t u; float f; } v;
    if (rest >= 0x7C00u) {
        v.u = sign | 0x7F800000u | ((rest & 0x3FFu) << 13);
    } else {
        v.u = rest << 13;
        v.f *= 0x1p112f;
        v.u |= sign;
    }
    return v.f;
}
/* Two halves by two halves onto a float, in the order a sum over the columns has. */
static inline float nn__silicon__dot2(uint32_t a, uint32_t b, float c) {
    return (c + nn__silicon__half_to_float((uint16_t)(a & 0xffffu)) * nn__silicon__half_to_float((uint16_t)(b & 0xffffu)))
         + nn__silicon__half_to_float((uint16_t)(a >> 16)) * nn__silicon__half_to_float((uint16_t)(b >> 16));
}

#endif /* SILVANN__SILICON_FAMILIES_HOST_NN_CUH */
