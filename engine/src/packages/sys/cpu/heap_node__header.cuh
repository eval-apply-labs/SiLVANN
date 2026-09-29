#ifndef SILVANN__PACKAGES_SYS_CPU_HEAP_NODE__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_HEAP_NODE__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"
#include "../contracts/objects/heap_node.cuh" /* the layouts it reads and writes */

/* ══ ⭐⭐ READING A CELL AS A REAL NUMBER — PUBLISHED, BECAUSE EVERY PACKAGE THAT TAKES ONE NEEDS IT ══
 *
 * ⚖ The float kind arrived and holds an fp64 in `args[0]` AS ITS BITS. These two are the
 * only sanctioned way to cross between the bits and the number.
 * ⛳ THEY ARE PUBLISHED RATHER THAN `zzprivate_` FOR A REASON THE VISIBILITY TIERS MAKE CONCRETE: a
 * private pair would be out of `nn`'s reach, so every package outside `sys` that takes a float argument
 * would write its own union — a second definition of the one thing that must not vary.
 * ⇒ ★ A CONVERSION EVERY PACKAGE NEEDS BELONGS WHERE THE THING IT CONVERTS LIVES.
 * ⛔ THEY ARE A REINTERPRETATION AND NOT A CONVERSION: nothing rounds, so a number that goes in comes
 * back bit for bit. ⛳ Written with a union rather than through the silicon seam, because the seam
 * carries what DIFFERS between backends and an exact bit move cannot. */
static __device__ inline double sys__heap_node__real(uint64_t bits) {
    union { uint64_t u; double d; } cast;
    cast.u = bits;
    return cast.d;
}

static __device__ inline uint64_t sys__heap_node__real_bits(double value) {
    union { uint64_t u; double d; } cast;
    cast.d = value;
    return cast.u;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_HEAP_NODE__HEADER_CUH */
