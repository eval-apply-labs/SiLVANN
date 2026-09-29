#ifndef SILVANN__PACKAGES_NN_CPU_DELTANET__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_DELTANET__IMPL_CUH
#include "deltanet__header.cuh"
#include "primitives__header.cuh"       /* the resolver, and the fp16 bound these two do NOT use */

/* ⛔⛔ THE STATE'S OWN BOUND, AND IT IS SEPARATE FROM `nn__primitives__fits` ON PURPOSE. That helper
 * multiplies by `NN__PRIMITIVES__ELEMENT_BYTES`, which is 2 — fp16's width and the width of every other
 * array in this package. The state is fp32, so borrowing that bound would admit a matrix TWICE the size
 * of the slot it was given, silently, and the overrun would land in the neighbouring head's state where
 * every number is plausible. ⇒ ★★ A BOUND IS A STATEMENT ABOUT A WIDTH, SO A BOUND REUSED AT A
 * DIFFERENT WIDTH IS NOT A WEAKER CHECK — IT IS THE WRONG ONE. */
static __device__ inline bool nn__deltanet__zzpackage_fits32(uint64_t n, uint64_t room) {
    if (n != 0ull && n > 0xFFFFFFFFFFFFFFFFull / 4ull) return false;
    return n * 4ull <= room;
}

/* Both verbs want the same three numbers out of two cells, and getting the wrap test subtly different
 * in two places is how a bound stops being one. */
static __device__ inline bool nn__deltanet__zzpackage_shape(sys__heap_node kc, sys__heap_node vc,
                                                            uint64_t* k_dim, uint64_t* v_dim,
                                                            uint64_t* cells) {
    if (kc.dtype != SYS__KIND__VALUE_INT || vc.dtype != SYS__KIND__VALUE_INT) return false;
    *k_dim = kc.args[0];
    *v_dim = vc.args[0];
    if (*k_dim == 0ull || *v_dim == 0ull) return false;
    if (*k_dim > 0xFFFFFFFFFFFFFFFFull / *v_dim) return false;
    *cells = *k_dim * *v_dim;
    return true;
}

#endif /* SILVANN__PACKAGES_NN_CPU_DELTANET__IMPL_CUH */
