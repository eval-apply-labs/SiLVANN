#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_HADAMARD_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_HADAMARD_CUH

/* This file needs nothing: every name in it is its own. */
#define NN__HADAMARD__FAULT_LENGTH 0x4E48444Cull   /* "NHDL" — a length that is not a power of two */

/* ⛔ THE BLOCK THE BUTTERFLY WALKS WHOLE. ⛳ In the header because the implementation and `kernels.cuh`
 * both read it, and a constant two files read belongs to the header. */
#define NN__HADAMARD__WHOLE_BELOW  512ull

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_HADAMARD_CUH */
