#ifndef SILVANN__PACKAGES_NN_MANIFEST__GPU_CUH
#define SILVANN__PACKAGES_NN_MANIFEST__GPU_CUH

/* ══ nn, AS A SILICON FAMILY'S BINARY COMPILES IT ═════════════════════════════════════════════════════
 * What nn asks of a family — its doors, the launch and the arithmetic (`contracts/abi/gpu.cuh`) — then the
 * declarations its compute bodies use, and the doors themselves, which include the bodies. Nothing of the
 * language: the compute stands alone, and `scripts/src_compute_stands_alone.sh` holds it to that.
 * ▶ `../manifest__gpu.cuh`. */
#include "contracts/abi/gpu.cuh"
#include "cpu/primitives__header.cuh"
#include "cpu/hadamard__header.cuh"
#include "cpu/deltanet__header.cuh"
#include "cpu/turboquant__header.cuh"
#include "gpu/doors.cuh"

#endif /* SILVANN__PACKAGES_NN_MANIFEST__GPU_CUH */
