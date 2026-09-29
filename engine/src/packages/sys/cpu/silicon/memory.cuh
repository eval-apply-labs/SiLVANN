#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_MEMORY_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_MEMORY_CUH
#include "silicon__header.cuh"   /* the contract these primitives answer */
/* ══ SCRATCH ROOM, ON A CPU ══════════════════════════════════════════════════════════════════════════
 * ⛳ THIS IS THE SEAM'S *DEVICE-SIDE* ALLOCATOR — what a verb reaches for when it needs room of its own
 * while it runs. On a card that is device `malloc` under a launch, with all the doubt that carries —
 * whether it is available, what it costs under contention, how large the device heap is allowed to
 * grow, none of it measured; on a host it is the ordinary one, and the doubt goes away with it.
 * ⛔ IT IS NOT THE LISP HEAP. `sys__heap__*` carves from the published arena and knows nothing about
 * this; conflating the two is how a chunk ends up in a different allocator's hands. */
#include <stdlib.h>

static inline void* sys__silicon__alloc(unsigned int bytes) { return malloc((size_t)bytes); }
static inline void  sys__silicon__free(void* room)          { if (room) free(room); }

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_MEMORY_CUH */
