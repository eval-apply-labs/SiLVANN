#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_NODE_ARRAY_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_NODE_ARRAY_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap.cuh"     /* how wide a chunk is, and where in one the first allocation can start */

/* The smallest array there can be: a head and one element. It is what the kind's row states, so making
 * one without saying how many gives an array of one rather than an array of nothing. */
#define SYS__NODE_ARRAY__NODES_MIN 2u

#define SYS__NODE_ARRAY__FAULT_EMPTY 0x4E415245ull   /* "NARE" — asked for an array with no elements  */
#define SYS__NODE_ARRAY__FAULT_RANGE 0x4E41524Full   /* "NARO" — an element Outside the array         */
#define SYS__NODE_ARRAY__FAULT_KIND  0x4E41524Eull   /* "NARN" — Not one, or a placement refused      */
#define SYS__NODE_ARRAY__FAULT_SPLIT 0x4E415253ull   /* "NARS" — a run asked for whole is Split        */

/* ── A LONG ARRAY IS A CHAIN OF RUNS ─────────────────────────────────────────────────────────────────
 * One allocation is carved from one heap chunk, so the most cells an allocation can give an array is
 * what a chunk holds past its own head, less the array's head. An array that fits is one run of exactly
 * the cells it asked for. One that does not is several runs, each a full allocation whose last cell is a
 * CONE: a cell that holds no element and names the run that carries on.
 * ⛳ ONLY A FULL RUN CAN END IN A CONE, which is what lets an array that fits answer its length off its
 * head alone: a run shorter than this is never chained, so its last cell need not be read. */
#define SYS__NODE_ARRAY__RUN_CELLS ((uint64_t)SYS__HEAP__CHUNK_NODES - (uint64_t)SYS__CHUNK__FIRST - 1ull)

/* ⭐ AND A CHAINED ARRAY KEEPS A TABLE OF ITS RUNS, so an element is one hop away however far along it is:
 * a node array holding, in order, where each run's first element sits — numbers, not references, so it
 * owns none of them; the cones do. The first run's head names it here and owns it, and it dies with the
 * array. ⛔ THE WORD IS READ ONLY WHEN THE FIRST RUN ENDS IN A CONE: a head that is not the start of a
 * chain was never written there, so what it holds is whatever the room held before. */
#define SYS__NODE_ARRAY__INDEXER 1u

/* Where a walk over an array is. A VALUE the caller keeps on its own stack, like a list's walk — it
 * allocates nothing and takes no hold, so a loop that breaks leaves nothing behind. */
struct sys__node_array_walk {
    uint64_t cell;    /* the element the walk stands on, 0 once it has run off the end                 */
    uint64_t last;    /* the last cell of the run it is in — the run's final element, or its cone      */
};

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_NODE_ARRAY_CUH */
