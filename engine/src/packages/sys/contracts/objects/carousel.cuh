#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_CAROUSEL_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_CAROUSEL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../defaults.cuh"

/* Turned in `contracts/defaults.cuh`, guaranteed here — see the note on the lock's pair in
 * `heap_object__header.cuh` for why both files carry a value. */
#ifndef SYS__CAROUSEL__WAIT_TRIES
#define SYS__CAROUSEL__WAIT_TRIES 64u
#endif
#ifndef SYS__CAROUSEL__WAIT_CYCLES
#define SYS__CAROUSEL__WAIT_CYCLES 10u
#endif

/* ── THE STORAGE WIDTH — the value IS the number of bytes, so nothing converts ──────────────────────── */
#define SYS__CAROUSEL__U8   1u
#define SYS__CAROUSEL__U16  2u
#define SYS__CAROUSEL__U32  4u

#define SYS__CAROUSEL__FAULT_GEOMETRY 0x43524747ull   /* "CRGG" — a ring that is not a power of two
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#define SYS__CAROUSEL__FAULT_NO_RING  0x43524E52ull   /* "CRNR" — asked of a ring that has no storage  */
#define SYS__CAROUSEL__FAULT_ZERO     0x43525A52ull   /* "CRZR" — zero was added, and zero means empty */
#define SYS__CAROUSEL__FAULT_FULL     0x4352464Cull   /* "CRFL" — the sizing rule was broken upstream  */
#define SYS__CAROUSEL__FAULT_STALE    0x43525354ull   /* "CRST" — a reserved slot never got its value  */
#define SYS__CAROUSEL__FAULT_WIDTH    0x43525744ull   /* "CRWD" — not one of the three storage widths  */
#define SYS__CAROUSEL__FAULT_RANGE    0x4352524Eull   /* "CRRN" — a value too wide for the storage     */
#define SYS__CAROUSEL__FAULT_TOO_BIG  0x43524247ull   /* "CRBG" — a size whose room cannot be counted  */
#define SYS__CAROUSEL__FAULT_NO_ROOM  0x43524F4Dull   /* "CROM" — the platform had no room to give     */
#define SYS__CAROUSEL__FAULT_NOT_EMPTY 0x43524E45ull  /* "CRNE" — disposed with things still waiting   */

/* ⛔ AND THE FIRST FIELD IS A WITNESS, BECAUSE THE ROOM COMES FROM THE PLATFORM AND NOT FROM A POOL.
 * `dispose` tells an entry from a stranger by its VALUE and not by pointer arithmetic against a pool. Room
 * from the platform has no range to test against, so the structure has to say what it is.
 * ⚠ IT IS NOT A COMPLETE GUARD AND CLAIMING OTHERWISE WOULD BE WORSE THAN NOT HAVING IT: after a `free`
 * the bytes belong to the allocator, so a DOUBLE dispose reads memory this code does not own and may find
 * the word still there, or find anything. What it does catch, every time, is a pointer to something that
 * was never a carousel — which is the case a caller can actually make by mistake. */
#define SYS__CAROUSEL__LIVE 0x43524C56u                /* "CRLV", written by create, cleared by dispose */

/* The cursors are plain counters that only rise. They are NOT the index; the index is what is left after
 * masking, and keeping the counter whole is what lets `head - tail` be the number waiting without a
 * separate count. They wrap at their own width, and the difference stays correct across the wrap because
 * it can never approach it — at most one fewer entry than the caller owns is ever outstanding.
 *
 * ⛳ `slot` IS UNTYPED BECAUSE THE WIDTH IS DATA. It is read and written through the two accessors in the
 * implementation and nowhere else, which is what keeps the untyped pointer from spreading. */
typedef struct {
    unsigned int live;     /* SYS__CAROUSEL__LIVE while this is a carousel, and zero once it is not */
    void*        slot;     /* the ring itself — `slots x width` bytes */
    unsigned int mask;     /* slots - 1, which is why slots must be a power of two */
    unsigned int width;    /* bytes per slot: one of the three widths */
    unsigned int head;     /* where the next thing is ADDED */
    unsigned int tail;     /* where the next thing is TAKEN from */
} sys__carousel;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_CAROUSEL_CUH */
