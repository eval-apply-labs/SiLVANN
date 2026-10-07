#ifndef SILVANN__PACKAGES_SYS_CPU_CAROUSEL__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_CAROUSEL__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"
#include "carousel__header.cuh"
#include "silicon/silicon__header.cuh"
/* ══ THE CAROUSEL, DEFINED ═══════════════════════════════════════════════════════════════════════════
 * The contract, the arguments and the type are in `carousel__header.cuh`. This file owes nothing to
 * anything else in the package: a compare-and-swap and a wait from the silicon seam, room from the same
 * seam, and the fault channel to say no on.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


/* ── THE TWO PLACES THE WIDTH IS KNOWN, AND THEY ARE THE ONLY TWO ─────────────────────────────────────
 * Everything else in this file works in `uint32_t` and never looks at `slot` directly. Keeping the untyped
 * pointer behind exactly two accessors is what stops "the storage is bytes" from becoming a fact every
 * function has to remember. The branch is predictable and constant for the life of a ring.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint32_t sys__carousel__zzprivate_read(const sys__carousel* ring, unsigned int at) {
    if (ring->width == SYS__CAROUSEL__U8)  return (uint32_t)((const unsigned char*)ring->slot)[at];
    if (ring->width == SYS__CAROUSEL__U16) return (uint32_t)((const unsigned short*)ring->slot)[at];
    return ((const uint32_t*)ring->slot)[at];
}

static __device__ inline void sys__carousel__zzprivate_write(sys__carousel* ring, unsigned int at,
                                                             uint32_t value) {
    if (ring->width == SYS__CAROUSEL__U8)       ((unsigned char*)ring->slot)[at]  = (unsigned char)value;
    else if (ring->width == SYS__CAROUSEL__U16) ((unsigned short*)ring->slot)[at] = (unsigned short)value;
    else                                        ((uint32_t*)ring->slot)[at]       = value;
}

/* The largest value a width can carry. `create` is not the only place this matters: `add` refuses a value
 * that would not survive the store, because a truncated chunk index is a valid-looking index to somewhere
 * else — the one failure shape a narrow ring introduces that a wide one does not have. */
static __device__ inline uint32_t sys__carousel__zzprivate_ceiling(unsigned int width) {
    if (width == SYS__CAROUSEL__U8)  return 0xFFu;
    if (width == SYS__CAROUSEL__U16) return 0xFFFFu;
    return 0xFFFFFFFFu;
}

/* How many are waiting. A difference of two rising counters, so it is correct without either being read
 * at the same instant as the other — which is what makes this safe to ask while others are adding. */
static __device__ inline unsigned int sys__carousel__count(const sys__carousel* ring) {
    return (ring == 0) ? 0u : (ring->head - ring->tail);
}

/* Put one at the head.
 *
 * The cursor is taken FIRST and the value written after, which is the order that lets a taker notice it
 * has arrived early — see the header. ⭐ THE RESERVATION IS A COMPARE-AND-SWAP RATHER THAN AN INCREMENT
 * BECAUSE IT IS CONDITIONAL, NOT BECAUSE THE SEAM IS SHORT OF VERBS: `sys__silicon__add_u32` is there and
 * is one instruction, but an increment takes the cursor before the fullness test below can say no, and a
 * cursor other adders are already walking past cannot be handed back. `REASONED` — the refusal is the
 * whole of the difference, and a counter with no refusal to make uses the increment instead (see
 * `sys__heap__zzpackage_take_up`). The retry is the ordinary one: somebody else took the slot, so read
 * the cursor again and try the next. */
static __device__ inline bool sys__carousel__add(sys__carousel* ring, uint32_t value) {
    if (ring == 0 || ring->slot == 0) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_NO_RING);
        return false;
    }
    if (value == 0u) {                       /* zero is the empty marker and cannot also be a member */
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_ZERO);
        return false;
    }
    if (value > sys__carousel__zzprivate_ceiling(ring->width)) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_RANGE);
        return false;
    }
    unsigned int h = ring->head;
    for (;;) {
        /* The most that can ever be WAITING is one fewer than there are slots — at `slots` the two masked
         * cursors land on the same place and a full ring would read as an empty one. `mask` IS slots-1,
         * so it is both the wrap and the ceiling, and they are the same number for the same reason. */
        if ((h - ring->tail) >= ring->mask) {
            sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_FULL);
            return false;
        }
        const unsigned int seen = sys__silicon__cas_u32(&ring->head, h, h + 1u);
        if (seen == h) break;
        h = seen;
    }
    sys__carousel__zzprivate_write(ring, h & ring->mask, value);
    return true;
}

/* Take one from the tail, or answer false when there is nothing waiting.
 *
 * ⛳ THE TWO READS ARE IN THIS ORDER FOR A REASON. Tail is read first and head second, so head is the
 * FRESHER of the two — and that is what makes `t == h` mean genuinely empty rather than merely
 * out-of-date. Tail never passes head, so if head is seen equal to a tail that was read earlier, tail
 * cannot have been anything but that same value at the instant head was read. Reading them the other way
 * round would allow a stale head to report an empty pool that has something in it — a refusal to allocate
 * with memory available, which is the worst answer this function can give. */
static __device__ inline bool sys__carousel__take(sys__carousel* ring, uint32_t* out) {
    if (ring == 0 || ring->slot == 0 || out == 0) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_NO_RING);
        return false;
    }
    unsigned int t;
    for (;;) {
        t = ring->tail;
        if (t == ring->head) return false;               /* empty, and see the note above on the order */
        if (sys__silicon__cas_u32(&ring->tail, t, t + 1u) == t) break;
    }
    const unsigned int at = t & ring->mask;
    for (unsigned int k = 0u; k < SYS__CAROUSEL__WAIT_TRIES; ++k) {
        const uint32_t v = sys__carousel__zzprivate_read(ring, at);
        if (v != 0u) {
            sys__carousel__zzprivate_write(ring, at, 0u);   /* say it is spent, for the next time round */
            *out = v;
            return true;
        }
        sys__silicon__wait_cycles(SYS__CAROUSEL__WAIT_CYCLES);
    }
    /* The slot was reserved by an adder that never wrote it. The reservation is not given back — putting
     * the cursor back races every other taker — so one entry is lost, loudly. */
    sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_STALE);
    return false;
}


/* Ask the platform for room, lay the carousel at the front of it, and answer that address.
 *
 * The room comes from `sys__silicon__alloc`, which is the seam, so there is no ceiling here that a build
 * decides — a ring is as large as the machine will give and no larger.
 *
 * ⛳ ONE ALLOCATION, NOT TWO. The structure sits at the front of the room and the ring follows it, so the
 * address `create` answers and the address `dispose` frees are the same one — which is what lets `dispose`
 * take nothing but the pointer a caller was given, with no handle to keep in step. `sys__carousel` begins
 * with a word and holds a pointer, so the room after it is aligned for anything the ring can be. */
static __device__ inline sys__carousel* sys__carousel__create(unsigned int slots, unsigned int width) {
    if (width != SYS__CAROUSEL__U8 && width != SYS__CAROUSEL__U16 && width != SYS__CAROUSEL__U32) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_WIDTH);
        return (sys__carousel*)0;
    }
    if (slots == 0u || (slots & (slots - 1u)) != 0u) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_GEOMETRY);
        return (sys__carousel*)0;
    }
    /* The only ceiling left is arithmetic: a request whose room cannot be COUNTED in the width the seam
     * takes would wrap, and an allocation smaller than asked for is the one failure that reads as success
     * all the way to the first write past the end. */
    const unsigned int header = (unsigned int)sizeof(sys__carousel);
    if (slots > (0xFFFFFFFFu - header) / width) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_TOO_BIG);
        return (sys__carousel*)0;
    }
    void* room = sys__silicon__alloc(header + slots * width);
    if (room == 0) {
        /* The seam answers NULL rather than raising, because only a caller knows whether it can carry on
         * without the room. This one cannot, so the fault is raised HERE, where that is known. */
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_NO_ROOM);
        return (sys__carousel*)0;
    }
    sys__carousel* ring = (sys__carousel*)room;
    ring->live  = SYS__CAROUSEL__LIVE;
    ring->slot  = (void*)((unsigned char*)room + header);
    ring->mask  = slots - 1u;
    ring->width = width;
    ring->head  = 0u;
    ring->tail  = 0u;
    for (unsigned int k = 0u; k < slots; ++k) sys__carousel__zzprivate_write(ring, k, 0u);
    return ring;
}

/* Give the room back — the same address `create` answered, freed through the seam it came from.
 *
 * ⛔⛔ IT REFUSES A RING WITH THINGS IN IT, and the reason has nothing to do with memory. What is waiting
 * is not bytes: for the pool's own free
 * list they are CHUNK INDICES, and abandoning them loses those chunks forever with every count still
 * balancing. ⇒ ★ AN EMPTY CONTAINER AND A FULL ONE ARE THE SAME SIZE, so nothing about freeing one
 * notices the difference. Drain first — draining is where a caller finds out there was something left.
 *
 * ⛳ IT IS NOT `release`, AND THE DIFFERENCE IS THE POINT. A release is a hold being given up and MAY end
 * the thing; a dispose ends it. A carousel is held by exactly one C caller and is named by no value, so
 * there is no second holder for a count to be tracking. */
static __device__ inline void sys__carousel__dispose(sys__carousel* ring) {
    if (ring == 0 || ring->live != SYS__CAROUSEL__LIVE) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_NO_RING);
        return;
    }
    if (sys__carousel__count(ring) != 0u) {
        sys__fault__raise(0ull, SYS__CAROUSEL__FAULT_NOT_EMPTY);
        return;                              /* and the room stays — nothing is half-disposed */
    }
    ring->live = 0u;                         /* before the free, so the witness is gone either way */
    ring->slot = 0;
    sys__silicon__free((void*)ring);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_CAROUSEL__IMPL_CUH */
