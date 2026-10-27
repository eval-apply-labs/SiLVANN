#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_PAD_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_PAD_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>

/* ══ ⭐⭐ A CARD'S LANDING PAD — PINNED RAM BETWEEN THE CARD AND THE REST OF THE MACHINE (NN-115) ═══════════════════════
 * ⚖ 2026-10-24: *"it still makes sense to have some memory that is mapped because it allows us to not depend on the cpu
 * for too many sync steps, and we could do it double buffered too"* · *"4 256mb chunks, so we have full duplex"* · *"a
 * parameter in the config especially for home users that might want to have 4x125mb"* · *"we need to create the data
 * structure and the transfer mechanism first, and then use it"*.
 *   A card's pad is chunks of pinned memory, some carrying bytes up (RAM -> card) and the rest down, each way a ring
 * with its own side channel, so the two ways run at once. Whoever moves bytes CLAIMS a region of its way's open
 * chunk — several threads at once, each its own region — fills it (up) and SENDS it, or FETCHES into it (down) and
 * RELEASES it once read. Each send or fetch is queued at once on the way's channel with a mark after it, and answers a
 * TICKET: its number in its way. The copies of a way run in the order they were queued, so a ticket is reached once its
 * mark is, and a chunk is filled again once the last copy queued from it is reached — while the copy engine empties one
 * chunk, the next is filled. The host waits for a ticket (`landed`) only to reuse a chunk or to read what came down; the
 * card's own queue may wait for one instead (`card_waits`), so a kernel reads what was sent with no host step between.
 * ⛳ A family without marks (OpenCL) has done each copy when its call returns: the pad moves the same bytes, one
 *   chunk at a time. */
#define SYS__PAD__CHUNKS_MAX  32u    /* a pad's chunks at most, both ways */
#define SYS__PAD__MARKS       64u    /* a way's marks, a send's each in turn: tickets further back are waited through a later one */
#define SYS__PAD__PADS_MAX    16u    /* cards with a pad at once */
/* ⛳ THE TWO WAYS NEED NOT BE THE SAME SIZE: a chunk up is free once its copy has landed, but what an exclusive tier sends
 *   down waits in its chunk until the swap commits (▶ nn's `nn__expert_tier__commit`), several visits on — so the down way
 *   is also the parking its evictions wait in, a chunk a swap in flight (⚖ 2026-10-24 *"we have to keep the parking if we
 *   want to do it in full duplex"*), and a chunk is taken wherever one is free, not only the next in its ring. */
#define SYS__PAD__UP           0u    /* RAM -> card */
#define SYS__PAD__DOWN         1u    /* card -> RAM */

#define SYS__PAD__FAULT_OPEN  0x5350414Full   /* "SPAO" — the pad could not be pinned, or its channels and marks opened */
#define SYS__PAD__FAULT_MOVE  0x5350414Dull   /* "SPAM" — no pad on this card, or a copy through it failed */

/* A region of a chunk, claimed: the way, the chunk, where in it, how long. */
typedef struct sys__pad__hold { unsigned way, chunk; uint64_t offset, bytes; } sys__pad__hold;
/* A send's or a fetch's place in its way: reached when the way's copies are past it. */
typedef struct sys__pad__ticket { unsigned way; uint64_t number; } sys__pad__ticket;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_PAD_CUH */
