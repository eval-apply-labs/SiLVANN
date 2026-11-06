#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_DICTIONARY_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_DICTIONARY_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap.cuh"

/* The dictionary itself: a head and four nodes — two trees, one of them the front readers read, the other
 * the back the writer writes, and the write the back has not had yet (▶ `cpu/dictionary.cuh`, "TWO TREES"). */
#define SYS__DICTIONARY__NODES  5u
/* node 1, the trees: tree t's root (a leaf, a routing node, or nothing), its live keys, its removals since it
 * was last rebuilt */
#define SYS__DICTIONARY__TREES   1u
#define SYS__DICTIONARY__ROOT_0  0u
#define SYS__DICTIONARY__ROOT_1  1u
#define SYS__DICTIONARY__LIVE_0  2u
#define SYS__DICTIONARY__LIVE_1  3u
#define SYS__DICTIONARY__STALE_0 4u
#define SYS__DICTIONARY__STALE_1 5u
/* node 2, the sides: which tree is the front, how many readers stand on each, the write pending for the back,
 * and how many keys were ever put — the count, which only grows */
#define SYS__DICTIONARY__SIDES     2u
#define SYS__DICTIONARY__FRONT     0u
#define SYS__DICTIONARY__READERS_0 1u
#define SYS__DICTIONARY__READERS_1 2u
#define SYS__DICTIONARY__PENDING   3u
#define SYS__DICTIONARY__COUNT     4u
/* node 3 the pending write's key and node 4 its value, whole cells, the value held while it waits */
#define SYS__DICTIONARY__PENDING_KEY   3u
#define SYS__DICTIONARY__PENDING_VALUE 4u
/* A removed pair's value: a null whose op code is this mark. ⛳ A STORED NULL IS NOT A REMOVAL — a name bound to
 * a null is bound (the bindings ask `find` whether a spelled name is there) — so the dictionary writes every null
 * it is given with an op code of zero, and only a removal writes this. */
#define SYS__DICTIONARY__REMOVED 0xFFFFFFFFFFFFFFFFull   /* every bit: no fault word, no opcode */
/* what the pending write is */
#define SYS__DICTIONARY__PENDING_NONE   0ull
#define SYS__DICTIONARY__PENDING_PUT    1ull
#define SYS__DICTIONARY__PENDING_REMOVE 2ull
/* a tree is rebuilt from its live keys once its removals since the last rebuild reach its live keys, and at least
 * this many — so a small dictionary is not rebuilt on every removal */
#define SYS__DICTIONARY__REBUILD_MIN 8ull
/* how long a writer waits for a reader still on the back — one that came onto it before the last flip, one
 * lookup long — before it calls the dictionary broken */
#define SYS__DICTIONARY__READERS_TRIES 1048576u

/* A routing node: a head and the node that is its bound. */
#define SYS__TREE_ROUTING_NODE__NODES  2u
#define SYS__TREE_ROUTING_NODE__LEFT   1u   /* args[1] of the head — args[0] is the lock in every kind */
#define SYS__TREE_ROUTING_NODE__RIGHT  2u   /* args[2] of the head                                     */
#define SYS__TREE_ROUTING_NODE__HEIGHT 3u   /* args[3] of the head: one more than its taller child     */

/* A leaf: an array granule, its head counting the pairs. */
#define SYS__DICTIONARY_LEAF__NODES  SYS__ALLOC__ARRAY_GRANULE
#define SYS__DICTIONARY_LEAF__USED   1u     /* args[1] of the head: how many pairs are in it */
#define SYS__DICTIONARY_LEAF__PAIRS  ((SYS__DICTIONARY_LEAF__NODES - 1u) / 2u)

/* args[1] of the dictionary's HEAD — args[0] is the lock in every kind: set once by `seal`. */
#define SYS__DICTIONARY__READONLY 1u

/* How deep a descent may go before it is taken for a broken tree; `cpu/dictionary.cuh` says what it bounds. */
#define SYS__DICTIONARY__DEPTH_MAX   48u

static_assert(SYS__DICTIONARY_LEAF__PAIRS >= 2u,
              "a leaf that holds fewer than two pairs cannot split into two leaves that each hold one");

#define SYS__DICTIONARY__FAULT_KIND     0x4443544Eull   /* "DCTN" — asked of something that is Not one  */
#define SYS__DICTIONARY__FAULT_NO_KEY   0x4443544Bull   /* "DCTK" — handed no Key                        */
#define SYS__DICTIONARY__FAULT_NO_VALUE 0x44435456ull   /* "DCTV" — handed no Value                      */
#define SYS__DICTIONARY__FAULT_DEEP     0x44435444ull   /* "DCTD" — a descent past any sound tree's Depth */
#define SYS__DICTIONARY__FAULT_READONLY 0x44435452ull   /* "DCTR" — a write to a Read-only one            */
#define SYS__DICTIONARY__FAULT_WAITED   0x44435457ull   /* "DCTW" — a reader on the back never left: Waited  */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_DICTIONARY_CUH */
