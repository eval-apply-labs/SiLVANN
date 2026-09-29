#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_DICTIONARY_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_DICTIONARY_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap.cuh"

/* The dictionary itself: a head and the node holding its two fields. */
#define SYS__DICTIONARY__NODES  2u
#define SYS__DICTIONARY__ROOT   0u   /* args[0] of the payload: the root, or nothing when it is empty */
#define SYS__DICTIONARY__COUNT  1u   /* args[1] of the payload: how many keys it holds                */

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

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_DICTIONARY_CUH */
