#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_STRING_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_STRING_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "heap_node.cuh"

/* The head and the hash: what an empty string is, and so the floor the kind's row states. */
#define SYS__STRING__NODES_MIN 2u
/* args[1] of the head: how many BYTES of text there are. */
#define SYS__STRING__LENGTH    1u
/* ⛔ THE ONE PLACE IN A PACKAGE THAT NAMES A NODE'S WIDTH IN BYTES, AND IT IS PINNED. Text is bytes, so
 * turning a length into nodes needs the width, and the hash is exactly one node, so a node of any other
 * width breaks the layout above. ⚖ ARCHITECT: *"accept the static_assert exception, so at least changing
 * the size raises the check engine light."* The claim gate's `node_width_only_at_the_seam` admits this
 * file's arithmetic only while this line stands. */
static_assert(sizeof(sys__heap_node) == 64u,
              "a string's hash is exactly one node and its text is 64 bytes to a node: a node of another "
              "width breaks string.cuh's layout and its byte arithmetic, so change them together");

/* The most text one string can hold, derived from how wide a chunk is — see above. */
#define SYS__STRING__BYTES_MAX (((uint64_t)SYS__HEAP__CHUNK_NODES - (uint64_t)SYS__CHUNK__FIRST          \
                                 - (uint64_t)SYS__STRING__NODES_MIN) * (uint64_t)sizeof(sys__heap_node))

#define SYS__STRING__FAULT_KIND     0x5354524Eull   /* "STRN" — asked of something that is Not a string */
#define SYS__STRING__FAULT_LONG     0x5354524Cull   /* "STRL" — too Long to fit one chunk              */
#define SYS__STRING__FAULT_NO_BYTES 0x53545242ull   /* "STRB" — text promised and no Bytes handed in   */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_STRING_CUH */
