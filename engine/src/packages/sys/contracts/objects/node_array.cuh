#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_NODE_ARRAY_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_NODE_ARRAY_CUH

/* This file needs nothing: every name in it is its own. */
/* The smallest array there can be: a head and one element. It is what the kind's row states, so making
 * one without saying how many gives an array of one rather than an array of nothing. */
#define SYS__NODE_ARRAY__NODES_MIN 2u

#define SYS__NODE_ARRAY__FAULT_EMPTY 0x4E415245ull   /* "NARE" — asked for an array with no elements  */
#define SYS__NODE_ARRAY__FAULT_RANGE 0x4E41524Full   /* "NARO" — an element Outside the array         */
#define SYS__NODE_ARRAY__FAULT_KIND  0x4E41524Eull   /* "NARN" — Not one, or a placement refused      */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_NODE_ARRAY_CUH */
