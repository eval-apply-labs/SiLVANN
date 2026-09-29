#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SYMBOL_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SYMBOL_CUH

/* This file needs nothing: every name in it is its own. */
/* What a refused registration answers: a number no name is ever dealt. */
#define SYS__SYMBOL__NONE (~0ull)

#define SYS__SYMBOL__FAULT_NO_TABLE  0x53594D54ull   /* "SYMT" — no symbol Table is standing              */
#define SYS__SYMBOL__FAULT_COLLISION 0x53594D43ull   /* "SYMC" — two texts, one hash: a Collision          */
#define SYS__SYMBOL__FAULT_UNKNOWN   0x53594D55ull   /* "SYMU" — a number no name was dealt: Unknown       */
#define SYS__SYMBOL__FAULT_LOOSE     0x53594D4Cull   /* "SYML" — a string name whose string is Loose, not
                                                        the one the table holds for its text          */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SYMBOL_CUH */
