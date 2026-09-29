#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_FAULT_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_FAULT_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>

/* ── WHAT A FAULT IS, ONCE IT HAS LANDED ─────────────────────────────────────────────────────────────
 *   raised   how many were. ⛔ IT IS NOT ATOMIC, AND THE PROPERTY THAT MATTERS SURVIVES THAT: two blocks
 *            raising at once can both read the same number and both write one past it, so the count can
 *            UNDERSTATE a storm — but neither of them can write a zero, so "nothing went wrong" is exact
 *            and only the magnitude is approximate. ⛳ THE PRICE OF EXACTNESS IS SMALL, AND IT IS SAID
 *            HERE SO THE TRADE IS NOT ARGUED FROM A COST THE TREE DOES NOT HAVE: `raise` is kept out of
 *            line (▶ `cpu/fault__impl.cuh`), so anything put inside it exists ONCE in the module
 *            rather than once per refusing verb, and the seam carries `sys__silicon__add_u32`, a
 *            single instruction and not a retry loop. One atomic add on a cold path is the whole
 *            bill. `REASONED` — the plain add stands because nobody asks this for the magnitude, not
 *            because an atomic would cost.
 *   pc       the program counter of the LAST raiser, and `op` its word. Last rather than first because
 *            nothing has asked for the first one; a cascade is visible in the count either way.
 *            ⚠ Two faults therefore leave one context, which is the known limit of a single-slot
 *            channel and the reason a per-computation fault exists beside it — a refusal that
 *            belongs to ONE program is reported through the computing base, and this is for the case
 *            where there is nobody to return to.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
typedef struct {
    unsigned int raised;
    uint64_t     pc;
    uint64_t     op;
} sys__fault_channel;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_FAULT_CUH */
