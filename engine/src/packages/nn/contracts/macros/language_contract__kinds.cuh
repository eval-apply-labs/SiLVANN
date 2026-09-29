#ifndef SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__KINDS_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__KINDS_CUH

/* This file needs nothing: a macro body names nothing until something expands it. */
/* ── WHAT A TYPE ROW WILL CARRY ──────────────────────────────────────────────────────────────────────
 * Four columns: the package forwarded in, the kind's own number counted from zero, the spelling a host
 * and a program's author use, and the constant this tree compares. A row here is a kind's IDENTITY and
 * says nothing about what it DOES — so a kind is named and numbered first, and being allocatable is
 * something some of them also are. The language has twenty-seven of these against twelve object rows,
 * and the difference is not an oversight: a cell tag and a value are never carved.
 *
 * ⛳ A KIND NEEDS BOTH HALVES OF A ROW HERE BEFORE IT IS OF ANY USE OUTSIDE THE DEVICE. The number is
 * what the dispatch compares; the spelling is the only way anything off the card can say which kind it
 * means, and every kind a test writes is resolved through the map the host publishes. A number on its
 * own dispatches correctly and cannot be named, which leaves no way to write the check that it worked.
 * Both tables are gathered across the roster, so this package appears in them by being listed.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

#define nn__LANGUAGE_CONTRACT__KINDS(X, PKG)                                                           \
    /* The value type of a model program: what a compute primitive takes and what one answers.         */ \
    /* Its bytes come from a pool this package owns rather than from the heap, which is why its        */ \
    /* OBJECTS row names a provider for the 64-byte reference and nothing at all for the run it        */ \
    /* points into: that run was carved out of nn's span before any program ran.                       */ \
    /* ⛳ AND IT NOW HAS ITS OBJECTS ROW TOO, because the pool it is carved from exists. A kind keeps    */ \
    /* its KINDS row either way: that row is its IDENTITY and says nothing about how it is built.      */ \
    X(PKG,  0, "nn__buffer",          NN__KIND__BUFFER)                                                \
    /* A place in the conversation's KV cache: an address, and the FORMAT the bytes there are in.      */ \
    /* ⛔ ITS BYTES ARE NOT THE HEAP'S AND ARE NOT THIS OBJECT'S EITHER — they belong to the cartridge, */ \
    /* which outlives every reference to it. ⚖ *"the kv cache is not deallocated"*, so this kind names */ \
    /* bytes without owning them, and its OBJECTS row names no release hook at all.                    */ \
    /* ⭐ THE FORMAT TRAVELS WITH THE ADDRESS BECAUSE A SUBSECTION MAY CROSS A TIER: ⚖ *"(kv_subsection */ \
    /* ((from to) (from to) (from to))) to pick a subsection of the kv cache even between different    */ \
    /* types"* — a consumer handed two formats and no way to tell them apart can decode neither.       */ \
    X(PKG,  1, "nn__kv_ref",          NN__KIND__KV_REF)                                                \
    /* ⭐⭐ WEIGHTS RESIDENT IN A PAGE — an address and a length, and it owns NEITHER. ⚖ RULED           */ \
    /* *"a second kind, minted from the index"*, and ⚖ S99 named the shape long before:     */ \
    /* *"experts will not be in the sys heap but only their references will be there as a special heap  */ \
    /* node"*. ⛳ IT IS THE SAME OBJECT `nn__kv_ref` IS, pointed at a different owner: the page holds    */ \
    /* the bytes and outlives every name for them, so release has nothing to give back.                 */ \
    /* ⛔ WHY A SECOND KIND RATHER THAN A FLAG ON `nn__buffer`: a release hook is per KIND, so one kind  */ \
    /* that sometimes owns its bytes and sometimes does not is a bit every reader carries at every      */ \
    /* site — and getting it wrong is a leak with no event. That argument is already written two        */ \
    /* paragraphs down about `nn__kv_ref`; this is the same call, made the same way, a second time.     */ \
    X(PKG,  2, "nn__weights",         NN__KIND__WEIGHTS)

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__KINDS_CUH */
