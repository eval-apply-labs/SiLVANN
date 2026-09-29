#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PACKAGE_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PACKAGE_CUH

/* This file needs nothing: every name in it is its own. */
#define NN__PACKAGE__SPAN_TERMS 7   /* buffers · L1 arena · cartridge tiers · full-attention layers ·
                                     * the conversation's text · the deltanet state · the layer
                                     * types array, which sizes nothing and ORDERS everything  */

/* ── THIS PACKAGE'S OWN DEVICE-WIDE STATE — `hatch[nn][row]` ────────────────────────────────────────
 *
 * ⚖ RULED: a package's own state is a SUB-ARRAY behind its hatch entry, indexed by its own
 * enum, so *"finding values is system_register[PACKAGE_ID][SETTING_ENUM_ID]"*. Row 0 is the room and is
 * the boot's; these start after it and `sys` never learns their names.
 *
 * ⛳ WHY A ROW RATHER THAN A BINDING, WHICH IS THE TEST `system_register__header.cuh` SETS: *"would a
 * second copy of it still BE it"*. Two copies of the buffer class table are two free masks over one run
 * of bytes, and the second holder walks straight past the first — the same argument that puts a lock
 * here. So it is not shadowable, and a per-scope binding would be incorrect rather than merely slower.
 *
 * ⛳ THE FIRST TWO ARE WRITTEN BY THE OPCODE INIT'S CARVE and read by every allocation and every release.
 * They are two rows rather than two halves of one because they answer different questions at different
 * rates: a `getnew` reads every size to pick a class and then one list; a release reads one list and no
 * size.
 * ⛳ THE THIRD IS A DIFFERENT KIND OF ROW AND IT IS WORTH SAYING SO: the buffer rows name OBJECTS this
 * package made, and the arena row is a pair of INTEGERS naming a region of memory it already had. So it
 * holds no reference, needs no teardown, and its `VALUE_INT` is the same shape the hatch's own ROOM row
 * uses — which is the point, because it is the same question asked of a region instead of an allocation.
 * ⛔ IT IS WRITTEN AFTER THE CARVE AND NOT BESIDE IT: the arena begins where the carve's running total
 * ENDS, so the order is a dependency. ▶ `expert__header.cuh`. */
#define nn__HATCH_ROW_LIST(X)                                                                           \
    X(NN__HATCH__BUFFERS)        /* a node_array on the buffer types: each entry that class's FREE LIST */ \
    X(NN__HATCH__BUFFER_SIZEOF)  /* a node_array on the buffer types: each entry a heap int, its bytes  */ \
    X(NN__HATCH__EXPERT_ARENA)   /* a VALUE_INT: where the expert L1 arena starts, and its bytes        */ \
    /* ⭐⭐⭐ THE TYPE TABLE — a node_array on the matrix types, each entry one collection's run: where it \
     * starts, how many slots, one slot's bytes, and where its up half ends. ⚖ *"one page collection per   \
     * matrix type"*, and ⚖ the offset lives *"in the page list descriptor, not the single page"*.         \
     * ⛳ IT IS WHAT MAKES A SLOT'S ADDRESS ARITHMETIC, and therefore what empties the page node. */        \
    X(NN__HATCH__EXPERT_TYPES)   /* a node_array on the types: AT · SLOTS · BYTES · OFFSET · FREE      */ \
    /* ⭐ THE FREE LISTS' LINKS, IN THE HEAP. They were threaded through the slots' own first eight bytes, \
     * which on a card is memory the CPU evaluator cannot touch; a link is a slot NUMBER, so nothing about \
     * it needed the card. A node_array on the types, each entry a node_array of pages of              \
     * NN__EXPERT__LINKS_PER_PAGE links — two levels because one node_array must fit one chunk and a real \
     * cache has thousands of slots. The head is still the type row's FREE word; only the links moved.  */ \
    X(NN__HATCH__EXPERT_LINKS)   /* a node_array on the types: each entry that collection's link pages */ \
    /* ⭐⭐ THE HANDED LIST — a `sys__list` of slots RESERVED and not yet published. ⚖ *"mark that handed    \
     * memory address into a handed list, so when the abi does the assignment you can revoke it from there  \
     * and you'll have the list of hanging references."* ⇒ ★★ THE LEAK BECOMES AN OBSERVABLE SET RATHER     \
     * THAN A HAZARD: reserve appends, assign removes, and what remains IS the outstanding set — so         \
     * "reserved but never published" is not a state anyone has to DETECT, it is a list anyone can READ.  */ \
    X(NN__HATCH__EXPERT_HANDED)  /* a sys__list: one node per handed slot — address · type             */ \
    X(NN__HATCH__EXPERTS)        /* the expert index's OUTER node_array, one slot per MODEL layer   */ \
    X(NN__HATCH__EXPERT_LRU)     /* a node_array on the layers: each that layer's LRU two ends     */ \
    /* ⭐ THE CARTRIDGE'S FOUR. The first three are REGIONS — a base and a length naming memory this       \
     * package already had — so they are `VALUE_INT`, hold nothing and need no teardown, exactly like the  \
     * arena row above. Only the tier table is an object.                                                  \
     * ⛳ THE TIER TABLE IS THE ONE ROW HERE THAT CHANGES AFTER THE BOOT: a rotation writes its cursors,    \
     * and a rollback writes them back. ⚖ *"an opcode that does a rollback and resets the cursors"* —      \
     * which is affordable precisely because a tier's whole mutable state is six words of one node. */     \
    X(NN__HATCH__CARTRIDGE)          /* a VALUE_INT: the region's base, its bytes, its layer count     */ \
    X(NN__HATCH__CARTRIDGE_TIERS)    /* a node_array on the tiers: each entry that tier's six cursors  */ \
    X(NN__HATCH__CARTRIDGE_LAYERS)   /* a node_array on the MODEL's layers: each its mechanism + lane  */ \
    X(NN__HATCH__CARTRIDGE_TEXT)     /* a VALUE_INT: where the conversation's text is, and its bytes   */ \
    X(NN__HATCH__CARTRIDGE_DELTANET) /* a VALUE_INT: the deltanet state region, and its bytes          */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PACKAGE_CUH */
