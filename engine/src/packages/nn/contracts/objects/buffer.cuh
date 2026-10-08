#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_BUFFER_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_BUFFER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../sys/contracts/objects/package.cuh"
/* ⚖ FIVE, RULED — *"im cool with 5"*. One class per distinct buffer ROLE, so a request matches
 * a class exactly and the round-up waste is zero. ⛳ THE ORDER IS THE ID ORDER AND THE LAYOUT ORDER, and
 * the first three are v1's `main/resid/alt` in the roles they actually held. */
#define NN__BUFFER_CLASS_LIST(X)                                                                        \
    X(0, resid,     NN__BUFFER__RESID)          /* the residual stream, carried the length of a layer  */ \
    X(1, main,      NN__BUFFER__MAIN)           /* the widest intermediate — a GEMM's contiguous out   */ \
    X(2, attn_out,  NN__BUFFER__ATTN_OUT)       /* attention's answer, before the projection           */ \
    X(3, logits,    NN__BUFFER__LOGITS)         /* the vocabulary-wide one, live only at the head      */ \
    X(4, partials,  NN__BUFFER__PARTIALS)       /* flash-decoding's per-chunk partials, grid-wide      */

/* The ids a class is named by. Nobody writes the count: it is one per row, so a list that grows and a
 * count that did not cannot happen. */
#define NN__BUFFER__ZZPRIVATE_ENUM_ROW(id, name, konst)   konst = (id),
#define NN__BUFFER__ZZPRIVATE_TALLY_ROW(id, name, konst)  + 1u

enum {
    NN__BUFFER_CLASS_LIST(NN__BUFFER__ZZPRIVATE_ENUM_ROW)
    NN__BUFFER__CLASSES = 0u NN__BUFFER_CLASS_LIST(NN__BUFFER__ZZPRIVATE_TALLY_ROW)
};

/* ⛔ THE DENSE-AND-ORDERED CHECK. A second enum takes each row's POSITION — no id column reaches it — and
 * every row then asserts that its stated id is that position. A row inserted in the middle without
 * renumbering, or a copied id, stops the build here and names the class it is about. */
#define NN__BUFFER__ZZPRIVATE_POS_ROW(id, name, konst)  NN__BUFFER__ZZPRIVATE_POS__##name,
enum { NN__BUFFER_CLASS_LIST(NN__BUFFER__ZZPRIVATE_POS_ROW) };

#define NN__BUFFER__ZZPRIVATE_DENSE_ROW(id, name, konst)                                                \
    static_assert((int)(konst) == (int)NN__BUFFER__ZZPRIVATE_POS__##name,                               \
                  "nn buffer class ids must be dense and in list order — the class table is indexed by "  \
                  "the id and the runs are laid out in list order, so an id that is not its own "        \
                  "position hands a class another class's bytes and every total stays right.");
NN__BUFFER_CLASS_LIST(NN__BUFFER__ZZPRIVATE_DENSE_ROW)

/* The row macros have done their work and nothing outside this file expands them. They are taken back so
 * that what this header leaves behind is the list, the ids, the count and the keys — and not five names a
 * reader would have to check the emptiness of. */
#undef NN__BUFFER__ZZPRIVATE_ENUM_ROW
#undef NN__BUFFER__ZZPRIVATE_TALLY_ROW
#undef NN__BUFFER__ZZPRIVATE_POS_ROW
#undef NN__BUFFER__ZZPRIVATE_DENSE_ROW

/* What the paired sum is handed, so the check and the walk cannot drift apart in how they spell a key. */
/* ⭐ THE CLASSES THAT LIVE IN PINNED RAM THE CARD CAN WRITE — ⚖ RULED: *"the buffers need to be
 * like nn buffer pool, so out of the heap but in ram."* Their ids CONTINUE after the card's, so every table
 * indexed by class serves both; they are carved from the room's RAM span; and a program asks for one by
 * name (`nn__buffer__getnew_ram`) — `getnew` chooses among the card's classes only, so a small ask never
 * lands in RAM by accident. ⛳ OPTIONAL IN A CONFIG, unlike the card's: a file naming none has no RAM span.
 * Their keys have their own prefix, because the card classes' check sums every `nn__buffer_` key. */
#define NN__BUFFER_RAM_CLASS_LIST(X)                                                                    \
    X(5, result,    NN__BUFFER__RESULT)         /* where a value verb's answer lands — ▶ `result.cuh`  */ \
    X(6, experts,   NN__BUFFER__EXPERTS)        /* a layer's routed experts, the store a prompt streams */ \
                                                /* to the card from — ▶ ai_qwen_3's `moe_rows`          */ \
    X(7, table,     NN__BUFFER__TABLE)          /* a table the card reads in place, a row at a time —   */ \
                                                /* a model's embedding, so a program embeds a token     */

#define NN__BUFFER__ZZPRIVATE_RAM_ENUM_ROW(id, name, konst)   konst = (id),
#define NN__BUFFER__ZZPRIVATE_RAM_TALLY_ROW(id, name, konst)  + 1u
enum {
    NN__BUFFER_RAM_CLASS_LIST(NN__BUFFER__ZZPRIVATE_RAM_ENUM_ROW)
    NN__BUFFER__ALL_CLASSES = NN__BUFFER__CLASSES NN__BUFFER_RAM_CLASS_LIST(NN__BUFFER__ZZPRIVATE_RAM_TALLY_ROW)
};
#define NN__BUFFER__ZZPRIVATE_RAM_POS_ROW(id, name, konst)  NN__BUFFER__ZZPRIVATE_RAMPOS__##name,
enum { NN__BUFFER__ZZPRIVATE_RAMPOS__ZZBASE = (int)NN__BUFFER__CLASSES - 1,
       NN__BUFFER_RAM_CLASS_LIST(NN__BUFFER__ZZPRIVATE_RAM_POS_ROW) };
#define NN__BUFFER__ZZPRIVATE_RAM_DENSE_ROW(id, name, konst)                                            \
    static_assert((int)(konst) == (int)NN__BUFFER__ZZPRIVATE_RAMPOS__##name,                            \
                  "nn's RAM buffer classes must follow the card's, dense and in list order");
NN__BUFFER_RAM_CLASS_LIST(NN__BUFFER__ZZPRIVATE_RAM_DENSE_ROW)
#undef NN__BUFFER__ZZPRIVATE_RAM_ENUM_ROW
#undef NN__BUFFER__ZZPRIVATE_RAM_TALLY_ROW
#undef NN__BUFFER__ZZPRIVATE_RAM_POS_ROW
#undef NN__BUFFER__ZZPRIVATE_RAM_DENSE_ROW

#define NN__BUFFER__KEY_RAM_PREFIX  "nn__ram_buffer_"
#define NN__BUFFER__KEY_PREFIX      "nn__buffer_"
#define NN__BUFFER__KEY_TAIL_SIZE   "__size_bytes"
#define NN__BUFFER__KEY_TAIL_QTY    "__qty"

/* ══ WHAT A BUFFER IS — A POINTER AND ITS CLASS, AND NOTHING ELSE ════════════════════════════════════
 *
 * ⚖⚖ RULED: *"i think buffer_reference could be a pure pointer, with the buffer type. no one
 * is going to access it by index from the heap."*
 *
 * ⭐⭐ SO A SLOT HAS NO NUMBER, AND THAT IS WHAT THE FREE LIST BOUGHT. A mask needs an index because a bit
 * position IS the name of a slot; a list holds the buffers themselves, so the ADDRESS is the identity and
 * release has nothing to look up — it reads the class out of the node and files the node in that class's
 * list. ⛳ `buffer[kind][num]` survives as a shape — `hatch[nn][BUFFERS][kind]` — with the second tier a
 * list rather than an indexed run.
 *
 * ── ⭐⭐ AND ITS RELEASE IS NOT A DEALLOCATE ─────────────────────────────────────────────────────────
 * ⚖ *"its deallocate is not a real deallocate it just reinserts itself in the freemem list for its buffer
 * type."* That maps onto machinery already here rather than needing any: `zzprivate_deallocate` runs a
 * kind's release hook and only then asks *"is it still dead"* —
 *     `if (SYS__HEAP_OBJECT__COUNT(head) == 0u) { give the room back }`
 * — a door the computing base already uses, because *"a release reaching it is a mistake and it outlives
 * everything"*. Filing a buffer into its list TAKES A HOLD, so the count goes 0 → 1 and the room is simply
 * not reclaimed. ⇒ ★ THE RESURRECTION IS NOT A TRICK PLAYED ON THE ALLOCATOR; IT IS THE ALLOCATOR'S OWN
 * QUESTION ANSWERED HONESTLY. The object is not dead — the free list wants it.
 * ⛳ WHICH IS ALSO WHY NOTHING NEEDS A GENERATION COUNTER. A release only fires at ZERO HOLDERS, so at the
 * instant a buffer rejoins its list nobody is holding it; a stale reference cannot exist to be detected.
 * The slab designs that carry generations recycle a slot independently of the references to it, and this
 * one cannot.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#define NN__BUFFER__AT     0u   /* args[0]: the address — a pure pointer into `nn`'s span */
#define NN__BUFFER__CLASS  1u   /* args[1]: which class, so a release knows which list is its home */
/* ⭐ WHERE THE PROGRAM SEES THE SAME BYTES, AND WHICH ANSWER IT IS WAITING FOR — ▶ `result.cuh`. A buffer
 * in mapped pinned RAM has two addresses, the card's (`AT`, what every door is handed) and the program's
 * (`HOST_AT`); a buffer only the card can reach carries 0 there. `STAMP` is the stamp of the last value
 * a verb was asked to write into it, 0 before any. */
#define NN__BUFFER__HOST_AT 2u   /* args[2]: the program's address for the same bytes, or 0             */
#define NN__BUFFER__STAMP   3u   /* args[3]: the stamp a value written here must carry, or 0            */
/* ⭐ WHOSE IT IS — ⚖ *"build owner check"*: the worker whose pool carved it, plus one, so 0 reads "nobody said"
 * and refuses nothing. A verb is handed only its own worker's buffers (`nn__primitives__room`); the one crossing is
 * `nn__buffer__copy`, which reads its source whatever worker holds it. */
#define NN__BUFFER__OWNER   4u   /* args[4]: the owning worker plus one, or 0                           */
#define NN__BUFFER__NODES  2u   /* the head, and one node to hold those words */

#define NN__BUFFER__FAULT_NO_POOL   0x4E424E50ull   /* "NBNP" — asked for a buffer before the carve */
#define NN__BUFFER__FAULT_EXHAUSTED 0x4E424558ull   /* "NBEX" — the class has none free right now   */
#define NN__BUFFER__FAULT_SIZE      0x4E42535Aull   /* "NBSZ" — no class is big enough for this ask */
#define NN__BUFFER__FAULT_CONFIG    0x4E424346ull   /* "NBCF" — the card cannot read what the host summed */
#define NN__BUFFER__FAULT_ROOM      0x4E42524Dull   /* "NBRM" — the carve does not fit the span it was given */
#define NN__BUFFER__FAULT_OWNER     0x4E424F57ull   /* "NBOW" — a buffer of another worker, handed to this one's verb */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_BUFFER_CUH */
