#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_EXPERT_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_EXPERT_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../sys/contracts/objects/heap.cuh"
#include "../../../sys/contracts/objects/package.cuh"
#define NN__EXPERT__FAULT_CONFIG  0x4E454346ull   /* "NECF" — the card cannot read what the host summed */
#define NN__EXPERT__FAULT_ROOM    0x4E45524Dull   /* "NERM" — the arena does not fit what is left       */

/* ⛔⛔ 4 KiB, AND THIS COMMENT CLAIMED A REQUIREMENT THE SHIPPING BUNDLE REFUTES — CORRECTED.
 * It read *"IT IS THE DISK PATH'S REQUIREMENT RATHER THAN A PREFERENCE — the O_DIRECT / pinned-DMA read
 * path needs every sub-slot aligned to it."*
 * `MEASURED` on `/mnt/data/bigdisk/qwen36_35b_v3`, the per-expert SOURCE strides:
 * ```
 *     up_data   2,097,152  aligned      down_data  1,048,576  aligned
 *     up_lut        2,048  NOT          down_lut       4,096  aligned
 * ```
 * ⇒ one of the four is already unaligned at source, so either the loader is not O_DIRECT or it reads a
 * lut field whole and slices it in host memory. Either way *"every sub-slot"* is false.
 * ⛔ AND THE DEEPER REASON IT DOES NOT APPLY: a slot boundary is a VRAM DESTINATION address. O_DIRECT
 * constrains file offsets, transfer lengths and HOST buffer addresses; padding a half moves where the
 * next half lands on the card and changes no file offset at all. Nothing here reads disk into VRAM
 * directly, so this alignment buys nothing measurable today.
 * ⚖ IT IS KEPT ANYWAY, RULED — *"pad at the boundary then"* — because it costs 0.065% and it
 * is what a direct-landing path would need. `ASSUMED` that such a path is worth insuring against;
 * FALSIFIED BY a measurement showing an aligned and an unaligned slot transfer at the same rate, which
 * nothing in this tree has taken. ⇒ ★ A THRIFT AND A REQUIREMENT READ THE SAME ONCE WRITTEN DOWN, and
 * this one was read as a requirement for as long as it stood. */
#define NN__EXPERT__PAGE_ALIGN  4096ull

/* The arena's row holds the same two words the hatch's ROOM row does, and by the same names — it is the
 * same question asked about a region rather than about an allocation. */
#define NN__EXPERT__ARENA_AT     SYS__PACKAGE__HATCH_AT
#define NN__EXPERT__ARENA_BYTES  SYS__PACKAGE__HATCH_BYTES

/* ══ ⭐⭐⭐ THE TYPE TABLE — THE PAGE LIST DESCRIPTOR, AND WHAT REPLACES THE SIX SIZE CLASSES ═════════
 *
 * ⚖⚖ RULED: *"since we do not work with variable bitrates we have one expert per page, and one
 * page collection per matrix type. this way there is no page walk to be done, there is only one lru of the
 * expert and one free list of pages (now more correctly described as slots)."*
 *
 * ⭐⭐ THE QUANT DECISION IS WHAT PAYS FOR IT, AND IT IS WORTH SAYING WHY THE OLD MACHINERY EXISTED: six
 * size classes are an answer to *"how do I fit things that DIFFER"*, and they stopped differing. The width
 * is one number per TENSOR, so every expert of a matrix type is the same size — and once that holds, a
 * slot's address is `base + i x slot_bytes`, which is ARITHMETIC and not a stored field.
 * ⇒ ★★ THE PAGE NODE HAS NOTHING LEFT TO HOLD. The LRU is the expert's, the vacancy byte went with
 * sharing, the class is the collection you are already in, and the offset is this row's.
 * ⛳ AND IT IS THE SHAPE `activation_buffer_pool.md` ALREADY RULES FOR BUFFERS — *"bytes as a run whose
 * address is arithmetic"*. Two structures arriving at one shape from opposite ends is the cheapest
 * evidence there is that the shape is right.
 *
 * ── ⚖ WHAT A ROW HOLDS, AND WHY THE OFFSET IS HERE AND NOT ON A PAGE ────────────────────────────────
 * ⚖ ARCHITECT: *"it is possible that the up and down gate have different quants and therefore have
 * different sizes. what we could do is save the offset somewhere else and then we can gather the up from
 * the base until the offset and the down from the page start plus offset to the end"* — and, asked where
 * *"somewhere else"* is: *"i mean the page list descriptor, not the single page."*
 * ⇒ `MEASURED` on `/mnt/data/bigdisk/qwen36_35b_v3`, all 40 layers: ONE geometry — gate_up
 * 1024 rows x 2048 B + 2 B/row of lut, down 2048 x 512 + 2 B/row — so the offset is a SINGLE distinct
 * value that a per-page field would have written 10,240 times. A fact stored once cannot disagree with
 * itself. ⛳ The per-half quants are independent BY CONSTRUCTION: the offset IS up's size, and the
 * remainder is down's, so two different widths need nothing beyond this one word.
 * ```
 *   [0] AT      the run's base — a pointer into the arena, and the only address a slot needs
 *   [1] SLOTS   how many slots the run holds
 *   [2] BYTES   one slot, padded — every slot in this collection is this size, which is the whole ruling
 *   [3] OFFSET  where the first half ends and the second begins — an aligned boundary inside the slot,
 *               and a GEOMETRY rather than a meaning: what sits either side is the collection's business
 *   [4] FREE    the free head. ⛳ WRITTEN BY THE FREE LIST, WHICH IS NOT THIS PIECE — it stands at
 *               `NN__EXPERT__TYPE_FREE_NONE` here and the next piece is what fills it.
 *   [5] EVICTABLE whether this collection's slots may ever be taken back. ⚖ RULED — ARCHITECT:
 *               *"the eviction can be a property of the page list so we can maintain some tables as
 *               anchored"*. ⛳ STORED AS THE INVERSE OF WHAT IT IS CALLED — ▶ the polarity note below.
 * ```
 * ⛳ THAT WORD WAS SPARE AND ITS COMMENT SAID IT WOULD STAY SPARE — *"the room freed here is a dividend of
 * the collapse and not an opportunity to find tenants for"*. That argument was about FINDING a tenant for
 * empty room; this is a RULED property of a collection arriving and needing somewhere to live. The
 * distinction is the one the old comment was defending, so it is honoured rather than overridden.
 *
 * ── ⛔⛔ AND THE POLARITY IS THE DESIGN: **ABSENT MEANS ANCHORED**, WHICH IS BACKWARDS FROM EVERY OTHER KEY
 * A collection is anchored unless a config says `anchored:0` in so many words. ⛳ THE REASON IS THIS TIER'S
 * OWN LIMIT, argued four paragraphs up: the settings readers answer a BOOL, so they **cannot distinguish an
 * ABSENT key from a MALFORMED one**. Under the ordinary polarity (absent ⇒ evictable) a mistyped key name,
 * a bad value, or a forgotten line all read as *"you may take this slot back"* — and for a DeltaNet state
 * that is THE ONLY COPY OF SOMETHING NO RE-READ CAN RECOVER. Under this polarity every one of those
 * failures lands on *"anchored"*: the card fills and `place` REFUSES, loudly, at the moment of the mistake.
 * ⇒ ★★ WHERE A READER CANNOT TELL ABSENT FROM WRONG, POINT THE DEFAULT AT THE FAILURE YOU CAN SURVIVE.
 * ⛳ It costs nothing today — no eviction policy stands yet (▶ `place`) — so nothing in the tree changes
 * behaviour; what it buys is that the dangerous direction can never be reached by an edit that went wrong.
 * ⭐⭐ AND THE WORD IS STORED AS `EVICTABLE` WHILE THE ACCESSOR IS CALLED `anchored`, WHICH IS THE WHOLE
 * TRICK AND NOT A NAMING SLIP. Every way the reader can fail — no table stood, a type past the end, a row
 * of the wrong dtype, a word never written — answers **zero**, and zero means *not evictable*. So the safe
 * answer is what the mechanism produces when it is broken, rather than a case somebody remembered to write.
 * ⇒ ★ A DEFAULT ENFORCED BY A SPECIAL CASE IS ONE `return` AWAY FROM BEING LOST; A DEFAULT THAT IS THE
 * ZERO OF THE REPRESENTATION CANNOT BE.
 *
 * ── ⚖⚖ THE PAD GOES AT THE BOUNDARY — RULED, AND IT IS WORTH ONE LINE WHY ────────────────
 * `offset = align(up)` and `slot = offset + align(down)`, so ONE pad buys BOTH the down half's alignment
 * and every subsequent slot's. Padding at the END instead costs the identical 2,048 bytes and aligns only
 * the slot start. ⛳ 0.065% on this checkpoint's 3,151,872-byte expert.
 * ⛔ AND IT IS INSURANCE, NOT A REQUIREMENT — ▶ `NN__EXPERT__PAGE_ALIGN`, whose old comment claimed a
 * requirement this bundle refutes.
 *
 * ── ⭐⭐⭐ AND A DENSE TENSOR IS THIS SAME OBJECT, WITH NOTHING ADDED — ⚖ RULED ────────────
 * ⚖ ARCHITECT: *"a dense tensor is a collection with 40 slots, one per layer."* So a `q_proj`, a
 * layernorm or a router gate needs no structure of its own and **no code that is not already here**:
 * it is a collection whose `SLOTS` is the model's layer count and whose slot `i` holds layer `i`'s copy.
 * ```
 *   an EXPERT's slot    up = [gate_up codes | gate_up scales]     down = [down codes | down scales]
 *   a DENSE slot        up = [codes]                              down = [scales]
 * ```
 * ⭐⭐ THE SECOND ROW IS THE PACKED RECORD'S OWN SHAPE — a v3 record is `vbr_data` then `vbr_lut`, and
 * that is exactly a slot's two halves. ⇒ ★ THE SHAPE THAT NEEDED NO NEW MECHANISM IS EVIDENCE THE
 * COLLAPSE WAS THE RIGHT ONE: under the page design a dense tensor would have wanted its own index, its
 * own geometry keys and its own walk.
 *
 * ⛔⛔ AND A ONE-HALF COLLECTION (`down_bytes: 0`) WAS BUILT, MEASURED AND TAKEN BACK OUT — the reason is
 * worth more than the code was. `sys__settings__size` refuses a figure of zero, so a dense type would
 * have had to declare its down half by OMITTING the key; and the device's settings readers answer a
 * BOOL, so they cannot tell an ABSENT key from a MALFORMED one. ⇒ ⛔ `down_byte:1234` — one character
 * wrong — would have read as *"this collection has one half"* and carved every slot at two thirds of its
 * intended size, silently, with the arena's totals all still right.
 * ⇒ ★★ AN OPTIONAL KEY IS ONLY SAFE WHERE THE READER CAN DISTINGUISH ABSENT FROM WRONG, and this tier
 * cannot — `settings.cuh` says so in its own prose, arguing that an init's response to all three is the
 * same. **That argument is false here**, which is the one thing worth keeping from the attempt: absent
 * would have meant *one half* and malformed must mean *refuse*. Nothing needs the capability, so the
 * guard stays and `NN__EXPERT__FAULT_TYPES` goes on meaning what it meant.
 *
 * ⛔⛔ HOW A PROGRAM *NAMES* A DENSE TENSOR IS **NOT** RULED, AND NOTHING HERE DECIDES IT. `plane` reaches
 * an expert through the index as `[layer][expert]`, and a dense tensor at `[layer][0]` would collide with
 * expert 0 of that layer. The three candidates — a per-collection index, a reserved band above the expert
 * count, or pure `(type, layer)` arithmetic with no index at all — differ in what they cost and in what
 * they let the LRU do. ⛳ THE LOADER AND THE COLLECTION DO NOT DEPEND ON THE ANSWER, which is why this
 * piece ships without it. ▶ `RESUME.md`, the open question.
 *
 * ── HOW MANY COLLECTIONS, AND WHY THE CAP IS A LIST RATHER THAN A NUMBER ────────────────────────────
 * A type's keys must be COMPILE-TIME strings, because a settings key is looked up by its bytes and this
 * tier builds no text. So the slots are a generated list and `nn__expert__types` says how many of them are
 * live. ⛳ A model needs one collection per (shape, quant) pair — the contract's *"per shape and per
 * quant"* — which is ONE for this 35B across all 40 layers.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⚖⚖ THIRTY-TWO, RULED — ARCHITECT: *"raise max_types to 32"*. It was EIGHT, and eight was
 * picked when this package held no model: `MEASURED` the same day, the MoE block ALONE stands FOUR
 * (routed gate_up, routed down, shared gate_up, shared down) and a whole 35B layer wants roughly twenty
 * — the four attention projections, the five DeltaNet projections, `conv1d`, `A_log`, `dt_bias`, the two
 * norms, the router gate, embed and lm_head — before DeltaNet's own two collections are counted.
 * ⛳ THE CAP IS A LIST AND NOT A NUMBER because a settings key is looked up by its BYTES, so `type17__slots`
 * has to be a string the compiler wrote; raising it is this line and nothing else. */
#define NN__EXPERT_TYPE_SLOT_LIST(X)                                                                    \
    X(0)  X(1)  X(2)  X(3)  X(4)  X(5)  X(6)  X(7)  X(8)  X(9)  X(10) X(11) X(12) X(13) X(14) X(15)     \
    X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)

#define NN__EXPERT__ZZPRIVATE_TYPE_TALLY(n)  + 1u
enum { NN__EXPERT__MAX_TYPES = 0u NN__EXPERT_TYPE_SLOT_LIST(NN__EXPERT__ZZPRIVATE_TYPE_TALLY) };
#undef NN__EXPERT__ZZPRIVATE_TYPE_TALLY

/* The row's arguments, by name. ▶ the block above for what each one is and which ruling put it there. */
#define NN__EXPERT__TYPE_AT      0u
#define NN__EXPERT__TYPE_SLOTS   1u
#define NN__EXPERT__TYPE_BYTES   2u
#define NN__EXPERT__TYPE_OFFSET  3u
#define NN__EXPERT__TYPE_FREE    4u
#define NN__EXPERT__TYPE_EVICTABLE 5u

/* ⛳ THE FREE HEAD'S EMPTY IS NOT `0`, BECAUSE SLOT 0 IS AN ORDINARY SLOT — the same sentinel problem the
 * expert LRU's links have, bought the same way: the head is stored as `slot + 1`. ▶ `expert_lru_node_array.md` §ⓑ②. */
#define NN__EXPERT__TYPE_FREE_NONE  0ull

#define NN__EXPERT__FAULT_TYPES  0x4E455459ull   /* "NETY" — the type table does not describe the arena */

/* ══ ⭐⭐ THE EXPERT INDEX — WHERE AN EXPERT IS, AND WHEN IT WAS LAST WANTED ══════════════════════════
 *
 * ⚖⚖ RULED COMPLETE (`expert_lru_node_array.md` §⓪, §ⓑ): *"the three things seem something that will be
 * found out in implementation, not rulings on my end."* This is the act of finding them out.
 *
 * ⭐⭐ IT IS ONE STRUCTURE WHERE THE OLD TREE HAD FOUR, AND THE SAVING IS THE INDEXING. The struct ledger
 * files `SilvannExpertRegistry` (166 references, four residency modes multiplexed), `LruService` (328 B,
 * 24 members, 8 of them debug), `MoeDirRow` and `MoeDirTier` all as DISSOLVING into this — and the reason
 * they can is that **the index IS the identity**. The old `MoeDirectory` carries an
 * `unordered_map<uint64_t,int>` keyed on a device pointer, so a lookup is a hash; here `[layer][expert]`
 * names the slot with no search at all.
 *
 * ── ⚖ SPARSE ON THE OUTSIDE, RULED ────────────────────────────────────────────────────────
 * *"lets keep the indexes for everyone but they'd point at a null subarray."* The outer array is sized to
 * the MODEL's layer count, so the index is the GLOBAL layer id on every rank and there is no per-rank
 * translation map to keep in step with a shard plan.
 * ⭐ AND IT PAYS TWICE: a rank that does not HOLD a layer and a layer that has NO EXPERTS are the same
 * thing here — both hold nothing. So the dense-vs-MoE question never has to be asked at all, which
 * matters because the old tree answers it by READING THE WEIGHT FILE (`composer.py:1867`), and an init on
 * the card cannot do that.
 * ⛳ THE LAYER COUNT COMES FROM THE CARTRIDGE'S LAYER TABLE, which is the one thing on this card that
 * knows how many layers the model has. ▶ `cartridge__header.cuh`; it was built the same day as this.
 *
 * ── ⚖ RESIDENCY IS A PROPERTY OF A SLOT, NOT A LIFETIME ──────────────────────────────────────────────
 * *"they still would be resident, an evicted expert gets an offset of 0 and no previous/next."*
 * ⇒ ⭐⭐ NOTHING IS EVER ALLOCATED OR FREED PER EXPERT. The arrays are made once and live as long as the
 * model; an eviction is `AT := 0` plus chain surgery and an admission is the reverse. **There is no
 * per-expert allocation on the hot path at all**, which is the property that makes this cheap.
 * ⛳ AND "EVICTED" BECOMES A ONE-WORD TEST: `AT == 0`.
 *
 * ── ⚖ POINTERS, NOT OFFSETS (`expert_lru_node_array.md` §ⓟ) ─────────────────────────────────────────────
 * `[0]` and `[3]` are ADDRESSES. An offset is for a thing that must survive its base being re-established;
 * an expert's residency does not — it is re-established by loading the expert again. ⇒ the consumer takes
 * the L1 pointer if it is non-zero, the L2 pointer if it is not, and neither means fetch from disk.
 * ⭐ THE MEMORY IS ENCODED BY ARGUMENT POSITION, which is a third answer to the kind-vs-argument question in
 * `src/docs/design/the_heap.md` ⑧ — and the only one that can say an expert is in TWO memories at once, which
 * is the normal state of a warm cache. ══════════════════════════════════════════════════════════════ */

/* A slot's six words, FIVE OF THEM SPOKEN FOR. ⛳ The page design filled all six on purpose while a page
 * was shared; one expert to a slot gives a word back. ▶ `expert_lru_node_array.md` §⓪. */
#define NN__EXPERT__SLOT_AT     0u   /* the L1 address. 0 MEANS NOT IN VRAM, and that is the whole test */
#define NN__EXPERT__SLOT_PREV   1u   /* the LRU, as expert+1 — ▶ the sentinel note below               */
#define NN__EXPERT__SLOT_NEXT   2u
#define NN__EXPERT__SLOT_AT_L2  3u   /* the RAM tier's address. 0 means not there either               */
#define NN__EXPERT__SLOT_TYPE   4u   /* which COLLECTION holds it. ⛔ STATIC — ▶ why, below            */
/* ⛳ AND ARGUMENT 5 IS FREE. A page id lived there while a slot had to say which page it was cut from;
 * a collection's slots are one run at a fixed stride, so the slot index is `(at - base) / bytes` and the
 * word that stored it has nothing left to store. ⛔ IT STAYS FREE — ▶ the type table's own note on the
 * page node's spare words, and for the same reason: the room is a dividend, not an opening. */

/* ⛔⛔ THE LINKS ARE `expert + 1`, AND 0 MEANS NONE. **AN EXPERT ID OF 0 IS PERFECTLY ORDINARY**, so 0
 * cannot mean "none" until the id is shifted by one — the same shift the free list makes, `slot + 1`,
 * for the same reason: slot 0 is ordinary too. */
#define NN__EXPERT__NONE        0ull

/* ⭐ WHERE AN EXPERT LIVES ON DISK — the backing an expert of a (layer, type) band is read from when it is
 * not in memory (⚖ *"the lru as a ram construct … load the misses from disk"*). An expert is `SLICES`
 * byte runs of one file: run `i` of expert `e` is `bytes[i]` at `from[i] + e · bytes[i]`, and it lands at
 * `into[i]` inside the expert's slot. `file` is a handle from the family's `file_open`; zero is no backing. */
#define NN__EXPERT__SLICES 4u
typedef struct nn__expert__backing {
    uint64_t file;
    uint64_t from[NN__EXPERT__SLICES];
    uint64_t bytes[NN__EXPERT__SLICES];
    uint64_t into[NN__EXPERT__SLICES];
} nn__expert__backing;

/* ⭐ AND THE SAME BACKING AS A PROGRAM HOLDS IT — a node array of integers, so a boot program can write one
 * and a verb can be handed one: the file, then each slice's start, one expert's length, and its place in the slot. */
#define NN__EXPERT__BACKING__FILE      0u
#define NN__EXPERT__BACKING__FROM(i)   (1u + 3u * (i))
#define NN__EXPERT__BACKING__BYTES(i)  (2u + 3u * (i))
#define NN__EXPERT__BACKING__INTO(i)   (3u + 3u * (i))
#define NN__EXPERT__BACKING__LENGTH    13u
#define NN__EXPERT__FAULT_BACKING      0x4E45424Bull   /* "NEBK" — a backing that is not 13 integers */

/* ⭐ THE LOADER — reads in flight at once, the threads that do them, and what it counts. ▶ `cpu/loader__impl.cuh`. */
#define NN__EXPERT__FLIGHT_MAX        64u   /* reads queued or in flight at once, over every layer */
#define NN__EXPERT__LOADERS_STANDARD   4u    /* threads, unless SILVANN_EXPERT_LOADERS says otherwise */
#define NN__EXPERT__LOADERS_MAX       32u
#define NN__EXPERT__COUNT_MISSES          0u   /* reads a pick had to ask for itself            */
#define NN__EXPERT__COUNT_PREDICTED       1u   /* reads a prediction queued                     */
#define NN__EXPERT__COUNT_PREDICTED_USED  2u   /* ...of which a pick then asked for the expert  */
#define NN__EXPERT__COUNT_WAIT_NS         3u   /* time a layer waited on its reads, in ns      */
#define NN__EXPERT__COUNT_WAITS           4u   /* ...and how many times it had to              */
#define NN__EXPERT__COUNTS                5u
/* What a request found: the expert in memory, its read on the way, or no way to have it. */
#define NN__EXPERT__RESIDENT   0
#define NN__EXPERT__ARRIVING   1
#define NN__EXPERT__REFUSED    2

/* A layer's LRU ends, one pair per matrix type. ⛳ THEY LIVE IN A HATCH ROW OF THEIR OWN —
 * `NN__HATCH__EXPERT_LRU`, a node_array on the layers whose entries are node_arrays on the types — AND NOT
 * IN THE INNER ARRAY'S HEAD NODE, though the head's `args[1..5]` really are free (`num_args` is the hold
 * count, `op_code` the allocation's reach, `args[0]` the lock). The reason: reaching into another
 * package's allocation head from `nn` would make this file depend on a layout `sys` is free to change. */
#define NN__EXPERT__LRU_RECENT  0u   /* the most recently touched expert of this layer, as a LINK */
#define NN__EXPERT__LRU_OLDEST  1u   /* the eviction candidate, as a LINK                         */
#define NN__EXPERT__LRU_COUNT   2u   /* how many are in the chain — this layer's residents         */

/* ⛔⛔ 509 EXPERTS PER LAYER IS A REAL BOUND AND IT IS NOT FAR AWAY. An allocation must fit the room left
 * in ONE heap chunk, and a `node_array` of E elements costs E+1 nodes against `SYS__HEAP__CHUNK_NODES`
 * (512) less `SYS__CHUNK__FIRST` (2). ⇒ 256 experts is comfortable, **512 IS NOT EXPRESSIBLE**, and the
 * next doubling of expert counts meets this.
 * ⛳ ITS RETIREMENT CONDITION: `SYS__HEAP__CHUNK_NODES` is a compile-time `#define`, so raising it is a
 * rebuild rather than a redesign — but it widens EVERY chunk in the pool, so it is a memory decision and
 * not a local one. The alternative is a third level, which costs an indirection on every lookup.
 * **Neither is owed until a model needs it; what is owed is that the bound is stated rather than met.** */
#define NN__EXPERT__MAX_PER_LAYER  (SYS__HEAP__CHUNK_NODES - SYS__CHUNK__FIRST - 1ull)

#define NN__EXPERT__FAULT_INDEX  0x4E455831ull   /* "NEX1" — the index could not be stood up */

#define NN__EXPERT__FAULT_ROOMLESS  0x4E45524Cull   /* "NERL" — no slot of that class anywhere        */
#define NN__EXPERT__FAULT_STRAY     0x4E455354ull   /* "NEST" — an address nobody reserved            */
#define NN__EXPERT__FAULT_ABSENT    0x4E454142ull   /* "NEAB" — that expert is not resident           */
#define NN__EXPERT__FAULT_SPAN      0x4E455350ull   /* "NESP" — the span asked for leaves the slot    */
/* ⛔ AND THREE MORE, BECAUSE THE LOAD PATH HAS THREE MORE WAYS TO FAIL AND THEY ARE DIFFERENT MISTAKES.
 * ⇒ ★ A FAULT CODE IS THE ONLY THING A CALLER GETS, so one that names the wrong cause sends the reader
 * to the wrong half of the machine — which is why `stand_arena` keeps its own three apart, in this file,
 * for this reason. ⛳ THE DISCRIMINATOR IS WHAT THE CALLER WOULD DO NEXT: no room at all is a card that
 * is full; a ledger failure is a card with room and no way to record it; a heap failure is neither and
 * belongs to the allocator. */
#define NN__EXPERT__FAULT_LEDGER    0x4E454C44ull   /* "NELD" — a slot was found, the handed row was not */
#define NN__EXPERT__FAULT_HEAP      0x4E454850ull   /* "NEHP" — the heap would not carve the object      */
#define NN__EXPERT__FAULT_REFUSED   0x4E455246ull   /* "NERF" — the index would not take this admission  */
/* ⛳ AND `FAULT_INDEX` KEEPS ITS OWN MEANING — *"the index could not be stood up"* — rather than being
 * stretched to cover "that expert is not IN the index", which was a fourth borrowing. */

/* ══ ⭐⭐ `nn__weights` — A NAME FOR BYTES A PAGE OWNS ════════════════════════════════════════════════
 *
 * ⚖ RULED: *"a second kind, minted from the index"* — and ⚖ S99 named the shape long before
 * there was a tenant for it: *"experts will not be in the sys heap but only their references will be
 * there as a special heap node."*
 *
 * ⛔⛔ WHAT IT FIXES, AND IT WAS BLOCKING A FORWARD PASS. The load path puts weights in PAGES; every
 * arithmetic verb takes an `nn__buffer`, whose bytes come from the POOL. So until this kind existed a
 * real weight had to be loaded TWICE to be used ONCE — once into the page it lives in and once into a
 * buffer something could compute from. ⇒ ★ THE ORACLE THAT PROVED BOTH HALVES AND NOT THE JOIN WAS
 * MEASURING EXACTLY THIS GAP.
 *
 * ⭐ IT IS THE SAME OBJECT `nn__kv_ref` IS, POINTED AT A DIFFERENT OWNER. Both name memory the heap did
 * not lend; a kv reference's bytes belong to the cartridge and these belong to a page, and in both cases
 * the owner outlives every name for it, so release has nothing to give back.
 * ⛳ AND IT IS SIMPLER THAN `nn__kv_ref` BY EXACTLY THE FIELD THAT MADE THAT ONE HARD: a KV reference
 * carries a FORMAT because a subsection may cross a tier. A page slot is one span of bytes in one
 * format, decided when the expert was packed — so there is nothing here but where and how much. */
#define NN__WEIGHTS__AT     0u   /* the address, inside some page's slot                              */
#define NN__WEIGHTS__BYTES  1u   /* how many, and a reader needs no second lookup to know             */
/* ⭐⭐⭐ WHAT THIS VIEW HOLDS, OR ZERO — ⚖ RULED. ARCHITECT: *"i think the view does a hold
 * once on the held object, and then other references to the view object are held by the view object
 * itself … the base method should hold and it is the developer responsibility to do a nohold only if it
 * is sensible."*
 * ⛳ SO THE KIND SERVES TWO OWNERS AND SAYS WHICH IN ONE WORD. A PAGE view (`nn__expert__plane`) stores
 * **0**: a page outlives every name for it and there is nothing to give back. A BUFFER view
 * (`nn__vector__range`) stores the buffer's handle and RETAINED it, so the pool cannot re-hand those
 * bytes while a view of them is alive.
 * ⛔ A NESTED VIEW HOLDS THE VIEW IT CAME FROM, not the buffer underneath — and the release loop drains
 * the chain, so the buffer's hold goes when the last link does. That is the architect's own description
 * and it needed no special case.
 * ⇒ ★★ THE SAFE THING IS THE DEFAULT AND THE CHEAP THING IS THE EXCEPTION. A borrowed span is a real
 * optimisation and an unsafe default: it produces a correct-LOOKING answer out of somebody else's
 * bytes, which is the failure this tier has the least defence against. */
#define NN__WEIGHTS__HELD   2u   /* the object this view holds, or 0 for one that owns nothing        */
#define NN__WEIGHTS__HOST_AT 3u  /* the program's address for the same bytes, or 0 — ▶ NN__BUFFER__HOST_AT */
#define NN__WEIGHTS__STAMP  4u   /* the stamp a value written here must carry, or 0 — ▶ `result.cuh`  */
#define NN__WEIGHTS__OWNER  5u   /* the owning worker plus one, or 0 — ▶ NN__BUFFER__OWNER            */
#define NN__WEIGHTS__NODES  2u   /* the head, and one node holding the words above                    */

/* ⛳ A SHAPE THE OPERANDS CANNOT HOLD, kept apart from `FAULT_ROOM` (the arena) and `FAULT_BOUNDS`-style
 * codes elsewhere: the caller gave `rows`/`cols` that do not fit the buffers it also gave, which is a
 * mistake in the CALL and not a shortage on the card. ⇒ ★ the discriminator is again what the caller
 * would do next — shrink the ask, or fix the arithmetic that produced it. */
#define NN__EXPERT__FAULT_SHAPE     0x4E455348ull   /* "NESH" — rows/cols do not fit the operands given  */

/* The key the host sized the span's third term by, spelled once. ⛳ ITS SUFFIX IS WHAT PICKS THE READER —
 * `_mb` is a unit, so this is a `sys__settings__size` and `sys__settings__count` would refuse it. */
#define NN__EXPERT__KEY_ARENA  "nn__expert_cache_l1__size_mb"

/* ⭐ THE TYPE TABLE'S KEYS — three to a collection, plus the count. ⛳ THE SUFFIXES PICK THE DOORS exactly
 * as the pair above does: `_bytes` is a unit of shift 0, so the two halves go through `sys__settings__size`
 * and cannot be read by the count door, while `__types` and `__slots` name no unit and cannot be read by
 * the size one. ⛔ `_bytes` AND NOT `_kb`: a half's size is `rows x row_bytes + 2 B a row`, and nothing
 * says that lands on a kibibyte for a model nobody has packed yet. */
#define NN__EXPERT__KEY_TYPES  "nn__expert__types"

/* ══ THE HANDED LIST — the slots RESERVED and not yet published ══════════════════════════════════════
 * ⚖ *"mark that handed memory address into a handed list, so when the abi does the assignment you can
 * revoke it from there and you'll have the list of hanging references."*
 * ⇒ ★★ THE LEAK BECOMES AN OBSERVABLE SET RATHER THAN A HAZARD: reserve appends, assign removes, and
 * what remains IS the outstanding set — so "reserved but never published" is not a state anyone has to
 * DETECT, it is a list anyone can READ.
 * ⛳ AN ENTRY IS AN ADDRESS AND A TYPE. The SLOT is not stored: it is `(at - base) / bytes`, the same
 * division `evict` does, so storing it would be a second copy of one number. */
#define NN__EXPERT__HANDED_AT    0u
#define NN__EXPERT__HANDED_TYPE  1u

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_EXPERT_CUH */
