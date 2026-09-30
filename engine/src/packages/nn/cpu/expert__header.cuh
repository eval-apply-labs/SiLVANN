#ifndef SILVANN__PACKAGES_NN_CPU_EXPERT__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_EXPERT__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/package__header.cuh"   /* the hatch this package's own rows are declared into */
#include "../../sys/cpu/heap__header.cuh"      /* the chunk geometry the expert bound is derived from */
#include "../contracts/objects/expert.cuh" /* its constants and fault words */
#include "../../sys/cpu/silicon/silicon__header.cuh"   /* the family a fetch reads through */
/* ══ nn — THE EXPERT L1 ARENA, AND WHERE IT BEGINS ═══════════════════════════════════════════════════
 *
 * ⚖⚖ RULED (`expert_lru_node_array.md` §⑤ⓐ): *"3 SPANS — the arena is a REGION of nn's span, not its
 * own allocation."* So this takes no memory. It NAMES memory the package already holds, and the whole of
 * ① is that the naming needs no arithmetic that did not already exist.
 *
 * ── ⭐⭐ THE BASE IS WHERE THE BUFFER CARVE STOPPED, AND NOTHING COMPUTES IT TWICE ────────────────────
 * The carve walks the class list laying runs down from the span's start, keeping a running total. When it
 * finishes, that total IS the first byte after the last buffer — which is the first byte of the arena.
 * ⇒ ★ THE FIGURE WAS ALREADY IN A LOCAL VARIABLE; ① IS THE ACT OF NOT THROWING IT AWAY. A second walk
 * summing the same five products would be a second copy of one number, which is the thing this package
 * has now spent three mechanisms avoiding — the span's total, the buffer runs, and the expert count.
 * ⛔ SO THE ORDER IS LOAD-BEARING: the arena cannot be stood up before the carve, because the carve is
 * what says where it goes. That is a real dependency and not a convention, and the init reflects it.
 *
 * ── THE SIZE IS THE ONE KEY THIS CLOSES ──────────────────────────────────────────────────────────────
 * `nn__expert_cache_l1__size_mb`, which the host already reads to SIZE the span and which nothing on the
 * card has ever read to CARVE it. ⛳ It is the third tenant of `nn_span_layout.md` §⓪ and the first of the
 * three keys that had no reader at all.
 * ⛔ AND IT IS READ THROUGH `sys__settings__size`, WHICH IS NEWER THAN THIS FILE'S SUBJECT. Until the card
 * had a unit reader it would have answered this key in MEBIBYTES to a caller expecting BYTES — a factor of
 * 1,048,576, with the span's total still right and every address still resolving. ▶ `settings.cuh`.
 *
 * ── ⛔ WHAT MUST FIT, AND WHY IT IS CHECKED HERE RATHER THAN TRUSTED ─────────────────────────────────
 * The host summed buffers + arena + cartridge and asked for that; so the arena fitting is implied by the
 * host having read the same keys. It is checked anyway, for the reason the carve checks its own runs: the
 * two tiers are two readers of one text, and the whole point of the arrangement is that a disagreement
 * between them surfaces as a REFUSAL rather than as a region that hands out addresses belonging to
 * nobody. ⇒ ★ AN IMPLICATION IS NOT A CHECK — what makes it true is the host, and this is on the card.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* Name the arena: the bytes from `from` onward that the config asked for. Answers false when the key is
 * unreadable or the region does not fit the span it was handed.
 * ⛳ `from` IS THE CARVE'S ANSWER and not a figure this recomputes — ▶ the block above. */
static __device__ inline bool nn__expert__zzpackage_stand_arena(uint64_t from,
                                                                uint64_t span_at,
                                                                uint64_t span_bytes);

/* Where the arena starts, and how many bytes of it there are — 0 when no config named one.
 * ⛳ A PACKAGE WITH NO SPAN HAS NO ARENA AND THAT IS NOT A FAULT, which is the `e.boot()` case the whole
 * device suite runs in. What refuses is a page asked for out of an arena that is not there. */
static __device__ inline uint64_t nn__expert__arena_at(void);
static __device__ inline uint64_t nn__expert__arena_bytes(void);

/* Read the type table out of the config and record it. Answers false on a config that cannot be one:
 * a count past the cap, a live type missing any of its three keys, a zero, arithmetic that would wrap, or
 * runs that do not fit the arena. ⛳ NAMING NO TYPES AT ALL IS NOT A REFUSAL — that is the no-config boot,
 * and a card with no collections simply holds no experts. */
static __device__ inline bool nn__expert__zzpackage_stand_types(void);

/* How many collections stand — 0 when no config named any. */
static __device__ inline uint64_t nn__expert__types(void);

/* One collection's four figures. Each answers 0 for a type that does not stand, which is the same answer
 * they give before any config is read — a type nobody named and a table nobody stood are the same nothing. */
static __device__ inline uint64_t nn__expert__type_at(uint64_t type);
static __device__ inline uint64_t nn__expert__type_slots(uint64_t type);
static __device__ inline uint64_t nn__expert__type_bytes(uint64_t type);
static __device__ inline uint64_t nn__expert__type_offset(uint64_t type);

/* TRUE when this collection's slots may never be taken back. ⛳ ANSWERS TRUE FOR A TYPE THAT DOES NOT
 * STAND, and for a table that could not be read at all — ▶ the polarity note in the block above, which is
 * where that falls out of the representation rather than out of a check. */
static __device__ inline bool nn__expert__type_anchored(uint64_t type);

/* The collection's free head, stored as `slot + 1` so that `NONE` and slot 0 are different things.
 * ⛳ IT STANDS AT SLOT 0 AND THE LIST BEGINS AT ITS FULL LENGTH — every slot is free on a fresh card. */
static __device__ inline uint64_t nn__expert__type_free(uint64_t type);

/* ⭐⭐⭐ TAKE AND GIVE — THE WHOLE OF PLACEMENT, AND THE LIST COSTS NO MEMORY.
 * ⚖ *"the pages go in freemem"*, the same word and the same mechanism the buffer pool already has. The
 * list is threaded through the FREE SLOTS THEMSELVES: a free slot's bytes are dead — `plane` bounds every
 * read to an expert's own slot, so a free slot belongs to no reader — and its first word holds the next
 * free slot. Only the head is stored, in the row above.
 * ⛳ `take` IS A POP WITH NO SECOND PATH: one free list is the only source of slots, so there is no bump
 * pointer to branch against. A collection with none left answers false, which is an ORDINARY answer —
 * ⚖ `place` refuses on a full card by design and no eviction policy stands behind it yet.
 * ⛔⛔ `give` TRUSTS THAT THE SLOT IS NOT ALREADY FREE. Giving one twice links it to itself and then two
 * experts are handed one slot, silently. What makes that unreachable is UPSTREAM — `evict` refuses an
 * expert that is not resident — so ⇒ ★ THE TEST THAT HOLDS THIS INVARIANT IS THE ONE ABOUT A SECOND
 * EVICT, and it lives with `evict` rather than here. A scan would catch it and is refused: the list
 * starts at every slot, so scanning is O(slots) on a path that runs per eviction. */
static __device__ inline bool nn__expert__type_take(uint64_t type, uint64_t* slot, uint64_t* at);
static __device__ inline bool nn__expert__type_give(uint64_t type, uint64_t slot);

/* ⭐⭐ AND THE TWO THAT REPLACE THE PAGE NODE. `slot_at` is the address arithmetic the collapse buys;
 * `slot_of` is it run backwards, which is what `slot_in_page` did from a stored page id and now does from
 * nothing at all. Both answer false outside the run rather than a plausible address.
 * ⛔ `slot_of` REFUSES A REMAINDER: an address inside the run but not ON a slot boundary is not a slot,
 * and answering the slot it lands in would turn a caller's arithmetic mistake into a silent aliasing. */
static __device__ inline bool nn__expert__type_slot_at(uint64_t type, uint64_t slot, uint64_t* at);
static __device__ inline bool nn__expert__type_slot_of(uint64_t type, uint64_t at, uint64_t* slot);
#define NN__EXPERT__LINK(e)     ((e) + 1ull)
#define NN__EXPERT__UNLINK(l)   ((l) - 1ull)

/* ── WHAT THE INDEX PUBLISHES ────────────────────────────────────────────────────────────────────────
 * ⛳ Every one answers zero or false when no index was stood up, which is the `e.boot()` case. */
static __device__ inline bool     nn__expert__zzpackage_stand_index(uint64_t layers);
static __device__ inline uint64_t nn__expert__index_layers(void);

/* One (layer, type) band's expert array, made when the loader states the layer's width through the
 * `layer` verb. ⛔ REFUSES PAST THE BOUND ABOVE rather than allocating something that cannot fit a
 * chunk, and refuses a type the table does not have. */
static __device__ inline bool     nn__expert__layer_stand(uint64_t layer, uint64_t type, uint64_t experts);
static __device__ inline uint64_t nn__expert__layer_experts(uint64_t layer, uint64_t type);

/* One slot, read and written. ⛳ BORROWED, like the tier and layer rows, and for the same reason. */
static __device__ inline bool nn__expert__slot(uint64_t layer, uint64_t type, uint64_t expert,
                                               sys__heap_node* answer);

/* ⭐ ADMIT · TOUCH · EVICT — the three things that happen to an expert, and none of them allocates.
 * ⛔ `type` IS WRITTEN AT ADMIT AND SURVIVES EVICTION (`expert_lru_node_array.md` §⓪): it says which
 * collection this expert belongs to, and eviction has to know that in order to give the slot back to the
 * right free list at the moment the slot stops being the expert's. */
static __device__ inline bool nn__expert__admit(uint64_t layer, uint64_t expert, uint64_t at,
                                                uint64_t type);
static __device__ inline bool nn__expert__touch(uint64_t layer, uint64_t type, uint64_t expert);
static __device__ inline bool nn__expert__evict(uint64_t layer, uint64_t type, uint64_t expert);

/* What an `nn__weights` hands over when its last hold goes — ▶ `NN__WEIGHTS__HELD`. A page view holds
 * nothing and returns at once; a buffer view moves the object it retained onto the chain the release
 * loop drains. ⛳ IT LETS GO OF NOTHING ITSELF, which is what keeps a chain of nested views from
 * recursing on a device with no stack to spare. */
static __device__ __noinline__ void nn__weights__zzpackage_release_internal(sys__heap_node* head,
                                                                           uint64_t* releaser_stack);

/* The LRU's two ends of one (layer, type) band, as EXPERT IDS. False when that band holds nothing
 * resident. */
static __device__ inline bool nn__expert__recent(uint64_t layer, uint64_t type, uint64_t* expert);
static __device__ inline bool nn__expert__oldest(uint64_t layer, uint64_t type, uint64_t* expert);
static __device__ inline uint64_t nn__expert__layer_resident(uint64_t layer, uint64_t type);

/* ══ ⭐⭐ THE ADMITTER — AND IT IS NOW A POP ═════════════════════════════════════════════════════════
 *
 * ⚖ *"the host asks for space to host an expert of TYPE T, the device finds the appropriate slot and
 * hands him a pointer"* — and the type is back, ⚖ ruled: *"reserve should take the type."*
 * ⛳ THE RULING SAID THIS ALL ALONG AND THE SIZE CLASSES TALKED THE DESIGN OUT OF IT. While a page held
 * experts of several sizes a class WAS a consequence of an expert's bytes, and asking the host for one
 * was asking for an answer the device could derive — that argument was right and its premise is gone.
 * Every expert of a collection is one size now, so bytes determine nothing and a caller that does not
 * name a type has not said what it wants.
 *
 * ⭐⭐ THERE IS ONE PLACE TO LOOK AND NO ORDER TO IT: the collection's free list. The three-way search —
 * a class's partial chain, then an empty page re-typed, then the arena minted — existed to find a page
 * with ROOM FOR THIS SIZE IN IT, and a slot either is free or is not.
 *
 * ⛔ IT DOES NOT EVICT. A collection with no free slot answers false, and WHAT TO THROW AWAY is a policy
 * that has not been ruled — so the refusal is the honest answer rather than a placeholder eviction that
 * would quietly become the policy by being there first. */
static __device__ inline bool nn__expert__place(uint64_t type, uint64_t* address, uint64_t* slot);

/* ══ ⭐⭐ AN EXPERT IN MEMORY, FROM DISK IF IT IS NOT — THE LOADER ═════════════════════════════════════════
 * ⚖ *"the lru as a ram construct to have some experts in ram while we load the misses from disk"* · ⚖ *"go
 * ahead with the loader pool and prefetch"*. ▶ `loader__impl.cuh`.
 *   request   a resident expert is TOUCHED (it becomes its band's most recent) and its address answered —
 *             RESIDENT. One whose read is already on the way answers ARRIVING. One that is neither gets a
 *             slot — a free one of its collection or the least recently used expert's, never one of
 *             `pinned` — and its read is queued for the loader threads: ARRIVING. `predicted` marks a read a
 *             prediction asked for rather than a pick, for the counts.
 *   settle    waits for every read of a (layer, type) band and admits each: after it, every expert that was
 *             ARRIVING is resident, or its slot went back.
 *   fetch     the two in one: the expert, in memory, when it returns — or false.
 *   count     reads asked for by picks, reads a prediction queued, and how many of those a pick then wanted.
 * ⛳ THE LRU IS PER LAYER — the index keeps one chain per (layer, type) band — so a victim is the oldest of the
 *   band that needs the room, else of the first other band that has one. A global order is a list the index
 *   does not keep yet. */
static __device__ inline int nn__expert__request(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t expert,
                                                 const nn__expert__backing* backing, bool predicted,
                                                 const uint64_t* pinned, unsigned npinned, uint64_t* at);
static __device__ inline bool nn__expert__settle(uint64_t layer, uint64_t type);
/* `settle` for one expert: waits for its read alone, if one is on the way, and admits it — so a verb can compute an
 * expert that has arrived while the reads it asked for after it are still arriving. */
static __device__ inline bool nn__expert__settle_one(uint64_t layer, uint64_t type, uint64_t expert);
static __device__ inline bool nn__expert__fetch(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t expert,
                                                const nn__expert__backing* backing, uint64_t* at, bool* missed);
static __device__ inline uint64_t nn__expert__count(unsigned which);
static __device__ inline uint64_t nn__expert__zzpackage_owner(void);
/* ⭐ WHERE THE SLOTS ARE A MAPPED FILE, "RESIDENT" SAYS ONLY THAT THE EXPERT HAS AN ADDRESS: its bytes are in memory
 * when the system has kept the file's pages. `in_memory` answers whether every byte of the slot at `at` is there now —
 * and when one is not, starts its reading without waiting, so a verb computes the experts that are here while the rest
 * arrive. Always true where the slots are memory of their own. */
static __device__ inline bool nn__expert__in_memory(sys__silicon_family__id family, uint64_t type, uint64_t at);
/* A backing node array, read into the struct the loader takes. False unless it is 13 integers. */
static __device__ inline bool nn__expert__backing_of(uint64_t array, nn__expert__backing* backing);
/* The loader threads joined, at the package's teardown. */
static void nn__loader__zzpackage_stop(void);

/* ══ ⭐⭐ THE LOAD PROTOCOL — THREE WORDS, AND THE HOST DRIVES IT ═════════════════════════════════════
 *
 * ⚖ *"the host asks for space to host an expert of type t, the device finds the appropriate slot and
 * hands him a pointer and adds that pointer to the pending list, the host writes with DMA and then calls
 * the completion of expert e at position p, then moves to the next one."*
 * ```
 *   (nn__expert__reserve type)                  -> an address, and a row in the handed list
 *   nn_abi_write(address, bytes, ...)              the host's DMA — already shipping
 *   (nn__expert__assign layer type expert at)   -> residency recorded, the row removed
 *   (nn__expert__revoke address)                -> the slot given back, the row removed
 * ```
 * ⭐⭐ `assign` TAKES NO CLASS AND NO PAGE, WHICH IS THE ONE PLACE THIS DIFFERS FROM THE SPEC AS SPOKEN.
 * Both are in the handed row already, put there by the device that chose them. ⇒ ★ ASKING THE HOST TO
 * REPEAT THEM WOULD BE ASKING FOR AN ANSWER IT COULD GET WRONG — the same argument that makes `reserve`
 * take BYTES rather than a quant name, applied to the other end of the protocol.
 * ⛔ AND IT IS WHY AN ADDRESS THAT WAS NEVER RESERVED IS REFUSED RATHER THAN ADMITTED: `assign` has
 * nowhere to learn the page from, so a stray address is not a thing it can half-do.
 *
 * ⛳ THEY ARE VERBS AND NOT RAW ABI DOORS, AND THE CHOICE WAS MADE FOR US: the page index lives in the
 * device heap, so anything that walks it runs on the card. ⛳ COST, `REASONED`: ~10k experts on the 35B
 * at ~15 µs a settling launch is ~150 ms of one-time load, which is not a hot path. */
static __device__ __noinline__ void nn__expert__zzpackage_apply_reserve(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__expert__zzpackage_apply_assign(sys__heap_node* base, uint64_t form);
static __device__ __noinline__ void nn__expert__zzpackage_apply_revoke(sys__heap_node* base, uint64_t form);

/* ⛳ AND ONE READER, because the handed list is only worth having if somebody can ask it. */
static __device__ __noinline__ void nn__expert__zzpackage_apply_handed(sys__heap_node* base, uint64_t form);

/* ⭐⭐ AND THE WORD THAT MAKES A LAYER EXIST AT ALL, WHICH THE PROTOCOL ABOVE NEEDED AND DID NOT HAVE.
 * ⚖ *"lets keep the indexes for everyone but they'd point at a null subarray"* — the index is sparse, so
 * the outer array is stood up by the boot from the MODEL's layer count and the inner arrays are made on
 * demand. ⛔ NOTHING MADE THEM. `assign` refused every expert on a freshly booted card, because a layer
 * that holds nothing has no slot to write and that is the correct behaviour of a structure nobody has
 * told how wide a layer is. ⇒ ★ A SPARSE INDEX NEEDS A WORD THAT SAYS "THIS LAYER IS THIS WIDE", and the
 * loader is the only thing that knows — the count is in the model's file, not on the card.
 * ⛳ It refuses a second stand rather than silently re-making the array: that would strand every expert
 * already recorded in it, with the layer still reading the right width. */
static __device__ __noinline__ void nn__expert__zzpackage_apply_layer(sys__heap_node* base, uint64_t form);

/* ══ ⭐⭐⭐ THE INDEX IS THREE DEEP — `[layer][type][expert]` ══════════════════════════════════════════
 *
 * ⚖⚖ RULED: *"my instinct would have been [layer][type][expert] … i would maintain the
 * layer addressing by index, keeping some null array values is a worthy trade for direct addressing."*
 * ```
 *   index[layer]          the MODEL's layers, stood at boot
 *     -> band[type]       MAX_TYPES entries, stood at boot — mostly null, and that is the trade
 *          -> [expert]    stood on demand by `nn__expert__layer`, because only a loader knows how wide
 * ```
 * ⭐⭐ THIS IS WHAT ENDS `NN-5`. An expert of layer L is `[L][t][e]`; a dense tensor of layer L is
 * `[L][t'][0]`. They cannot collide because the TYPE separates them before the expert index is read,
 * and a dense collection's inner array is simply one wide.
 * ⛳ SO A DENSE TENSOR IS `[layer][type][0]` — the `[0]` the architect named as his fallback, arrived at
 * from the other side: with the type in the path there is nothing left for the expert index to say.
 *
 * ⛳ THE MIDDLE ARRAYS ARE EAGER AND THE INNER ONES ARE NOT. A band is `MAX_TYPES` nodes, so every
 * `[layer][type]` is addressable with no "does it exist" question on the read path; an inner array is
 * as wide as a layer's expert count and only a loader knows it. ⚖ *"keeping some null array values is
 * a worthy trade for direct addressing."*
 *
 * ⭐⭐ AND THE LRU MOVED DOWN WITH THE INDEX — ONE CHAIN PER (LAYER, TYPE). That is a CORRECTNESS
 * argument and not symmetry: the chain exists to choose a victim whose SLOT is then reused, and a slot
 * only ever serves its own collection. A chain mixing two collections would offer a victim whose slot
 * is the wrong size for the expert that wanted room. ⇒ ★ AN EVICTION POLICY IS ONLY MEANINGFUL OVER A
 * SET OF INTERCHANGEABLE THINGS. ⛳ It also keeps a link a bare expert index, where one chain a layer
 * would have had to carry `(type, expert)` in a single word.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⭐⭐ THE ONLY WAY TO MAKE ONE, AND IT GOES THROUGH THE INDEX. `(nn__expert__plane layer type expert offset
 * bytes)` — the expert is looked UP, and the span is checked against that expert's OWN slot width.
 * ⇒ ★ A PROGRAM CAN NEVER NAME MEMORY IT WAS NOT GIVEN, because the bound is applied by the thing that
 * knows the slot, not by the caller that wants to read it. The alternative — a door taking a raw device
 * address — is the line `abi.cuh` declined to cross (*"a different and much larger promise"*), and a
 * door inside the language would be that same promise made to every program rather than to one host.
 * ⛳ THE `offset` IS WHAT MAKES ONE SLOT HOLD SEVERAL PLANES, which the real bundle requires: an expert
 * is `gate_up` + `down` in one slot, and each is codes then scales at its own offset. ▶ `weight_load.md` §②. */
static __device__ __noinline__ void nn__expert__zzpackage_apply_plane(sys__heap_node* base, uint64_t form);

/* ══ ⭐⭐⭐ `nn__expert__multiply_fp16` — THE MATVEC AT FULL PRECISION ════════════════════════════════
 *
 * ⚖ NAMED BY THE ARCHITECT: *"nn__expert__multiply_fp16 is the verb you are looking for."*
 *
 * `(nn__expert__multiply_fp16 weights x out rows cols)` — `out[r] = Σ_c W[r][c] · x[c]`, every operand
 * fp16, the accumulator fp32. It answers the OUT operand so a program can chain it.
 *
 * ⛳ WHY IT IS ITS OWN VERB RATHER THAN A WIDTH OF THE CODEC. `nn__turboquant__gemv` at `d == 16` computes the
 * identical product — `MEASURED` on a real Qwen2.5-7B `q_proj`, max |abs err| 5.5e-05 against numpy
 * fp64, below the fp16 output resolution — so this is not a capability that was missing. What it is, is
 * the SHAPE the dispatch needs: ⚖ *"nn__expert__multiply will be a lisp defun … for every expert type
 * there is the related function"*, so each type's function is a separate named thing the table can
 * point at, and the fp16 one carries no `lut` operand and no width argument because it has neither.
 * ⇒ ★ A VERB THAT TAKES ARGUMENTS IT IGNORES IS A WORSE TABLE ENTRY THAN ONE THAT DOES NOT TAKE THEM:
 * the codec's arm needs a scale plane and a `d`, and handing it two dead cells at every call site is a
 * cost paid by the caller to save a row here.
 *
 * ⛔⛔ AND IT IS WHAT A FORWARD PASS NEEDS FIRST, WHICH IS THE ORDERING ARGUMENT. ⚖ *"the goal is to
 * prove the system here and then replicate it on the nn_tq."* (⚖ *"keep together"*: the codec
 * stays in `nn`, and "there" is the codec's own path.) A pass at fp16 needs no rotation, no
 * codebook and no scale — so the one thing that could make a correct-looking pass compute a WRONG
 * product (an activation not rotated to match the weights) cannot arise. ⇒ ⛳ THE PRECISION IS NOT THE
 * POINT OF STARTING HERE; THE ABSENCE OF THE ROTATION IS.
 *
 * ⭐ THE ACCUMULATOR IS fp32 WHILE BOTH OPERANDS AND THE ANSWER ARE HALVES — ⚖ *"fp16 storage, fp32
 * accumulators"*. `cols` is thousands; an fp16 accumulator would lose more to its own rounding than q8
 * loses to quantisation. ⛔ AND THE ORDER OF THE REDUCTION IS NOT THIS SOURCE'S TO DECIDE — the same
 * `MEASURED` finding as the codec's gemv, where a fixture built to tell showed the device summing in an
 * order other than the written one, with no fast-math in the build. Repeatability is untouched;
 * agreement with an external reference that assumes the written order is not.
 *
 * ⛳ IT TAKES ITS OPERANDS THROUGH `nn__primitives__room`, so a PAGE (`nn__weights`) and a POOL buffer
 * are both accepted in every slot, and a real resident weight needs no second copy. That is the join
 * `nn__weights` exists for, and this verb inherits it by asking the same question every other one does. */

#endif /* SILVANN__PACKAGES_NN_CPU_EXPERT__HEADER_CUH */
