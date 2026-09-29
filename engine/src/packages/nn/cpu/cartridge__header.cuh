#ifndef SILVANN__PACKAGES_NN_CPU_CARTRIDGE__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_CARTRIDGE__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/package__header.cuh"   /* the hatch this package's own rows are declared into */
#include "../contracts/objects/cartridge.cuh" /* its constants and fault words */
/* ══ nn — THE CONVERSATION CARTRIDGE: WHERE A TOKEN'S KV BYTES ARE, AND WHAT FORMAT THEY ARE IN ══════
 *
 * ⚖⚖ RULED: *"it only contains the 2MB text and the kv cache that should be indexed as an
 * array like cache[layer][token] and shielded by an opcode so the deltanet can be masked"*, and
 * decisively: *"i plan to access the kv cache via an opcode that returns me a node pointer, not to stash
 * the whole kv cache as a node."*
 *
 * ⭐⭐ SO THE INDEX IS ARITHMETIC AND NOT A STRUCTURE, AND THAT IS WHAT MAKES TOKEN GRANULARITY
 * AFFORDABLE. `cache[layer][token]` is the INTERFACE, spelled where a program reads it; nothing anywhere
 * stores a slot per token. ⛔ THE ALTERNATIVE WAS PRICED: a node per token per layer is 3,072,000 slots at
 * the device fixture's 256k × 12 — 6,036 `node_array`s of 32 KiB = **193 MiB of index, IN THE HEAP**, for
 * a cartridge whose data is in the SPAN, against a 128 MiB heap. On the 36-layer model, ~580 MiB.
 * ⇒ ★★ A STRUCTURE WOULD HAVE COST MORE THAN THE THING IT INDEXES.
 *
 * ── ⚖⚖ THE TIER IS A TOKEN'S AGE, NOT ITS POSITION IN A FIXED PLAN — RULED ────────────────
 * ⚖ *"the compression to turboquant should be a lisp function so we can migrate at the end of a trigger
 * or something, like the end of a sentence. for now it trips at the filling of the cache tier and
 * triggers a cache tier rotation (move stuff to tq4, then fp32 to tq6) the three tiers are separate, you
 * have the start and end token num so you can find the base you were looking for."*
 *
 * ⛳ A TOKEN IS BORN IN THE HOT TIER AND SINKS. When a tier fills, its oldest run is handed down — so each
 * tier holds a CONTIGUOUS RUN of token numbers, and the runs are in age order. That is what keeps the
 * resolver arithmetic: finding a token's tier is four comparisons and no lookup.
 *
 * ── ⭐ AND THE TIER IS A RING, WHICH COSTS THREE INSTRUCTIONS AND NO MEMORY MOVE ──────────────────────
 * ⚖ *"q-a portion"* — a rotation moves PART of a tier, so the tier has a hole at the front and writes
 * continue at the back. Its bytes never move; only the cursors do.
 *     phys = phys_first + (token − first);   if (phys >= capacity) phys -= capacity;
 * ⭐ THE CONDITIONAL SUBTRACT IS EXACT AND NEEDS NO MODULO AND NO POWER-OF-TWO CAPACITY: both terms are
 * below `capacity` by their own invariants, so their sum is below twice it and one subtract lands it.
 * ⛳ WHICH MATTERS BECAUSE THE POWER-OF-TWO ROUTE IS NOT FREE HERE — rounding the shipping 192,000-token
 * tq4 tier up to 262,144 is +405 MiB for an instruction this arithmetic does not need.
 *
 * ── ⚖ THE SINK TIER, AND IT IS WHY THIS LIST HAS FOUR ROWS AND NOT THREE ─────────────────────────────
 * ⚖ *"make the system prompt 4th tier as fixed for 50 tokens so the starting pulse does not degrade"*,
 * and *"make it sized like the others"*.
 * ⛔⛔ PURE AGE WOULD CRUSH THE OPENING TOKENS FIRST, WHICH IS THE ONE PLACE IT IS WORST. They are the
 * oldest thing in the cache, so they are the first thing rotated to tq4 — and the old tree knew this: it
 * carried `sys_filled_tokens` as its own region and made fp32 mean *"HOT + SYSTEM"* precisely so the
 * opening never degraded (`MODEL_CMD_GATHER_ATTENTION.cuh:55,105`). This row is that region, re-derived.
 * ⇒ ★ IT IS THE CHEAPEST ROW IN THE FILE AND THE ONE WITH THE MOST LEVERAGE: 50 × 49,152 B = 2.34 MiB
 * against a 1.80 GiB cartridge — 0.13% — to stop the attention sink being quantised away.
 * ⛳ AND IT IS A CONFIG PAIR LIKE EVERY OTHER TIER, not a constant. ⚖ *"it is a parameter after all."*
 * ⛔ THE NAME IS `sink` AND NOT `system`, DELIBERATELY: 50 tokens is an attention sink and not a system
 * prompt, which runs to hundreds. A key spelled `system_prompt` invites a later reader to "fix" it to
 * hold the real prompt — which would either blow the budget or quietly stop it being pinned, and neither
 * announces itself. ⇒ ★ NAME A THING FOR WHAT IT DOES WHEN THE OBVIOUS NAME WOULD INVITE A WRONG EDIT.
 *
 * ── ⛔ ROTATION IS LOSSY, SO IT IS A COMMIT POINT ────────────────────────────────────────────────────
 * ⚖ Q4 ruled the cache is never deallocated — *"at most there will be an opcode that does a rollback and
 * resets the cursors"*, which is what makes a divergent path affordable: copy the divergence, restore the
 * cursors, never clone the base. ⭐ THE CURSORS ARE SIX WORDS A ROW, so a rollback is a WRITE and not a
 * data operation — that is the property to keep whole.
 * ⛔ BUT fp32 → tq6 DESTROYS BITS, so a rollback CANNOT cross a rotation. The bound is real and belongs
 * beside the mechanism rather than in a design note: **you can unwind within the hot tier, and not past
 * the last rotation.** ⛳ The cheap lever if that ever bites is a minimum age on rotation, so the youngest
 * N tokens are never handed down; it is a comparison in the trigger and nothing here changes.
 *
 * ── ASSUMED: K AND V SPLIT THE STRIDE IN HALF ────────────────────────────────────────────────────────
 * `bytes_per_token_layer` covers K and V together — 4,096 B of fp32 is 2 (K+V) × 2 kv-heads × 256 dims ×
 * 4 B — and this file lays each layer's run down as a K plane then a V plane of equal length.
 * ⛔ WHAT WOULD FALSIFY IT: an attention with asymmetric K and V. MLA's latent KV is exactly that, and it
 * is on the far roadmap (`docs/deepseek_v4_mla_dsa_design.md`). ⛳ WHAT DEPENDS ON IT: the plane offset
 * below, and nothing else — a per-plane stride pair in the tier row is what it would become, which is one
 * more word in a row that has one to spare. The build refuses an odd stride rather than halving it, so an
 * asymmetric format arrives as a REFUSAL and not as a misread address.
 * ⇒ ★ THE TWO PLANES ARE SEPARATE RATHER THAN INTERLEAVED FOR A REASON THAT OUTLIVES THE ASSUMPTION:
 * attention reads all of K for the scores and then all of V for the weighted sum, so each phase streams
 * one plane contiguously where an interleave would halve the useful bytes of every cache line. The old
 * tree agrees — `SilvannKVCachePage` carries `k_data` and `v_data` as separate pointers.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* The two keys a tier is sized by, spelled by STRINGIFICATION so neither can be misspelled in one place
 * and not the other. ⛳ NEITHER SUFFIX NAMES A UNIT — `__tokens` counts things and
 * `__bytes_per_token_layer` is a raw byte figure whose tail is `_layer` — so both are read by the
 * COUNT door, and the SIZE door refuses them. That is the suffix rule doing its work, not an exception
 * to it: `sys__settings__{size,count}` refuse each other's keys by construction. ▶ `sys/settings.cuh`. */
#define NN__CARTRIDGE__KEY_TOKENS(name)  "nn__cartridge_" #name "__tokens"
#define NN__CARTRIDGE__KEY_BYTES(name)   "nn__cartridge_" #name "__bytes_per_token_layer"

/* ── WHAT THIS FILE PUBLISHES ────────────────────────────────────────────────────────────────────────
 * ⛳ THE CARVE TAKES WHERE THE ARENA ENDED, exactly as the arena takes where the buffer carve ended. The
 * order is a dependency and not a style: each region is DEFINED as the first byte after the last one. */
static __device__ inline bool nn__cartridge__zzpackage_carve(uint64_t from,
                                                             uint64_t span_at,
                                                             uint64_t span_bytes,
                                                             uint64_t* ended);

/* ⛔⛔ TWO LAYER COUNTS, AND MISREADING ONE FOR THE OTHER IS THE DEFECT THIS FILE WAS REPAIRED FOR.
 *     `nn__cartridge__layers`        HOW MANY LANES THE TIERS ARE CUT INTO — the FULL-ATTENTION count,
 *                                    which is what a tier's region is a multiple of. On the shipping
 *                                    35B it is 10.
 *     `nn__cartridge__model_layers`  HOW MANY LAYERS THE MODEL HAS. On the same 35B it is 40.
 * ⇒ ★ THEY ARE NOT THE SAME NUMBER AND NOTHING MAKES THEM LOOK DIFFERENT AT A CALL SITE, which is why
 * the second is spelled with `model_` in it and this note sits between them. A model layer becomes a
 * lane only by going through `nn__cartridge__layer`; there is no other honest route.
 * ⛳ All of these answer zero or false when no cartridge was carved — the `e.boot()` case the device
 * suite runs in, where a config names no nn key at all. */
static __device__ inline uint64_t nn__cartridge__layers(void);
static __device__ inline uint64_t nn__cartridge__model_layers(void);
static __device__ inline bool     nn__cartridge__tier(uint64_t tier, sys__heap_node* answer);

/* ⭐⭐ THE TRANSLATION — a MODEL layer to the mechanism that serves it and its LANE within that
 * mechanism. ▶ the array's own account above for where the answer comes from and what it cost to go
 * without it. `false` when that layer is past the model's end, and it refuses QUIETLY: "no such layer"
 * is an answer the resolver returns, not a fault it files. */
static __device__ inline bool nn__cartridge__layer(uint64_t layer, uint64_t* mechanism, uint64_t* lane);

/* ⭐ THE RESOLVER. `(layer, token, plane)` to an address and a format, or false when that token is not
 * resident. ⛳ THE LAYER IS A MODEL LAYER — it is translated through the table above before anything is
 * addressed, so a hybrid model's deltanet layers refuse a K plane instead of being handed some other
 * layer's KV bytes. ⛳ IT TAKES NO CARTRIDGE ARGUMENT AND THAT IS DELIBERATE — ⚖ *"no second conversation"* is
 * ruled, so there is exactly one, and a parameter with one possible value is not a parameter. The
 * argument arrives the day a cartridge is a nameable object (⚖ `CARTRIDGE_PTR` is ruled *"a let/set on
 * the program to point at the current conversation"*); until then it would be a column every caller
 * fills with the same word. There are no callers yet, so adding it later costs nothing.
 * ⛳ IT FILLS THE REFERENCE'S BODY NODE RATHER THAN A HANDFUL OF OUT-PARAMETERS, which is what the caller
 * wanted anyway and keeps the `BOTH` case from growing the signature a seventh argument. */
static __device__ inline bool nn__cartridge__resolve(uint64_t layer, uint64_t token, uint64_t plane,
                                                     sys__heap_node* answer);

/* The word a model program says. Answers a `nn__kv_ref`, or an error naming which of the two it was. */
static __device__ __noinline__ void nn__cartridge__zzpackage_apply_reference(sys__heap_node* base,
                                                                            uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_CARTRIDGE__HEADER_CUH */
