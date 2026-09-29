#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_CARTRIDGE_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_CARTRIDGE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "../../../sys/contracts/objects/package.cuh"
/* ⭐ THE ORDER IS THE ID ORDER, THE LAYOUT ORDER *AND* THE TOKEN ORDER — oldest first. A token sinks from
 * the last row toward the first, and the sink never hands anything on.
 * ⛳ PRECISION IS DELIBERATELY NOT MONOTONE DOWN THIS LIST: row 0 is fp32 and row 1 is the coarsest there
 * is. That is the whole point of the sink and is worth meeting here rather than as a surprise.
 * ⛳ THE FOURTH COLUMN IS A BUILD-TIME PROPERTY AND SO IT LIVES IN THE LIST RATHER THAN IN THE ROW: a
 * tier's row holds what a ROTATION changes, and whether it rotates at all is not something a rotation
 * can change. */
#define NN__CARTRIDGE__PINNED   0u
#define NN__CARTRIDGE__ROTATES  1u

#define NN__CARTRIDGE_TIER_LIST(X)                                                                      \
    X(0, sink, NN__CARTRIDGE__SINK, NN__CARTRIDGE__PINNED)  /* the starting pulse, fp32, never handed on */ \
    X(1, tq4,  NN__CARTRIDGE__TQ4,  NN__CARTRIDGE__ROTATES) /* the cold tail — where length is carried   */ \
    X(2, tq6,  NN__CARTRIDGE__TQ6,  NN__CARTRIDGE__ROTATES) /* the warm middle                           */ \
    X(3, fp32, NN__CARTRIDGE__FP32, NN__CARTRIDGE__ROTATES) /* the hot window every token is born into   */

/* The ids a tier is named by. Nobody writes the count: it is one per row, so a list that grows and a
 * count that did not cannot happen. */
#define NN__CARTRIDGE__ZZPRIVATE_ENUM_ROW(id, name, konst, rot)   konst = (id),
#define NN__CARTRIDGE__ZZPRIVATE_TALLY_ROW(id, name, konst, rot)  + 1u

enum {
    NN__CARTRIDGE_TIER_LIST(NN__CARTRIDGE__ZZPRIVATE_ENUM_ROW)
    NN__CARTRIDGE__TIERS = 0u NN__CARTRIDGE_TIER_LIST(NN__CARTRIDGE__ZZPRIVATE_TALLY_ROW)
};

/* ⛔ THE DENSE-AND-ORDERED CHECK, the same one the buffer classes carry and for the same reason: the tier
 * table is indexed BY THE ID and the regions are laid out as a running total IN LIST ORDER, so an id that
 * is not its own position silently hands every tier another tier's bytes. Every total stays right, the
 * array is the right length, and every address resolves — which is why this is generated and not trusted. */
#define NN__CARTRIDGE__ZZPRIVATE_POS_ROW(id, name, konst, rot)  NN__CARTRIDGE__ZZPRIVATE_POS__##name,
enum { NN__CARTRIDGE_TIER_LIST(NN__CARTRIDGE__ZZPRIVATE_POS_ROW) };

#define NN__CARTRIDGE__ZZPRIVATE_DENSE_ROW(id, name, konst, rot)                                        \
    static_assert((int)(konst) == (int)NN__CARTRIDGE__ZZPRIVATE_POS__##name,                            \
                  "nn cartridge tier ids must be dense and in list order — the tier table is indexed "   \
                  "by the id and the regions are laid out in list order, so an id that is not its own "  \
                  "position hands a tier another tier's bytes and every total stays right.");
NN__CARTRIDGE_TIER_LIST(NN__CARTRIDGE__ZZPRIVATE_DENSE_ROW)

#undef NN__CARTRIDGE__ZZPRIVATE_ENUM_ROW
#undef NN__CARTRIDGE__ZZPRIVATE_TALLY_ROW
#undef NN__CARTRIDGE__ZZPRIVATE_POS_ROW
#undef NN__CARTRIDGE__ZZPRIVATE_DENSE_ROW

/* What the paired sum is handed, so the check and the walk cannot drift apart in how they spell a key. */
#define NN__CARTRIDGE__KEY_PREFIX      "nn__cartridge_"
#define NN__CARTRIDGE__KEY_TAIL_TOKENS "__tokens"
#define NN__CARTRIDGE__KEY_TAIL_BYTES  "__bytes_per_token_layer"

/* How many of a model's layers hold a per-token cache. ⛔ NOT THE LAYER TOTAL: a deltanet layer's state is
 * constant in the sequence length, so it does not scale with tokens and cannot be part of a per-token
 * product. It is not free either — ▶ the deltanet term beside this. */
#define NN__CARTRIDGE__KEY_LAYERS   "nn__model__kv_cache_layers_fullattn"

/* The conversation's text. ⚖ *"add the field for the conversation text, and also its config parameter
 * sizer."* One size, like the arena — it is a flat region with no interior structure this package knows. */
#define NN__CARTRIDGE__KEY_TEXT     "nn__conversation__text_size_kb"

/* ⚖ THE DELTANET STATE — *"use the max deltanet state for the context window so we have no oom."*
 * ⛳ IT IS O(1) IN THE SEQUENCE LENGTH, being a recurrent state rather than a per-token cache, so "for the
 * context window" is the model's own maximum and not a figure that grows with the conversation. Sizing it
 * at that maximum is what turns a mid-run out-of-memory into a refusal at boot.
 * ⭐ `MEASURED` from the old tree (`session/alloc.py:476`, `types.py:113`): conv_state(qkv_dim × 3) +
 * recurrent_state(num_v_heads × head_dim²) = 1,114,112 fp32 = 4,456,448 B ≈ 4.25 MiB per layer, and
 * Qwen3.5-MoE has 36 of them — ≈ 153 MiB, ~7.7% on top of the cartridge. It was costed and deliberately
 * unsummed until a ruling named it; this is that ruling.
 * ⛳⛳ AND IT ADDS EXACTLY ONE KEY, NOT TWO — THE LAYER COUNT WAS ALREADY IN THE FILE. Configs have
 * carried `nn__model__kv_cache_layers_deltanet` since the span was first summed, for the layer-total of
 * `nn_span_layout.md` §①ⓐ, and it is the same count this term multiplies. Minting a second key for it
 * would be two writable numbers obliged to agree with nothing checking them — the one thing this format
 * exists to refuse. ⛳ The finding at the foot of `package__impl.cuh` predicted this exactly: *"the
 * deltanet layer count, which the config file already carries"*. */
#define NN__CARTRIDGE__KEY_DELTANET_BYTES  "nn__deltanet_state__bytes_per_layer"
#define NN__CARTRIDGE__KEY_DELTANET_LAYERS "nn__model__kv_cache_layers_deltanet"

/* ── ⭐⭐ THE ARRAY THAT SAYS WHICH LAYER IS WHICH — AND IT IS THE AUTHORITY, NOT A HINT ──────────────
 *
 * ⚖ ARCHITECT: *"not a map, but an array that specifies what attention type is used by that
 * layer. this makes it possible to have a different layer if needed at zero cost"* — and, earlier, on the
 * shape of the thing: *"the authoritative orderer is the array itself."*
 *
 * ⛔⛔ WITHOUT IT A MODEL LAYER WAS READ AS A DENSE KV SLOT, WHICH IS A WRONG ADDRESS AND NOT A REFUSAL.
 * `MEASURED` from the shipping 35B (`full_attention_interval` 4 over 40 layers): full attention sits at
 * layers 3, 7, 11 … 39 and the other thirty are deltanet. A resolver bounding `layer < 10` answers model
 * layer 5 with SLOT 5's bytes — a layer that holds no KV at all — and refuses layer 19, which holds
 * lane 4. It fails in BOTH directions and neither one raises anything.
 *
 * ⭐ ONE CHARACTER PER LAYER, AND THE CHARACTERS ARE LETTERS RATHER THAN DIGITS ON PURPOSE. A digit
 * string is a number to `sys__settings__count`, so `…:1110` would read as one thousand one hundred and
 * ten and a caller reaching for the wrong door would get a plausible figure instead of a refusal. Letters
 * make that read impossible — which is the same property the unit suffixes buy, by the same means.
 *
 * ⛳ AND IT IS HF's OWN `layer_types` CARRIED ACROSS VERBATIM, which is why it is spelled with that name:
 * the composer reads `text_config["layer_types"]` as its PRIMARY source and falls back to an interval
 * only when the list is absent (`composer.py:173-184`). Taking the interval here would have built on the
 * weaker of two sources the host already ranks — and `sys/settings.cuh` had already refused it in the
 * architect's words: *"an interval would need its derivation rewritten the day a third mechanism
 * arrives."* An array needs nothing rewritten; it needs one more row in the list below. */
#define NN__CARTRIDGE__KEY_LAYER_TYPES "nn__model__layer_types"

/* ── THE MECHANISMS, ONE ROW EACH — id · the letter that spells it · the key that counts it ───────────
 * ⭐ THIS IS WHERE "AT ZERO COST" IS ACTUALLY PAID. A third attention mechanism is ONE ROW here: it gets
 * an id, a letter the array can carry, and the key whose count it must agree with. No resolver arm, no
 * parser change, no new reader.
 * ⛔ AND AN UNKNOWN LETTER IS A REFUSAL, NEVER A SKIP. A mechanism this build has never heard of has no
 * region carved for it, so there is no address to hand out — answering anything at all would be inventing
 * one. ⇒ ★ REFUSING TO BOOT IS THE ONLY HONEST ANSWER TO A MODEL THIS BINARY CANNOT ADDRESS. */
#define NN__CARTRIDGE__MECHANISM_LIST(X)                                                                \
    X(0u, 'f', FULLATTN, NN__CARTRIDGE__KEY_LAYERS)                                                     \
    X(1u, 'd', DELTANET, NN__CARTRIDGE__KEY_DELTANET_LAYERS)

#define NN__CARTRIDGE__ZZPRIVATE_MECH_ROW(id, letter, name, key)  NN__CARTRIDGE__MECH__##name = (id),
#define NN__CARTRIDGE__ZZPRIVATE_MECH_TALLY(id, letter, name, key)  + 1u
enum {
    NN__CARTRIDGE__MECHANISM_LIST(NN__CARTRIDGE__ZZPRIVATE_MECH_ROW)
    NN__CARTRIDGE__MECHANISMS = 0u NN__CARTRIDGE__MECHANISM_LIST(NN__CARTRIDGE__ZZPRIVATE_MECH_TALLY)
};
#undef NN__CARTRIDGE__ZZPRIVATE_MECH_ROW
#undef NN__CARTRIDGE__ZZPRIVATE_MECH_TALLY

/* ── A LAYER'S ROW IN THE TABLE — WHAT IT IS, AND WHICH ONE OF THOSE IT IS ───────────────────────────
 * ⛳ THE LANE IS STORED RATHER THAN COUNTED. Deriving it means counting that mechanism's letters before
 * this layer, which is O(layers) on a path that runs once per attention head per token. Counting it ONCE
 * at the carve turns the hot path back into a single borrow and an add — ⚖ *"the lane table derived once
 * at init, never per resolve."* */
#define NN__CARTRIDGE__LAYER_MECHANISM 0u   /* which row of the list above                              */
#define NN__CARTRIDGE__LAYER_LANE      1u   /* its index AMONG LAYERS OF THAT MECHANISM — not its own   */

/* ── A TIER'S ROW — SIX WORDS, WHICH IS EXACTLY WHAT A NODE HAS ──────────────────────────────────────
 * ⛳ THE FIT IS WORTH NAMING because it is what keeps a rollback a write: a tier's whole mutable state is
 * one node, so saving and restoring it is a copy of 64 bytes and never a walk.
 * ⛳ `FIRST` AND `END` ARE BOTH STORED THOUGH THE TIERS ARE ADJACENT and one tier's end is the next tier's
 * first. The redundancy is deliberate: a tier answers its own range without consulting its neighbours, so
 * the resolver reads ONE row per lookup instead of two. The adjacency is checked rather than relied on. */
#define NN__CARTRIDGE__TIER_AT        0u   /* the tier's first byte, inside nn's span                    */
#define NN__CARTRIDGE__TIER_CAPACITY  1u   /* how many tokens it holds — from the config, never moves    */
#define NN__CARTRIDGE__TIER_STRIDE    2u   /* bytes per token per layer, K and V together                */
#define NN__CARTRIDGE__TIER_FIRST     3u   /* the lowest token number resident in it                     */
#define NN__CARTRIDGE__TIER_END       4u   /* one past the highest — so END − FIRST is what it holds     */
#define NN__CARTRIDGE__TIER_PHYS      5u   /* the ring slot FIRST occupies                               */

/* Which half of a token's bytes is wanted. ⚖ *"i think each is valid and we could have a separate and a
 * together opcode to cast them."* The read is where they are separate — attention streams all of K, then
 * all of V — and TOGETHER is the write's shape, both arriving at once when a token is processed. */
#define NN__CARTRIDGE__PLANE_K        0u
#define NN__CARTRIDGE__PLANE_V        1u
#define NN__CARTRIDGE__PLANE_BOTH     2u   /* the whole token — and it needs TWO addresses; ▶ below      */
#define NN__CARTRIDGE__PLANES         3u

/* ⭐ WHAT A REFERENCE'S TIER WORD SAYS WHEN THE BYTES ARE NOT IN A TIER AT ALL. A deltanet layer's state
 * is a recurrent blob in its own region, not a run inside one of the four token tiers — so it needs a
 * FORMAT word that cannot collide with a real tier, and one past the last id is exactly that. The
 * dense-and-ordered static_assert above is what keeps it one past the last. */
#define NN__CARTRIDGE__TIER_DELTANET  ((uint64_t)NN__CARTRIDGE__TIERS)

/* ── WHAT A KV REFERENCE IS — AN ADDRESS THAT KNOWS ITS OWN FORMAT ───────────────────────────────────
 * ⚖ *"like reference (it will give you a datatype that tells you the type of kv cache object and the
 * address)."*
 * ⭐⭐ THE TIER IS NOT DECORATION ON THE ADDRESS, IT IS WHAT MAKES A CROSS-TIER READ POSSIBLE. ⚖ *"maybe
 * we need to have (kv_subsection ((from to) (from to) (from to))) to pick a subsection of the kv cache
 * even between different types"* — a subsection spanning a rotation boundary hands back elements in two
 * FORMATS, and a consumer that cannot tell which is which cannot decode either. So the format travels
 * with the address or the subsection verb is unwritable.
 * ⛔ AND ITS BYTES ARE NOT THE HEAP'S. A reference is a 64-byte node naming a run inside nn's span; when
 * its last holder goes the node's room comes back and the span is untouched. ⚖ *"the kv cache is not
 * deallocated"* — ⇒ its release hook is the default, and that is the whole of it.
 *
 * ── ⛔⛔ AND THE "TOGETHER" FORM NEEDS TWO ADDRESSES, WHICH IS WHERE TWO DECISIONS MEET ──────────────
 * The planes are SEPARATE runs, so a token's K and V are not adjacent and NO single (address, length)
 * pair can name the token whole. That is not a flaw in either decision — it is the two of them being
 * honest with each other. ⇒ a `BOTH` reference carries the V address in a word of its own, and `BYTES`
 * stays what it always is: the length of ONE run. ⛳ THE ALTERNATIVE WAS TO INTERLEAVE K AND V so the
 * token is contiguous, and it is refused for the reason in this file's header: each attention phase
 * streams one plane, and interleaving halves the useful bytes of every cache line to save a word here.
 * ⇒ ★ WHEN A CONVENIENCE AND A LAYOUT DISAGREE, THE LAYOUT IS THE ONE THAT IS MEASURED. */
#define NN__KV_REF__AT      0u   /* where the bytes are — K's address when the plane is BOTH            */
#define NN__KV_REF__TIER    1u   /* which tier — which is to say, what FORMAT they are in               */
#define NN__KV_REF__PLANE   2u   /* K, V, or the token whole                                            */
#define NN__KV_REF__BYTES   3u   /* the length of ONE run, so a reader needs no second lookup           */
#define NN__KV_REF__AT_V    4u   /* V's address — written only for BOTH, and zero for the single planes */
#define NN__KV_REF__NODES   2u   /* the head, and one node holding the five above                       */

/* ── THE FAULTS THIS RAISES ──────────────────────────────────────────────────────────────────────────
 * Four letters each, so a fault word read off a dump names its own cause without a table. */
#define NN__CARTRIDGE__FAULT_CONFIG   0x4E434346ull  /* "NCCF" — the card cannot read what the host summed */
#define NN__CARTRIDGE__FAULT_ROOM     0x4E43524Dull  /* "NCRM" — the regions do not fit what is left       */
#define NN__CARTRIDGE__FAULT_RANGE    0x4E435247ull  /* "NCRG" — that token or layer is not resident       */
#define NN__CARTRIDGE__FAULT_NO_CACHE 0x4E434E43ull  /* "NCNC" — asked for a token before the carve ran    */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_CARTRIDGE_CUH */
