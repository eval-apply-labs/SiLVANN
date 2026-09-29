#ifndef SILVANN__PACKAGES_NN_CPU_CARTRIDGE__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_CARTRIDGE__IMPL_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "cartridge__header.cuh"         /* the contract these definitions answer */
#include "package__header.cuh"           /* this package's id, and its hatch rows */
#include "../../sys/cpu/package__header.cuh"    /* where a package keeps its own device-wide state */
#include "../../sys/cpu/node_array__header.cuh" /* the tier table: one row per tier, indexed by its id */
#include "../../sys/cpu/list__header.cuh"       /* a form is a list, and the verb reads its arguments */
#include "../../sys/cpu/heap__header.cuh"       /* where a kv reference node is carved from */
#include "../../sys/cpu/heap_object__header.cuh" /* holds taken and given back */
#include "../../sys/cpu/settings.cuh"           /* the card's own reader for the keys the host summed */
#include "../../sys/cpu/opcodes/opcodes.cuh"            /* how a verb answers, and how it refuses */
/* ══ nn — THE CARTRIDGE, CARVED AND RESOLVED ═════════════════════════════════════════════════════════
 *
 * Three things live here: where the regions ARE (the carve), where a token IS (the resolver), and the
 * word a program says to ask (the verb). ▶ `cartridge__header.cuh` for why each is shaped as it is.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


/* ── THE ROWS, READ BY KNOWING WHO THIS PACKAGE IS ───────────────────────────────────────────────────
 * Every one answers a zero or a false when no cartridge was carved. ⛳ THAT IS NOT A FAILURE PATH — it is
 * the `e.boot()` case the whole device suite runs in, where a config names no nn key at all. */
static __device__ inline sys__heap_node nn__cartridge__zzprivate_region_row(void) {
    return sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE);
}

static __device__ inline uint64_t nn__cartridge__at(void) {
    const sys__heap_node row = nn__cartridge__zzprivate_region_row();
    if (row.dtype != SYS__KIND__VALUE_INT) return 0ull;
    return row.args[0];
}

static __device__ inline uint64_t nn__cartridge__bytes(void) {
    const sys__heap_node row = nn__cartridge__zzprivate_region_row();
    if (row.dtype != SYS__KIND__VALUE_INT) return 0ull;
    return row.args[1];
}

static __device__ inline uint64_t nn__cartridge__layers(void) {
    const sys__heap_node row = nn__cartridge__zzprivate_region_row();
    if (row.dtype != SYS__KIND__VALUE_INT) return 0ull;
    return row.args[2];
}

static __device__ inline uint64_t nn__cartridge__zzprivate_tiers(void) {
    const sys__heap_node row = sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE_TIERS);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

/* ⛳ BORROWED, NOT `get`. The tier table is read on every single resolve and the row is copied out by
 * value; taking a hold on a table this package will hold for the life of the boot would be a retain and a
 * release around a read that cannot outlive either. ▶ the same reasoning at `buffer__impl.cuh`. */
static __device__ inline bool nn__cartridge__tier(uint64_t tier, sys__heap_node* answer) {
    if (answer == 0 || tier >= (uint64_t)NN__CARTRIDGE__TIERS) return false;
    const uint64_t table = nn__cartridge__zzprivate_tiers();
    if (table == 0ull) return false;
    const sys__heap_node row = sys__node_array__borrow(table, tier);
    if (row.dtype != SYS__KIND__VALUE_INT) return false;
    *answer = row;
    return true;
}

/* ── ⭐⭐ WHAT A MODEL LAYER ACTUALLY IS — THE ONE LOOKUP THAT STOPS A LAYER BEING READ AS A LANE ──────
 * ⛳ BORROWED FOR THE SAME REASON THE TIER ROW IS: this runs on every resolve and the row is copied out
 * by value, so a hold taken here would be a retain and a release around a read that cannot outlive it.
 * ⛔⛔ AND THE LENGTH IS CHECKED BEFORE THE BORROW RATHER THAN LEFT TO IT. `sys__node_array__borrow`
 * RAISES `FAULT_RANGE` on an index past the end — correct for a program that indexed off the end, and
 * wrong here, where "that layer does not exist" is an ANSWER this function returns as `false` and the
 * verb turns into its own fault word. Letting the borrow raise would file a second, differently-named
 * fault for the same one question and leave it on the channel after an honest refusal. */
static __device__ inline uint64_t nn__cartridge__zzprivate_layer_table(void) {
    const sys__heap_node row = sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE_LAYERS);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

static __device__ inline uint64_t nn__cartridge__model_layers(void) {
    const uint64_t table = nn__cartridge__zzprivate_layer_table();
    if (table == 0ull) return 0ull;
    return sys__node_array__length(table);
}

static __device__ inline bool nn__cartridge__layer(uint64_t layer, uint64_t* mechanism, uint64_t* lane) {
    if (mechanism == 0 || lane == 0) return false;
    const uint64_t table = nn__cartridge__zzprivate_layer_table();
    if (table == 0ull) return false;
    if (layer >= sys__node_array__length(table)) return false;      /* ⛔ quietly — ▶ the note above */
    const sys__heap_node row = sys__node_array__borrow(table, layer);
    if (row.dtype != SYS__KIND__VALUE_INT) return false;
    *mechanism = row.args[NN__CARTRIDGE__LAYER_MECHANISM];
    *lane      = row.args[NN__CARTRIDGE__LAYER_LANE];
    return true;
}

/* The deltanet state region: where it starts, how many bytes ONE layer's state is. */
static __device__ inline sys__heap_node nn__cartridge__zzprivate_deltanet_row(void) {
    return sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE_DELTANET);
}

/* ── THE CARVE ───────────────────────────────────────────────────────────────────────────────────────
 *
 * ⭐⭐ IT READS THE SAME KEYS THE HOST SUMMED, which is the arrangement the whole span is built on: one
 * text, two readers, and no total written down anywhere for the two to disagree with. A key this side
 * cannot find means the two parsers do not agree about the same bytes — a fault, never a default.
 *
 * ⛔ THE ORDER IS THE LIST'S ORDER, which is the id order, which is the layout order, which is the token
 * order. `cartridge__header.cuh` generates a static_assert that the first three are the same thing,
 * because nothing at runtime could tell.
 *
 * ⚖⚖ ABSENT IS FINE, PARTIAL REFUSES — the same ruling the host init obeys, applied to the same keys from
 * the other side. THIRTEEN keys make a cartridge: four tiers of two, one layer count per mechanism, the
 * text, the deltanet byte figure, and the layer-types array. Naming NONE of them stands with no
 * cartridge; naming SOME refuses, because half a cartridge is an edit that went wrong and not a default
 * anybody chose. ⛳ THE COUNT IS NOT WRITTEN DOWN AS A CONSTANT — `terms` is added up as the keys are
 * read, and the mechanism rows add their own, so a new mechanism cannot leave a stale total behind it.
 * ⇒ ⛳ AND THE COUNT IS TAKEN RATHER THAN ASSUMED FROM THE HOST HAVING PASSED. The host reads the same
 * text and would have refused a partial set — so this can only fire when the two READERS disagree, which
 * is precisely the failure it exists to catch and the one nothing downstream could diagnose. */
static __device__ inline bool nn__cartridge__zzpackage_carve(uint64_t from,
                                                             uint64_t span_at,
                                                             uint64_t span_bytes,
                                                             uint64_t* ended) {
    if (span_at == 0ull || ended == 0) return false;
    *ended = from;

    /* Everything the twelve keys say, read before a byte is laid down. */
    uint64_t tokens[NN__CARTRIDGE__TIERS], stride[NN__CARTRIDGE__TIERS];
    unsigned spoken = 0u, terms = 0u;

#define NN__CARTRIDGE__ZZPRIVATE_READ_ROW(id, name, konst, rot)                                         \
    {                                                                                                   \
        uint64_t n = 0ull, b = 0ull;                                                                    \
        const bool a1 = sys__settings__count((const uint8_t*)NN__CARTRIDGE__KEY_TOKENS(name),           \
                                             sizeof(NN__CARTRIDGE__KEY_TOKENS(name)) - 1ull, &n);       \
        const bool a2 = sys__settings__count((const uint8_t*)NN__CARTRIDGE__KEY_BYTES(name),            \
                                             sizeof(NN__CARTRIDGE__KEY_BYTES(name)) - 1ull, &b);        \
        /* ⛔ A TOKEN COUNT WITHOUT ITS BYTE FIGURE IS NOT A TIER. Taking one of the two would size a     \
         * region from a figure whose other half somebody deleted — which carves fine and is wrong by a  \
         * factor, and every total downstream stays healthy. */                                          \
        if (a1 != a2) return false;                                                                      \
        tokens[konst] = n; stride[konst] = b;                                                            \
        terms += 2u; if (a1) spoken += 2u;                                                                \
    }
    NN__CARTRIDGE_TIER_LIST(NN__CARTRIDGE__ZZPRIVATE_READ_ROW)
#undef NN__CARTRIDGE__ZZPRIVATE_READ_ROW

    /* ⭐ THE PER-MECHANISM LAYER COUNTS, ONE ROW OF THE LIST EACH — so a third mechanism brings its own
     * key into the required set with no edit here. `declared[]` is what the HOST summed the span from,
     * and the array below has to agree with it letter for letter. */
    uint64_t declared[NN__CARTRIDGE__MECHANISMS];
#define NN__CARTRIDGE__ZZPRIVATE_READ_MECH(id, letter, name, key)                                       \
    {                                                                                                   \
        uint64_t n = 0ull;                                                                              \
        if (sys__settings__count((const uint8_t*)(key), sizeof(key) - 1ull, &n)) spoken += 1u;          \
        declared[id] = n; terms += 1u;                                                                  \
    }
    NN__CARTRIDGE__MECHANISM_LIST(NN__CARTRIDGE__ZZPRIVATE_READ_MECH)
#undef NN__CARTRIDGE__ZZPRIVATE_READ_MECH

    uint64_t text = 0ull, dn_bytes = 0ull;
    const bool k2 = sys__settings__size((const uint8_t*)NN__CARTRIDGE__KEY_TEXT,
                                        sizeof(NN__CARTRIDGE__KEY_TEXT) - 1ull, &text);
    const bool k3 = sys__settings__count((const uint8_t*)NN__CARTRIDGE__KEY_DELTANET_BYTES,
                                         sizeof(NN__CARTRIDGE__KEY_DELTANET_BYTES) - 1ull, &dn_bytes);

    /* ⭐⭐ THE ARRAY ITSELF, READ AS TEXT AND NOT AS A FIGURE. ⚖ *"an array that specifies what attention
     * type is used by that layer."* It goes through the RAW value door because it is not a number and
     * must never be read as one — ▶ the letters-not-digits argument in the header. */
    const uint64_t settings = sys__settings__published();
    const uint64_t types = (settings == 0ull) ? 0ull
        : sys__settings__value(settings, (const uint8_t*)NN__CARTRIDGE__KEY_LAYER_TYPES,
                               sizeof(NN__CARTRIDGE__KEY_LAYER_TYPES) - 1ull);
    const bool k5 = (types != 0ull) && sys__string__is(types);

    terms += 3u;
    if (k2) spoken += 1u;
    if (k3) spoken += 1u;
    if (k5) spoken += 1u;

    const uint64_t layers    = declared[NN__CARTRIDGE__MECH__FULLATTN];
    const uint64_t dn_layers = declared[NN__CARTRIDGE__MECH__DELTANET];

    if (spoken == 0u) return true;        /* no cartridge key anywhere: no cartridge, and that stands */
    if (spoken != terms) return false;    /* half a cartridge is a mistake, not a default             */

    /* ⛔ A LAYER COUNT OF ZERO WOULD MAKE EVERY TIER A ZERO-LENGTH REGION and every resolve answer an
     * address inside somebody else's bytes. The host multiplied by this figure too, so a zero here means
     * it summed a cartridge of nothing and asked for it — caught at the one moment there is something to
     * be done about it. */
    if (layers == 0ull) return false;

    const uint64_t table = sys__node_array__create((uint64_t)NN__CARTRIDGE__TIERS);
    if (table == 0ull) return false;

    bool ok = true;

    /* ── ⭐⭐ THE LAYER TABLE — READ ONCE HERE SO NOTHING EVER COUNTS LETTERS ON THE HOT PATH ─────────
     *
     * ⚖ *"the authoritative orderer is the array itself."* Each layer gets the mechanism its letter
     * spells and its LANE — the index among layers of that same mechanism, which is what the regions are
     * actually laid out by. `MEASURED` on the shipping 35B: model layer 19 is full attention at lane 4,
     * and layer 5 is deltanet at lane 4 — two different regions reached by the same number, which is
     * precisely the collision a dense-slot read could not see.
     *
     * ⛳ THE THREE FACTS ARE MADE TO CHECK EACH OTHER RATHER THAN MERELY CO-EXIST. The array's LENGTH is
     * the model's layer total, and each letter's TALLY is that mechanism's declared count — so the
     * config's counts and its array cannot disagree without this refusing. ⇒ ★ IT IS THE SAME RULE THE
     * SPAN TOTAL OBEYS, POINTED THE OTHER WAY: there, the figure nobody writes cannot be wrong; here, two
     * figures somebody DID write are made to answer for each other.
     * ⚠ AND THE SIZING STILL USES THE DECLARED COUNT, DELIBERATELY — the host summed the span from that
     * key, so carving from the same one keeps the two readers provably on the same input. The array is
     * the authority on ORDER, which is the question the host never has to ask. */
    const uint64_t total = sys__string__length(types);
    uint64_t layer_table = 0ull;
    {
        uint64_t summed = 0ull;
        for (unsigned m = 0u; m < (unsigned)NN__CARTRIDGE__MECHANISMS; ++m) summed += declared[m];
        if (total == 0ull || total != summed) ok = false;
    }
    if (ok) {
        layer_table = sys__node_array__create(total);
        if (layer_table == 0ull) ok = false;
    }
    if (ok) {
        const uint8_t* letters = sys__string__bytes(types);
        uint64_t seen[NN__CARTRIDGE__MECHANISMS];
        for (unsigned m = 0u; m < (unsigned)NN__CARTRIDGE__MECHANISMS; ++m) seen[m] = 0ull;
        for (uint64_t l = 0ull; ok && l < total; ++l) {
            unsigned which = (unsigned)NN__CARTRIDGE__MECHANISMS;
            switch (letters[l]) {
#define NN__CARTRIDGE__ZZPRIVATE_LETTER_ROW(id, letter, name, key)                                      \
                case (uint8_t)(letter): which = (id); break;
                NN__CARTRIDGE__MECHANISM_LIST(NN__CARTRIDGE__ZZPRIVATE_LETTER_ROW)
#undef NN__CARTRIDGE__ZZPRIVATE_LETTER_ROW
                default: break;
            }
            /* ⛔ AN UNKNOWN LETTER REFUSES THE WHOLE CARVE. This build has carved no region for a
             * mechanism it has never heard of, so there is no address it could honestly hand out for
             * that layer — and a model whose layers it cannot all address is not a model it can run. */
            if (which >= (unsigned)NN__CARTRIDGE__MECHANISMS) { ok = false; break; }
            sys__heap_node lrow;
            lrow.dtype = SYS__KIND__VALUE_INT; lrow.num_args = 0u; lrow.op_code = 0ull;
            lrow.args[NN__CARTRIDGE__LAYER_MECHANISM] = (uint64_t)which;
            lrow.args[NN__CARTRIDGE__LAYER_LANE]      = seen[which];
            lrow.args[2] = 0ull; lrow.args[3] = 0ull; lrow.args[4] = 0ull; lrow.args[5] = 0ull;
            seen[which] += 1ull;
            ok = sys__node_array__set(layer_table, l, &lrow);
        }
        /* ⛔ AND EVERY TALLY MUST MATCH ITS OWN KEY, not just add up to the right total — `fdfd` and
         * `ffdd` have the same length and the same two tallies, and a config that swapped the two counts
         * would pass a sum check while putting every lane one region away from its bytes. */
        for (unsigned m = 0u; ok && m < (unsigned)NN__CARTRIDGE__MECHANISMS; ++m)
            if (seen[m] != declared[m]) ok = false;
    }
    uint64_t at = from;                          /* the running total, in list order */

#define NN__CARTRIDGE__ZZPRIVATE_CARVE_ROW(id, name, konst, rot)                                        \
    if (ok) {                                                                                           \
        const uint64_t n = tokens[konst], b = stride[konst];                                            \
        /* ⛔ AN ODD STRIDE CANNOT BE SPLIT INTO TWO PLANES, and halving it anyway would put every V       \
         * address half a byte wrong for the whole run. ▶ the ASSUMED note in the header: an asymmetric   \
         * K and V arrives here as a REFUSAL rather than as a misread address, which is the only way a    \
         * format this file was not written for can announce itself. */                                  \
        if ((b & 1ull) != 0ull) ok = false;                                                              \
        /* The region must land inside the room the host took. A carve that walks past the end hands out  \
         * addresses that belong to nobody, and every count stays healthy. */                            \
        uint64_t run = 0ull;                                                                             \
        if (ok) {                                                                                        \
            if (n != 0ull && b > 0xFFFFFFFFFFFFFFFFull / n) ok = false;                                   \
            else {                                                                                       \
                run = n * b;                                                                             \
                if (run != 0ull && layers > 0xFFFFFFFFFFFFFFFFull / run) ok = false;                      \
                else run *= layers;                                                                      \
            }                                                                                            \
        }                                                                                                \
        if (ok && ((at - span_at) > span_bytes || run > span_bytes - (at - span_at))) ok = false;         \
        if (ok) {                                                                                        \
            /* ⭐ EVERY TIER IS BORN EMPTY — FIRST, END AND PHYS ALL ZERO. An empty tier answers no token  \
             * at all (the resolver asks END > FIRST first), so a freshly booted cartridge correctly      \
             * holds nothing, and the first writer is what gives these words meaning. */                 \
            sys__heap_node row;                                                                          \
            row.dtype = SYS__KIND__VALUE_INT; row.num_args = 0u; row.op_code = 0ull;                      \
            row.args[NN__CARTRIDGE__TIER_AT]       = at;                                                 \
            row.args[NN__CARTRIDGE__TIER_CAPACITY] = n;                                                  \
            row.args[NN__CARTRIDGE__TIER_STRIDE]   = b;                                                  \
            row.args[NN__CARTRIDGE__TIER_FIRST]    = 0ull;                                               \
            row.args[NN__CARTRIDGE__TIER_END]      = 0ull;                                               \
            row.args[NN__CARTRIDGE__TIER_PHYS]     = 0ull;                                               \
            ok = sys__node_array__set(table, (uint64_t)(konst), &row);                                    \
        }                                                                                                \
        if (ok) at += run;                                                                                \
    }
    NN__CARTRIDGE_TIER_LIST(NN__CARTRIDGE__ZZPRIVATE_CARVE_ROW)
#undef NN__CARTRIDGE__ZZPRIVATE_CARVE_ROW

    /* ── THE TWO FLAT REGIONS, AFTER THE TIERS AND IN THE ORDER THE HOST SUMMED THEM ──────────────────
     * ⛳ NEITHER HAS ANY INTERIOR STRUCTURE THIS PACKAGE KNOWS: the text is the conversation's bytes and
     * the deltanet state is a per-layer run an opcode will address. Both are a base and a length, which
     * is all a region is. */
    uint64_t text_at = 0ull, dn_at = 0ull, dn_run = 0ull;
    if (ok) {
        if ((at - span_at) > span_bytes || text > span_bytes - (at - span_at)) ok = false;
        else { text_at = at; at += text; }
    }
    if (ok) {
        if (dn_bytes != 0ull && dn_layers > 0xFFFFFFFFFFFFFFFFull / dn_bytes) ok = false;
        else {
            dn_run = dn_bytes * dn_layers;
            if ((at - span_at) > span_bytes || dn_run > span_bytes - (at - span_at)) ok = false;
            else { dn_at = at; at += dn_run; }
        }
    }

    sys__heap_node row;
    if (ok) {
        row.dtype = SYS__KIND__OBJECT_REFERENCE; row.num_args = 0u; row.op_code = 0ull;
        row.args[0] = table;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE_TIERS, &row);
    }
    if (ok) {
        row.dtype = SYS__KIND__VALUE_INT; row.num_args = 0u; row.op_code = 0ull;
        row.args[0] = from; row.args[1] = at - from; row.args[2] = layers;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE, &row);
    }
    if (ok) {
        row.dtype = SYS__KIND__VALUE_INT; row.num_args = 0u; row.op_code = 0ull;
        row.args[0] = text_at; row.args[1] = text;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE_TEXT, &row);
    }
    if (ok) {
        row.dtype = SYS__KIND__VALUE_INT; row.num_args = 0u; row.op_code = 0ull;
        /* ⛳ THE PER-LAYER FIGURE IS CARRIED, NOT RE-DERIVED. `dn_run / dn_layers` would recover it, but
         * a division that is only correct because a multiply produced it is a fact resting on a fact —
         * and the word was free. A deltanet lane's address is `dn_at + lane × args[2]`. */
        row.args[0] = dn_at; row.args[1] = dn_run; row.args[2] = dn_bytes;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE_DELTANET, &row);
    }
    if (ok) {
        row.dtype = SYS__KIND__OBJECT_REFERENCE; row.num_args = 0u; row.op_code = 0ull;
        row.args[0] = layer_table;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__CARTRIDGE_LAYERS, &row);
    }

    /* The rows took their own holds. These two are the carve's, and they go back whether it worked or
     * not — on the failing path that is what takes the half-built tables apart. */
    (void)sys__heap_object__release(table);
    if (layer_table != 0ull) (void)sys__heap_object__release(layer_table);

    /* ⭐ AND WHERE THE WALK STOPPED IS WHAT COMES NEXT, written only on success — a refused carve laid
     * some unknown prefix of the regions down, so its `at` describes a half-built layout and naming
     * anything from it would be worse than having no cartridge at all. */
    if (ok) *ended = at;
    return ok;
}

/* ── ⭐⭐ THE NEWEST TOKEN THE CARTRIDGE HOLDS, WHICH IS THE ONLY ONE A RECURRENT STATE CAN ANSWER FOR ─
 * The tiers state their own ranges and nothing orders them, so this is a max over four rows rather than
 * a read of the last one. ⛳ NOTHING RESIDENT IS `false` AND NOT A ZERO: token 0 is a real token, and a
 * freshly carved cartridge holding nothing must not report that it holds token 0's state. */
static __device__ inline bool nn__cartridge__zzprivate_newest(uint64_t* newest) {
    if (newest == 0) return false;
    bool any = false; uint64_t high = 0ull;
    for (uint64_t t = 0ull; t < (uint64_t)NN__CARTRIDGE__TIERS; ++t) {
        sys__heap_node row;
        if (!nn__cartridge__tier(t, &row)) return false;
        const uint64_t end = row.args[NN__CARTRIDGE__TIER_END];
        if (end <= row.args[NN__CARTRIDGE__TIER_FIRST]) continue;      /* that tier holds nothing */
        if (!any || end > high) { high = end; any = true; }
    }
    if (!any) return false;
    *newest = high - 1ull;
    return true;
}

/* ── ⛔⛔ A DELTANET LAYER, AND WHY IT ANSWERS FOR ONE TOKEN ONLY ─────────────────────────────────────
 *
 * ⚖ THE RULING WAS *"the deltanet resolve is a partial view up to that token, which will be used for amb
 * if we want to simulate different paths."* ⛔ THAT VIEW DOES NOT EXIST TO BE TAKEN, AND SAYING SO IS THE
 * point of this function. `MEASURED` from the old tree (`MODEL_CMD_LINEAR_ATTN.cuh:178,193,207`):
 * `state_base = v_head_idx * head_dim² + v_idx * head_dim` HAS NO TOKEN TERM, and the kernel writes
 * `recurrent_state[state_base + k] = h_kv` — IN PLACE, over the previous token's state. A GatedDeltaNet
 * layer keeps ONE state, not a history, and it is O(1) in the sequence length (which is exactly why it
 * was excluded from the per-token product this file sums). There is no byte anywhere in the region that
 * holds what the state was at an earlier token.
 * ⇒ ★★ SO THE HONEST ANSWER FOR A PAST TOKEN IS A REFUSAL, AND THE REFUSAL IS THE FEATURE. Handing back
 * the CURRENT state for a question about token 5 would be the identical defect this whole file is being
 * repaired for — a wrong answer that resolves, rather than a right one that refuses.
 *
 * ⚖⚖ TODO, RULED AND DELIBERATELY NOT BUILT: *"if we ask for a token between the checkpoint
 * and now add a todo that deltanet can be recalculated. dont implement it now tho, just add it as a
 * note."* ⇒ THIS REFUSAL IS THE CALL SITE THAT WORK FILLS IN: with checkpoints at the rotations, a token
 * between the newest checkpoint and now is reachable by REPLAYING FORWARD from it, and only then can
 * `amb` explore two paths. ⛳ THE PRICE IS ALREADY COSTED and is why it is not built today: the state is
 * 4.25 MiB per layer × 36 layers ≈ 153 MiB per divergence point, against a KV rollback of six words.
 * Checkpoint-and-replay is what turns that into TWO state sets rather than N.
 *
 * ⛔ AND K/V ARE REFUSED RATHER THAN IGNORED. A recurrent state has no planes — there is no half of it to
 * ask for — so a caller naming one has confused a deltanet layer for an attention layer, which is the
 * very confusion this translation exists to catch. Saying yes to a meaningless argument is how it would
 * go on being confused. */
static __device__ inline bool nn__cartridge__zzprivate_resolve_deltanet(uint64_t lane, uint64_t token,
                                                                        uint64_t plane,
                                                                        sys__heap_node* answer) {
    if (plane != (uint64_t)NN__CARTRIDGE__PLANE_BOTH) return false;

    const sys__heap_node region = nn__cartridge__zzprivate_deltanet_row();
    if (region.dtype != SYS__KIND__VALUE_INT) return false;
    const uint64_t at    = region.args[0];
    const uint64_t run   = region.args[1];
    const uint64_t bytes = region.args[2];
    if (bytes == 0ull || run == 0ull) return false;   /* and `run / bytes` below needs the first of these */
    /* ⛳ THRIFT, NOT CORRECTNESS — `MEASURED` by deleting it: the suite stays GREEN. The lane
     * cannot reach this bound today, because `run / bytes` IS `dn_layers`, which the carve has already
     * checked equals the number of `d`s in the array, which is what lanes were handed out from. The two
     * facts come from the same carve, so they cannot disagree. ⇒ ★ IT IS KEPT AS A BOUND ON AN ADDRESS
     * COMPUTATION, WHICH IS A CHEAP THING TO KEEP AND AN EXPENSIVE THING TO WANT BACK — but it is NOT
     * what makes the code correct, and a comment saying otherwise would be the third time this file
     * dressed a skip-early as an invariant. ⛳ RETIREMENT: it becomes load-bearing the day the deltanet
     * region is sized by anything other than the mechanism count the layer table is built from. */
    if (lane >= run / bytes) return false;

    uint64_t newest = 0ull;
    if (!nn__cartridge__zzprivate_newest(&newest)) return false;   /* nothing resident: no state either */
    if (token != newest) return false;                             /* ▶ the TODO above */

    answer->dtype = SYS__KIND__VALUE_INT; answer->num_args = 0u; answer->op_code = 0ull;
    answer->args[NN__KV_REF__AT]    = at + lane * bytes;
    answer->args[NN__KV_REF__TIER]  = NN__CARTRIDGE__TIER_DELTANET;   /* its FORMAT, and not a tier */
    answer->args[NN__KV_REF__PLANE] = plane;
    answer->args[NN__KV_REF__BYTES] = bytes;
    answer->args[NN__KV_REF__AT_V]  = 0ull;        /* there is no second run — the state is one blob */
    answer->args[5]                 = 0ull;
    return true;
}

/* ── THE RESOLVER — FOUR COMPARISONS AND AN ADD ──────────────────────────────────────────────────────
 *
 * ⭐⭐ THIS IS THE WHOLE OF `cache[layer][token]`, AND IT TOUCHES NO STRUCTURE. The tiers are walked in
 * list order and each is asked whether it holds this token; the first that does answers, and the address
 * is arithmetic from that point on.
 *
 * ⛔⛔ AN EMPTY TIER NEEDS NO TEST OF ITS OWN, AND I WROTE ONE AND CALLED IT LOAD-BEARING. The line was
 * `if (end <= first) continue;`, with a paragraph here saying that without it a token of 0 would resolve
 * inside all four empty tiers. **Deleting it: GREEN.** The range test below subsumes it completely —
 * `token >= end` with `end == first` rejects every token there is, and so does any inverted row.
 * ⇒ ★★ THIRD TIME IN TWO DAYS I HAVE DRESSED A SKIP-EARLY AS AN INVARIANT, and a careful read passed it
 * both times. ⛳ The one thing that found it was deleting the line and running the suite — which is now a
 * standing check on this tree and not a thing to be remembered.
 *
 * ⛳ AND THE WALK RELIES ON NO ORDERING BETWEEN TIERS. The ranges happen to be adjacent and in age order,
 * but nothing here reads them that way: each row states its own range and is believed about it alone.
 * ⇒ ★ WHICH IS WHY A ROTATION CANNOT BREAK THIS BY BEING HALF DONE — a token mid-move is in one range or
 * neither, and neither is an honest `false` rather than a wrong address. */
static __device__ inline bool nn__cartridge__resolve(uint64_t layer, uint64_t token, uint64_t plane,
                                                     sys__heap_node* answer) {
    if (answer == 0 || plane >= (uint64_t)NN__CARTRIDGE__PLANES) return false;

    /* ⭐⭐ THE LAYER IS TRANSLATED BEFORE ANYTHING IS ADDRESSED, AND THAT IS THE WHOLE FIX. What arrives
     * is a MODEL layer; what the regions are laid out by is a LANE within one mechanism. Reading the
     * first as the second is an address, not a refusal — ▶ the header's account of what it cost. */
    uint64_t mechanism = 0ull, lane = 0ull;
    if (!nn__cartridge__layer(layer, &mechanism, &lane)) return false;

    if (mechanism == (uint64_t)NN__CARTRIDGE__MECH__DELTANET)
        return nn__cartridge__zzprivate_resolve_deltanet(lane, token, plane, answer);
    if (mechanism != (uint64_t)NN__CARTRIDGE__MECH__FULLATTN) return false;

    for (uint64_t t = 0ull; t < (uint64_t)NN__CARTRIDGE__TIERS; ++t) {
        sys__heap_node row;
        if (!nn__cartridge__tier(t, &row)) return false;

        const uint64_t first    = row.args[NN__CARTRIDGE__TIER_FIRST];
        const uint64_t end      = row.args[NN__CARTRIDGE__TIER_END];
        const uint64_t capacity = row.args[NN__CARTRIDGE__TIER_CAPACITY];
        /* An empty tier has END == FIRST, so this one comparison rejects every token in it. */
        if (token < first || token >= end) continue;
        if (capacity == 0ull) return false;               /* a range inside a tier with no room for it */

        /* ⭐ THE RING, AND THE CONDITIONAL SUBTRACT IS EXACT. `token − first` is below `capacity` because
         * the range never exceeds the room, and `phys_first` is below it by its own invariant — so the
         * sum is below twice `capacity` and one subtraction lands it. No modulo, and no power-of-two
         * capacity: rounding the shipping 192,000-token tq4 tier up to 262,144 would cost 405 MiB to
         * save an instruction this does not need. */
        uint64_t phys = row.args[NN__CARTRIDGE__TIER_PHYS] + (token - first);
        if (phys >= capacity) phys -= capacity;

        const uint64_t half       = row.args[NN__CARTRIDGE__TIER_STRIDE] / 2ull;
        const uint64_t plane_run  = capacity * half;      /* one plane, one layer */
        const uint64_t layer_base = row.args[NN__CARTRIDGE__TIER_AT] + lane * (plane_run * 2ull);
        const uint64_t k_at       = layer_base + phys * half;
        const uint64_t v_at       = layer_base + plane_run + phys * half;

        answer->dtype = SYS__KIND__VALUE_INT; answer->num_args = 0u; answer->op_code = 0ull;
        answer->args[NN__KV_REF__AT]    = (plane == (uint64_t)NN__CARTRIDGE__PLANE_V) ? v_at : k_at;
        answer->args[NN__KV_REF__TIER]  = t;
        answer->args[NN__KV_REF__PLANE] = plane;
        answer->args[NN__KV_REF__BYTES] = half;
        answer->args[NN__KV_REF__AT_V]  = (plane == (uint64_t)NN__CARTRIDGE__PLANE_BOTH) ? v_at : 0ull;
        answer->args[5]                 = 0ull;
        return true;
    }
    return false;                                          /* that token is not resident anywhere */
}

/* ── THE WORD A PROGRAM SAYS ─────────────────────────────────────────────────────────────────────────
 *
 * `(nn__kv__reference layer token plane)` — three integers in, a `nn__kv_ref` out.
 *
 * ⛔⛔ THE TWO REFUSALS ARE DIFFERENT FACTS AND ARE KEPT APART, which is the thing this tree has been
 * caught collapsing four times in two days:
 *     NO_CACHE   there is no cartridge on this card at all — nobody asked for one in the config
 *     RANGE      there is one, and it does not hold that token or that layer
 * ⇒ ★ A CALLER CAN DO SOMETHING ABOUT THE SECOND AND NOTHING ABOUT THE FIRST, so one word for both would
 * be a diagnosis thrown away at the only place it was available. */
static __device__ __noinline__ void nn__cartridge__zzpackage_apply_reference(sys__heap_node* base,
                                                                            uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 4ull) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY);
        return;
    }
    const sys__heap_node layer = sys__sublist__nth(form, 1ull);
    const sys__heap_node token = sys__sublist__nth(form, 2ull);
    const sys__heap_node plane = sys__sublist__nth(form, 3ull);
    if (layer.dtype != SYS__KIND__VALUE_INT || token.dtype != SYS__KIND__VALUE_INT
        || plane.dtype != SYS__KIND__VALUE_INT) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }
    /* ⛳ THE LAYER TABLE IS WHAT IS ASKED FOR, not the lane count — because it is what the resolver
     * needs, and the two refusals below it are only distinct if this one is exactly "there is no
     * cartridge here". Both rows are written by the same carve, so either would answer; naming the one
     * the next line depends on is what keeps that true if they ever stop being written together. */
    if (nn__cartridge__model_layers() == 0ull) {
        sys__opcodes__fails(form, NN__CARTRIDGE__FAULT_NO_CACHE);
        return;
    }

    sys__heap_node body;
    if (!nn__cartridge__resolve(layer.args[0], token.args[0], plane.args[0], &body)) {
        sys__opcodes__fails(form, NN__CARTRIDGE__FAULT_RANGE);
        return;
    }

    sys__heap_node* made = sys__heap__make(NN__KIND__KV_REF);
    if (made == 0) { sys__opcodes__fails(form, NN__CARTRIDGE__FAULT_ROOM); return; }
    made[1] = body;

    const uint64_t where = sys__heap__offset(made + 1);
    const sys__heap_node ref = sys__heap_object__reference_to(where);
    /* The form takes its own hold and the one `make` left is this verb's, so it goes back — the same two
     * lines every verb in this tree ends with. */
    sys__opcodes__becomes(form, &ref);
    (void)sys__heap_object__release(where);
}

/* ── RELEASE: THERE IS NOTHING TO DO, AND THAT IS THE RULING ─────────────────────────────────────────
 * ⚖ *"the kv cache is not deallocated, at most there will be an opcode that does a rollback and resets
 * the cursors."* A reference is a NAME for bytes the cartridge owns; when its last holder goes, the
 * node's 64 bytes come back to the heap and the span is untouched.
 * ⇒ ⛳ SO THIS KIND NEEDS NO RELEASE HOOK AT ALL, and its OBJECTS row says so by naming the default. What
 * would be wrong is a hook that filed it somewhere, the way a buffer's does — a buffer is a thing lent
 * from a pool, and a kv reference is a sentence about an address. ▶ `language_contract.cuh`. */

#endif /* SILVANN__PACKAGES_NN_CPU_CARTRIDGE__IMPL_CUH */
