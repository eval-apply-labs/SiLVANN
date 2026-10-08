#ifndef SILVANN__PACKAGES_NN_CPU_ROUTED__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_ROUTED__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "routed__header.cuh"
#include "doors.cuh"                          /* nn's doors, launched on the worker's silicon */
#include <stdlib.h>                           /* a chunk's picks, inverted on the host: malloc, free */

/* ══ A MODEL'S ROUTED EXPERTS, AS nn RUNS THEM — ▶ `routed__header.cuh` ══════════════════════════════════════════════ */

static uint64_t nn__routed__top_k(sys__engine__ctx* ctx, const nn__doors* doors, uint64_t picks, uint64_t weights, uint64_t logits,
                                  uint64_t bias, float scale, uint64_t n, uint64_t k, uint64_t* chosen) {
    if (k == 0ull || k > NN__VECTOR__TOP_K_MAX || (bias != 0ull && k > NN__VECTOR__ROUTER_K_MAX) || sys__node_array__length(picks) < k)
        return NN__PRIMITIVES__FAULT_BOUNDS;
    /* ⛔ ONLY VALUE WORDS ARE WRITTEN, AND ONLY INTO ELEMENTS THAT HOLD NO REFERENCE: an int is written over, a null
     *   becomes an int, anything else is refused — so no hold is made or lost behind the array's back */
    sys__heap_node zero = sys__heap_node__nothing();
    zero.dtype = SYS__KIND__VALUE_INT; zero.args[0] = 0ull;
    sys__node_array_walk w;
    if (!sys__node_array__walk(picks, 0ull, &w)) return SYS__OPCODES__FAULT_TYPE;
    for (uint64_t j = 0u; j < k; ++j, sys__node_array__next(&w)) {
        const sys__heap_node* cell = sys__node_array__walk_cell(&w);
        if (cell == 0 || (cell->dtype != SYS__KIND__VALUE_INT && cell->dtype != SYS__KIND__VALUE_NULL)) return SYS__OPCODES__FAULT_TYPE;
        if (!sys__node_array__walk_set(&w, &zero)) return SYS__OPCODES__FAULT_TYPE;
    }
    sys__heap_node* cells = sys__node_array__cells(picks, k);
    if (cells == 0) return SYS__OPCODES__FAULT_TYPE;
    /* the words from one element's value to the next, taken from two elements rather than from a width */
    const uint64_t words = (uint64_t)(&cells[1].args[0] - &cells[0].args[0]);
    if (ctx->heap_card != 0 && cells >= ctx->heap_host && (uint64_t)(cells - ctx->heap_host) + k <= ctx->heap_nodes) {
        sys__heap_node* card = ctx->heap_card + (cells - ctx->heap_host);
        if (bias != 0ull)
            doors->vector_top_k_biased(&card[0].args[0], words, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits,
                                       (const uint16_t*)(uintptr_t)bias, n, k, scale);
        else
            doors->vector_top_k(&card[0].args[0], words, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits, n, k);
        if (!sys__gpu__compute_completed(ctx->family)) return NN__PRIMITIVES__FAULT_NO_DEVICE;
        if (chosen != 0) for (uint64_t j = 0u; j < k; ++j) chosen[j] = cells[j].args[0];
        return 0u;
    }
    void* landing = 0;
    uint64_t got[NN__VECTOR__TOP_K_MAX];
    if (!sys__gpu__memory_allocate(ctx->family, &landing, k * sizeof(uint64_t))) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    if (bias != 0ull)
        doors->vector_top_k_biased((uint64_t*)landing, 1ull, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits,
                                   (const uint16_t*)(uintptr_t)bias, n, k, scale);
    else
        doors->vector_top_k((uint64_t*)landing, 1ull, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits, n, k);
    const bool read = sys__gpu__memory_read(ctx->family, got, landing, k * sizeof(uint64_t));
    sys__gpu__memory_free(ctx->family, landing);
    if (!read) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    for (uint64_t j = 0u; j < k; ++j) { cells[j].args[0] = got[j]; if (chosen != 0) chosen[j] = got[j]; }
    return 0u;
}

/* The activation of `pairs` gate-and-up pairs: SwiGLU, clamped where the model's is. */
static void nn__routed__zzprivate_activate(const nn__doors* doors, unsigned int* over, const nn__routed* r, uint64_t act, uint64_t gu,
                                           uint64_t pairs) {
    if (r->limit != 0.0f)
        doors->swiglu_clamped((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, pairs, r->inter, r->limit, over);
    else
        doors->swiglu_pairs((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, pairs, r->inter, over);
}

/* The gate-and-up rows of `n` picks — the one in the slot at `slot[i]` into place `at[i]` of `gu` — over the hand's `x`. */
static void nn__routed__zzprivate_up(const nn__doors* doors, unsigned int* over, const nn__routed* r, uint64_t x, uint64_t gu,
                                     const uint64_t* slot, const uint64_t* at, uint64_t n) {
    nn__turboquant__groups up = {};
    for (uint64_t i = 0u; i < n; ++i) {
        up.codes[i] = slot[i]; up.luts[i] = slot[i] + r->up_lut;
        up.rows[i] = 2u * r->inter; up.d[i] = r->bits; up.out_at[i] = at[i] * 2u * r->inter;
    }
    up.count = n;
    if (r->int8) doors->turboquant_gemv_groups_int8((uint16_t*)(uintptr_t)gu, up, (const uint16_t*)(uintptr_t)x, r->hidden, over);
    else         doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)gu, up, (const uint16_t*)(uintptr_t)x, r->hidden, over);
}

/* `routed = zero + Σ w · down(rotate(act(gu)))` over the `n` picks whose gate-and-up rows are `gu`'s first `n` — pick
 * `which[i]` in the slot at `slot[i]`, its weight the hand's `which[i]`-th — summed in that order. */
static void nn__routed__zzprivate_down(const nn__doors* doors, unsigned int* over, const nn__routed* r, uint64_t hand, uint64_t routed,
                                       uint64_t gu, uint64_t act, uint64_t actr, const uint64_t* slot, const uint64_t* which, uint64_t n) {
    const uint64_t H = r->hidden, I = r->inter;
    nn__turboquant__groups down = {};
    for (uint64_t i = 0u; i < n; ++i) {
        down.codes[i] = slot[i] + r->down; down.luts[i] = slot[i] + r->down_lut;
        down.rows[i] = H; down.d[i] = r->bits; down.out_at[i] = which[i];   /* its pick's weight */
    }
    down.count = n;
    nn__routed__zzprivate_activate(doors, over, r, act, gu, n);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)r->signs, n * I, over);
    if (r->int8)
        doors->turboquant_gemv_groups_sum_int8((uint16_t*)(uintptr_t)routed, down, (const uint16_t*)(uintptr_t)actr,
                                               (const uint16_t*)(uintptr_t)(hand + 2u * H), (const uint16_t*)(uintptr_t)r->zero, H, I, over);
    else
        doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)routed, down, (const uint16_t*)(uintptr_t)actr,
                                          (const uint16_t*)(uintptr_t)(hand + 2u * H), (const uint16_t*)(uintptr_t)r->zero, H, I, over);
}

/* An expert's slot, where its index says it is: 0 when it has none. */
static uint64_t nn__routed__zzprivate_slot(const nn__routed* r, uint64_t expert) {
    sys__heap_node me;
    return nn__expert__slot(r->layer, r->type, expert, &me) ? me.args[NN__EXPERT__SLOT_AT] : 0ull;
}

static uint64_t nn__routed__position(sys__engine__ctx* ctx, const nn__doors* doors, const nn__routed* r, uint64_t hand,
                                     uint64_t picks, const nn__expert_tier* tier, uint64_t routed, nn__routed__done* done) {
    const uint64_t H = r->hidden, I = r->inter, E = r->experts, K = r->top_k, first = r->first, count = r->count;
    const uint64_t layer = r->layer, type = r->type;
    if (H == 0u || I == 0u || K == 0u || K > E || K > NN__TURBOQUANT__GROUPS_MAX || H > (1ull << 24) || I > (1ull << 24)
     || count == 0u || first >= E || count > E - first || sys__node_array__length(picks) < K
     || (tier != 0 && (r->backing.file != 0ull || tier->top_k != K)))
        return NN__ROUTED__FAULT_SHAPE;
    if (r->scratch_room < 8u * K * I) return NN__ROUTED__FAULT_ROOM;
    unsigned int* over = ctx->fault_word;
    /* every pick read once; the ones this worker holds, as its collection numbers them */
    uint64_t ids[NN__TURBOQUANT__GROUPS_MAX], held[NN__TURBOQUANT__GROUPS_MAX], n_held = 0u;
    sys__node_array_walk pw;
    if (!sys__node_array__walk(picks, 0ull, &pw)) return NN__ROUTED__FAULT_EXPERT;
    for (uint64_t j = 0u; j < K; ++j, sys__node_array__next(&pw)) {
        const sys__heap_node* p = sys__node_array__walk_cell(&pw);
        /* a pick numbered `experts` or past it is one a card's tier holds */
        if (p == 0 || p->dtype != SYS__KIND__VALUE_INT || p->args[0] >= 2u * E) return NN__ROUTED__FAULT_EXPERT;
        ids[j] = p->args[0];
        if (ids[j] >= first && ids[j] < first + count) held[n_held++] = ids[j] - first;
    }
    /* a tier's visit: which picks it holds, and the promotions of those it does not */
    bool on_card[NN__TURBOQUANT__GROUPS_MAX] = {};
    if (tier != 0 && !nn__expert_tier__visit(ctx->family, tier, ids, on_card)) return NN__ROUTED__FAULT_EXPERT;
    /* ① the picks this worker holds: the ones in memory now, and the others on their way — read into a slot where the
     *   slots are a cache, their pages started where the slots are a mapped file */
    uint64_t slot[NN__TURBOQUANT__GROUPS_MAX], which[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t away_slot[NN__TURBOQUANT__GROUPS_MAX], away[NN__TURBOQUANT__GROUPS_MAX], away_id[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t ready = 0u, n_away = 0u;
    for (uint64_t j = 0u; j < K; ++j) {
        const uint64_t id = ids[j];
        if (id < first || id >= first + count) continue;            /* another worker's, or held already */
        if (r->backing.file != 0ull) {
            uint64_t at = 0ull;
            const int got = nn__expert__request(ctx->family, layer, type, id - first, &r->backing, false, held, (unsigned)n_held, &at);
            if (got == NN__EXPERT__REFUSED) return NN__ROUTED__FAULT_EXPERT;
            if (got == NN__EXPERT__RESIDENT && nn__expert__in_memory(ctx->family, type, at)) { slot[ready] = at; which[ready] = j; ++ready; }
            else { away_id[n_away] = id - first; away[n_away] = j; ++n_away; }
            continue;
        }
        if (tier != 0) {
            /* a tier: the picks it holds are this worker's, marked held for the others; the rest are left to them */
            if (!on_card[j]) continue;
            const uint64_t at = nn__routed__zzprivate_slot(r, id - first);
            if (at == 0ull) return NN__ROUTED__FAULT_EXPERT;
            sys__heap_node mark = sys__heap_node__nothing();
            mark.dtype = SYS__KIND__VALUE_INT; mark.num_args = 0u; mark.op_code = 0ull; mark.args[0] = id + E;
            if (!sys__node_array__set(picks, j, &mark)) return NN__ROUTED__FAULT_EXPERT;
            slot[ready] = at; which[ready] = j; ++ready;
            continue;
        }
        /* an expert an exclusive tier just sent back from the card is in its slot once its write has landed */
        nn__expert__wait_written(layer, type, id - first);
        const uint64_t at = nn__routed__zzprivate_slot(r, id - first);
        if (at == 0ull) return NN__ROUTED__FAULT_EXPERT;
        if (nn__expert__in_memory(ctx->family, type, at)) { slot[ready] = at; which[ready] = j; ++ready; }
        else { away_slot[n_away] = at; away_id[n_away] = id - first; away[n_away] = j; ++n_away; }
    }
    const uint64_t n = ready + n_away;
    const uint64_t gu = r->scratch, act = gu + 4u * K * I, actr = act + 2u * K * I;
    if (done != 0) {
        for (uint64_t j = 0u; j < K; ++j) done->ids[j] = ids[j];
        done->n = n; done->actr = actr;
    }
    /* ② the gate-and-up of the ones in memory while the rest arrive; ③ the layer's reads settled — the picks' and any a
     *   prediction queued — and the rest's; then every pick's down summed in pick order. A pick's place is its rank
     *   among this worker's picks. */
    uint64_t at_ready[NN__TURBOQUANT__GROUPS_MAX], at_away[NN__TURBOQUANT__GROUPS_MAX];
    for (uint64_t a = 0u, b = 0u; a < ready || b < n_away; ) {
        if (b == n_away || (a < ready && which[a] < away[b])) { at_ready[a] = a + b; ++a; }
        else { at_away[b] = a + b; ++b; }
    }
    uint64_t all_slot[NN__TURBOQUANT__GROUPS_MAX], all_which[NN__TURBOQUANT__GROUPS_MAX];
    if (ready != 0u) nn__routed__zzprivate_up(doors, over, r, hand, gu, slot, at_ready, ready);
    for (uint64_t k = 0u; k < ready; ++k) { all_slot[at_ready[k]] = slot[k]; all_which[at_ready[k]] = which[k]; }
    if (r->backing.file != 0ull && !nn__expert__settle(layer, type)) return NN__ROUTED__FAULT_EXPERT;
    if (n_away != 0u) {
        for (uint64_t k = 0u; k < n_away; ++k) {
            if (r->backing.file != 0ull) {
                away_slot[k] = nn__routed__zzprivate_slot(r, away_id[k]);
                if (away_slot[k] == 0ull) return NN__ROUTED__FAULT_EXPERT;
            }
            all_slot[at_away[k]] = away_slot[k]; all_which[at_away[k]] = away[k];
        }
        nn__routed__zzprivate_up(doors, over, r, hand, gu, away_slot, at_away, n_away);
    }
    if (n == 0u) {                                                   /* every pick another worker's: a sum of nothing */
        doors->vector_add((uint16_t*)(uintptr_t)routed, (const uint16_t*)(uintptr_t)r->zero, (const uint16_t*)(uintptr_t)r->zero, H, over);
        return 0u;
    }
    nn__routed__zzprivate_down(doors, over, r, hand, routed, gu, act, actr, all_slot, all_which, n);
    if (done != 0) for (uint64_t k = 0u; k < n; ++k) done->which[k] = all_which[k];
    return 0u;
}

/* ⛳ SCRATCH, in bytes: xs (n·H halves) · pairs (16·P) · the weights (P halves) · gu (P·2I halves) · act · actr (P·I
 *   each) · the sums (n·K·H floats), P the picks this worker holds — each part on 8 bytes. */
static uint64_t nn__routed__rows(sys__engine__ctx* ctx, const nn__doors* doors, const nn__routed* r, uint64_t hands, uint64_t n,
                                 const nn__expert_tier* tier, uint64_t routed, nn__routed__rows_mask mask, void* model) {
    const uint64_t H = r->hidden, I = r->inter, E = r->experts, K = r->top_k, first = r->first, count = r->count;
    const uint64_t layer = r->layer, type = r->type, D = r->bits;
    const bool holds = tier != 0;
    if (H == 0u || I == 0u || K == 0u || K > E || K > NN__VECTOR__TOP_K_MAX || H > (1ull << 24) || I > (1ull << 24)
     || count == 0u || first >= E || count > E - first || n == 0u || r->hand_bytes < 2u * H
     || r->picks_at + 8u * K > r->hand_bytes || (holds && r->backing.file != 0ull)
     || (r->backing.file != 0ull && (r->batch == 0u || r->batch > NN__ROUTED__BATCH_MAX)))
        return NN__ROUTED__FAULT_SHAPE;
    unsigned int* over = ctx->fault_word;
    /* ① every row's picks, read; the ones this worker holds inverted — each expert used, and its picks */
    uint64_t chosen[NN__VECTOR__TOP_K_MAX];
    uint16_t weight[NN__VECTOR__TOP_K_MAX];
    uint32_t* rows_of = (uint32_t*)malloc(sizeof(uint32_t) * n * K);       /* a held pick's row, in listing order */
    uint32_t* slot_of = (uint32_t*)malloc(sizeof(uint32_t) * n * K);       /* its row's pick: row · K + j, its sum's slot */
    uint16_t* w_of = (uint16_t*)malloc(sizeof(uint16_t) * n * K);          /* its weight */
    uint32_t* expert_of = (uint32_t*)malloc(sizeof(uint32_t) * n * K);     /* its expert, less `first` */
    uint32_t* picked = (uint32_t*)calloc(count, sizeof(uint32_t));        /* each expert's picks */
    uint32_t* begin = (uint32_t*)calloc(count, sizeof(uint32_t));         /* where they begin, in `order` */
    uint32_t* fill = (uint32_t*)calloc(count, sizeof(uint32_t));
    uint32_t* order = (uint32_t*)malloc(sizeof(uint32_t) * 2u * count);    /* the used ones, in the order they run */
    uint32_t* pairs = 0;
    uint32_t* laid = 0;                                                     /* each pick's expert, as laid out */
    uint16_t* wj = 0;
    uint64_t fault = 0u, P = 0u;
    if (rows_of == 0 || slot_of == 0 || w_of == 0 || expert_of == 0 || picked == 0 || begin == 0 || fill == 0 || order == 0)
        fault = NN__ROUTED__FAULT_ROOM;
    for (uint64_t t = 0u; t < n && fault == 0u; ++t) {
        const uint64_t hand = hands + t * r->hand_bytes;
        if (!sys__gpu__memory_read(ctx->family, chosen, (const void*)(uintptr_t)(hand + r->picks_at), 8u * K)
         || !sys__gpu__memory_read(ctx->family, weight, (const void*)(uintptr_t)(hand + 2u * H), 2u * K)) { fault = NN__PRIMITIVES__FAULT_NO_DEVICE; break; }
        for (uint64_t j = 0u; j < K; ++j) {
            if (chosen[j] >= 2u * E) { fault = NN__ROUTED__FAULT_EXPERT; break; }
            const bool marked = chosen[j] >= E;                                    /* held by a card's tier */
            if (marked != holds) continue;
            const uint64_t id = marked ? chosen[j] - E : chosen[j];
            if (id < first || id >= first + count) continue;                       /* another worker's */
            rows_of[P] = (uint32_t)t; slot_of[P] = (uint32_t)(t * K + j); w_of[P] = weight[j]; expert_of[P] = (uint32_t)(id - first);
            ++picked[expert_of[P]];
            ++P;
        }
    }
    /* the used experts in the order they run, their picks laid out in that order. Where the slots are a cache, or a tier's,
     * from the least picked to the most — so the ones this chunk used most are its most recent when it ends, the ones a
     * token will want next (`MEASURED` on the 35B: in number order the answer after a prompt of rows hit 83%, where a
     * prompt read a position at a time left 94%). Where they are a mapped file, the ones in memory first and the others'
     * pages started on their way, so the chunk computes with what is here while the rest arrive */
    uint64_t used = 0u, n_away = 0u;
    for (uint64_t e = 0u; e < count && fault == 0u; ++e) {
        if (picked[e] == 0u) continue;
        if (r->backing.file != 0ull || holds) {
            uint64_t k = used++;
            while (k > 0u && picked[order[k - 1u]] > picked[e]) { order[k] = order[k - 1u]; --k; }
            order[k] = (uint32_t)e;
            continue;
        }
        const uint64_t at = nn__routed__zzprivate_slot(r, e);
        if (at == 0ull) { fault = NN__ROUTED__FAULT_EXPERT; break; }
        if (nn__expert__in_memory(ctx->family, type, at)) order[used++] = (uint32_t)e;
        else order[count + n_away++] = (uint32_t)e;
    }
    for (uint64_t k = 0u; k < n_away; ++k) order[used++] = order[count + k];
    for (uint64_t o = 0u, at = 0u; o < used; ++o) { begin[order[o]] = (uint32_t)at; at += picked[order[o]]; }
    const uint64_t xs = r->scratch, pairs_at = xs + ((2u * n * H + 7u) & ~7ull), wj_at = pairs_at + 16u * P;
    const uint64_t gu = (wj_at + 2u * P + 7u) & ~7ull, act = gu + 4u * P * I, actr = act + 2u * P * I, acc = (actr + 2u * P * I + 7u) & ~7ull;
    if (fault == 0u && acc + 4u * n * K * H - xs > r->scratch_room) fault = NN__ROUTED__FAULT_ROOM;
    if (fault == 0u) {
        pairs = (uint32_t*)malloc(sizeof(uint32_t) * 4u * (P + 1u));
        laid = (uint32_t*)malloc(sizeof(uint32_t) * (P + 1u));
        wj = (uint16_t*)malloc(sizeof(uint16_t) * (P + 1u));
        if (pairs == 0 || laid == 0 || wj == 0) fault = NN__ROUTED__FAULT_ROOM;
    }
    if (fault == 0u) {
        /* the picks in expert order: up pairs (row -> j), then down pairs (j -> its sum's slot), and the weights */
        uint32_t* up = pairs;
        uint32_t* down = pairs + 2u * P;
        for (uint64_t p = 0u; p < P; ++p) {
            const uint64_t j = begin[expert_of[p]] + fill[expert_of[p]]++;
            up[2u * j] = rows_of[p];   up[2u * j + 1u] = (uint32_t)j;
            down[2u * j] = (uint32_t)j; down[2u * j + 1u] = slot_of[p];
            wj[j] = w_of[p]; laid[j] = expert_of[p];
        }
        for (uint64_t t = 0u; t < n; ++t)                    /* the rows' inputs, contiguous */
            doors->vector_copy((uint16_t*)(uintptr_t)(xs + 2u * t * H), (const uint16_t*)(uintptr_t)(hands + t * r->hand_bytes), H);
        if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)pairs_at, pairs, 16u * P)
         || !sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)wj_at, wj, 2u * P)
         || !sys__gpu__memory_zerofill(ctx->family, (void*)(uintptr_t)acc, 4u * n * K * H))
            fault = NN__PRIMITIVES__FAULT_NO_DEVICE;
    }
    /* ② batch by batch: each expert's gate-and-up over its rows, twelve experts a launch; the activations and rotations
     *   over the batch's picks; each expert's down weighted into its picks' sums. A tier computes four experts a batch,
     *   its first one alone, so the card begins after one copy has landed and a batch runs while the next ones land */
    const uint64_t batch = r->backing.file != 0ull ? r->batch : holds ? 4u : used;
    uint64_t pinned[2u * NN__ROUTED__BATCH_MAX];
    for (uint64_t b0 = 0u, b_end = holds ? 1u : batch; b0 < used && fault == 0u && P > 0u; b0 = b_end, b_end = b0 + batch) {
        const uint64_t b1 = b_end < used ? b_end : used;
        if (holds)                                                   /* a copy still on its way, waited for here */
            for (uint64_t o = b0; o < b1 && fault == 0u; ++o)
                if (!nn__expert__settle_one(layer, type, order[o])) fault = NN__ROUTED__FAULT_EXPERT;
        if (r->backing.file != 0ull) {
            /* this batch's reads, if not asked for already, and the next batch's, both kept from being given up */
            const uint64_t n1 = b1 + batch < used ? b1 + batch : used;
            unsigned np = 0u;
            for (uint64_t o = b0; o < n1; ++o) pinned[np++] = order[o];
            for (uint64_t o = b0; o < n1 && fault == 0u; ++o)
                if (nn__expert__request(ctx->family, layer, type, order[o], &r->backing, false, pinned, np, 0) == NN__EXPERT__REFUSED)
                    fault = NN__ROUTED__FAULT_EXPERT;
            for (uint64_t o = b0; o < b1 && fault == 0u; ++o)
                if (!nn__expert__settle_one(layer, type, order[o])) fault = NN__ROUTED__FAULT_EXPERT;
        }
        for (uint64_t o = b0; o < b1 && fault == 0u; ) {
            nn__expert__groups g = {};
            uint64_t k = 0u;
            for (; o < b1 && k < NN__EXPERT__GROUPS_MAX; ++o) {
                const uint64_t e0 = order[o];
                if (!holds) nn__expert__wait_written(layer, type, e0);    /* ▶ `position`: a write still landing */
                const uint64_t at = nn__routed__zzprivate_slot(r, e0);
                if (at == 0ull) { fault = NN__ROUTED__FAULT_EXPERT; break; }
                g.codes[k] = at; g.luts[k] = at + r->up_lut; g.out_rows[k] = 2u * I; g.d[k] = D;
                g.out_at[k] = 0u; g.pairs_at[k] = begin[e0]; g.pairs[k] = picked[e0];
                ++k;
            }
            g.count = k;
            if (k > 0u && fault == 0u) {
                if (r->int8) doors->expert_groups_int8((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)xs, (const uint32_t*)(uintptr_t)pairs_at, H, over);
                else         doors->expert_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)xs, (const uint32_t*)(uintptr_t)pairs_at, H, over);
            }
        }
        if (fault != 0u) break;
        /* the batch's picks: every pick where it is one batch of all, else its experts' run of them, in order */
        const uint64_t p0 = begin[order[b0]], p1 = begin[order[b1 - 1u]] + picked[order[b1 - 1u]];
        nn__routed__zzprivate_activate(doors, over, r, act + 2u * p0 * I, gu + 4u * p0 * I, p1 - p0);
        doors->hadamard_rotate((uint16_t*)(uintptr_t)(actr + 2u * p0 * I), (const uint16_t*)(uintptr_t)(act + 2u * p0 * I),
                               (const uint16_t*)(uintptr_t)r->signs, (p1 - p0) * I, over);
        for (uint64_t o = b0; o < b1 && fault == 0u; ++o) {
            const uint64_t e = order[o];
            const uint64_t at = nn__routed__zzprivate_slot(r, e);
            if (at == 0ull) { fault = NN__ROUTED__FAULT_EXPERT; break; }
            (r->int8 ? doors->expert_rows_sum_int8 : doors->expert_rows_sum)((float*)(uintptr_t)acc, (const uint8_t*)(uintptr_t)(at + r->down),
                                   r->down_lut - r->down, (const uint8_t*)(uintptr_t)(at + r->down_lut), 2u * H,
                                   (const uint16_t*)(uintptr_t)actr, (const uint32_t*)(uintptr_t)(pairs_at + 8u * (P + begin[e])),
                                   (const uint16_t*)(uintptr_t)wj_at, picked[e], D, H, I);
        }
    }
    /* ③ a model's own addition to the picks' sums, then the rows: each its picks' sums added in pick order */
    if (fault == 0u && mask != 0 && P != 0u) fault = mask(model, ctx, P, laid, actr, wj_at, pairs_at + 8u * P, acc);
    if (fault == 0u) {
        if (!sys__gpu__memory_zerofill(ctx->family, (void*)(uintptr_t)routed, 2u * n * H)) fault = NN__PRIMITIVES__FAULT_NO_DEVICE;
        else doors->expert_rows_reduce((uint16_t*)(uintptr_t)routed, (float*)(uintptr_t)acc, (const uint16_t*)(uintptr_t)routed, n, K, H, over);
    }
    free(rows_of); free(slot_of); free(w_of); free(expert_of); free(picked); free(begin); free(fill); free(order);
    free(pairs); free(laid); free(wj);
    return fault;
}

#endif /* SILVANN__PACKAGES_NN_CPU_ROUTED__IMPL_CUH */
