#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_PAD_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_PAD_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include <string.h>                                 /* the transfers' copies into and out of the chunks */
#include "../../../../operating_systems/os.cuh"     /* the pads' lock and the condition a freed chunk wakes */
#include "../../contracts/objects/pad.cuh"          /* what a pad is: its holds, its tickets, its fault words */
#include "silicon__header.cuh"                      /* the family's doors the pad is moved through */

/* ══ ⭐⭐ A CARD'S LANDING PAD — ▶ `contracts/objects/pad.cuh` for what it is ═══════════════════════════════════════════
 * A pad is opened on its card's own worker, whose device is bound, and is used from any thread after: its channels and
 * marks are its card's, and a copy queued on them runs there. Every pad shares one lock — a claim, a send or a release
 * holds it for a few calls that only queue work — and the threads that wait for a chunk wait on one condition. */
typedef struct sys__pad__chunk {
    uint8_t* host;
    unsigned holds;         /* regions claimed and not yet sent (up) or released (down) */
    bool     full;          /* its way moved on: no more claims until it is filled again */
    uint64_t used;          /* bytes claimed since it was last filled from the start */
    uint64_t last;          /* the last copy queued from it, by number: it is free once that is reached */
} sys__pad__chunk;

typedef struct sys__pad__way {
    void*    channel;
    unsigned first, count;  /* its chunks */
    unsigned open;          /* the chunk claims go to */
    void*    mark[SYS__PAD__MARKS];
    uint64_t marked[SYS__PAD__MARKS];   /* the copy each mark was last recorded after */
    uint64_t queued;        /* copies queued, the last one's number */
    uint64_t reached;       /* the highest known reached */
} sys__pad__way;

typedef struct sys__pad {
    bool     open;
    sys__silicon_family__id family;
    unsigned device;
    unsigned chunks;
    uint64_t chunk_bytes;
    void*    base;
    void*    base_card;
    bool     marks;         /* the family has marks; without, a copy is done when its call returns */
    void*    compute[2];    /* two marks on the card's compute queue, recorded by `mark_compute` */
    sys__pad__chunk chunk[SYS__PAD__CHUNKS_MAX];
    sys__pad__way   way[2];
} sys__pad;

static sys__pad           sys__pad__zzprivate_pads[SYS__PAD__PADS_MAX];
static sys__os__lock      sys__pad__zzprivate_lock  = SYS__OS__LOCK_INIT;
static sys__os__condition sys__pad__zzprivate_freed = SYS__OS__CONDITION_INIT;   /* a hold given back, a copy reached */

/* The open pad of a card, or 0. */
static inline sys__pad* sys__pad__of(sys__silicon_family__id family, unsigned device) {
    sys__pad* found = 0;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    for (unsigned i = 0u; i < SYS__PAD__PADS_MAX && found == 0; ++i) {
        sys__pad* p = &sys__pad__zzprivate_pads[i];
        if (p->open && p->family == family && p->device == device) found = p;
    }
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return found;
}

/* Where a hold's bytes are, in RAM. */
static inline uint8_t* sys__pad__at(const sys__pad* p, const sys__pad__hold* h) {
    return p->chunk[h->chunk].host + h->offset;
}

/* Everything a pad took, given back; the pad's copies all reached first. Lock not held. */
static void sys__pad__zzprivate_drop(sys__pad* p) {
    for (unsigned c = 0u; c < 2u; ++c) {
        if (p->compute[c] == 0) continue;
        (void)sys__gpu__mark_reached(p->family, p->compute[c]);
        sys__gpu__mark_close(p->family, p->compute[c]);
        p->compute[c] = 0;
    }
    for (unsigned w = 0u; w < 2u; ++w) {
        sys__pad__way* y = &p->way[w];
        for (unsigned m = 0u; m < SYS__PAD__MARKS; ++m) {
            if (y->mark[m] == 0) continue;
            if (p->marks) (void)sys__gpu__mark_reached(p->family, y->mark[m]);
            sys__gpu__mark_close(p->family, y->mark[m]);
            y->mark[m] = 0;
        }
        if (y->channel != 0) sys__gpu__side_close(p->family, y->channel);
        y->channel = 0;
    }
    if (p->base != 0) sys__gpu__memory_free_host_ram(p->family, p->base);
    const sys__pad none = {};
    *p = none;
}

/* A way's copy `number` reached, waited for on this thread: false where the family failed. Lock not held. */
static inline bool sys__pad__zzprivate_reach(sys__pad* p, unsigned way, uint64_t number) {
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    sys__pad__way* y = &p->way[way];
    if (!p->marks || number <= y->reached) { sys__os__lock_give(&sys__pad__zzprivate_lock); return true; }
    /* the mark recorded after `number`, or after a later copy that reused it — whose reaching is `number`'s too */
    const unsigned m = (unsigned)(number % SYS__PAD__MARKS);
    void* const mark = y->mark[m];
    const uint64_t after = y->marked[m];
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    const bool ok = sys__gpu__mark_reached(p->family, mark);
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    if (ok && after > y->reached) { y->reached = after; sys__os__condition_wake_all(&sys__pad__zzprivate_freed); }
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return ok;
}

/* ⭐ A CARD'S PAD OPENED — `up` chunks up and `down` down, each `chunk_bytes`, pinned at once — on the card's own worker.
 * The same shape again is the pad already there; another shape replaces it once nothing holds it. */
static bool sys__pad__open(sys__silicon_family__id family, unsigned device, unsigned up, unsigned down, uint64_t chunk_bytes) {
    const unsigned chunks = up + down;
    if (up == 0u || down == 0u || chunks > SYS__PAD__CHUNKS_MAX || chunk_bytes == 0ull || (chunk_bytes & 4095ull) != 0ull)
        return false;
    sys__pad* p = sys__pad__of(family, device);
    if (p != 0 && p->way[SYS__PAD__UP].count == up && p->way[SYS__PAD__DOWN].count == down && p->chunk_bytes == chunk_bytes)
        return true;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    if (p != 0) {
        for (unsigned c = 0u; c < p->chunks; ++c)
            if (p->chunk[c].holds != 0u) { sys__os__lock_give(&sys__pad__zzprivate_lock); return false; }
        p->open = false;
    } else {
        for (unsigned i = 0u; i < SYS__PAD__PADS_MAX && p == 0; ++i)
            if (!sys__pad__zzprivate_pads[i].open && sys__pad__zzprivate_pads[i].base == 0) p = &sys__pad__zzprivate_pads[i];
    }
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    if (p == 0) return false;
    if (p->base != 0) sys__pad__zzprivate_drop(p);
    p->family = family; p->device = device; p->chunks = chunks; p->chunk_bytes = chunk_bytes;
    bool ok = sys__gpu__memory_allocate_host_ram_mapped(family, &p->base, &p->base_card, (size_t)(chunks * chunk_bytes));
    for (unsigned w = 0u; w < 2u && ok; ++w) ok = sys__gpu__side_open(family, &p->way[w].channel);
    /* marks where the family has them: the first refused says it has none */
    p->marks = ok && sys__gpu__mark_open(family, &p->way[0].mark[0]);
    for (unsigned w = 0u; w < 2u && ok && p->marks; ++w)
        for (unsigned m = (w == 0u ? 1u : 0u); m < SYS__PAD__MARKS && ok; ++m) ok = sys__gpu__mark_open(family, &p->way[w].mark[m]);
    for (unsigned c = 0u; c < 2u && ok && p->marks; ++c) ok = sys__gpu__mark_open(family, &p->compute[c]);
    if (!ok) { sys__pad__zzprivate_drop(p); return false; }
    p->way[SYS__PAD__UP].first = 0u;    p->way[SYS__PAD__UP].count = up;
    p->way[SYS__PAD__DOWN].first = up;  p->way[SYS__PAD__DOWN].count = down;
    for (unsigned w = 0u; w < 2u; ++w) p->way[w].open = p->way[w].first;
    for (unsigned c = 0u; c < chunks; ++c) p->chunk[c].host = (uint8_t*)p->base + c * chunk_bytes;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    p->open = true;
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return true;
}

/* How many chunks a way of a pad has. */
static inline unsigned sys__pad__chunks(const sys__pad* p, unsigned way) {
    return p == 0 || way > SYS__PAD__DOWN ? 0u : p->way[way].count;
}

/* A card's pad closed, its copies all reached. */
static void sys__pad__close(sys__silicon_family__id family, unsigned device) {
    sys__pad* p = sys__pad__of(family, device);
    if (p == 0) return;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    p->open = false;
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    sys__pad__zzprivate_drop(p);
}

/* ⭐ A REGION CLAIMED: `bytes` of the way's open chunk — or, where they do not fit, of the next chunk in its ring with no
 * region out, once the last copy queued from it is reached (the double buffering: while the card empties one chunk, the
 * next is filled; a parked chunk given back out of turn is taken as soon as it is free). Answers where the bytes go, or 0
 * where they are more than a chunk — or, `wait` false, where every other chunk still has regions out: a thread that holds
 * down regions of its own reads one back and releases it rather than wait, or it could wait on itself. A reach it waits
 * for always comes. */
static uint8_t* sys__pad__zzprivate_claim(sys__pad* p, unsigned way, uint64_t bytes, sys__pad__hold* h, bool wait) {
    if (p == 0 || way > SYS__PAD__DOWN || bytes == 0ull || bytes > p->chunk_bytes) return 0;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    for (;;) {
        sys__pad__way* y = &p->way[way];
        sys__pad__chunk* c = &p->chunk[y->open];
        if (!c->full && c->used + bytes <= p->chunk_bytes) {
            h->way = way; h->chunk = y->open; h->offset = c->used; h->bytes = bytes;
            c->used += bytes;
            ++c->holds;
            sys__os__lock_give(&sys__pad__zzprivate_lock);
            return c->host + h->offset;
        }
        c->full = true;
        unsigned next = y->open;
        sys__pad__chunk* n = 0;
        for (unsigned k = 1u; k <= y->count && n == 0; ++k) {
            const unsigned i = y->first + (y->open - y->first + k) % y->count;
            if (p->chunk[i].holds == 0u) { next = i; n = &p->chunk[i]; }
        }
        if (n == 0) {
            if (!wait) { sys__os__lock_give(&sys__pad__zzprivate_lock); return 0; }
            sys__os__condition_wait(&sys__pad__zzprivate_freed, &sys__pad__zzprivate_lock);
            continue;
        }
        if (p->marks && n->last > y->reached) {
            const uint64_t last = n->last;
            sys__os__lock_give(&sys__pad__zzprivate_lock);
            const bool ok = sys__pad__zzprivate_reach(p, way, last);
            sys__os__lock_take(&sys__pad__zzprivate_lock);
            if (!ok) { sys__os__lock_give(&sys__pad__zzprivate_lock); return 0; }
            continue;
        }
        if (y->open != next || n->full) { n->full = false; n->used = 0ull; y->open = next; }
    }
}
static inline uint8_t* sys__pad__claim(sys__pad* p, unsigned way, uint64_t bytes, sys__pad__hold* h) {
    return sys__pad__zzprivate_claim(p, way, bytes, h, true);
}
/* A claim that does not wait for a chunk to be given back: 0 where every chunk but the open one still has regions out. */
static inline uint8_t* sys__pad__try_claim(sys__pad* p, unsigned way, uint64_t bytes, sys__pad__hold* h) {
    return sys__pad__zzprivate_claim(p, way, bytes, h, false);
}

/* The copy just queued on way `way`'s channel numbered, and a mark recorded after it: its ticket. Lock held. */
static bool sys__pad__zzprivate_number(sys__pad* p, unsigned way, sys__pad__ticket* t) {
    sys__pad__way* y = &p->way[way];
    const uint64_t number = ++y->queued;
    bool ok = true;
    if (p->marks) {
        const unsigned m = (unsigned)(number % SYS__PAD__MARKS);
        ok = sys__gpu__side_mark_record(p->family, y->mark[m], y->channel);
        y->marked[m] = number;
    } else {
        y->reached = number;
    }
    t->way = way; t->number = number;
    return ok;
}

/* A copy of a hold queued on its way's channel, with a mark after it — under the lock, so the channel's order is the
 * tickets' order. `up`: the hold's bytes to `card`; else `card`'s bytes into the hold. Lock held. */
static bool sys__pad__zzprivate_queue(sys__pad* p, const sys__pad__hold* h, void* card, sys__pad__ticket* t) {
    sys__pad__way* y = &p->way[h->way];
    sys__pad__chunk* c = &p->chunk[h->chunk];
    const bool queued = h->way == SYS__PAD__UP
            ? sys__gpu__side_memory_write_queued(p->family, card, c->host + h->offset, (size_t)h->bytes, y->channel)
            : sys__gpu__side_memory_read_queued(p->family, c->host + h->offset, card, (size_t)h->bytes, y->channel);
    const bool ok = sys__pad__zzprivate_number(p, h->way, t) && queued;
    c->last = t->number;
    return ok;
}

/* ⭐ AN UP HOLD SENT: its bytes, filled, queued to the card at `card_to` — the hold given back, the bytes the card's once
 * the ticket is reached. */
static bool sys__pad__send(sys__pad* p, const sys__pad__hold* h, void* card_to, sys__pad__ticket* t) {
    if (p == 0 || h->way != SYS__PAD__UP) return false;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    const bool ok = sys__pad__zzprivate_queue(p, h, card_to, t);
    --p->chunk[h->chunk].holds;
    sys__os__condition_wake_all(&sys__pad__zzprivate_freed);
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return ok;
}

/* ⭐ BYTES SENT FROM OUTSIDE THE PAD: `bytes` at `host_from` — RAM pinned already, registered with the card — queued to
 * the card at `card_to` on the up way's channel, in its order and under its marks, with no chunk claimed. */
static bool sys__pad__send_from(sys__pad* p, const void* host_from, void* card_to, uint64_t bytes, sys__pad__ticket* t) {
    if (p == 0 || bytes == 0ull) return false;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    const bool queued = sys__gpu__side_memory_write_queued(p->family, card_to, host_from, (size_t)bytes,
                                                           p->way[SYS__PAD__UP].channel);
    const bool ok = sys__pad__zzprivate_number(p, SYS__PAD__UP, t) && queued;
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return ok;
}

/* ⭐ BYTES FETCHED TO OUTSIDE THE PAD: `bytes` of the card at `card_from` queued into RAM at `host_to` — pinned already —
 * on the down way's channel, in its order and under its marks, with no chunk claimed. */
static bool sys__pad__fetch_to(sys__pad* p, void* host_to, const void* card_from, uint64_t bytes, sys__pad__ticket* t) {
    if (p == 0 || bytes == 0ull) return false;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    const bool queued = sys__gpu__side_memory_read_queued(p->family, host_to, card_from, (size_t)bytes,
                                                          p->way[SYS__PAD__DOWN].channel);
    const bool ok = sys__pad__zzprivate_number(p, SYS__PAD__DOWN, t) && queued;
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return ok;
}

/* ⭐ A DOWN HOLD FETCHED: the card's bytes at `card_from` queued into it — read once the ticket is reached, then the hold
 * released. */
static bool sys__pad__fetch(sys__pad* p, const sys__pad__hold* h, const void* card_from, sys__pad__ticket* t) {
    if (p == 0 || h->way != SYS__PAD__DOWN) return false;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    const bool ok = sys__pad__zzprivate_queue(p, h, (void*)card_from, t);
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return ok;
}

/* A hold given back without a send — a down hold read, or an up hold not used. */
static void sys__pad__release(sys__pad* p, const sys__pad__hold* h) {
    if (p == 0) return;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    if (p->chunk[h->chunk].holds != 0u) --p->chunk[h->chunk].holds;
    sys__os__condition_wake_all(&sys__pad__zzprivate_freed);
    sys__os__lock_give(&sys__pad__zzprivate_lock);
}

/* The host waits until a ticket is reached: its bytes on the card (up), or in the hold (down). */
static inline bool sys__pad__landed(sys__pad* p, sys__pad__ticket t) {
    return p != 0 && sys__pad__zzprivate_reach(p, t.way, t.number);
}

/* The card's own queue waits, on the card, until a ticket is reached — work queued on it after this call starts only
 * then, and the host does not wait. Called with the card bound. */
static inline bool sys__pad__card_waits(sys__pad* p, sys__pad__ticket t) {
    if (p == 0) return false;
    sys__os__lock_take(&sys__pad__zzprivate_lock);
    sys__pad__way* y = &p->way[t.way];
    const bool done = !p->marks || t.number <= y->reached;
    void* const mark = done ? 0 : y->mark[t.number % SYS__PAD__MARKS];
    sys__os__lock_give(&sys__pad__zzprivate_lock);
    return done || sys__gpu__mark_wait(p->family, mark);
}

/* One thread's share of a move: `piece` bytes a claim each way, its down holds read back in the order they were fetched. */
static bool sys__pad__zzprivate_move(sys__pad* p, void* card_to, const void* host_from, uint64_t up_bytes,
                                     void* host_to, const void* card_from, uint64_t down_bytes, uint64_t piece) {
    sys__pad__hold pending[SYS__PAD__CHUNKS_MAX];
    sys__pad__ticket tickets[SYS__PAD__CHUNKS_MAX];
    uint64_t into[SYS__PAD__CHUNKS_MAX];
    unsigned head = 0u, n = 0u;
    const unsigned most = p->way[SYS__PAD__DOWN].count;
    uint64_t up_at = 0ull, down_at = 0ull, down_done = 0ull;
    sys__pad__ticket last_up = { SYS__PAD__UP, 0ull };
    bool ok = true;
    while (ok && (up_at < up_bytes || down_done < down_bytes)) {
        if (up_at < up_bytes) {
            const uint64_t b = up_bytes - up_at < piece ? up_bytes - up_at : piece;
            sys__pad__hold h;
            uint8_t* at = sys__pad__claim(p, SYS__PAD__UP, b, &h);
            ok = at != 0;
            if (ok) {
                memcpy(at, (const uint8_t*)host_from + up_at, (size_t)b);
                ok = sys__pad__send(p, &h, (uint8_t*)card_to + up_at, &last_up);
                up_at += b;
            }
        }
        if (!ok) break;
        const unsigned k = (head + n) % SYS__PAD__CHUNKS_MAX;
        const uint64_t b = down_bytes - down_at < piece ? down_bytes - down_at : piece;
        if (down_at < down_bytes && n < most && sys__pad__zzprivate_claim(p, SYS__PAD__DOWN, b, &pending[k], n == 0u) != 0) {
            ok = sys__pad__fetch(p, &pending[k], (const uint8_t*)card_from + down_at, &tickets[k]);
            into[k] = down_at;
            down_at += b;
            ++n;
        } else if (down_at < down_bytes && n == 0u) {
            ok = false;                                 /* a waiting claim fails only where a reach did */
        } else if (n > 0u) {
            ok = sys__pad__landed(p, tickets[head]);
            if (ok) memcpy((uint8_t*)host_to + into[head], sys__pad__at(p, &pending[head]), (size_t)pending[head].bytes);
            down_done += pending[head].bytes;
            sys__pad__release(p, &pending[head]);
            head = (head + 1u) % SYS__PAD__CHUNKS_MAX;
            --n;
        }
    }
    for (; n > 0u; --n, head = (head + 1u) % SYS__PAD__CHUNKS_MAX) {   /* a failure's holds given back */
        (void)sys__pad__landed(p, tickets[head]);
        sys__pad__release(p, &pending[head]);
    }
    return ok && (last_up.number == 0ull || sys__pad__landed(p, last_up));
}

typedef struct sys__pad__share {
    sys__pad* p;
    uint8_t* card_to; const uint8_t* host_from; uint64_t up_bytes;
    uint8_t* host_to; const uint8_t* card_from; uint64_t down_bytes;
    uint64_t piece;
    bool ok;
} sys__pad__share;
static void* sys__pad__zzprivate_share_run(void* arg) {
    sys__pad__share* s = (sys__pad__share*)arg;
    s->ok = sys__pad__zzprivate_move(s->p, s->card_to, s->host_from, s->up_bytes, s->host_to, s->card_from, s->down_bytes,
                                     s->piece);
    return 0;
}

/* ⭐ THE CARD'S COMPUTE QUEUE, MARKED: compute mark `which` (0 or 1) placed after everything queued on the bound card's
 * compute queue so far — so that memory the kernels queued so far read can be written again once it is reached
 * (▶ `compute_reached`), the host never waiting for the whole card. Called with the card bound. */
static inline bool sys__pad__mark_compute(sys__pad* p, unsigned which) {
    if (p == 0 || which > 1u) return false;
    return !p->marks || sys__gpu__mark_record(p->family, p->compute[which]);
}
/* The host waits until compute mark `which`'s latest recording is reached — without marks, until the card is done. */
static inline bool sys__pad__compute_reached(sys__pad* p, unsigned which) {
    if (p == 0 || which > 1u) return false;
    return p->marks ? sys__gpu__mark_reached(p->family, p->compute[which]) : sys__gpu__compute_completed(p->family);
}

/* ⭐⭐ BOTH WAYS AT ONCE: `up_bytes` of RAM at `host_from` to the card at `card_to`, and `down_bytes` of the card at
 * `card_from` into RAM at `host_to` — each way cut into `threads` shares moved side by side, every thread claiming its
 * pieces of the same open chunk, a chunk filled while the last is copied; every copy reached when it answers. What the
 * card wrote before is waited for first. Either way may be empty. */
#define SYS__PAD__THREADS_MAX 16u
static bool sys__pad__move(sys__pad* p, void* card_to, const void* host_from, uint64_t up_bytes,
                           void* host_to, const void* card_from, uint64_t down_bytes, unsigned threads) {
    if (p == 0 || threads == 0u || threads > SYS__PAD__THREADS_MAX) return false;
    if (down_bytes != 0ull && !sys__gpu__compute_completed(p->family)) return false;
    /* a piece a thread's share of a chunk, whole pages, so every thread fills the open chunk at once */
    uint64_t piece = (p->chunk_bytes / threads) & ~4095ull;
    if (piece == 0ull) piece = p->chunk_bytes;
    sys__pad__share share[SYS__PAD__THREADS_MAX];
    sys__os__thread helper[SYS__PAD__THREADS_MAX];
    bool started[SYS__PAD__THREADS_MAX];
    const uint64_t up_each = (up_bytes / threads + 4095ull) & ~4095ull, down_each = (down_bytes / threads + 4095ull) & ~4095ull;
    for (unsigned t = 0u; t < threads; ++t) {
        const uint64_t u0 = t * up_each < up_bytes ? t * up_each : up_bytes;
        const uint64_t u1 = u0 + up_each < up_bytes ? u0 + up_each : up_bytes;
        const uint64_t d0 = t * down_each < down_bytes ? t * down_each : down_bytes;
        const uint64_t d1 = d0 + down_each < down_bytes ? d0 + down_each : down_bytes;
        const sys__pad__share s = { p, (uint8_t*)card_to + u0, (const uint8_t*)host_from + u0, u1 - u0,
                                              (uint8_t*)host_to + d0, (const uint8_t*)card_from + d0, d1 - d0, piece, false };
        share[t] = s;
        started[t] = t != 0u && sys__os__thread_start(&helper[t], sys__pad__zzprivate_share_run, &share[t]);
        if (t != 0u && !started[t]) (void)sys__pad__zzprivate_share_run(&share[t]);
    }
    (void)sys__pad__zzprivate_share_run(&share[0]);
    bool ok = true;
    for (unsigned t = 0u; t < threads; ++t) {
        if (t != 0u && started[t]) sys__os__thread_join(helper[t]);
        ok = ok && share[t].ok;
    }
    return ok;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_PAD_CUH */
