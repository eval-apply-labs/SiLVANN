#ifndef SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../operating_systems/os.cuh"   /* threads, locks, conditions and the clock: the system's */
#include <stdlib.h>
#include <string.h>                             /* the slice's copies: memcpy, memset */
#include "expert__header.cuh"                           /* the index this reads into, and what it publishes */
#include "../../sys/cpu/silicon/silicon__header.cuh"   /* the family's file_read */
#include "../../sys/cpu/silicon/pad.cuh"                /* a card's landing pad, what a promotion is moved through */
#include "../contracts/objects/routed.cuh"               /* a streamed prompt's batch, at the most */

/* ══ ⭐⭐ THE LOADER — AN EXPERT'S DISK READ, OFF THE EVALUATOR'S THREAD ════════════════════════════════════
 * ⚖ *"go ahead with the loader pool and prefetch"*. A few host threads do nothing but read: each takes a queued
 * READ — an expert's four slices, from its file into a slot reserved for it — and does it with the worker
 * family's `file_read`. Everything the INDEX knows stays on the evaluator's thread: the slot is taken (or an
 * old expert evicted for it) when the read is queued, and the expert is admitted when the read is settled.
 * ⛳ SO A LOADER THREAD TOUCHES BYTES AND NEVER THE INDEX, and a read in flight is in no LRU chain — nothing
 *   can choose it as a victim while its bytes are arriving.
 * ⛳ A READ IS SETTLED BY WHOEVER NEEDS ITS LAYER: `settle` waits for every read of a (layer, type) band and
 *   admits each, whether a program picked it or a prediction did — so no reserved slot outlives its layer.
 * The threads start the first time a read is queued (`SILVANN_EXPERT_LOADERS`, 4 by default) and are joined
 * at the package's teardown. */

#define NN__LOADER__FREE     0u
#define NN__LOADER__QUEUED   1u
#define NN__LOADER__READING  2u
#define NN__LOADER__DONE     3u
#define NN__LOADER__FAILED   4u
#define NN__LOADER__WAITING  5u   /* half duplex: an evacuation until its gather lands, a write until the CPUs are done — no thread takes it */

typedef struct nn__loader__read {
    uint32_t state;
    bool     predicted;            /* queued by a prediction, not by a pick */
    uint64_t owner;                /* the index of the worker that asked — several workers share these threads */
    sys__silicon_family__id family;
    uint64_t layer, type, expert, slot, room;
    nn__expert__backing backing;
    uint64_t seq;                  /* when it was queued: the threads take the oldest first */
    uint64_t made;                 /* when its worker asked for it — the order `settle` admits a worker's reads in */
    bool gather;                   /* a promotion: `runs` gathered from memory into `room`, `bytes` long, not a file read */
    uint64_t bytes;
    nn__expert__gather runs;
    uint32_t job;                  /* what it is — ▶ NN__LOADER__JOB_*; `gather` stays true for a GATHER */
    unsigned channel;              /* an EVACUATE's or a WRITE's evacuation channel */
    uint64_t by;                   /* a WRITE: the worker whose commit queued it, the one whose program releases it */
    uint64_t queued_ns;            /* a GATHER: when it was queued, for the time it takes to land */
    unsigned device;               /* a GATHER's or a stage's: its card, whose pad it is moved through */
    bool     in_place;             /* a GATHER: every piece in RAM registered with the card, copied from where it lies */
    sys__pad__ticket ticket;       /* a GATHER: its last copy queued — on the card once the card's queue reaches it */
    sys__pad__ticket after;        /* an EVACUATE in half duplex: its swap's gather, landed before it is fetched */
    /* a READ's slices, each a thread's (NN-97): every slice with bytes, those a thread has taken, those landed, and
     *   whether any failed — the read DONE or FAILED when the last lands. Set when its first slice is taken. */
    uint32_t slices_all, slices_taken, slices_landed;
    bool     slices_failed;
} nn__loader__read;
/* ⭐ WHAT A JOB IS. A READ (a file into a slot) and a GATHER (a promotion: the CPUs' memory into a card slot) are admitted
 *   by whoever settles their layer. The exclusive tier's two are not: an EVACUATE (a card slot's bytes into an evacuation
 *   channel) waits for the commit that sends its expert to RAM, and a WRITE (a channel into an expert's RAM slot) frees
 *   itself — what waits for it is a CPU about to compute that expert (`nn__expert__wait_written`). So the settles, the
 *   finds and a promotion's in-flight test see only the first two. An exclusive group's slice change has two of its own,
 *   in a table of their own (▶ `stage_jobs`): a STAGE_DOWN (a card slot's bytes into a stage, freed when it lands) and a
 *   STAGE_UP (a stage into a card slot, WAITING until its stage is filled, admitted by its card's switch). */
#define NN__LOADER__JOB_READ        0u
#define NN__LOADER__JOB_GATHER      1u
#define NN__LOADER__JOB_EVACUATE    2u
#define NN__LOADER__JOB_WRITE       3u
#define NN__LOADER__JOB_STAGE_DOWN  4u
#define NN__LOADER__JOB_STAGE_UP    5u
#define NN__LOADER__ADMITTED(r)   ((r)->job == NN__LOADER__JOB_READ || (r)->job == NN__LOADER__JOB_GATHER)

static nn__loader__read nn__loader__zzprivate_reads[NN__EXPERT__FLIGHT_MAX];
/* ⭐ A SLICE CHANGE'S STAGES (NN-106): pinned memory an expert's bytes wait in between two cards of an exclusive group —
 * read off the card that holds it, written into each card that takes it. `gen` the change it was last given to, `filled`
 * the change whose bytes landed in it (`failed` one whose read did not), `takers` the cards still to write it into
 * themselves: the next change waits for none. A stage job's `channel` is its stage and its `by` the change. */
typedef struct nn__loader__stage { void* host; uint64_t bytes, gen, filled, failed; unsigned takers; } nn__loader__stage;
static nn__loader__stage nn__loader__zzprivate_stages[NN__EXPERT_TIER__STAGES];
static nn__loader__read nn__loader__zzprivate_stage_jobs[NN__EXPERT_TIER__STAGE_JOBS];
static void nn__expert_tier__zzprivate_gathered(const nn__loader__read* gather);   /* ▶ the exclusive tier, below */
static bool nn__expert_tier__zzprivate_kept(uint64_t l, uint64_t e);               /* ▶ a group's slice, below */
static void nn__expert_tier__zzprivate_forget(void);
static sys__os__lock    nn__loader__zzprivate_lock = SYS__OS__LOCK_INIT;
static sys__os__condition nn__loader__zzprivate_work = SYS__OS__CONDITION_INIT;   /* a read was queued */
static sys__os__condition nn__loader__zzprivate_done = SYS__OS__CONDITION_INIT;   /* a read finished   */
static sys__os__thread  nn__loader__zzprivate_threads[NN__EXPERT__LOADERS_MAX];
static unsigned         nn__loader__zzprivate_started = 0u;
static bool             nn__loader__zzprivate_stopping = false;
static uint64_t         nn__loader__zzprivate_count[NN__EXPERT__COUNTS];
static uint64_t         nn__loader__zzprivate_seq = 1ull << 62;       /* the queue's clock, under the lock */
static uint64_t         nn__loader__zzprivate_front = 0ull;           /* the clock of the jobs taken ahead (▶ `nn__expert__ahead`),
                                                                       * below every other job's */
/* writes to RAM queued or in flight — an exclusive tier's demotions — so a wait for one costs nothing where there are none */
static uint64_t         nn__loader__zzprivate_writes_out = 0ull;
/* The evacuation channels (▶ the exclusive tier, below): a region of its card's pad's down way an expert's bytes wait in,
 * between the card and the RAM slots they are written into — the pad's parking (▶ sys's `pad.cuh`). `held` from the
 * evacuation until the commit hands the bytes on; `pending` the writes still reading it — the region is given back to the
 * pad when both are clear, and a channel's next user waits while it is pending. */
typedef struct nn__loader__channel { sys__pad* pad; sys__pad__hold hold; bool held; unsigned pending; } nn__loader__channel;
static nn__loader__channel nn__loader__zzprivate_channels[NN__EXPERT_TIER__CHANNELS];

/* A channel whose evacuation is no longer held and whose writes have all landed gives its region back to the pad. Lock
 * held. */
static void nn__loader__zzprivate_channel_settle(nn__loader__channel* ch) {
    if (ch->held || ch->pending != 0u || ch->pad == 0) return;
    sys__pad__release(ch->pad, &ch->hold);
    ch->pad = 0;
}

/* ⭐ THE CPUS' EXPERTS REGISTERED WITH A CARD (NN-115, ▶ `contracts/objects/expert.cuh`): each span RAM this process holds,
 * page-locked and mapped for one family's copy engine. Written on the evaluator's thread when a model loads, read by the
 * loader threads under the loader's lock. */
typedef struct nn__loader__span { uint64_t lo, hi; sys__silicon_family__id family; } nn__loader__span;
static nn__loader__span nn__loader__zzprivate_spans[NN__EXPERT__SPANS_MAX];
static unsigned nn__loader__zzprivate_n_spans = 0u;

/* `bytes` from `at` registered with `family` (the calling thread's card bound): true when it was, or already is. */
static bool nn__loader__register(sys__silicon_family__id family, uint64_t at, uint64_t bytes) {
    if (bytes == 0ull) return true;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    for (unsigned i = 0u; i < nn__loader__zzprivate_n_spans; ++i) {
        const nn__loader__span* s = &nn__loader__zzprivate_spans[i];
        if (s->family == family && s->lo <= at && at + bytes <= s->hi) { sys__os__lock_give(&nn__loader__zzprivate_lock); return true; }
    }
    bool ok = nn__loader__zzprivate_n_spans < NN__EXPERT__SPANS_MAX;
    void* card_at = 0;
    ok = ok && sys__gpu__memory_register_host(family, (void*)(uintptr_t)at, (size_t)bytes, &card_at);
    if (ok) {
        const nn__loader__span s = { at, at + bytes, family };
        nn__loader__zzprivate_spans[nn__loader__zzprivate_n_spans++] = s;
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return ok;
}

/* Whether every piece of a promotion's runs lies in RAM registered with `family`, in few enough pieces to copy each
 * straight from where it lies (▶ NN__EXPERT__IN_PLACE_PIECES). Lock not held. */
static bool nn__loader__zzprivate_in_place(sys__silicon_family__id family, const nn__expert__gather* runs) {
    uint64_t pieces = 0ull;
    bool ok = runs->runs != 0u;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    for (unsigned i = 0u; i < runs->runs && ok; ++i) {
        const nn__expert__run* u = &runs->run[i];
        pieces += u->count;
        if (u->count == 0ull) continue;
        const uint64_t lo = u->from, hi = u->from + (u->count - 1ull) * u->from_stride + u->bytes;
        bool inside = false;
        for (unsigned s = 0u; s < nn__loader__zzprivate_n_spans && !inside; ++s) {
            const nn__loader__span* p = &nn__loader__zzprivate_spans[s];
            inside = p->family == family && p->lo <= lo && hi <= p->hi;
        }
        ok = inside && pieces <= NN__EXPERT__IN_PLACE_PIECES;
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return ok;
}

/* How many threads fill a promotion: two halves of its slot through the pad — one thread's memcpy is ~9 GB/s and the link
 * 13.6, so two keep it full (`MEASURED` measurements/2026-10-24_nn115_pad_node03.md: 13.3 GB/s for GLM's expert, 12.4 for
 * the 35B's) — and one where it is copied from where it lies, or is smaller than two pages. */
static inline uint32_t nn__loader__zzprivate_halves(const nn__loader__read* r) {
    return (r->in_place || r->bytes < 8192ull) ? 1u : 2u;
}

/* ⭐ A PROMOTION'S BYTES, ON A LOADER THREAD, THROUGH ITS CARD'S PAD (▶ sys's `pad.cuh`): from registered RAM, each piece
 * queued to the card's slot from where it lies; else this thread's half of the slot (`half`, ▶ `…_halves`) copied from the
 * runs into a region of the pad's open chunk in the slot's layout, and sent. Answered once its copies are QUEUED, with the
 * last one's ticket in `t`: the card's own queue waits for it where the expert is admitted (▶ `…_card_waits`), so neither
 * this thread nor the card's waits for the copy on the host. The copies run on the card's own channel, beside the kernels
 * on its compute queue. */
static bool nn__loader__zzprivate_gather(const nn__loader__read* r, unsigned half, sys__pad__ticket* t) {
    sys__pad* p = sys__pad__of(r->family, r->device);
    if (p == 0) return false;
    if (r->in_place) {
        for (unsigned i = 0u; i < r->runs.runs; ++i) {
            const nn__expert__run* u = &r->runs.run[i];
            for (uint64_t c = 0ull; c < u->count; ++c) {
                if (u->into + c * u->into_stride + u->bytes > r->bytes) return false;
                if (!sys__pad__send_from(p, (const void*)(uintptr_t)(u->from + c * u->from_stride),
                                         (void*)(uintptr_t)(r->room + u->into + c * u->into_stride), u->bytes, t))
                    return false;
            }
        }
        return true;
    }
    const uint64_t split = (r->bytes / 2ull) & ~4095ull;
    const uint64_t lo = nn__loader__zzprivate_halves(r) == 1u || half == 0u ? 0ull : split;
    const uint64_t hi = nn__loader__zzprivate_halves(r) == 1u || half == 1u ? r->bytes : split;
    sys__pad__hold h;
    uint8_t* at = sys__pad__claim(p, SYS__PAD__UP, hi - lo, &h);
    if (at == 0) return false;
    /* the runs are this process's own memory — a CPU worker's slots — so they are read through the host family's door */
    static __thread sys__silicon_family__id host = SYS__SILICON_FAMILY__NONE;
    if (host == SYS__SILICON_FAMILY__NONE) host = sys__silicon_family__named("host");
    bool ok = true;
    for (unsigned i = 0u; i < r->runs.runs && ok; ++i) {
        const nn__expert__run* u = &r->runs.run[i];
        for (uint64_t c = 0ull; c < u->count && ok; ++c) {
            const uint64_t into = u->into + c * u->into_stride;
            ok = into + u->bytes <= r->bytes;
            const uint64_t a = into > lo ? into : lo, b = into + u->bytes < hi ? into + u->bytes : hi;
            if (ok && a < b)
                ok = sys__gpu__memory_read(host, at + (a - lo), (const void*)(uintptr_t)(u->from + c * u->from_stride + (a - into)),
                                           (size_t)(b - a));
        }
    }
    if (!ok) { sys__pad__release(p, &h); return false; }
    return sys__pad__send(p, &h, (void*)(uintptr_t)(r->room + lo), t);
}

/* A gather about to be admitted: the card's own queue waits until its copies have landed — what the card computes after
 * reads the expert, and the host does not wait (`sys__pad__card_waits`). On the card's own thread. */
static inline bool nn__loader__zzprivate_card_waits(const nn__loader__read* r) {
    if (r->job != NN__LOADER__JOB_GATHER || r->ticket.number == 0ull) return true;
    return sys__pad__card_waits(sys__pad__of(r->family, r->device), r->ticket);
}

/* An EVACUATE: the card slot at `room` fetched into its channel's region of the pad, on the pad's down channel — beside
 * the kernels, and beside the promotions going up. Answered once it has landed. */
static bool nn__loader__zzprivate_evacuate(const nn__loader__read* r) {
    if (r->channel >= NN__EXPERT_TIER__CHANNELS) return false;
    const nn__loader__channel* c = &nn__loader__zzprivate_channels[r->channel];
    if (c->pad == 0 || c->hold.bytes < r->bytes) return false;
    /* half duplex: the swap's other copy on the link first, landed (▶ `nn__expert_tier__zzprivate_gathered`) */
    if (r->after.number != 0ull && !sys__pad__landed(c->pad, r->after)) return false;
    sys__pad__ticket t;
    return sys__pad__fetch(c->pad, &c->hold, (const void*)(uintptr_t)r->room, &t) && sys__pad__landed(c->pad, t);
}
/* A WRITE: the runs copied out of the channel into an expert's RAM slot — both this process's memory, `from` and `into`
 *   absolute addresses. */
static bool nn__loader__zzprivate_write(const nn__loader__read* r) {
    static __thread sys__silicon_family__id host = SYS__SILICON_FAMILY__NONE;
    if (host == SYS__SILICON_FAMILY__NONE) host = sys__silicon_family__named("host");
    for (unsigned i = 0u; i < r->runs.runs; ++i) {
        const nn__expert__run* u = &r->runs.run[i];
        for (uint64_t c = 0ull; c < u->count; ++c)
            if (!sys__gpu__memory_read(host, (void*)(uintptr_t)(u->into + c * u->into_stride),
                                       (const void*)(uintptr_t)(u->from + c * u->from_stride), (size_t)u->bytes))
                return false;
    }
    return true;
}
/* A STAGE_DOWN or a STAGE_UP: a card slot read into its stage, or its stage written into a card slot — on the card's pad's
 *   channels, beside the kernels. Answered once it has landed. */
static bool nn__loader__zzprivate_stage_copy(const nn__loader__read* r) {
    if (r->channel >= NN__EXPERT_TIER__STAGES) return false;
    const nn__loader__stage* s = &nn__loader__zzprivate_stages[r->channel];
    sys__pad* p = sys__pad__of(r->family, r->device);
    if (p == 0 || s->host == 0 || s->bytes < r->bytes) return false;
    sys__pad__ticket t;
    const bool queued = r->job == NN__LOADER__JOB_STAGE_DOWN
                      ? sys__pad__fetch_to(p, s->host, (const void*)(uintptr_t)r->room, r->bytes, &t)
                      : sys__pad__send_from(p, s->host, (void*)(uintptr_t)r->room, r->bytes, &t);
    return queued && sys__pad__landed(p, t);
}

/* A stage's read landed (or failed) for change `gen`: every card's write of it that waits goes, or fails with it. Lock
 * held. */
static void nn__loader__zzprivate_stage_filled(unsigned stage, uint64_t gen, bool ok) {
    if (stage >= NN__EXPERT_TIER__STAGES) return;
    if (ok) nn__loader__zzprivate_stages[stage].filled = gen;
    else    nn__loader__zzprivate_stages[stage].failed = gen;
    for (unsigned i = 0u; i < NN__EXPERT_TIER__STAGE_JOBS; ++i) {
        nn__loader__read* u = &nn__loader__zzprivate_stage_jobs[i];
        if (u->state != NN__LOADER__WAITING || u->job != NN__LOADER__JOB_STAGE_UP || u->channel != stage || u->by != gen) continue;
        u->state = ok ? NN__LOADER__QUEUED : NN__LOADER__FAILED;
        u->seq = ++nn__loader__zzprivate_seq;
    }
    sys__os__condition_wake_all(&nn__loader__zzprivate_work);
}

static void* nn__loader__zzprivate_run(void* unused) {
    (void)unused;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    for (;;) {
        /* the oldest first — so a layer's promotions land in the order its verb will compute them: a job queued, or a
         *   READ or a GATHER another thread is moving whose slices are not all taken (NN-97: an expert's slices read side
         *   by side, where the pack lays them apart — the file's one run was one read; NN-115: a promotion's two halves
         *   filled side by side through the pad). The jobs taken ahead (`nn__expert__ahead`) are older than any other:
         *   their clock runs below the queue's */
        nn__loader__read* job = 0;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX + NN__EXPERT_TIER__STAGE_JOBS; ++i) {
            nn__loader__read* c = i < NN__EXPERT__FLIGHT_MAX ? &nn__loader__zzprivate_reads[i]
                                                              : &nn__loader__zzprivate_stage_jobs[i - NN__EXPERT__FLIGHT_MAX];
            const bool sliced = (c->job == NN__LOADER__JOB_READ && !c->gather) || c->job == NN__LOADER__JOB_GATHER;
            const bool open = c->state == NN__LOADER__QUEUED
                           || (c->state == NN__LOADER__READING && sliced && (c->slices_all & ~c->slices_taken) != 0u);
            if (open && (job == 0 || c->seq < job->seq)) job = c;
        }
        if (job == 0) {
            if (nn__loader__zzprivate_stopping) break;
            sys__os__condition_wait(&nn__loader__zzprivate_work, &nn__loader__zzprivate_lock);
            continue;
        }
        const bool gather = job->job == NN__LOADER__JOB_GATHER;
        const bool slices = (job->job == NN__LOADER__JOB_READ && !job->gather) || gather;
        if (job->state == NN__LOADER__QUEUED) {
            job->state = NN__LOADER__READING;
            if (slices) {
                job->slices_all = 0u; job->slices_taken = 0u; job->slices_landed = 0u; job->slices_failed = false;
                if (gather) {
                    job->slices_all = (1u << nn__loader__zzprivate_halves(job)) - 1u;
                    job->ticket.way = SYS__PAD__UP; job->ticket.number = 0ull;
                }
                else for (unsigned i = 0u; i < NN__EXPERT__SLICES; ++i) if (job->backing.bytes[i] != 0ull) job->slices_all |= 1u << i;
            }
        }
        /* a READ's or a GATHER's: this thread's slice, the lowest not taken — and the others woken, to take the rest */
        unsigned slice = NN__EXPERT__SLICES;
        if (slices) {
            const uint32_t open = job->slices_all & ~job->slices_taken;
            if (open != 0u) {
                slice = (unsigned)__builtin_ctz(open);
                job->slices_taken |= 1u << slice;
                if ((job->slices_all & ~job->slices_taken) != 0u) sys__os__condition_wake_all(&nn__loader__zzprivate_work);
            }
        }
        const nn__loader__read r = *job;
        sys__os__lock_give(&nn__loader__zzprivate_lock);
        bool ok = true;
        sys__pad__ticket queued = { SYS__PAD__UP, 0ull };
        if (r.job == NN__LOADER__JOB_STAGE_DOWN || r.job == NN__LOADER__JOB_STAGE_UP) ok = nn__loader__zzprivate_stage_copy(&r);
        else if (r.job == NN__LOADER__JOB_EVACUATE) ok = nn__loader__zzprivate_evacuate(&r);
        else if (r.job == NN__LOADER__JOB_WRITE) ok = nn__loader__zzprivate_write(&r);
        else if (r.gather) ok = slice < NN__EXPERT__SLICES && nn__loader__zzprivate_gather(&r, slice, &queued);
        else if (slice < NN__EXPERT__SLICES) {
            const uint64_t step = r.backing.stride[slice] != 0ull ? r.backing.stride[slice] : r.backing.bytes[slice];
            ok = sys__gpu__file_read(r.family, r.backing.file, r.backing.from[slice] + r.expert * step, r.backing.bytes[slice],
                                     (void*)(uintptr_t)(r.room + r.backing.into[slice]));
        }
        if (slices) {
            sys__os__lock_take(&nn__loader__zzprivate_lock);
            if (slice < NN__EXPERT__SLICES) job->slices_landed |= 1u << slice;
            if (!ok) job->slices_failed = true;
            if (gather && queued.number > job->ticket.number) job->ticket = queued;   /* the channel's order: the last covers all */
            const bool last = job->slices_landed == job->slices_all;
            if (last) {
                job->state = job->slices_failed ? NN__LOADER__FAILED : NN__LOADER__DONE;
                if (gather && !job->slices_failed) {
                    if (job->in_place) ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_IN_PLACE];
                    nn__expert_tier__zzprivate_gathered(job);
                }
                sys__os__condition_wake_all(&nn__loader__zzprivate_done);
            }
            continue;                                   /* the lock held, as the loop's top wants it */
        }
        sys__os__lock_take(&nn__loader__zzprivate_lock);
        if (r.job == NN__LOADER__JOB_WRITE) {
            /* a write frees itself: its expert is in RAM now, its channel one reader fewer. ⛔ A FAILED WRITE IS LOUD — the
             *   expert's RAM slot holds the wrong bytes, and nothing could tell — so it is counted, and the run stops */
            if (r.channel < NN__EXPERT_TIER__CHANNELS && nn__loader__zzprivate_channels[r.channel].pending > 0u) {
                --nn__loader__zzprivate_channels[r.channel].pending;
                nn__loader__zzprivate_channel_settle(&nn__loader__zzprivate_channels[r.channel]);
            }
            if (!ok) ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_WRITE_FAILED];
            job->state = NN__LOADER__FREE;
            __atomic_sub_fetch(&nn__loader__zzprivate_writes_out, 1ull, __ATOMIC_RELEASE);
        } else if (r.job == NN__LOADER__JOB_STAGE_DOWN) {
            /* a stage filled: the read frees itself, and each card's write of it waiting goes (▶ `…_stage_filled`) */
            job->state = NN__LOADER__FREE;
            nn__loader__zzprivate_stage_filled(r.channel, r.by, ok);
        } else {
            job->state = ok ? NN__LOADER__DONE : NN__LOADER__FAILED;
        }
        sys__os__condition_wake_all(&nn__loader__zzprivate_done);
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return 0;
}

/* The threads, the first time they are wanted. Called with the lock held. */
static bool nn__loader__zzprivate_start(void) {
    if (nn__loader__zzprivate_started != 0u) return true;
    unsigned want = NN__EXPERT__LOADERS_STANDARD;
    const char* said = getenv("SILVANN_EXPERT_LOADERS");
    if (said != 0) { const long n = strtol(said, 0, 10); if (n > 0) want = (unsigned)n; }
    if (want > NN__EXPERT__LOADERS_MAX) want = NN__EXPERT__LOADERS_MAX;
    nn__loader__zzprivate_stopping = false;
    for (unsigned t = 0u; t < want; ++t) {
        if (!sys__os__thread_start(&nn__loader__zzprivate_threads[t], nn__loader__zzprivate_run, 0)) break;
        ++nn__loader__zzprivate_started;
    }
    return nn__loader__zzprivate_started != 0u;
}

/* The package's teardown: every thread told to stop once the queue is empty, and joined. */
static void nn__loader__zzpackage_stop(void) {
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    const unsigned n = nn__loader__zzprivate_started;
    nn__loader__zzprivate_stopping = true;
    sys__os__condition_wake_all(&nn__loader__zzprivate_work);
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    for (unsigned t = 0u; t < n; ++t) sys__os__thread_join(nn__loader__zzprivate_threads[t]);
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    nn__loader__zzprivate_started = 0u;
    /* ⛔⛔ AND NOTHING OF THIS MACHINE'S IS LEFT FOR THE NEXT. A read or a promotion that landed and was never settled
     *   stays in the table, and a machine booted after this one in the same process — the text model booted again after a
     *   picture, a model the card was lent to — has an index whose owner can be the same handle: its first settle admitted
     *   the old machine's copies into its own tier, experts named resident in slots nothing was copied to, computed from
     *   whatever the card held there. `MEASURED` 2026-10-17 on the precision: the 35B with an 11 GB tier, booted a second
     *   time in a process, answered ".\n" over and over (measurements/2026-10-17_v0110rc1_on_precision.md). The table is
     *   emptied here, and the evacuation channels — regions of the pads of the machine that is going — forgotten. */
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) nn__loader__zzprivate_reads[i].state = NN__LOADER__FREE;
    for (unsigned c = 0u; c < NN__EXPERT_TIER__CHANNELS; ++c) {
        const nn__loader__channel none = {};
        nn__loader__zzprivate_channels[c] = none;
    }
    for (unsigned i = 0u; i < NN__EXPERT_TIER__STAGE_JOBS; ++i) nn__loader__zzprivate_stage_jobs[i].state = NN__LOADER__FREE;
    for (unsigned s = 0u; s < NN__EXPERT_TIER__STAGES; ++s) {
        const nn__loader__stage none = {};
        nn__loader__zzprivate_stages[s] = none;
    }
    /* the registered RAM given back to its families, before the memory under it goes with the machine */
    for (unsigned i = 0u; i < nn__loader__zzprivate_n_spans; ++i)
        sys__gpu__memory_unregister_host(nn__loader__zzprivate_spans[i].family, (void*)(uintptr_t)nn__loader__zzprivate_spans[i].lo);
    nn__loader__zzprivate_n_spans = 0u;
    nn__expert_tier__zzprivate_forget();
    nn__loader__zzprivate_writes_out = 0ull;
    sys__os__lock_give(&nn__loader__zzprivate_lock);
}

/* This worker's read of `expert` in flight, or none. Called with the lock held. ⛳ A READ IS ITS WORKER'S: two workers of
 *   one program each keep their own index and slots, and the same expert of each is two reads into two slots. */
static nn__loader__read* nn__loader__zzprivate_find(uint64_t owner, uint64_t layer, uint64_t type, uint64_t expert) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__FREE && NN__LOADER__ADMITTED(r) && r->owner == owner && r->layer == layer && r->type == type
         && r->expert == expert) return r;
    }
    return 0;
}

/* A slot for a read. ⭐ EVERY LAYER KEEPS ITS FAIR SHARE — the collection's slots over the layers that have experts of the
 * type: a layer holding its share gives up its own least recently used expert (one not `pinned`); one holding less takes
 * a free slot, and with none free, the least recently used of the layer that holds the most.
 * ⛳ WITHOUT THE SHARE, the first layers of a prompt read as rows took every free slot — a chunk wants most of a layer's
 *   experts at once — and a token's misses, each evicting from its own layer, never gave them back: `MEASURED` on the
 *   35B with 8 GB, 83% of reads from memory after a prompt of rows against 94% after one read a position at a time. */
/* Layer `l`'s least recently used expert that is not one of `pinned` (they are the asking layer's), evicted for its slot.
 * ⛳ A PINNED ONE CAN BE THE OLDEST: a batch's resident experts move to the front only as the batch's requests reach them,
 *   so the walk goes on toward the newer ones — at most one step a pinned expert. */
static bool nn__loader__zzprivate_evict_oldest(uint64_t l, uint64_t type, uint64_t layer, const uint64_t* pinned, unsigned npinned,
                                               uint64_t* slot, uint64_t* room) {
    uint64_t victim = 0ull;
    if (nn__expert__layer_experts(l, type) == 0ull || !nn__expert__oldest(l, type, &victim)) return false;
    for (unsigned step = 0u; step <= npinned; ++step) {
        bool held = false;
        for (unsigned p = 0u; l == layer && p < npinned; ++p) held = held || pinned[p] == victim;
        if (!held) return nn__expert__evict(l, type, victim) && nn__expert__type_take(type, slot, room);
        sys__heap_node me;
        if (!nn__expert__slot(l, type, victim, &me) || me.args[NN__EXPERT__SLOT_PREV] == NN__EXPERT__NONE) return false;
        victim = NN__EXPERT__UNLINK(me.args[NN__EXPERT__SLOT_PREV]);
    }
    return false;
}
static bool nn__loader__zzprivate_room(uint64_t layer, uint64_t type, const uint64_t* pinned, unsigned npinned,
                                       uint64_t* slot, uint64_t* room) {
    const uint64_t layers = nn__expert__index_layers();
    uint64_t banded = 0ull;
    for (uint64_t l = 0ull; l < layers; ++l) banded += nn__expert__layer_experts(l, type) != 0ull ? 1ull : 0ull;
    const uint64_t share = banded == 0ull ? 0ull : nn__expert__type_live(type) / banded;
    if (share != 0ull && nn__expert__layer_resident(layer, type) >= share
     && nn__loader__zzprivate_evict_oldest(layer, type, layer, pinned, npinned, slot, room))
        return true;
    if (nn__expert__type_take(type, slot, room)) return true;
    uint64_t most = layers, held = 0ull;
    for (uint64_t l = 0ull; l < layers; ++l) {
        const uint64_t n = nn__expert__layer_resident(l, type);
        if (n > held) { held = n; most = l; }
    }
    if (most < layers && nn__loader__zzprivate_evict_oldest(most, type, layer, pinned, npinned, slot, room)) return true;
    if (nn__loader__zzprivate_evict_oldest(layer, type, layer, pinned, npinned, slot, room)) return true;
    /* ⛳ AND ANY OTHER LAYER HOLDING SOME: a streamed prompt's ring is a few slots spread over the layers, so the layer
     *   holding the most can be the asking one with all of its own pinned while the rest lie in others (▶ nn's STREAM) */
    for (uint64_t l = 0ull; l < layers; ++l)
        if (l != layer && l != most && nn__expert__layer_resident(l, type) != 0ull
         && nn__loader__zzprivate_evict_oldest(l, type, layer, pinned, npinned, slot, room)) return true;
    return false;
}


static __device__ inline int nn__expert__request(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t expert,
                                                 const nn__expert__backing* backing, bool predicted,
                                                 const uint64_t* pinned, unsigned npinned, uint64_t* at) {
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me)) return NN__EXPERT__REFUSED;
    if (me.args[NN__EXPERT__SLOT_AT] != 0ull) {
        if (at != 0) *at = me.args[NN__EXPERT__SLOT_AT];
        return nn__expert__touch(layer, type, expert) ? NN__EXPERT__RESIDENT : NN__EXPERT__REFUSED;
    }
    const uint64_t owner = nn__expert__zzpackage_owner();
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    nn__loader__read* r = nn__loader__zzprivate_find(owner, layer, type, expert);
    if (r != 0) {
        if (!predicted && r->predicted) { r->predicted = false; ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PREDICTED_USED]; }
        sys__os__lock_give(&nn__loader__zzprivate_lock);
        return NN__EXPERT__ARRIVING;
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    if (backing == 0 || backing->file == 0ull) return NN__EXPERT__REFUSED;

    uint64_t slot = 0ull, room = 0ull;
    if (!nn__loader__zzprivate_room(layer, type, pinned, npinned, &slot, &room)) return NN__EXPERT__REFUSED;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    nn__loader__read* free_read = 0;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX && free_read == 0; ++i)
        if (nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE) free_read = &nn__loader__zzprivate_reads[i];
    if (free_read == 0 || !nn__loader__zzprivate_start()) {
        sys__os__lock_give(&nn__loader__zzprivate_lock);
        (void)nn__expert__type_give(type, slot);
        return NN__EXPERT__REFUSED;
    }
    free_read->state = NN__LOADER__QUEUED;
    free_read->predicted = predicted;
    free_read->owner = owner;
    free_read->family = family;
    free_read->layer = layer; free_read->type = type; free_read->expert = expert;
    free_read->slot = slot; free_read->room = room;
    free_read->backing = *backing;
    free_read->gather = false;
    free_read->job = NN__LOADER__JOB_READ;
    free_read->made = free_read->seq = ++nn__loader__zzprivate_seq;
    ++nn__loader__zzprivate_count[predicted ? NN__EXPERT__COUNT_PREDICTED : NN__EXPERT__COUNT_MISSES];
    sys__os__condition_wake_one(&nn__loader__zzprivate_work);
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

/* ⭐ A WAIT FOR A READ, TIMED — the lock held, as `pthread_cond_wait` needs it: the time is what a layer spent waiting on
 *   the disk, the thing a cache's misses cost beyond the compute that overlaps them (`nn__expert__count`). */
/* Now, in ns, for the gathers' time to land. */
static inline uint64_t nn__loader__zzprivate_now_ns(void) {
    struct timespec a;
    sys__os__clock(&a);
    return (uint64_t)a.tv_sec * 1000000000ull + (uint64_t)a.tv_nsec;
}

/* A gather landed and settled: its time from queued to now counted. Lock held. */
static inline void nn__loader__zzprivate_landed(const nn__loader__read* r) {
    if (r->job != NN__LOADER__JOB_GATHER || r->queued_ns == 0ull) return;
    nn__loader__zzprivate_count[NN__EXPERT__COUNT_LAND_NS] += nn__loader__zzprivate_now_ns() - r->queued_ns;
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_LANDED];
}

/* ⭐ A PROMOTION HELD UNTIL THE CPUS ARE DONE (NN-99 ④): its gather — a CPU's copy of the expert's pieces into pinned
 *   staging, 12 MB on GLM — runs beside the sockets' experts for the memory and the cores when it starts at once, and those
 *   layers ran ~4x slower — that much `MEASURED` (2026-10-21, a rocprof trace of GLM on one MI50 of the ML350: 6 % of
 *   layers, 23 % of the time). ⚖ *"send the copy over at the end … so it does the write while attention is being
 *   computed"*: held in WAITING like a half duplex write, let go by `nn__expert_tier__written` — the program's word that
 *   the CPUs are done — or by anyone who must wait for it to land. `SILVANN_HOLD_PROMOTIONS=1`; OFF, as on the ML350 it
 *   changed nothing (115.8 against 117.0 ms a token, GLM on one card, 18 GB) — where that card's link is PCIe 3 x8 and a
 *   promotion's 12 MB take ~2.1 ms at the link's ~6 GB/s. ⛳ To be measured where the DMA is wider (the precision). */
static bool nn__loader__zzprivate_hold_gathers(void) {
    static int hold = -1;
    if (hold < 0) { const char* e = getenv("SILVANN_HOLD_PROMOTIONS"); hold = (e != 0 && e[0] == '1') ? 1 : 0; }
    return hold == 1;
}
/* A held promotion let go now, because someone waits for it. Called with the lock held. */
static inline void nn__loader__zzprivate_let_go(nn__loader__read* r) {
    if (r->state != NN__LOADER__WAITING || r->job != NN__LOADER__JOB_GATHER) return;
    r->state = NN__LOADER__QUEUED;
    r->seq = ++nn__loader__zzprivate_seq;
    sys__os__condition_wake_all(&nn__loader__zzprivate_work);
}

static inline void nn__loader__zzprivate_wait(void) {
    struct timespec a, b;
    sys__os__clock(&a);
    sys__os__condition_wait(&nn__loader__zzprivate_done, &nn__loader__zzprivate_lock);
    sys__os__clock(&b);
    nn__loader__zzprivate_count[NN__EXPERT__COUNT_WAIT_NS] += (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull + (uint64_t)(b.tv_nsec - a.tv_nsec);
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_WAITS];
}

static __device__ inline bool nn__expert__settle(uint64_t layer, uint64_t type) {
    bool ok = true;
    const uint64_t owner = nn__expert__zzpackage_owner();
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    /* ⛳ IN THE ORDER THIS WORKER MADE THEM, NOT THE TABLE'S: an admission makes its expert the band's most recent, and
     *   which entry a read took depends on the other workers' reads around it */
    for (;;) {
        nn__loader__read* next = 0;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
            nn__loader__read* r = &nn__loader__zzprivate_reads[i];
            if (r->state == NN__LOADER__FREE || !NN__LOADER__ADMITTED(r) || r->owner != owner || r->layer != layer || r->type != type) continue;
            nn__loader__zzprivate_let_go(r);
            if (next == 0 || r->made < next->made) next = r;
        }
        if (next == 0) break;
        if (next->state == NN__LOADER__QUEUED || next->state == NN__LOADER__READING) { nn__loader__zzprivate_wait(); continue; }
        /* done or failed: the index learns of it here, on this thread */
        const bool read = next->state == NN__LOADER__DONE && nn__loader__zzprivate_card_waits(next);
        const uint64_t slot = next->slot, room = next->room, expert = next->expert;
        if (read) nn__loader__zzprivate_landed(next);
        next->state = NN__LOADER__FREE;
        if (!read || !nn__expert__admit(layer, expert, room, type)) {
            (void)nn__expert__type_give(type, slot);
            ok = false;
        }
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return ok;
}

static __device__ inline int nn__expert__promote(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t expert,
                                                 const nn__expert__gather* gather, const uint64_t* pinned, unsigned npinned) {
    sys__heap_node me;
    if (gather == 0 || gather->runs == 0u || gather->runs > NN__EXPERT__GATHER_RUNS || !nn__expert__slot(layer, type, expert, &me))
        return NN__EXPERT__REFUSED;
    if (me.args[NN__EXPERT__SLOT_AT] != 0ull) return NN__EXPERT__RESIDENT;
    const uint64_t owner = nn__expert__zzpackage_owner(), bytes = nn__expert__type_bytes(type);
    const bool in_place = nn__loader__zzprivate_in_place(family, gather);
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    nn__loader__read* free_read = 0;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__FREE && NN__LOADER__ADMITTED(r) && r->owner == owner && r->layer == layer && r->type == type
         && r->expert == expert) {
            sys__os__lock_give(&nn__loader__zzprivate_lock);
            return NN__EXPERT__ARRIVING;
        }
        if (r->state == NN__LOADER__FREE && free_read == 0) free_read = r;
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    if (free_read == 0 || bytes == 0ull) return NN__EXPERT__REFUSED;      /* no room in flight: the CPU keeps it this time */
    uint64_t slot = 0ull, room = 0ull;
    if (!nn__loader__zzprivate_room(layer, type, pinned, npinned, &slot, &room)) return NN__EXPERT__REFUSED;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    /* ⛳ AN ENTRY ANOTHER WORKER TOOK MEANWHILE IS NOT A REFUSAL: a group's cards promote at once, and refusing the second
     *   would make which promotion each card starts a matter of timing — their tiers part from boot to boot */
    for (unsigned i = 0u; free_read->state != NN__LOADER__FREE && i < NN__EXPERT__FLIGHT_MAX; ++i)
        free_read = &nn__loader__zzprivate_reads[i];
    if (free_read->state != NN__LOADER__FREE || !nn__loader__zzprivate_start()) {
        sys__os__lock_give(&nn__loader__zzprivate_lock);
        (void)nn__expert__type_give(type, slot);
        return NN__EXPERT__REFUSED;
    }
    const bool hold = nn__loader__zzprivate_hold_gathers();
    free_read->state = hold ? NN__LOADER__WAITING : NN__LOADER__QUEUED;
    free_read->by = owner;
    free_read->predicted = false;
    free_read->owner = owner;
    free_read->family = family;
    free_read->layer = layer; free_read->type = type; free_read->expert = expert;
    free_read->slot = slot; free_read->room = room;
    free_read->gather = true; free_read->bytes = bytes; free_read->runs = *gather;
    free_read->job = NN__LOADER__JOB_GATHER;
    free_read->device = sys__silicon__bound_device(family); free_read->in_place = in_place;
    free_read->made = free_read->seq = ++nn__loader__zzprivate_seq;
    free_read->queued_ns = nn__loader__zzprivate_now_ns();
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PROMOTED];
    if (!hold) sys__os__condition_wake_one(&nn__loader__zzprivate_work);
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

static __device__ inline bool nn__expert__ahead(uint64_t layer, uint64_t type, uint64_t expert) {
    const uint64_t owner = nn__expert__zzpackage_owner();
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    nn__loader__read* r = nn__loader__zzprivate_find(owner, layer, type, expert);
    const bool queued = r != 0 && r->state == NN__LOADER__QUEUED;
    if (queued) r->seq = ++nn__loader__zzprivate_front;
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return queued;
}

static __device__ inline unsigned nn__expert__flight_mine(void) {
    const uint64_t owner = nn__expert__zzpackage_owner();
    unsigned n = 0u;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        const nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        n += r->state != NN__LOADER__FREE && r->owner == owner;
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return n;
}

static __device__ inline unsigned nn__expert__flight_free(void) {
    unsigned n = 0u;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    /* ⛳ A WRITE COUNTS AS FREE: it frees itself when it lands, and a promotion waits for it — counted as taken, how many a
     *   chunk promotes would be the writes' timing */
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        const nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        n += r->state == NN__LOADER__FREE || r->job == NN__LOADER__JOB_WRITE;
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return n;
}

static __device__ inline bool nn__expert__settle_one(uint64_t layer, uint64_t type, uint64_t expert) {
    bool ok = true;
    const uint64_t owner = nn__expert__zzpackage_owner();
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    for (;;) {
        nn__loader__read* r = nn__loader__zzprivate_find(owner, layer, type, expert);
        if (r == 0) break;
        nn__loader__zzprivate_let_go(r);
        if (r->state == NN__LOADER__QUEUED || r->state == NN__LOADER__READING) {
            nn__loader__zzprivate_wait();
            continue;
        }
        const bool read = r->state == NN__LOADER__DONE && nn__loader__zzprivate_card_waits(r);
        const uint64_t slot = r->slot, room = r->room;
        r->state = NN__LOADER__FREE;
        if (!read || !nn__expert__admit(layer, expert, room, type)) {
            (void)nn__expert__type_give(type, slot);
            ok = false;
        }
        break;
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return ok;
}

static __device__ inline bool nn__expert__in_memory(sys__silicon_family__id family, uint64_t type, uint64_t at) {
    return sys__gpu__memory_prefetch(family, (const void*)(uintptr_t)at, (size_t)nn__expert__type_bytes(type));
}

static __device__ inline uint64_t nn__expert__count(unsigned which) {
    if (which >= NN__EXPERT__COUNTS) return 0ull;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    const uint64_t n = nn__loader__zzprivate_count[which];
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return n;
}

static __device__ inline bool nn__expert__backing_of(uint64_t array, nn__expert__backing* backing) {
    if (backing == 0 || !sys__node_array__is(array)) return false;
    const uint64_t length = sys__node_array__length(array);
    const bool four = length >= NN__EXPERT__BACKING__FOUR && length < NN__EXPERT__BACKING__LENGTH;
    const unsigned cells = four ? NN__EXPERT__BACKING__FOUR : NN__EXPERT__BACKING__LENGTH;
    if (length < cells) return false;
    uint64_t v[NN__EXPERT__BACKING__LENGTH] = {};
    sys__node_array_walk w;
    if (!sys__node_array__walk(array, 0ull, &w)) return false;
    for (unsigned i = 0u; i < cells; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0 || n->dtype != SYS__KIND__VALUE_INT) return false;
        v[i] = n->args[0];
    }
    const nn__expert__backing none = {};
    *backing = none;
    backing->file = v[NN__EXPERT__BACKING__FILE];
    for (unsigned i = 0u; i < NN__EXPERT__SLICES; ++i) {
        if (four && i >= 4u) break;
        /* the thirteen: a slice is three integers and steps by its length */
        backing->from[i] = v[four ? 1u + 3u * i : NN__EXPERT__BACKING__FROM(i)];
        backing->bytes[i] = v[four ? 2u + 3u * i : NN__EXPERT__BACKING__BYTES(i)];
        backing->into[i] = v[four ? 3u + 3u * i : NN__EXPERT__BACKING__INTO(i)];
        backing->stride[i] = four ? 0ull : v[NN__EXPERT__BACKING__STRIDE(i)];
    }
    return true;
}

/* ── A CARD'S TIER ───────────────────────────────────────────────────────────────────────────────────── ▶ the header */
static __device__ inline bool nn__expert_tier__of(uint64_t binding, uint64_t layer, nn__expert_tier* tier) {
    if (tier == 0 || !sys__node_array__is(binding) || sys__node_array__length(binding) <= (uint64_t)sys__silicon__worker()) return false;
    const sys__heap_node entry = sys__node_array__borrow(binding, (uint64_t)sys__silicon__worker());
    if (!sys__heap_node__carries_reference(entry.dtype) || !sys__node_array__is(entry.args[0])
     || sys__node_array__length(entry.args[0]) < NN__EXPERT_TIER__LENGTH) return false;
    uint64_t v[NN__EXPERT_TIER__LENGTH];
    sys__node_array_walk w;
    if (!sys__node_array__walk(entry.args[0], 0ull, &w)) return false;
    for (unsigned i = 0u; i < NN__EXPERT_TIER__LENGTH; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0) return false;
        if (i == NN__EXPERT_TIER__SOURCES && sys__heap_node__carries_reference(n->dtype) && sys__node_array__is(n->args[0])) {
            v[i] = n->args[0];
            continue;
        }
        if (n->dtype != SYS__KIND__VALUE_INT || (i == NN__EXPERT_TIER__SOURCES && n->args[0] != 0ull)) return false;
        v[i] = n->args[0];
    }
    /* this layer's sources, or none */
    uint64_t sources = 0u;
    if (v[NN__EXPERT_TIER__SOURCES] != 0ull && layer < sys__node_array__length(v[NN__EXPERT_TIER__SOURCES])) {
        const sys__heap_node at = sys__node_array__borrow(v[NN__EXPERT_TIER__SOURCES], layer);
        if (sys__heap_node__carries_reference(at.dtype) && sys__node_array__is(at.args[0])) sources = at.args[0];
        else if (at.dtype != SYS__KIND__VALUE_INT || at.args[0] != 0ull) return false;
    }
    uint64_t ex[4] = {0u, 0u, 0u, 0u};                      /* EXCLUSIVE, PART_WORKER, PART_TYPE, DUPLEX — absent: inclusive */
    if (sys__node_array__length(entry.args[0]) >= NN__EXPERT_TIER__LENGTH_EXCLUSIVE)
        for (unsigned i = 0u; i < 4u; ++i) {
            const sys__heap_node n = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__EXCLUSIVE + i);
            if (n.dtype != SYS__KIND__VALUE_INT) return false;
            ex[i] = n.args[0];
        }
    nn__expert_tier t = {
        sources, layer, v[NN__EXPERT_TIER__TYPE], v[NN__EXPERT_TIER__EXPERTS], v[NN__EXPERT_TIER__TOP_K],
        v[NN__EXPERT_TIER__HIDDEN], v[NN__EXPERT_TIER__INTER], v[NN__EXPERT_TIER__UP_LUT], v[NN__EXPERT_TIER__DOWN],
        v[NN__EXPERT_TIER__DOWN_LUT], v[NN__EXPERT_TIER__PARTS], v[NN__EXPERT_TIER__P_UP_LUT], v[NN__EXPERT_TIER__P_DOWN],
        v[NN__EXPERT_TIER__P_DOWN_LUT], v[NN__EXPERT_TIER__GATE_ROW], v[NN__EXPERT_TIER__DOWN_ROW], v[NN__EXPERT_TIER__LANDING],
        ex[0], ex[1], ex[2], ex[3], v[NN__EXPERT_TIER__SOURCES], v[NN__EXPERT_TIER__TOP_K], 1u, 0u, 0u, 0u };
    if (t.exclusive > 1u || t.duplex > 1u || (t.exclusive == 1u && t.layers == 0u)) return false;   /* exclusive needs its sources to move */
    if (t.experts == 0u || t.top_k == 0u || t.top_k > t.experts || t.top_k > NN__EXPERT__TIER_K_MAX || t.parts == 0u
     || 5u * t.parts + 1u > NN__EXPERT__GATHER_RUNS || t.inter % t.parts != 0u || t.down_row % t.parts != 0u
     || (sources != 0u && sys__node_array__length(sources) < t.experts * t.parts))
        return false;
    if (sys__node_array__length(entry.args[0]) >= NN__EXPERT_TIER__LENGTH_LINE) {
        const sys__heap_node n = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__LANDING_LINE);
        if (n.dtype != SYS__KIND__VALUE_INT) return false;
        t.landing_line = n.args[0];
    }
    if (sys__node_array__length(entry.args[0]) >= NN__EXPERT_TIER__LENGTH_GROUP) {
        const sys__heap_node n = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__RANKS);
        if (n.dtype != SYS__KIND__VALUE_INT || n.args[0] == 0u || n.args[0] > NN__EXPERT_TIER__RANKS_MAX
         || (uint64_t)sys__silicon__worker() >= n.args[0]) return false;   /* a group's tier: a card's */
        t.ranks = n.args[0];
    }
    if (sys__node_array__length(entry.args[0]) >= NN__EXPERT_TIER__LENGTH_SHARED) {
        const sys__heap_node n = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__SHARED);
        if (n.dtype != SYS__KIND__VALUE_INT || n.args[0] > t.experts) return false;
        t.shared = n.args[0];
    }
    if (sys__node_array__length(entry.args[0]) >= NN__EXPERT_TIER__LENGTH_STREAM) {
        const sys__heap_node s = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__STREAM);
        const sys__heap_node b = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__STREAM_BATCH);
        if (s.dtype != SYS__KIND__VALUE_INT || b.dtype != SYS__KIND__VALUE_INT
         || (s.args[0] != 0u && (b.args[0] == 0u || b.args[0] > NN__ROUTED__BATCH_MAX))) return false;
        t.stream = s.args[0]; t.stream_batch = b.args[0];
    }
    *tier = t;
    return true;
}

/* Part `p`'s runs, gathering it from `src` (the part's RAM slot) into a whole card slot — its gate rows, its up rows, their
 * scales, its columns of every down row and, with `scales`, every down row's scale (each part holds them whole). A part
 * that IS a card slot (one part, the same plan) is one run. Answers false when the runs would not fit. */
static __device__ inline bool nn__expert_tier__part_runs(const nn__expert_tier* t, uint64_t p, uint64_t src, bool scales,
                                                         nn__expert__gather* g) {
    const uint64_t P = t->parts, H = t->hidden, I = t->inter, PI = I / P, cols = t->down_row / P;
    if (P == 1u && t->p_up_lut == t->up_lut && t->p_down == t->down && t->p_down_lut == t->down_lut) {
        if (g->runs + 1u > NN__EXPERT__GATHER_RUNS) return false;
        const nn__expert__run whole = { src, 0u, 0u, 0u, t->down_lut + 2u * H, 1u };
        g->run[g->runs++] = whole;
        return true;
    }
    if (g->runs + 5u + (scales ? 1u : 0u) > NN__EXPERT__GATHER_RUNS) return false;
    const nn__expert__run runs[5] = {
        { src,                           0u,   p * PI * t->gate_row,                      0u,          PI * t->gate_row, 1u },
        { src + PI * t->gate_row,        0u,   I * t->gate_row + p * PI * t->gate_row,    0u,          PI * t->gate_row, 1u },
        { src + t->p_up_lut,             0u,   t->up_lut + p * PI * 2u,                   0u,          PI * 2u,          1u },
        { src + t->p_up_lut + PI * 2u,   0u,   t->up_lut + I * 2u + p * PI * 2u,          0u,          PI * 2u,          1u },
        { src + t->p_down,               cols, t->down + p * cols,                        t->down_row, cols,             H  },
    };
    for (unsigned r = 0u; r < 5u; ++r) g->run[g->runs++] = runs[r];
    if (scales) {
        const nn__expert__run sc = { src + t->p_down_lut, 0u, t->down_lut, 0u, H * 2u, 1u };
        g->run[g->runs++] = sc;
    }
    return true;
}

static __device__ inline bool nn__expert_tier__stitch(const nn__expert_tier* t, uint64_t x, nn__expert__gather* g) {
    if (t->sources == 0ull || x >= t->experts) return false;
    g->runs = 0u;
    for (uint64_t p = 0u; p < t->parts; ++p) {
        const sys__heap_node at = sys__node_array__borrow(t->sources, x * t->parts + p);
        if (at.dtype != SYS__KIND__VALUE_INT || at.args[0] == 0ull) return false;
        if (!nn__expert_tier__part_runs(t, p, at.args[0], p == 0u, g)) return false;
    }
    return true;
}

/* ══ ⭐⭐ THE EXCLUSIVE TIER — FREE, OUTGOING, DEMOTION (NN-48) ═════════════════════════════════════════════════════════
 * ⚖ *"we need to have an array of demotion candidates using the pointers, and the move is demotion candidates to recycled
 * landing slots … three arrays, demotion, outgoing, free"* · *"the evacuation channels stay held until they are written to
 * the ram destination slots, so when they are recycled if they are marked as pending the writer does wait"*.
 *   FREE       the card collection's free list: recycled slots, each one whose last expert has left for RAM
 *   OUTGOING   `outgoing` below: a promotion's arriving expert gathered into a FREE slot, and the band's demotion
 *              candidate — its least recently used expert not pinned and not already outgoing — read off the card into an
 *              evacuation channel while the card still computes it
 *   the commit at a layer's visit or a chunk's note, the CPUs idle: the arriving expert, landed, is the card's; the
 *              outgoing one leaves the card (its slot back to FREE) and takes, on every socket, the RAM slot the arriving
 *              one gave up (`nn__expert__hand_over`); the channel is written into them (WRITE jobs) and a CPU about to
 *              compute that expert waits for its write (`nn__expert__wait_written`). The tier's sources follow.
 * ⛳ every expert lives in one place, and is computable everywhere a reader looks for it: an arriving one stays in RAM until
 *   the commit, an outgoing one stays on the card until its bytes are safely in a channel. */
typedef struct nn__expert_tier__out {
    bool used;
    uint64_t owner, lx, x, lv, v;          /* who asked; the arriving expert and its layer; the outgoing one and its */
    unsigned channel;
    nn__loader__read* evacuate;
    uint64_t made;                         /* its gather's: the order its worker's commits take the swaps in */
    unsigned visits;                       /* its worker's visits since it was made (▶ COMMIT_VISITS) */
} nn__expert_tier__out;
static nn__expert_tier__out nn__expert_tier__zzprivate_out[NN__EXPERT_TIER__OUTGOING];

/* A gather queued, on a loader thread: the swap it is the arriving half of sends its outgoing expert where half duplex held
 * it — once the gather has landed (its `after`), the link free of this swap's other copy then — so it is on its way well
 * before the commit waits for it. Lock held. */
static void nn__expert_tier__zzprivate_gathered(const nn__loader__read* gather) {
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING; ++i) {
        nn__expert_tier__out* o = &nn__expert_tier__zzprivate_out[i];
        if (!o->used || o->owner != gather->owner || o->lx != gather->layer || o->x != gather->expert
         || o->evacuate->state != NN__LOADER__WAITING) continue;
        o->evacuate->after = gather->ticket;
        o->evacuate->state = NN__LOADER__QUEUED;
        o->evacuate->seq = ++nn__loader__zzprivate_seq;
        sys__os__condition_wake_all(&nn__loader__zzprivate_work);
    }
}

/* Every write held until the CPUs are done let go, of any worker: someone waits for a write's record or its bytes. Lock
 * held. */
static void nn__loader__zzprivate_let_writes_go(void) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__WAITING || r->job != NN__LOADER__JOB_WRITE) continue;
        r->state = NN__LOADER__QUEUED;
        r->seq = ++nn__loader__zzprivate_seq;
        sys__os__condition_wake_all(&nn__loader__zzprivate_work);
    }
}

/* How many more swaps worker `owner` may start: each swap parks its outgoing expert in a chunk of its card's pad's down
 * way until its commit, so a card has as many in flight as that way has chunks — a group's share of the parking, each
 * card its own, so what one card is refused never hangs on another's timing (▶ `expert_tier.tier`). Lock held. */
static uint64_t nn__expert_tier__zzprivate_swaps_left(uint64_t owner, const sys__pad* pad) {
    const uint64_t share = sys__pad__chunks(pad, SYS__PAD__DOWN);
    uint64_t mine = 0u;
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING; ++i)
        mine += nn__expert_tier__zzprivate_out[i].used && nn__expert_tier__zzprivate_out[i].owner == owner;
    return mine < share ? share - mine : 0u;
}

/* The records free now. Lock held. */
static unsigned nn__loader__zzprivate_free_records(void) {
    unsigned n = 0u;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) n += nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE;
    return n;
}

/* Nanoseconds since `a`. */
static inline uint64_t nn__loader__zzprivate_since(const struct timespec* a) {
    struct timespec b;
    sys__os__clock(&b);
    return (uint64_t)(b.tv_sec - a->tv_sec) * 1000000000ull + (uint64_t)(b.tv_nsec - a->tv_nsec);
}

/* A channel for an evacuation of `bytes`: a region of `pad`'s down way. ⛳ A CHANNEL STILL PENDING — written into RAM by
 *   some earlier commit — IS WAITED FOR, as ruled; one HELD (its expert not yet committed) cannot be, because the commit
 *   that frees it runs on this same thread later: with every chunk of the way held, the answer is none. Lock held. */
static int nn__loader__zzprivate_channel_take(sys__pad* pad, uint64_t bytes) {
    for (;;) {
        int pending = -1, free_channel = -1;
        for (unsigned c = 0u; c < NN__EXPERT_TIER__CHANNELS; ++c) {
            nn__loader__channel* ch = &nn__loader__zzprivate_channels[c];
            if (ch->held) continue;
            if (ch->pending != 0u) { if (ch->pad == pad) pending = (int)c; continue; }
            if (free_channel < 0) free_channel = (int)c;
        }
        if (free_channel >= 0) {
            nn__loader__channel* ch = &nn__loader__zzprivate_channels[free_channel];
            if (sys__pad__try_claim(pad, SYS__PAD__DOWN, bytes, &ch->hold) != 0) {
                ch->pad = pad;
                ch->held = true;
                return free_channel;
            }
        }
        if (pending < 0) { ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_CHANNEL_FULL]; return -1; }
        /* ⛳ A CHANNEL PENDING ON A HELD WRITE IS LET GO FIRST: the held writes wait for the program to say the CPUs are done,
         *   which cannot come while this visit waits here — the commit just before it may have held them (MEASURED: the
         *   9-layer test hung, 2026-10-05) */
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
            nn__loader__read* r = &nn__loader__zzprivate_reads[i];
            if (r->state == NN__LOADER__WAITING && r->job == NN__LOADER__JOB_WRITE && r->channel == (unsigned)pending) {
                r->state = NN__LOADER__QUEUED;
                r->seq = ++nn__loader__zzprivate_seq;
                sys__os__condition_wake_all(&nn__loader__zzprivate_work);
            }
        }
        struct timespec a;
        sys__os__clock(&a);
        nn__loader__zzprivate_wait();
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_CHANNEL_WAIT_NS] += nn__loader__zzprivate_since(&a);
    }
}

/* Whether expert `e` of band (`l`, `type`) is in a swap not yet committed — outgoing, or ARRIVING: an arriving expert
 * is resident on the card from its settle, but its RAM slots are still its own until the commit hands them on, so a
 * second swap taking it as its victim would evict it with nowhere to write it — and the first swap's commit would
 * then give its RAM slots away, leaving it in neither place (`REASONED`: the whole GLM's decode faulted at token 20
 * once the commit settled arrivals itself, which widened this window). Lock held. */
static bool nn__expert_tier__zzprivate_outgoing(uint64_t owner, uint64_t l, uint64_t e) {
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING; ++i) {
        const nn__expert_tier__out* o = &nn__expert_tier__zzprivate_out[i];
        if (o->used && o->owner == owner && ((o->lv == l && o->v == e) || (o->lx == l && o->x == e))) return true;
    }
    return false;
}

/* The demotion candidate: the least recently used expert of `layer`'s band while it holds its share, else of the band
 * holding the most — never one of `pinned` (the asking layer's picks), never one already outgoing, never a group's slice's
 * nor one its change waits on (▶ `…_kept`). Lock held. */
static bool nn__expert_tier__zzprivate_victim(uint64_t owner, uint64_t layer, uint64_t type, const uint64_t* pinned, unsigned npinned,
                                              uint64_t* vl, uint64_t* v) {
    const uint64_t layers = nn__expert__index_layers();
    uint64_t banded = 0ull, most = layers, held = 0ull;
    for (uint64_t l = 0ull; l < layers; ++l) {
        const uint64_t n = nn__expert__layer_resident(l, type);
        banded += nn__expert__layer_experts(l, type) != 0ull ? 1ull : 0ull;
        if (n > held) { held = n; most = l; }
    }
    const uint64_t share = banded == 0ull ? 0ull : nn__expert__type_live(type) / banded;
    const uint64_t order[2] = { (share != 0ull && nn__expert__layer_resident(layer, type) >= share) ? layer : most, layer };
    for (unsigned k = 0u; k < 2u; ++k) {
        const uint64_t l = order[k];
        if (l >= layers) continue;
        uint64_t e = 0ull;
        if (!nn__expert__oldest(l, type, &e)) continue;
        for (unsigned step = 0u; step < 64u; ++step) {
            bool skip = nn__expert_tier__zzprivate_outgoing(owner, l, e) || nn__expert_tier__zzprivate_kept(l, e);
            for (unsigned p = 0u; !skip && l == layer && p < npinned; ++p) skip = pinned[p] == e;
            if (!skip) { *vl = l; *v = e; return true; }
            sys__heap_node me;
            if (!nn__expert__slot(l, type, e, &me) || me.args[NN__EXPERT__SLOT_PREV] == NN__EXPERT__NONE) break;
            e = NN__EXPERT__UNLINK(me.args[NN__EXPERT__SLOT_PREV]);
        }
    }
    return false;
}

/* A free read record, or none. Lock held. */
static nn__loader__read* nn__loader__zzprivate_free_record(void) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i)
        if (nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE) return &nn__loader__zzprivate_reads[i];
    return 0;
}

/* Whether a WRITE of expert `e` (band `l`, `type`) is still on its way, for any worker. Lock held. */
static bool nn__loader__zzprivate_writing(uint64_t l, uint64_t type, uint64_t e) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        const nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__FREE && r->job == NN__LOADER__JOB_WRITE && r->layer == l && r->type == type && r->expert == e)
            return true;
    }
    return false;
}

/* ⭐ A PROMOTION INTO AN EXCLUSIVE TIER: `x` (in RAM) gathered into a FREE card slot, and a demotion candidate sent
 * OUTGOING beside it. RESIDENT / ARRIVING / REFUSED, as `nn__expert__promote`; REFUSED leaves everything as it was. */
static __device__ inline int nn__expert_tier__promote(sys__silicon_family__id family, const nn__expert_tier* t, uint64_t x,
                                                      const uint64_t* pinned, unsigned npinned) {
    const uint64_t layer = t->layer, type = t->type, owner = nn__expert__zzpackage_owner(), bytes = nn__expert__type_bytes(type);
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, x, &me)) return NN__EXPERT__REFUSED;
    if (me.args[NN__EXPERT__SLOT_AT] != 0ull) return NN__EXPERT__RESIDENT;
    nn__expert__gather g;
    if (bytes == 0ull || !nn__expert_tier__stitch(t, x, &g)) return NN__EXPERT__REFUSED;     /* not in RAM: none to gather */
    const bool in_place = nn__loader__zzprivate_in_place(family, &g);
    sys__pad* pad = sys__pad__of(family, sys__silicon__bound_device(family));
    if (pad == 0) return NN__EXPERT__REFUSED;                                                /* no pad: no promotion */
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    /* ⛳ THE WAITS FIRST, AND EACH LETS THE LOCK GO — so nothing is reserved until the last is behind, and what they wait for
     *   is asked again after each: a channel (it may wait for a pending one); an expert still being written into RAM —
     *   demoted a moment ago — gathered back once its bytes are there; and the swap's two records, where writes hold them.
     *   ⛔ WAITED FOR, NOT REFUSED: a write frees itself when it lands, so a refusal would make which promotions start the
     *   writes' timing (⚖ one config, one answer) */
    int channel = -1;
    if (nn__expert_tier__zzprivate_swaps_left(owner, pad) == 0u) { sys__os__lock_give(&nn__loader__zzprivate_lock); return NN__EXPERT__REFUSED; }
    for (;;) {
        if (nn__loader__zzprivate_find(owner, layer, type, x) != 0) {
            if (channel >= 0) {
                nn__loader__zzprivate_channels[channel].held = false;
                nn__loader__zzprivate_channel_settle(&nn__loader__zzprivate_channels[channel]);
            }
            sys__os__lock_give(&nn__loader__zzprivate_lock);
            return NN__EXPERT__ARRIVING;
        }
        if (channel < 0) {
            channel = nn__loader__zzprivate_channel_take(pad, bytes);
            if (channel < 0) { sys__os__lock_give(&nn__loader__zzprivate_lock); return NN__EXPERT__REFUSED; }
            continue;
        }
        const bool writes = __atomic_load_n(&nn__loader__zzprivate_writes_out, __ATOMIC_ACQUIRE) != 0ull;
        if (nn__loader__zzprivate_writing(layer, t->part_type, x) || (writes && nn__loader__zzprivate_free_records() < 2u)) {
            nn__loader__zzprivate_let_writes_go();
            nn__loader__zzprivate_wait();
            continue;
        }
        break;
    }
    nn__loader__channel* ch = &nn__loader__zzprivate_channels[channel];
    nn__expert_tier__out* o = 0;
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING && o == 0; ++i)
        if (!nn__expert_tier__zzprivate_out[i].used) o = &nn__expert_tier__zzprivate_out[i];
    uint64_t vl = 0ull, v = 0ull, slot = 0ull, room = 0ull;
    nn__loader__read *rg = 0, *re = 0;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX && re == 0; ++i)
        if (nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE) { if (rg == 0) rg = &nn__loader__zzprivate_reads[i]; else re = &nn__loader__zzprivate_reads[i]; }
    /* ⚖ *"if so we can launch the read and write together, if not we need to send to vram first and receive to ram later"*:
     *   half duplex reserves the evacuation's record now and launches it at the commit after the gather lands */
    sys__heap_node vm;
    if (o == 0 || re == 0 || !nn__expert_tier__zzprivate_victim(owner, layer, type, pinned, npinned, &vl, &v)
     || !nn__expert__slot(vl, type, v, &vm) || vm.args[NN__EXPERT__SLOT_AT] == 0ull || !nn__loader__zzprivate_start()
     || !nn__expert__type_take(type, &slot, &room)) {
        ch->held = false;
        nn__loader__zzprivate_channel_settle(ch);
        sys__os__lock_give(&nn__loader__zzprivate_lock);
        return NN__EXPERT__REFUSED;                                   /* no room in flight, no candidate, no FREE slot */
    }
    rg->state = NN__LOADER__QUEUED;
    rg->predicted = false; rg->owner = owner; rg->family = family;
    rg->layer = layer; rg->type = type; rg->expert = x; rg->slot = slot; rg->room = room;
    rg->gather = true; rg->bytes = bytes; rg->runs = g; rg->job = NN__LOADER__JOB_GATHER;
    rg->device = sys__silicon__bound_device(family); rg->in_place = in_place;
    rg->queued_ns = nn__loader__zzprivate_now_ns();
    rg->made = rg->seq = ++nn__loader__zzprivate_seq;
    re->state = t->duplex == 1u ? NN__LOADER__QUEUED : NN__LOADER__WAITING;
    re->predicted = false; re->owner = owner; re->family = family;
    re->layer = vl; re->type = type; re->expert = v; re->slot = 0ull; re->room = vm.args[NN__EXPERT__SLOT_AT];
    re->gather = false; re->bytes = bytes; re->job = NN__LOADER__JOB_EVACUATE; re->channel = (unsigned)channel;
    re->after.way = SYS__PAD__UP; re->after.number = 0ull;
    re->seq = ++nn__loader__zzprivate_seq;
    o->used = true; o->owner = owner; o->lx = layer; o->x = x; o->lv = vl; o->v = v; o->channel = (unsigned)channel; o->evacuate = re;
    o->made = rg->made; o->visits = 0u;
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PROMOTED];
    sys__os__condition_wake_all(&nn__loader__zzprivate_work);
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

/* Every WRITE that worker `by`'s commits held back, let go — the threads take them from here. Lock held. */
static void nn__loader__zzprivate_release_writes(uint64_t by) {
    bool any = false;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__WAITING || (r->job != NN__LOADER__JOB_WRITE && r->job != NN__LOADER__JOB_GATHER) || r->by != by)
            continue;
        r->state = NN__LOADER__QUEUED;
        r->seq = ++nn__loader__zzprivate_seq;
        any = true;
    }
    if (any) sys__os__condition_wake_all(&nn__loader__zzprivate_work);
}

/* One cell of the tier's sources — part `p` of expert `e` of layer `l` — set to `at` (0: on the card). */
static bool nn__expert_tier__zzprivate_source(const nn__expert_tier* t, uint64_t l, uint64_t e, uint64_t p, uint64_t at) {
    if (l >= sys__node_array__length(t->layers)) return false;
    const sys__heap_node arr = sys__node_array__borrow(t->layers, l);
    if (!sys__heap_node__carries_reference(arr.dtype) || !sys__node_array__is(arr.args[0])) return false;
    sys__heap_node cell = sys__heap_node__nothing();
    cell.dtype = SYS__KIND__VALUE_INT; cell.args[0] = at;
    return sys__node_array__set(arr.args[0], e * t->parts + p, &cell);
}

/* ⭐ THE COMMIT — every swap of this worker that is due: its arriving expert admitted, its outgoing one leaving the card
 * for the arriving one's RAM slots, the channel written into them. Called at a layer's visit and a chunk's note, before
 * the CPUs are handed anything, so no CPU is reading a slot this changes. A swap whose gather failed is undone: the
 * outgoing expert stays on the card. False on a step that could not be done.
 * ⭐ A SWAP IS DUE AT A SET VISIT, ITS COPIES WAITED FOR THERE IF LATE: its arriving layer's next visit, or its worker's
 *   COMMIT_VISITS-th since it was made, whichever comes first. The visit an expert leaves the card at decides whether the
 *   card or the CPUs compute it, and the two round apart — so committing the swaps whose copies happen to have landed
 *   would make the answer the copies' timing (⚖ one config, one answer). A visit takes a few ms, an expert's copy one or
 *   two. And the due ones in the order they were made, as `settle` admits a worker's reads: the order the indices learn
 *   of them is the order their LRU chains are in. */
static __device__ inline bool nn__expert_tier__commit(sys__silicon_family__id family, const nn__expert_tier* t) {
    const uint64_t owner = nn__expert__zzpackage_owner(), type = t->type, P = t->parts;
    const int acting = sys__silicon__host_acting_as();
    bool ok = true;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    /* ⛳ writes still held from an earlier commit go now — a program that never says its CPUs are done must not leave
     *   their channels pending for ever */
    nn__loader__zzprivate_release_writes(owner);
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING; ++i) {
        nn__expert_tier__out* o = &nn__expert_tier__zzprivate_out[i];
        if (o->used && o->owner == owner) ++o->visits;
    }
    for (;;) {
        /* the due swap made first — asked again after every wait, which lets the lock go */
        nn__expert_tier__out* o = 0;
        for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING; ++i) {
            nn__expert_tier__out* c = &nn__expert_tier__zzprivate_out[i];
            if (!c->used || c->owner != owner || (c->lx != t->layer && c->visits < NN__EXPERT_TIER__COMMIT_VISITS)) continue;
            if (o == 0 || c->made < o->made) o = c;
        }
        if (o == 0) break;
        nn__loader__read* rx = nn__loader__zzprivate_find(owner, o->lx, type, o->x);
        if (rx != 0) {
            /* ⛳ ITS GATHER SETTLED HERE, WHATEVER LAYER THIS VISIT IS FOR: left to its own layer's visit it waits a token,
             *   holding its channel all that time — MEASURED, 16 channels then carried 15.3 swaps a token, 60 more a
             *   token were refused for want of one, and the card waited 19 ms a token for one to come free. The commit
             *   runs on the card's thread between layers, so no one is reading that layer's index. */
            nn__loader__zzprivate_let_go(rx);
            if (rx->state != NN__LOADER__DONE && rx->state != NN__LOADER__FAILED) { nn__loader__zzprivate_wait(); continue; }
            /* ⛳ LANDED BEFORE ITS RAM SLOT IS HANDED ON: a copy from registered RAM reads the slot the outgoing expert is
             *   about to be written into — waited for here, the lock let go meanwhile (only this worker settles it) */
            if (rx->state == NN__LOADER__DONE && rx->ticket.number != 0ull) {
                sys__pad* pad = sys__pad__of(rx->family, rx->device);
                const sys__pad__ticket ticket = rx->ticket;
                sys__os__lock_give(&nn__loader__zzprivate_lock);
                const bool reached = sys__pad__landed(pad, ticket);
                sys__os__lock_take(&nn__loader__zzprivate_lock);
                if (!reached) rx->state = NN__LOADER__FAILED;
                rx->ticket.number = 0ull;
            }
            const bool got = rx->state == NN__LOADER__DONE;
            const uint64_t slot = rx->slot, room = rx->room;
            if (got) nn__loader__zzprivate_landed(rx);
            rx->state = NN__LOADER__FREE;
            if (!got || !nn__expert__admit(o->lx, o->x, room, type)) (void)nn__expert__type_give(type, slot);
        }
        nn__loader__read* re = o->evacuate;
        if (re->state == NN__LOADER__WAITING) {         /* half duplex, its gather landed (▶ `…_gathered`, which sends it) */
            sys__heap_node lm;
            if (nn__expert__slot(o->lx, type, o->x, &lm) && lm.args[NN__EXPERT__SLOT_AT] != 0ull) {
                re->state = NN__LOADER__QUEUED;
                re->seq = ++nn__loader__zzprivate_seq;
                sys__os__condition_wake_all(&nn__loader__zzprivate_work);
            } else {
                re->state = NN__LOADER__FAILED;                      /* the gather failed: undone below */
            }
        }
        if (re->state == NN__LOADER__QUEUED || re->state == NN__LOADER__READING) { nn__loader__zzprivate_wait(); continue; }
        /* ⛳ A RECORD FOR EACH PART'S WRITE, WAITED FOR WHILE WRITES HOLD THEM — a write frees its own when it lands */
        if (re->state == NN__LOADER__DONE && nn__loader__zzprivate_free_records() < P
         && __atomic_load_n(&nn__loader__zzprivate_writes_out, __ATOMIC_ACQUIRE) != 0ull) {
            nn__loader__zzprivate_let_writes_go();
            nn__loader__zzprivate_wait();
            continue;
        }
        sys__heap_node xm;
        const bool landed = re->state == NN__LOADER__DONE && nn__expert__slot(o->lx, type, o->x, &xm) && xm.args[NN__EXPERT__SLOT_AT] != 0ull;
        re->state = NN__LOADER__FREE;
        nn__loader__channel* ch = &nn__loader__zzprivate_channels[o->channel];
        o->used = false;
        if (!landed) { ch->held = false; nn__loader__zzprivate_channel_settle(ch); continue; }   /* undone: v stays */
        /* the card: the outgoing expert leaves, its slot back to FREE */
        if (!nn__expert__evict(o->lv, type, o->v)) { ch->held = false; nn__loader__zzprivate_channel_settle(ch); ok = false; continue; }
        /* the sockets: the arriving expert's slot handed to the outgoing one, and the channel written into it */
        ch->pending = 0u;
        for (uint64_t p = 0u; p < P; ++p) {
            sys__silicon__host_act_as((int)(t->part_worker + p));
            sys__heap_node pm;
            nn__expert__gather w;
            w.runs = 0u;
            const bool have = nn__expert__slot(o->lx, t->part_type, o->x, &pm) && pm.args[NN__EXPERT__SLOT_AT] != 0ull;
            const uint64_t at = have ? pm.args[NN__EXPERT__SLOT_AT] : 0ull;
            nn__loader__read* rw = nn__loader__zzprivate_free_record();
            /* the write's runs: the gather's for this part, turned round — out of the channel, in the card's layout, into the
             *   part's RAM slot */
            nn__expert__gather gp;
            gp.runs = 0u;
            if (!have || rw == 0 || !nn__expert_tier__part_runs(t, p, at, true, &gp)
             || !nn__expert__hand_over(o->lx, o->x, o->lv, o->v, t->part_type)) { ok = false; continue; }
            for (unsigned r = 0u; r < gp.runs; ++r) {
                const nn__expert__run u = gp.run[r];
                const nn__expert__run back = { (uint64_t)(uintptr_t)sys__pad__at(ch->pad, &ch->hold) + u.into, u.into_stride, u.from,
                                               u.from_stride, u.bytes, u.count };
                w.run[w.runs++] = back;
            }
            const uint64_t writer = nn__expert__zzpackage_owner();
            /* ⚖ *"mark the write to ram after the cpu compute is done (it is the whole point of the half duplex path)"*:
             *   half duplex holds the write until the program says the CPUs are done (▶ `nn__expert_tier__written`), so it
             *   runs while the card computes attention and not against the CPUs' experts for the memory; full duplex
             *   writes at once — which of the two serves best there is the EPYC's to measure */
            rw->state = t->duplex == 1u ? NN__LOADER__QUEUED : NN__LOADER__WAITING;
            rw->by = owner;
            rw->predicted = false; rw->owner = writer; rw->family = family;
            rw->layer = o->lv; rw->type = t->part_type; rw->expert = o->v; rw->slot = 0ull; rw->room = at;
            rw->gather = false; rw->bytes = 0ull; rw->runs = w; rw->job = NN__LOADER__JOB_WRITE; rw->channel = o->channel;
            rw->seq = ++nn__loader__zzprivate_seq;
            __atomic_add_fetch(&nn__loader__zzprivate_writes_out, 1ull, __ATOMIC_RELEASE);
            ++ch->pending;
            ok = nn__expert_tier__zzprivate_source(t, o->lx, o->x, p, 0ull) && ok;
            ok = nn__expert_tier__zzprivate_source(t, o->lv, o->v, p, at) && ok;
        }
        sys__silicon__host_act_as(acting);
        ch->held = false;
        nn__loader__zzprivate_channel_settle(ch);
        ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_DEMOTED];
        sys__os__condition_wake_all(&nn__loader__zzprivate_work);
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    return ok;
}

/* ⭐ A CPU ABOUT TO COMPUTE expert `e` of band (`layer`, `type`) waits here for its write, if one is still on its way.
 * ⛳ A WRITE STILL HELD IS LET GO HERE: it is held until the CPUs are done, so a CPU that needs it now would otherwise wait
 *   for a release that comes only after it. */
static __device__ inline void nn__expert__wait_written(uint64_t layer, uint64_t type, uint64_t e) {
    /* ⛳ NO WRITE ON ITS WAY, NO LOCK: a verb asks this once a pick, and the lock is the loader threads' too, busy with a
     *   tier's promotions */
    if (__atomic_load_n(&nn__loader__zzprivate_writes_out, __ATOMIC_ACQUIRE) == 0ull) return;
    const uint64_t owner = nn__expert__zzpackage_owner();
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    for (;;) {
        bool writing = false;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX && !writing; ++i) {
            nn__loader__read* r = &nn__loader__zzprivate_reads[i];
            writing = r->state != NN__LOADER__FREE && r->job == NN__LOADER__JOB_WRITE && r->owner == owner && r->layer == layer
                   && r->type == type && r->expert == e;
            if (writing && r->state == NN__LOADER__WAITING) {
                r->state = NN__LOADER__QUEUED;
                r->seq = ++nn__loader__zzprivate_seq;
                sys__os__condition_wake_all(&nn__loader__zzprivate_work);
            }
        }
        if (!writing) break;
        struct timespec a;
        sys__os__clock(&a);
        nn__loader__zzprivate_wait();
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_WRITTEN_WAIT_NS] += nn__loader__zzprivate_since(&a);
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
}

/* ⭐ THE CPUS ARE DONE: the writes this worker's commits held back go now, while the card computes what comes next. */
static __device__ inline void nn__expert_tier__written(void) {
    const uint64_t owner = nn__expert__zzpackage_owner();
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    nn__loader__zzprivate_release_writes(owner);
    sys__os__lock_give(&nn__loader__zzprivate_lock);
}

static __device__ inline bool nn__expert_tier__visit(sys__silicon_family__id family, const nn__expert_tier* t, const uint64_t* ids,
                                                     bool* held) {
    const uint64_t layer = t->layer, type = t->type;
    const unsigned k = (unsigned)t->top_k;
    const bool fed = t->sources != 0ull;
    /* ⭐ A PROMOTION COUNTS AT ITS LAYER'S NEXT VISIT, LANDED OR WAITED FOR: every one this worker started on the layer —
     *   at the visit before — is admitted here, so which experts a card holds at a visit is the same on every boot (⚖ one
     *   config, one answer); admitting those landed by then would make it the copies' timing. A layer's next visit is a
     *   position later — ~250 ms in a whole model, where a copy lands in ~12 ms. */
    if (fed) (void)nn__expert__settle(layer, type);
    if (t->exclusive == 1u && !nn__expert_tier__commit(family, t)) return false;   /* the swaps that are ready: ▶ commit */
    const uint64_t now = nn__expert__tick();
    uint64_t pinned[NN__EXPERT__TIER_K_MAX];
    unsigned n_pinned = 0u;
    unsigned hits = 0u;
    for (unsigned j = 0u; j < k; ++j) {
        sys__heap_node me;
        if (!nn__expert__slot(layer, type, ids[j], &me)) return false;
        held[j] = me.args[NN__EXPERT__SLOT_AT] != 0ull;
        if (held[j]) { (void)nn__expert__touch(layer, type, ids[j]); pinned[n_pinned++] = ids[j]; ++hits; }
    }
    {
        const uint64_t owner = nn__expert__zzpackage_owner();
        sys__os__lock_take(&nn__loader__zzprivate_lock);
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_PICKS] += k;
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_HITS] += hits;
        for (unsigned j = 0u; j < k; ++j)
            if (!held[j] && nn__loader__zzprivate_find(owner, layer, type, ids[j]) != 0)
                ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PICKED_ARRIVING];
        sys__os__lock_give(&nn__loader__zzprivate_lock);
    }
    /* ⚖ 2026-10-06 — HOW MANY TO COPY IS A FUNCTION OF HOW LONG THE CPUS TAKE: above the line of hits one, at or below it
     *   the tier's `landing` — few picks on the card leave the CPUs a long share, and two copies hide behind it.
     * ⚖ AND THE BEST OF THEM, NOT THE FIRST: the qualifying picks in the order of their score (the older of their last two
     *   calls, the most recent first), so a visit that may start one starts the strongest. Each asked before its call
     *   is recorded: the call `qualifies` reads is the one before this. */
    const unsigned land = hits > t->landing_line ? 1u : (unsigned)t->landing;
    unsigned cand[NN__EXPERT__TIER_K_MAX];
    uint64_t score[NN__EXPERT__TIER_K_MAX];
    unsigned n_cand = 0u;
    for (unsigned j = 0u; j < k && fed && land > 0u; ++j) {
        if (held[j] || !nn__expert__qualifies(layer, type, ids[j])) continue;
        uint64_t s = 0ull;
        (void)nn__expert__zzpackage_older(layer, type, ids[j], &s);    /* 0 while the tier fills first-come */
        unsigned at = n_cand++;
        while (at > 0u && score[at - 1u] < s) { cand[at] = cand[at - 1u]; score[at] = score[at - 1u]; --at; }
        cand[at] = j; score[at] = s;
    }
    unsigned started = 0u;
    for (unsigned c = 0u; c < n_cand && started < land; ++c) {
        const unsigned j = cand[c];
        nn__expert__gather g;
        if (t->exclusive == 1u ? nn__expert_tier__promote(family, t, ids[j], pinned, n_pinned) == NN__EXPERT__ARRIVING
                               : (nn__expert_tier__stitch(t, ids[j], &g)
                                  && nn__expert__promote(family, layer, type, ids[j], &g, pinned, n_pinned) == NN__EXPERT__ARRIVING))
            ++started;
    }
    for (unsigned j = 0u; j < k; ++j) (void)nn__expert__called(layer, type, ids[j], now);
    return true;
}

/* ══ ⭐ A POSITION'S PICKS SHARED OUT AMONG A GROUP'S CARDS (NN-99 ②) ══════════════════════════════════════════════════
 * ⚖ 2026-10-20: *"we see which experts are not on the shared slice and map them each to their gpu, then we see which gpu
 * has less experts and give it enough experts so it computes the same amount as the others. then we compute the ram
 * experts and add it all up"*. Every card visits its own tier — each reads only its own index — then posts its picks
 * and which of them it holds on the board, waits for the others' posts of the same call, and makes the assignment every
 * card makes from the same posts. ⛳ The board is one, for one group: its cards call it layer by layer in the same order.
 * Two calls' posts alternate, so a card a call ahead never writes over the one a slower card still reads — it cannot be
 * two ahead, as it waits for the slower card's post of the call between. */
typedef struct nn__expert_tier__board {
    uint64_t gen[NN__EXPERT_TIER__RANKS_MAX];                                     /* a card's last call posted */
    uint64_t calls[NN__EXPERT_TIER__RANKS_MAX];                                   /* its own count, its thread's */
    uint64_t ids[2][NN__EXPERT_TIER__RANKS_MAX][NN__EXPERT__TIER_K_MAX];
    uint8_t  held[2][NN__EXPERT_TIER__RANKS_MAX][NN__EXPERT__TIER_K_MAX];
    uint64_t promoted[NN__EXPERT_TIER__RANKS_MAX][NN__EXPERT_TIER__RANKS_MAX];    /* each card's count of every card's */
    uint64_t computed[NN__EXPERT_TIER__RANKS_MAX][NN__EXPERT_TIER__RANKS_MAX];    /* and of every card's picks computed */
} nn__expert_tier__board;
static nn__expert_tier__board nn__expert_tier__zzprivate_board;

/* ⭐ A CARD'S VIEW OF EVERY LAYER'S SHARED SLICE (▶ SHARED) — each card keeps its own, made from the same shared picks and
 * the same counts in the same order, so every card's is the same; its own thread alone writes it. */
#define NN__EXPERT_TIER__NO_KEEPER 0xFFu
/* An exclusive group's slice change, made at a choice and switched at its layer's next visit (▶ `…_change`, `…_switch`):
 * `x` joins from card `a`, `s` leaves for card `d`, and where d is not a, `e` goes from d to a (else `e` is `experts`).
 * x's bytes wait in stage `sx` for change `gx`, e's in `se` for `ge`; `up` is this card's write of what it takes, into
 * free slot `slot` — or none. */
typedef struct nn__expert_tier__change {
    uint64_t x, s, e, gx, ge, slot;
    uint32_t a, d;
    unsigned sx, se;
    nn__loader__read* up;
} nn__expert_tier__change;
typedef struct nn__expert_tier__slice {
    uint64_t layers, experts;
    uint32_t* count;                     /* [layer · experts + e]: picks, halved at every choice */
    uint64_t* visits;                    /* [layer] */
    uint8_t* in;                         /* 1: in the layer's slice */
    uint8_t* keeper;                     /* the card an expert held outside the slice is computed by, or NO_KEEPER */
    /* an exclusive group's: whether a layer's slice is known yet (▶ `…_slice_ready`), its changes waiting for their
     *   switch, and the changes and stages handed out so far — the same on every card */
    uint8_t* known;                      /* [layer] */
    uint8_t* changing;                   /* [layer]: how many of `change` wait */
    nn__expert_tier__change* change;     /* [layer · SLICE_PAIRS + i] */
    uint64_t changes, stages;
} nn__expert_tier__slice;
static nn__expert_tier__slice nn__expert_tier__zzprivate_slices[NN__EXPERT_TIER__RANKS_MAX];

static nn__expert_tier__slice* nn__expert_tier__zzprivate_slice(uint32_t me, uint64_t experts) {
    nn__expert_tier__slice* v = &nn__expert_tier__zzprivate_slices[me];
    if (v->count != 0) return v->experts == experts ? v : 0;
    const uint64_t layers = nn__expert__index_layers(), n = layers * experts;
    if (layers == 0ull) return 0;
    v->count = (uint32_t*)calloc(n, sizeof(uint32_t));
    v->visits = (uint64_t*)calloc(layers, sizeof(uint64_t));
    v->in = (uint8_t*)calloc(n, 1u);
    v->keeper = (uint8_t*)malloc(n);
    v->known = (uint8_t*)calloc(layers, 1u);
    v->changing = (uint8_t*)calloc(layers, 1u);
    v->change = (nn__expert_tier__change*)calloc(layers * NN__EXPERT_TIER__SLICE_PAIRS, sizeof(nn__expert_tier__change));
    if (v->count == 0 || v->visits == 0 || v->in == 0 || v->keeper == 0 || v->known == 0 || v->changing == 0 || v->change == 0) {
        free(v->count); free(v->visits); free(v->in); free(v->keeper); free(v->known); free(v->changing); free(v->change);
        v->count = 0; return 0;
    }
    memset(v->keeper, NN__EXPERT_TIER__NO_KEEPER, n);
    v->layers = layers; v->experts = experts;
    v->changes = 0ull; v->stages = 0ull;
    return v;
}

/* Every card's view forgotten — the package's teardown: a machine booted after this one in the same process starts its
 * slices from its own deal, not from this one's (▶ `nn__loader__zzpackage_stop`). */
static void nn__expert_tier__zzprivate_forget(void) {
    for (unsigned r = 0u; r < NN__EXPERT_TIER__RANKS_MAX; ++r) {
        nn__expert_tier__slice* v = &nn__expert_tier__zzprivate_slices[r];
        free(v->count); free(v->visits); free(v->in); free(v->keeper); free(v->known); free(v->changing); free(v->change);
        const nn__expert_tier__slice none = {};
        *v = none;
    }
}

/* Whether expert `e` of layer `l` is the slice's or waits in a slice change of this card's view — an exclusive tier's
 * promotion never sends either back (▶ `…_victim`). Lock held, or the card's own thread. */
static bool nn__expert_tier__zzprivate_kept(uint64_t l, uint64_t e) {
    const uint32_t me = (uint32_t)sys__silicon__worker();
    if (me >= NN__EXPERT_TIER__RANKS_MAX) return false;
    const nn__expert_tier__slice* v = &nn__expert_tier__zzprivate_slices[me];
    if (v->count == 0 || l >= v->layers || e >= v->experts) return false;
    if (v->in[l * v->experts + e]) return true;
    for (unsigned i = 0u; i < v->changing[l]; ++i) {
        const nn__expert_tier__change* c = &v->change[l * NN__EXPERT_TIER__SLICE_PAIRS + i];
        if (c->x == e || c->e == e) return true;
    }
    return false;
}

/* The layer's slice chosen again: its `shared` most picked (the lowest numbered at a tie), the counts halved. An expert
 * leaving it is kept by the card with the fewest picks computed so far, the lowest numbered at a tie. */
static void nn__expert_tier__zzprivate_choose(nn__expert_tier__slice* v, uint64_t layer, uint64_t shared, uint32_t ranks,
                                              const uint64_t* computed) {
    const uint64_t E = v->experts;
    uint32_t* c = v->count + layer * E;
    uint8_t* in = v->in + layer * E;
    uint8_t* keeper = v->keeper + layer * E;
    uint32_t least = 0u;
    for (uint32_t r = 1u; r < ranks; ++r) if (computed[r] < computed[least]) least = r;
    uint8_t* was = (uint8_t*)malloc(E);
    if (was == 0) return;
    memcpy(was, in, E);
    memset(in, 0, E);
    for (uint64_t taken = 0u; taken < shared; ++taken) {
        uint64_t best = E;
        for (uint64_t e = 0u; e < E; ++e)
            if (!in[e] && c[e] != 0u && (best == E || c[e] > c[best])) best = e;
        if (best == E) break;
        in[best] = 1u;
    }
    for (uint64_t e = 0u; e < E; ++e) {
        if (was[e] && !in[e]) keeper[e] = (uint8_t)least;
        if (in[e]) keeper[e] = NN__EXPERT_TIER__NO_KEEPER;
        c[e] >>= 1;
    }
    free(was);
}

/* Every other card's post of `call` waited for on a board's `gen`; false when one never comes. */
static inline bool nn__expert_tier__zzprivate_posted(const uint64_t* gen, uint32_t me, uint32_t ranks, uint64_t call) {
    for (uint32_t r = 0u; r < ranks; ++r) {
        if (r == me) continue;
        if (__atomic_load_n(&gen[r], __ATOMIC_ACQUIRE) < call) {
            struct timespec t0, t;
            sys__os__clock(&t0);
            for (uint32_t spin = 1u; __atomic_load_n(&gen[r], __ATOMIC_ACQUIRE) < call; ++spin) {
                if (spin < 256u) continue;
                sys__os__yield();
                if ((spin & 1023u) != 0u) continue;
                sys__os__clock(&t);
                if ((uint64_t)(t.tv_sec - t0.tv_sec) > (uint64_t)NN__EXPERT_TIER__PATIENCE) return false;
            }
        }
    }
    return true;
}

/* Card `me` of `ranks` posts its `k` picks and which it holds, and waits for every other card's post of the same call;
 * false when one never comes, or routed the position otherwise — the cards would sum different experts. `ids` and
 * `held` come back every card's, card by card. */
static inline bool nn__expert_tier__zzprivate_share(uint32_t me, uint32_t ranks, unsigned k, const uint64_t* ids,
                                                   const bool* held, uint8_t all_held[][NN__EXPERT__TIER_K_MAX]) {
    nn__expert_tier__board* b = &nn__expert_tier__zzprivate_board;
    const uint64_t call = ++b->calls[me], at = call & 1u;
    for (unsigned j = 0u; j < k; ++j) { b->ids[at][me][j] = ids[j]; b->held[at][me][j] = held[j] ? 1u : 0u; }
    __atomic_store_n(&b->gen[me], call, __ATOMIC_RELEASE);
    if (!nn__expert_tier__zzprivate_posted(b->gen, me, ranks, call)) return false;
    for (uint32_t r = 0u; r < ranks; ++r)
        for (unsigned j = 0u; j < k; ++j) {
            if (b->ids[at][r][j] != ids[j]) return false;
            all_held[r][j] = b->held[at][r][j];
        }
    return true;
}

/* ⭐ A CHUNK'S HOLDINGS SHARED OUT AMONG A GROUP'S EXCLUSIVE TIERS (NN-99): every card posts which of the layer's
 * experts it holds — after its commit, so every card's post is the layout the chunk runs on — and reads the others'. Its
 * cards call it layer by layer in the same order, and its posts alternate as the board's do. */
#define NN__EXPERT_TIER__CHUNK_WORDS 16u   /* a layer's experts a chunk's post holds, 64 a word */
typedef struct nn__expert_tier__chunk_board {
    uint64_t gen[NN__EXPERT_TIER__RANKS_MAX];
    uint64_t calls[NN__EXPERT_TIER__RANKS_MAX];
    uint64_t held[2][NN__EXPERT_TIER__RANKS_MAX][NN__EXPERT_TIER__CHUNK_WORDS];
} nn__expert_tier__chunk_board;
static nn__expert_tier__chunk_board nn__expert_tier__zzprivate_chunk_board;

/* Card `me` posts `mine` (1: it holds that expert) and waits for every other card's post of the same call: `other` comes
 * back 1 where another card holds the expert. False when a card never posts, or the layer has more experts than a post
 * holds. */
static inline bool nn__expert_tier__zzprivate_share_held(uint32_t me, uint32_t ranks, uint64_t experts, const uint8_t* mine,
                                                        uint8_t* other) {
    if (experts > 64u * NN__EXPERT_TIER__CHUNK_WORDS) return false;
    nn__expert_tier__chunk_board* b = &nn__expert_tier__zzprivate_chunk_board;
    const uint64_t call = ++b->calls[me], at = call & 1u;
    for (unsigned w = 0u; w < NN__EXPERT_TIER__CHUNK_WORDS; ++w) b->held[at][me][w] = 0ull;
    for (uint64_t e = 0u; e < experts; ++e) if (mine[e]) b->held[at][me][e >> 6] |= 1ull << (e & 63u);
    __atomic_store_n(&b->gen[me], call, __ATOMIC_RELEASE);
    if (!nn__expert_tier__zzprivate_posted(b->gen, me, ranks, call)) return false;
    for (uint64_t e = 0u; e < experts; ++e) {
        other[e] = 0u;
        for (uint32_t r = 0u; r < ranks; ++r)
            if (r != me && (b->held[at][r][e >> 6] >> (e & 63u) & 1ull) != 0ull) other[e] = 1u;
    }
    return true;
}

/* ⭐ AN EXCLUSIVE GROUP'S SLICE TRADED (NN-106): every card posts which of the layer's experts it holds outside any swap,
 * and how many free slots it set aside, and reads the others'. Its cards call it layer by layer in the same order — at a
 * layer's first visit or chunk, and at its choices — and its posts alternate as the board's do. */
typedef struct nn__expert_tier__slice_board {
    uint64_t gen[NN__EXPERT_TIER__RANKS_MAX];
    uint64_t calls[NN__EXPERT_TIER__RANKS_MAX];
    uint64_t held[2][NN__EXPERT_TIER__RANKS_MAX][NN__EXPERT_TIER__CHUNK_WORDS];
    uint64_t room[2][NN__EXPERT_TIER__RANKS_MAX];
} nn__expert_tier__slice_board;
static nn__expert_tier__slice_board nn__expert_tier__zzprivate_slice_board;

/* Card `me` posts what it holds of `layer`'s experts outside any swap, and its `room`, and waits for every other card's post
 * of the same call: `holders[e]` comes back how many cards hold expert e, `who[e]` the lowest numbered of them, `least` the
 * fewest slots any card set aside. False when a card never posts, or the layer has more experts than a post holds. */
static bool nn__expert_tier__zzprivate_share_slice(uint32_t me, uint32_t ranks, uint64_t layer, uint64_t type, uint64_t experts,
                                                   uint64_t room, uint8_t* holders, uint8_t* who, uint64_t* least) {
    if (experts > 64u * NN__EXPERT_TIER__CHUNK_WORDS) return false;
    nn__expert_tier__slice_board* b = &nn__expert_tier__zzprivate_slice_board;
    const uint64_t owner = nn__expert__zzpackage_owner(), call = ++b->calls[me], at = call & 1u;
    for (unsigned w = 0u; w < NN__EXPERT_TIER__CHUNK_WORDS; ++w) b->held[at][me][w] = 0ull;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    for (uint64_t e = 0u; e < experts; ++e) {
        sys__heap_node n;
        if (nn__expert__slot(layer, type, e, &n) && n.args[NN__EXPERT__SLOT_AT] != 0ull
         && !nn__expert_tier__zzprivate_outgoing(owner, layer, e))
            b->held[at][me][e >> 6] |= 1ull << (e & 63u);
    }
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    b->room[at][me] = room;
    __atomic_store_n(&b->gen[me], call, __ATOMIC_RELEASE);
    if (!nn__expert_tier__zzprivate_posted(b->gen, me, ranks, call)) return false;
    *least = room;
    for (uint32_t r = 0u; r < ranks; ++r) if (b->room[at][r] < *least) *least = b->room[at][r];
    for (uint64_t e = 0u; e < experts; ++e) {
        holders[e] = 0u;
        who[e] = NN__EXPERT_TIER__NO_KEEPER;
        for (uint32_t r = 0u; r < ranks; ++r)
            if ((b->held[at][r][e >> 6] >> (e & 63u) & 1ull) != 0ull && holders[e]++ == 0u) who[e] = (uint8_t)r;
    }
    return true;
}

/* Every layer's slice learned at the first visit or chunk of any: the runtime dealt it to every card at boot, so every
 * card trades what it holds of each layer in turn, and the experts all of them hold are its slice. Every layer at once,
 * before any promotion: a promotion's victim may be of another layer's band (▶ `…_victim`). False when a trade fails. */
static bool nn__expert_tier__zzprivate_slice_ready(const nn__expert_tier* t, nn__expert_tier__slice* v, uint32_t me) {
    const uint64_t E = t->experts;
    if (v->known[t->layer]) return true;
    uint8_t* holders = (uint8_t*)malloc(E);
    uint8_t* who = (uint8_t*)malloc(E);
    uint64_t least = 0u;
    bool ok = holders != 0 && who != 0;
    for (uint64_t l = 0u; l < v->layers && ok; ++l) {
        if (nn__expert__layer_experts(l, t->type) == 0ull) continue;
        ok = nn__expert_tier__zzprivate_share_slice(me, (uint32_t)t->ranks, l, t->type, E, 0u, holders, who, &least);
        for (uint64_t e = 0u; e < E && ok; ++e) v->in[l * E + e] = holders[e] == t->ranks ? 1u : 0u;
        v->known[l] = ok ? 1u : 0u;
    }
    free(holders); free(who);
    return ok && v->known[t->layer];
}

/* A stage job's record, or none. Lock held. */
static nn__loader__read* nn__loader__zzprivate_stage_job(void) {
    for (unsigned i = 0u; i < NN__EXPERT_TIER__STAGE_JOBS; ++i)
        if (nn__loader__zzprivate_stage_jobs[i].state == NN__LOADER__FREE) return &nn__loader__zzprivate_stage_jobs[i];
    return 0;
}

/* This card's expert `x` of `layer` read into `stage` for change `gen`, which `takers` cards then write into themselves —
 * once the change before has been written out of it. Lock held (each wait lets it go). */
static bool nn__expert_tier__zzprivate_stage_down(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t x,
                                                  unsigned stage, uint64_t gen, unsigned takers, uint64_t bytes) {
    nn__loader__stage* s = &nn__loader__zzprivate_stages[stage];
    /* ⛳ THE CHANGE BEFORE IS OUT OF IT ALREADY: a stage comes round again two choices later, at least two positions on,
     *   and every card switches the one before at the position after its choice — so this waits only on a slow copy */
    while (s->takers != 0u) nn__loader__zzprivate_wait();
    if (s->host == 0 || s->bytes < bytes) {
        void* card = 0;
        if (s->host != 0) sys__gpu__memory_free_host_ram(family, s->host);
        s->host = 0; s->bytes = 0u;
        if (!sys__gpu__memory_allocate_host_ram_mapped(family, &s->host, &card, (size_t)bytes)) return false;
        s->bytes = bytes;
    }
    sys__heap_node n;
    nn__loader__read* r = nn__loader__zzprivate_stage_job();
    if (r == 0 || !nn__expert__slot(layer, type, x, &n) || n.args[NN__EXPERT__SLOT_AT] == 0ull || !nn__loader__zzprivate_start())
        return false;
    s->gen = gen; s->takers = takers;
    r->predicted = false; r->owner = nn__expert__zzpackage_owner(); r->family = family;
    r->layer = layer; r->type = type; r->expert = x; r->slot = 0ull; r->room = n.args[NN__EXPERT__SLOT_AT];
    r->gather = false; r->bytes = bytes; r->runs.runs = 0u; r->job = NN__LOADER__JOB_STAGE_DOWN; r->channel = stage; r->by = gen;
    r->device = sys__silicon__bound_device(family);
    r->seq = ++nn__loader__zzprivate_seq;
    r->state = NN__LOADER__QUEUED;
    sys__os__condition_wake_all(&nn__loader__zzprivate_work);
    return true;
}

/* This card's write of expert `x` of `layer` out of `stage` (change `gen`) into its card slot at `room` — waiting until the
 * stage is filled, which another card's read does. The record, or none. Lock held. */
static nn__loader__read* nn__expert_tier__zzprivate_stage_up(sys__silicon_family__id family, uint64_t layer, uint64_t type,
                                                             uint64_t x, uint64_t room, unsigned stage, uint64_t gen,
                                                             uint64_t bytes) {
    nn__loader__read* r = nn__loader__zzprivate_stage_job();
    if (r == 0 || !nn__loader__zzprivate_start()) return 0;
    const nn__loader__stage* s = &nn__loader__zzprivate_stages[stage];
    r->predicted = false; r->owner = nn__expert__zzpackage_owner(); r->family = family;
    r->layer = layer; r->type = type; r->expert = x; r->slot = 0ull; r->room = room;
    r->gather = false; r->bytes = bytes; r->runs.runs = 0u; r->job = NN__LOADER__JOB_STAGE_UP; r->channel = stage; r->by = gen;
    r->device = sys__silicon__bound_device(family);
    r->seq = ++nn__loader__zzprivate_seq;
    r->state = s->filled == gen ? NN__LOADER__QUEUED : s->failed == gen ? NN__LOADER__FAILED : NN__LOADER__WAITING;
    sys__os__condition_wake_all(&nn__loader__zzprivate_work);
    return r;
}

/* What card `me` takes and gives up in change `c` — `experts` where nothing: where d is a, every card but a takes x for s;
 * else a takes e for s, d takes x for e, and every other card x for s. */
static inline void nn__expert_tier__zzprivate_roles(const nn__expert_tier__change* c, uint32_t me, uint64_t experts,
                                                    uint64_t* take, uint64_t* give) {
    *take = experts; *give = experts;
    if (c->a == c->d) { if (me != c->a) { *take = c->x; *give = c->s; } }
    else if (me == c->a) { *take = c->e; *give = c->s; }
    else if (me == c->d) { *take = c->x; *give = c->e; }
    else                 { *take = c->x; *give = c->s; }
}

/* ⭐ A CHOICE OF AN EXCLUSIVE GROUP'S SLICE — at the layer's choice visit, after its promotions, on every card alike. Every
 * card sets aside up to SLICE_PAIRS free slots and trades what it holds outside any swap (▶ `…_share_slice`); the slice is
 * chosen again among its members and the experts one card holds — the most picked, a member at a tie, then the lowest
 * numbered — and each joiner X is paired with a leaver S, as many as every card set slots aside for. S stays with d, the
 * card with the fewest picks computed so far; where X's card a is not d, d gives a its least picked expert of its own, E,
 * for the room. This card then queues its share: X read into a stage where it is a, E where it is d, and a write of what
 * it takes into a slot it set aside; the slots it does not need go back, and the counts are halved. The change is switched
 * at the layer's next visit (▶ `…_switch`). False when the trade or a queueing fails. */
static bool nn__expert_tier__zzprivate_change(sys__silicon_family__id family, const nn__expert_tier* t, nn__expert_tier__slice* v,
                                              uint32_t me, const uint64_t* computed) {
    const uint64_t layer = t->layer, type = t->type, E = t->experts, bytes = nn__expert__type_bytes(type);
    const uint32_t ranks = (uint32_t)t->ranks;
    uint32_t* cnt = v->count + layer * E;
    const uint8_t* in = v->in + layer * E;
    uint64_t slot[NN__EXPERT_TIER__SLICE_PAIRS], room[NN__EXPERT_TIER__SLICE_PAIRS], aside = 0u;
    sys__os__lock_take(&nn__loader__zzprivate_lock);
    while (aside < NN__EXPERT_TIER__SLICE_PAIRS && nn__expert__type_take(type, &slot[aside], &room[aside])) ++aside;
    sys__os__lock_give(&nn__loader__zzprivate_lock);
    uint8_t* holders = (uint8_t*)malloc(E);
    uint8_t* who = (uint8_t*)malloc(E);
    uint64_t* order = (uint64_t*)malloc(sizeof(uint64_t) * E);
    uint64_t least = 0u;
    bool ok = holders != 0 && who != 0 && order != 0
           && nn__expert_tier__zzprivate_share_slice(me, ranks, layer, type, E, aside, holders, who, &least);
    unsigned n = 0u;
    if (ok) {
        /* the candidates, best first: the members every card holds and the experts one card holds */
        uint64_t m = 0u, members = 0u;
        for (uint64_t e = 0u; e < E; ++e) {
            if (!(in[e] ? holders[e] == ranks : holders[e] == 1u)) continue;
            members += in[e];
            uint64_t q = m++;
            while (q > 0u && (cnt[order[q - 1u]] < cnt[e] || (cnt[order[q - 1u]] == cnt[e] && !in[order[q - 1u]] && in[e]))) {
                order[q] = order[q - 1u];
                --q;
            }
            order[q] = e;
        }
        /* the slice keeps its size: the joiners are the others within its first `members`, the leavers the members past */
        uint64_t join[NN__EXPERT_TIER__SLICE_PAIRS], leave[NN__EXPERT_TIER__SLICE_PAIRS];
        unsigned nj = 0u, nl = 0u;
        for (uint64_t q = 0u; q < members && nj < NN__EXPERT_TIER__SLICE_PAIRS; ++q) if (!in[order[q]]) join[nj++] = order[q];
        for (uint64_t q = m; q-- > members && nl < NN__EXPERT_TIER__SLICE_PAIRS; ) if (in[order[q]]) leave[nl++] = order[q];
        uint32_t d = 0u;
        for (uint32_t r = 1u; r < ranks; ++r) if (computed[r] < computed[d]) d = r;
        for (unsigned i = 0u; i < nj && i < nl && i < least; ++i) {
            const uint32_t a = who[join[i]];
            uint64_t e = E;
            if (a != d) {                                 /* d's least picked of its own, the lowest numbered at a tie */
                for (uint64_t f = 0u; f < E; ++f) {
                    if (in[f] || holders[f] != 1u || who[f] != d) continue;
                    bool taken = false;
                    for (unsigned j = 0u; j < nj && !taken; ++j) taken = join[j] == f;
                    for (unsigned j = 0u; j < n && !taken; ++j) taken = v->change[layer * NN__EXPERT_TIER__SLICE_PAIRS + j].e == f;
                    if (!taken && (e == E || cnt[f] < cnt[e])) e = f;
                }
                if (e == E) break;                        /* d holds none of its own to give */
            }
            nn__expert_tier__change* c = &v->change[layer * NN__EXPERT_TIER__SLICE_PAIRS + n++];
            c->x = join[i]; c->s = leave[i]; c->e = e; c->a = a; c->d = d; c->up = 0; c->slot = 0ull;
            c->sx = (unsigned)(v->stages++ % NN__EXPERT_TIER__STAGES); c->gx = ++v->changes;
            if (e < E) { c->se = (unsigned)(v->stages++ % NN__EXPERT_TIER__STAGES); c->ge = ++v->changes; }
            else       { c->se = NN__EXPERT_TIER__STAGES; c->ge = 0ull; }
        }
        /* this card's share: its reads into the stages, then its writes out of them into the slots it set aside */
        unsigned used = 0u;
        sys__os__lock_take(&nn__loader__zzprivate_lock);
        for (unsigned i = 0u; i < n && ok; ++i) {
            nn__expert_tier__change* c = &v->change[layer * NN__EXPERT_TIER__SLICE_PAIRS + i];
            if (me == c->a) ok = nn__expert_tier__zzprivate_stage_down(family, layer, type, c->x, c->sx, c->gx, ranks - 1u, bytes);
            if (ok && c->e < E && me == c->d) ok = nn__expert_tier__zzprivate_stage_down(family, layer, type, c->e, c->se, c->ge, 1u, bytes);
        }
        for (unsigned i = 0u; i < n && ok; ++i) {
            nn__expert_tier__change* c = &v->change[layer * NN__EXPERT_TIER__SLICE_PAIRS + i];
            uint64_t take, give;
            nn__expert_tier__zzprivate_roles(c, me, E, &take, &give);
            if (take >= E) continue;
            c->slot = slot[used];
            c->up = nn__expert_tier__zzprivate_stage_up(family, layer, type, take, room[used], take == c->x ? c->sx : c->se,
                                                        take == c->x ? c->gx : c->ge, bytes);
            ok = c->up != 0;
            ++used;
        }
        sys__os__lock_give(&nn__loader__zzprivate_lock);
        aside -= used;
        for (uint64_t e = 0u; e < E; ++e) cnt[e] >>= 1;
        v->changing[layer] = (uint8_t)n;
        for (uint64_t i = 0u; i < aside; ++i) (void)nn__expert__type_give(type, slot[used + i]);
        aside = 0u;
    }
    for (uint64_t i = 0u; i < aside; ++i) (void)nn__expert__type_give(type, slot[i]);
    free(holders); free(who); free(order);
    return ok;
}

/* ⭐ THE SWITCH — this card's share of the changes made at the layer's last choice, at its next visit or chunk, before
 * anything is traded: what it takes, waited for in the slot it set aside and admitted; what it gives up evicted, its slot
 * back to FREE; the slice and the keepers as every card now has them. False on a step that could not be done — a stage
 * whose read failed among them. */
static bool nn__expert_tier__zzprivate_switch(const nn__expert_tier* t, nn__expert_tier__slice* v, uint32_t me) {
    const uint64_t layer = t->layer, type = t->type, E = t->experts;
    bool ok = true;
    for (unsigned i = 0u; i < v->changing[layer]; ++i) {
        nn__expert_tier__change* c = &v->change[layer * NN__EXPERT_TIER__SLICE_PAIRS + i];
        uint64_t take, give;
        nn__expert_tier__zzprivate_roles(c, me, E, &take, &give);
        if (take < E && c->up == 0) ok = false;           /* its write was never queued: the choice failed */
        else if (take < E) {
            nn__loader__read* u = c->up;
            sys__os__lock_take(&nn__loader__zzprivate_lock);
            while (u->state != NN__LOADER__DONE && u->state != NN__LOADER__FAILED) nn__loader__zzprivate_wait();
            const bool got = u->state == NN__LOADER__DONE;
            const uint64_t room = u->room;
            u->state = NN__LOADER__FREE;
            nn__loader__stage* from = &nn__loader__zzprivate_stages[take == c->x ? c->sx : c->se];
            if (from->takers > 0u) --from->takers;
            sys__os__condition_wake_all(&nn__loader__zzprivate_done);
            sys__os__lock_give(&nn__loader__zzprivate_lock);
            if (!got || !nn__expert__admit(layer, take, room, type)) { (void)nn__expert__type_give(type, c->slot); ok = false; }
        }
        if (give < E && !nn__expert__evict(layer, type, give)) ok = false;
        v->in[layer * E + c->s] = 0u;
        v->in[layer * E + c->x] = 1u;
        v->keeper[layer * E + c->s] = (uint8_t)c->d;
        v->keeper[layer * E + c->x] = NN__EXPERT_TIER__NO_KEEPER;
        if (c->e < E) v->keeper[layer * E + c->e] = (uint8_t)c->a;
        if (me == 0u) {
            sys__os__lock_take(&nn__loader__zzprivate_lock);
            ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_SLICE_CHANGED];
            sys__os__lock_give(&nn__loader__zzprivate_lock);
        }
    }
    v->changing[layer] = 0u;
    return ok;
}

/* A group's tier's visit (▶ the board): which of the position's picks THIS card computes (`mine`) and which some card
 * does (`any`, marked held for the CPUs), and whether this card adds the model's shared expert (`shared_here`). In order:
 * a pick outside the slice that cards hold is its keeper's where the keeper holds it, else the one of them with the
 * fewest so far — which keeps it from then on; then the slice's picks that cards hold and the shared expert, one at a
 * time to the card with the fewest of those that can (the lowest numbered at a tie); a pick none holds is the CPUs', and
 * the card with the fewest — then the one the fewest promotions were given so far — is the one that may promote it, by
 * the tier's own qualifier and landing; and every card not holding a pick of the slice promotes it. Every card counts
 * every card's picks and promotions the same way, so their assignments never part. A card touches only what it computes
 * and the slice it holds, so a copy no longer computed there ages out. */
static __device__ inline bool nn__expert_tier__visit_group(sys__silicon_family__id family, const nn__expert_tier* t,
                                                           const uint64_t* ids, bool* mine, bool* any, bool* shared_here) {
    const uint64_t layer = t->layer, type = t->type, E = t->experts;
    const unsigned k = (unsigned)t->top_k;
    const uint32_t ranks = (uint32_t)t->ranks, me = (uint32_t)sys__silicon__worker();
    const bool fed = t->sources != 0ull;
    if (fed) (void)nn__expert__settle(layer, type);           /* the visit before's promotions, as `visit` admits them */
    /* an exclusive tier's swaps that are due, before this card says what it holds (▶ commit): every card's post is then the
     *   layout this position runs on */
    if (t->exclusive == 1u && !nn__expert_tier__commit(family, t)) return false;
    nn__expert_tier__slice* v = t->shared != 0u ? nn__expert_tier__zzprivate_slice(me, E) : 0;
    if (t->shared != 0u && (v == 0 || layer >= v->layers)) return false;
    /* an exclusive group's slice: the last choice's changes switched, and at the layer's first visit the slice learned —
     *   both before this card says what it holds (▶ NN-106) */
    if (v != 0 && t->exclusive == 1u
     && (!nn__expert_tier__zzprivate_switch(t, v, me) || !nn__expert_tier__zzprivate_slice_ready(t, v, me))) return false;
    const uint64_t now = nn__expert__tick();
    bool held[NN__EXPERT__TIER_K_MAX];
    uint64_t pinned[NN__EXPERT__TIER_K_MAX];
    unsigned n_pinned = 0u, hits = 0u;
    for (unsigned j = 0u; j < k; ++j) {
        sys__heap_node e;
        if (ids[j] >= E || !nn__expert__slot(layer, type, ids[j], &e)) return false;
        held[j] = e.args[NN__EXPERT__SLOT_AT] != 0ull;
        if (held[j]) pinned[n_pinned++] = ids[j];
    }
    uint8_t all[NN__EXPERT_TIER__RANKS_MAX][NN__EXPERT__TIER_K_MAX];
    if (!nn__expert_tier__zzprivate_share(me, ranks, k, ids, held, all)) return false;
    uint64_t* computed = nn__expert_tier__zzprivate_board.computed[me];          /* this card's count of every card's */
    uint8_t* in = v != 0 ? v->in + layer * E : 0;
    uint8_t* keeper = v != 0 ? v->keeper + layer * E : 0;
    uint32_t load[NN__EXPERT_TIER__RANKS_MAX] = {}, owner_of[NN__EXPERT__TIER_K_MAX], promoter[NN__EXPERT__TIER_K_MAX];
    for (unsigned j = 0u; j < k; ++j) { owner_of[j] = ranks; promoter[j] = ranks; }
    /* the picks outside the slice that cards hold: their keeper's, or the least loaded holder, who keeps them */
    for (unsigned j = 0u; j < k; ++j) {
        if (in != 0 && in[ids[j]]) continue;
        uint32_t best = ranks;
        const uint32_t kept = keeper != 0 ? keeper[ids[j]] : NN__EXPERT_TIER__NO_KEEPER;
        if (kept < ranks && all[kept][j]) best = kept;
        else for (uint32_t r = 0u; r < ranks; ++r)
            if (all[r][j] && (best == ranks || load[r] < load[best])) best = r;
        if (best == ranks) continue;
        owner_of[j] = best; ++load[best];
        if (keeper != 0) keeper[ids[j]] = (uint8_t)best;
    }
    /* the slice's picks that cards hold, then the model's shared expert — any card's — one at a time to the fewest */
    for (unsigned j = 0u; in != 0 && j < k; ++j) {
        if (!in[ids[j]]) continue;
        uint32_t best = ranks;
        for (uint32_t r = 0u; r < ranks; ++r)
            if (all[r][j] && (best == ranks || load[r] < load[best])) best = r;
        if (best == ranks) continue;
        owner_of[j] = best; ++load[best];
    }
    if (shared_here != 0) {
        uint32_t best = 0u;
        for (uint32_t r = 1u; r < ranks; ++r) if (load[r] < load[best]) best = r;
        *shared_here = best == me;
        ++load[best];
    }
    for (unsigned j = 0u; j < k; ++j) {
        any[j] = owner_of[j] < ranks;
        mine[j] = owner_of[j] == me;
        if (mine[j]) ++hits;
        if (held[j] && (mine[j] || (in != 0 && in[ids[j]]))) (void)nn__expert__touch(layer, type, ids[j]);
    }
    uint64_t* promoted = nn__expert_tier__zzprivate_board.promoted[me];          /* this card's count of every card's */
    for (unsigned j = 0u; j < k; ++j) {
        if (any[j]) continue;
        uint32_t best = 0u;
        for (uint32_t r = 1u; r < ranks; ++r)
            if (load[r] < load[best] || (load[r] == load[best] && promoted[r] < promoted[best])) best = r;
        promoter[j] = best;
        ++promoted[best];
    }
    for (uint32_t r = 0u; r < ranks; ++r) computed[r] += load[r];
    /* the slice: this layer's picks counted, and chosen again every SLICE_EVERY visits — an exclusive group's after this
     *   visit's promotions, at visits its layer's number apart from another layer's (▶ `…_change`) */
    bool choose = false;
    if (v != 0) {
        for (unsigned j = 0u; j < k; ++j) ++v->count[layer * E + ids[j]];
        ++v->visits[layer];
        if (t->exclusive == 1u) choose = (v->visits[layer] + layer) % NN__EXPERT_TIER__SLICE_EVERY == 0u;
        else if (v->visits[layer] % NN__EXPERT_TIER__SLICE_EVERY == 0u)
            nn__expert_tier__zzprivate_choose(v, layer, t->shared, ranks, computed);
    }
    {
        const uint64_t owner = nn__expert__zzpackage_owner();
        sys__os__lock_take(&nn__loader__zzprivate_lock);
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_PICKS] += k;
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_HITS] += hits;
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_CARD_HITS + me] += hits;
        if (me == 0u) {                                   /* every card assigns alike: card 0 counts the busiest */
            unsigned routed[NN__EXPERT_TIER__RANKS_MAX] = {}, most = 0u;
            for (unsigned j = 0u; j < k; ++j) if (owner_of[j] < ranks) ++routed[owner_of[j]];
            for (uint32_t r = 0u; r < ranks; ++r) if (routed[r] > most) most = routed[r];
            nn__loader__zzprivate_count[NN__EXPERT__COUNT_BUSIEST] += most;
            ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_GROUP_VISITS];
        }
        for (unsigned j = 0u; j < k; ++j)
            if (!held[j] && nn__loader__zzprivate_find(owner, layer, type, ids[j]) != 0)
                ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PICKED_ARRIVING];
        sys__os__lock_give(&nn__loader__zzprivate_lock);
    }
    /* this card's promotions: the picks no card holds that fell to it, by the qualifier, and the slice's it does not hold
     *   — the best first (▶ `nn__expert_tier__visit`); an exclusive group's slice is every card's already */
    const unsigned land = hits > t->landing_line ? 1u : (unsigned)t->landing;
    unsigned cand[NN__EXPERT__TIER_K_MAX];
    uint64_t score[NN__EXPERT__TIER_K_MAX];
    unsigned n_cand = 0u;
    for (unsigned j = 0u; j < k && fed && land > 0u; ++j) {
        const bool sliced = t->exclusive != 1u && in != 0 && in[ids[j]] && !held[j];
        if (!sliced && (promoter[j] != me || !nn__expert__qualifies(layer, type, ids[j]))) continue;
        uint64_t sc = 0ull;
        (void)nn__expert__zzpackage_older(layer, type, ids[j], &sc);
        unsigned at = n_cand++;
        while (at > 0u && score[at - 1u] < sc) { cand[at] = cand[at - 1u]; score[at] = score[at - 1u]; --at; }
        cand[at] = j; score[at] = sc;
    }
    /* ⭐ AN INCLUSIVE GROUP'S CARD PROMOTES WITHIN ITS SHARE OF THE TABLE OF READS: a token's promotions are admitted at
     *   the band's next visit, so a card holds up to `land` a layer across every layer at once — more than the table
     *   holds for two cards — and which card's promotion found the last read free was the two cards' timing, the answer
     *   after it with it (NN-115 ④, MEASURED: two processes' tokens' logits apart from the second). Its own reads change
     *   only by its own promotions and admissions, so counted against its share alone a card decides the same way on
     *   every run. An exclusive tier's swaps keep their own count (▶ `…_swaps_left`). */
    unsigned room = land;
    if (ranks > 1u && t->exclusive != 1u) {
        const unsigned mine = NN__EXPERT__FLIGHT_MAX / ranks, held_now = nn__expert__flight_mine();
        room = held_now >= mine ? 0u : mine - held_now;
    }
    unsigned started = 0u;
    for (unsigned c = 0u; c < n_cand && started < land && started < room; ++c) {
        nn__expert__gather g;
        if (t->exclusive == 1u ? nn__expert_tier__promote(family, t, ids[cand[c]], pinned, n_pinned) == NN__EXPERT__ARRIVING
                               : (nn__expert_tier__stitch(t, ids[cand[c]], &g)
                                  && nn__expert__promote(family, layer, type, ids[cand[c]], &g, pinned, n_pinned) == NN__EXPERT__ARRIVING))
            ++started;
    }
    if (choose && !nn__expert_tier__zzprivate_change(family, t, v, me, computed)) return false;
    for (unsigned j = 0u; j < k; ++j) (void)nn__expert__called(layer, type, ids[j], now);
    return true;
}

static __device__ inline bool nn__expert_tier__chunk(sys__silicon_family__id family, const nn__expert_tier* t, uint64_t* chosen,
                                                     uint64_t n, uint64_t threshold) {
    const uint64_t layer = t->layer, type = t->type, experts = t->experts;
    const unsigned k = (unsigned)t->top_k;
    const bool fed = t->sources != 0ull;
    if (fed) (void)nn__expert__settle(layer, type);           /* the chunk before's promotions, as `visit` admits them */
    if (t->exclusive == 1u && !nn__expert_tier__commit(family, t)) return false;   /* the swaps that are due: ▶ commit */
    /* ⭐ A GROUP'S EXCLUSIVE TIERS EACH TOLD OF THE CHUNK (NN-99): each card computes the chunk's picks it holds, and the
     *   others' are marked `2·experts` past their number, which this card's experts and the CPUs both leave (▶
     *   `nn__routed__rows`) — card 0's hands are the ones the CPUs are handed. An inclusive group's chunk is card 0's
     *   alone: its CPUs hold every expert. */
    const bool group = t->ranks > 1u && t->exclusive == 1u;
    const uint32_t ranks = (uint32_t)t->ranks, rank = (uint32_t)sys__silicon__worker();
    /* ⭐ STREAMED (NN-115 ② ③): a line of 0, or one with NOTE_STREAM, where the tier has a ring — every pick the card's;
     *   the experts it does not hold are promoted into the ring as the card computes the chunk (▶ `nn__routed__rows`), and
     *   with a line the ones picked that often into the tier first, as below. ⭐ AND IN AN INCLUSIVE GROUP (NN-115 ④): every
     *   card streams the experts numbered its own modulo the group, the others' marked `2·experts` past as an exclusive
     *   group's are — a pick computed once, by the card it falls to, the sites' group sum adding the cards' routed rows */
    const bool stream = t->stream != 0u && (threshold == 0u || (threshold & NN__EXPERT_TIER__NOTE_STREAM) != 0u)
                     && fed && !group && t->exclusive != 1u;
    const bool share = stream && t->ranks > 1u;
    threshold &= ~NN__EXPERT_TIER__NOTE_STREAM;
    /* an exclusive group's slice: the last choice's changes switched, and at the layer's first chunk the slice learned (▶
     *   NN-106) — a chunk never chooses one */
    nn__expert_tier__slice* v = group && t->shared != 0u ? nn__expert_tier__zzprivate_slice(rank, experts) : 0;
    if (group && t->shared != 0u
     && (v == 0 || layer >= v->layers || !nn__expert_tier__zzprivate_switch(t, v, rank)
         || !nn__expert_tier__zzprivate_slice_ready(t, v, rank))) return false;
    uint32_t* picked = (uint32_t*)calloc(experts, sizeof(uint32_t));
    uint8_t* card = (uint8_t*)calloc(experts, 1u);                 /* 1: the card computes this expert for the chunk */
    uint8_t* other = group || share ? (uint8_t*)calloc(experts, 1u) : 0;    /* 1: another card of the group does */
    uint64_t* list = (uint64_t*)malloc(sizeof(uint64_t) * experts);
    uint64_t* pinned = (uint64_t*)malloc(sizeof(uint64_t) * experts);
    bool ok = picked != 0 && card != 0 && list != 0 && pinned != 0 && (!(group || share) || other != 0);
    /* ⭐ EVERY CALL RECORDED AS A ROW AT A TIME WOULD RECORD IT, ONCE AN EXPERT: a row a tick, an expert's two newest calls
     *   the ticks of the last two rows that picked it, and the resident ones touched in the order of their last picks —
     *   the chain as a touch a pick leaves it. Done a pick at a time it was ~50 ms of a 10,000-row slice's layer, the
     *   card idle (NN-115 ④). */
    uint64_t* last_pick = ok ? (uint64_t*)malloc(sizeof(uint64_t) * experts) : 0;   /* its last pick: row · k + j */
    uint64_t* rows_two = ok ? (uint64_t*)malloc(sizeof(uint64_t) * 2u * experts) : 0; /* the rows of its last two picks */
    ok = ok && last_pick != 0 && rows_two != 0;
    for (uint64_t i = 0u; i < n * k && ok; ++i) {
        const uint64_t id = chosen[i];
        if (id >= experts) { ok = false; break; }
        ++picked[id];
        last_pick[id] = i;
        rows_two[2u * id] = rows_two[2u * id + 1u]; rows_two[2u * id + 1u] = i / k;
    }
    const uint64_t first_tick = ok ? nn__expert__ticks(n) : 0u;            /* row r's: first_tick + r */
    uint64_t touched = 0u;
    for (uint64_t e = 0u; e < experts && ok; ++e) {
        if (picked[e] == 0u) continue;
        sys__heap_node me;
        if (!nn__expert__slot(layer, type, e, &me)) { ok = false; break; }
        if (me.args[NN__EXPERT__SLOT_AT] != 0ull) {                      /* resident: touched, by its last pick's order */
            card[e] = 1u;
            uint64_t q = touched++;
            while (q > 0u && last_pick[list[q - 1u]] > last_pick[e]) { list[q] = list[q - 1u]; --q; }
            list[q] = e;
        }
        if (picked[e] > 1u) (void)nn__expert__called(layer, type, e, (first_tick + rows_two[2u * e]) & 0xFFFFFFFFull);
        (void)nn__expert__called(layer, type, e, (first_tick + rows_two[2u * e + 1u]) & 0xFFFFFFFFull);
    }
    for (uint64_t q = 0u; q < touched && ok; ++q) (void)nn__expert__touch(layer, type, list[q]);
    free(last_pick); free(rows_two);
    if (ok && group) ok = nn__expert_tier__zzprivate_share_held(rank, ranks, experts, card, other);
    /* a pick of the slice, which every card holds: the card its number modulo the group names computes it */
    for (uint64_t e = 0u; e < experts && ok && v != 0; ++e)
        if (v->in[layer * experts + e] && card[e] && e % ranks != rank) { card[e] = 0u; other[e] = 1u; }
    /* a streaming group's: the experts not numbered this card's are the others' — held here or not */
    for (uint64_t e = 0u; e < experts && ok && share; ++e)
        if (picked[e] != 0u && e % ranks != rank) { card[e] = 0u; other[e] = 1u; }
    unsigned n_pinned = 0u;
    for (uint64_t e = 0u; e < experts && ok; ++e) if (card[e]) pinned[n_pinned++] = e;
    /* the line: the experts not on the card picked `threshold` times or more, the most picked first, as many as there is room */
    if (ok && fed && !(stream && threshold == 0u)) {
        uint64_t m = 0u;
        for (uint64_t e = 0u; e < experts; ++e) {
            if (card[e] || picked[e] == 0u || picked[e] < threshold) continue;
            /* in a group, an expert another card holds is not this one's to take, and each card takes the ones numbered
             *   its own modulo the group — the same split on every card, from the same picks */
            if ((group || share) && (other[e] || e % ranks != rank)) continue;
            uint64_t q = m++;
            while (q > 0u && picked[list[q - 1u]] < picked[e]) { list[q] = list[q - 1u]; --q; }
            list[q] = e;
        }
        /* an exclusive tier: as many as this card may start swaps (▶ `…_swaps_left`), the most picked — each holds a
         *   channel until its commit */
        uint64_t room = 0u;
        if (t->exclusive == 1u) {
            sys__os__lock_take(&nn__loader__zzprivate_lock);
            room = nn__expert_tier__zzprivate_swaps_left(nn__expert__zzpackage_owner(),
                                                         sys__pad__of(family, sys__silicon__bound_device(family)));
            sys__os__lock_give(&nn__loader__zzprivate_lock);
        } else {
            room = nn__expert__flight_free();
            /* a streaming group's cards promote at once from one table of reads, and a read is free again only when the card
             *   that made it settles it: each card's line takes at most its share of the table less a batch of the ring's,
             *   so its stream always finds a read free (▶ `nn__routed__zzprivate_stream_one`) — and how many the line
             *   promotes is the same on every run, not the other cards' timing */
            if (share) {
                /* ...less what this card holds already — a token's promotions not yet admitted, before a prompt read
                 *   later in a conversation */
                const uint64_t mine = NN__EXPERT__FLIGHT_MAX / ranks, held_now = nn__expert__flight_mine();
                const uint64_t line = mine > t->stream_batch + held_now ? mine - t->stream_batch - held_now : 0u;
                if (room > line) room = line;
            }
        }
        if (m > room) m = room;
        for (uint64_t q = m; q-- > 0u; ) {                         /* queued from the fewest picks to the most */
            nn__expert__gather g;
            /* ⛳ IN A GROUP A PROMOTION IS NOT THE CHUNK'S: the others cannot know it started, so the CPUs compute it this
             *   chunk — from the RAM an exclusive tier's arriving expert keeps until its commit */
            if (t->exclusive == 1u ? nn__expert_tier__promote(family, t, list[q], pinned, n_pinned) == NN__EXPERT__ARRIVING
                                   : (nn__expert_tier__stitch(t, list[q], &g)
                                      && nn__expert__promote(family, layer, type, list[q], &g, pinned, n_pinned) == NN__EXPERT__ARRIVING))
                card[list[q]] = group ? 0u : 1u;
        }
    }
    for (uint64_t e = 0u; e < experts && ok && stream; ++e) if (picked[e] != 0u && !(share && other[e])) card[e] = 1u;
    for (uint64_t i = 0u; i < n * k && ok; ++i) {
        if (card[chosen[i]]) chosen[i] += experts;
        else if ((group || share) && other[chosen[i]]) chosen[i] += 2u * experts;
    }
    free(picked); free(card); free(other); free(list); free(pinned);
    return ok;
}

#endif /* SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH */
