#ifndef SILVANN__SILICON_FAMILIES_CHOOSE_CUH
#define SILVANN__SILICON_FAMILIES_CHOOSE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include <stdbool.h>
#include "../packages/sys/contracts/abi/silicon_family.cuh"   /* the bound on a family's number, and NONE */

/* ══ ⭐⭐⭐ WHICH FAMILY A CARD GOES TO ══════════════════════════════════════════════════════════════════
 *
 * ⚖ *"we move from arch to arch until we find one that works"* · *"arch get done from the highest family
 * list to the smallest"* · *"we could also verify that rocm6 is installed on the same check"*. A card is
 * offered to every family that includes it, highest number first, and goes to the first one that WORKS:
 *
 *     its binary opens        which is the check that its runtime is installed — the family's binary
 *                             links against its runtime by name (`libamdhip64.so.6`), so on a box
 *                             without that runtime the open fails and the next family down is asked.
 *                             ⛳ A failed open has started nothing, so moving on is free.
 *     and it sees its cards   once open, it counts at least one card it includes.
 *
 * ⛔ AN OPEN FAMILY THAT SEES NONE IS A FAILURE, NOT A REASON TO MOVE ON. Opening it started its runtime;
 *   asking the next family of the same vendor would start a second one in the same process. The card is
 *   reported, by name, with the family that could not see it — a driver older than the runtime, most often.
 *
 * ⛔⛔ AND TWO FAMILIES OF ONE VENDOR ARE NEVER OPEN IN ONE PROCESS. `REASONED` from the libraries on
 *   node03: HIP (`libamdhip64.so.6`) needs `libhsa-runtime64.so.1`, and the loader matches libraries by
 *   that name, so a second ROCm's HIP opened beside the first would be bound to the FIRST one's HSA runtime
 *   rather than its own. `ASSUMED`: that the next ROCm still names it `.so.1` — what would falsify it is a
 *   ROCm that ships it under another name, and even then two runtimes driving one kernel driver is
 *   unmeasured. ⇒ An MI50 on ROCm 6 and an MI350 on a later ROCm in one box is TWO PROCESSES, one per GPU
 *   (`docs/design/pending/evaluator_and_providers.md` §①): each loads only its own family. In one process, the card whose family would
 *   be the vendor's second is reported as needing a process of its own.
 *
 * ⛳ WRITTEN OVER A LIST AND TWO CALLS, NOT OVER THE ROSTER AND `dlopen`, so it can be checked against a
 *   made-up roster — the case this box cannot show, two families that both include one card, is the case
 *   the order exists for. The program hands it the real roster (`silicon_families__rows`) and the wheel's
 *   own open and count. ▶ `test/src_silicon_family_choice.cpp`. */

/* A family as the chooser sees it. */
typedef struct silicon_families__row {
    sys__silicon_family__id id;             /* its number — what a machine records, and the preference */
    const char* name;                       /* its folder, and the name its binary is found by */
    const char* vendor;                     /* `amd`, `nvidia`, `host` */
    bool      (*includes)(const char* card);
} silicon_families__row;

/* What offering a card to one family came to. */
enum {
    SILICON_FAMILIES__TAKEN            = 1u,   /* it opened, sees its cards, and this card is its       */
    SILICON_FAMILIES__WONT_OPEN        = 2u,   /* its binary would not open here — its runtime is not
                                                  installed, or the binary is missing or broken          */
    SILICON_FAMILIES__SEES_NONE        = 3u,   /* it opened and counts no card of its own: its runtime
                                                  started, so nothing below it is asked                  */
    SILICON_FAMILIES__VENDOR_ALREADY   = 4u    /* another family of its vendor is open in this process   */
};

/* One offer, in the order they were made. */
typedef struct silicon_families__attempt {
    uint32_t family;                        /* its number */
    uint32_t outcome;                       /* SILICON_FAMILIES__TAKEN … */
} silicon_families__attempt;

/* What a process remembers from one card to the next: which families it has opened, and which would not
 * open, so a binary is opened once whatever number of cards it takes. Zero-initialised is "nothing yet". */
typedef struct silicon_families__session {
    uint8_t state[SYS__SILICON_FAMILY__SLOTS];     /* by position in the roster: 0 not asked · 1 open · 2 would not open */
} silicon_families__session;

typedef bool     (*silicon_families__open_fn)(const char* family, void* context);
typedef uint32_t (*silicon_families__count_fn)(const char* family, void* context);

static inline bool silicon_families__zzprivate_same(const char* a, const char* b) {
    while (*a && *a == *b) { ++a; ++b; }
    return *a == *b;
}

/* The number of the family `card` goes to, or `SYS__SILICON_FAMILY__NONE` — which, with no attempts
 * written, means no family includes the card (it is UNSUPPORTED), and with some, says why each failed. */
static inline sys__silicon_family__id silicon_families__choose(const silicon_families__row* rows, uint32_t count,
                                                               silicon_families__session* session, const char* card,
                                                               silicon_families__open_fn open,
                                                               silicon_families__count_fn devices, void* context,
                                                               silicon_families__attempt* attempts, uint32_t most,
                                                               uint32_t* made) {
    *made = 0u;
    if (count > SYS__SILICON_FAMILY__SLOTS) count = SYS__SILICON_FAMILY__SLOTS;
    for (uint32_t i = count; i-- > 0u; ) {
        if (!rows[i].includes(card)) continue;
        uint32_t outcome = 0u;
        if (session->state[i] == 0u) {
            for (uint32_t j = 0u; j < count; ++j)
                if (j != i && session->state[j] == 1u && silicon_families__zzprivate_same(rows[j].vendor, rows[i].vendor))
                    outcome = SILICON_FAMILIES__VENDOR_ALREADY;
            if (outcome == 0u) session->state[i] = open(rows[i].name, context) ? 1u : 2u;
        }
        if (outcome == 0u)
            outcome = (session->state[i] == 2u)                ? SILICON_FAMILIES__WONT_OPEN
                    : (devices(rows[i].name, context) > 0u)    ? SILICON_FAMILIES__TAKEN
                                                               : SILICON_FAMILIES__SEES_NONE;
        if (*made < most) { attempts[*made].family = rows[i].id; attempts[*made].outcome = outcome; ++*made; }
        if (outcome == SILICON_FAMILIES__TAKEN) return rows[i].id;
        if (outcome == SILICON_FAMILIES__SEES_NONE) return SYS__SILICON_FAMILY__NONE;
    }
    return SYS__SILICON_FAMILY__NONE;
}

#endif /* SILVANN__SILICON_FAMILIES_CHOOSE_CUH */
