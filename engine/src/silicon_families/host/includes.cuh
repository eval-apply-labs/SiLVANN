#ifndef SILVANN__SILICON_FAMILIES_HOST_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_HOST_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE HOST TAKES NO CARD ══════════════════════════════════════════════════════════════════════════
 * The host family is the machine the program runs on, compiled into it and offered when it starts; it is
 * never chosen for a card, so it includes none. It answers the question anyway, so the roster can ask
 * every family the same thing in the same order. */
static inline bool host__includes(const char* card) {
    (void)card;
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_HOST_INCLUDES_CUH */
