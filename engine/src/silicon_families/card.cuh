#ifndef SILVANN__SILICON_FAMILIES_CARD_CUH
#define SILVANN__SILICON_FAMILIES_CARD_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdbool.h>

/* ══ A CARD, BY NAME ═════════════════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"find the device name and check if it is included inside that family"*. A card is named by the
 * instruction set it runs, because that is what decides whether a build has code for it:
 *
 *     AMD       `gfx906`, `gfx90a`, `gfx1100` — the target a compiler is given. A driver may add feature
 *               flags after a colon (`gfx906:sramecc+:xnack-`); they do not change which build runs, so
 *               the name is read up to the colon.
 *     NVIDIA    `sm_86`, `sm_120` — the compute capability, as a compiler's `-arch` spells it.
 *
 * ⛳ NOT THE MARKETING NAME. "Radeon Instinct MI50" and "Radeon VII" are one `gfx906`; one marketing name
 *   has shipped on two instruction sets. The ISA string is the only name a build can be checked against. */

/* Whether `card` is the target `target`, reading the card's name only as far as its first colon. */
static inline bool silicon_families__card_is(const char* card, const char* target) {
    if (card == 0 || target == 0) return false;
    while (*target && *card == *target) { ++card; ++target; }
    return *target == 0 && (*card == 0 || *card == ':');
}

#endif /* SILVANN__SILICON_FAMILIES_CARD_CUH */
