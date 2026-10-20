#ifndef SILVANN__SILICON_FAMILIES_OPENCL_VERSIONS_CUH
#define SILVANN__SILICON_FAMILIES_OPENCL_VERSIONS_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdbool.h>
#include "card.cuh"   /* what a card's name is */

/* ══ ⭐ WHICH CARDS' OPENCL IS 1.2 — THE TWO OPENCL FAMILIES' SPLIT, A BEST GUESS (NN-96) ═══════════════════════════════
 *
 * ⚖ 2026-10-19, the architect: *"add all the other nvidia pre 20 series to the opencl block, and figure out which nvidia
 * and amd and intel gpus go to the opencl 1 vs 2 block. it is a best effort guess, so that 80% of the users will be on
 * the right setup and the rest will open a ticket."* `khronos_opencl2` needs OpenCL 2.0 with fine-grained SVM;
 * `khronos_opencl1_igpu` needs only 1.2. A card named here is the 1.2 family's and never the 2's — the 2 family, of the
 * same vendor (`khronos`), is not asked again once it is open, so a card it would open and then refuse would get no
 * family at all. `REASONED` from the vendors' published OpenCL versions, none of it run but NVIDIA's sm_120:
 *
 *   NVIDIA   every card (`sm_*`): its OpenCL C is 1.2 whatever the card, and `nvidia_cuda12` is asked first — so a
 *            card CUDA 12 builds for (Turing and after, sm_75 …) is CUDA's, and Kepler, Maxwell, Pascal and Volta
 *            (CUDA 13 builds none of them), or any card whose CUDA will not open, come here
 *   Intel    the iGPUs before Gen9: Ivy Bridge's HD 2500 / 4000, Haswell's HD 4200 … 5200 and Iris 5100 / Iris Pro
 *            5200, Bay Trail's "HD Graphics" with no number. Broadwell (Gen8, OpenCL 2.0), Skylake and after
 *            (HD / UHD 5xx 6xx 7xx, Iris Xe) and Arc are the 2 family's
 *   AMD      TeraScale (HD 5000 / 6000 and its APUs: Evergreen and Northern Islands, OpenCL 1.2 at most) and GCN's
 *            first generation (Tahiti, Pitcairn, Cape Verde, Oland, Hainan — `gfx600 … 602`), whose drivers give
 *            OpenCL 2.0 without fine-grained SVM. GCN 2 (Hawaii, Bonaire, Kaveri …) and after are the 2 family's
 *
 * ⛳ BY NAME, AS THE SYSTEM GIVES IT: an NVIDIA card by its compute capability, any other by its OpenCL device name —
 *   on Windows AMD's driver names a GCN or TeraScale card by its chip (`Tahiti`, `Cypress`) and a later one by its
 *   `gfx`; Intel's names the product (`Intel(R) HD Graphics 4600`). A name is matched from its start, so the system's
 *   truncation (`SILICON_FAMILIES__CARD_NAME_BYTES`) and a driver's suffix change nothing. A card wrongly placed is a
 *   ticket: its OpenCL version from `clinfo`, and the row it moves to. */

static const char* const silicon_families__opencl1_names[] = {
    /* Intel, before Gen9 */
    "Intel(R) HD Graphics 2500", "Intel(R) HD Graphics 4000", "Intel(R) HD Graphics 4200", "Intel(R) HD Graphics 4400",
    "Intel(R) HD Graphics 4600", "Intel(R) HD Graphics 5000", "Intel(R) Iris(TM) Graphics 5100",
    "Intel(R) Iris(TM) Pro Graphics 5200",
    /* AMD TeraScale: Evergreen, Northern Islands and their APUs */
    "Cypress", "Hemlock", "Juniper", "Redwood", "Cedar", "Cayman", "Antilles", "Barts", "Turks", "Caicos",
    "Sumo", "SuperSumo", "WinterPark", "BeaverCreek", "Loveland", "Devastator", "Scrapper", "Wrestler",
    /* AMD GCN 1 */
    "Tahiti", "Pitcairn", "Capeverde", "Cape Verde", "Oland", "Hainan", "gfx600", "gfx601", "gfx602",
};

/* Whether `card` is named from the start by `prefix`. */
static inline bool silicon_families__name_starts(const char* card, const char* prefix) {
    if (card == 0 || prefix == 0) return false;
    while (*prefix && *card == *prefix) { ++card; ++prefix; }
    return *prefix == 0;
}

/* Whether `card`'s OpenCL is 1.2 — the 1.2 family's, and never the 2's. */
static inline bool silicon_families__opencl1_only(const char* card) {
    if (card == 0 || card[0] == 0) return false;
    if (card[0] == 's' && card[1] == 'm' && card[2] == '_') return true;                  /* NVIDIA's, whatever the card */
    if (silicon_families__card_is(card, "Intel(R) HD Graphics")) return true;             /* Bay Trail: no number */
    for (unsigned i = 0; i < sizeof silicon_families__opencl1_names / sizeof silicon_families__opencl1_names[0]; ++i)
        if (silicon_families__name_starts(card, silicon_families__opencl1_names[i])) return true;
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_OPENCL_VERSIONS_CUH */
