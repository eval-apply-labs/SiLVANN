#ifndef SILVANN__SILICON_FAMILIES_MANIFEST_CUH
#define SILVANN__SILICON_FAMILIES_MANIFEST_CUH

/* ══ THE SILICON FAMILIES, AS THE PROGRAM COMPILES THEM ═══════════════════════════════════════════════
 * ⚖ *"only through a manifest so i dont have to keep track of them"*. The program takes two things from
 * this tier: the roster — which families there are, and how a card is given to one — and the host family,
 * the only one compiled in rather than loaded. It includes this file and nothing else of the tier.
 * ⛳ A HARNESS BRINGS ITS OWN STAND-IN FOR THE HOST (`test/src_host_shim.cuh`), so it takes the roster and
 *   not the host family. */
#include "roster.cuh"
#ifndef SYS__SILICON__HARNESS
#include "host/manifest.cuh"
#endif

#endif /* SILVANN__SILICON_FAMILIES_MANIFEST_CUH */
