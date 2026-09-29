#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_MANIFEST_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_MANIFEST_CUH

/* ══ x86_avx2 — ITS ANSWERS TO EVERY PACKAGE IT SUPPORTS ══════════════════════════════════════════════
 * ⚖ *"a folder for primitives and one for overrides so it is clear what is expected to run and what are
 * performance upgrades"*: `primitives/` is what the family must answer — the seam, without which nothing
 * runs — and `overrides/` is a door's faster body, which the family runs in place of the generic one and
 * could lose without losing anything but speed. This file names the primitives; `unit.cuh` adds the
 * overrides after every package's doors exist. */
#include "primitives/sys.cuh"   /* sys's doors */
#include "primitives/nn.cuh"    /* nn's requests */

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_MANIFEST_CUH */
