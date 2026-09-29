#ifndef SILVANN__PACKAGES_MANIFEST__GPU_CUH
#define SILVANN__PACKAGES_MANIFEST__GPU_CUH
/* ⛔ GENERATED FILE — DO NOT EDIT. Your change will be overwritten the next time anything builds.
 *
 * Written by `scripts/src_assemble.sh` from the roster in `manifest__header.cuh`, beside `manifest.cuh`:
 * THE PACKAGES, AS A SILICON FAMILY'S BINARY COMPILES THEM. A family's binary holds each package's card
 * side — its doors and the kernels behind them — compiled against the family's answers; which headers
 * that takes is each package's business, named in its own `<pkg>/manifest__gpu.cuh`, and a family
 * includes this file and nothing else of the packages. ⛳ EVERY PACKAGE HAS ONE, even one that runs
 * nothing on a card: it carries the package's requests (`contracts/abi/gpu.cuh`), which every family's
 * table is gathered over. The order is the roster's: the lists first, then each package.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "manifest__header.cuh"          /* the roster, and each package's id */

/* The lists — they expand nothing, so a card side can take them. */
#include "sys/language_contract.cuh"
#include "nn/language_contract.cuh"
#include "ai_qwen_3/language_contract.cuh"
#include "ai_glm_5_3/language_contract.cuh"
#include "language_contract_kinds.cuh"

/* Each package's card side. */
#include "sys/manifest__gpu.cuh"
#include "nn/manifest__gpu.cuh"
#include "ai_qwen_3/manifest__gpu.cuh"
#include "ai_glm_5_3/manifest__gpu.cuh"

#endif /* SILVANN__PACKAGES_MANIFEST__GPU_CUH */
