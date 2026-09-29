#ifndef SILVANN__PACKAGES_MANIFEST_CUH
#define SILVANN__PACKAGES_MANIFEST_CUH
/* ⛔ GENERATED FILE — DO NOT EDIT. Your change will be overwritten the next time anything builds.
 *
 * Written by `scripts/src_assemble.sh` from the roster in `manifest__header.cuh`, which is where the
 * reasoning for both files lives; a package is added by its folder, which `scripts/src_enroll.py` enrols.
 * This one is a pure function of that row set: one include per package per phase, in row order.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "manifest__header.cuh"

/* ── PHASE ZERO — WHAT EVERY PACKAGE PUBLISHES INTO THE LANGUAGE ─────────────────────────────────
 *
 * A contract file needs nothing and expands nothing, so every package's rows can be read before
 * anything else in the tree has been. That is what makes this phase possible, and the kinds are why it
 * is necessary: a kind is compared inside declarations — whether a node carries a reference, what an
 * empty register row answers — so its constants have to exist before the first header, while a verb is
 * only ever compared inside an implementation and can be named after the last one. */
#include "sys/language_contract.cuh"
#include "nn/language_contract.cuh"
#include "ai_qwen_3/language_contract.cuh"
#include "ai_glm_5_3/language_contract.cuh"

/* Every package has now written its rows, and the tag that makes a kind unique is applied once, here,
 * for all of them. */
#include "language_contract_kinds.cuh"

/* ── PHASE ONE — WHAT EVERY PACKAGE DECLARES ────────────────────────────────────────────────────── */
#include "sys/manifest__header.cuh"
#include "nn/manifest__header.cuh"
#include "ai_qwen_3/manifest__header.cuh"
#include "ai_glm_5_3/manifest__header.cuh"

#define PACKAGE_HEADER_PRESENT_CHECK(id, name, ...) \
    static_assert(PACKAGE_##name##_HEADER_PRESENT == 1, \
                  "package '" #name "' is listed and its header phase is not included");
PACKAGE_LIST(PACKAGE_HEADER_PRESENT_CHECK)
#undef PACKAGE_HEADER_PRESENT_CHECK

/* ── THE VERBS, NAMED ────────────────────────────────────────────────────────────────────────────
 *
 * Every package has now written its rows, and nothing before this point asks what a verb is called. A
 * gather that reads EVERY package cannot sit inside one of them: the packages listed after it have not
 * been read yet, so it pastes a macro nobody has defined and the compiler reports a syntax error inside
 * whichever package happened to be next — pages away from the row that caused it. Anything that reads
 * all the packages belongs beside the list, and runs once they have all spoken. */
#include "language_contract_verbs.cuh"

/* ── PHASE TWO — WHAT EVERY PACKAGE DEFINES ─────────────────────────────────────────────────────── */
#include "sys/manifest.cuh"
#include "nn/manifest.cuh"
#include "ai_qwen_3/manifest.cuh"
#include "ai_glm_5_3/manifest.cuh"

#define PACKAGE_PRESENT_CHECK(id, name, ...) \
    static_assert(PACKAGE_##name##_PRESENT == 1, \
                  "package '" #name "' is listed and its implementation is not included");
PACKAGE_LIST(PACKAGE_PRESENT_CHECK)
#undef PACKAGE_PRESENT_CHECK

#endif /* SILVANN__PACKAGES_MANIFEST_CUH */
