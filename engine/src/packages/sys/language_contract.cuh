#ifndef SILVANN__PACKAGES_SYS_LANGUAGE_CONTRACT_CUH
#define SILVANN__PACKAGES_SYS_LANGUAGE_CONTRACT_CUH
/* What this package contributes to the language, as three lists: what its kinds are called, which both
 * sides read; the words a program can call; and what each kind does when it is released, cloned or made,
 * which only the program runs. Every package has this file, because the roster reads it by name.
 * ⛔ IT MUST NOT BE BYTE-IDENTICAL TO ANOTHER PACKAGE'S, which is why it names `sys` here: the includes are
 * relative, so two packages' copies can be the same text. ⛳ And every header is guarded by its path's name
 * rather than `#pragma once`, because GCC's `#pragma once` takes two files of the same size and modification
 * time for ONE even when their bytes differ — `MEASURED`: a fresh checkout of the release, every file
 * of one time, lost `ai_glm_5_3`'s kinds (260 bytes, the same as `ai_qwen_3`'s) and failed to build. */
#include "contracts/macros/language_contract__kinds.cuh"
#include "contracts/macros/language_contract__verbs.cuh"
#include "contracts/macros/language_contract__objects.cuh"

#endif /* SILVANN__PACKAGES_SYS_LANGUAGE_CONTRACT_CUH */
