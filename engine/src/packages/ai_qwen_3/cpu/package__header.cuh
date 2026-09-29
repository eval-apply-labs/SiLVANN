#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_PACKAGE__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_PACKAGE__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../sys/contracts/objects/kind.cuh"          /* what a thing IS */
#include "../../sys/cpu/package__header.cuh"             /* the hatch this package's rows are declared into */
#include "../contracts/objects/moe.cuh"                  /* its constants and fault words */

/* No device-wide state of its own: no hatch rows, no room. */
#define ai_qwen_3__HATCH_ROW_LIST(X)
SYS__PACKAGE__DECLARE_HATCH_ROWS(ai_qwen_3)

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_PACKAGE__HEADER_CUH */
