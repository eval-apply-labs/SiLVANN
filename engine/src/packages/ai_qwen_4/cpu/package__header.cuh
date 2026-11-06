#ifndef SILVANN__PACKAGES_AI_QWEN_4_CPU_PACKAGE__HEADER_CUH
#define SILVANN__PACKAGES_AI_QWEN_4_CPU_PACKAGE__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../sys/contracts/objects/kind.cuh"          /* what a thing IS */
#include "../../sys/cpu/package__header.cuh"             /* the hatch this package's rows are declared into */
#include "../contracts/objects/layer.cuh"                /* a layer's table, the shape table and the fault words */
/* No device-wide state of its own: no hatch rows, no room. */
#define ai_qwen_4__HATCH_ROW_LIST(X)
SYS__PACKAGE__DECLARE_HATCH_ROWS(ai_qwen_4)

#endif /* SILVANN__PACKAGES_AI_QWEN_4_CPU_PACKAGE__HEADER_CUH */
