#ifndef SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_PACKAGE__HEADER_CUH
#define SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_PACKAGE__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../sys/contracts/objects/kind.cuh"          /* what a thing IS */
#include "../../sys/cpu/package__header.cuh"             /* the hatch this package's rows are declared into */
#include "../contracts/objects/experts.cuh"              /* its experts table, the hand and the fault words */
#include "../contracts/objects/layer.cuh"                /* a layer's three tables: its weights, an application's memory, the work */
/* No device-wide state of its own: no hatch rows, no room. */
#define ai_thinkingmachines__inkling_2__HATCH_ROW_LIST(X)
SYS__PACKAGE__DECLARE_HATCH_ROWS(ai_thinkingmachines__inkling_2)

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_PACKAGE__HEADER_CUH */
