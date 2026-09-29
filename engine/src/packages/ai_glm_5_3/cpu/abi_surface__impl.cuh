#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CPU_ABI_SURFACE__IMPL_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CPU_ABI_SURFACE__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/abi/cpu.cuh"   /* the call, and — from sys — the table it is described by */

/* No calls of its own for a host language: a program reaches the package through its verbs. */
#ifndef SYS__SILICON__HARNESS
static const sys__abi_surface ai_glm_5_3__abi_surface__zzprivate_surface = {
    "ai_glm_5_3: GLM 5.3 Flash's layer, a verb a device seam; nothing to call from a host.",
    0u, 0,
};
extern "C" const sys__abi_surface* ai_glm_5_3_abi_surface(void) { return &ai_glm_5_3__abi_surface__zzprivate_surface; }
#endif

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CPU_ABI_SURFACE__IMPL_CUH */
