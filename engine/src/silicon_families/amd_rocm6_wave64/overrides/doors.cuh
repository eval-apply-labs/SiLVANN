#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DOORS_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DOORS_CUH
/* ══ THE OVERRIDES, INTO nn's TABLE — amd_rocm6_wave64 ════════════════════════════════════════════════════════════════
 * Each door the files beside this run faster, in its place in nn's table; every other door stays nn's generic one. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "turboquant.cuh"
#include "hadamard.cuh"
#include "attention.cuh"
#include "deltanet.cuh"
#include "expert.cuh"
#include "primitives.cuh"
#include "result.cuh"
#include "vector.cuh"

/* Into nn's table, once, when the family is first asked for it. */
static inline void amd_rocm6_wave64__override_doors(nn__doors* doors) {
    doors->turboquant_gemv      = amd_rocm6_wave64__override_turboquant_gemv;
    doors->turboquant_gemv_int8 = amd_rocm6_wave64__override_turboquant_gemv_int8;
    doors->hadamard_blocks              = amd_rocm6_wave64__override_hadamard_blocks;
    doors->attention_weights            = amd_rocm6_wave64__override_attention_weights;
    doors->attention_residual_mix       = amd_rocm6_wave64__override_attention_residual_mix;
    doors->attention_scores             = amd_rocm6_wave64__override_attention_scores;
    doors->attention_mix                = amd_rocm6_wave64__override_attention_mix;
    doors->deltanet_steps               = amd_rocm6_wave64__override_deltanet_steps;
    doors->deltanet_step                = amd_rocm6_wave64__override_deltanet_step;
    doors->hadamard_rotate      = amd_rocm6_wave64__override_hadamard_rotate;
    doors->turboquant_gemv_groups     = amd_rocm6_wave64__override_gemv_groups;
    doors->turboquant_gemv_groups_sum = amd_rocm6_wave64__override_gemv_groups_sum;
    doors->expert_rows          = amd_rocm6_wave64__override_expert_rows;
    doors->expert_rows_sum      = amd_rocm6_wave64__override_expert_rows_sum;
    doors->expert_groups        = amd_rocm6_wave64__override_expert_groups;
    doors->rmsnorm              = amd_rocm6_wave64__override_rmsnorm;
    doors->rmsnorm_rows         = amd_rocm6_wave64__override_rmsnorm_rows;
    doors->argmax               = amd_rocm6_wave64__override_argmax;
    doors->vector_top_k         = amd_rocm6_wave64__override_vector_top_k;
    doors->vector_penalize      = amd_rocm6_wave64__override_vector_penalize;
    doors->attention_merge      = amd_rocm6_wave64__override_attention_merge;
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DOORS_CUH */
