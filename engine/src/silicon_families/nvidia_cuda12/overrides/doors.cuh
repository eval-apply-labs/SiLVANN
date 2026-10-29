#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_DOORS_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_DOORS_CUH
/* ══ THE OVERRIDES, INTO nn's TABLE — nvidia_cuda12 ═══════════════════════════════════════════════════════════════════
 * Each door the files beside this run faster, in its place in nn's table; every other door stays nn's generic one. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "turboquant.cuh"
#include "hadamard.cuh"
#include "deltanet.cuh"
#include "attention.cuh"
#include "result.cuh"
#include "expert.cuh"
#include "conv.cuh"
#include "vector.cuh"
#include "primitives.cuh"
#include "index.cuh"
#include "kda.cuh"
#include "hyper.cuh"

/* Into nn's table, once, when the family is first asked for it. */
static inline void nvidia_cuda12__override_doors(nn__doors* doors) {
    doors->turboquant_gemv            = nvidia_cuda12__override_turboquant_gemv;
    doors->turboquant_gemv_groups     = nvidia_cuda12__override_turboquant_gemv_groups;
    doors->turboquant_gemv_groups_sum = nvidia_cuda12__override_turboquant_gemv_groups_sum;
    doors->hadamard_rotate      = nvidia_cuda12__override_hadamard_rotate;
    doors->hadamard_blocks      = nvidia_cuda12__override_hadamard_blocks;
    doors->hadamard_ladder_rows = nvidia_cuda12__override_hadamard_ladder_rows;
    doors->deltanet_steps       = nvidia_cuda12__override_deltanet_steps;
    doors->deltanet_step        = nvidia_cuda12__override_deltanet_step;
    doors->attention_causal_scores_grouped = nvidia_cuda12__override_attention_causal_scores_grouped;
    doors->attention_causal_mix            = nvidia_cuda12__override_attention_causal_mix;
    doors->attention_causal_scores         = nvidia_cuda12__override_attention_causal_scores;
    doors->attention_scores                = nvidia_cuda12__override_attention_scores;
    doors->attention_mix                   = nvidia_cuda12__override_attention_mix;
    doors->attention_weights               = nvidia_cuda12__override_attention_weights;
    doors->attention_residual_mix          = nvidia_cuda12__override_attention_residual_mix;
    doors->attention_weights_tq            = nvidia_cuda12__override_attention_weights_tq;
    doors->attention_residual_mix_tq       = nvidia_cuda12__override_attention_residual_mix_tq;
    doors->attention_weights_rows          = nvidia_cuda12__override_attention_weights_rows;
    doors->attention_residual_mix_rows     = nvidia_cuda12__override_attention_residual_mix_rows;
    doors->attention_weights_tq_rows       = nvidia_cuda12__override_attention_weights_tq_rows;
    doors->attention_residual_mix_tq_rows  = nvidia_cuda12__override_attention_residual_mix_tq_rows;
    doors->attention_set                   = nvidia_cuda12__override_attention_set;
    doors->attention_limits                = nvidia_cuda12__override_attention_limits;
    doors->argmax               = nvidia_cuda12__override_argmax;
    doors->expert_rows          = nvidia_cuda12__override_expert_rows;
    doors->expert_rows_sum      = nvidia_cuda12__override_expert_rows_sum;
    doors->expert_groups        = nvidia_cuda12__override_expert_groups;
    doors->unfold_rows          = nvidia_cuda12__override_unfold_rows;
    doors->vector_top_k         = nvidia_cuda12__override_vector_top_k;
    doors->vector_top_k_scored  = nvidia_cuda12__override_vector_top_k_scored;
    doors->vector_sum           = nvidia_cuda12__override_vector_sum;
    doors->rmsnorm_fp32         = nvidia_cuda12__override_rmsnorm_fp32;
    doors->kda_steps_decayed             = nvidia_cuda12__override_kda_steps_decayed;
    doors->kda_step                      = nvidia_cuda12__override_kda_step;
    doors->hyper_logits                  = nvidia_cuda12__override_hyper_logits;
    doors->hyper_logits_fp32             = nvidia_cuda12__override_hyper_logits_fp32;
    doors->hyper_pre                     = nvidia_cuda12__override_hyper_pre;
    doors->hyper_pre_fp32                = nvidia_cuda12__override_hyper_pre_fp32;
    doors->index_scores                  = nvidia_cuda12__override_index_scores;
    doors->index_scores_rows             = nvidia_cuda12__override_index_scores_rows;
    doors->index_select                  = nvidia_cuda12__override_index_select;
    doors->index_select_rows             = nvidia_cuda12__override_index_select_rows;
    doors->attention_scores_grouped      = nvidia_cuda12__override_attention_scores_grouped;
    doors->attention_scores_grouped_rows = nvidia_cuda12__override_attention_scores_grouped_rows;
    doors->attention_mix_rows            = nvidia_cuda12__override_attention_mix_rows;
    doors->attention_absorb              = nvidia_cuda12__override_attention_absorb;
    doors->attention_absorb_rows         = nvidia_cuda12__override_attention_absorb_rows;
    doors->attention_expand              = nvidia_cuda12__override_attention_expand;
    doors->attention_expand_rows         = nvidia_cuda12__override_attention_expand_rows;
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_DOORS_CUH */
