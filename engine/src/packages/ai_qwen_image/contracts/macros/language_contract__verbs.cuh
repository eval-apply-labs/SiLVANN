#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
/* This file needs nothing: a macro body names nothing until something expands it. */
/* ══ ai_qwen_image's VERBS — Qwen-Image 2.1's diffusion transformer, a block a verb ══════════════════════════════
 * ⚖ 2026-10-15: *"make qwen image 0.10.0"*, its block a package verb as the vision tower's (the architect's choice). The
 * block over a planes table, launching nn's doors directly (▶ `contracts/objects/block.cuh`); and what a step does
 * around the blocks, over the transformer's own table (▶ `contracts/objects/step.cuh`); and the VAE's convolution and its
 * middle's attention (▶ `contracts/objects/vae.cuh`). */
#define ai_qwen_image__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                      \
    X(PKG, 0, "ai_qwen_image__block",   ai_qwen_image__block__zzabi_adapter,   AI_QWEN_IMAGE__BLOCK)         \
    X(PKG, 1, "ai_qwen_image__time",    ai_qwen_image__time__zzabi_adapter,    AI_QWEN_IMAGE__TIME)          \
    X(PKG, 2, "ai_qwen_image__text",    ai_qwen_image__text__zzabi_adapter,    AI_QWEN_IMAGE__TEXT)          \
    X(PKG, 3, "ai_qwen_image__latents", ai_qwen_image__latents__zzabi_adapter, AI_QWEN_IMAGE__LATENTS)       \
    X(PKG, 4, "ai_qwen_image__out",     ai_qwen_image__out__zzabi_adapter,     AI_QWEN_IMAGE__OUT)           \
    X(PKG, 5, "ai_qwen_image__conv",    ai_qwen_image__conv__zzabi_adapter,    AI_QWEN_IMAGE__CONV)          \
    X(PKG, 6, "ai_qwen_image__vae_attention", ai_qwen_image__vae_attention__zzabi_adapter, AI_QWEN_IMAGE__VAE_ATTENTION)

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
