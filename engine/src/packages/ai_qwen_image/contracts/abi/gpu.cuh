#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_ABI_GPU_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_ABI_GPU_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
/* ══ ai_qwen_image's DOORS — NONE ═══════════════════════════════════════════════════════════════════════════
 * Its verb launches nn's doors. The table still exists, holding only its size, because every listed package has one
 * in each family. */
#define ai_qwen_image__CONTRACT__DOORS(X, PKG)
#ifndef __OPENCL_C_VERSION__
typedef struct ai_qwen_image__doors {
    uint32_t size;                                   /* sizeof this table as the family built it */
} ai_qwen_image__doors;
#endif

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_ABI_GPU_CUH */
