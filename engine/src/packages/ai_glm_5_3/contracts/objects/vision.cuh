#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_OBJECTS_VISION_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_OBJECTS_VISION_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_GLM_5_3__VISION__FAULT_TABLE 0x474C3556ull  /* "GL5V" — a vision verb's table is not what it reads   */
#define AI_GLM_5_3__VISION__FAULT_ROOM  0x474C3557ull  /* "GL5W" — a buffer is too small for the picture        */

/* ══ A PICTURE THROUGH GLM 5.3'S VISION TOWER — three verbs over its planes ═══════════════════════════════════════
 * (▶ `cpu/opcodes/vision__abi.cuh`; the arithmetic, Hugging Face's `Glm5NextVisionModel`, written out in numpy in
 * `python/silvann_runtime/vision_glm.py`.) A table is its planes, then its integers from `HIDDEN`, then its reals.
 *
 * `(ai_glm_5_3__vision_embed pixels table rows x)` — a picture's patches (`pixels`, a row of 3 x 2 x 14 x 14 halves a
 * patch) through the patch embedding and its bias into `x`. */
#define AI_GLM_5_3__VE__PATCH         0u    /* the patch embedding: a row of `IN` halves an output (16 bits)          */
#define AI_GLM_5_3__VE__PATCH_LUT     1u    /* not read at 16 bits                                                   */
#define AI_GLM_5_3__VE__PATCH_B       2u
#define AI_GLM_5_3__VE__SCRATCH       3u    /* the pairs the GEMM door takes, eight bytes a row                      */
#define AI_GLM_5_3__VE__PLANES        4u
#define AI_GLM_5_3__VE__HIDDEN        4u    /* integers from here on                                                 */
#define AI_GLM_5_3__VE__IN            5u
#define AI_GLM_5_3__VE__D             6u
#define AI_GLM_5_3__VE__TABLE         7u

/* `(ai_glm_5_3__vision_block x positions table rows)` — a block over a picture's `rows` patches, its residual `x` in place:
 * RMSNorm -> q, k, v (with their bias) -> each head's q and k RMSNormed -> the axial rotary from `positions` (a patch's
 * row and column, two uint32) -> every patch to every patch -> the projection (with its bias), added; RMSNorm -> gate
 * and up (with their bias, the bias plane the two side by side) -> the clamped SwiGLU -> down (with its bias), added —
 * the MLP `CHUNK` rows a step. */
#define AI_GLM_5_3__VB__NORM1_W       0u
#define AI_GLM_5_3__VB__QKV           1u
#define AI_GLM_5_3__VB__QKV_LUT       2u
#define AI_GLM_5_3__VB__QKV_B         3u
#define AI_GLM_5_3__VB__Q_NORM        4u
#define AI_GLM_5_3__VB__K_NORM        5u
#define AI_GLM_5_3__VB__PROJ          6u
#define AI_GLM_5_3__VB__PROJ_LUT      7u
#define AI_GLM_5_3__VB__PROJ_B        8u
#define AI_GLM_5_3__VB__NORM2_W       9u
#define AI_GLM_5_3__VB__GATE          10u
#define AI_GLM_5_3__VB__GATE_LUT      11u
#define AI_GLM_5_3__VB__UP            12u
#define AI_GLM_5_3__VB__UP_LUT        13u
#define AI_GLM_5_3__VB__GU_B          14u   /* the gate's bias, then the up's: a row of the two answers' width       */
#define AI_GLM_5_3__VB__DOWN          15u
#define AI_GLM_5_3__VB__DOWN_LUT      16u
#define AI_GLM_5_3__VB__DOWN_B        17u
#define AI_GLM_5_3__VB__SIGNS         18u   /* the pack's first 512 signs, as halves: the rotation a row of codes was made with */
#define AI_GLM_5_3__VB__SCRATCH       19u   /* the pairs, then in halves h · hrot · qkv · gu · act · actr · m, then the angles */
#define AI_GLM_5_3__VB__PLANES        20u
#define AI_GLM_5_3__VB__HIDDEN        20u   /* integers from here on                                                 */
#define AI_GLM_5_3__VB__HEADS         21u
#define AI_GLM_5_3__VB__HEAD_DIM      22u
#define AI_GLM_5_3__VB__INTER         23u
#define AI_GLM_5_3__VB__D_QKV         24u
#define AI_GLM_5_3__VB__D_PROJ        25u
#define AI_GLM_5_3__VB__D_GATE        26u   /* the gate's and the up's: one width                                    */
#define AI_GLM_5_3__VB__D_DOWN        27u
#define AI_GLM_5_3__VB__THETA         28u
#define AI_GLM_5_3__VB__CHUNK         29u
#define AI_GLM_5_3__VB__EPS           30u   /* reals from here on                                                    */
#define AI_GLM_5_3__VB__LIMIT         31u
#define AI_GLM_5_3__VB__TABLE         32u

/* `(ai_glm_5_3__vision_tail x table rows out)` — the tower's last word: RMSNorm a patch; `MERGE` neighbouring patches (in
 * the processor's order, side by side) as one row through the downsampling (16 bits, its columns laid out for the
 * patches as they lie) and its bias; then the merger — a projection, LayerNorm (with its bias), GELU's erf form, the
 * clamped SwiGLU — into a row of `out` a token. The merger's matrices carry no bias. Its MLP `CHUNK` tokens a step. */
#define AI_GLM_5_3__VT__NORM_W        0u
#define AI_GLM_5_3__VT__DOWNS         1u
#define AI_GLM_5_3__VT__DOWNS_LUT     2u    /* not read at 16 bits                                                   */
#define AI_GLM_5_3__VT__DOWNS_B       3u
#define AI_GLM_5_3__VT__PROJ          4u
#define AI_GLM_5_3__VT__PROJ_LUT      5u
#define AI_GLM_5_3__VT__LN_W          6u
#define AI_GLM_5_3__VT__LN_B          7u
#define AI_GLM_5_3__VT__GATE          8u
#define AI_GLM_5_3__VT__GATE_LUT      9u
#define AI_GLM_5_3__VT__UP            10u
#define AI_GLM_5_3__VT__UP_LUT        11u
#define AI_GLM_5_3__VT__DOWN          12u
#define AI_GLM_5_3__VT__DOWN_LUT      13u
#define AI_GLM_5_3__VT__SIGNS         14u
#define AI_GLM_5_3__VT__SCRATCH       15u   /* the pairs, then in halves h · y · yrot · gu · act · actr               */
#define AI_GLM_5_3__VT__PLANES        16u
#define AI_GLM_5_3__VT__HIDDEN        16u   /* integers from here on                                                 */
#define AI_GLM_5_3__VT__MERGE         17u
#define AI_GLM_5_3__VT__OUT           18u
#define AI_GLM_5_3__VT__INTER         19u   /* the merger's gate and up width                                        */
#define AI_GLM_5_3__VT__D_DOWNS       20u
#define AI_GLM_5_3__VT__D_PROJ        21u
#define AI_GLM_5_3__VT__D_GATE        22u
#define AI_GLM_5_3__VT__D_DOWN        23u
#define AI_GLM_5_3__VT__CHUNK         24u
#define AI_GLM_5_3__VT__EPS           25u   /* reals from here on: the patches' RMSNorm                              */
#define AI_GLM_5_3__VT__LN_EPS        26u   /* the merger's LayerNorm                                                */
#define AI_GLM_5_3__VT__LIMIT         27u
#define AI_GLM_5_3__VT__TABLE         28u

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_OBJECTS_VISION_CUH */
