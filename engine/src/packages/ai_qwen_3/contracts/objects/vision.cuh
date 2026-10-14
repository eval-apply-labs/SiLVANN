#ifndef SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_VISION_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_VISION_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_QWEN_3__VISION__FAULT_TABLE 0x51335654ull   /* "Q3VT" — a vision verb's table is not what it reads  */
#define AI_QWEN_3__VISION__FAULT_ROOM  0x51335655ull   /* "Q3VU" — a buffer is too small for the picture       */

/* The tower's norms' epsilon: Qwen 3.5's vision blocks and merger (`nn.LayerNorm(eps=1e-6)`), every size. */
#define AI_QWEN_3__VISION__EPS 1e-6f

/* ══ A PICTURE THROUGH QWEN 3.5'S VISION TOWER — three verbs over its planes ═══════════════════════════════════════
 * (▶ `cpu/opcodes/vision__abi.cuh`; the arithmetic, Hugging Face's: `python/silvann_runtime/vision.py`.)
 *
 * `(ai_qwen_3__vision_embed pixels pos table rows x)` — a picture's patches (`pixels`, a row of 3 x 2 x 16 x 16 halves a
 * patch) through the patch embedding, its bias and the position table's rows for them (`pos`, a row a patch) into `x`. */
#define AI_QWEN_3__VE__PATCH         0u    /* the patch embedding's codes: a row of `IN` halves an output (16 bits)   */
#define AI_QWEN_3__VE__PATCH_LUT     1u
#define AI_QWEN_3__VE__PATCH_B       2u
#define AI_QWEN_3__VE__SCRATCH       3u    /* the pairs the GEMM door takes, eight bytes a row                      */
#define AI_QWEN_3__VE__HIDDEN        4u    /* integers from here on                                                 */
#define AI_QWEN_3__VE__IN            5u
#define AI_QWEN_3__VE__D             6u
#define AI_QWEN_3__VE__TABLE         7u

/* `(ai_qwen_3__vision_block x positions table rows)` — a block over a picture's `rows` patches, its residual `x` in place:
 * LayerNorm -> q, k, v (with their bias) -> the axial rotary from `positions` (a patch's row and column, two uint32) ->
 * every patch to every patch -> the projection, added; LayerNorm -> the MLP (GELU's tanh form), added — the MLP
 * `CHUNK` rows a step, so its rows are the only ones the scratch holds a step's worth of. */
#define AI_QWEN_3__VB__NORM1_W       0u
#define AI_QWEN_3__VB__NORM1_B       1u
#define AI_QWEN_3__VB__QKV           2u
#define AI_QWEN_3__VB__QKV_LUT       3u
#define AI_QWEN_3__VB__QKV_B         4u
#define AI_QWEN_3__VB__PROJ          5u
#define AI_QWEN_3__VB__PROJ_LUT      6u
#define AI_QWEN_3__VB__PROJ_B        7u
#define AI_QWEN_3__VB__NORM2_W       8u
#define AI_QWEN_3__VB__NORM2_B       9u
#define AI_QWEN_3__VB__FC1           10u
#define AI_QWEN_3__VB__FC1_LUT       11u
#define AI_QWEN_3__VB__FC1_B         12u
#define AI_QWEN_3__VB__FC2           13u
#define AI_QWEN_3__VB__FC2_LUT       14u
#define AI_QWEN_3__VB__FC2_B         15u
#define AI_QWEN_3__VB__SIGNS         16u   /* the pack's first 512 signs, as halves: the rotation a row of codes was made with */
#define AI_QWEN_3__VB__SCRATCH       17u   /* the pairs, then in halves h · hrot · qkv · m1 · m2, then the angles (floats):
                                                hrot the larger of `rows` rows of HIDDEN and CHUNK rows of INTER        */
#define AI_QWEN_3__VB__HIDDEN        18u   /* integers from here on                                                 */
#define AI_QWEN_3__VB__HEADS         19u
#define AI_QWEN_3__VB__HEAD_DIM      20u
#define AI_QWEN_3__VB__INTER         21u
#define AI_QWEN_3__VB__D_QKV         22u
#define AI_QWEN_3__VB__D_PROJ        23u
#define AI_QWEN_3__VB__D_FC1         24u
#define AI_QWEN_3__VB__D_FC2         25u
#define AI_QWEN_3__VB__THETA         26u
#define AI_QWEN_3__VB__CHUNK         27u
#define AI_QWEN_3__VB__TABLE         28u

/* `(ai_qwen_3__vision_merger x table rows out)` — the tower's last word: LayerNorm a patch, `MERGE` neighbouring patches
 * (in the processor's order, already side by side) as one row, -> GELU's erf form -> the text model's width, a row of
 * `out` a token. */
#define AI_QWEN_3__VM__NORM_W        0u
#define AI_QWEN_3__VM__NORM_B        1u
#define AI_QWEN_3__VM__FC1           2u
#define AI_QWEN_3__VM__FC1_LUT       3u
#define AI_QWEN_3__VM__FC1_B         4u
#define AI_QWEN_3__VM__FC2           5u
#define AI_QWEN_3__VM__FC2_LUT       6u
#define AI_QWEN_3__VM__FC2_B         7u
#define AI_QWEN_3__VM__SIGNS         8u
#define AI_QWEN_3__VM__SCRATCH       9u    /* the pairs, then in halves h · hrot · m · mrot                         */
#define AI_QWEN_3__VM__HIDDEN        10u   /* integers from here on                                                 */
#define AI_QWEN_3__VM__MERGE         11u
#define AI_QWEN_3__VM__OUT           12u
#define AI_QWEN_3__VM__D_FC1         13u
#define AI_QWEN_3__VM__D_FC2         14u
#define AI_QWEN_3__VM__TABLE         15u

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_VISION_CUH */
