#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_OBJECTS_VISION_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_OBJECTS_VISION_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_DEEPSEEK_4__VISION__FAULT_TABLE 0x44533456ull  /* "DS4V" — a vision verb's table is not what it reads    */
#define AI_DEEPSEEK_4__VISION__FAULT_ROOM  0x44533457ull  /* "DS4W" — a buffer is too small for the picture         */

/* ══ A PICTURE THROUGH DEEPSEEK-V4.1'S VISION TOWER — three verbs over its planes ═══════════════════════════════════
 * (▶ `cpu/opcodes/vision__abi.cuh`; the arithmetic, the release's `inference/vision.py`.) A table is its planes, then its
 * integers from `HIDDEN`, then its reals.
 *
 * `(ai_deepseek_4__vision_embed pixels table rows x)` — a picture's patches (`pixels`, a row of 3 x 14 x 14 halves a
 * patch, channel by channel) through the patch embedding and its bias into `x`. */
#define AI_DEEPSEEK_4__VE__PATCH         0u    /* the patch embedding: a row of `IN` halves an output (16 bits)          */
#define AI_DEEPSEEK_4__VE__PATCH_LUT     1u    /* not read at 16 bits                                                   */
#define AI_DEEPSEEK_4__VE__PATCH_B       2u
#define AI_DEEPSEEK_4__VE__SCRATCH       3u    /* the pairs the GEMM door takes, eight bytes a row                      */
#define AI_DEEPSEEK_4__VE__PLANES        4u
#define AI_DEEPSEEK_4__VE__HIDDEN        4u    /* integers from here on                                                 */
#define AI_DEEPSEEK_4__VE__IN            5u
#define AI_DEEPSEEK_4__VE__D             6u
#define AI_DEEPSEEK_4__VE__TABLE         7u

/* `(ai_deepseek_4__vision_block x positions table rows)` — a block over a picture's `rows` patches, its residual `x` in
 * place: RMSNorm -> q, k, v (with their bias) -> the axial rotary from `positions` (a patch's row and column, two uint32)
 * on q and k -> every patch to every patch -> the projection (with its bias), added; RMSNorm -> gate and up, one matrix of
 * the two (gate's rows, then up's) -> SwiGLU -> down, added — the MLP `CHUNK` rows a step, its matrices without bias. */
#define AI_DEEPSEEK_4__VB__NORM1_W       0u
#define AI_DEEPSEEK_4__VB__QKV           1u
#define AI_DEEPSEEK_4__VB__QKV_LUT       2u
#define AI_DEEPSEEK_4__VB__QKV_B         3u
#define AI_DEEPSEEK_4__VB__PROJ          4u
#define AI_DEEPSEEK_4__VB__PROJ_LUT      5u
#define AI_DEEPSEEK_4__VB__PROJ_B        6u
#define AI_DEEPSEEK_4__VB__NORM2_W       7u
#define AI_DEEPSEEK_4__VB__GU            8u    /* gate's `INTER` rows, then up's                                        */
#define AI_DEEPSEEK_4__VB__GU_LUT        9u
#define AI_DEEPSEEK_4__VB__DOWN          10u
#define AI_DEEPSEEK_4__VB__DOWN_LUT      11u
#define AI_DEEPSEEK_4__VB__SIGNS         12u   /* the pack's first 512 signs, as halves: the rotation a row of codes was made with */
#define AI_DEEPSEEK_4__VB__SCRATCH       13u   /* the pairs, then in halves h · hrot · qkv · gu · act · actr · m, then the angles */
#define AI_DEEPSEEK_4__VB__PLANES        14u
#define AI_DEEPSEEK_4__VB__HIDDEN        14u   /* integers from here on                                                 */
#define AI_DEEPSEEK_4__VB__HEADS         15u
#define AI_DEEPSEEK_4__VB__HEAD_DIM      16u
#define AI_DEEPSEEK_4__VB__INTER         17u
#define AI_DEEPSEEK_4__VB__D_QKV         18u
#define AI_DEEPSEEK_4__VB__D_PROJ        19u
#define AI_DEEPSEEK_4__VB__D_GU          20u
#define AI_DEEPSEEK_4__VB__D_DOWN        21u
#define AI_DEEPSEEK_4__VB__THETA         22u
#define AI_DEEPSEEK_4__VB__CHUNK         23u
#define AI_DEEPSEEK_4__VB__EPS           24u   /* reals from here on                                                    */
#define AI_DEEPSEEK_4__VB__TABLE         25u

/* `(ai_deepseek_4__vision_tail x table h w out)` — the tower's last norm over its `h` x `w` patches, then the aligner:
 * each 3 x 3 block of patches (the picture's last rows and columns padded with zeros to whole blocks) as one row of
 * nine patches side by side — the block's top row first, a patch's channels together — through the first map (its
 * columns in that order) and its bias, GELU's erf form, the second map and its bias, into a row of `out` a block, the
 * blocks in reading order. */
#define AI_DEEPSEEK_4__VT__NORM_W        0u
#define AI_DEEPSEEK_4__VT__W1            1u
#define AI_DEEPSEEK_4__VT__W1_LUT        2u
#define AI_DEEPSEEK_4__VT__W1_B          3u
#define AI_DEEPSEEK_4__VT__W2            4u
#define AI_DEEPSEEK_4__VT__W2_LUT        5u
#define AI_DEEPSEEK_4__VT__W2_B          6u
#define AI_DEEPSEEK_4__VT__SIGNS         7u
#define AI_DEEPSEEK_4__VT__SCRATCH       8u    /* the pairs, then in halves h · u · urot · y                            */
#define AI_DEEPSEEK_4__VT__PLANES        9u
#define AI_DEEPSEEK_4__VT__HIDDEN        9u    /* integers from here on                                                 */
#define AI_DEEPSEEK_4__VT__RATIO         10u   /* the block's side                                                      */
#define AI_DEEPSEEK_4__VT__OUT           11u
#define AI_DEEPSEEK_4__VT__D_W1          12u
#define AI_DEEPSEEK_4__VT__D_W2          13u
#define AI_DEEPSEEK_4__VT__EPS           14u   /* reals from here on                                                    */
#define AI_DEEPSEEK_4__VT__TABLE         15u

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_OBJECTS_VISION_CUH */
