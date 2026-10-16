#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_BLOCK_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_BLOCK_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_QWEN_IMAGE__BLOCK__FAULT_TABLE 0x51495454ull   /* "QITT" — the block's table is not what it reads      */
#define AI_QWEN_IMAGE__BLOCK__FAULT_ROOM  0x51495455ull   /* "QITU" — a buffer is too small for the rows           */

/* Qwen-Image 2.1's norms' epsilon: the blocks' LayerNorms and the per-head RMSNorms of q and k (`eps=1e-6`). */
#define AI_QWEN_IMAGE__EPS 1e-6f

/* ══ A BLOCK OF QWEN-IMAGE 2.1's DIFFUSION TRANSFORMER — one verb over its planes ═════════════════════════════
 * (▶ `cpu/opcodes/block__abi.cuh`; the arithmetic, diffusers' `QwenImage21TransformerBlock`.)
 *
 * `(ai_qwen_image__block x mod cs limits table rows first prefix)` — the block over the sequence's rows `first ..
 * first + rows`, their residual `x` in place (a row of HIDDEN halves each):
 *   h = LN(x)·w₁; q, k, v = W·h; q and k normed a head (RMSNorm) and turned by the row's angles (`cs`, the sequence's
 *   rotary table: a row of HEAD_DIM floats, the cosines then the sines); every row to the keys `0 .. limits[row]`;
 *   x += g₁·(W_out·a); h = LN(x)·w₂; x += g₂·W_down(silu(W_gate·h)·W_up·h)
 * `mod` holds the step's modulation, two rows of four HIDDEN halves — the prompt's (t = 0), then the picture's: `w₁`
 * (1 + scale₁), `g₁` (tanh of gate₁), `w₂`, `g₂` — and a row of the sequence below `prefix` takes the first. The
 * prompt's keys and values (the rows below `prefix`) are the block's own, kept in `PREFIX_K` / `PREFIX_V` when `first`
 * is 0 and read back from there when it is not: a later step computes the picture's rows alone (`first` = `prefix`).
 * q and k are the pack's rows reordered a head, as the runtime loads them, so that nn's rotary pairs element k with
 * k + HEAD_DIM/2 where the transformer paired 2k with 2k + 1; their norms' weights likewise. */
#define AI_QWEN_IMAGE__B__Q            0u    /* each matrix: its codes, then its scales (a row's fp16)                 */
#define AI_QWEN_IMAGE__B__Q_LUT        1u
#define AI_QWEN_IMAGE__B__K            2u
#define AI_QWEN_IMAGE__B__K_LUT        3u
#define AI_QWEN_IMAGE__B__V            4u
#define AI_QWEN_IMAGE__B__V_LUT        5u
#define AI_QWEN_IMAGE__B__OUT          6u
#define AI_QWEN_IMAGE__B__OUT_LUT      7u
#define AI_QWEN_IMAGE__B__GATE         8u
#define AI_QWEN_IMAGE__B__GATE_LUT     9u
#define AI_QWEN_IMAGE__B__UP           10u   /* diffusers' `img_mlp.proj`                                             */
#define AI_QWEN_IMAGE__B__UP_LUT       11u
#define AI_QWEN_IMAGE__B__DOWN         12u   /* diffusers' `img_mlp.out`                                              */
#define AI_QWEN_IMAGE__B__DOWN_LUT     13u
#define AI_QWEN_IMAGE__B__NORM_Q       14u   /* HEAD_DIM halves, reordered as q is                                    */
#define AI_QWEN_IMAGE__B__NORM_K       15u
#define AI_QWEN_IMAGE__B__PREFIX_K     16u   /* PREFIX_MAX rows of HIDDEN halves: the prompt's keys, kept from step one */
#define AI_QWEN_IMAGE__B__PREFIX_V     17u
#define AI_QWEN_IMAGE__B__ZERO         18u   /* HIDDEN halves of 0: the LayerNorms' bias                              */
#define AI_QWEN_IMAGE__B__SIGNS        19u   /* the pack's first 512 signs, as halves: the rotation a row of codes was made with */
#define AI_QWEN_IMAGE__B__SCRATCH      20u   /* the pairs, then in halves h · hrot · q · keys · values · a · m1 · m2 —
                                                keys and values the sequence's rows, hrot the larger of `rows` rows of
                                                HIDDEN and CHUNK rows of INTER; every block's the same                 */
#define AI_QWEN_IMAGE__B__HIDDEN       21u   /* integers from here on                                                 */
#define AI_QWEN_IMAGE__B__HEADS        22u
#define AI_QWEN_IMAGE__B__HEAD_DIM     23u
#define AI_QWEN_IMAGE__B__INTER        24u
#define AI_QWEN_IMAGE__B__D_QKV        25u   /* the widths: q, k and v's; the output's; gate and up's; down's        */
#define AI_QWEN_IMAGE__B__D_OUT        26u
#define AI_QWEN_IMAGE__B__D_MLP        27u
#define AI_QWEN_IMAGE__B__D_DOWN       28u
#define AI_QWEN_IMAGE__B__PREFIX_MAX   29u
#define AI_QWEN_IMAGE__B__CHUNK        30u   /* the MLP's rows a step: the only rows the scratch holds INTER wide     */
#define AI_QWEN_IMAGE__B__TABLE        31u

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_BLOCK_CUH */
