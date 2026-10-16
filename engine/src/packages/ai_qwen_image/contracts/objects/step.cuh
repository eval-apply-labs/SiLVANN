#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_STEP_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_STEP_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_QWEN_IMAGE__STEP__FAULT_TABLE 0x51495354ull    /* "QIST" — the transformer's table is not what it reads */
#define AI_QWEN_IMAGE__STEP__FAULT_ROOM  0x51495355ull    /* "QISU" — a buffer is too small for the rows           */

/* ══ WHAT A STEP DOES AROUND THE BLOCKS — four verbs over the transformer's own planes ═══════════════════════════
 * (▶ `cpu/opcodes/step__abi.cuh`; the arithmetic, diffusers' `QwenImage21Transformer2DModel`.)
 *
 * `(ai_qwen_image__time sinusoids step mod row scale table)` — the modulation of the timestep whose sinusoid is row
 *   `step` of `sinusoids` (FREQ halves a row: cos(1000·t·fₖ), then sin, as the runtime writes them): temb =
 *   W₂·silu(W₁·sinusoid); `mod`'s row `row` (0 the prompt's, 1 the picture's — ▶ `block.cuh`) = 1 + scale₁, tanh(gate₁),
 *   1 + scale₂, tanh(gate₂) of W_mod·silu(temb); `scale` = 1 + W_norm_out·silu(temb), the last norm's.
 * `(ai_qwen_image__text encoder x table rows from first)` — a run of the prompt's text rows into the sequence: `x`'s rows
 *   `first ..` = W_out·gelu_tanh(W_in·(RMSNorm(e)·(1 + w))), `e` `encoder`'s rows `from ..`, rows of CONTEXT halves.
 * `(ai_qwen_image__latents latents x table rows from first)` — a picture's latent rows into the sequence: `x`'s rows
 *   `first ..` = W_img·l, `l` `latents`' rows `from ..`, rows of CHANNELS halves.
 * `(ai_qwen_image__out x scale prediction latents table rows first dsigma)` — the step's end over `x`'s rows `first ..`,
 *   the picture's: `prediction` = W_proj·(LayerNorm(x)·scale), rows of CHANNELS; `latents` += dsigma·prediction, the
 *   flow's Euler step.
 * Every product is nn's `expert_rows`, its rows rotated in the pack's ladder first where its codes are below 16 bits; the
 * latents' matrix is 64 columns wide, below the ladder's 128, so it is 16 bits — the runtime writes it decoded. */
#define AI_QWEN_IMAGE__G__IMG          0u    /* each matrix: its codes, then its scales (a row's fp16; unread at 16 bits) */
#define AI_QWEN_IMAGE__G__IMG_LUT      1u
#define AI_QWEN_IMAGE__G__TXT_IN       2u
#define AI_QWEN_IMAGE__G__TXT_IN_LUT   3u
#define AI_QWEN_IMAGE__G__TXT_OUT      4u
#define AI_QWEN_IMAGE__G__TXT_OUT_LUT  5u
#define AI_QWEN_IMAGE__G__T1           6u    /* the timestep's two: FREQ in, HIDDEN out; HIDDEN to HIDDEN                */
#define AI_QWEN_IMAGE__G__T1_LUT       7u
#define AI_QWEN_IMAGE__G__T2           8u
#define AI_QWEN_IMAGE__G__T2_LUT       9u
#define AI_QWEN_IMAGE__G__MOD          10u   /* the blocks' shared modulation: 4·HIDDEN rows                             */
#define AI_QWEN_IMAGE__G__MOD_LUT      11u
#define AI_QWEN_IMAGE__G__NORM_OUT     12u   /* the last norm's scale                                                   */
#define AI_QWEN_IMAGE__G__NORM_OUT_LUT 13u
#define AI_QWEN_IMAGE__G__PROJ         14u   /* HIDDEN in, CHANNELS out                                                  */
#define AI_QWEN_IMAGE__G__PROJ_LUT     15u
#define AI_QWEN_IMAGE__G__TXT_NORM     16u   /* CONTEXT halves: the text norm's 1 + w                                   */
#define AI_QWEN_IMAGE__G__ONES         17u   /* HIDDEN halves of 1                                                      */
#define AI_QWEN_IMAGE__G__MOD_ONE      18u   /* 4·HIDDEN halves: 1 where the modulation is a scale, 0 where a gate      */
#define AI_QWEN_IMAGE__G__ZERO         19u   /* HIDDEN halves of 0                                                      */
#define AI_QWEN_IMAGE__G__SIGNS        20u   /* the pack's first 512 signs, as halves                                   */
#define AI_QWEN_IMAGE__G__SCRATCH      21u   /* the pairs, then in halves h · hrot · m: ROWS rows each of the wider of
                                                HIDDEN and CONTEXT; every verb's                                         */
#define AI_QWEN_IMAGE__G__HIDDEN       22u   /* integers from here on                                                   */
#define AI_QWEN_IMAGE__G__CONTEXT      23u   /* the encoder's width                                                     */
#define AI_QWEN_IMAGE__G__CHANNELS     24u   /* a latent row's                                                          */
#define AI_QWEN_IMAGE__G__FREQ         25u   /* the sinusoid's                                                          */
#define AI_QWEN_IMAGE__G__ROWS         26u   /* the rows a verb takes at a time: the scratch holds this many            */
#define AI_QWEN_IMAGE__G__D_IMG        27u   /* the widths, a matrix each, in the planes' order                         */
#define AI_QWEN_IMAGE__G__D_TXT_IN     28u
#define AI_QWEN_IMAGE__G__D_TXT_OUT    29u
#define AI_QWEN_IMAGE__G__D_T1         30u
#define AI_QWEN_IMAGE__G__D_T2         31u
#define AI_QWEN_IMAGE__G__D_MOD        32u
#define AI_QWEN_IMAGE__G__D_NORM_OUT   33u
#define AI_QWEN_IMAGE__G__D_PROJ       34u
#define AI_QWEN_IMAGE__G__TABLE        35u

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_STEP_CUH */
