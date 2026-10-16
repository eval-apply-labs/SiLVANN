#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_VAE_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_VAE_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_QWEN_IMAGE__VAE__FAULT_TABLE 0x51495654ull     /* "QIVT" — a convolution's table is not what it reads      */
#define AI_QWEN_IMAGE__VAE__FAULT_ROOM  0x51495655ull     /* "QIVU" — a picture too large for its planes or scratch   */

/* The VAE's RMS norm (`QwenImage21RMS_norm`, x / max(‖x‖, 1e-12) · √C · γ) as nn's rmsnorm (x / √(mean x² + eps) · γ):
 * the two agree wherever ‖x‖² / C is far above this. */
#define AI_QWEN_IMAGE__VAE__EPS 1e-24f

/* ══ THE VAE ON A WORKER — a picture kept a pixel a row (H x W pixels of C halves), every layer a convolution or the
 * middle's attention (▶ `cpu/opcodes/vae__abi.cuh`; the arithmetic, diffusers' `AutoencoderKLQwenImage21` on one frame).
 *
 * `(ai_qwen_image__conv x y table H W)` -> `y`: the convolution of `x` (H x W of C) into `y`, by nn's `unfold_rows` and
 *   the product of its rows a chunk at a time, plus the bias. Its options, in the table:
 *     NORM     the input normed (RMS, γ) and through SiLU first — a resnet's half, the decoder's last layer
 *     UP       2: the input upsampled 2 x 2 by its nearest pixel first (the decoder's upsampling and its convolution)
 *     SHUFFLE  f > 1: the output's channels to space, O / f² of them at Ho·f x Wo·f (a dup-up's 1 x 1 selection)
 *     ADD      1: added to `y` rather than written (a residual)
 *   y is Ho x Wo (x f with SHUFFLE): Ho = (H·UP + PAD_TOP + PAD_BOTTOM − KH) / STRIDE + 1, Wo alike.
 * `(ai_qwen_image__vae_attention x table H W)` -> `x`: the middle block's one head over every pixel — `x` += the
 *   projection of softmax(q·kᵀ / √C)·v, q, k and v 1 x 1 convolutions of x normed (γ) — as products: the scores a row
 *   of N = H·W halves each, each row's softmax, the values turned to columns. */
#define AI_QWEN_IMAGE__VC__WEIGHT     0u    /* O rows of KH·KW·C halves, a row an output channel's kernel in (dy, dx, c) */
#define AI_QWEN_IMAGE__VC__BIAS       1u    /* O halves                                                                */
#define AI_QWEN_IMAGE__VC__GAMMA      2u    /* C halves: the input's norm's γ (read with NORM)                         */
#define AI_QWEN_IMAGE__VC__SCRATCH    3u    /* the pairs, a chunk's unfolded rows and its product, the normed input
                                               (NORM), and the output before its shuffle and after (SHUFFLE)           */
#define AI_QWEN_IMAGE__VC__C          4u    /* integers from here on                                                   */
#define AI_QWEN_IMAGE__VC__O          5u
#define AI_QWEN_IMAGE__VC__KH         6u
#define AI_QWEN_IMAGE__VC__KW         7u
#define AI_QWEN_IMAGE__VC__STRIDE     8u
#define AI_QWEN_IMAGE__VC__PAD_TOP    9u
#define AI_QWEN_IMAGE__VC__PAD_LEFT   10u
#define AI_QWEN_IMAGE__VC__PAD_BOTTOM 11u
#define AI_QWEN_IMAGE__VC__PAD_RIGHT  12u
#define AI_QWEN_IMAGE__VC__UP         13u
#define AI_QWEN_IMAGE__VC__NORM       14u
#define AI_QWEN_IMAGE__VC__SHUFFLE    15u
#define AI_QWEN_IMAGE__VC__ADD        16u
#define AI_QWEN_IMAGE__VC__CHUNK      17u   /* the output pixels a step: what the scratch's unfolded rows hold          */
#define AI_QWEN_IMAGE__VC__TABLE      18u

#define AI_QWEN_IMAGE__VA__Q          0u    /* the attention's: q, k, v and the projection, C rows of C halves each   */
#define AI_QWEN_IMAGE__VA__Q_BIAS     1u
#define AI_QWEN_IMAGE__VA__K          2u
#define AI_QWEN_IMAGE__VA__K_BIAS     3u
#define AI_QWEN_IMAGE__VA__V          4u
#define AI_QWEN_IMAGE__VA__V_BIAS     5u
#define AI_QWEN_IMAGE__VA__PROJ       6u
#define AI_QWEN_IMAGE__VA__PROJ_BIAS  7u
#define AI_QWEN_IMAGE__VA__GAMMA      8u
#define AI_QWEN_IMAGE__VA__SCRATCH    9u    /* the pairs, then n · q · k · v · vᵀ · the answers (N x C each), the scores
                                               (N x N)                                                                 */
#define AI_QWEN_IMAGE__VA__C          10u   /* integers from here on                                                   */
#define AI_QWEN_IMAGE__VA__TABLE      11u

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CONTRACTS_OBJECTS_VAE_CUH */
