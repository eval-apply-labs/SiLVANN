#ifndef SILVANN__PACKAGES_AI_QWEN_4_CONTRACTS_OBJECTS_LAYER_CUH
#define SILVANN__PACKAGES_AI_QWEN_4_CONTRACTS_OBJECTS_LAYER_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_QWEN_4__LAYER__FAULT_TABLE 0x51344c54ull     /* "Q4LT" — a layer's table or the shape table is not what a verb reads */
#define AI_QWEN_4__LAYER__FAULT_ROOM  0x51344c55ull     /* "Q4LU" — a buffer or the scratch too small for the shape            */

/* ══ QWEN3.8-FLASH-NEXT's SITES, A VERB EACH (`qwen4_exp`) ══════════════════════════════════════════════════════
 * Each verb is a site of `lisp/qwen4_exp.lisp` — the same nn doors in the same order, over the same layer table — so a
 * program that calls it computes what the site's nn words compute, to the bit. ▶ `cpu/opcodes/layer__abi.cuh`.
 *
 * `(ai_qwen_4__gated_pre streams table site into inj shape)` -> `into`: the gated residual's before-step over the four
 *   fp32 streams — each stream normed by its slice of the gain, the mix's two maps, the weighted mean into `into` (H
 *   halves) and, where `inj` is a buffer, the four injection weights into it (2·σ(¼·W·n)); `inj` 0 for none (the last
 *   mixer's). `site` is the table cell of the site's norm: its down, up and injection matrices follow it (▶ LT).
 * `(ai_qwen_4__attention m table kc vc pos y shape)` -> `y`: Qwen Sparse Attention's layer while it chooses every block —
 *   q (with its gate), k and v of `m`, each head's q and k normed and roped on its first ROT, k and v into the cache at
 *   `pos`, the attention over 0 … pos, gated by σ(gate), the output projection.
 * `(ai_qwen_4__deltanet m table s32 cwin y shape)` -> `y`: the Gated DeltaNet's layer at one position — its projections,
 *   β = σ(b) and g = −exp(A_log)·softplus(a + dt_bias), the short conv, every head's step in one launch with its output
 *   gated by σ(z), the output projection.
 * `(ai_qwen_4__shared x xr table out shape)` -> `out`: the shared expert over `x` (`xr` it rotated), gated by σ of its
 *   own gate.
 * `(ai_qwen_4__route m mr table logits picks hand k shape)` -> `picks`: `m` rotated into `mr`, the router into `logits`
 *   (the experts' count), `mr` into the hand's first H halves, then the `k` likeliest into `picks` (a node array) and
 *   their softmax into the hand after it — `nn__vector__top_k`'s, which waits for its kernel, so the CPU reads the picks
 *   next; the hand's copy is queued before it.
 *
 * ⭐ A PROMPT'S CHUNK, ITS `n` ROWS AT ONCE — the same sites over `n` rows a row a position, each matrix one GEMM over every
 * row (nn's `expert_rows`), the recurrence and the attention walking the rows in order inside their launches. A rows form
 * sums in its own order, so it is held to a position at a time's by a bound, not bits (CLAUDE.md: two forms). The shape
 * table a rows verb reads is the rows' own: its scratch sized for the chunk, its angles a row each, its scores the chunk's.
 * `(ai_qwen_4__gated_pre_rows streams table site into inj n shape)` -> `into`: `n` rows of the fp32 streams (a row S
 *   floats) into `n` rows of H, and `inj` `n` rows of STREAMS halves (or 0).
 * `(ai_qwen_4__inject_rows streams y inj n shape)` -> `streams`: row r's streams += inj_r · y_r, a row of `inj` STREAMS
 *   halves; the shape's MIX a 24-float mix a row, its comb the identity (▶ the program's `inject`).
 * `(ai_qwen_4__deltanet_rows m table s32 cwin y n shape)` -> `y`: `n` positions after the state's.
 * `(ai_qwen_4__attention_rows m table kc vc first n y shape)` -> `y`: positions `first .. first + n`, every row's key and
 *   value into the cache before any row's query reads it, row r attending 0 … first + r; the angles the shape's, a row
 *   ROT floats.
 * `(ai_qwen_4__shared_rows x xr table out n shape)` -> `out`.
 * `(ai_qwen_4__route_rows m mr table logits hands n shape)` -> `logits`: `m` rotated into `mr`, the router over every
 *   row into `logits` (`n` rows of the experts' count), and each row's `mr` into its hand row (HAND_ROW bytes a row) — the
 *   picks are the program's (`nn__vector__top_k_rows`). */

/* ── a layer's table — the cells `python/silvann_runtime/qwen4_exp.py` CELLS writes, in its order. A matrix is three
 *    cells (its codes, its scales, its bits), a norm or a vector one; a cell a layer does not have is the integer 0. */
#define AI_QWEN_4__LT__AN       0u    /* the attention site's gated residual: its norm, then down, up, inject   */
#define AI_QWEN_4__LT__MN      10u    /* the MLP site's, the same four                                            */
#define AI_QWEN_4__SITE__DOWN   1u    /* a site's cells after its norm                                            */
#define AI_QWEN_4__SITE__UP     4u
#define AI_QWEN_4__SITE__INJ    7u
#define AI_QWEN_4__LT__QKV     20u    /* the DeltaNet's                                                           */
#define AI_QWEN_4__LT__Z       23u
#define AI_QWEN_4__LT__A       26u
#define AI_QWEN_4__LT__B       29u
#define AI_QWEN_4__LT__CONV    32u
#define AI_QWEN_4__LT__ALOG    33u
#define AI_QWEN_4__LT__DTB     34u
#define AI_QWEN_4__LT__GNORM   35u
#define AI_QWEN_4__LT__OUT     36u
#define AI_QWEN_4__LT__Q       39u    /* the attention's                                                          */
#define AI_QWEN_4__LT__K       42u
#define AI_QWEN_4__LT__V       45u
#define AI_QWEN_4__LT__O       48u
#define AI_QWEN_4__LT__QN      51u
#define AI_QWEN_4__LT__KN      52u
#define AI_QWEN_4__LT__GATE    65u    /* the router                                                               */
#define AI_QWEN_4__LT__SG      68u    /* the shared expert's gate, up, down and its own gate (the MoE's)          */
#define AI_QWEN_4__LT__SU      71u
#define AI_QWEN_4__LT__SD      74u
#define AI_QWEN_4__LT__SEG     77u
#define AI_QWEN_4__LT__TABLE   80u

/* ── the shape table — one a position's, one a prompt's chunk's, the same layout: its buffers first, then its integers,
 *    then its epsilon ─────────────────────────────────────────────────────────────────────────────────────────────── */
#define AI_QWEN_4__SH__SIGNS    0u    /* the rotation's 512 signs, halves                                          */
#define AI_QWEN_4__SH__ONES     1u    /* STREAMS·H halves of 1 (silu as a combine with ones)                       */
#define AI_QWEN_4__SH__SCRATCH  2u    /* every verb's own, from its start (▶ `ai_qwen_4__layer__zzprivate_take`)   */
#define AI_QWEN_4__SH__CS       3u    /* the angles: ROT floats a position, a row each for a chunk                 */
#define AI_QWEN_4__SH__SCORES   4u    /* the attention's scores, 2 · QH · positions halves a row                   */
#define AI_QWEN_4__SH__MIX      5u    /* a mix a row, 2·M + M² floats — M of pre, M of post, M² of comb (▶ inject_rows) */
#define AI_QWEN_4__SH__HIDDEN   6u    /* integers from here on                                                     */
#define AI_QWEN_4__SH__STREAMS  7u
#define AI_QWEN_4__SH__RANK     8u    /* the gated residual's rank, padded to the rotation's 256 and 128           */
#define AI_QWEN_4__SH__KH       9u    /* the DeltaNet's key heads, value heads and head width                      */
#define AI_QWEN_4__SH__VH      10u
#define AI_QWEN_4__SH__LD      11u
#define AI_QWEN_4__SH__QH      12u    /* the attention's query heads, key-value heads, head width, rotated width   */
#define AI_QWEN_4__SH__KVH     13u
#define AI_QWEN_4__SH__AD      14u
#define AI_QWEN_4__SH__ROT     15u
#define AI_QWEN_4__SH__INTER   16u    /* the shared expert's inner width                                           */
#define AI_QWEN_4__SH__EXPERTS 17u    /* the router's experts                                                      */
#define AI_QWEN_4__SH__HAND    18u    /* a hand row's bytes (ai_qwen_3's AI_QWEN_3__HAND__ROW)                    */
#define AI_QWEN_4__SH__ROWS    19u    /* the most rows its scratch, angles, scores and mixes hold (1: a position's) */
#define AI_QWEN_4__SH__EPS     20u    /* a float: the norms' epsilon                                               */
#define AI_QWEN_4__SH__TABLE   21u

#define AI_QWEN_4__RANK_FIRST  256u   /* the rank's rotation: a block of 256, then one of 128 (▶ the program's rotate_rank) */
#define AI_QWEN_4__RANK_LAST   128u
#define AI_QWEN_4__GATE_SIGMOID 1u    /* nn's NN__DELTANET__GATE_SIGMOID: the DeltaNet's output gated by σ(z)      */

#endif /* SILVANN__PACKAGES_AI_QWEN_4_CONTRACTS_OBJECTS_LAYER_CUH */
