#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_NGRAM_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_NGRAM_CUH
/* ══ AN N-GRAM MEMORY'S ROWS — A POSITION'S, HASHED AND READ INSIDE THE PROGRAM ═════════════════════════════════════
 * ⚖ 2026-11-01 *"yes then it makes sense to turn it into an nn verb"*. Qwen3.8-Flash-Next's n-gram embedding and
 * DeepSeek V4's engram hash the n-grams ending at a position into rows of a table on disk: the token and the ones before
 * it, each times its multiplier, xored one after another, and each n-gram length's rolling value modulo each head's size,
 * offset into the one table. Read on the host, a position's rows made every token a program of its own; read by this
 * verb, a run of tokens is one program.
 *
 * `(nn__ngram__rows tok back table out)` -> `out`: the rows of token `tok`'s n-grams after the tokens in `back`, as
 * halves, into `out` — then `back` moved on, `tok` its newest.
 *   tok    an integer: the position's token, negative for a place no token holds (DeepSeek's picture positions)
 *   back   a buffer of N − 1 32-bit tokens, the oldest first: the ones before `tok`, part of the conversation's state
 *   table  a node array, the cells below
 *   out    a buffer of (N − 1) · HEADS rows of ROW_BYTES halves: the n-gram of two tokens' heads first
 * The look back stops at the first place before the position, at a negative token, and — a step back or more — at the
 * STOP token: that place and every one before it read as FILL. A live token reads as itself, or as MAP's entry for it.
 * ⛳ THE READS GO THROUGH THE SYSTEM'S CACHE — a handle's ordinary descriptor, not its direct one — since a common
 *   n-gram's rows are read again and again; NN__NGRAM__READERS threads read a position's rows at once. */
#define NN__NGRAM__FILE        0u   /* the table's file, as `sys__file__open` answers it */
#define NN__NGRAM__DATA_AT     1u   /* the byte row 0's values start at */
#define NN__NGRAM__ROW_BYTES   2u   /* a row's values: FP8 E4M3, a byte each */
#define NN__NGRAM__SCALE_FILE  3u   /* 0: every value times SCALE; else the file of each row's E8M0 scales, one a 32 values */
#define NN__NGRAM__SCALE_AT    4u   /* the byte row 0's scales start at */
#define NN__NGRAM__N           5u   /* the longest n-gram: the token and the N − 1 before it */
#define NN__NGRAM__HEADS       6u   /* the rows an n-gram length hashes into */
#define NN__NGRAM__STOP        7u   /* the token that ends the look back a step back or more (an end of text), or −1 */
#define NN__NGRAM__FILL        8u   /* what a place past the look back's end reads as */
#define NN__NGRAM__MAP         9u   /* a node array: a token's id as hashed (a compressed vocabulary), or 0 */
#define NN__NGRAM__SCALE      10u   /* a float: every value's scale where SCALE_FILE is 0 */
#define NN__NGRAM__MULT       11u   /* the N multipliers, the token s steps back's at MULT + s; then the (N − 1) · HEADS
                                     * sizes — the n-gram of k + 1 tokens' head h at (k − 1) · HEADS + h — then as many
                                     * offsets in the same order */

#define NN__NGRAM__N_MAX       8u
#define NN__NGRAM__HEADS_MAX  16u
#define NN__NGRAM__ROWS_MAX   (NN__NGRAM__HEADS_MAX * (NN__NGRAM__N_MAX - 1u))
#define NN__NGRAM__ROW_MAX  8192u   /* a row's values at most */
#define NN__NGRAM__READERS     4u   /* the threads a call reads its rows on */

#define NN__NGRAM__FAULT_TABLE 0x4E475442ull   /* "NGTB" — a table, a token or a buffer not what its place says */
#define NN__NGRAM__FAULT_READ  0x4E475244ull   /* "NGRD" — a row the file did not give */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_NGRAM_CUH */
