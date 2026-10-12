#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_ATTENTION_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_ATTENTION_CUH

/* This file needs nothing: every name in it is its own. */

/* ══ ⭐ GROUPED-QUERY ATTENTION'S PLANE TABLE — the half of a layer before its MLP, in its plainest form ═════════════════
 * `(nn__attention__rows xs table first h1s rows)` -> `h1s = xs + o(attention(rmsnorm(xs)))` for a prompt's `rows`
 * positions from `first`: each row's query, key and value, its key and value written into the layer's cache at its
 * position, every row attending the cache up to its own position. RoPE over the whole head from tables of angles a
 * position a row — cosines then sines, fp32, `head_dim` floats a row — the queries' with any query scale folded in. No
 * norm on the query or the key, no gate: what a model with those has is its own package's (Qwen's). Its matrices are
 * TurboQuant's, each with its scales; its input rotated as the pack rotated their columns, the residual and the heads'
 * widths whole blocks of 512 so a rotation over the rows is each row's. A model's one-position form is nn's own words
 * in its program (Ministral's: `lisp/mistral3.lisp`). */
#define NN__ATTENTION__NORM          0u
#define NN__ATTENTION__Q             1u
#define NN__ATTENTION__Q_SCALES      2u
#define NN__ATTENTION__K             3u
#define NN__ATTENTION__K_SCALES      4u
#define NN__ATTENTION__V             5u
#define NN__ATTENTION__V_SCALES      6u
#define NN__ATTENTION__O             7u
#define NN__ATTENTION__O_SCALES      8u
#define NN__ATTENTION__KEYS          9u    /* the layer's key cache: [position][kv head][head dim] in halves */
#define NN__ATTENTION__VALUES        10u   /* and its values, the same shape */
#define NN__ATTENTION__ANGLES_Q      11u   /* the queries' angles, a position a row */
#define NN__ATTENTION__ANGLES_K      12u   /* the keys' */
#define NN__ATTENTION__SIGNS         13u
#define NN__ATTENTION__SCRATCH       14u
#define NN__ATTENTION__SCORES        15u   /* fp32, q_heads × (first + rows) a row: the rows attend as many at a time as it holds */
#define NN__ATTENTION__HIDDEN        16u   /* integers from here on */
#define NN__ATTENTION__Q_HEADS       17u
#define NN__ATTENTION__KV_HEADS      18u
#define NN__ATTENTION__HEAD_DIM      19u
#define NN__ATTENTION__D_Q           20u
#define NN__ATTENTION__D_K           21u
#define NN__ATTENTION__D_V           22u
#define NN__ATTENTION__D_O           23u
#define NN__ATTENTION__EPS           24u   /* a float: the norm's epsilon */
#define NN__ATTENTION__TABLE         25u

/* A prompt's rows: up to this many a call. The scratch starts with their two lists of pairs, 8 bytes a row each. */
#define NN__ATTENTION__ROWS_MAX      256u

#define NN__ATTENTION__FAULT_TABLE   0x4E415442ull   /* "NATB" — a table cell that is not what its place says */
#define NN__ATTENTION__FAULT_ROOM    0x4E41524Dull   /* "NARM" — a plane too small for the widths and the rows */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_ATTENTION_CUH */
