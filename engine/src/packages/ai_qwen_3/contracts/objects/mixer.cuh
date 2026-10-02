#ifndef SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_MIXER_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_MIXER_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_QWEN_3__MIXER__FAULT_TABLE 0x51334D58ull   /* "Q3MX" — the mixer's table is not what the verb reads */
#define AI_QWEN_3__MIXER__FAULT_ROOM  0x51334D59ull   /* "Q3MY" — a buffer is too small for the shapes         */

/* ══ THE MIXERS' PLANE TABLES — the half of a layer before the MoE, each owning its state ═════════════════
 * Both answer `h1 = x + mixer(rmsnorm(x))` — or `residual + mixer(rmsnorm(x))` when a last argument names one: under
 * tensor parallelism a card holds some of the heads and the same columns of the out projection, and every card
 * but one sums onto zeros, so the cards' outputs add up to the layer's. The DeltaNet owns its recurrent state and conv window; the
 * attention owns its KV cache. Planes first, then the layer's integers. */

/* `(ai_qwen_3__deltanet x planes h1)` */
#define AI_QWEN_3__DN__INPUT_NORM   0u
#define AI_QWEN_3__DN__QKV          1u
#define AI_QWEN_3__DN__QKV_LUT      2u
#define AI_QWEN_3__DN__Z            3u
#define AI_QWEN_3__DN__Z_LUT        4u
#define AI_QWEN_3__DN__A            5u
#define AI_QWEN_3__DN__A_LUT        6u
#define AI_QWEN_3__DN__B            7u
#define AI_QWEN_3__DN__B_LUT        8u
#define AI_QWEN_3__DN__CONV_W       9u
#define AI_QWEN_3__DN__A_LOG        10u
#define AI_QWEN_3__DN__DT_BIAS      11u
#define AI_QWEN_3__DN__NORM_W       12u   /* the gated norm's weight, plain `w` */
#define AI_QWEN_3__DN__OUT          13u
#define AI_QWEN_3__DN__OUT_LUT      14u
#define AI_QWEN_3__DN__STATE        15u   /* the recurrent state S, fp32 — written in place */
#define AI_QWEN_3__DN__WINDOW       16u   /* the conv window — written in place */
#define AI_QWEN_3__DN__SIGNS        17u
#define AI_QWEN_3__DN__MINUS1       18u   /* a buffer whose element 0 is -1: the decay's sign */
#define AI_QWEN_3__DN__SCRATCH      19u
#define AI_QWEN_3__DN__HIDDEN       20u   /* integers from here on */
#define AI_QWEN_3__DN__K_HEADS      21u
#define AI_QWEN_3__DN__V_HEADS      22u
#define AI_QWEN_3__DN__HEAD_DIM     23u
#define AI_QWEN_3__DN__D_QKV        24u
#define AI_QWEN_3__DN__D_Z          25u
#define AI_QWEN_3__DN__D_A          26u
#define AI_QWEN_3__DN__D_B          27u
#define AI_QWEN_3__DN__D_OUT        28u
#define AI_QWEN_3__DN__TABLE        29u
/* ⭐ THE RESIDUAL ROTATED — an optional cell past the table, 1 when the model was packed with its residual stream kept
 * rotated and the residual norms' gains folded into the matrices that read them (`nn_pack_model.py --residual`):
 * the input is normed and read as it is, with no rotation before the projections. Absent or 0, the verb rotates it. */
#define AI_QWEN_3__DN__RESIDUAL_ROTATED 29u

/* `(ai_qwen_3__attention x planes pos h1)` — `pos` is this token's position: its cache row, and the length */
#define AI_QWEN_3__AT__INPUT_NORM   0u
#define AI_QWEN_3__AT__Q            1u    /* q_proj: each head's query and its output gate, interleaved */
#define AI_QWEN_3__AT__Q_LUT        2u
#define AI_QWEN_3__AT__K            3u
#define AI_QWEN_3__AT__K_LUT        4u
#define AI_QWEN_3__AT__V            5u
#define AI_QWEN_3__AT__V_LUT        6u
#define AI_QWEN_3__AT__O            7u
#define AI_QWEN_3__AT__O_LUT        8u
#define AI_QWEN_3__AT__Q_NORM       9u
#define AI_QWEN_3__AT__K_NORM       10u
#define AI_QWEN_3__AT__K_CACHE      11u   /* [position][kv head][head dim], fp16 — written in place */
#define AI_QWEN_3__AT__V_CACHE      12u
#define AI_QWEN_3__AT__ANGLES       13u   /* this position's cos and sin (fp32), from nn__rope__angles */
#define AI_QWEN_3__AT__SCORES       14u   /* fp32, q_heads x the cache's positions */
#define AI_QWEN_3__AT__SIGNS        15u
#define AI_QWEN_3__AT__SCRATCH      16u
#define AI_QWEN_3__AT__HIDDEN       17u   /* integers from here on */
#define AI_QWEN_3__AT__Q_HEADS      18u
#define AI_QWEN_3__AT__KV_HEADS     19u
#define AI_QWEN_3__AT__HEAD_DIM     20u
#define AI_QWEN_3__AT__ROT          21u   /* the rotated leading dims of a head (partial RoPE) */
#define AI_QWEN_3__AT__D_Q          22u
#define AI_QWEN_3__AT__D_K          23u
#define AI_QWEN_3__AT__D_V          24u
#define AI_QWEN_3__AT__D_O          25u
#define AI_QWEN_3__AT__TABLE        26u
/* ⭐ AND FOR A PROMPT'S ROWS AT ONCE (`ai_qwen_3__attention_rows`), one more: the angles are made inside the
 * verb, a position a row, so it needs RoPE's base. The one-position verb reads no further than `TABLE`. */
#define AI_QWEN_3__AT__THETA        26u
#define AI_QWEN_3__AT__ROWS_TABLE   27u
/* ⭐ THE RESIDUAL ROTATED, as the DeltaNet's — after the rows' RoPE base, so a table may carry both. */
#define AI_QWEN_3__AT__RESIDUAL_ROTATED 27u
/* ⭐⭐ THE KV CACHE IN TIERS — `(ai_qwen_3__attention_tiered x planes pos h1 [residual])`. ⚖ *"rotated from the start,
 * tq8 for the warm tier … and tq4 for the cold"* · *"the most recent quarter, excluding the first 256 tokens and the most
 * recent 4k tokens"*. Cells 0..27 are the attention's, the SINK standing where its cache stood (11, 12: the first
 * positions, fp16); every key and value is kept rotated a head at a time. Past them, the tiers: */
#define AI_QWEN_3__AT__HOT_K        28u   /* the newest positions past the sink, fp16, a ring of HOT slots       */
#define AI_QWEN_3__AT__HOT_V        29u
#define AI_QWEN_3__AT__WARM_KC      30u   /* the positions before those, TurboQuant 8, a ring of WARM slots: codes */
#define AI_QWEN_3__AT__WARM_KS      31u   /*   and a half scale a row, a row a kv head                              */
#define AI_QWEN_3__AT__WARM_VC      32u
#define AI_QWEN_3__AT__WARM_VS      33u
#define AI_QWEN_3__AT__COLD_KC      34u   /* every older one, TurboQuant 4, by its place past the sink             */
#define AI_QWEN_3__AT__COLD_KS      35u
#define AI_QWEN_3__AT__COLD_VC      36u
#define AI_QWEN_3__AT__COLD_VS      37u
#define AI_QWEN_3__AT__TIER_SCRATCH 38u   /* four residuals, the query and the answer rotated, a key and value, a row cooled */
#define AI_QWEN_3__AT__SINK         39u   /* integers: how many positions the sink, the hot ring and the warm ring hold */
#define AI_QWEN_3__AT__HOT          40u
#define AI_QWEN_3__AT__WARM         41u
#define AI_QWEN_3__AT__TIERED_TABLE 42u
/* ⭐ `(ai_qwen_3__attention_tiered_rows xs planes first h1s n)` — a prompt's rows through the tiered cache: the tiered
 *   table, then the rows' scratch and angles, a window the cache's fp16 positions are gathered into in order (keys,
 *   values), and the rows' residuals (three a row). ⚖ *"the tiers happen in chunks with the rotation that sends a 64
 *   chunk in and rotates down other 64 values"*: the chunk's rows take their hot slots and the rows they displace are
 *   cooled together, before the chunk attends. */
#define AI_QWEN_3__AT__ROWS_SCRATCH 42u
#define AI_QWEN_3__AT__ROWS_ANGLES  43u
#define AI_QWEN_3__AT__WINDOW_K     44u
#define AI_QWEN_3__AT__WINDOW_V     45u
#define AI_QWEN_3__AT__ROWS_RES     46u
#define AI_QWEN_3__AT__ROWS_SCORES  47u   /* fp32, the rows' scores over the window: while it is the whole cache the rows
                                             attend it at once, causally */
#define AI_QWEN_3__AT__TIERED_ROWS_TABLE 48u

/* ⭐ A RANK-1 MASK ON THE OUTPUT PROJECTION — a changed model kept as `out_proj + r·vᵀ` (▶ nn's `rank1_*` doors and
 * `python/nn_rank1_mask.py`). Three optional cells past the longest table of each mixer: `r` (H halves), `v` (the
 * projection's input width, kept rotated as that input is) and a scratch for the dots, a float a row. An integer 0 in
 * the first cell, or a table that stops short of it, is a layer the mask leaves alone; a table shorter than the longest
 * is padded with zeros up to it. */
#define AI_QWEN_3__DN__MASK         30u
#define AI_QWEN_3__AT__MASK         48u

/* ══ THE DENSE MLP'S PLANE TABLE — the half of a layer after the mixer, in a model without experts ══════════════
 * `(ai_qwen_3__mlp h1 planes out [residual])` -> `out = residual + down(swiglu(gate, up)(rmsnorm(h1)))`, the residual
 * `h1` when none is named. ⭐ UNDER TENSOR PARALLELISM a card holds INTER of the whole: its gate and up rows and the same
 * columns of down — the activation is rotated in whole blocks of 512 and every part's slice is whole blocks, so each
 * part's product is exact — and every part but one sums onto a zero residual, so the parts' outputs add up to the
 * layer's. */
#define AI_QWEN_3__MLP__NORM         0u
#define AI_QWEN_3__MLP__GATE         1u
#define AI_QWEN_3__MLP__GATE_LUT     2u
#define AI_QWEN_3__MLP__UP           3u
#define AI_QWEN_3__MLP__UP_LUT       4u
#define AI_QWEN_3__MLP__DOWN         5u
#define AI_QWEN_3__MLP__DOWN_LUT     6u
#define AI_QWEN_3__MLP__SIGNS        7u
#define AI_QWEN_3__MLP__ONE          8u    /* a buffer whose element 0 is 1: the down's one weight */
#define AI_QWEN_3__MLP__SCRATCH      9u
#define AI_QWEN_3__MLP__HIDDEN       10u   /* integers from here on */
#define AI_QWEN_3__MLP__INTER        11u   /* this card's part of the intermediate width */
#define AI_QWEN_3__MLP__D_GATE       12u
#define AI_QWEN_3__MLP__D_UP         13u
#define AI_QWEN_3__MLP__D_DOWN       14u
#define AI_QWEN_3__MLP__TABLE        15u
#define AI_QWEN_3__MLP__RESIDUAL_ROTATED 15u   /* ⭐ optional, as the DeltaNet's */

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_MIXER_CUH */
