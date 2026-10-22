#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_OBJECTS_LAYER_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_OBJECTS_LAYER_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_DEEPSEEK_4__LAYER__FAULT_TABLE  0x44533454ull   /* "DS4T" — a table is not what the verb reads          */
#define AI_DEEPSEEK_4__LAYER__FAULT_ROOM   0x44533452ull   /* "DS4R" — a buffer is too small for the shapes        */

/* ══ THE PLANE TABLES — a site's planes first, then its integers, then its reals ═══════════════════════════════
 * Every card table begins with its site's hyper-connection map and norm, then two MIXES: OWN, which the verb writes —
 * this site's `pre`, `post` and `comb` (2 · (2 + mult) · mult floats, the map's logits passing through its second half)
 * — and EARLIER, which it reads: the mix the site before it wrote, whose `pre` collapses this site's streams. The model
 * collapses a site by the weights its previous site computed, so a program hands each site's OWN to the next as its
 * EARLIER (two buffers, the attention's and the MLP's, in turn), and the first layer's EARLIER is a one-hot `pre`.
 * A table's planes end with the rotation's signs and a SCRATCH buffer, nothing in which outlives the verb but the MoE
 * site's first `hidden` halves — the shared expert's answer — kept from its `shared` verb to its `close`. */
#define AI_DEEPSEEK_4__MIX_BYTES      256ull                    /* twice (2 + mult) · mult floats, mult ≤ 4 */

#define AI_DEEPSEEK_4__HC_FN          0u
#define AI_DEEPSEEK_4__HC_BASE        1u
#define AI_DEEPSEEK_4__HC_SCALE       2u
#define AI_DEEPSEEK_4__NORM           3u
#define AI_DEEPSEEK_4__OWN            4u
#define AI_DEEPSEEK_4__EARLIER        5u

/* `(ai_deepseek_4__attention streams table pos)` — every head of the query over the window's ring of rows, a sink a head.
 * A row is the normed latent `wkv` makes of the position, its last `rope` halves turned — both the key and the value of
 * every head. The ring holds `window` rows, position `p` at row `p % window`; the angles a row of `rope` floats a
 * position (`rope/2` cosines, then the sines), as the layer's kind turns them (YaRN or not). */
#define AI_DEEPSEEK_4__ATT__WQA       6u      /* wq_a and wkv read the same input: one grouped launch */
#define AI_DEEPSEEK_4__ATT__WKV       8u
#define AI_DEEPSEEK_4__ATT__WQB       10u
#define AI_DEEPSEEK_4__ATT__WOA       12u     /* `groups` maps of `o_rank` rows, each over its own heads' answer */
#define AI_DEEPSEEK_4__ATT__WOB       14u
#define AI_DEEPSEEK_4__ATT__Q_NORM    16u
#define AI_DEEPSEEK_4__ATT__KV_NORM   17u
#define AI_DEEPSEEK_4__ATT__SINK      18u     /* a sink logit a head, halves */
#define AI_DEEPSEEK_4__ATT__RING      19u     /* window · head_dim halves, written at `pos % window` */
#define AI_DEEPSEEK_4__ATT__ANGLES    20u     /* rope floats a position, from position 0 */
#define AI_DEEPSEEK_4__ATT__SCORES    21u     /* heads · window floats */
#define AI_DEEPSEEK_4__ATT__SIGNS     22u
#define AI_DEEPSEEK_4__ATT__SCRATCH   23u
#define AI_DEEPSEEK_4__ATT__PLANES    24u
#define AI_DEEPSEEK_4__ATT__HIDDEN    24u     /* integers */
#define AI_DEEPSEEK_4__ATT__MULT      25u
#define AI_DEEPSEEK_4__ATT__ITERS     26u
#define AI_DEEPSEEK_4__ATT__HEADS     27u
#define AI_DEEPSEEK_4__ATT__HEAD_DIM  28u
#define AI_DEEPSEEK_4__ATT__ROPE      29u
#define AI_DEEPSEEK_4__ATT__Q_RANK    30u
#define AI_DEEPSEEK_4__ATT__O_RANK    31u
#define AI_DEEPSEEK_4__ATT__GROUPS    32u
#define AI_DEEPSEEK_4__ATT__WINDOW    33u
#define AI_DEEPSEEK_4__ATT__D         34u     /* wq_a, wkv, wq_b, wo_a, wo_b */
#define AI_DEEPSEEK_4__ATT__EPS       39u     /* reals */
#define AI_DEEPSEEK_4__ATT__HC_EPS    40u
#define AI_DEEPSEEK_4__ATT__TABLE     41u
/* ⭐ THE COMPRESSED LATENTS, cells past the table — a layer whose ratio is not 0 attends its window AND the compressed
 *   latents of its source layer: one latent a `ratio` positions, the group pooled by a softmax gate a channel (ratio 1:
 *   every position its own). The source layer compresses — its own projections, its norm, the partial group in a ring
 *   of `ratio` rows (the latents, then their gates) — and the layers after it read its cache. A latent is normed, its
 *   tail turned at the group's first position, and written at row `position / ratio`. The rows a query attends — the
 *   window's, then the compressed — are gathered into one buffer for the scores and the mix.
 *   ⛳ EVERY COMPRESSED ROW SO FAR IS ATTENDED: the model's indexer keeps the best `index_topk` of them, which is every
 *   one up to `index_topk` rows — so a layer refuses a position past that until the indexer is here. */
#define AI_DEEPSEEK_4__CMP__CACHE     41u     /* the source's latents, halves, a row of head_dim a group */
#define AI_DEEPSEEK_4__CMP__WKV       42u     /* the source's: the latent's map (pair) */
#define AI_DEEPSEEK_4__CMP__WGATE     44u     /* its gate's map (pair) — read where ratio > 1 */
#define AI_DEEPSEEK_4__CMP__NORM      46u
#define AI_DEEPSEEK_4__CMP__RING      47u     /* ratio rows of latents, then ratio rows of gates, halves */
#define AI_DEEPSEEK_4__CMP__ZERO      48u     /* ratio · head_dim halves of zero: the pool's place bias, which this model has none of */
#define AI_DEEPSEEK_4__CMP__GATHER    49u     /* the rows a query attends: window + attended latents, head_dim halves each */
#define AI_DEEPSEEK_4__CMP__PLANES    50u
#define AI_DEEPSEEK_4__CMP__RATIO     50u     /* integers */
#define AI_DEEPSEEK_4__CMP__SOURCE    51u     /* 1: this layer compresses; 0: it reads its source's cache */
#define AI_DEEPSEEK_4__CMP__ROWS      52u     /* the cache's rows */
#define AI_DEEPSEEK_4__CMP__TOPK      53u     /* the most latents a query attends: the indexer's `index_topk` */
#define AI_DEEPSEEK_4__CMP__D         54u     /* wkv, wgate */
#define AI_DEEPSEEK_4__CMP__TABLE     56u
/* ⭐ THE INDEXER, cells past those — with them, past `topk` latents a query attends the best `topk` of them, scored by a
 *   small attention of its own (`heads` of `dim`): a key a latent, which the source writes as it compresses (its latent
 *   before the rope, through `wk` and a norm, its tail turned), and a query and a weight a head, which an INDEX SOURCE
 *   layer makes at each position, scores and selects with (nn's index doors, a pool a latent); the layers after it reuse
 *   its selection. Without them a position past `topk` latents is refused. */
#define AI_DEEPSEEK_4__IDX__KEYS      56u     /* the source's index keys, a row of dim halves a latent */
#define AI_DEEPSEEK_4__IDX__WQB       57u     /* the index source's: its query from the q rank (pair) */
#define AI_DEEPSEEK_4__IDX__WP        59u     /* a weight a head (pair) */
#define AI_DEEPSEEK_4__IDX__WK        61u     /* the source's: the key from the latent (pair) */
#define AI_DEEPSEEK_4__IDX__KNORM     63u
#define AI_DEEPSEEK_4__IDX__SCORES    64u     /* a float a latent */
#define AI_DEEPSEEK_4__IDX__SELECT    65u     /* the index source's selection: topk u32, then their count */
#define AI_DEEPSEEK_4__IDX__PLANES    66u
#define AI_DEEPSEEK_4__IDX__INDEXER   66u     /* 1: this layer scores and selects; 0: it reads its index source's selection */
#define AI_DEEPSEEK_4__IDX__HEADS     67u
#define AI_DEEPSEEK_4__IDX__DIM       68u
#define AI_DEEPSEEK_4__IDX__D         69u     /* wq_b, weights, wk */
#define AI_DEEPSEEK_4__IDX__TABLE     72u

/* ⭐ A MASK, `W' = W + Σ_i r_i·v_iᵀ` beside a matrix that writes into the streams (▶ nn's rank-1 doors) — three cells past
 *   a site's table: `r` (`rank` rows of `hidden` halves), `v` (`rank` rows of the matrix's input, rotated as that input is,
 *   this card's columns of it) and a scratch of `rank` floats; an integer 0 in the first, or no cells, where the site has
 *   none. The rank is `v`'s rows. The attention's mask follows the table's last part: at ATT__TABLE, CMP__TABLE or
 *   IDX__TABLE, whichever the table carries (▶ `ai_deepseek_4__attention__zzprivate_mask_first`); it is on wo_b's answer.
 *   The MoE site's is on the shared expert's down. */
#define AI_DEEPSEEK_4__MASK__RANK_MAX 8u

/* `(ai_deepseek_4__route streams table hand picks)` · `(ai_deepseek_4__shared hand table)` · `(ai_deepseek_4__close streams
 * table routed)` — the MoE site, one table for its three card verbs. The HAND is what crosses to the CPU: the rotated
 * input (hidden halves), then the picks' weights (top_k halves). The router scores √softplus(logit); the bias chooses
 * and does not weigh. */
#define AI_DEEPSEEK_4__MOE__ROUTER    6u      /* lossless: halves, read against the input unrotated */
#define AI_DEEPSEEK_4__MOE__BIAS      7u
#define AI_DEEPSEEK_4__MOE__GATE      8u      /* the shared expert */
#define AI_DEEPSEEK_4__MOE__UP        10u
#define AI_DEEPSEEK_4__MOE__DOWN      12u
#define AI_DEEPSEEK_4__MOE__SIGNS     14u
#define AI_DEEPSEEK_4__MOE__SCRATCH   15u
#define AI_DEEPSEEK_4__MOE__PLANES    16u
#define AI_DEEPSEEK_4__MOE__HIDDEN    16u
#define AI_DEEPSEEK_4__MOE__MULT      17u
#define AI_DEEPSEEK_4__MOE__ITERS     18u
#define AI_DEEPSEEK_4__MOE__EXPERTS   19u
#define AI_DEEPSEEK_4__MOE__TOP_K     20u
#define AI_DEEPSEEK_4__MOE__INTER     21u     /* the shared expert's */
#define AI_DEEPSEEK_4__MOE__D         22u     /* gate, up, down */
#define AI_DEEPSEEK_4__MOE__EPS       25u
#define AI_DEEPSEEK_4__MOE__HC_EPS    26u
#define AI_DEEPSEEK_4__MOE__SCALE     27u     /* the routed weights' factor */
#define AI_DEEPSEEK_4__MOE__LIMIT     28u
#define AI_DEEPSEEK_4__MOE__TABLE     29u
#define AI_DEEPSEEK_4__MOE__MASK      AI_DEEPSEEK_4__MOE__TABLE   /* the shared expert's down: its mask (▶ AI_DEEPSEEK_4__MASK__*) */

/* `(ai_deepseek_4__engram streams table)` — the n-gram memory's write into the streams, before the layer's attention site.
 * The program writes the position's looked-up rows (`in` halves: each n-gram size's heads' rows of the layer's table,
 * dequantised) into ROWS; the verb maps them to a key a stream and one value, and each stream takes the value gated by
 * how well it matches its key (▶ nn's `engram_gate_fp32`). W is the gate's two maps' product, a row a stream. */
#define AI_DEEPSEEK_4__ENG__ROWS      0u
#define AI_DEEPSEEK_4__ENG__WKV       1u      /* pair */
#define AI_DEEPSEEK_4__ENG__W         3u
#define AI_DEEPSEEK_4__ENG__SIGNS     4u
#define AI_DEEPSEEK_4__ENG__SCRATCH   5u      /* the rows rotated, then the keys and the value */
#define AI_DEEPSEEK_4__ENG__PLANES    6u
#define AI_DEEPSEEK_4__ENG__HIDDEN    6u
#define AI_DEEPSEEK_4__ENG__MULT      7u
#define AI_DEEPSEEK_4__ENG__IN        8u
#define AI_DEEPSEEK_4__ENG__D         9u
#define AI_DEEPSEEK_4__ENG__EPS       10u     /* reals */
#define AI_DEEPSEEK_4__ENG__TABLE     11u

/* `(ai_deepseek_4__experts hand picks table routed)` — on a worker that holds routed experts: those numbered `first` up to
 * `first + count`, in its collection as `id - first`. An expert is one slot: its gate and up rows as one matrix (codes,
 * then scales), then its down rows. ▶ nn's `nn__routed`. */
#define AI_DEEPSEEK_4__EXP__SIGNS     0u
#define AI_DEEPSEEK_4__EXP__ZERO      1u      /* hidden halves of zero, what the weighted sum starts from */
#define AI_DEEPSEEK_4__EXP__SCRATCH   2u
#define AI_DEEPSEEK_4__EXP__PLANES    3u
#define AI_DEEPSEEK_4__EXP__LAYER     3u
#define AI_DEEPSEEK_4__EXP__TYPE      4u      /* the experts' collection */
#define AI_DEEPSEEK_4__EXP__UP_LUT    5u      /* offsets in a slot: the gate-and-up scales, the down codes, its scales */
#define AI_DEEPSEEK_4__EXP__DOWN      6u
#define AI_DEEPSEEK_4__EXP__DOWN_LUT  7u
#define AI_DEEPSEEK_4__EXP__HIDDEN    8u
#define AI_DEEPSEEK_4__EXP__INTER     9u
#define AI_DEEPSEEK_4__EXP__EXPERTS   10u
#define AI_DEEPSEEK_4__EXP__TOP_K     11u
#define AI_DEEPSEEK_4__EXP__D         12u
#define AI_DEEPSEEK_4__EXP__FIRST     13u
#define AI_DEEPSEEK_4__EXP__COUNT     14u
#define AI_DEEPSEEK_4__EXP__INT8      15u     /* 1: the products in integers (nn's int8 gemvs), 0: exact */
#define AI_DEEPSEEK_4__EXP__LIMIT     16u
#define AI_DEEPSEEK_4__EXP__TABLE     17u
#define AI_DEEPSEEK_4__EXP__BATCH     16u     /* nn's: a prompt's experts a batch, where the slots are a cache */

#define AI_DEEPSEEK_4__TOP_K_MAX      12u     /* nn's grouped launch takes twelve matrices */
/* A HAND, what the card hands the CPU for one position: the input rotated (hidden halves), the router's weights (`k`
 * halves), and its picks as words (`k`), at the next eight bytes. */
#define AI_DEEPSEEK_4__HAND_PICKS(hidden, k)  ((2ull * (hidden) + 2ull * (k) + 7ull) & ~7ull)
#define AI_DEEPSEEK_4__HAND(hidden, k)        (AI_DEEPSEEK_4__HAND_PICKS(hidden, k) + 8ull * (k))
#define AI_DEEPSEEK_4__ROWS_MAX       4096ull /* a chunk's rows, at most */

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_OBJECTS_LAYER_CUH */
