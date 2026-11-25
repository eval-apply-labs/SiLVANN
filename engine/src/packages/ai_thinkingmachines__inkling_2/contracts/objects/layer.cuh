#ifndef SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_OBJECTS_LAYER_CUH
#define SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_OBJECTS_LAYER_CUH
/* This file needs nothing: it is offsets and fault words. */

#define AI_THINKINGMACHINES__INKLING_2__LAYER__FAULT_TABLE 0x49324C54ull       /* "I2LT" — a table, a buffer's room or a position is not what the verbs read */

/* ══ A LAYER'S APPLICATION, IN TWO VERBS AROUND ITS EXPERTS ═══════════════════════════════════════════════════════════════
 *   `(ai_thinkingmachines__inkling_2__before_experts layer application work with_messages pos ring_row picks)`
 *   `(ai_thinkingmachines__inkling_2__after_experts layer application work routed)`
 * Three tables: the LAYER's weights (24 of them), the APPLICATION's own memory (48: a loop layer is applied three times,
 * each with its own cache), and the WORK every application shares — the stream, the block states, the buffers a site
 * leaves its answer in, the model's shape.
 *
 * ── the layer's table: buffers, then integers ──
 * A MATRIX is two buffers — its codes, then its row scales — matrix `m` at cells 2m and 2m + 1, its width (4, 5, 8, or
 * 16 for one kept exact) the integer at BITS + m. A cell the layer has no use for holds any buffer: a first layer's
 * query, a sliding layer's made taps, a layer outside the loop's controllers. Matrices that read one vector may lie one
 * after another — the second's codes where the first's end, its scales where the first's end — and the verbs then
 * take them in one product: the attention's over the query's input in the order query, bias state, sink, output
 * gate, key, value, and a gate's forget and gain. */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY            0u      /* matrices: the attention's five over the query's input … */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__KEY              1u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE            2u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT           3u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__OUTPUT_GATE      4u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_STATE       5u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__SINK             6u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__ROUTER           7u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__KEY_TAPS_MADE    8u      /* a full layer's: a head's taps made from the position's state */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE_TAPS_MADE  9u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLERS      10u     /* a loop layer's four message gates — the query's, the key's, the value's, */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__CONTROLLER       3u      /* the experts' — each three matrices: down, forget, gain                   */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__MATRICES         22u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_SCALE      44u     /* vectors and tables, a buffer each */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_SCALE       45u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__EXPERTS_NORM     46u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__ROUTER_BIAS      47u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__QUERY_NORM       48u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__KEY_NORM         49u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE_NORM       50u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__KEY_TAPS         51u     /* a short convolution's taps: a row a lag, this position's first */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__VALUE_TAPS       52u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__ATTENTION_TAPS   53u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__EXPERTS_TAPS     54u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BIAS_TABLE       55u     /* the bias a distance, a row a distance, the furthest first */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__ATTENTION_QUERY  56u     /* the learned queries: the attention's (three in a full layer), */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__EXPERTS_QUERY    57u     /* the experts', and a loop layer's four over the pass before    */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__MESSAGE_QUERIES  58u
#define AI_THINKINGMACHINES__INKLING_2__LAYER__PLANES           59u     /* integers from here on */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BITS             59u     /* matrix m's width at BITS + m */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__FULL             81u     /* 1: it attends the whole context — three queries, taps made from the state */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_START      82u     /* 1: a block of six begins here — the stream joins the block states */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCK_ROW        83u     /* that block's row of the states */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_ATTENDED  84u     /* the block states its attention's queries read, */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__BLOCKS_MIXED     85u     /* and its experts' query                         */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__FIRST            86u     /* 1: the model's first layer — its attention reads the stream itself */
#define AI_THINKINGMACHINES__INKLING_2__LAYER__TABLE            87u

/* ── the application's table: six buffers ── */
#define AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEYS               0u   /* its cache: a row a position, or a ring of the window */
#define AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUES             1u
#define AI_THINKINGMACHINES__INKLING_2__APPLICATION__KEY_HISTORY        2u   /* a convolution's three inputs before this position, */
#define AI_THINKINGMACHINES__INKLING_2__APPLICATION__VALUE_HISTORY      3u   /* the newest first                                    */
#define AI_THINKINGMACHINES__INKLING_2__APPLICATION__ATTENTION_HISTORY  4u
#define AI_THINKINGMACHINES__INKLING_2__APPLICATION__EXPERTS_HISTORY    5u
#define AI_THINKINGMACHINES__INKLING_2__APPLICATION__TABLE              6u

/* ── the work's table: buffers, integers, reals ── */
#define AI_THINKINGMACHINES__INKLING_2__WORK__STREAM               0u
#define AI_THINKINGMACHINES__INKLING_2__WORK__STATES               1u    /* the block states normed, a row each, the current one after them */
#define AI_THINKINGMACHINES__INKLING_2__WORK__PASS_STATES          2u    /* what a pass leaves the next: two rows */
#define AI_THINKINGMACHINES__INKLING_2__WORK__MESSAGES             3u    /* a loop layer's four messages */
#define AI_THINKINGMACHINES__INKLING_2__WORK__HAND                 4u    /* what the experts read: ▶ experts.cuh */
#define AI_THINKINGMACHINES__INKLING_2__WORK__RESIDUAL             5u    /* the stream with the attention's answer: kept across the experts */
#define AI_THINKINGMACHINES__INKLING_2__WORK__INPUTS               6u    /* what the attention read: three rows */
#define AI_THINKINGMACHINES__INKLING_2__WORK__ATTENTION_ANSWER     7u
#define AI_THINKINGMACHINES__INKLING_2__WORK__ATTENTION_CONVOLVED  8u
#define AI_THINKINGMACHINES__INKLING_2__WORK__EXPERTS_MIX          9u    /* what the experts read, before its gate and its norm */
#define AI_THINKINGMACHINES__INKLING_2__WORK__SCORES               10u   /* floats: a query head's row over the cache */
#define AI_THINKINGMACHINES__INKLING_2__WORK__BIAS                 11u   /* floats: the same shape */
#define AI_THINKINGMACHINES__INKLING_2__WORK__MIX_SCORES           12u   /* floats: a learned query's few */
#define AI_THINKINGMACHINES__INKLING_2__WORK__SIGNS                13u   /* the rotation's signs, 512 halves */
#define AI_THINKINGMACHINES__INKLING_2__WORK__ONES                 14u   /* HIDDEN ones: a norm with no weight */
#define AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE       15u   /* a half a position: a full layer's scale past POSITION_SCALE_FROM */
#define AI_THINKINGMACHINES__INKLING_2__WORK__SCRATCH              16u   /* the verbs' own: ▶ layer__abi.cuh */
#define AI_THINKINGMACHINES__INKLING_2__WORK__PLANES               17u   /* integers from here on */
#define AI_THINKINGMACHINES__INKLING_2__WORK__HIDDEN               17u
#define AI_THINKINGMACHINES__INKLING_2__WORK__QUERY_HEADS          18u
#define AI_THINKINGMACHINES__INKLING_2__WORK__KEY_VALUE_HEADS      19u
#define AI_THINKINGMACHINES__INKLING_2__WORK__HEAD_WIDTH           20u
#define AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_RANK            21u
#define AI_THINKINGMACHINES__INKLING_2__WORK__GATE_RANK            22u
#define AI_THINKINGMACHINES__INKLING_2__WORK__EXPERTS              23u
#define AI_THINKINGMACHINES__INKLING_2__WORK__TOP_K                24u
#define AI_THINKINGMACHINES__INKLING_2__WORK__TAPS                 25u
#define AI_THINKINGMACHINES__INKLING_2__WORK__WINDOW               26u
#define AI_THINKINGMACHINES__INKLING_2__WORK__CONTEXT              27u
#define AI_THINKINGMACHINES__INKLING_2__WORK__POSITION_SCALE_FROM  28u
#define AI_THINKINGMACHINES__INKLING_2__WORK__REALS                29u   /* reals from here on */
#define AI_THINKINGMACHINES__INKLING_2__WORK__EPSILON              29u
#define AI_THINKINGMACHINES__INKLING_2__WORK__QUERY_KEY_LENGTH     30u   /* the length a query's and a key's head is clipped to */
#define AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_LENGTH          31u
#define AI_THINKINGMACHINES__INKLING_2__WORK__ROUTE_SCALE          32u
#define AI_THINKINGMACHINES__INKLING_2__WORK__BIAS_RANK_ROOT       33u   /* √BIAS_RANK: nn's scores divide by it */
#define AI_THINKINGMACHINES__INKLING_2__WORK__CONVOLUTION_SCALE    34u   /* 1 / √(TAPS + 1) */
#define AI_THINKINGMACHINES__INKLING_2__WORK__MADE_SCALE           35u   /* 1 / √(2·TAPS + 1): a full layer's keys and values */
#define AI_THINKINGMACHINES__INKLING_2__WORK__TABLE                36u

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_OBJECTS_LAYER_CUH */
