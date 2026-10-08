#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_MLP_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_MLP_CUH

/* This file needs nothing: every name in it is its own. */

/* ══ ⭐ A DENSE MLP'S PLANE TABLE — the half of a layer after its mixer, in a model without experts ═══════════════════
 * `(nn__mlp__apply h1 table out [residual])` -> `out = residual + down(swiglu(gate, up)(rmsnorm(h1)))`, the residual
 * `h1` when none is named; `(nn__mlp__rows h1 table out rows)` the same for `rows` rows, `h1`'s their residual. Its
 * matrices are TurboQuant's, each with its scales; its input rotated as the pack rotated their columns.
 * ⭐ UNDER TENSOR PARALLELISM a card holds INTER of the whole: its gate and up rows and the same columns of down — the
 * activation is rotated in whole blocks of 512 and every part's slice is whole blocks, so each part's product is exact —
 * and every part but one sums onto a zero residual, so the parts' outputs add up to the layer's. */
#define NN__MLP__NORM            0u
#define NN__MLP__GATE            1u
#define NN__MLP__GATE_SCALES     2u
#define NN__MLP__UP              3u
#define NN__MLP__UP_SCALES       4u
#define NN__MLP__DOWN            5u
#define NN__MLP__DOWN_SCALES     6u
#define NN__MLP__SIGNS           7u
#define NN__MLP__ONE             8u    /* a buffer whose element 0 is 1: the down's one weight */
#define NN__MLP__SCRATCH         9u
#define NN__MLP__HIDDEN          10u   /* integers from here on */
#define NN__MLP__INTER           11u   /* this card's part of the intermediate width */
#define NN__MLP__D_GATE          12u
#define NN__MLP__D_UP            13u
#define NN__MLP__D_DOWN          14u
#define NN__MLP__EPS             15u   /* a float: the norm's epsilon */
#define NN__MLP__TABLE           16u
/* ⭐ Optional: 1 where the model keeps its residual stream rotated — the norm's answer is then the matrices' input, with
 *   no rotation between them. */
#define NN__MLP__RESIDUAL_ROTATED 16u

/* A prompt's rows: up to this many a call. The rows form's scratch starts with their pairs, 8 bytes a row. */
#define NN__MLP__ROWS_MAX        256u

#define NN__MLP__FAULT_TABLE     0x4E4D5442ull   /* "NMTB" — a table cell that is not what its place says */
#define NN__MLP__FAULT_ROOM      0x4E4D524Dull   /* "NMRM" — a scratch too small for the widths */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_MLP_CUH */
