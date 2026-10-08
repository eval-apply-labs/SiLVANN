#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_ROUTED_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_ROUTED_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "expert.cuh"                                  /* a backing */
#include "../abi/gpu.cuh"                              /* the grouped launches' width */
#include "../../../sys/contracts/objects/verb_abi.cuh" /* a verb's context, which a mask is handed */

/* ══ ⭐⭐ A MODEL'S ROUTED EXPERTS, AS nn RUNS THEM ═══════════════════════════════════════════════════════════════════
 * The picks of a position, or of a prompt chunk's rows, each expert's gate-and-up read from its slot, the activation,
 * the rotation, and the down weighted into the sum. A model's experts verb reads its own table into this and keeps what
 * is its own: the router, the hand's layout, the shared expert, a mask. ▶ `cpu/routed__header.cuh`.
 *
 * A HAND is what the card hands a worker for one position: its input (`hidden` halves) at 0, the router's weights after
 * it (2·hidden bytes in), the picks as words at `picks_at`; a chunk's hands are rows `hand_bytes` apart.
 * A SLOT is one expert: the gate-and-up codes at 0, then `up_lut`, `down` and `down_lut` — the layout nn's tier keeps. */
typedef struct nn__routed {
    uint64_t layer, type;                       /* the collection: a layer's band of the experts' type             */
    uint64_t hidden, inter, experts, top_k;     /* H, I, E, K                                                       */
    uint64_t bits;                              /* the experts' width, 4 or 8                                       */
    uint64_t up_lut, down, down_lut;            /* each plane's place in a slot                                     */
    uint64_t first, count;                      /* the experts this worker holds: `first` to `first + count`        */
    bool     int8;                              /* the products in integers                                         */
    float    limit;                             /* the activation: SwiGLU, its gate and up clamped at `limit`, or —
                                                 * 0 — SwiGLU as nn's `swiglu_pairs` computes it                    */
    uint64_t signs, zero;                       /* card memory: the rotation's signs, a row of zeros (H halves)     */
    uint64_t scratch, scratch_room;             /* the verb's scratch, and its bytes                                */
    uint64_t hand_bytes, picks_at;              /* a hand row's width and where its picks are in it, in bytes       */
    uint64_t batch;                             /* a prompt's experts a batch where the slots are a cache           */
    nn__expert__backing backing;                /* where the slots are a cache, the file they are read from; 0 none */
} nn__routed;

/* What a position's routine leaves for a model's own addition after it — a mask over the routed downs: the picks
 * (all `top_k`, a held one marked past `experts`), the `n` it computed — pick `which[i]` the i-th — and where their
 * rotated activations are, `inter` halves each in that order. */
typedef struct nn__routed__done {
    uint64_t ids[NN__TURBOQUANT__GROUPS_MAX], which[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t n, actr;
} nn__routed__done;

/* What a prompt's routine hands a model's mask over the routed downs, before the rows are summed: the `count` picks it
 * computed in the order it laid them out — pick `p`'s expert `expert_of[p]` (less `first`), its rotated activation at
 * `actr + 2·p·inter`, its weight the `p`-th half at `weights`, the slot of the sum it lands in the second word of each
 * pair at `down_pairs` — and the sum, `sums` floats a slot of `hidden`. 0, or the fault to answer with. */
typedef uint64_t (*nn__routed__rows_mask)(void* model, sys__engine__ctx* ctx, uint64_t count, const uint32_t* expert_of,
                                          uint64_t actr, uint64_t weights, uint64_t down_pairs, uint64_t sums);

#define NN__ROUTED__BATCH_MAX    32u            /* a prompt's experts a batch, at the most                          */
#define NN__ROUTED__FAULT_EXPERT 0x4E4E5245ull  /* "NNRE" — a pick is not an expert in memory                       */
#define NN__ROUTED__FAULT_ROOM   0x4E4E5252ull  /* "NNRR" — the scratch is too small for the shapes                 */
#define NN__ROUTED__FAULT_SHAPE  0x4E4E5253ull  /* "NNRS" — a shape or a setting the routine does not take          */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_ROUTED_CUH */
