#ifndef SILVANN__PACKAGES_NN_CPU_ROUTED__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_ROUTED__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/abi/gpu.cuh"          /* the shape of nn's door table */
#include "expert__header.cuh"                 /* where an expert's slot is, the loader, the tier */
#include "../contracts/objects/routed.cuh"    /* what a model's verb hands these */

/* ══ ⭐⭐ A MODEL'S ROUTED EXPERTS, AS nn RUNS THEM — the three a model's experts verbs call ═════════════════════════════
 *   top_k     the router's `k` largest of `n` logits — σ + `bias` where `bias` is not 0, its weights σ normalised and
 *             times `scale` — into the first `k` elements of the node array `picks` (and into `chosen`, where it is not
 *             0), their weights into `weights`. The card writes the nodes itself where the heap is registered with it,
 *             and a card buffer lands them otherwise. Waits for the card: the picks are what comes next.
 *   position  one position's picks this worker holds into `routed = Σ w_j · down_j(act(gate_up_j(x)))`, `x` and the
 *             weights the hand's — and 0 where it holds none. The picks in memory are computed while the rest arrive
 *             (read into a slot where the slots are a cache, a mapped file's pages started), and every pick's down is
 *             summed in pick order: the order the experts arrived in changes nothing in the sum. With a card's tier
 *             (`tier` not 0) the picks it holds are computed here and marked held — their number plus `experts` — for
 *             the other workers to skip. `done`, where it is not 0, says what was computed, for a model's mask.
 *   rows      the same over a chunk's `n` hand rows, EXPERT-MAJOR: every expert the chunk used runs once over the rows
 *             that picked it; where the slots are a cache a batch at a time, the next batch read while one computes,
 *             from the least picked to the most (so the most used are the most recent when it ends); where they are a
 *             mapped file the ones in memory first. Each pick's down lands in a sum of its own and the rows are added
 *             in pick order at the end, so the answer is the same whatever order the experts ran in. With a tier, only
 *             the picks marked held are this worker's, four experts a batch, each waited for only while its copy is on
 *             its way. `mask`, where it is not 0, is handed the picks before the rows are summed.
 * Each answers 0, or the fault to answer with. */
static uint64_t nn__routed__top_k(sys__engine__ctx* ctx, const nn__doors* doors, uint64_t picks, uint64_t weights, uint64_t logits,
                                  uint64_t bias, float scale, uint64_t n, uint64_t k, uint64_t* chosen);
static uint64_t nn__routed__position(sys__engine__ctx* ctx, const nn__doors* doors, const nn__routed* r, uint64_t hand,
                                     uint64_t picks, const nn__expert_tier* tier, uint64_t routed, nn__routed__done* done);
static uint64_t nn__routed__rows(sys__engine__ctx* ctx, const nn__doors* doors, const nn__routed* r, uint64_t hands, uint64_t n,
                                 const nn__expert_tier* tier, uint64_t routed, nn__routed__rows_mask mask, void* model);

#endif /* SILVANN__PACKAGES_NN_CPU_ROUTED__HEADER_CUH */
