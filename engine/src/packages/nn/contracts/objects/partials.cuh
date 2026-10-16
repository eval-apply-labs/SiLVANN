#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PARTIALS_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PARTIALS_CUH

/* This file needs nothing: every name in it is its own. */

/* ══ ⭐ A TENSOR-PARALLEL GROUP — devices whose partial sums of one vector add up to it ═══════════════════════════════
 * ⚖ *"tensor parallelism requires DMA"* · *"the addition to gpu0 by adding up in pairs (maybe in blocks of 4 would be
 * more effective) and transmit down at the same way"* · *"go with the stream wait events"*.
 * `(nn__partials__join group rank ranks fan)` — the calling worker is rank `rank` of `ranks` in group `group`, its sums
 *   taken `fan` at a time. Each rank joins on its own worker; rank 0 first, which starts the group afresh. A rank that
 *   cannot read and write every other joined rank's memory directly is refused (`NN__PARTIALS__FAULT_NO_DMA`).
 * `(nn__partials__reduce group part out n [residual])` -> `out` — every rank calls it with its own partial of `n`
 *   halves; each rank's `out` ends holding the residual (rank 0's, where it names one) plus every rank's partial.
 *   Up the tree: a rank whose place is a multiple of `fan^(k+1)` adds the partials of the `fan - 1` ranks `fan^k` apart
 *   above it onto its own, in rank order, until rank 0 adds the last onto the residual into its `out`. Down the same
 *   tree: each rank copies the sum from the rank it sent its partial to. Every read is of another card's memory, in
 *   place; every wait is the reading card's queue waiting on the writer's mark — the host waits only for the other
 *   rank's thread to have QUEUED its part (a generation each call), never for a card.
 * ⛳ BETWEEN TWO CALLS a rank's `part` and `out` are read by the others, so it writes neither of them except through its
 *   next partial: its next `part` is queued after its copy of the sum, which every reader of its previous one has
 *   waited for. */
#define NN__PARTIALS__GROUPS_MAX 8u
#define NN__PARTIALS__RANKS_MAX  8u
#define NN__PARTIALS__FAN_MAX    7u    /* the root adds its residual, its own and `fan - 1` others: 8 parts at most */
/* How long a rank waits for another rank's thread to queue its part before it gives up, in seconds. */
#define NN__PARTIALS__PATIENCE   60u

#define NN__PARTIALS__FAULT_NO_DMA  0x4E50444Dull   /* "NPDM" — tensor parallelism requires DMA: a pair of the group's devices
                                                       cannot read each other's memory */
#define NN__PARTIALS__FAULT_GROUP   0x4E504750ull   /* "NPGP" — no such group, rank or fan, or a rank not joined yet */
#define NN__PARTIALS__FAULT_PEER    0x4E505052ull   /* "NPPR" — another rank never queued its part */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PARTIALS_CUH */
