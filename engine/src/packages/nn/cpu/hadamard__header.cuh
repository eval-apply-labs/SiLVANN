#ifndef SILVANN__PACKAGES_NN_CPU_HADAMARD__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_HADAMARD__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/heap_node__header.cuh" /* the node a form's arguments arrive in */
#include "../contracts/objects/hadamard.cuh" /* its constants and fault words */
/* ══ nn — THE ROTATION, WHICH IS THE GAUSSIAN FAMILY'S PRECONDITION ══════════════════════════════════
 *
 * ⚖ ARCHITECT: *"build that quant format with the global nf table, fwht row scaler and no
 * block scaler."*
 *
 * ── ⭐⭐ WHAT IT IS, AND WHY IT IS ONE VERB AND NOT TWO ───────────────────────────────────────────────
 * ```
 *     R(v) = H · (S ⊙ v)
 * ```
 * — a random sign flip, then the normalised Fast Walsh-Hadamard Transform. `S` is a vector of ±1, `H`
 * is the Hadamard matrix scaled by `1/√n`, and `n` must be a power of two.
 * ⛔⛔ THE SIGN FLIP IS NOT SEPARABLE FROM THE TRANSFORM AND MUST NOT BE OFFERED SEPARATELY. `H` alone
 * is a rotation too, and a bad one: it maps a Hadamard basis vector to a spike, which is the exact
 * opposite of what this is for. ⇒ ★ IF A CALLER CAN APPLY HALF OF A TRANSFORM, SOME CALLER WILL, AND
 * THE RESULT IS NOT AN ERROR — IT IS A WORSE DISTRIBUTION THAT STILL DECODES. Fusing them makes the
 * correct thing the only thing available.
 *
 * ── ⭐⭐⭐ WHY THIS EXISTS: A FIXED CODEBOOK IS ONLY CORRECT IF EVERY ROW IS GAUSSIAN ──────────────────
 * `NN__TURBOQUANT__FAMILY_GAUSSIAN` reads levels from a build constant instead of from the file. That is only
 * legitimate if every row really has the shape the table was cut from, and a row of trained weights
 * does not. Each output of the transform is `(1/√n)·Σ ±w_i` — a sum of `n` independent-ish terms — so
 * the central limit theorem makes it Gaussian whatever the input was.
 * ⇒ ⭐ THE ROTATION IS WHAT BUYS THE SHARED TABLE, and the shared table is what takes the codebook off
 * the critical path. ▶ `turboquant__header.cuh` §FAMILY 2.
 *
 * ── ⭐⭐ WHY IT WORKS, AND IT IS ONE LINE ─────────────────────────────────────────────────────────────
 * The packer stores `w' = R(w)` for each ROW; the runtime computes `x' = R(x)`. Then
 * ```
 *     RᵀR = (HS)ᵀ(HS) = SᵀHᵀHS = S·I·S = S² = I        ⇒  R IS ORTHOGONAL
 *     w'·x' = (Rw)ᵀ(Rx) = wᵀRᵀRx = wᵀx                 ⇒  the product is unchanged
 * ```
 * ⇒ ★★ **THE WEIGHTS AND THE ACTIVATION GET THE IDENTICAL TRANSFORM, AND THE OUTPUT GETS NONE.** The
 * rotation lives entirely on the CONTRACTION axis, which is what makes it nearly free: one rotation of
 * length `cols` per matvec, shared by every one of the `rows` dot products that follow it.
 *
 * ⛔⛔ AND `R` IS **NOT** ITS OWN INVERSE — THIS FILE CLAIMED IT WAS, AND THE SUITE CAUGHT IT. `H` is an
 * involution and `S` is an involution, but `R = H∘S` is not: `R∘R = H∘S∘H∘S`, and `H` and `S` do not
 * commute. `MEASURED` on the device at n=4 with `S = [1,−1,1,−1]`: `R([1,2,3,4]) = [−1,5,0,−2]` and
 * `R(R([1,2,3,4])) = [−2,1,−4,3]`, which is not the input. The inverse is `R⁻¹ = Rᵀ = S⊙(H v)` — the
 * same two operations in the OTHER ORDER — and only the packer ever needs it, to check itself.
 * ⇒ ★★ THE CONCLUSION WAS RIGHT AND THE ROUTE TO IT WAS WRONG, WHICH IS THE DANGEROUS COMBINATION: an
 * argument from "both are involutions" reaches the correct "apply R to both sides" and would go on
 * reaching wrong answers everywhere else. The honest reason is orthogonality, and it needs no
 * involution at all. ⛳ Nothing in this package needs a backward transform — but that is because the
 * codec only ever applies `R` to BOTH operands, not because applying it twice undoes it.
 *
 * ── ⚖⚖ WHAT IS **NOT** DECIDED HERE, AND IT IS THE ARCHITECT'S TO DECIDE ─────────────────────────────
 * **WHERE `S` COMES FROM.** This verb takes it as a buffer and therefore takes no position. The three
 * candidates and their prices:
 * ```
 *   a SEED        0 bytes    reproducible both sides; needs one agreed PRNG
 *   a PLANE       cols bits  per tensor, in the file; survives any change of PRNG
 *   a CONSTANT    0 bytes    cannot vary per tensor, so no per-tensor search is possible
 * ```
 * ⛳ The reference packer offers a seed (`scripts/nn_gaussian_pack.py`, splitmix64 indexed by column
 * so a parallel machine can compute `S[i]` from `i` with no state). ⛔ `MEASURED` and it
 * bears on the ruling: searching `S` greedily buys only **+0.030 bits** at full tensor size, but a
 * bank of K=8 vectors with a per-row choice buys **+0.18…0.21** — so the answer is not obviously "one
 * vector", and a seed that cannot be searched forecloses the better option.
 *
 * ── ⛔ WHAT IS NOT CLAIMED ───────────────────────────────────────────────────────────────────────────
 * Nothing about speed. The door runs this as one block of one thread (`gpu/doors.cuh`) and the loop is serial; the `n log n` is a property
 * of the ALGORITHM and not of anything observed on this tree. ▶ `nn_port.md` §⑧.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* `(nn__hadamard__rotate x signs out n)` — `out = H·(S⊙x)`, answering OUT.
 * ⛳ ALIASING `x` AND `out` IS ALLOWED and is the expected call: the transform is written to reach `out`
 * first and work there, so rotating an activation in place costs no scratch.
 * ⛔ `n` MUST BE A POWER OF TWO — refused, not rounded. A Walsh-Hadamard transform of a non-power-of-two
 * length is not a smaller transform, it is a different object, and the butterfly would silently read
 * and write outside the half it was given. */

/* ⭐ DECLARED HERE FOR THE REASON `turboquant__header.cuh`'S PAIR ARE: `kernels.cuh` gathers the compute
 * bodies so they can become kernels, and it is included before this package's impls — so a body that
 * calls the butterfly has to be able to see it. ▶ `kernels.cuh`. */
static __device__ inline void nn__hadamard__zzpackage_one_block(uint16_t* out, const uint16_t* in,
                                                          const uint16_t* sign, uint64_t b,
                                                          bool* over);

#endif /* SILVANN__PACKAGES_NN_CPU_HADAMARD__HEADER_CUH */
